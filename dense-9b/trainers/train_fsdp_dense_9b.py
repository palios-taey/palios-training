#!/usr/bin/env python3
"""Dense Qwen3.5-9B-Base FSDP LoRA SFT — surgical adaptation of train_fsdp_v3.py.

Same proven 4-Spark FSDP+LoRA pipeline (rank-split mmap load,
accelerator.prepare(), explicit FSDP MixedPrecision policy, Adafactor
optimizer, summon_full_params save, session-limit fragmentation exit).
Only the MoE-specific lines were changed:

  1. Import dense decoder layer (Qwen3_5DecoderLayer) instead of the
     MoE one (Qwen3_5MoeDecoderLayer).
  2. LoRA target_modules: drop `shared_expert.*` (no shared expert on
     dense), add `mlp.{gate,up,down}_proj` (dense analogue).
  3. Drop `output_router_logits = False` (no router on dense).
  4. The FREEZE_CONFIG branches that reference `mlp.experts.` /
     `shared_expert_gate` are no-ops on dense — that's fine, LoRA is
     the trainable surface.
"""

import os, gc, random, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")
os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLA_DISABLE_CAUSAL_CONV1D", "1")
os.environ.setdefault("TRITON_AUTOTUNE_DISABLE", "1")
os.environ.setdefault("FLA_USE_TMA", "0")
os.environ.setdefault("NCCL_IB_TIMEOUT", "23")

# ── DISABLE FLA (Blackwell/GB10 Triton linear-attention kernels) ──
# Original patched qwen3_5_MOE = WRONG module for the DENSE qwen3_5 run (no-op). py-spy 2026-07-23
# proved the wedge is a rank-divergent STALL in the activation-checkpointed linear_attn forward (all
# 4 ranks hung in modeling_qwen3_5 forward at DIFFERENT lines). Null the FLA kernels in the DENSE
# module BEFORE model construction so linear_attn uses the torch fallback (torch_chunk_gated_delta_rule
# etc.) — SAME MATH, deterministic, no Triton kernel. Patch BOTH modules to be safe; verify it took.
if os.environ.get("DISABLE_FLA", "0") == "1":
    import importlib
    for _mod in ("transformers.models.qwen3_5.modeling_qwen3_5",
                 "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        try:
            _q = importlib.import_module(_mod)
            _q.chunk_gated_delta_rule = None
            _q.fused_recurrent_gated_delta_rule = None
            _q.FusedRMSNormGated = None
            _q.causal_conv1d_fn = None
            _q.causal_conv1d_update = None
            if hasattr(_q, "is_fast_path_available"):
                _q.is_fast_path_available = False
            print(f"[DIAG] FLA DISABLED in {_mod} (chunk_gated_delta_rule={_q.chunk_gated_delta_rule}, "
                  f"is_fast_path_available={getattr(_q, 'is_fast_path_available', 'n/a')}) — torch fallback")
        except ModuleNotFoundError:
            pass
os.environ.setdefault("NCCL_IB_RETRY_CNT", "7")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")

import hashlib
import json
import math
import collections
import logging
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
import psutil
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Keystone Layers — selected by routing probe (Mar 29, 2026)
# These have the strongest super-expert concentrations (50-80% of tokens).
# Expert tensors in these layers are fully unfrozen (ESFT).
# All 256 experts train, but routing focuses gradient on hot experts.
# ═══════════════════════════════════════════════════════════════════
KEYSTONE_LAYERS = [17, 28]  # 2 layers — 3.2GB optimizer/node, fits 5.7GB headroom
# L17: 16 T1 safe infra experts. L28: 19 T1 safe infra experts (densest).
# 2 layers = 1.6B params, FSDP shards to 400M/node, 3.2GB optimizer/node

# Per keystone layer: ~805M params (gate_up_proj [256,1024,2048] + down_proj [256,2048,512])

# Trainable components (besides keystone expert tensors):
# - shared_expert gate/up/down_proj (all 40 layers) — LoRA r=64
# - attention projections (all 40 layers, both SDPA + DeltaNet) — LoRA r=64
# - router gates mlp.gate + shared_expert_gate (all 40 layers) — full, lower LR
# - layernorms (all layers) — full
# - embeddings/lm_head — frozen for v1 (conservative)


def _clean_fsdp_name(name):
    """Strip FSDP wrapper prefixes to get canonical model parameter name."""
    return name.replace("_fsdp_wrapped_module.", "").replace("module.", "")


def _is_trainable(name, keystone_layers):
    """Determine if a parameter should be trainable in the hybrid approach."""
    clean = _clean_fsdp_name(name)

    # LoRA adapter weights — always trainable (PEFT handles this)
    if 'lora_' in clean.lower():
        return True

    # Keystone expert tensors — direct ESFT on selected layers
    for kl in keystone_layers:
        if f'layers.{kl}.mlp.experts.' in clean:
            return True

    # Router gates — full fine-tune (lower LR group)
    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
        return True

    # Layer norms — full fine-tune
    if 'layernorm' in clean or 'norm' in clean:
        return True

    # shared_expert projections that PEFT wraps — PEFT handles requires_grad
    # (These will be set by get_peft_model if they're in modules_to_save or target_modules)

    return False

def save_lora_only_fsdp(
    model,
    accelerator,
    out_dir,
    adapter_name="default",
    tokenizer=None,
):
    """Save only trainable PEFT weights from an FSDP-wrapped model."""
    import hashlib

    accelerator.wait_for_everyone()

    # Derive one global key list, then perform one explicit full-tensor gather per key.
    # The implicit get_model_state_dict traversal can issue rank-local collectives in
    # different orders; this fixed loop makes count and order uniform by construction.
    state = model.state_dict()
    local_keys = sorted(k for k in state if "lora_" in k.lower())
    gathered_keys = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered_keys, local_keys)
    keys = sorted(set().union(*map(set, gathered_keys)))
    key_hash = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
    collectives = len(keys)
    log.info("LoRA-only write plan rank=%s count=%d hash=%s collectives=%d",
             accelerator.process_index, len(keys), key_hash, collectives)
    plans = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(plans, (len(keys), key_hash, collectives))
    if any(p != plans[0] for p in plans):
        raise RuntimeError(f"LoRA-only write plan mismatch: {plans}")
    if len(keys) != 704:
        raise RuntimeError(f"LoRA-only write expected 704 keys, found {len(keys)}")
    missing = [key for key in keys if key not in state]
    missing_all = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(missing_all, missing)
    if any(missing_all):
        raise RuntimeError(f"LoRA-only write missing keys by rank: {missing_all}")
    trainable_state = {}
    for key in keys:
        value = state[key]
        full = value.full_tensor() if hasattr(value, "full_tensor") else value
        if accelerator.is_main_process:
            trainable_state[key] = full.detach().cpu().clone()
        del full

    total_bytes = 0
    if accelerator.is_main_process:
        if not trainable_state:
            raise RuntimeError("adapter-only save found no trainable tensors")
        nonfinite = [
            key
            for key, value in trainable_state.items()
            if not torch.isfinite(value).all().item()
        ]
        if nonfinite:
            raise RuntimeError(
                f"adapter-only save found non-finite tensors: {nonfinite[:5]}"
            )
        topology = [
            (key, str(value.dtype), list(value.shape))
            for key, value in sorted(trainable_state.items())
        ]
        topology_hash = hashlib.sha256(
            json.dumps(topology, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        total_bytes = sum(
            value.numel() * value.element_size()
            for value in trainable_state.values()
        )
        total_gb = total_bytes / 1e9
        unwrapped = accelerator.unwrap_model(model)
        log.info(
            "Saving %d finite trainable tensors to %s "
            "(%.2f GB, topology_sha256=%s)",
            len(trainable_state),
            out_dir,
            total_gb,
            topology_hash,
        )
        unwrapped.save_pretrained(
            out_dir,
            selected_adapters=[adapter_name],
            state_dict=trainable_state,
            safe_serialization=True,
            save_embedding_layers=False,
            is_main_process=True,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
    else:
        total_gb = 0.0

    accelerator.wait_for_everyone()
    return total_gb


def _lora_fingerprint(model):
    import hashlib

    digest = hashlib.sha256()
    tensor_count = 0
    for name, param in sorted(model.named_parameters()):
        if "lora_" not in name.lower() or not param.requires_grad:
            continue
        if param.is_meta:
            raise RuntimeError(f"cannot fingerprint meta LoRA parameter {name}")
        value = param.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise RuntimeError("cannot fingerprint LoRA initialization: no trainable LoRA tensors")
    return digest.hexdigest(), tensor_count


def _tokenize_sft_pair(
    messages, tokenizer, tools=None, *, require_assistant_labels=False
):
    """Tokenize SFT conversation with assistant-only loss.

    Uses incremental template application to find exact assistant token boundaries.
    Handles Qwen3.5 chat template which adds special tokens around content.

    `tools`: the record's top-level tool schema. Passed to the chat template so
    the available-tools list is present in the prompt at train time (#1). Caller
    must pass None when a system message already embeds <tools> (avoid double-inject).

    Exact-dose training sets `require_assistant_labels` so a template mismatch
    cannot silently change the objective from assistant-only to full-conversation.
    """
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools,
                                                  add_generation_prompt=False, enable_thinking=False)
    except Exception as exc:
        if require_assistant_labels:
            raise RuntimeError("full chat-template rendering failed") from exc
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        full_text = "\n".join(parts) + tokenizer.eos_token

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(full_ids)

    # Build prefix up to each assistant message to find exact token boundaries
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # Template up to (but not including) this assistant message
        prefix_msgs = messages[:i]
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs, tokenize=False, tools=tools, add_generation_prompt=True, enable_thinking=False)
        except Exception as exc:
            if require_assistant_labels:
                raise RuntimeError(
                    f"assistant-prefix template rendering failed at message {i}"
                ) from exc
            prefix_text = ""

        # Template including this assistant message
        incl_msgs = messages[:i+1]
        try:
            incl_text = tokenizer.apply_chat_template(
                incl_msgs, tokenize=False, tools=tools, add_generation_prompt=False, enable_thinking=False)
        except Exception as exc:
            if require_assistant_labels:
                raise RuntimeError(
                    f"assistant-inclusive template rendering failed at message {i}"
                ) from exc
            incl_text = full_text

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        incl_ids = tokenizer.encode(incl_text, add_special_tokens=False)

        # Assistant tokens are between prefix end and incl end
        start = len(prefix_ids)
        end = len(incl_ids)
        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    # Preserve the legacy full-conversation fallback outside exact-dose runs.
    if all(l == -100 for l in labels):
        if require_assistant_labels:
            raise RuntimeError("chat template produced no assistant loss tokens")
        labels = list(full_ids)

    return full_ids, labels


class CappedMixtureSampler(torch.utils.data.Sampler):
    """Lane-weighted mixture with a per-example cap on tiny lanes (Gaia's binding recipe).

    Yields a GLOBAL index list (identical on every rank — seeded by set_seed) and lets
    accelerate.prepare() do the SINGLE correct shard. Deliberately NOT a DistributedSampler:
    stacking one on top of accelerate's shard is the proven double-shard bug (only 1/4 the corpus
    trained). Per-lane target draws = weight * epoch_size, capped at lane_size*cap for lanes below
    `threshold`, so 21 values rows are not drawn ~17x and memorized verbatim.
    """
    def __init__(self, lanes, lane_weights, cap=3, threshold=500, seed=0):
        import collections as _c
        self.lanes = lanes
        self.seed = int(seed)
        self.epoch = 0
        self.tgt = {}
        for kv in lane_weights.split(","):
            k, v = kv.split("="); self.tgt[k.strip()] = float(v)
        counts = _c.Counter(lanes)
        missing = [l for l in counts if l not in self.tgt]
        if missing:
            raise SystemExit(f"ABORT: LANE_WEIGHTS missing a weight for present lanes: {missing}")
        self.lane_idx = _c.defaultdict(list)
        for i, l in enumerate(lanes):
            self.lane_idx[l].append(i)
        epoch = len(lanes)
        self.per_lane_draws = {}
        for l, idxs in self.lane_idx.items():
            n = len(idxs)
            eff_cap = cap if n < threshold else 10**9
            self.per_lane_draws[l] = min(round(self.tgt[l] * epoch), n * eff_cap)

    def set_epoch(self, epoch):
        """DistributedSampler contract — the training loop calls this at every epoch boundary.

        Without it the run dies at the end of epoch 0 with AttributeError (observed 2026-07-21
        17:42, r0.log). Reseeding by epoch (rather than consuming global RNG) also guarantees
        every rank builds a BYTE-IDENTICAL index list, which is required: this sampler yields a
        GLOBAL list and accelerate.prepare() does the single shard from it. If ranks disagreed on
        the list they would silently train different data under the same step count.
        """
        self.epoch = int(epoch)

    def _build(self):
        # Explicit generator, NOT global RNG: deterministic + identical on every rank + varies
        # per epoch. Global RNG state can diverge between ranks and would desync the shard.
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        pool = []
        for l, idxs in sorted(self.lane_idx.items()):      # sorted → rank-stable iteration order
            n, draws = len(idxs), self.per_lane_draws[l]
            full = draws // n
            for _ in range(full):
                pool += idxs
            rem = draws - full * n
            if rem:
                perm = torch.randperm(n, generator=g).tolist()
                pool += [idxs[j] for j in perm[:rem]]
        perm = torch.randperm(len(pool), generator=g).tolist()
        return [pool[j] for j in perm]

    def __iter__(self):
        return iter(self._build())

    def __len__(self):
        return sum(self.per_lane_draws.values())


def _supervised_sft_windows(input_ids, labels, max_seq, context_overlap=256):
    if len(input_ids) != len(labels):
        raise RuntimeError(
            f"SFT token/label length mismatch: {len(input_ids)} != {len(labels)}"
        )
    if max_seq <= context_overlap:
        raise RuntimeError(
            f"max_seq={max_seq} must exceed context_overlap={context_overlap}"
        )
    if len(input_ids) <= max_seq:
        return [(input_ids, labels)]

    supervised_positions = [
        index for index, label in enumerate(labels) if label != -100
    ]
    if not supervised_positions:
        raise RuntimeError("overlength SFT row has no assistant loss tokens")

    windows = []
    cursor = 0
    start = 0
    while cursor < len(supervised_positions):
        first_unassigned = supervised_positions[cursor]
        end = min(len(input_ids), start + max_seq)
        if first_unassigned >= end:
            remaining_last = supervised_positions[-1]
            if remaining_last - first_unassigned + 1 <= max_seq:
                start = max(0, remaining_last + 1 - max_seq)
            else:
                start = max(0, first_unassigned - context_overlap)
            end = min(len(input_ids), start + max_seq)

        stop = cursor
        while (
            stop < len(supervised_positions)
            and supervised_positions[stop] < end
        ):
            stop += 1
        if stop == cursor:
            raise RuntimeError(
                "SFT supervised-window construction made no label progress"
            )

        chunk_ids = input_ids[start:end]
        chunk_labels = [-100] * len(chunk_ids)
        for position in supervised_positions[cursor:stop]:
            chunk_labels[position - start] = labels[position]
        windows.append((chunk_ids, chunk_labels))
        cursor = stop
        start = max(0, end - context_overlap)

    return windows


class BucketSFTDataset(Dataset):
    """Pre-tokenized, length-sorted SFT dataset for bucket batching.

    Per Perplexity DR (2026-04-28): the original CombinedSFTDataset used
    `random.random()` weighted sampling and ignored the DistributedSampler's
    indices, producing variable-length batches and probabilistic epoch
    coverage rather than partitioned. Replaced with a deterministic dataset
    that tokenizes everything up front, sorts by length, and lets the
    DistributedSampler partition the sorted indices across ranks.

    Overlength rows become loss-bearing supervised windows with 256 tokens
    of context overlap. Every assistant label is emitted exactly once:
    overlap labels are masked after their first window, and prompt-only
    windows are never admitted as training samples.
    """

    def __init__(self, sft_jsonl_path, tokenizer, max_seq, strict_one_sample_per_row=False):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.strict_one_sample_per_row = strict_one_sample_per_row

        # ── Cache key ─────────────────────────────────────────────────────
        # Bind cached tensors to the corpus bytes, tokenizer path, and window
        # semantics. Size+mtime alone can alias an in-place corpus correction.
        import hashlib, pickle, time
        corpus_st = os.stat(sft_jsonl_path)
        corpus_sha = hashlib.sha256()
        with open(sft_jsonl_path, "rb") as corpus_file:
            for chunk in iter(lambda: corpus_file.read(1024 * 1024), b""):
                corpus_sha.update(chunk)
        cache_key_parts = [
            os.path.abspath(sft_jsonl_path),
            str(corpus_st.st_size),
            str(int(corpus_st.st_mtime)),
            f"corpus_sha256={corpus_sha.hexdigest()}",
            getattr(tokenizer, "name_or_path", "?"),
            f"max_seq={max_seq}",
            "v3-supervised-windows",
        ]
        if strict_one_sample_per_row:
            cache_key_parts.extend([
                "strict_one_sample_per_row=v2-assistant-labels",
            ])
        cache_key = hashlib.sha256("|".join(cache_key_parts).encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(sft_jsonl_path), "tokenized_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"sft_{cache_key}.pkl")

        # ── Try cache load ────────────────────────────────────────────────
        if os.path.exists(cache_path):
            try:
                t0 = time.time()
                with open(cache_path, "rb") as f:
                    self.samples, self.lanes = pickle.load(f)
                log.info(
                    f"BucketSFTDataset: loaded {len(self.samples)} pre-tokenized "
                    f"samples from cache {cache_path} in {time.time()-t0:.1f}s "
                    f"(saved ~7min of re-tokenization)"
                )
                return
            except Exception as e:
                log.warning(f"Cache load failed ({e}); falling back to fresh tokenize")

        # ── Fresh tokenize ────────────────────────────────────────────────
        self.samples = []  # list of (input_ids, labels)
        self.lanes = []    # parallel list: meta.lane per SAMPLE (chunks inherit the parent row's lane)
        _recs = []         # (ids, labels, lane) before length-sort
        log.info(f"Pre-tokenizing SFT corpus from {sft_jsonl_path} (no cache hit)...")
        n_rows = 0
        n_split = 0
        with open(sft_jsonl_path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    if strict_one_sample_per_row:
                        raise RuntimeError(f"blank SFT row at line {line_number}")
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    if strict_one_sample_per_row:
                        raise RuntimeError(
                            f"invalid SFT JSON at line {line_number}: {exc}"
                        ) from exc
                    continue
                msgs = row.get("messages")
                _lane = (row.get("meta") or {}).get("lane", "_unlaned")
                if not isinstance(msgs, list) or not msgs:
                    if strict_one_sample_per_row:
                        raise RuntimeError(
                            f"SFT row {line_number} has no non-empty messages list"
                        )
                    continue
                # Pass the top-level tools schema into the template so the model
                # sees the available-tools list at train time. ~60% of records
                # (xlam/toolace/swe) carry tools ONLY here, not inlined in a
                # system message; without this they trained ungrounded (#1).
                # Guard double-injection: hermes records already embed <tools>
                # in their system message, so skip tools= for those.
                row_tools = row.get("tools")
                if row_tools and any(
                    m.get("role") == "system" and "<tools>" in (m.get("content") or "")
                    for m in msgs
                ):
                    row_tools = None
                try:
                    ids, labels = _tokenize_sft_pair(
                        msgs,
                        tokenizer,
                        tools=row_tools,
                        require_assistant_labels=strict_one_sample_per_row,
                    )
                except Exception as exc:
                    if strict_one_sample_per_row:
                        raise RuntimeError(
                            f"SFT tokenization failed at line {line_number}: {exc}"
                        ) from exc
                    continue
                n_rows += 1
                if len(ids) <= max_seq:
                    _recs.append((ids, labels, _lane))
                else:
                    if strict_one_sample_per_row:
                        raise RuntimeError(
                            f"SFT row {line_number} tokenized to {len(ids)} tokens, "
                            f"exceeding max_seq={max_seq}"
                        )
                    windows = _supervised_sft_windows(ids, labels, max_seq)
                    for chunk_ids, chunk_labels in windows:
                        _recs.append((chunk_ids, chunk_labels, _lane))
                    n_split += len(windows) - 1

        # REAL bucket batching: sort by length so a length-grouped sampler
        # (or simple sorted sampler) produces batches of similar-length
        # samples. Padding happens dynamically in collate_fn — pads only to
        # the batch max, NOT to max_seq.
        _recs.sort(key=lambda r: len(r[0]))
        self.samples = [(r[0], r[1]) for r in _recs]
        self.lanes = [r[2] for r in _recs]

        log.info(f"BucketSFTDataset: {n_rows} rows -> {len(self.samples)} samples "
                 f"({n_split} extra from outlier splits, length-sorted, "
                 f"DYNAMIC padding via collate)")

        # ── Save cache (atomic via temp+rename) ───────────────────────────
        # Multi-rank race-safe: all 4 ranks tokenize independently first time
        # (waste of CPU but correct), each writes to its own temp, last one
        # to rename wins. Subsequent launches all hit the cache.
        try:
            tmp_path = cache_path + f".tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump((self.samples, self.lanes), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, cache_path)
            log.info(f"Saved tokenized cache to {cache_path}")
        except Exception as e:
            log.warning(f"Cache save failed ({e}); training will proceed without cache")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return {"input_ids": ids, "labels": labels, "is_dpo": False}


class ExactEpochDataset(Dataset):
    def __init__(self, dataset, expected_real_samples, global_batch_size):
        if expected_real_samples <= 0:
            raise ValueError("expected_real_samples must be positive")
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if len(dataset) != expected_real_samples:
            raise RuntimeError(
                "EXACT SFT source mismatch: "
                f"tokenized_samples={len(dataset)} expected_real={expected_real_samples}"
            )
        self.dataset = dataset
        self.real_samples = expected_real_samples
        self.padding_samples = (-expected_real_samples) % global_batch_size
        self.total_samples = expected_real_samples + self.padding_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < self.real_samples:
            sample = dict(self.dataset[idx])
            sample["is_exact_padding"] = False
            return sample
        sample = dict(self.dataset[0])
        sample["labels"] = [-100] * len(sample["input_ids"])
        sample["is_exact_padding"] = True
        return sample


class PackedCPTDataset(Dataset):
    """Fixed-length PACKED CPT dataset — every item is EXACTLY seq_len pre-tokenized ids
    (produced by dense-9b/data/pack_corpus.py). Uniform micro-batch shape every step, so the
    CUDA caching allocator reuses the same freed blocks → NO fragmentation (fixes the first-step
    NV_ERR_NO_MEMORY that variable length-bucketing triggers) AND zero padding waste (throughput).
    No token-budget keys → routes to the legacy per-batch labels-loss path (uniform length ⇒
    sequence-average == token-average, so the loss is correct). Loss is masked on the single
    partial-final block's eos padding (pad_tail)."""

    def __init__(self, path):
        self.records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.records.append((obj["input_ids"], int(obj.get("pad_tail", 0))))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ids, pad_tail = self.records[idx]
        labels = list(ids)
        if pad_tail > 0:                     # mask the final block's eos padding from loss
            for i in range(len(labels) - pad_tail, len(labels)):
                labels[i] = -100
        return {"input_ids": list(ids), "labels": labels, "is_dpo": False}


class BucketCPTDataset(Dataset):
    """Byte-offset CPT dataset with cached token lengths for bucket batching."""

    def __init__(self, cpt_path, tokenizer, max_seq):
        self.cpt_path = cpt_path
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise RuntimeError("CPT training requires tokenizer.eos_token_id")

        import hashlib, pickle, time
        corpus_st = os.stat(cpt_path)
        cache_key_parts = [
            os.path.abspath(cpt_path),
            str(corpus_st.st_size),
            str(int(corpus_st.st_mtime)),
            getattr(tokenizer, "name_or_path", "?"),
            f"max_seq={max_seq}",
            "bucket_cpt_index_v1",
        ]
        cache_key = hashlib.sha256("|".join(cache_key_parts).encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(cpt_path), "tokenized_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cpt_index_{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                t0 = time.time()
                with open(cache_path, "rb") as f:
                    self.entries = pickle.load(f)
                log.info(
                    f"BucketCPTDataset: loaded {len(self.entries)} length-index entries "
                    f"from {cache_path} in {time.time()-t0:.1f}s"
                )
                self._log_stats()
                return
            except Exception as e:
                log.warning(f"CPT index cache load failed ({e}); rebuilding")

        self.entries = []
        bucket_counts = CounterLike()
        log.info(f"Building CPT length index from {cpt_path}...")
        t0 = time.time()
        with open(cpt_path, "rb") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(row.get("input_ids"), list):
                    length = len(row["input_ids"])
                else:
                    text = str(row.get("text", "")).strip()
                    if not text:
                        continue
                    length = len(tokenizer.encode(text, add_special_tokens=False)) + 1
                if length > max_seq:
                    raise RuntimeError(
                        f"CPT row exceeds max_seq: {length}>{max_seq} at byte offset {offset}; "
                        "rebuild the corpus before launching training"
                    )
                bucket = self.bucket_for_length(length)
                entry = {
                    "offset": offset,
                    "length": length,
                    "loss_tokens": max(length - 1, 1),
                    "bucket": bucket,
                }
                self.entries.append(entry)
                bucket_counts[bucket] += 1

        if not self.entries:
            raise RuntimeError(f"No CPT rows indexed from {cpt_path}")

        log.info(
            f"BucketCPTDataset: indexed {len(self.entries)} rows in {time.time()-t0:.1f}s "
            f"buckets={dict(bucket_counts)}"
        )
        try:
            tmp_path = cache_path + f".tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, cache_path)
            log.info(f"Saved CPT length index cache to {cache_path}")
        except Exception as e:
            log.warning(f"CPT index cache save failed ({e}); training will proceed without cache")
        self._log_stats()

    @staticmethod
    def bucket_for_length(length):
        if length < 2048:
            return "short"
        if length < 8192:
            return "mid"
        return "long"

    def _log_stats(self):
        lengths = sorted(e["length"] for e in self.entries)
        bucket_counts = CounterLike()
        for entry in self.entries:
            bucket_counts[entry["bucket"]] += 1
        p90 = lengths[int(0.9 * (len(lengths) - 1))]
        log.info(
            f"BucketCPTDataset stats: rows={len(lengths)} min={lengths[0]} "
            f"median={lengths[len(lengths)//2]} p90={p90} max={lengths[-1]} "
            f"buckets={dict(bucket_counts)}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            idx, loss_denom_tokens, group_end, group_id, micro_id, pad_to_length = key
        else:
            idx, loss_denom_tokens, group_end, group_id, micro_id, pad_to_length = key, 0, True, -1, 0, 0
        if idx < 0:
            return {
                "input_ids": [(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id)] * max(1, int(pad_to_length)),
                "labels": [-100] * max(1, int(pad_to_length)), "is_dpo": False,
                "loss_denom_tokens": int(loss_denom_tokens), "group_end": bool(group_end), "group_id": int(group_id),
                "micro_id": int(micro_id), "bucket": "padding", "length": 0,
                "pad_to_length": int(pad_to_length), "is_padding": True,
            }
        entry = self.entries[idx]
        with open(self.cpt_path, "rb") as f:
            f.seek(entry["offset"])
            raw = f.readline()
        row = json.loads(raw.decode("utf-8"))
        if isinstance(row.get("input_ids"), list):
            ids = [int(token) for token in row["input_ids"]]
        else:
            text = str(row.get("text", "")).strip()
            ids = self.tokenizer.encode(text, add_special_tokens=False) + [self.eos_token_id]
        if len(ids) > self.max_seq:
            raise RuntimeError(f"CPT row exceeds max_seq after tokenize: {len(ids)}>{self.max_seq}")
        return {
            "input_ids": ids,
            "labels": list(ids),
            "is_dpo": False,
            "loss_denom_tokens": int(loss_denom_tokens),
            "group_end": bool(group_end),
            "group_id": int(group_id),
            "micro_id": int(micro_id),
            "bucket": entry["bucket"],
            "length": entry["length"],
            "pad_to_length": int(pad_to_length),
        }


class CounterLike(dict):
    def __missing__(self, key):
        self[key] = 0
        return 0


class DistributedTokenBucketBatchSampler:
    """Accelerate-sharded bucket sampler with exact token-budget groups.

    The sampler returns an interleaved global schedule:
    [micro0-rank0, micro0-rank1, ..., micro1-rank0, ...]. Accelerate's
    BatchSamplerShard(split_batches=False) then gives each process its rank's
    slice while preserving the same bucket and microbatch ordinal across ranks.
    """

    def __init__(
        self,
        dataset,
        num_replicas,
        rank,
        bucket_batch_sizes,
        target_tokens_per_step,
        seed=42,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_batch_sizes = bucket_batch_sizes
        self.target_tokens_per_step = target_tokens_per_step
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self._rank_batches())

    def __len__(self):
        return len(self._rank_batches())

    def _rank_batches(self):
        rng = random.Random(self.seed + self.epoch)
        bucket_to_indices = {"short": [], "mid": [], "long": []}
        for idx, entry in enumerate(self.dataset.entries):
            bucket_to_indices[entry["bucket"]].append(idx)

        microbatches = []
        for bucket in ("short", "mid", "long"):
            per_rank = self.bucket_batch_sizes[bucket]
            global_size = per_rank * self.num_replicas
            indices = sorted(bucket_to_indices[bucket], key=lambda i: (self.dataset.entries[i]["length"], i))
            if not indices:
                continue
            padding = (-len(indices)) % global_size
            indices = indices + [-1] * padding
            for start in range(0, len(indices), global_size):
                global_indices = indices[start : start + global_size]
                rank_indices = [
                    global_indices[r * per_rank : (r + 1) * per_rank]
                    for r in range(self.num_replicas)
                ]
                real = [i for i in global_indices if i >= 0]
                global_loss_tokens = sum(self.dataset.entries[i]["loss_tokens"] for i in real)
                max_length = max([self.dataset.entries[i]["length"] for i in real] or [1])
                microbatches.append({
                    "bucket": bucket,
                    "indices": rank_indices,
                    "global_loss_tokens": global_loss_tokens,
                    "max_length": max_length,
                })

        rng.shuffle(microbatches)

        batches = []
        group = []
        group_tokens = 0
        group_id = 0

        def flush_group():
            nonlocal group, group_tokens, group_id
            if not group:
                return
            for micro_id, micro in enumerate(group):
                group_end = micro_id == len(group) - 1
                for rank_indices in micro["indices"]:
                    batches.append([
                        (idx, group_tokens, group_end, group_id, micro_id, micro["max_length"])
                        for idx in rank_indices
                    ])
            group = []
            group_tokens = 0
            group_id += 1

        for micro in microbatches:
            group.append(micro)
            group_tokens += micro["global_loss_tokens"]
            if group_tokens >= self.target_tokens_per_step:
                flush_group()
        flush_group()
        return batches

    def coverage_receipt(self):
        seen = []
        for batch in self._rank_batches():
            for item in batch:
                seen.append(item[0])
        real = [i for i in seen if i >= 0]
        duplicates = len(real) - len(set(real))
        counts = {b: sum(1 for e in self.dataset.entries if e["bucket"] == b)
                  for b in ("short", "mid", "long")}
        padding = sum((-n) % (self.bucket_batch_sizes[b] * self.num_replicas)
                      for b, n in counts.items())
        return {"real_unique": len(set(real)), "omitted": len(self.dataset.entries)-len(set(real)),
                "duplicates": duplicates, "expected_padding": padding,
                "emitted_padding": sum(1 for i in seen if i < 0), "buckets": counts,
                "optimizer_groups": len({item[3] for batch in self._rank_batches() for item in batch})}


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def install_liger_backbone_kernels(model):
    """Swap RMSNorm and SwiGLU for Liger's fused kernels, and PROVE the swap took.

    Ported verbatim in behaviour from careers-qwen/train_ddp_lora.py, where this has run in
    production. The receipt is the load-bearing part: apply_liger_kernel_to_qwen3_5 is a
    monkey-patch, and a monkey-patch that silently fails to bind leaves the model running stock
    kernels while the log says otherwise. Walking the module tree afterwards and raising on any
    module that is NOT the Liger class is the difference between "we applied it" and "it is
    applied".
    """
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5

    cls = type(model).__name__
    if cls != "Qwen3_5ForCausalLM":
        raise RuntimeError(
            f"Liger backbone kernels expect Qwen3_5ForCausalLM, got "
            f"{type(model).__module__}.{cls}"
        )
    # rope / cross_entropy / fused_linear_cross_entropy stay False: this is a PORT of a
    # production-proven configuration, not the place to introduce an untested kernel combination.
    apply_liger_kernel_to_qwen3_5(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=True,
        swiglu=True,
        model=model,
    )
    text_model = model.model
    rms_norms = [text_model.norm]
    for layer in text_model.layers:
        rms_norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    unexpected_norms = [m._get_name() for m in rms_norms if m._get_name() != "LigerRMSNorm"]
    unexpected_mlps = [
        layer.mlp._get_name()
        for layer in text_model.layers
        if layer.mlp._get_name() != "LigerQwen3MoeSwiGLUMLP"
    ]
    if unexpected_norms or unexpected_mlps:
        raise RuntimeError(
            "Liger backbone patch receipt is incomplete: "
            f"unexpected_norms={unexpected_norms[:5]} "
            f"unexpected_mlps={unexpected_mlps[:5]}"
        )
    return {
        "architecture": cls,
        "liger_kernel": _package_version("liger-kernel"),
        "rms_norm": "LigerRMSNorm",
        "rms_norm_modules": len(rms_norms),
        "swiglu": "LigerQwen3MoeSwiGLUMLP",
        "swiglu_modules": len(text_model.layers),
    }


_QUARANTINE_MARKER_GLOB = "QUARANTINE*"


def _quarantined_digests():
    """sha256 digests of every corpus a marker or manifest declares un-trainable.

    Sourced from QUARANTINE_DIGESTS (a file of `sha256  # note` lines) named by the
    QUARANTINE_DIGESTS env var, so the list is data rather than code and can be updated by the
    corpus owner without touching the trainer.
    """
    out = {}
    # DEFAULTED, not merely read from the environment. This gate was INERT for the first hours of
    # its life because QUARANTINE_DIGESTS was never set anywhere — the code was present, the
    # commit was real, and the list it consulted was empty, so it refused nothing. Found by
    # treasurer 2026-08-02 by checking the RUNNING ENVIRONMENT rather than the code.
    # A control whose operative input is optional is a control that is off by default.
    _default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "careers-qwen", "governance", "QUARANTINE_DIGESTS.txt",
    )
    reg = os.environ.get("QUARANTINE_DIGESTS", "") or _default
    # FAIL CLOSED. Absence used to be silent; now it stops the run. An empty or missing registry
    # is indistinguishable at load time from "nothing is quarantined", and the whole point of this
    # gate is that the difference matters.
    if not os.path.isfile(reg):
        raise RuntimeError(
            f"REFUSE: quarantine digest registry not found at {reg}. This gate cannot be "
            f"silently inert. Restore the file, or set QUARANTINE_DIGESTS to a readable one."
        )
    if reg and os.path.isfile(reg):
        for line in open(reg):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) >= 1 and len(parts[0]) == 64:
                out[parts[0].lower()] = parts[1].strip(" #") if len(parts) > 1 else ""
    return out


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_not_quarantined(path):
    """Refuse to index a data file that is quarantined BY CONTENT or sits beside a marker.

    CONTENT first, and this is the load-bearing half. The marker check below is DIRECTORY-scoped,
    so it protects a location rather than a file: the same bytes copied to a second path are
    uncovered. Found 2026-08-02 — a credential-bearing corpus existed at two paths, sha
    cdb345826b6d6b11 identical, one under a marker and one not. The manifest caught the second
    copy at build time; this runtime gate would NOT have, which is the gap, since this gate exists
    precisely so the quarantine is mechanical rather than administrative.
    A digest follows the bytes wherever they are copied. A path protects a location.

    The loader discovers corpora by globbing a directory; it does not read
    PAIRS_MANIFEST. That made a quarantine ADMINISTRATIVE — a marker file the
    trainer never opened — so a file marked do-not-train was still trainable by
    pointing the loader at its directory. This makes the marker MECHANICAL.

    The marker is directory-scoped by its own wording ("do not tokenize, train
    on, or copy files in this directory"), so any marker in a file's directory
    quarantines that file.

    This RAISES rather than skipping. A silently-dropped corpus is worse than a
    stopped run: training would report success over data nobody chose, and the
    step count would look normal. Refusing is the whole point — a gate that
    quietly narrows the corpus is the failure it was built to prevent.
    """
    digests = _quarantined_digests()
    if not digests:
        raise RuntimeError(
            "REFUSE: quarantine digest registry is EMPTY. An empty registry and a missing one "
            "look identical to a corpus that is genuinely un-quarantined; this gate refuses "
            "rather than assume the benign reading."
        )
    if digests:
        d = _sha256_file(path)
        if d in digests:
            raise RuntimeError(
                f"REFUSE: {path} matches a QUARANTINED corpus by content (sha256 {d[:16]}...). "
                f"{digests[d]} A digest-keyed quarantine follows the bytes; renaming or copying "
                f"the file does not escape it."
            )

    directory = os.path.dirname(os.path.abspath(path))
    markers = sorted(glob.glob(os.path.join(directory, _QUARANTINE_MARKER_GLOB)))
    if markers:
        names = ", ".join(os.path.basename(m) for m in markers)
        raise RuntimeError(
            f"REFUSE: {path} lies in a QUARANTINED directory (marker: {names}). "
            f"A quarantine is mechanical, not advisory. Close the owning task and "
            f"remove the marker, or point the loader at sanctioned bytes."
        )


class CombinedSFTDataset(Dataset):
    """Combined SFT dataset — zero-footprint byte-offset indexing.
    Weighted sampling: 55% SFT, 30% CPT (constitutional+infra), 15% general.
    """

    def __init__(self, sft_dir, cpt_path, general_dir, tokenizer, max_seq):
        self.tokenizer = tokenizer
        self.max_seq = max_seq

        log.info("Building byte-offset indexes...")
        sft_files = sorted(glob.glob(os.path.join(sft_dir, "*.jsonl"))) if sft_dir and os.path.isdir(sft_dir) else []
        cpt_files = [cpt_path] if cpt_path and os.path.exists(cpt_path) else []
        general_files = sorted(glob.glob(os.path.join(general_dir, "*.jsonl"))) if general_dir and os.path.isdir(general_dir) else []

        self.sft_index = self._build_index(sft_files)
        self.cpt_index = self._build_index(cpt_files)
        self.general_index = self._build_index(general_files)

        total = len(self.sft_index) + len(self.cpt_index) + len(self.general_index)
        self.total_len = max(total, 1)
        log.info(f"Dataset: {len(self.sft_index)} SFT + {len(self.cpt_index)} CPT + "
                 f"{len(self.general_index)} General = {self.total_len}")

    def _build_index(self, file_paths):
        index = []
        for path in file_paths:
            if not os.path.isfile(path):
                continue
            _assert_not_quarantined(path)
            with open(path, 'rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        index.append((path, offset))
        return index

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Weighted sampling: 55% SFT, 30% CPT, 15% general
        r = random.random()
        if r < 0.55 and self.sft_index:
            source, is_cpt = self.sft_index, False
        elif r < 0.85 and self.cpt_index:
            source, is_cpt = self.cpt_index, True
        elif self.general_index:
            source, is_cpt = self.general_index, False
        elif self.sft_index:
            source, is_cpt = self.sft_index, False
        elif self.cpt_index:
            source, is_cpt = self.cpt_index, True
        else:
            return {"input_ids": [0] * self.max_seq, "labels": [-100] * self.max_seq}

        path, offset = source[idx % len(source)]
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.readline()

        try:
            data = json.loads(raw.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"input_ids": [self.tokenizer.pad_token_id] * self.max_seq,
                    "labels": [-100] * self.max_seq}

        if is_cpt:
            text = data.get("text", "")
            tokens = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            labels = list(tokens)
            assert len(tokens) <= self.max_seq, (
                f"CPT row exceeds max_seq: {len(tokens)}>{self.max_seq}; corpus must be pre-chunked"
            )
            return {"input_ids": tokens, "labels": labels, "is_dpo": False}
        elif "messages" in data:
            try:
                tokens, labels = _tokenize_sft_pair(data["messages"], self.tokenizer)
            except Exception:
                tokens = [self.tokenizer.pad_token_id] * self.max_seq
                labels = [-100] * self.max_seq
        elif "chosen" in data and "prompt" in data and "rejected" in data:
            # DPO item — tokenize both chosen and rejected
            chosen_msgs = [{"role": "user", "content": data["prompt"]},
                           {"role": "assistant", "content": data["chosen"]}]
            rejected_msgs = [{"role": "user", "content": data["prompt"]},
                             {"role": "assistant", "content": data["rejected"]}]
            try:
                tokens, labels = _tokenize_sft_pair(chosen_msgs, self.tokenizer)
                rej_tokens, rej_labels = _tokenize_sft_pair(rejected_msgs, self.tokenizer)
            except Exception:
                tokens = [self.tokenizer.pad_token_id] * self.max_seq
                labels = [-100] * self.max_seq
                rej_tokens = tokens[:]
                rej_labels = labels[:]

            # Pad/truncate rejected
            if len(rej_tokens) > self.max_seq:
                # NEVER TRUNCATE (Jesse, standing invariant). A cut row is a silently
                # corrupted training target: the model cannot tell what is missing.
                # Over-length input is a CORPUS defect — fix it upstream by chunking or
                # windowing, never by discarding tokens here.
                raise RuntimeError(
                    f"row exceeds max_seq: len(rej_tokens)={len(rej_tokens)} > {self.max_seq}; "
                    f"corpus must be pre-chunked or windowed — truncation is not permitted"
                )
            elif len(rej_tokens) < self.max_seq:
                pad_len = self.max_seq - len(rej_tokens)
                rej_tokens += [self.tokenizer.pad_token_id] * pad_len
                rej_labels += [-100] * pad_len

            # Pad/truncate chosen (handled below), store rejected
            if len(tokens) > self.max_seq:
                # NEVER TRUNCATE (Jesse, standing invariant). A cut row is a silently
                # corrupted training target: the model cannot tell what is missing.
                # Over-length input is a CORPUS defect — fix it upstream by chunking or
                # windowing, never by discarding tokens here.
                raise RuntimeError(
                    f"row exceeds max_seq: len(tokens)={len(tokens)} > {self.max_seq}; "
                    f"corpus must be pre-chunked or windowed — truncation is not permitted"
                )
            elif len(tokens) < self.max_seq:
                pad_len = self.max_seq - len(tokens)
                tokens += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len

            return {"input_ids": tokens, "labels": labels,
                    "rejected_input_ids": rej_tokens, "rejected_labels": rej_labels,
                    "is_dpo": True}
        else:
            tokens = [self.tokenizer.pad_token_id] * self.max_seq
            labels = [-100] * self.max_seq

        if len(tokens) > self.max_seq:
            # NEVER TRUNCATE (Jesse, standing invariant). A cut row is a silently
            # corrupted training target: the model cannot tell what is missing.
            # Over-length input is a CORPUS defect — fix it upstream by chunking or
            # windowing, never by discarding tokens here.
            raise RuntimeError(
                f"row exceeds max_seq: len(tokens)={len(tokens)} > {self.max_seq}; "
                f"corpus must be pre-chunked or windowed — truncation is not permitted"
            )
        elif len(tokens) < self.max_seq:
            pad_len = self.max_seq - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {"input_ids": tokens, "labels": labels, "is_dpo": False}


# (emitted name, environment name). The emitted names are the CONTRACT: they are what
# careers-qwen/post_cpt_pipeline.sh:47-54 and finalize_post_cpt_candidate.sh:44-48 read
# back. The first six are REQUIRED by both consumers; the rest are provenance and unknown
# keys are ignored by them.
_RUN_CONFIG_KEYS = (
    ("CPT_PATH_FROM_LOG", "CPT_DATA"),
    ("TRAIN_BASE", "MODEL_PATH"),
    ("TOTAL_STEPS", "TOTAL_STEPS"),
    ("SESSION_LIMIT", "SESSION_LIMIT"),
    ("WARMUP_STEPS", "WARMUP_STEPS"),
    ("LR", "LR"),
    ("LR_LORA", "LR_LORA"),
    ("CORPUS_INPUTS", "CORPUS_INPUTS"),
    ("MAX_SEQ", "MAX_SEQ"),
    ("EPOCHS", "EPOCHS"),
    ("RESUME_DELTA", "RESUME_DELTA"),
    ("SAVE_EVERY", "SAVE_EVERY"),
    ("CHECKPOINT_DCP", "CHECKPOINT_DCP"),
)


def _execed_environment():
    """The environment this process was EXECed with, per README Rule 5.

    os.environ is the LIVE mapping and any in-process assignment mutates it, so it can
    disagree with what the worker actually received. /proc/self/environ is the exec-time
    block and cannot be edited from inside the process, which is exactly why the rule
    names it. The two disagreeing is a finding to record, not a difference to smooth over.
    """
    with open("/proc/self/environ", "rb") as handle:
        raw = handle.read()
    execed = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        if separator:
            execed[name.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return execed


def capture_run_config(output_dir):
    """Write run_config.env beside the run's checkpoints. Rank 0, once per session.

    This file IS the run. careers-qwen/post_cpt_pipeline.sh:41-45 refuses to bake without
    it and states why: "The live trainer is the only authoritative source; this
    configuration cannot be reconstructed." Four consumers read it and none may default a
    value. Until now nothing wrote it -- four readers, zero producers -- so a completed run
    could not be baked at all.

    A key is emitted ONLY when this worker actually received it. Absence is preserved
    deliberately so a dropped variable reaches the consumer as MISSING and it aborts. A
    capture that filled in defaults would make a variable that never arrived read as
    configured, which is the failure this file exists to detect: run_4node_27b_cpt.sh:58-60
    records that LR and WARMUP_STEPS were not forwarded AT ALL until 2026-07-13, so every
    run before that trained at the trainer default no matter what the operator set.

    Failing to write is fatal on purpose. A run whose configuration was never captured
    cannot be baked, so dying in the first seconds is strictly cheaper than discovering it
    after the run has spent its hours -- which is what happened to cpt_prod_v4_repos_1ep.
    """
    execed = _execed_environment()
    lines = [
        "# run_config.env -- written by the trainer process that ran, from /proc/self/environ.",
        "# README Rule 5: captured once per SESSION; it cannot be reconstructed after exit.",
        "# A required name ABSENT here was never received by this worker. Do not add it by hand.",
    ]
    divergences = []
    for emitted_name, environment_name in _RUN_CONFIG_KEYS:
        if environment_name not in execed:
            continue
        value = execed[environment_name]
        live = os.environ.get(environment_name)
        if live is not None and live != value:
            divergences.append(f"{environment_name} exec={value!r} live={live!r}")
        lines.append(f"{emitted_name}={value}")
    lines.extend(f"# DIVERGENCE {entry}" for entry in divergences)

    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, "run_config.env")
    # A prior capture belongs to a prior session and is never silently overwritten: it is
    # the only record of how that session was configured.
    if os.path.exists(destination):
        for index in range(1, 1000):
            preserved = f"{destination}.session{index}"
            if not os.path.exists(preserved):
                os.rename(destination, preserved)
                print(f"  CAPTURE: preserved earlier capture as {preserved}", flush=True)
                break
        else:
            raise RuntimeError(f"cannot preserve prior {destination}: 999 sessions already kept")
    partial = destination + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    for entry in divergences:
        print(f"  CAPTURE DIVERGENCE: {entry}", flush=True)
    print(f"  CAPTURE: {destination} ({len(lines)} lines)", flush=True)
    return destination


def main():
    gc.collect(2)
    gc.freeze()
    gc.set_threshold(50_000, 500, 50)

    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta
    # FIX (2026-07-09): bind THIS rank's CUDA device BEFORE process-group init so NCCL does not
    # "Guess device ID based on global rank" — the un-set device is what hung the first collective
    # (exp6). device_id also opts into EAGER NCCL init (the RoCE fabric opens at init, recoverable,
    # instead of lazily at the first optimizer step). PyTorch's own log prescribed this exact fix.
    _local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(_local_rank)
    try:
        pg_timeout = InitProcessGroupKwargs(
            timeout=timedelta(hours=1),
            device_id=torch.device(f"cuda:{_local_rank}"),
        )
    except TypeError:
        # older accelerate: device_id unsupported on the kwargs — set_device above still binds it
        pg_timeout = InitProcessGroupKwargs(timeout=timedelta(hours=1))
    accelerator = Accelerator(kwargs_handlers=[pg_timeout])
    set_seed(42)

    # ── Config ──
    model_path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Huihui-Qwen3.5-35B-A3B-abliterated"))
    delta_path = os.environ.get("RESUME_DELTA", "")
    # FSDP2-native resumable checkpoint (DCP). Default ON — replaces the FSDP1 summon_full_params
    # 54GB-gather that OOMs. _is_dcp_ckpt: the resume target is a DCP checkpoint (COMPLETE + dcp/)
    # → load POST-prepare (needs the FSDP2 model to exist to place shards), skip the pre-prepare path.
    _use_dcp = os.environ.get("CHECKPOINT_DCP", "1") == "1"
    _is_dcp_ckpt = bool(delta_path) and os.path.exists(os.path.join(delta_path, "COMPLETE")) \
        and os.path.isdir(os.path.join(delta_path, "dcp"))
    sft_dir = os.environ.get("SFT_DIR", "/var/spark/isma/training/sft")
    cpt_data = os.environ.get("CPT_DATA", "/var/spark/isma/training/infra_soul_cpt.jsonl")
    general_dir = os.environ.get("GENERAL_DIR", "")
    output_dir = os.environ.get("OUTPUT_DIR", "/var/spark/models/taey-lora-v1")
    final_lora_dir = os.environ.get("FINAL_LORA_DIR", "")
    max_seq = int(os.environ.get("MAX_SEQ", "8192"))
    total_steps = int(os.environ.get("TOTAL_STEPS", "3000"))
    save_every = int(os.environ.get("SAVE_EVERY", "50"))
    session_limit = int(os.environ.get("SESSION_LIMIT", "250"))
    _expected_sft_samples = int(os.environ.get("EXPECTED_SFT_SAMPLES", "0"))
    if _expected_sft_samples < 0:
        raise RuntimeError("EXPECTED_SFT_SAMPLES must be zero or positive")
    _exact_sft_epoch = os.environ.get("EXACT_SFT_EPOCH", "0") == "1"
    _expected_real_samples = int(os.environ.get("EXPECTED_REAL_SAMPLES", "0"))
    # ONE definition of "is this run a bake/export rather than a training run", for the same
    # reason EPOCHS is defined once below. A bake loads a checkpoint and writes an artifact: it
    # takes no optimizer steps, so TOTAL_STEPS=1 is a formality there and any gate reasoning about
    # the training horizon is a FALSE POSITIVE on that path. It was previously computed inside the
    # LR-scheduler block, where it was both conditionally scoped and invisible to the horizon
    # contract added later — which would then have blocked every packed bake, reproducing the
    # 2026-07-28 warmup-guard failure documented at its own use site.
    _is_bake = bool(os.environ.get("BAKE_TO_HF", "") or os.environ.get("EXPORT_DCP", ""))
    # ONE definition of EPOCHS, read once and closed over by everything below.
    # It is defined HERE rather than at each use because the first version read it inside
    # _make_optim only, and the END-OF-RUN dose check 900 lines away kept its own one-epoch
    # assumption — the run completed all 30 steps and then died at the final accounting with
    # `real=120/60`, which is 2 epochs of 60 measured against a 1-epoch expectation. Two places
    # deriving the same quantity independently is the bug; one definition is the fix.
    _epochs = int(os.environ.get("EPOCHS", "1"))
    if _epochs < 1:
        raise RuntimeError(f"EPOCHS={_epochs} must be >= 1")
    # Capture this session's configuration before anything else can fail, because it cannot
    # be recovered afterwards. Guarded on _is_bake: a bake reloads a completed run and must
    # never overwrite that run's capture with the bake's own environment -- the training
    # capture is the provenance the bake is being judged against. RANK is what torchrun sets
    # per worker, so exactly one process writes.
    if not _is_bake and int(os.environ.get("RANK", "0")) == 0:
        capture_run_config(output_dir)
    # MANDATORY for cumulative LoRA resume (tutor 2026-07-25). Opt-in was the defect.
    # Without this flag the adapter load is `if exists / elif exists` with NO else: a rank
    # that cannot find the adapter sets loaded_delta=False and SILENTLY CONTINUES, training
    # from BASE weights while rank0 trains from the adapter. Under FSDP2 every rank loads
    # for itself (see _load_delta_here below — `or _fsdp_v2`), so a partially-distributed
    # adapter breaks cumulative lineage on 3 of 4 ranks and produces a checkpoint whose
    # provenance receipt claims a lineage it does not have. That is undetectable from the
    # artifact afterwards, which makes it the expensive failure rather than a loud one.
    # Occasion: module3_exact159_adapter_hf was present on .68 and ABSENT on .80/.12/.19
    # (found by tutor-codex on a four-node probe). Nothing would have stopped the launch.
    # So: if you are resuming a non-DCP adapter in LORA_MODE, parity is REQUIRED, not asked
    # for. The env var can still turn it ON elsewhere; it can no longer turn it OFF here.
    _require_lora_init_parity = os.environ.get("REQUIRE_LORA_INIT_PARITY", "0") == "1"
    if (os.environ.get("LORA_MODE", "0") == "1"
            and delta_path
            and not _is_dcp_ckpt):
        if not _require_lora_init_parity:
            log.info("REQUIRE_LORA_INIT_PARITY forced ON: cumulative LoRA resume "
                     "(a rank that cannot find the adapter would silently train from base)")
        _require_lora_init_parity = True
    _nsys_profile_step = int(os.environ.get("NSYS_PROFILE_STEP", "0"))
    if _nsys_profile_step < 0:
        raise RuntimeError(
            "NSYS_PROFILE_STEP must be zero or a positive absolute optimizer step"
        )
    # Rank-0-only profiling measures HOW MUCH time goes to collectives; it cannot say WHICH rank
    # is late. Under a collective, every rank blocks until the slowest arrives, so a straggler
    # shows up as the rank spending the LEAST time waiting in NCCL while its peers spend the most
    # — a comparison that needs a timeline from every rank, not one. Default off: unset, behaviour
    # is byte-identical to rank-0-only.
    _nsys_profile_all_ranks = os.environ.get("NSYS_PROFILE_ALL_RANKS", "0") == "1"
    # Memory-guard thresholds (tutor 2026-07-22). Two bands because the free-memory drop is allocator
    # CACHE (hoarded reserved blocks on variable batch sizes), not live allocation — so when free gets
    # tight we RECLAIM the cache first and only exit if that isn't enough:
    #   RECLAIM band: free < this -> gc + empty_cache to release hoarded blocks, then re-measure.
    #   EXIT floor:   free STILL < this after reclaim -> checkpoint + reboot (genuine ceiling — even
    #                 releasing the cache can't recover contiguous space; only a reboot defrags).
    # 25GB floor sits just above the 20GB where session 2's 5GB all-gather died. 45GB reclaim band
    # triggers release well before the floor. Tune via MEM_RECLAIM_FREE_GB / MEM_EXIT_FREE_GB.
    _MEM_RECLAIM_FREE_B = float(os.environ.get("MEM_RECLAIM_FREE_GB", "45")) * 1e9
    _MEM_EXIT_FREE_B = float(os.environ.get("MEM_EXIT_FREE_GB", "25")) * 1e9
    # Keystone layers configurable via env var (JSON array) or default
    keystone_env = os.environ.get("KEYSTONE_LAYERS", "")
    if keystone_env:
        keystone_layers = json.loads(keystone_env)
    else:
        keystone_layers = KEYSTONE_LAYERS

    warmup_steps = int(os.environ.get("WARMUP_STEPS", "25"))

    if accelerator.is_main_process:
        log.info(f"=== PALIOS-TAEY v3: Hybrid LoRA + ESFT ===")
        log.info(f"Model: {model_path}")
        log.info(f"Keystone layers: {keystone_layers}")
        log.info(f"Seq={max_seq}, session={session_limit}, save_every={save_every}")
        mem = torch.cuda.mem_get_info()
        log.info(f"UMA: free={mem[0]/1e9:.1f}GB total={mem[1]/1e9:.1f}GB")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Canonical tool-call chat template (train == inference) ──
    # The base model's bundled template instructs XML tool calls (<function=>);
    # serving uses the Hermes-JSON canonical template. Load the SAME canonical
    # template here so the train-time prompt (tools schema + tool-call format
    # instruction) is byte-identical to what serving renders — closing the
    # Cycle-C train/inference gap and grounding the tools schema (#1). One template
    # file, two consumers (train here, serve on the host). CHAT_TEMPLATE=none keeps
    # the bundled template (e.g. pure-CPT runs with no tool structure).
    _ct = os.environ.get(
        "CHAT_TEMPLATE",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "inference", "qwen3.5-tooluse.jinja"),
    )
    if _ct and _ct.lower() != "none" and os.path.isfile(_ct):
        tokenizer.chat_template = open(_ct, encoding="utf-8").read()
        if accelerator.is_main_process:
            log.info(f"Chat template: loaded canonical {_ct} (train==inference tool-call format)")
    elif _exact_sft_epoch:
        raise RuntimeError(
            f"EXACT_SFT_EPOCH requires the canonical chat template, missing: {_ct}"
        )
    elif accelerator.is_main_process:
        log.info(f"Chat template: using model bundled template (CHAT_TEMPLATE={_ct})")

    # ── Rank-split model loading ──
    # UMA constraint: 119GB system memory (GPU firmware reserves 9.5GB of 128.5GB UMA).
    # Loading 71GB model to CUDA creates 71GB page cache (mmap) + 71GB CUDA = 142GB → OOM.
    # Solution: rank 0 loads to CPU (zero-copy mmap from safetensors, no CUDA allocation).
    # FSDP sync_module_states broadcasts rank 0's CPU params → CUDA on all ranks during wrap.
    # Other ranks: device_map="meta" = zero memory.
    # FSDP: only rank0 loads real weights (sync_module_states broadcasts to meta ranks during wrap).
    # DDP (no fsdp_plugin): NO such broadcast — EVERY rank must load the full real model to CPU
    # (accelerator.prepare() then moves each rank's copy to its GPU). 9B bf16 ≈ 18GB/node, fits 120GB.
    _is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    # FSDP2 has NO sync_module_states — the rank0-real/others-meta trick leaves the UNWRAPPED top-level
    # params (embed_tokens/norm/lm_head) on meta on the workers → fully_shard's _validate_no_meta_params
    # raises. So under FSDP2 EVERY rank loads real weights (each node has 128GB; 27B bf16 ≈ 54GB transient
    # CPU load, then fully_shard shards to ~13.5GB/rank). Only FSDP1 (sync_module_states) uses meta workers.
    _fsdp_v2 = _is_fsdp and int(getattr(accelerator.state.fsdp_plugin, "fsdp_version", 1)) == 2
    if accelerator.is_main_process or not _is_fsdp or _fsdp_v2:
        log.info(f"Rank {accelerator.process_index}: loading FULL model to CPU (real weights; fsdp={_is_fsdp} v2={_fsdp_v2})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 2026-07-13 FP32 MASTER (5-lane Family consult root-cause): the null runs were bf16
            # masters — Adafactor's sub-ULP updates at lr<=1e-5 rounded to zero on write-back
            # (model.norm bit-identical, decoder ~1e-4 = bf16 ULP). Load fp32 so the sharded master
            # the optimizer updates is fp32; MixedPrecisionPolicy(param_dtype=bf16) casts to bf16 for
            # compute only. Env FP32_MASTER=0 reverts to the (broken) bf16-master path.
            torch_dtype=(torch.float32 if os.environ.get("FP32_MASTER", "0") == "1" else torch.bfloat16),
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        vm = psutil.virtual_memory()
        log.info(f"Rank {accelerator.process_index}: model loaded to CPU. RAM used={vm.used/1e9:.1f}GB free={vm.available/1e9:.1f}GB")
    else:
        log.info(f"Rank {accelerator.process_index}: loading model on meta device (zero memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 2026-07-13 FP32 MASTER (5-lane Family consult root-cause): the null runs were bf16
            # masters — Adafactor's sub-ULP updates at lr<=1e-5 rounded to zero on write-back
            # (model.norm bit-identical, decoder ~1e-4 = bf16 ULP). Load fp32 so the sharded master
            # the optimizer updates is fp32; MixedPrecisionPolicy(param_dtype=bf16) casts to bf16 for
            # compute only. Env FP32_MASTER=0 reverts to the (broken) bf16-master path.
            torch_dtype=(torch.float32 if os.environ.get("FP32_MASTER", "0") == "1" else torch.bfloat16),
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map="meta",
        )

    model.config.use_cache = False
    # output_router_logits skipped — dense Qwen3.5-9B has no router
    # gradient checkpointing ENABLED for full-FT — without this the autograd
    # graph for all 8.95B trainable params at seq=8192 holds ~12GB activations
    # per rank for the backward pass, which silently OOMs UMA at first training
    # step (manifests as 96% GPU spin-wait at 18W). The 11x-slower comment in
    # train_fsdp_v3.py applied to the LoRA case where activation memory was
    # tiny; for full-FT this is the standard fix and matches every other
    # full-FT script in the repo (train_cpt_qwen35_dense.py:96,
    # train_recovery_sft_qwen35_dense.py:88, train_cpt_fsdp.py:117, etc.).
    # ── exp9 (2026-07-10, 3-lane fetch consult: grok+chatgpt-DR+perplexity-DR CONVERGED) ──
    # REMOVED the HF gradient_checkpointing_enable() call. All 3 lanes (live PyTorch/accelerate
    # docs) agree it is the CULPRIT: under FSDP2 (fully_shard) + the Qwen3.6 multimodal outer
    # wrapper it only sets a Python attribute (.gradient_checkpointing=True propagates to all 64
    # layers — Observed) but does NOT install torch.utils.checkpoint around the inner
    # Qwen3_5DecoderLayer forward, so activations are never recomputed (measured: 119GB peak,
    # +13GB per batch-doubling = activations NOT checkpointed). It also CONFLICTS with accelerate's
    # own AC. Fix: rely on accelerate's fsdp_activation_checkpointing=true (config), which applies
    # checkpoint_wrapper(NO_REENTRANT) to the layer children INSIDE prepare(), BEFORE fully_shard
    # (the mandatory order). Verified post-prepare by the CheckpointWrapper count below.
    # if the count is 0, escalate to a manual pre-prepare apply_activation_checkpointing.
    log.info("AC: HF gradient_checkpointing_enable REMOVED (exp9) — using accelerate fsdp_activation_checkpointing")

    # ── LIGER BACKBONE KERNELS (2026-08-01) ──
    # Ported from careers-qwen/train_ddp_lora.py:235, where these have run in production and are
    # recorded in the stage-2 SFT run manifest (liger_kernel 0.8.1, LigerRMSNorm x129,
    # LigerQwen3MoeSwiGLUMLP x64). The SFT path had them; THIS path never did. Same model family,
    # same hardware, one trainer optimised and the other not — the cost was paid on every CPT step
    # since the kernels were installed.
    # Applied HERE: after from_pretrained, before accelerate.prepare/fully_shard, because the patch
    # swaps module classes and must happen while the module tree is still local and unwrapped.
    # rope/cross_entropy/fused_linear_cross_entropy stay FALSE to match the configuration already
    # proven in production rather than introducing an untested combination alongside a port.
    # LIGER=0 disables it, so an A/B is one env var and the previous behaviour is one flag away.
    if os.environ.get("LIGER", "1") == "1":
        try:
            _liger_receipt = install_liger_backbone_kernels(model)
            log.info(f"LIGER: {_liger_receipt}")
        except Exception as _e:
            # Loud, not silent. A throughput optimisation that quietly no-ops leaves the run
            # slower than it should be with nothing in the log to say why — which is how this
            # gap survived in the first place.
            raise RuntimeError(
                f"LIGER install failed: {type(_e).__name__}: {_e}. "
                f"Set LIGER=0 to run without the fused kernels."
            ) from _e
    else:
        log.info("LIGER: DISABLED by env (LIGER=0) — stock RMSNorm/SwiGLU")

    # ── LORA_MODE (2026-07-21) — module training on the FROZEN CPT base ──
    # Attach PEFT here (BEFORE accelerator.prepare/fully_shard, so the adapter modules exist when
    # FSDP2 shards each decoder layer and requires_grad is set pre-wrap). Everything downstream is
    # UNCHANGED PRODUCTION: the Adafactor build already filters `p.requires_grad` (-> LoRA-only),
    # _save_checkpoint_dcp already writes the SHARDED (full_state_dict=False) DCP that does NOT
    # all-gather — the path that avoids "the wedge" — and RESUME_DELTA / SESSION_LIMIT / the NCCL
    # fabric / run_4node_27b_cpt.sh / run_till_done_v3.sh all apply as-is.
    # LORA_MODE unset => byte-identical legacy full-FT behavior.
    _lora_mode = os.environ.get("LORA_MODE", "0") == "1"
    if _lora_mode:
        from peft import LoraConfig, get_peft_model
        _targets = os.environ.get(
            "LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj").split(",")
        model = get_peft_model(model, LoraConfig(
            r=int(os.environ.get("LORA_R", "16")),
            lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
            lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
            target_modules=_targets, bias="none", task_type="CAUSAL_LM",
            modules_to_save=[]))          # purely additive — base stays bit-identical
        # fail-loud: nothing outside the adapter may be trainable
        _bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
        if _bad:
            raise SystemExit(f"ABORT LORA_MODE: {len(_bad)} non-LoRA params trainable, e.g. {_bad[:3]}")
        if accelerator.is_main_process:
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_to = sum(p.numel() for p in model.parameters())
            log.info(f"LORA: {n_tr/1e6:.1f}M trainable / {n_to/1e9:.2f}B total "
                     f"({100*n_tr/max(n_to,1):.3f}%) targets={_targets}")
    else:
        # ── FULL-PARAMETER SFT — every weight is trainable ──
        # Per TOOLS.md Phase 1 plan ("Phase 1: SFT (~70K tools+chat examples,
        # 2-3 epochs)") and Perplexity DR diagnosis 2026-04-28, this is
        # full-parameter SFT, not LoRA. We unfreeze every parameter explicitly
        # before FSDP wraps the model so FSDP respects requires_grad=True
        # uniformly.
        for param in model.parameters():
            param.requires_grad_(True)
        if accelerator.is_main_process:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            log.info(f"FULL-FT: {n_trainable/1e9:.2f}B trainable / {n_total/1e9:.2f}B total")

    # ── NCCL warm-up ──
    dist.barrier()
    if accelerator.is_main_process:
        log.info("NCCL connections established")

    # ── Checkpoint reload ──
    resume_step = 0
    if _require_lora_init_parity and (not delta_path or _is_dcp_ckpt):
        raise RuntimeError(
            "REQUIRE_LORA_INIT_PARITY requires a non-DCP LoRA adapter in RESUME_DELTA"
        )
    if delta_path and not _is_dcp_ckpt:
        # (DCP checkpoints resume POST-prepare — see the _load_checkpoint_dcp call after prepare.)
        adapter_file = os.path.join(delta_path, "adapter_model.safetensors")
        meta_file = os.path.join(delta_path, "trainer_meta.pt")
        _load_delta_here = accelerator.is_main_process or not _is_fsdp or _fsdp_v2
        if _require_lora_init_parity and not _lora_mode:
            raise RuntimeError("REQUIRE_LORA_INIT_PARITY requires LORA_MODE=1")
        if _require_lora_init_parity and _is_fsdp and not _fsdp_v2:
            raise RuntimeError(
                "REQUIRE_LORA_INIT_PARITY requires every rank to hold real pre-FSDP weights"
            )
        initial_fingerprint = None
        if _require_lora_init_parity:
            initial_fingerprint, _ = _lora_fingerprint(model)
        loaded_delta = False
        loaded_lora_tensor_count = 0
        if _load_delta_here:
            # Universal resume: try trainable_weights.safetensors first (covers ALL configs)
            universal_file = os.path.join(delta_path, "trainable_weights.safetensors")
            if os.path.exists(universal_file):
                log.info(
                    f"Rank {accelerator.process_index}: loading trainable weights from {delta_path}..."
                )
                from safetensors.torch import load_file
                all_state = load_file(universal_file)
                lora_keys = {k: v for k, v in all_state.items() if 'lora_' in k.lower()}
                other_keys = {k: v for k, v in all_state.items() if 'lora_' not in k.lower()}
                if lora_keys:
                    from peft import set_peft_model_state_dict
                    load_result = set_peft_model_state_dict(
                        model, lora_keys, adapter_name="default"
                    )
                    missing_lora = [
                        key for key in load_result.missing_keys if "lora_" in key.lower()
                    ]
                    unexpected_lora = [
                        key for key in load_result.unexpected_keys if "lora_" in key.lower()
                    ]
                    if missing_lora or unexpected_lora:
                        raise RuntimeError(
                            "LoRA checkpoint load was incomplete: "
                            f"missing={missing_lora[:5]} unexpected={unexpected_lora[:5]}"
                        )
                    loaded_lora_tensor_count = len(lora_keys)
                    log.info(f"  LoRA: {len(lora_keys)} tensors via set_peft_model_state_dict")
                if other_keys:
                    model.load_state_dict(other_keys, strict=False)
                    log.info(f"  Non-LoRA: {len(other_keys)} tensors via load_state_dict")
                log.info(f"  Total: {len(all_state)} trainable tensors loaded")
                del all_state, lora_keys, other_keys
                gc.collect()
                loaded_delta = True
            elif os.path.exists(adapter_file):
                # Legacy fallback: separate per-type files
                log.info(
                    f"Rank {accelerator.process_index}: loading legacy checkpoint from {delta_path}..."
                )
                from safetensors.torch import load_file
                from peft import set_peft_model_state_dict
                adapter_state = load_file(adapter_file)
                load_result = set_peft_model_state_dict(
                    model, adapter_state, adapter_name="default"
                )
                missing_lora = [
                    key for key in load_result.missing_keys if "lora_" in key.lower()
                ]
                unexpected_lora = [
                    key for key in load_result.unexpected_keys if "lora_" in key.lower()
                ]
                if missing_lora or unexpected_lora:
                    raise RuntimeError(
                        "LoRA checkpoint load was incomplete: "
                        f"missing={missing_lora[:5]} unexpected={unexpected_lora[:5]}"
                    )
                loaded_lora_tensor_count = len(adapter_state)
                log.info(f"  Legacy PEFT: {len(adapter_state)} tensors")
                del adapter_state; gc.collect()
                for fname, label in [("router_gates.safetensors", "router"),
                                      ("expert_weights.safetensors", "expert")]:
                    fpath = os.path.join(delta_path, fname)
                    if os.path.exists(fpath):
                        from safetensors.torch import load_file as lf
                        st = lf(fpath)
                        model.load_state_dict(st, strict=False)
                        log.info(f"  Legacy {label}: {len(st)} tensors")
                        del st; gc.collect()
                loaded_delta = True
            if os.path.exists(meta_file):
                meta = torch.load(meta_file, map_location="cpu", weights_only=False)
                resume_step = meta.get("step", 0)
                log.info(f"Resuming from step {resume_step}")
                del meta
            else:
                # Fallback: parse step from directory name (checkpoint-300 → 300)
                import re
                m = re.search(r'checkpoint-(\d+)', delta_path)
                if m:
                    resume_step = int(m.group(1))
                    log.info(f"Resuming from step {resume_step} (parsed from directory name)")
        if _require_lora_init_parity:
            if not loaded_delta:
                raise RuntimeError(
                    f"LoRA initialization parity failed: no loadable adapter in {delta_path}"
                )
            local_fingerprint, local_tensor_count = _lora_fingerprint(model)
            if loaded_lora_tensor_count != local_tensor_count:
                raise RuntimeError(
                    "LoRA initialization parity failed: "
                    f"loaded_tensors={loaded_lora_tensor_count} "
                    f"trainable_tensors={local_tensor_count}"
                )
            if local_fingerprint == initial_fingerprint:
                raise RuntimeError(
                    "LoRA initialization parity failed: adapter load did not change trainable tensors"
                )
            local_receipt = torch.tensor(
                [local_tensor_count, *bytes.fromhex(local_fingerprint)],
                dtype=torch.long,
                device=accelerator.device,
            )
            gathered_receipts = [
                torch.empty_like(local_receipt) for _ in range(accelerator.num_processes)
            ]
            dist.all_gather(gathered_receipts, local_receipt)
            receipts = [
                (int(receipt[0].item()), bytes(receipt[1:].tolist()).hex())
                for receipt in gathered_receipts
            ]
            if len(set(receipts)) != 1:
                raise RuntimeError(
                    "LoRA initialization parity failed: "
                    f"per-rank receipts={receipts}"
                )
            if accelerator.is_main_process:
                log.info(
                    "LORA INIT PARITY: "
                    f"ranks={accelerator.num_processes} tensors={local_tensor_count} "
                    f"sha256={local_fingerprint}"
                )

    dist.barrier()
    resume_tensor = torch.tensor([resume_step], dtype=torch.long, device="cuda")
    dist.broadcast(resume_tensor, src=0)
    resume_step = resume_tensor.item()

    # ── Full-parameter SFT — no FREEZE_CONFIG branches ──
    # All params already set to requires_grad=True after model load. No
    # router/expert categorization (no MoE on dense). No LoRA-vs-base
    # safety checks (no LoRA on full FT). Single trainable surface.
    router_params = []   # kept as empty so the optimizer-builder code below
    expert_params = []   # doesn't break; both groups are zero-length here.

    # ── FSDP wrapping (transformer-block layer only — no LoRA wrap) ──
    import functools
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    layer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3_5DecoderLayer},
    )
    # DDP (MULTI_GPU) has no fsdp_plugin — full model resident per node, no wrap policy.
    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = layer_policy
        if accelerator.is_main_process:
            log.info("FSDP wrap policy: Qwen3_5DecoderLayer (full-param FT, no LoRA wrap)")
    elif accelerator.is_main_process:
        log.info("DDP mode: no FSDP wrap policy (full 9B resident per node)")

    # ── Dataset selection: SFT (Phase 1) vs CPT (Phase 2) ─────────────────
    # SFT mode: SFT_DIR points at a dir with *.jsonl messages-format data.
    #           Use BucketSFTDataset (length-sorted, dynamic-padding bucket).
    # CPT mode: SFT_DIR is empty AND CPT_DATA points at a jsonl with
    #           {"text": "..."} entries. Use BucketCPTDataset by default:
    #           length buckets, dynamic padding, token-average loss, and
    #           token-budget accumulation. Set CPT_BUCKETING=0 only for a
    #           diagnostic rollback to the old random CombinedSFTDataset path.
    cpt_bucket_mode = False
    _bucket_groups_by_epoch = None
    # PACKED mode: pre-tokenized fixed-length blocks (uniform shape → no allocator fragmentation →
    # fixes the first-step OOM + zero padding waste). Triggered by CPT_PACKED=1 or a "packed" filename.
    _packed = bool(cpt_data) and (os.environ.get("CPT_PACKED", "0") == "1"
                                  or "packed" in os.path.basename(cpt_data))
    _natural_sft_mode = False
    _fixed_packed_cpt = False
    if cpt_data and _packed and (not sft_dir or not os.path.isdir(sft_dir)):
        _fixed_packed_cpt = True
        if accelerator.is_main_process:
            log.info(f"CPT mode: dataset = PackedCPTDataset(cpt_path={cpt_data}) "
                     f"[FIXED-LENGTH PACKING — uniform shape, no bucketing, BATCH_SIZE_PER_RANK]")
        dataset = PackedCPTDataset(cpt_data)
        cpt_bucket_mode = False   # → else-branch DistributedSampler + fixed BATCH_SIZE_PER_RANK
    elif cpt_data and (not sft_dir or not os.path.isdir(sft_dir)):
        if os.environ.get("CPT_BUCKETING", "1") == "0":
            if accelerator.is_main_process:
                log.info(f"CPT mode: dataset = CombinedSFTDataset(cpt_path={cpt_data}) [BUCKETING DISABLED]")
            dataset = CombinedSFTDataset(
                sft_dir="", cpt_path=cpt_data, general_dir="",
                tokenizer=tokenizer, max_seq=max_seq,
            )
        else:
            if accelerator.is_main_process:
                log.info(f"CPT mode: dataset = BucketCPTDataset(cpt_path={cpt_data})")
            dataset = BucketCPTDataset(cpt_data, tokenizer, max_seq)
            cpt_bucket_mode = True
    else:
        sft_jsonl = os.environ.get(
            "SFT_JSONL",
            os.path.join(sft_dir, "tools_sft.jsonl"),
        )
        dataset = BucketSFTDataset(
            sft_jsonl,
            tokenizer,
            max_seq,
            strict_one_sample_per_row=_exact_sft_epoch,
        )
        _natural_sft_mode = True
        if _expected_sft_samples:
            if len(dataset) != _expected_sft_samples:
                raise RuntimeError(
                    "SFT sample-count contract failed: "
                    f"tokenized_samples={len(dataset)} "
                    f"expected={_expected_sft_samples}"
                )
            if accelerator.is_main_process:
                log.info(
                    "SFT SAMPLE CONTRACT: "
                    f"tokenized_samples={len(dataset)} "
                    f"max_seq={max_seq}"
                )

    if _exact_sft_epoch:
        if not _natural_sft_mode:
            raise RuntimeError("EXACT_SFT_EPOCH is only valid for natural SFT")
        if not _lora_mode:
            raise RuntimeError("EXACT_SFT_EPOCH requires LORA_MODE=1")
        if not _require_lora_init_parity:
            raise RuntimeError("EXACT_SFT_EPOCH requires REQUIRE_LORA_INIT_PARITY=1")
        if not _use_dcp:
            raise RuntimeError("EXACT_SFT_EPOCH requires CHECKPOINT_DCP=1")
        if not final_lora_dir:
            raise RuntimeError("EXACT_SFT_EPOCH requires FINAL_LORA_DIR")
        if _expected_real_samples <= 0:
            raise RuntimeError("EXACT_SFT_EPOCH requires EXPECTED_REAL_SAMPLES > 0")
        if os.environ.get("LANE_WEIGHTS", ""):
            raise RuntimeError("EXACT_SFT_EPOCH forbids replacement-sampling LANE_WEIGHTS")
        exact_batch_per_rank = int(os.environ.get("BATCH_SIZE_PER_RANK", "4"))
        if exact_batch_per_rank != 1:
            raise RuntimeError(
                "EXACT_SFT_EPOCH requires BATCH_SIZE_PER_RANK=1 so every real row "
                "has one equal sequence-mean coefficient"
            )
        exact_global_batch = exact_batch_per_rank * accelerator.num_processes
        dataset = ExactEpochDataset(
            dataset,
            expected_real_samples=_expected_real_samples,
            global_batch_size=exact_global_batch,
        )
        if accelerator.is_main_process:
            log.info(
                "EXACT SFT DATASET: "
                f"real={dataset.real_samples} padding={dataset.padding_samples} "
                f"total={len(dataset)} global_batch={exact_global_batch}"
            )

    # Dynamic-padding collate: pad to BATCH-MAX (not max_seq), aligned to a
    # multiple of 64 for tensor-core friendliness. Per Perplexity DR
    # (SFT_PHASE1_ISSUES.md Issue 6) this is the actual throughput win that
    # bucket batching enables — short batches stay short, long batches grow.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    PAD_TO_MULTIPLE_OF = 64

    def _round_up(n, k):
        return ((n + k - 1) // k) * k

    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        if "pad_to_length" in batch[0]:
            max_len = max(max_len, max(int(b["pad_to_length"]) for b in batch))
        target = _round_up(max_len, PAD_TO_MULTIPLE_OF)
        n = len(batch)
        input_ids = torch.full((n, target), pad_id, dtype=torch.long)
        labels = torch.full((n, target), -100, dtype=torch.long)
        attention_mask = torch.zeros((n, target), dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            lbl = b["labels"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_dpo": False,
        }
        if "is_exact_padding" in batch[0]:
            out["is_exact_padding"] = torch.tensor(
                [1 if b["is_exact_padding"] else 0 for b in batch],
                dtype=torch.long,
            )
        if "loss_denom_tokens" in batch[0]:
            denoms = [int(b["loss_denom_tokens"]) for b in batch]
            if not denoms or min(denoms) <= 0 or len(set(denoms)) != 1:
                raise RuntimeError("CPT bucket group denominator must be identical and positive")
            if any(bool(b.get("is_padding", False)) and any(v != -100 for v in b["labels"])
                   for b in batch):
                raise RuntimeError("CPT bucket padding labels must be fully masked")
            if any(not b.get("is_padding", False) and not any(v != -100 for v in b["labels"])
                   for b in batch):
                raise RuntimeError("CPT bucket real rows must contribute loss")
            out["is_padding"] = torch.tensor(
                [1 if b.get("is_padding", False) else 0 for b in batch], dtype=torch.long
            )
            out["loss_denom_tokens"] = torch.tensor(
                [int(b["loss_denom_tokens"]) for b in batch], dtype=torch.long
            )
            out["group_end"] = torch.tensor(
                [1 if b["group_end"] else 0 for b in batch], dtype=torch.long
            )
            out["group_id"] = torch.tensor([int(b["group_id"]) for b in batch], dtype=torch.long)
            out["micro_id"] = torch.tensor([int(b["micro_id"]) for b in batch], dtype=torch.long)
            out["length"] = torch.tensor([int(b["length"]) for b in batch], dtype=torch.long)
            out["pad_to_length"] = torch.tensor([int(b["pad_to_length"]) for b in batch], dtype=torch.long)
            out["bucket"] = [b["bucket"] for b in batch]
        return out

    if cpt_bucket_mode:
        bucket_batch_sizes = {
            "short": int(os.environ.get("CPT_SHORT_BATCH", "16")),
            "mid": int(os.environ.get("CPT_MID_BATCH", "4")),
            "long": int(os.environ.get("CPT_LONG_BATCH", "1")),
        }
        token_budget_per_step = int(os.environ.get("TOKEN_BUDGET_PER_STEP", "262144"))
        sampler = DistributedTokenBucketBatchSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            bucket_batch_sizes=bucket_batch_sizes,
            target_tokens_per_step=token_budget_per_step,
            seed=42,
        )
        _coverage = sampler.coverage_receipt()
        _bucket_groups_by_epoch = []
        for _epoch in range(_epochs):
            sampler.set_epoch(_epoch)
            _bucket_groups_by_epoch.append(len({item[3] for batch in sampler._rank_batches() for item in batch}))
        sampler.set_epoch(0)
        if (_coverage["real_unique"] != len(dataset) or _coverage["omitted"] or
                _coverage["duplicates"] or _coverage["emitted_padding"] != _coverage["expected_padding"] or
                _coverage["optimizer_groups"] <= 0):
            raise RuntimeError(f"CPT BUCKET COVERAGE FAILED: {_coverage}")
        if accelerator.is_main_process:
            log.info(f"CPT BUCKET COVERAGE PASS: {_coverage}")
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=0,
        )
        if accelerator.is_main_process:
            log.info(
                "CPT bucket batching: "
                f"short<2K batch/rank={bucket_batch_sizes['short']}, "
                f"mid2-8K batch/rank={bucket_batch_sizes['mid']}, "
                f"long8-16K batch/rank={bucket_batch_sizes['long']}, "
                f"target_tokens/optimizer_step={token_budget_per_step}"
            )
    elif _fixed_packed_cpt:
        # PACKED CPT (2026-07-12 DOUBLE-SHARD FIX): do NOT build a manual DistributedSampler here.
        # accelerator.prepare() ALREADY shards the dataloader across processes. A manual
        # DistributedSampler(num_replicas=world) on TOP of that double-shards: 12255 blocks / 4
        # (DistributedSampler) / 4 (accelerate) → each rank sees only 12255/16 ≈ 766 samples/epoch,
        # so one "epoch" was only ~192 optimizer-steps = 3072 blocks = 1/4 of the corpus, and with
        # shuffle=False it was the SAME 1/4 every epoch (3/4 NEVER trained). Proven by DCP metadata:
        # step250 epoch=1 data_pos=58 → 192-step epoch. Let accelerate do the SINGLE correct shard:
        # plain DataLoader → 12255/4 = 3064/rank → batch 4 → ~766 steps/epoch = FULL corpus coverage.
        train_batch_size = int(os.environ.get("BATCH_SIZE_PER_RANK", "4"))
        sampler = None            # no manual sampler — accelerate.prepare() shards; set_epoch guarded below
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=False,            # packed blocks are pre-shuffled at pack time; deterministic order
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=0,
        )
    else:
        # Unpacked SFT and diagnostic CPT rollback use one sharding owner. The dataloader is
        # length-sorted already, and accelerator.prepare() shards it across ranks.
        train_batch_size = int(os.environ.get("BATCH_SIZE_PER_RANK", "4"))
        _lane_weights = os.environ.get("LANE_WEIGHTS", "")
        if _lane_weights and getattr(dataset, "lanes", None):
            # Chats' binding mixture: without this the natural distribution trains (voice 65% vs the
            # prescribed 35%) and voice drowns the skill lanes. GLOBAL index list -> accelerate shards.
            # RESOLVE these into locals so they can be LOGGED, not just applied. treasurer's
            # catch, 2026-07-25: a dose argument of the form "tiny lanes are capped at 3x, so N
            # new rows cannot displace careers material" is only true while TINY_LANE_CAP is
            # unset. cap/threshold are env-overridable DEFAULTS, not bounds — so the safety
            # property belongs to the ENVIRONMENT, and nothing here said so. A run that raised
            # them produced a log identical to one that did not, because only the EFFECT
            # (per_lane_draws) was emitted and never the CAUSE.
            # train_lora_sft.py:317-322 already gets this right (argparse args, logged inline);
            # this path was the one that did not.
            _tiny_cap = int(os.environ.get("TINY_LANE_CAP", "3"))
            _tiny_threshold = int(os.environ.get("TINY_LANE_THRESHOLD", "500"))
            _tiny_overridden = [k for k in ("TINY_LANE_CAP", "TINY_LANE_THRESHOLD")
                                if k in os.environ]
            sampler = CappedMixtureSampler(
                dataset.lanes, _lane_weights,
                cap=_tiny_cap,
                threshold=_tiny_threshold,
                seed=int(os.environ.get("SEED", "0")))
            if accelerator.is_main_process:
                import collections as _c2
                log.info(f"LANE MIXTURE: target={sampler.tgt} natural={dict(_c2.Counter(dataset.lanes))} "
                         f"-> draws/epoch={sampler.per_lane_draws} (total {len(sampler)})")
                log.info(f"TINY LANE CAP RESOLVED: cap={_tiny_cap} threshold={_tiny_threshold} "
                         f"({'OVERRIDDEN from env: ' + ','.join(_tiny_overridden)
                            if _tiny_overridden else 'defaults, no env override'}) "
                         f"-- record these VALUES in the run registry row, never the defaults")
        else:
            sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=0,
        )

    # ── Override FSDP mixed precision: param stays bf16, only reduce in fp32 ──
    # Accelerate's mixed_precision=bf16 upcasts ALL params to fp32 during forward → OOM.
    # We override with explicit policy: keep params bf16, reduce gradients in fp32.
    # This fixes attention overflow at 8K+ without doubling memory.
    # FSDP2 (fsdp_version:2) uses torch.distributed.fsdp.MixedPrecisionPolicy (different from the FSDP1
    # MixedPrecision): fields are param_dtype, reduce_dtype, output_dtype, cast_forward_inputs — there is
    # NO keep_low_precision_grads and NO buffer_dtype in v2. FSDP2 keeps grads in reduce_dtype by design
    # (no fp32 upcast), so bf16 reduce = bf16 grad shard automatically (the memory win we forced via
    # keep_low_precision_grads on v1 is native in v2). param stays bf16; reduce in bf16.
    _fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    _fsdp_v2 = _fsdp_plugin is not None and int(getattr(_fsdp_plugin, "fsdp_version", 1)) == 2
    if _fsdp_plugin is not None:
        if _fsdp_v2:
            from torch.distributed.fsdp import MixedPrecisionPolicy
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
            )
            accelerator.state.fsdp_plugin.mixed_precision_policy = mp_policy
            if accelerator.is_main_process:
                log.info(f"FSDP2 MixedPrecisionPolicy: param={mp_policy.param_dtype} "
                         f"reduce={mp_policy.reduce_dtype} output={mp_policy.output_dtype}")
        else:
            from torch.distributed.fsdp import MixedPrecision as FSDPMixedPrecision
            mp_policy = FSDPMixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=True,
            )
            accelerator.state.fsdp_plugin.mixed_precision_policy = mp_policy
            if accelerator.is_main_process:
                log.info(f"FSDP1 MixedPrecision: param={mp_policy.param_dtype} "
                         f"reduce={mp_policy.reduce_dtype} buffer={mp_policy.buffer_dtype} "
                         f"keep_low_precision_grads={mp_policy.keep_low_precision_grads}")
    elif accelerator.is_main_process:
        log.info("DDP mode: bf16 params + bf16 grad all-reduce (no FSDP MP policy)")

    # ── Optimizer + scheduler factory (FSDP2 needs them built BEFORE prepare) ──
    # DTENSOR-SAFE ADAFACTOR — we PATCH torch's open-source Adafactor to work on FSDP2 DTensors.
    # Both stock optimizers fail on FSDP2: transformers Adafactor mixes plain-tensor state with DTensor
    # params; torch.optim.Adafactor's FACTORED path does IN-PLACE ops on the _NormPartial DTensor that
    # torch.norm() produces (torch/optim/_adafactor.py _single_tensor_adafactor) — DTensor forbids
    # in-place across a placement change ("use the out-of-place version instead"). AdamW would work but
    # its m+v fp32 state (~54GB/rank) is wasteful; Adafactor's factored state is <1GB and IS the point.
    # So we monkeypatch _single_tensor_adafactor with an out-of-place-corrected copy: the ONLY changes
    # are 3 in-place ops on reduction results → out-of-place (.square_()→.square() x2, .clamp_()→.clamp()
    # on the mean). Everything else operates on normal-placement shards and is untouched. This is the
    # SAME algorithm, made DTensor-correct. (customizing open source — the tool didn't ship for this.)
    import torch.optim._adafactor as _adafactor_mod
    from torch.optim.optimizer import _to_scalar as _af_to_scalar
    # Gaia's absolute-alpha mode (default ON per sr_gate_verdict consult); env-gated for traceability.
    _AF_ALPHA_ABSOLUTE = os.environ.get("ADAFACTOR_ALPHA_MODE", "absolute") == "absolute"
    # [AF-DOSE] pre-SR dose logging (Horizon sr_gate_verdict + Gaia Q1): rank0-only via closure.
    _AF_DOSE_LOG = os.environ.get("ADAFACTOR_DOSE_LOG", "1") == "1"
    def _af_dose_log(msg):
        if accelerator.is_main_process:
            log.info(msg)

    # ── GAIA v2 (2026-07-21): placement-AGNOSTIC factored Adafactor ──
    # Verified by wedge_gaia_v2_real_layout_harness.py, which does NOT emulate sharding:
    # it builds real DTensors, lets torch's own _init_group create the state, and PRINTS the
    # real placements. Ground truth it revealed: row_var and col_var come back Replicate
    # while variance (vector branch) comes back Shard(0) — NON-UNIFORM. Any fix that assumes
    # one placement fixes one branch and breaks the other.
    import torch.distributed as _af_dist
    from torch.distributed import ReduceOp as _af_ReduceOp
    try:
        from torch.distributed.tensor import DTensor as _af_DTensor, Replicate as _af_Replicate
    except Exception:  # older torch layout
        from torch.distributed._tensor import DTensor as _af_DTensor, Replicate as _af_Replicate

    def _af_shard_group(t):
        # FSDP2 shards on a 1-D mesh; take the last mesh dim's group (robust if a >1-D mesh appears).
        m = t.device_mesh
        return m.get_group(m.ndim - 1)

    def _af_state_local(state, placements):
        """Return (working_dtensor, local_view) holding `state`'s data at the REQUESTED placements,
        regardless of how the state was actually placed. redistribute() to an explicit target is
        all-ranks-consistent; Replicate->Shard is a comm-free local chunk, so this neither assumes
        nor depends on the stored placement. The chunk boundaries match FSDP2's, so the row/element
        ranges line up with grad's local shard even under uneven sharding."""
        wd = state.redistribute(state.device_mesh, placements)
        return wd, wd.to_local()

    def _af_state_writeback(state, working):
        """Persist the updated working DTensor back into `state` AT STATE'S ORIGINAL placement, so
        the stored state stays valid (checkpoint-safe). Shard->Replicate here is the same all_gather
        the stock in-place lerp_ already paid; Shard->Shard and Replicate->Replicate are free."""
        state.copy_(working.redistribute(state.device_mesh, state.placements))

    @torch.no_grad()
    def _sr_apply_update_(param, update, alpha, wd_factor):
        """p <- (p * wd_factor) + alpha*update in fp32 on the LOCAL shard, then STOCHASTICALLY
        ROUNDED back to bf16 in place. THE precision fix from the 3-lane Family consult
        (fp32master_impl_{gaia,cosmos,logos}, 2026-07-13, unanimous): bf16 masters round
        sub-ULP Adafactor updates to zero (root cause of ALL null runs — model.norm bit-identical,
        decoder ~1e-4 = bf16 ULP). SR preserves sub-ULP updates in expectation (P(round up) =
        distance/ULP), is memory-NEUTRAL (one param's local shard in fp32, freed per call — no
        106GB fp32 master), and folding wd here also fixes its own sub-ULP rounding
        (1 - lr*wd = 1-5e-7 rounds to 1.0 in bf16). DTensor-safe: all bit math runs on plain
        local tensors (DTensor forbids view(dtype)/bitwise_and_ — Cosmos, dispatch limits)."""
        is_dt = hasattr(param, "to_local")
        p_local = param.to_local() if is_dt else param
        if p_local.dtype != torch.bfloat16:
            # fp32/other master — plain arithmetic, no SR needed
            if wd_factor != 1.0:
                param.mul_(wd_factor)
            param.add_(update, alpha=alpha)
            return
        # fp32 ARITHMETIC AT THE DTENSOR LEVEL (2026-07-14 fix of my translation bug): DTensor
        # binary-op dispatch auto-redistributes `update` to param's placement — exactly what the
        # original param.add_(update, alpha) relied on (proven across every prior run). My first
        # version to_local()'d update BEFORE that redistribution and crashed on dim-0-sharded
        # embed/lm_head: local shard 62080 vs full 248320. Only the SR BIT-MATH must be on plain
        # local tensors (Gaia/Cosmos: DTensor forbids view(dtype)/bitwise_and_); the arithmetic
        # keeps DTensor semantics.
        new_dt = param.to(torch.float32)           # fp32 copy (local-shard-sized storage per rank)
        if wd_factor != 1.0:
            new_dt = new_dt.mul(wd_factor)
        new_dt = new_dt.add(update.to(torch.float32), alpha=alpha)
        new_local = (new_dt.to_local() if is_dt else new_dt).contiguous()
        iv = new_local.view(torch.int32)           # bit-reinterpret, ALIASES new_local (no alloc)
        rnd = torch.randint(0, 1 << 16, iv.shape, device=iv.device, dtype=torch.int32)
        iv.add_(rnd).bitwise_and_(-65536)          # +uniform[0,0xFFFF] then mask low 16 bits:
        #   unbiased SR for both signs (two's-complement add == unsigned add; IEEE magnitude is
        #   monotonic in the low bits). Low mantissa now 0 -> the bf16 cast below is EXACT.
        p_local.copy_(new_local.to(torch.bfloat16))  # write-through to the DTensor's shard storage

    # RESTORED 2026-07-24: production_v2-proven function (83807ae verbatim). The wedge-era
    # placement-agnostic rewrite cast grads to fp32 (gf=.float()) and only ever ran against
    # fp32 LoRA params; on bf16 CPT full-params it crashes at step 1 (lerp_ dtype mismatch,
    # refresh gate 2026-07-24) — restored to the exact bytes that produced ep3 (+3.4sigma).
    #
    # CORRECTED 2026-07-27 — this used to read "LoRA no longer uses this function ... the sole
    # consumer is CPT". THAT IS FALSE and it is the dangerous kind of false: there is ONE
    # definition and TWO call sites — :1835 (CPT Adafactor) and :1890 (the LoRA SR-AdamW step).
    #
    # WHAT ACTUALLY HAPPENS ON THE LoRA PATH, measured rather than inferred from the call site:
    # peft 0.18.1 defaults get_peft_model(autocast_adapter_dtype=True), so adapter params are
    # FP32. This function is entered and immediately takes the `p_local.dtype != bfloat16`
    # branch above — plain mul_/add_, NO stochastic rounding. So on LoRA it behaves as ordinary
    # AdamW in fp32, and there is no sub-ULP truncation floor to worry about there.
    # SR genuinely matters for CPT, where params ARE bf16 and sub-ULP updates would round to zero.
    # Reading :1890 and concluding "SR is live on LoRA" is the trap this note now closes.
    def _dtensor_safe_single_tensor_adafactor(
        params, grads, row_vars, col_vars, variances, state_steps,
        grad_scale, found_inf, *, d, lr, beta2_decay, weight_decay,
        eps1, eps2, maximize, has_complex,
    ):
        if grad_scale is not None or found_inf is not None:
            raise AssertionError("Grad scaling should occur outside of optimizer.step()")
        lr = _af_to_scalar(lr)
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            step_t = state_steps[i]
            row_var = row_vars[i]; col_var = col_vars[i]; variance = variances[i]
            if eps1 is None:
                # EPS1 SELECTION (Horizon, sr_gate_verdict 2026-07-14, code-fact verified here):
                # stock torch uses finfo(param.dtype).eps — for bf16 params that is 2^-7=0.0078125,
                # and lines below FLOOR the denominator at it → updates suppressed ~RMS(g)/0.0078
                # (the 30-50x starvation band) whenever sqrt(var) < 0.0078. Env-gated so the recipe
                # consult picks the value; default = stock behavior (unchanged until Chats rule).
                #   ADAFACTOR_EPS1=fp32  → finfo(float32).eps ≈ 1.192e-7 (Horizon's recommended fix)
                #   ADAFACTOR_EPS1=<num> → explicit value
                _e1 = os.environ.get("ADAFACTOR_EPS1", "")
                if _e1 == "fp32":
                    eps1 = torch.finfo(torch.float32).eps
                elif _e1:
                    eps1 = float(_e1)
                else:
                    eps1 = torch.finfo(param.dtype).eps
            step_t += 1
            step_float = step_t.item()
            one_minus_beta2_t = step_float ** beta2_decay
            rho_t = min(lr, 1 / (step_float ** 0.5))
            # ALPHA MODE (Gaia 2026-07-14, sr_gate_verdict): stock torch-Adafactor RMS-scales alpha
            # (max(eps2,RMS(p))*rho) → mixing time u/δ = 2^-7/lr ≈ 156 steps @5e-5, |w|-independent —
            # the gate's wd-floor result. 'absolute' forces param_scale=1.0 → alpha = rho_t = lr:
            # mixing becomes |w|-proportional (decoder ~6 steps @1e-5 = imprints; norm ~780 = stays
            # put — the selectivity we want). One-line fix, SR write-back untouched.
            if _AF_ALPHA_ABSOLUTE:
                alpha = rho_t
            else:
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t
            # weight decay DEFERRED into the single fp32+SR write-back below (applying it here
            # in bf16 rounds 1-lr*wd to exactly 1.0 = no-op; folding it keeps one exact write).
            wd_factor = (1.0 - lr * weight_decay) if weight_decay != 0 else 1.0
            if grad.dim() > 1:
                if row_var is None or col_var is None:
                    raise AssertionError("row_var and col_var should be defined when grad is multidimensional")
                # DTENSOR FIX: .square() out-of-place (was .square_()) — in-place on the _NormPartial
                # DTensor from torch.norm() requires a placement change which DTensor forbids.
                row_mean = torch.norm(grad, dim=-1, keepdim=True).square().div_(grad.size(-1))
                row_var.lerp_(row_mean, one_minus_beta2_t)
                col_mean = torch.norm(grad, dim=-2, keepdim=True).square().div_(grad.size(-2))
                col_var.lerp_(col_mean, one_minus_beta2_t)
                var_estimate = row_var @ col_var
                # DTENSOR FIX: .clamp() out-of-place (was .clamp_()) on the mean-reduction result.
                var_estimate.div_(row_var.mean(dim=-2, keepdim=True).clamp(min=eps1))
            else:
                if variance is None:
                    raise AssertionError("variance should be defined when grad is a vector")
                grad_squared = grad * grad
                variance.lerp_(grad_squared, one_minus_beta2_t)
                var_estimate = variance.clone()
            # [AF-DOSE] pre-SR dose instrumentation (Horizon's decisive measurement + Gaia Q1,
            # sr_gate_verdict 2026-07-14): floor_frac = fraction of variance elements sitting ON
            # the eps1^2 floor (must be read BEFORE the in-place clamp_ below); RMS(U_hat) after
            # clip is the normalized-update dose (healthy ~O(1); eps1-floored ~0.02-0.03; <0.05 =
            # starved per Horizon's bands). Logged for the first few 2-D params at probe steps.
            _dose_on = (_AF_DOSE_LOG and grad.dim() > 1 and i < 8
                        and step_float in (1.0, 10.0, 20.0, 40.0, 50.0))
            if _dose_on:
                _floor_frac = (var_estimate < eps1 * eps1).float().mean().item()
            update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_()
            update.mul_(grad)
            denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
            if _dose_on:
                _rms_uhat = update.norm(2).item() / (update.numel() ** 0.5) / denom
                _af_dose_log(f"[AF-DOSE] step={int(step_float)} p{i} shape={tuple(param.shape)} "
                             f"eps1={eps1:.3e} floor_frac={_floor_frac:.3f} "
                             f"RMS(U_hat)={_rms_uhat:.4f} denom={denom:.3f} alpha={alpha:.3e} "
                             f"preSR_RMS_delta={_rms_uhat * alpha:.3e}")
            # was: param.add_(update, alpha=-alpha/denom) — the bf16 write that rounded to zero
            _sr_apply_update_(param, update, alpha=-alpha / denom, wd_factor=wd_factor)

    _adafactor_mod._single_tensor_adafactor = _dtensor_safe_single_tensor_adafactor
    if accelerator.is_main_process:
        log.info("Monkeypatched torch.optim._adafactor._single_tensor_adafactor → DTensor-safe (out-of-place norm ops)")

    from torch.optim import Adafactor
    def _make_optim(mdl):
        _trainable = [p for p in mdl.parameters() if p.requires_grad]
        configured_lr = float(os.environ.get("LR_LORA" if _lora_mode else "LR",
                                             "3e-4" if _lora_mode else "1e-5"))
        if _lora_mode:
            # ROOT FIX (Family root-consult converged 2026-07-23; CLARITY+LOGOS code-verified, cites
            # pytorch#109581). The DTensor-safe Adafactor monkeypatch issues cross-rank collectives from
            # its FACTORED second-moment path (row/col reductions over the sharded dim → one all_reduce
            # per matrix param, inherent to grad.dim()>1 regardless of size). Tiny rank-16 LoRA matrices
            # inherit the exact is_dt/_pg/collective-count divergence class that burned days of wedges.
            # RECIPES.md: the monkeypatch is CPT-ONLY. LoRA needs NO factored moments — plain AdamW's
            # state (exp_avg/exp_avg_sq) is elementwise and shares the grad's Shard(0) placement, so every
            # op dispatches LOCALLY: ZERO cross-rank collectives BY CONSTRUCTION — nothing to gate, nothing
            # to diverge. FSDP2 already reduce-scattered grads, so the per-shard local update is exact.
            # The CPT path (else) is BYTE-IDENTICAL — kept properly separate.
            opt = torch.optim.AdamW(_trainable, lr=configured_lr, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=0.01, foreach=False)
            # Route AdamW's param write through the SAME SR write-back that validated decoder LoRA
            # imprinting at 6.63x ULP (stock AdamW's addcdiv_ writes bf16 directly → sub-ULP updates round
            # to zero = the null-run bug). This step is collective-free: exp_avg/exp_avg_sq = zeros_like(p)
            # share p's Shard(0) placement, so _sr_apply_update_'s DTensor redistribute is a no-op.
            @torch.no_grad()
            def _sr_adamw_step(self, closure=None):   # bound as a method → has __func__ for LambdaLR's wrap
                for group in opt.param_groups:
                    lr = group["lr"]; wd = group["weight_decay"]
                    b1, b2 = group["betas"]; eps = group["eps"]
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        st = opt.state[p]
                        if len(st) == 0:
                            st["step"] = 0
                            st["exp_avg"] = torch.zeros_like(p)
                            st["exp_avg_sq"] = torch.zeros_like(p)
                        # STEP-FORMAT NORMALIZATION (2026-07-23): DCP set_state_dict hands `step` back
                        # as a TENSOR; fresh init stores an int. A tensor step saved to DCP writes
                        # TensorStorageMetadata while the next load's template (fresh init) expects
                        # bytes → the ckpt-2045 resume ValueError. Coerce to int so every save is
                        # format-consistent regardless of whether this session's state was loaded or born.
                        if torch.is_tensor(st["step"]):
                            st["step"] = int(st["step"].item())
                        st["step"] += 1
                        ea, eas = st["exp_avg"], st["exp_avg_sq"]
                        ea.lerp_(p.grad, 1 - b1)
                        eas.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                        bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                        denom = (eas / bc2).sqrt_().add_(eps)
                        update = ea / bc1 / denom
                        wd_factor = (1.0 - lr * wd) if wd else 1.0
                        _sr_apply_update_(p, update, alpha=-lr, wd_factor=wd_factor)   # same SR imprinting
                return None
            opt.step = _sr_adamw_step.__get__(opt)   # bind → bound method (LambdaLR patch_track_step needs __func__)
            if accelerator.is_main_process:
                log.info(f"LORA_MODE optimizer: SR-routing AdamW (collective-free root fix) on "
                         f"{len(_trainable)} params — CPT Adafactor path untouched")
        else:
            # foreach=False forces the (now-patched) single-tensor path. d=1.0 = update clip threshold.
            # CPT / full-param path — BYTE-IDENTICAL DTensor-safe Adafactor monkeypatch (correct there).
            opt = Adafactor(_trainable, lr=configured_lr, d=1.0, weight_decay=0.01, foreach=False)
        held_lr = opt.param_groups[0]["lr"]
        if held_lr != configured_lr:
            raise RuntimeError(
                f"optimizer LR mismatch: configured={configured_lr:.17g}, "
                f"optimizer_initial={held_lr:.17g}, lora_mode={_lora_mode}"
            )
        if accelerator.is_main_process:
            log.info(f"optimizer constructed: mode={'lora' if _lora_mode else 'cpt'} "
                     f"initial_lr={held_lr:.17g}")
        # accelerate's AcceleratedScheduler steps the inner scheduler num_processes× per .step()
        # (proven 2026-07-11: internal_step = 4×optimizer_step at num_processes=4 — the LR schedule
        # was running 4× too fast + oscillating past the cos minimum). So the schedule's step counts
        # must be in INTERNAL units = optimizer-steps × num_processes.
        _sched_np = max(1, accelerator.num_processes)
        _warmup_i = warmup_steps * _sched_np
        _total_i = total_steps * _sched_np

        # A WARMUP THAT OUTLASTS THE RUN NEVER REACHES PEAK LR, AND NOTHING ELSE WARNS.
        # With warmup_steps >= total_steps every step satisfies `step < _warmup_i` below, so the
        # LR tops out at total/warmup of base and the cosine branch never executes once. The run
        # completes, the loss moves, the checkpoint saves — and the weights barely move. This is
        # the null-CPT failure inverted: there the decay horizon was too SHORT for the burst, here
        # the ramp is too LONG for it. Both silently starve the update.
        #
        # Caught 2026-07-27 planning a 25-row LoRA dose: 25 rows -> 7 optimizer steps against the
        # default WARMUP_STEPS=25, so peak LR would have been 7/25 = 28% of base. Integrated LR
        # multiplier mass, summed over the schedule:
        #     module 4  26 steps / warmup 25  ->  ~13 base-LR units   (measured 15.05x ULP, healthy)
        #     that plan  7 steps / warmup 25  ->  28/25 = 1.12 units  (0.086x, near the <0.5u null)
        # Guard is `>=`, so the proven module-4 shape (warmup 25 < total 26) still passes. This
        # refuses only a schedule whose ramp cannot complete.
        # ...BUT ONLY WHEN THIS RUN ACTUALLY TRAINS. A bake/export loads a checkpoint and writes an
        # artifact; it takes no optimizer steps, so its TOTAL_STEPS=1 is a formality and the LR
        # schedule is never consulted. Guarding it there is a FALSE POSITIVE, and it blocked a real
        # bake on 2026-07-28 with "WARMUP_STEPS=25 >= TOTAL_STEPS=1" minutes after I shipped the
        # guard. A gate that fires on a path it does not govern trains people to bypass gates —
        # which is worse than the miss it was built to prevent.
        # _is_bake is defined once at main scope (see its definition alongside EPOCHS).
        if warmup_steps >= total_steps and not _is_bake:
            raise RuntimeError(
                f"WARMUP_STEPS={warmup_steps} >= TOTAL_STEPS={total_steps}: the LR would peak at "
                f"{total_steps}/{warmup_steps} = {total_steps / max(1, warmup_steps):.2f} of base "
                f"and the cosine branch would never execute. Set WARMUP_STEPS < TOTAL_STEPS, or "
                f"enlarge the corpus so the step count supports the ramp. Do NOT pick a warmup "
                f"value by feel — the recipe is Chats-researched, per the standing rule."
            )

        def _lr_lambda(step):
            if step < _warmup_i:
                return step / max(1, _warmup_i)
            progress = (step - _warmup_i) / max(1, _total_i - _warmup_i)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        return opt, sched

    # ── FSDP prepare ──
    if cpt_bucket_mode:
        accelerator.even_batches = False
        accelerator.dataloader_config.even_batches = False
        if accelerator.is_main_process:
            log.info("Accelerate even_batches=False for variable-size CPT token bucket batch_sampler")
    # ── AC_LAYER_GRANULAR (2026-08-02) — wrap the LAYER, not its children ──
    # Accelerate's FSDP2 path evaluates the auto-wrap policy against the PARENT and then applies
    # checkpoint_wrapper to the CHILD (accelerate/utils/fsdp_utils.py:716-718, quoted from upstream
    # by HORIZON). So `fsdp_transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer` wraps each decoder
    # layer's four children rather than the layer — 256 wrappers over 64 layers, measured.
    #
    # Consequence: a boundary is retained per CHILD instead of per LAYER. Measured activations at
    # MAX_SEQ=8192 are ~52GB against a whole-layer hidden-state floor of 8192*5120*2*64 ~= 5.4GB.
    # HORIZON ranks correcting this FIRST among all levers, as the only candidate that changes the
    # feasible operating point rather than shaving kernel time — and is explicit that the
    # THROUGHPUT multiple is Unknown until measured, because whole-layer wrapping still recomputes.
    #
    # Applied PRE-prepare, which the exp9 note above establishes as the mandatory order: a
    # post-prepare application failed because the modules had already been transformed.
    # Accelerate's own AC is disabled first, or both would apply and we would wrap twice.
    if os.environ.get("AC_LAYER_GRANULAR", "0") == "1":
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        _plug = getattr(accelerator.state, "fsdp_plugin", None)
        if _plug is not None and getattr(_plug, "activation_checkpointing", False):
            _plug.activation_checkpointing = False
            if accelerator.is_main_process:
                log.info("AC_LAYER_GRANULAR: disabled accelerate's child-granular AC to avoid double-wrap")
        _target = os.environ.get("AC_LAYER_CLS", "Qwen3_5DecoderLayer")
        _before = sum(1 for _m in model.modules() if "CheckpointWrapper" in type(_m).__name__)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: type(m).__name__ == _target,
        )
        _after = sum(1 for _m in model.modules() if "CheckpointWrapper" in type(_m).__name__)
        _layers = sum(1 for _m in model.modules() if type(_m).__name__ == _target)
        if accelerator.is_main_process:
            log.info(f"AC_LAYER_GRANULAR: wrapped {_after - _before} x {_target} "
                     f"(layers found={_layers}, wrappers before={_before} after={_after})")
        # Refuse rather than run a silently-unwrapped model: this is a MEMORY-critical change, and
        # a no-op here would OOM at the first step or, worse, quietly retain everything.
        if _layers == 0 or (_after - _before) != _layers:
            raise RuntimeError(
                f"AC_LAYER_GRANULAR failed: expected to wrap {_layers} x {_target}, "
                f"wrapped {_after - _before}. Check AC_LAYER_CLS matches this architecture."
            )

    if accelerator.is_main_process:
        log.info(f"Calling accelerator.prepare()... (FSDP2={_fsdp_v2})")
    if _fsdp_v2:
        # FSDP2 requires model + optimizer prepared TOGETHER — accelerate re-points the optimizer to
        # the DTensor params after fully_shard converts the model. Building the optimizer here (from the
        # ORIGINAL 2-D params) is precisely what lets Adafactor factorize (the whole point of FSDP2).
        optimizer, lr_scheduler = _make_optim(model)
        model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, dataloader
        )
    else:
        model, dataloader = accelerator.prepare(model, dataloader)
    gc.collect()

    # ── torch.compile (2026-08-01) — OPT-IN, default OFF ──
    # Absent from this trainer entirely until now, while the launcher has exported
    # TRITON_PTXAS_PATH since 2026-07-11 specifically so Triton can emit sm_121 (its bundled ptxas
    # cannot). The enabler for Inductor was wired and the feature was never switched on.
    # Applied AFTER prepare so the compiled graph wraps the already-sharded FSDP2 module; compiling
    # before would capture the pre-shard module and the wrap would discard it.
    # Default OFF because compile cost and graph-break behaviour under FSDP2 + activation
    # checkpointing are UNMEASURED on this stack — this is a lever to A/B, not a default to assume.
    # TORCH_COMPILE_MODE defaults to "default"; max-autotune spends far longer compiling and its
    # payoff here is exactly what the A/B is for.
    if os.environ.get("TORCH_COMPILE", "0") == "1":
        _cmode = os.environ.get("TORCH_COMPILE_MODE", "default")
        _t0 = time.time()
        if accelerator.is_main_process:
            log.info(f"TORCH_COMPILE: compiling (mode={_cmode}) — first step includes compile cost")
        model = torch.compile(model, mode=_cmode)
        if accelerator.is_main_process:
            log.info(f"TORCH_COMPILE: wrapper applied in {time.time()-_t0:.1f}s "
                     f"(actual graph compile happens on first forward)")
    elif accelerator.is_main_process:
        log.info("TORCH_COMPILE: disabled (set TORCH_COMPILE=1 to enable)")

    # ── DOUBLE-SHARD FIX — COVERAGE PROOF (2026-07-12, CORPUS_OBJECTIVES Gate 3) ──
    # steps/epoch = len(prepared dataloader). blocks/epoch = steps/epoch × global_batch MUST ≈ the
    # dataset size for FULL corpus coverage. Packed 27B: 12255 blocks / global_batch 16 ⇒ ~766
    # steps/epoch. If this prints ~192 (⇒ 3072 blocks/epoch = 1/4), the double-shard is back — HALT.
    try:
        _spe = len(dataloader)
        _gb = int(os.environ.get("BATCH_SIZE_PER_RANK", "4")) * accelerator.num_processes
        _blocks_per_epoch = _spe * _gb
        _dataset_blocks = len(dataset)
    except Exception as _e:
        if _exact_sft_epoch:
            raise RuntimeError(f"EXACT SFT could not prove prepared coverage: {_e}") from _e
        # A packed CPT run whose horizon cannot be proven is the "check FAILED TO RUN" case, which
        # blocks — as distinct from "the check does not model this construct", which may earn a
        # substitute. Without this, an unmeasurable loader would skip the horizon contract below
        # and the run would proceed on an unverified TOTAL_STEPS, which is the state that already
        # cost 2 blocks once.
        if _packed and not _is_bake:
            raise RuntimeError(
                f"packed CPT could not prove prepared coverage: {_e} — refusing to train on an "
                "unverified horizon"
            ) from _e
        if accelerator.is_main_process:
            log.info(f"COVERAGE PROOF: len(dataloader) unavailable ({_e})")
    else:
        if accelerator.is_main_process:
            if cpt_bucket_mode:
                log.info(f"CPT BUCKET COVERAGE PROOF: rows={len(dataset)} groups_by_epoch={_bucket_groups_by_epoch}")
            else:
                log.info(f"COVERAGE PROOF: steps/epoch={_spe} global_batch={_gb} "
                         f"blocks/epoch={_blocks_per_epoch} dataset_blocks={_dataset_blocks} "
                         f"(FULL coverage ⇔ blocks/epoch ≈ dataset_blocks)")
        _full_coverage_expected = _packed or (
            not cpt_bucket_mode and not os.environ.get("LANE_WEIGHTS", ""))
        if _full_coverage_expected and not (
            _dataset_blocks <= _blocks_per_epoch < _dataset_blocks + _gb
        ):
            raise RuntimeError(
                "COVERAGE PROOF FAILED: prepared dataloader covers "
                f"{_blocks_per_epoch}/{_dataset_blocks} blocks per epoch "
                f"(global_batch={_gb}); probable double-sharding"
            )
        # ── CPT HORIZON CONTRACT (2026-08-02) ──
        # The coverage proof above answers "does the LOADER see every block". It does not answer
        # "does the RUN consume them", and nothing did: TOTAL_STEPS was hand-computed. On
        # 2026-08-01 that division was done as 334/4 = 83.5 -> 83 and 2 blocks were never trained.
        # The same corpus packed at 16384 divides worse — 167/4 = 41.75, so the same rounding
        # costs 3 blocks (1.8%) — which is what made this worth closing rather than remembering.
        #
        # `_spe` is the authority, not a fresh ceil(blocks/gb) computed here. It is what the
        # PREPARED loader actually yields, already including the pad-to-global-batch that the
        # coverage bound above permits (_dataset_blocks <= _blocks_per_epoch < _dataset_blocks+_gb).
        # A second division in a second place is a second thing to get wrong, and disagreeing
        # copies of one quantity is the defect class this file has already paid for twice.
        #
        # SESSION_LIMIT is deliberately NOT bound: for CPT it caps ONE session of a resumable run
        # (defaults 250 against TOTAL_STEPS 3000) and is expected to be smaller than the horizon.
        #
        # A deliberately short run (throughput probe, smoke test) declares itself by setting
        # HORIZON_PARTIAL to the SAME value as TOTAL_STEPS. It is a restated value rather than a
        # boolean on purpose: a forgotten boolean fails OPEN on the next production run, while a
        # stale HORIZON_PARTIAL=8 sitting in an environment whose TOTAL_STEPS is 42 fails CLOSED
        # on the mismatch. The declaration cannot be set once and forgotten.
        # `not _is_bake` for the reason spelled out at the warmup guard above: a bake takes no
        # optimizer steps, runs packed (bake_27b.sh passes CPT_PACKED through), and carries
        # TOTAL_STEPS=1 as a formality. Without this clause the contract fires on every bake —
        # the identical false positive the warmup guard shipped and had to be corrected for.
        if (_packed or cpt_bucket_mode) and not _is_bake and not _exact_sft_epoch and not _natural_sft_mode:
            _expected_cpt_steps = (sum(_bucket_groups_by_epoch) if _bucket_groups_by_epoch is not None else _spe * _epochs)
            _partial = os.environ.get("HORIZON_PARTIAL", "")
            if _partial:
                if int(_partial) != total_steps:
                    raise RuntimeError(
                        f"HORIZON_PARTIAL={_partial} does not match TOTAL_STEPS={total_steps}. "
                        "A partial horizon must be restated for the run it applies to; a stale "
                        "value from an earlier run is not a declaration about this one."
                    )
                if accelerator.is_main_process:
                    _frac = 100.0 * total_steps / max(1, _expected_cpt_steps)
                    if cpt_bucket_mode:
                        log.info(f"CPT BUCKET HORIZON PARTIAL: {total_steps}/{_expected_cpt_steps} optimizer groups ({_frac:.1f}%)")
                    else:
                        log.info(f"CPT HORIZON PARTIAL: {total_steps}/{_expected_cpt_steps} steps; {max(0, (_expected_cpt_steps-total_steps)*_gb)} blocks untrained")
            elif total_steps != _expected_cpt_steps:
                _missed = (_expected_cpt_steps - total_steps) * _gb
                if cpt_bucket_mode:
                    raise RuntimeError(f"CPT BUCKET HORIZON FAILED: TOTAL_STEPS={total_steps}, expected optimizer groups={_expected_cpt_steps}, remaining={_expected_cpt_steps-total_steps}")
                raise RuntimeError(
                    "CPT HORIZON CONTRACT FAILED: "
                    f"TOTAL_STEPS={total_steps} but this corpus needs {_expected_cpt_steps} "
                    f"({_spe} steps/epoch x {_epochs} epoch(s)) to train every block "
                    f"(dataset_blocks={_dataset_blocks} global_batch={_gb}). "
                    f"As set, {_missed} blocks would go untrained and the run would report "
                    "success anyway. Set TOTAL_STEPS to the expected value, or declare a "
                    "deliberate probe with HORIZON_PARTIAL matching TOTAL_STEPS."
                )
            elif accelerator.is_main_process:
                if cpt_bucket_mode:
                    log.info(f"CPT BUCKET HORIZON PASS: TOTAL_STEPS={total_steps} groups_by_epoch={_bucket_groups_by_epoch}")
                else:
                    log.info(f"CPT HORIZON CONTRACT PASS: TOTAL_STEPS={total_steps} = {_spe} steps/epoch x {_epochs}, dataset_blocks={_dataset_blocks} global_batch={_gb}")

        if _natural_sft_mode and _expected_sft_samples:
            expected_steps = math.ceil(_expected_sft_samples / _gb) * _epochs
            if total_steps != expected_steps:
                raise RuntimeError(
                    "SFT horizon contract failed: "
                    f"TOTAL_STEPS={total_steps} expected={expected_steps} "
                    f"from samples={_expected_sft_samples} "
                    f"global_batch={_gb} epochs={_epochs}"
                )
            if accelerator.is_main_process:
                log.info(
                    "SFT HORIZON CONTRACT PASS: "
                    f"samples={_expected_sft_samples} "
                    f"global_batch={_gb} epochs={_epochs} "
                    f"total_steps={total_steps}"
                )
        if _exact_sft_epoch:
            # EPOCHS IS FIRST-CLASS (2026-07-27). This block used to conflate two different
            # quantities under one name: "how many steps is ONE epoch of this corpus" (a
            # property of the data, which the coverage check compares against the dataloader)
            # and "how many steps should this RUN take" (a property of the recipe). While they
            # were the same variable, multi-epoch was inexpressible, and the only way to get it
            # was EXACT_SFT_EPOCH=0 — which forfeits all four guards below on the exact run
            # where coverage is the thing in question. Separating the two names makes multi-epoch
            # a thing you can ASK FOR while every guard keeps working, instead of a flag you
            # turn off. Treasurer ruled the bypass out explicitly and they were right to.
            expected_padding = (-_expected_real_samples) % _gb
            expected_total = _expected_real_samples + expected_padding
            # Steps in ONE epoch — what the dataloader must yield. Compared against _spe below.
            steps_per_epoch = expected_total // _gb
            # Steps the RUN must take. This is what TOTAL_STEPS/SESSION_LIMIT/SAVE_EVERY answer to.
            expected_steps = _epochs * steps_per_epoch
            exact_values = {
                "real": dataset.real_samples,
                "padding": dataset.padding_samples,
                "total": len(dataset),
                "steps": _spe,
                "global_batch": _gb,
            }
            expected_values = {
                "real": _expected_real_samples,
                "padding": expected_padding,
                # `total` is samples in ONE epoch, and it must stay in this dict: the check is a
                # whole-dict equality, so a dropped key fails as loudly as a wrong value. (It was
                # dropped once, 2026-07-27, while splitting steps_per_epoch from expected_steps —
                # the guard caught its own maintainer, which is the argument for whole-dict
                # comparison over field-by-field asserts.)
                "total": expected_total,
                # PER-EPOCH, because `_spe` above is what the DATALOADER yields in one pass.
                # Putting the run total here would break the coverage check the moment EPOCHS>1
                # — and break it in the silent direction, by demanding the loader produce a
                # multi-epoch count it never produces.
                "steps": steps_per_epoch,
                "global_batch": _gb,
            }
            if exact_values != expected_values:
                raise RuntimeError(
                    f"EXACT SFT coverage mismatch: actual={exact_values} expected={expected_values}"
                )
            if total_steps != expected_steps or session_limit != expected_steps:
                raise RuntimeError(
                    f"EXACT SFT requires exactly {_epochs} uninterrupted epoch(s) "
                    f"({steps_per_epoch} steps/epoch): "
                    f"TOTAL_STEPS={total_steps} SESSION_LIMIT={session_limit} "
                    f"expected={expected_steps}"
                )
            if save_every <= expected_steps:
                raise RuntimeError(
                    "EXACT SFT requires final-only checkpointing: "
                    f"SAVE_EVERY={save_every} must exceed {expected_steps}"
                )
            if resume_step != 0:
                raise RuntimeError(
                    f"EXACT SFT must begin at optimizer step 0, got resume_step={resume_step}"
                )
            if accelerator.is_main_process:
                log.info(
                    "EXACT SFT COVERAGE PASS: "
                    f"real={_expected_real_samples} padding={expected_padding} "
                    f"total={expected_total} steps={expected_steps} "
                    "objective=equal-sequence-mean/FSDP-rank-average "
                    "unique-real=external-provenance-gate"
                )

    # ── MEMORY FIX 1 (exp5 RCA): free the stale mmap'd model-file page cache ──
    # low_cpu_mem_usage=True mmaps the safetensors; after FSDP copies weights into the sharded
    # flat param, the ~53GB mmap stays resident and pushes the UMA pool over 128GB at the first
    # optimizer step → OOM. Evict the model files from page cache now (uses the existing helper).
    try:
        _ev = 0
        for _sf in glob.glob(os.path.join(model_path, "*.safetensors")):
            _evict_page_cache(_sf); _ev += 1
        if accelerator.is_main_process:
            log.info(f"MEM: evicted {_ev} model safetensors from page cache (freed stale mmap)")
    except Exception as _e:
        log.info(f"MEM: page-cache evict skipped ({_e})")

    # ── MEMORY FIX 2 (exp5b RCA + exp9 3-lane consult): activation checkpointing via accelerate's
    # fsdp_activation_checkpointing=true (applied INSIDE prepare, BEFORE fully_shard — the
    # mandatory order). The earlier manual POST-prepare apply_activation_checkpointing failed
    # because it was the WRONG ORDER (wrappers already transformed). VERIFY it actually applied: ──
    # ── AC VERIFY — count is necessary but NOT sufficient (2026-08-02) ──
    # The previous check accepted ANY count > 0 as a pass. Production reported 256 wrappers across
    # 64 decoder layers and the gate went green — but 256 is 4-per-layer, i.e. Accelerate wrapped
    # each decoder layer's CHILDREN, not the layer. Mechanism, quoted from upstream by HORIZON and
    # verifiable at accelerate/utils/fsdp_utils.py:716-718: the auto-wrap policy is evaluated
    # against the PARENT and then `checkpoint_wrapper` is applied to the child. So a policy naming
    # Qwen3_5DecoderLayer wraps that layer's children.
    # Why it matters: child-level wrapping retains a boundary per CHILD instead of per LAYER, which
    # is the difference between a ~5.4GB hidden-state floor and the ~52GB of activations measured
    # at MAX_SEQ=8192. The gate passed on precisely the configuration it existed to catch — a check
    # of FORM (something is wrapped) reported as a check of TRUTH (the right thing is wrapped).
    # Now records WHICH classes are wrapped and how many, so the granularity is visible in the log
    # of every run rather than inferable only by dividing two numbers by hand.
    _acw = -1
    try:
        _wrapped = [_m for _m in model.modules() if "CheckpointWrapper" in type(_m).__name__]
        _acw = len(_wrapped)
        _inner = collections.Counter()
        for _m in _wrapped:
            _c = getattr(_m, "_checkpoint_wrapped_module", None)
            _inner[type(_c).__name__ if _c is not None else "?"] += 1
        _n_layers = sum(1 for _m in model.modules()
                        if type(_m).__name__.endswith("DecoderLayer"))
        if accelerator.is_main_process:
            log.info(f"AC VERIFY: {_acw} CheckpointWrapper modules over {_n_layers} decoder layers "
                     f"→ {(_acw / _n_layers) if _n_layers else float('nan'):.2f} per layer; "
                     f"wrapped classes = {dict(_inner)}")
            if _acw == 0:
                log.info("AC VERIFY: 0 wrappers → accelerate AC did NOT apply; ESCALATE to manual "
                         "pre-prepare apply_activation_checkpointing")
            elif _n_layers and _acw == _n_layers:
                log.info("AC VERIFY: LAYER-GRANULAR (1 wrapper per decoder layer) — minimum retained state")
            elif _n_layers:
                log.info(f"AC VERIFY: CHILD-GRANULAR — {_acw / _n_layers:.0f}x more retained boundaries "
                         f"than layer-granular. See AC_LAYER_GRANULAR=1.")
    except Exception as _e:
        log.info(f"AC VERIFY skipped ({_e})")
    if _exact_sft_epoch:
        ac_gate = torch.tensor(
            [1 if _acw > 0 else 0],
            device=accelerator.device,
            dtype=torch.long,
        )
        dist.all_reduce(ac_gate, op=dist.ReduceOp.MIN)
        if ac_gate.item() != 1:
            raise RuntimeError(
                "EXACT SFT activation-checkpointing gate failed on at least one rank"
            )
        if accelerator.is_main_process:
            log.info(f"EXACT SFT AC PASS: checkpoint_wrappers_per_rank={_acw}")

    # ── Fix 2: Broadcast non-persistent buffers (MoE gate correction bias) ──
    # Perplexity DR Root Cause 2: sync_module_states only broadcasts Parameters
    # and persistent buffers, NOT non-persistent buffers like e_score_correction_bias
    buf_count = 0
    for name, buf in model.named_buffers():
        if buf is not None and buf.is_cuda:
            dist.broadcast(buf.data, src=0)
            buf_count += 1
    dist.barrier()
    if accelerator.is_main_process:
        log.info(f"Broadcast {buf_count} buffers from rank 0")

    # ── DCP resume (FSDP2-native, POST-prepare — shards placed into the live FSDP2 model) ──
    _resume_epoch, _resume_data_pos = 0, 0
    if _is_dcp_ckpt:
        resume_step, _resume_data_pos, _resume_epoch = _load_checkpoint_dcp(
            model, optimizer, lr_scheduler, delta_path, accelerator)
        dist.barrier()

    # ── BAKE mode: consolidate this DCP checkpoint → a servable HF model dir, then EXIT ──
    # The STANDARD production bake. It reuses the EXACT FSDP2 build + dcp.load resume path above (the
    # only load path empirically proven correct for our use_collectives=False per-rank bundles), then
    # gathers the FULL model-only state dict on rank0 and save_pretrained. Because it runs in the SAME
    # 4-rank distributed context as the save, it sidesteps the offline per-rank-reassembly trap
    # entirely (those rank-local bundles are topology-locked / NOT offline-consolidatable — verified by
    # the 5-seat DCP consult). Model-only (no optimizer/activations) → rank0 peak ≈ own-shard + 54GB
    # gather, memory-safe on the 128GB pool. Launch with BAKE_TO_HF=<out_dir> + RESUME_DELTA=<ckpt>.
    bake_to = os.environ.get("BAKE_TO_HF", "")
    if bake_to:
        if _lora_mode:
            if accelerator.is_main_process:
                log.info(
                    f"BAKE: gathering trainable LoRA tensors only; frozen base excluded → {bake_to}"
                )
            adapter_gb = save_lora_only_fsdp(
                model,
                accelerator,
                bake_to,
                tokenizer=tokenizer,
            )
            if accelerator.is_main_process:
                log.info(f"BAKE COMPLETE: adapter-only artifact ({adapter_gb:.2f}GB) → {bake_to}")
            return
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        if accelerator.is_main_process:
            log.info(f"BAKE: gathering FULL model state dict on rank0 (cpu_offload) → {bake_to}")
        full_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            os.makedirs(bake_to, exist_ok=True)
            total_gb = sum(t.numel() * t.element_size() for t in full_sd.values()) / 1e9
            log.info(f"BAKE: save_pretrained {len(full_sd)} tensors ({total_gb:.1f}GB, safetensors) → {bake_to}")
            unwrapped.save_pretrained(bake_to, state_dict=full_sd,
                                      safe_serialization=True, max_shard_size="5GB")
            if tokenizer is not None:
                tokenizer.save_pretrained(bake_to)
            log.info(f"BAKE COMPLETE → {bake_to}")
        accelerator.wait_for_everyone()
        return

    # ── EXPORT_DCP mode: emit Artifact B (portable model-only COORDINATED sharded DCP), then EXIT ──
    # The production export path (UNANIMOUS Chats ruling, BAKE_ARCHITECTURE_27b.md 2026-07-14):
    # instead of the wedge-prone rank0 FULL-STATE gather (BAKE_TO_HF above), write a model-only
    # SHARDED state dict (full_state_dict=False → NO 51GB gather, no NCCL all-gather → the wedge
    # class cannot occur) with rank coordination ENABLED over a GLOO process group. Gloo exchanges
    # only KB-scale plan/metadata over TCP (sidesteps the RoCE fabric = the wedge's prime suspect);
    # each rank writes its OWN shard locally; a single global .metadata lands on the coordinator.
    # Mira then collects shards + metadata and converts to HF OFFLINE (bake_dcp_offline.py) — the
    # Sparks never run the full gather again. Launch with EXPORT_DCP=<dir> + RESUME_DELTA=<ckpt>.
    export_dcp = os.environ.get("EXPORT_DCP", "")
    if export_dcp:
        import hashlib, json as _json
        import torch.distributed as _dist
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemWriter
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        rank = accelerator.process_index
        # FABRIC PREFLIGHT (Gaia poka-yoke, BAKE_ARCHITECTURE_27b.md Q2): the resume is collective-free
        # (use_collectives=False reads only-local), so the NEXT NCCL collective — new_group()'s internal
        # barrier — is the "first collective post-reboot" that gray-fails on a cold RoCE rail. Force QP
        # bring-up with a 1-element all_reduce FIRST: if the fabric is dead it wedges HERE (localized,
        # diagnosable) instead of deep in the save, and with the Flight-Recorder env it aborts-with-stack.
        if accelerator.is_main_process:
            log.info("EXPORT_DCP: fabric preflight (1-elem NCCL all_reduce → force QP bring-up)...")
        _pf = torch.ones(1, device=accelerator.device)
        _dist.all_reduce(_pf)
        torch.cuda.synchronize()
        if accelerator.is_main_process:
            log.info(f"EXPORT_DCP: fabric preflight OK (all_reduce={_pf.item():.0f}, expect {accelerator.num_processes})")
        # Gloo PG for the coordination collective (NOT NCCL — dodges the RoCE fabric). Created on all
        # ranks in the same order. Assumes gloo is available (torch default build).
        export_pg = _dist.new_group(ranks=list(range(accelerator.num_processes)), backend="gloo")
        os.makedirs(export_dcp, exist_ok=True)
        if accelerator.is_main_process:
            log.info(f"EXPORT_DCP: model-only SHARDED coordinated save (gloo, NO full gather) → {export_dcp}")
        # monitored_barrier over gloo reports which rank fails to arrive (Q2 poka-yoke)
        _dist.monitored_barrier(group=export_pg, timeout=timedelta(minutes=2), wait_all_ranks=True)
        msd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=False, cpu_offload=False))
        writer = FileSystemWriter(export_dcp, single_file_per_rank=True,
                                  sync_files=True, thread_count=1, overwrite=True)
        dcp.save({"model": msd}, storage_writer=writer, process_group=export_pg,
                 use_collectives=True)
        _dist.monitored_barrier(group=export_pg, timeout=timedelta(minutes=5), wait_all_ranks=True)
        # per-rank manifest (filename, bytes, sha256) + READY marker → Mira collector verifies before convert
        my_files = sorted(f for f in os.listdir(export_dcp)
                          if f.endswith(".distcp") and f"__{rank}_" in f)
        man = {"rank": rank, "world": accelerator.num_processes, "files": []}
        for f in my_files:
            p = os.path.join(export_dcp, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            man["files"].append({"name": f, "bytes": os.path.getsize(p), "sha256": h})
        _json.dump(man, open(os.path.join(export_dcp, f"manifest.rank{rank}.json"), "w"), indent=2)
        with open(os.path.join(export_dcp, f"READY.rank{rank}"), "w") as _f:
            _f.write(f"rank={rank} files={len(my_files)}\n")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            has_meta = os.path.exists(os.path.join(export_dcp, ".metadata"))
            log.info(f"EXPORT_DCP COMPLETE → {export_dcp} | global .metadata={'PRESENT' if has_meta else 'MISSING!'} "
                     f"(collect per-rank *.distcp + manifests + this .metadata to Mira; convert offline)")
        accelerator.wait_for_everyone()
        return

    # ── Per-rank shard reload (post-FSDP, legacy non-DCP path) ──
    if delta_path and not _is_dcp_ckpt:
        rank = accelerator.process_index
        shard_file = os.path.join(delta_path, f"model_rank{rank}.pt")
        if not os.path.exists(shard_file):
            shard_file = os.path.join(delta_path, f"shard_{rank}.pt")
        if os.path.exists(shard_file):
            log.info(f"Rank {rank}: loading shard from {shard_file}")
            shard = torch.load(shard_file, map_location="cpu", mmap=True, weights_only=False)
            clean_to_param = {}
            for name, param in model.named_parameters():
                clean_to_param[_clean_fsdp_name(name)] = param
            loaded, skipped = 0, 0
            with torch.no_grad():
                for saved_key, tensor in shard.items():
                    clean_key = _clean_fsdp_name(saved_key)
                    if clean_key in clean_to_param:
                        live = clean_to_param[clean_key]
                        if live.shape == tensor.shape:
                            live.data.copy_(tensor.to(dtype=live.dtype, device=live.device))
                            loaded += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
            del shard
            gc.collect()
            torch.cuda.empty_cache()
            log.info(f"Rank {rank}: shard applied ({loaded} loaded, {skipped} skipped)")
        dist.barrier()

    # ── Freeze/unfreeze was moved BEFORE accelerator.prepare() ──
    # (see above — Perplexity Root Cause 1 fix)
    mem = torch.cuda.mem_get_info()
    if accelerator.is_main_process:
        log.info(f"  CUDA free: {mem[0]/1e9:.1f}GB")

    # ── Optimizer (FSDP1/DDP path — built AFTER model prepare) ──
    # FSDP2 already built + prepared the optimizer together with the model above.
    # Adafactor (not AdamW): factored row/column sums keep optimizer state small — BUT this only
    # engages when params are >=2-D. FSDP1 FULL_SHARD flat-shards to 1-D → dense fp32 state (26.9GB,
    # M5-confirmed) → use FSDP2 for the real memory win. This v1 path is the DDP/9B fallback.
    optimizer_name = "SR-routing AdamW" if _lora_mode else "Adafactor"
    if not _fsdp_v2:
        optimizer, lr_scheduler = _make_optim(model)
        if accelerator.is_main_process:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(
                f"Optimizer: {optimizer_name} "
                f"(single group, {n_trainable/1e9:.2f}B params @ lr={optimizer.param_groups[0]['lr']})"
            )
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    elif accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(
            f"Optimizer: {optimizer_name} "
                f"(FSDP2, prepared with model, {n_trainable/1e9:.2f}B params @ lr={optimizer.param_groups[0]['lr']})"
        )

    if resume_step > 0:
        # ALWAYS force the scheduler to resume_step (2026-07-11 fix): the DCP path's
        # load_state_dict(meta["sched"]) on the accelerate-WRAPPED scheduler did NOT restore the step
        # — the LR reset to WARMUP each cycle (observed: rising 6.6→7.4e-6 at global step 115, i.e. the
        # step/25 warmup ramp, when it should be cos-decaying ~7.5e-6 falling). Explicitly setting the
        # inner LambdaLR's last_epoch = resume_step makes the LR continue the single warmup+decay across
        # every resume (no per-cycle sawtooth). Authoritative + idempotent; applies to DCP and non-DCP.
        # INTERNAL units: the AcceleratedScheduler steps num_processes× per opt step, so last_epoch
        # counts internal steps = resume_step × num_processes (matches the _lr_lambda fix above).
        _np = max(1, accelerator.num_processes)
        _internal = resume_step * _np
        inner_sched = lr_scheduler.scheduler if hasattr(lr_scheduler, 'scheduler') else lr_scheduler
        inner_sched.last_epoch = _internal
        inner_sched._step_count = _internal + 1
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = inner_sched.base_lrs[i] * inner_sched.lr_lambdas[i](_internal)
        if accelerator.is_main_process:
            log.info(f"Scheduler fast-forwarded to opt-step {resume_step} (internal {_internal}, lr={optimizer.param_groups[0]['lr']:.2e})")
    dist.barrier()

    if accelerator.is_main_process:
        mem = torch.cuda.mem_get_info()
        log.info(f"Ready. CUDA free={mem[0]/1e9:.1f}GB")

    # ── Post-FSDP diagnostic forward ──
    log.info(f"Rank {accelerator.process_index}: running post-FSDP diagnostic forward...")
    model.eval()
    post_fsdp_finite = False
    with torch.no_grad():
        diag_ids = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_labels = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_out = model(input_ids=diag_ids, labels=diag_labels)
        diag_loss = diag_out.loss
        diag_logits = diag_out.logits
        post_fsdp_finite = bool(
            torch.isfinite(diag_loss).item() and torch.isfinite(diag_logits).all().item()
        )
        log.info(f"Rank {accelerator.process_index}: POST-FSDP diag: "
                 f"loss={diag_loss.item():.4f} "
                 f"loss_finite={torch.isfinite(diag_loss).item()} "
                 f"logits_shape={diag_logits.shape} "
                 f"logits_nan%={100*torch.isnan(diag_logits).float().mean().item():.1f}% "
                 f"logits_inf%={100*torch.isinf(diag_logits).float().mean().item():.1f}% "
                 f"logits_min={diag_logits.min().item():.4f} "
                 f"logits_max={diag_logits.max().item():.4f} "
                 f"logits_mean={diag_logits[torch.isfinite(diag_logits)].mean().item():.4f}")
        # Check first few param norms post-FSDP
        for i, (name, p) in enumerate(model.named_parameters()):
            if i >= 5:
                break
            pn = p.float().norm().item() if p.numel() > 0 and not p.is_meta else -1
            log.info(f"  param[{i}] {_clean_fsdp_name(name)}: norm={pn:.4f} dtype={p.dtype} device={p.device}")
        del diag_out, diag_logits

    # Test train mode forward (still no_grad, to isolate train vs grad)
    model.train()
    log.info(f"Rank {accelerator.process_index}: running train-mode (no_grad) diagnostic...")
    train_mode_finite = False
    with torch.no_grad():
        diag_ids2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_labels2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_out2 = model(input_ids=diag_ids2, labels=diag_labels2)
        train_mode_finite = bool(torch.isfinite(diag_out2.loss).item())
        log.info(f"Rank {accelerator.process_index}: TRAIN-MODE diag: "
                 f"loss={diag_out2.loss.item():.4f} "
                 f"finite={torch.isfinite(diag_out2.loss).item()}")
        del diag_out2
    if _exact_sft_epoch:
        diagnostic_gate = torch.tensor(
            [1 if post_fsdp_finite and train_mode_finite else 0],
            device=accelerator.device,
            dtype=torch.long,
        )
        dist.all_reduce(diagnostic_gate, op=dist.ReduceOp.MIN)
        if diagnostic_gate.item() != 1:
            raise RuntimeError(
                "EXACT SFT post-FSDP diagnostic gate failed on at least one rank"
            )
        if accelerator.is_main_process:
            log.info("EXACT SFT DIAGNOSTIC PASS: eval_and_train_mode_finite_on_all_ranks")
    dist.barrier()

    # ── DPO dataloader (separate, for periodic DPO steps) ──
    dpo_dir = os.environ.get("DPO_DIR", "")
    dpo_dataloader = None
    dpo_iter = None
    dpo_interval = int(os.environ.get("DPO_INTERVAL", "10"))  # DPO every N steps
    dpo_weight = float(os.environ.get("DPO_WEIGHT", "0.1"))   # DPO loss weight

    if _exact_sft_epoch and dpo_dir:
        raise RuntimeError("EXACT_SFT_EPOCH forbids DPO_DIR")
    if dpo_dir and os.path.isdir(dpo_dir):
        if cpt_bucket_mode:
            raise RuntimeError("DPO_DIR is incompatible with CPT token-budget bucket mode")
        dpo_dataset = CombinedSFTDataset(dpo_dir, "", "", tokenizer, max_seq)
        dpo_sampler = DistributedSampler(dpo_dataset, num_replicas=accelerator.num_processes,
                                          rank=accelerator.process_index, shuffle=True)
        dpo_dataloader = DataLoader(dpo_dataset, batch_size=1, sampler=dpo_sampler,
                                     collate_fn=collate_fn, pin_memory=False, num_workers=0)
        dpo_iter = iter(dpo_dataloader)
        if accelerator.is_main_process:
            log.info(f"DPO dataloader: {len(dpo_dataset)} items, interval={dpo_interval}, weight={dpo_weight}")

    # ── Training loop ──
    os.makedirs(output_dir, exist_ok=True)
    model.train()
    global_step = resume_step
    _sr_probe = {}   # SR write-through probe state (ref snapshot of one mid-net weight)
    _exact_seen_real = 0
    _exact_seen_padding = 0
    _profile_nvtx_active = False
    _profile_nvtx_complete = False

    def _profile_range(name):
        if _profile_nvtx_active:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    if accelerator.is_main_process:
        log.info(f"Starting: steps {resume_step}→{total_steps}, {accelerator.num_processes} nodes")
        if _nsys_profile_step:
            log.info(
                "NSYS PROFILE ARMED: "
                f"absolute_optimizer_step={_nsys_profile_step} "
                "trigger=cudaProfilerApi range=PALIOS_PROFILE_STEP "
                f"scope={'ALL_RANKS' if _nsys_profile_all_ranks else 'rank0'}"
            )

    for epoch in range(_resume_epoch, 100):
        # packed CPT has no manual sampler (accelerate shards); reshuffle via the prepared
        # dataloader if it supports it, else nothing (shuffle=False ⇒ deterministic full-coverage order).
        if sampler is not None:
            sampler.set_epoch(epoch)
        elif hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        # epoch_step = optimizer-step groups completed WITHIN this epoch. Saved as data_pos so a
        # resume can fast-forward past already-trained groups (no double-training, no data replay).
        epoch_step = 0
        # On the resume epoch, skip micro-batches until we've re-passed the groups already trained.
        _skipping = (_is_dcp_ckpt and epoch == _resume_epoch and _resume_data_pos > 0)
        if _skipping and accelerator.is_main_process:
            log.info(f"Resume: fast-forwarding data past {_resume_data_pos} groups in epoch {epoch} "
                     f"(no double-training)")
        for batch in dataloader:
            if global_step >= total_steps:
                break
            # tok/s instrumentation (roadmap §5.4; lands BEFORE any throughput claim/packing decision).
            # Local-rank padded-token count × world_size, windowed to the 10-step log cadence — an
            # APPROXIMATE aggregate (dynamic padding varies per rank; labeled ~ in the log line). No
            # new collectives (wedge discipline): purely local counters + wall clock.
            try:
                _tokwin_n += batch["input_ids"].numel()
            except NameError:
                import time as _time
                _tokwin_n = batch["input_ids"].numel()
                _tokwin_t0 = _time.time()

            # Data-position fast-forward on resume: consume (discard) already-trained groups with
            # NO forward/backward, counting group boundaries, until we reach the saved position.
            if _skipping:
                _ge = bool(batch["group_end"][0].item()) if "loss_denom_tokens" in batch else True
                if _ge:
                    epoch_step += 1
                    if epoch_step >= _resume_data_pos:
                        _skipping = False
                        if accelerator.is_main_process:
                            log.info(f"Resume skip complete: {epoch_step} groups skipped, "
                                     f"training resumes at the next unseen batch")
                continue

            if (
                (accelerator.is_main_process or _nsys_profile_all_ranks)
                and _nsys_profile_step == global_step + 1
                and not _profile_nvtx_complete
                and not _profile_nvtx_active
            ):
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("PALIOS_PROFILE_STEP")
                _profile_nvtx_active = True
                log.info(
                    f"NSYS PROFILE START: optimizer_step={global_step + 1} "
                    f"rank={accelerator.process_index}"
                )

            exact_sft_batch = "is_exact_padding" in batch
            if _exact_sft_epoch != exact_sft_batch:
                raise RuntimeError(
                    "EXACT SFT batch marker mismatch: "
                    f"mode={_exact_sft_epoch} marker={exact_sft_batch}"
                )
            token_budget_batch = "loss_denom_tokens" in batch
            group_end = True
            log_loss = None
            bucket_label = "legacy"
            group_denom_tokens = 0

            if exact_sft_batch:
                padding_mask = batch["is_exact_padding"].bool()
                if padding_mask.numel() != 1:
                    raise RuntimeError(
                        "EXACT SFT objective requires exactly one sample per rank"
                    )
                shift_labels = batch["labels"][..., 1:].contiguous()
                valid_tokens_per_sample = (shift_labels != -100).sum(dim=1)
                invalid_local = (
                    (padding_mask & (valid_tokens_per_sample != 0)).any()
                    or ((~padding_mask) & (valid_tokens_per_sample == 0)).any()
                )
                invalid_flag = torch.tensor(
                    [1 if invalid_local else 0],
                    device=accelerator.device,
                    dtype=torch.long,
                )
                dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX)
                if invalid_flag.item() != 0:
                    raise RuntimeError(
                        "EXACT SFT label-mask invariant failed on at least one rank"
                    )

                with _profile_range("forward"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                with _profile_range("cross_entropy"):
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    token_loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                local_real_tokens = (shift_labels != -100).sum().to(
                    device=accelerator.device,
                    dtype=torch.long,
                )
                local_real_samples = (~padding_mask).sum().to(
                    device=accelerator.device,
                    dtype=torch.long,
                )
                loss = token_loss_sum / local_real_tokens.clamp_min(1)
                local_sequence_loss = loss.detach().float()
                loss_stats = torch.stack([
                    local_sequence_loss,
                    local_real_samples.detach().float(),
                    local_real_tokens.detach().float(),
                ])
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                global_real_samples = int(loss_stats[1].item())
                global_real_tokens = int(loss_stats[2].item())
                if global_real_samples <= 0 or global_real_tokens <= 0:
                    raise RuntimeError(
                        "EXACT SFT optimizer step has no global real sample tokens"
                    )
                log_loss = loss_stats[0] / float(global_real_samples)
                _exact_seen_padding += int(padding_mask.sum().item())
                _exact_seen_real += int(local_real_samples.item())
                del (
                    outputs,
                    shift_logits,
                    shift_labels,
                    token_loss_sum,
                    local_real_tokens,
                    local_real_samples,
                    local_sequence_loss,
                    loss_stats,
                    padding_mask,
                    valid_tokens_per_sample,
                    invalid_flag,
                )
            elif token_budget_batch:
                group_end = bool(batch["group_end"][0].item())
                group_denom_tokens = int(batch["loss_denom_tokens"][0].item())
                bucket_label = batch["bucket"][0] if batch.get("bucket") else "unknown"
                if group_denom_tokens <= 0:
                    raise RuntimeError("CPT token-budget batch missing positive loss denominator")

                with _profile_range("forward"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                with _profile_range("cross_entropy"):
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    token_loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                local_loss_tokens = (shift_labels != -100).sum().to(device=accelerator.device, dtype=torch.long)
                loss = token_loss_sum * accelerator.num_processes / float(group_denom_tokens)

                loss_stats = torch.stack([
                    token_loss_sum.detach().float(),
                    local_loss_tokens.detach().float(),
                ])
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                log_loss = loss_stats[0] / torch.clamp(loss_stats[1], min=1.0)
                del outputs, shift_logits, shift_labels, token_loss_sum, local_loss_tokens, loss_stats
            else:
                # Legacy SFT / diagnostic CPT rollback path. This remains
                # sequence-average loss because those paths do not provide a
                # fixed token-budget group denominator.
                with _profile_range("forward_and_cross_entropy"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                loss = outputs.loss
                log_loss = loss.detach()
                del outputs

            # Periodic DPO step
            if dpo_dataloader and (global_step - resume_step) % dpo_interval == 0 and (global_step - resume_step) > 0:
                try:
                    dpo_batch = next(dpo_iter)
                except StopIteration:
                    dpo_iter = iter(dpo_dataloader)
                    dpo_batch = next(dpo_iter)
                # DPO uses chosen/rejected from the batch
                if dpo_batch.get("is_dpo", False):
                    dpo_beta = 0.1
                    chosen_out = model(input_ids=dpo_batch["input_ids"], labels=dpo_batch["labels"])
                    rej_out = model(input_ids=dpo_batch["rejected_input_ids"], labels=dpo_batch["rejected_labels"])
                    dpo_loss = -torch.nn.functional.logsigmoid(dpo_beta * (-chosen_out.loss - (-rej_out.loss)))
                    loss = loss + dpo_weight * dpo_loss
                    del chosen_out, rej_out

            # Distributed NaN veto
            nan_flag = torch.zeros(1, device=accelerator.device)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_flag.fill_(1.0)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)

            if nan_flag.item() > 0:
                if _exact_sft_epoch:
                    raise RuntimeError(
                        f"EXACT SFT non-finite loss at optimizer step {global_step + 1}"
                    )
                if accelerator.is_main_process:
                    log.warning(f"[step {global_step+1}] NaN/Inf — ALL ranks skipping")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_step += 1
                continue

            # DO NOT use model.no_sync() for accumulation. FSDP no_sync() accumulates the FULL
            # UNSHARDED gradients (~54GB bf16 for 27B) instead of gradient shards — the DOMINANT
            # term in the 45→102GB accumulation climb (torch FSDP no_sync docstring, verbatim:
            # "accumulate the full model gradients (instead of gradient shards) until the eventual
            # sync"; ChatGPT HORIZON lane, source-verified 2026-07-09). reduce_dtype/keep_low_
            # precision_grads DO NOT touch this (they affect the SHARDED grad after reduce-scatter).
            # Fix: reduce-scatter EVERY micro-backward → grads stay SHARDED (~13.5GB bf16) through
            # accumulation → accum ~102G→~58G. Costs ~100 reduce-scatters/step (more comm) but FITS.
            # optimizer.step() still deferred to group_end; FSDP sums the sharded .grad across
            # micro-backwards, token-budget loss normalization unchanged.
            with _profile_range("backward"):
                accelerator.backward(loss)
            if not group_end:
                continue

            # FSDP-AWARE grad clip. torch.nn.utils.clip_grad_norm_ on FSDP use_orig_params
            # grads is a documented misuse (accelerate: "Should be used in place of
            # torch.nn.utils.clip_grad_norm_") — it can gather/materialize the full UNSHARDED
            # params (~54GB) to compute the norm, spiking GPU allocation at the first optimizer
            # step past the ~92GB CUDA-free ceiling → NV_ERR_NO_MEMORY / node reset (measured
            # 2026-07-09: died at the opt-step on the tightest node). accelerator.clip_grad_norm_
            # routes to FSDP's own model.clip_grad_norm_ which clips the SHARDED grads in place
            # (all-reduces only the scalar norm) — no full-param materialization.
            # NCCL-SAFE anti-fragmentation defrag before the first optimizer steps. The opt-step
            # GPU-OOM'd (NV_ERR_NO_MEMORY, ~48GB free) is allocator FRAGMENTATION: after the
            # variable-size grad-accum + activation-checkpoint alloc/free churn, no contiguous
            # region serves Adafactor's ~13-27GB first-step workspace. expandable_segments:True
            # fixes fragmentation structurally BUT breaks multi-node RDMA (IBV_WC_RETRY_EXC_ERR →
            # node wedge; measured 2026-07-09). NCCL-safe fix instead: empty_cache() returns all
            # cached blocks to the driver → defrags the pool ahead of the big Adafactor alloc.
            # First few steps only (Adafactor state is allocated once at step 0 then reused; empty
            # per-step would add cudaMalloc latency every step).
            # (empty_cache removed: expandable_segments:True VMM handles fragmentation; per-step
            # empty_cache FOUGHT the VMM — caused a 9-12 NV_ERR/node storm at step 2-9.)
            # Grad clip. FSDP2 DTensor grads break accelerator.clip_grad_norm_ — it computes a
            # _NormPartial DTensor then does in-place aten.pow_ which DTensor refuses across a placement
            # change ("in-place operations that require placement changes are not supported"). Under FSDP2
            # we SKIP explicit grad-norm clipping: torch.optim.Adafactor's d=1.0 already clips the UPDATE
            # RMS to 1.0 (bounds the effective step regardless of grad magnitude), and the distributed
            # NaN-veto backstops any Inf/NaN — so grad clipping is redundant here. FSDP1/DDP keep it.
            if not _fsdp_v2:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # ── WEDGE FIX + FAIL-LOUD (Family 3-lane RCA, all code-verified 2026-07-22) ──
            # ROOT: torch _adafactor.py:82 `if p.grad is None: continue`. In the hybrid arch a trainable
            # param (e.g. in_proj_qkv) can have a grad on some ranks but None on others (data-dependent).
            # The patched single-tensor Adafactor loop then runs a DIFFERENT NUMBER of iterations per rank
            # → issues a mismatched number of 1-elem all_reduces (:1509/:1514/:1546) → NCCL pairs the k-th
            # with the (k+1)-th → the default_pg 1-element ALLREDUCE deadlock (the watchdog surfaces it at
            # the next guard collective on the non-diverging ranks — hence the nan_flag/mem_flag red herring).
            # FIX (GAIA v2 follow-up, code-verified; CLARITY+LOGOS concur): keep the DTensor-safe patch and
            # keep in_proj_qkv — make the optimizer param set RANK-INVARIANT by zero-filling missing grads.
            if _fsdp_v2 and dist.is_initialized():
                # C0 — diagnostic (first 50 steps): name the culprit if ranks ever disagree.
                if (global_step - resume_step) < 50:
                    _miss = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    _c = torch.tensor([len(_miss)], device=accelerator.device)
                    _allc = [torch.zeros_like(_c) for _ in range(accelerator.num_processes)]
                    dist.all_gather(_allc, _c)
                    if len({t.item() for t in _allc}) != 1 and accelerator.is_main_process:
                        log.error(f"WEDGE-GUARD C0 [step {global_step+1}]: rank-divergent grad-None "
                                  f"counts={[t.item() for t in _allc]} e.g. {_miss[:5]} — C1 zero-filling")
                # C1 — THE FIX: zero-fill missing grads so _adafactor.py:82 skips nothing on any rank
                # (torch.zeros_like on a DTensor param returns a DTensor of the same placement).
                for p in model.parameters():
                    if p.requires_grad and p.grad is None:
                        p.grad = torch.zeros_like(p)
                # C3 — FAIL LOUD, never hang: abort in ms if any residual param/grad-set divergence remains.
                # v2 (Family 3-lane RCA 2026-07-23; CLARITY+LOGOS code-verified, py-spy caught rank 3 alone
                # inside the is_dt-gated all_reduce): the round-1 signature checked shape + grad-presence but
                # NOT grad DTensor-ness (is_dt = hasattr(grad,"to_local")). The wedge IS a per-rank is_dt
                # divergence — the Adafactor's 3 "UNCONDITIONAL" all_reduces (:1523/:1528/:1560) are actually
                # gated `if is_dt:`, so one rank issues them and the others skip → 1-elem ALLREDUCE deadlock.
                # Add is_dt to the sig so this guard catches the ACTUAL wedge axis, and name the culprit param.
                _pnames = [n for n, p in model.named_parameters() if p.requires_grad]
                _isdt = [1 if hasattr(p.grad, "to_local") else 0
                         for _, p in model.named_parameters() if p.requires_grad]
                _sig = hash(tuple((tuple(p.shape), p.grad is not None, hasattr(p.grad, "to_local"))
                                  for p in model.parameters() if p.requires_grad)) % (2**31)
                _st = torch.tensor([_sig], device=accelerator.device, dtype=torch.long)
                _sg = [torch.zeros_like(_st) for _ in range(accelerator.num_processes)]
                dist.all_gather(_sg, _st)
                if len({x.item() for x in _sg}) != 1:
                    # Name the culprit: all_gather the per-param is_dt bit-vector, report the divergent params.
                    _iv = torch.tensor(_isdt, device=accelerator.device, dtype=torch.int32)
                    _ig = [torch.zeros_like(_iv) for _ in range(accelerator.num_processes)]
                    dist.all_gather(_ig, _iv)
                    _diff = [(_pnames[j], [g[j].item() for g in _ig])
                             for j in range(len(_pnames))
                             if len({g[j].item() for g in _ig}) != 1]
                    raise RuntimeError(f"WEDGE-GUARD C3 rank={dist.get_rank()} sig-divergent on the is_dt axis. "
                                       f"Diverging params (name, per-rank is_dt): {_diff[:8]} — "
                                       f"aborting fast in ms (NOT the 1hr NCCL hang).")
            with _profile_range("optimizer_update"):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_step += 1

            # ── SR WRITE-THROUGH PROBE (Gaia gate, 3-lane consult 2026-07-13) ──
            # The null runs' updates silently rounded to zero in the bf16 master; this catches that
            # class in ~20 steps instead of a wasted 16h epoch. Healthy: mean|dW| grows ~linearly,
            # ≫ bf16 half-ULP noise (rel 2^-9). Flat-at-noise = updates STILL vanishing → halt run.
            # ULP-ANCHORED bands (Gaia recalibration, sr_gate_verdict 2026-07-14): probe a decoder
            # mlp.down_proj (u = mean|w|*2^-7 ≈ 6.1e-5); liveness = cross 1 ULP by ~step 10-15;
            # PASS @40 = [0.5u, 20u]; FAIL-LOW <0.5u (starving/regression); FAIL-HIGH >20u (too hot).
            if accelerator.is_main_process and (global_step - resume_step) in (1, 10, 20, 40):
                try:
                    if "_name" not in _sr_probe:
                        # Prefer a decoder mlp down_proj (Gaia's calibrated target, u≈6.1e-5);
                        # FALL BACK to first non-embed/non-norm 2-D param (the previously-working
                        # predicate) so liveness is never silently lost to a naming mismatch.
                        # ROOT CAUSE of the revenue_regate all-steps skip (2026-07-14): activation
                        # checkpointing inserts a wrapper infix in runtime names — e.g.
                        # "linear_attn._checkpoint_wrapped_conv1d.weight" — so the literal
                        # "mlp.down_proj" substring NEVER matches ("mlp._checkpoint_wrapped_
                        # down_proj") → next() StopIteration, whose str() is "" (the silent skip).
                        # Match the two tokens independently instead.
                        _sr_probe["_name"] = next(
                            (n for n, p in model.named_parameters()
                             if p.requires_grad and p.ndim == 2
                             and "mlp" in n and "down_proj" in n),
                            None)
                        if _sr_probe["_name"] is None:
                            _sr_probe["_name"] = next(
                                n for n, p in model.named_parameters()
                                if p.requires_grad and p.ndim == 2
                                and "embed" not in n and "norm" not in n)
                            log.info(f"[SR-DELTA] no mlp.down_proj match in named_parameters — "
                                     f"FALLBACK target: {_sr_probe['_name']}")
                    _pp = dict(model.named_parameters())[_sr_probe["_name"]].detach()
                    _pl = (_pp.to_local() if hasattr(_pp, "to_local") else _pp).float()
                    if "_ref" not in _sr_probe:
                        _sr_probe["_ref"] = _pl.clone()
                        _sr_probe["_ulp"] = _pl.abs().mean().item() * 2 ** -7
                        log.info(f"[SR-DELTA] ref captured: {_sr_probe['_name']} "
                                 f"(step {global_step}, ULP u={_sr_probe['_ulp']:.3e})")
                    else:
                        _k = global_step - resume_step
                        _delta = (_pl - _sr_probe["_ref"]).abs().mean().item()
                        _u = _sr_probe["_ulp"]
                        _r = _delta / _u if _u > 0 else float("inf")
                        _sr_probe["_last_step"] = _k
                        _sr_probe["_last_ratio"] = _r
                        if _k >= 40:
                            _tag = ("PASS (in Gaia band [0.5u,20u])" if 0.5 <= _r <= 20 else
                                    "FAIL-LOW (<0.5u — starving/regression)" if _r < 0.5 else
                                    "FAIL-HIGH (>20u — too hot, lower lr)")
                        else:
                            _tag = ("LIVE (crossed 1 ULP)" if _r >= 1.0 else
                                    "warming (sub-ULP — expect crossing by ~step 10-15)")
                        log.info(f"[SR-DELTA] {_sr_probe['_name']}: mean|dW|={_delta:.3e} = {_r:.2f}x ULP → {_tag}")
                except Exception as _sre:
                    # typed + repr'd: StopIteration strs to "" which hid the 2026-07-14 skip cause
                    log.info(f"[SR-DELTA] probe skipped ({type(_sre).__name__}: {_sre!r})")

            # MEMGROWTH DIAGNOSTIC (2026-07-10): the run dies ~step 3 from per-step memory GROWTH
            # (79G→117-118G). Log EVERY step for the first 12 (then every 10) to catch it live and
            # tell leak-vs-caching apart: resNow=torch reserved (allocator's total, incl cached-freed
            # blocks), allocNow=live tensors. If resNow GROWS but allocNow FLAT → allocator caching
            # (variable-batch hoarding; fix gc_threshold/fixed-batch). If allocNow GROWS → real leak
            # (retained tensor/graph). sysUsed=system RAM (= GPU on unified) to catch CPU-side growth.
            if accelerator.is_main_process and (global_step <= resume_step + 12 or global_step % 10 == 0):
                mem = torch.cuda.mem_get_info()
                vm = psutil.virtual_memory()
                stats = torch.cuda.memory_stats()
                alloc = stats["allocated_bytes.all.current"]
                reserved = stats["reserved_bytes.all.current"]
                frag = (reserved - alloc) / reserved if reserved > 0 else 0
                # Transient peak since last window — the backward-pass high-water that the
                # inter-step free= trough never shows (wedge diagnostic, consult 2026-06-29).
                peak_res = stats["reserved_bytes.all.peak"] / 1e9
                peak_alloc = stats["allocated_bytes.all.peak"] / 1e9
                extra = (
                    f" bucket={bucket_label} group_tokens={group_denom_tokens}"
                    if token_budget_batch else ""
                )
                loss_str = f"{log_loss.item():.4f}" if log_loss is not None else "n/a"
                # windowed ~tok/s (local padded tokens × world, since last log line)
                try:
                    import time as _time
                    _dt = max(1e-6, _time.time() - _tokwin_t0)
                    _toks = f" ~tok/s={_tokwin_n * accelerator.num_processes / _dt:,.0f}"
                    _tokwin_n = 0; _tokwin_t0 = _time.time()
                except NameError:
                    _toks = ""
                log.info(f"[step {global_step}] loss={loss_str} "
                         f"lr={lr_scheduler.get_last_lr()[0]:.2e} free={mem[0]/1e9:.1f}GB "
                         f"resNow={reserved/1e9:.1f}GB allocNow={alloc/1e9:.1f}GB "
                         f"sysUsed={vm.used/1e9:.1f}GB frag={frag:.1%} "
                         f"peakRes={peak_res:.1f}GB peakAlloc={peak_alloc:.1f}GB{extra}{_toks}")
                torch.cuda.reset_peak_memory_stats()  # per-window peak

            if global_step == resume_step + 1 and accelerator.is_main_process:
                torch.cuda.synchronize()
                free_b, total_b = torch.cuda.mem_get_info()
                alloc_b = torch.cuda.memory_allocated()
                param_b = sum(p.numel() * p.element_size() for p in model.parameters())
                grad_b = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
                optim_b = sum(v.numel() * v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
                log.info(f"FIRST STEP: free={free_b/1e9:.1f}GB alloc={alloc_b/1e9:.1f}GB")
                log.info(f"  params={param_b/1e9:.1f}GB grads={grad_b/1e9:.1f}GB optim={optim_b/1e9:.1f}GB")
                # M5 (GAIA): is Adafactor factorizing? exp_avg_sq = DENSE Adam-style (bad, ~4B/param,
                # 26.9GB — factorization disengaged on FSDP1 1-D flat shards). exp_avg_sq_row/col =
                # FACTORED (good, <1GB). optim_dtypes shows if state is fp32 (the 4B/param tell).
                try:
                    _st0 = next(iter(optimizer.state.values()))
                    _keys = sorted(k for k in _st0.keys())
                    _dtypes = {k: str(v.dtype) for k, v in _st0.items() if torch.is_tensor(v)}
                    _factored = any("_row" in k or "_col" in k for k in _keys)
                    log.info(f"  M5 OPTIM STATE KEYS={_keys} dtypes={_dtypes} "
                             f"FACTORED={_factored} (DENSE exp_avg_sq={'exp_avg_sq' in _keys and not _factored})")
                except Exception as _e:
                    log.info(f"  M5 optim-state probe skipped ({_e})")

            # Periodic save (relative to session start, not global step)
            saved_this_step = False
            steps_this_session = global_step - resume_step
            if steps_this_session > 0 and steps_this_session % save_every == 0:
                if _use_dcp:
                    _save_checkpoint_dcp(model, optimizer, lr_scheduler, tokenizer,
                                         output_dir, global_step, epoch, epoch_step, accelerator)
                else:
                    _save_checkpoint(model, optimizer, lr_scheduler, tokenizer,
                                    output_dir, global_step, epoch, keystone_layers, accelerator)
                saved_this_step = True

            # VARIABLE-SHAPE MEMORY GUARD (tutor 2026-07-22).
            # WHY: under expandable_segments:False (mandatory — :True kills the node on this RoCE
            # fabric, f531d64) the allocator cannot defragment mid-run; fragmentation only clears on
            # the between-session REBOOT. The clean FRAGMENTATION EXIT below was gated on a FIXED
            # step count (session_limit), but fragmentation-per-step GROWS through the length-sorted
            # epoch: early sessions (short seqs) finish 250 steps fine, later sessions (long seqs)
            # hit the ceiling first — session 2 died at step 490 on a 5GB all-gather that could not
            # allocate at 83% frag, 10 steps short of its step-500 exit. A fixed step limit cannot
            # know how long a length-varying session can safely run; free memory can.
            # HOW: exit-checkpoint-reboot BEFORE the ceiling, triggered by measured free memory.
            # The flag is ALL-REDUCE MAX'd so if ANY rank is low, ALL exit together at this clean
            # step boundary — never one rank exiting into a collective the others still enter.
            # RECLAIM-THEN-EXIT (tutor 2026-07-22, refined after the guard fired at step 270 on a
            # transient trough): the free-memory drop is allocator CACHE, not live allocation —
            # allocNow stays flat ~13.7GB while resNow climbs 53->71GB as the allocator hoards blocks
            # for variable batch sizes. So when free dips below the RECLAIM band, first RELEASE the
            # hoarded cache (Jesse: "you cannot hold anything on GPUs"); only exit if free is STILL
            # below the hard EXIT floor AFTER reclaiming. empty_cache is a LOCAL op (no collective),
            # safe to call per-rank; the exit decision is all-reduce MAX'd so all ranks exit together.
            # Fixed packed CPT is a different allocation domain: every step has the same B×sequence shape,
            # so its 110.9GB cache is bounded and reusable. Reclaiming it every step forces hundreds of
            # cudaMalloc/cudaFree calls on the next identical step. Keep the session-limit checkpoint
            # boundary for packed CPT, but reserve this memory-driven guard for variable-shape data.
            _mem_exit = False
            _reclaimed = False
            _free_b = 0
            if not _fixed_packed_cpt:
                _free_b, _ = torch.cuda.mem_get_info()
                if _free_b < _MEM_RECLAIM_FREE_B:
                    gc.collect(); torch.cuda.empty_cache()
                    _free_b, _ = torch.cuda.mem_get_info()   # re-measure after releasing the cache
                    _reclaimed = True
                _mem_low = 1 if _free_b < _MEM_EXIT_FREE_B else 0
                if dist.is_initialized():
                    _flag = torch.tensor([_mem_low], device=accelerator.device)
                    dist.all_reduce(_flag, op=dist.ReduceOp.MAX)
                    _mem_exit = bool(_flag.item())
                else:
                    _mem_exit = bool(_mem_low)

            # Session limit OR memory floor (post-reclaim): clean exit + checkpoint + reboot-to-defrag
            steps_this_session = global_step - resume_step
            _resume_required = global_step < total_steps and (
                steps_this_session >= session_limit or _mem_exit
            )
            if _resume_required:
                if _mem_exit and accelerator.is_main_process:
                    log.info(f"[step {global_step}] MEMORY-GUARD EXIT: free={_free_b/1e9:.1f}GB "
                             f"< {_MEM_EXIT_FREE_B/1e9:.0f}GB floor after reclaim (reclaimed={_reclaimed}) "
                             f"— checkpoint + reboot to defrag (step count {steps_this_session}/{session_limit})")
                accelerator.wait_for_everyone()
                gc.collect()
                if not saved_this_step:
                    if _use_dcp:
                        _save_checkpoint_dcp(model, optimizer, lr_scheduler, tokenizer,
                                             output_dir, global_step, epoch, epoch_step, accelerator)
                    else:
                        _free_for_save(model, optimizer)
                        _save_checkpoint(model, None, lr_scheduler, tokenizer,
                                        output_dir, global_step, epoch, keystone_layers, accelerator)
                if accelerator.is_main_process:
                    log.info(f"[step {global_step}] FRAGMENTATION EXIT — "
                             f"Resume: RESUME_DELTA=...checkpoint-{global_step}")
                accelerator.wait_for_everyone()
                if dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sys.exit(0)

            if _profile_nvtx_active:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
                _profile_nvtx_active = False
                _profile_nvtx_complete = True
                log.info(f"NSYS PROFILE COMPLETE: optimizer_step={global_step}")

        if global_step >= total_steps:
            break

    if _exact_sft_epoch:
        exact_seen = torch.tensor(
            [_exact_seen_real, _exact_seen_padding],
            device=accelerator.device,
            dtype=torch.long,
        )
        dist.all_reduce(exact_seen, op=dist.ReduceOp.SUM)
        # RUN TOTALS, so these scale with EPOCHS. `dataset` is ONE epoch; the model sees it
        # `_epochs` times, so 2 epochs of 60 real rows is 120 real samples seen — not a dose
        # mismatch. This check kept a one-epoch expectation after the multiplier landed and
        # failed a run that had completed every step correctly.
        expected_padding = (len(dataset) - _expected_real_samples) * _epochs
        expected_real = _expected_real_samples * _epochs
        if (
            global_step != total_steps
            or exact_seen[0].item() != expected_real
            or exact_seen[1].item() != expected_padding
        ):
            raise RuntimeError(
                "EXACT SFT dose mismatch: "
                f"step={global_step}/{total_steps} "
                f"real={exact_seen[0].item()}/{expected_real} "
                f"padding={exact_seen[1].item()}/{expected_padding}"
            )
        if accelerator.is_main_process:
            log.info(
                "EXACT SFT DOSE PASS: "
                f"optimizer_steps={global_step} real={exact_seen[0].item()} "
                f"padding={exact_seen[1].item()}"
            )
        sr_probe_pass = 1
        if accelerator.is_main_process:
            last_probe_step = int(_sr_probe.get("_last_step", 0))
            last_probe_ratio = float(_sr_probe.get("_last_ratio", float("nan")))
            sr_probe_pass = int(
                last_probe_step >= 20
                and math.isfinite(last_probe_ratio)
                and 0.5 <= last_probe_ratio <= 20.0
            )
        sr_probe_gate = torch.tensor(
            [sr_probe_pass],
            device=accelerator.device,
            dtype=torch.long,
        )
        dist.all_reduce(sr_probe_gate, op=dist.ReduceOp.MIN)
        if sr_probe_gate.item() != 1:
            raise RuntimeError("EXACT SFT SR write-through gate failed at or before step 20")
        if accelerator.is_main_process:
            log.info(
                "EXACT SFT SR PASS: "
                f"step={last_probe_step} ratio={last_probe_ratio:.2f}x bf16 ULP"
            )

    if _use_dcp:
        _save_checkpoint_dcp(model, optimizer, lr_scheduler, tokenizer,
                             output_dir, global_step, epoch, epoch_step, accelerator, final=True)
    else:
        _free_for_save(model, optimizer)
        _save_checkpoint(model, None, lr_scheduler, tokenizer,
                        output_dir, global_step, epoch, keystone_layers, accelerator, final=True)
    if final_lora_dir:
        if not _lora_mode:
            raise RuntimeError("FINAL_LORA_DIR is only valid with LORA_MODE=1")
        adapter_gb = save_lora_only_fsdp(
            model,
            accelerator,
            final_lora_dir,
            tokenizer=tokenizer,
        )
        if accelerator.is_main_process:
            log.info(
                f"FINAL LORA COMPLETE: adapter-only artifact ({adapter_gb:.2f}GB) "
                f"→ {final_lora_dir}"
            )
    accelerator.wait_for_everyone()


def _evict_page_cache(filepath):
    """Evict file from page cache to protect UMA headroom."""
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def _free_for_save(model, optimizer):
    """Free optimizer state + gradients for save."""
    import gc
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if optimizer is not None:
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    mem = torch.cuda.mem_get_info()
    log.info(f"Pre-save cleanup: freed optimizer+grads, now {mem[0]/1e9:.1f}GB free GPU")


def _save_checkpoint_dcp(model, optimizer, lr_scheduler, tokenizer,
                         output_dir, step, epoch, data_pos, accelerator, final=False):
    """FSDP2-native RESUMABLE checkpoint via torch.distributed.checkpoint (DCP).
    Each rank writes its OWN DTensor shards — NO full-param gather (fixes the summon_full_params
    54GB-gather OOM). Saves model + optimizer (Adafactor factored state) + scheduler + counters + RNG.

    NO-SHARED-FS DESIGN (exp11, corrected 2026-07-10): use_collectives=False so DCP writes a
    SELF-CONTAINED per-rank bundle — `__<rank>.metadata` + `__<rank>_0.distcp` — instead of a single
    coordinated global `.metadata` that DEDUPLICATES replicated tensors across ranks. The default
    (collective) save puts each replicated param/buffer in only ONE rank's file, so on a no-shared-FS
    cluster a rank would need to read ANOTHER rank's file at load (FileNotFoundError — the real-27B
    failure). With use_collectives=False every rank's bundle is complete on its own node, so
    dcp.load reads ONLY-LOCAL (empirically proven, incl. replicated tensors). trainer_meta.pt +
    COMPLETE are also written per-rank so each node is fully self-contained (resume needs NO
    cross-node distribution). This is the enabler for the base+adapters architecture."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    ckpt_name = "final" if final else f"checkpoint-{step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    dcp_dir = os.path.join(ckpt_dir, "dcp")
    os.makedirs(dcp_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    # RECLAIM the allocator's reserved-but-unused cache BEFORE the save (2026-07-11 fix): at batch=4
    # the torch CUDA allocator holds ~94GB reserved (resNow~110GB vs allocNow~16GB) while the driver
    # sees only ~7GB free. dcp.save host-stages each ~7.3GB shard, and on GB10 UNIFIED memory that
    # staging draws from the SAME pool → the save died at step 60 for lack of driver-visible headroom
    # (shards wrote, metadata/COMPLETE didn't → unresumable). empty_cache returns the cache to the
    # driver (frees only UNUSED cached blocks — does NOT touch the live model/optimizer state we're
    # about to save). Safe: expandable_segments is OFF, and this is a ONE-TIME per-save reclaim (NOT
    # the banned per-step empty_cache that fought the VMM).
    import gc
    gc.collect(); torch.cuda.empty_cache(); gc.collect()
    mem = torch.cuda.mem_get_info()
    if accelerator.is_main_process:
        log.info(f"DCP save → {ckpt_dir} (sharded, self-contained per-rank, no gather) | free={mem[0]/1e9:.1f}GB")
    # sharded model+optimizer state (DTensors) — no gather; use_collectives=False → per-rank bundle
    msd, osd = get_state_dict(model, optimizer)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=dcp_dir, use_collectives=False)
    accelerator.wait_for_everyone()
    # Per-rank trainer meta + COMPLETE (each node self-contained; RNG is per-rank = correct on resume)
    meta = {
        "format": "dcp_v2", "step": step, "epoch": epoch, "data_pos": data_pos,
        "num_ranks": accelerator.num_processes,
        "max_seq": int(os.environ.get("MAX_SEQ", "8192")),
        "sched": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()},
    }
    torch.save(meta, os.path.join(ckpt_dir, "trainer_meta.pt"))
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)
    accelerator.wait_for_everyone()
    # ATOMIC completion marker LAST (per-rank) — resume/collect only trusts a bundle with COMPLETE
    with open(os.path.join(ckpt_dir, "COMPLETE"), "w") as _f:
        _f.write(f"step={step} epoch={epoch} data_pos={data_pos} rank={accelerator.process_index}\n")
    if accelerator.is_main_process:
        log.info(f"DCP save COMPLETE: checkpoint-{step} (per-rank self-contained bundles, atomic)")
    accelerator.wait_for_everyone()


def _load_checkpoint_dcp(model, optimizer, lr_scheduler, delta_path, accelerator):
    """Load a DCP checkpoint into the ALREADY-FSDP2-PREPARED model+optimizer (call POST-prepare).
    get_state_dict → dcp.load (each rank reads its own shards) → set_state_dict. Returns
    (resume_step, data_pos, epoch). Verifies the COMPLETE marker first (never resume a partial)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    if not os.path.exists(os.path.join(delta_path, "COMPLETE")):
        raise RuntimeError(f"DCP resume refused: no COMPLETE marker in {delta_path} (partial/corrupt)")
    dcp_dir = os.path.join(delta_path, "dcp")
    # RESUME_MODEL_ONLY (2026-07-23): PROACTIVE model-only resume for an OPTIMIZER SWITCH (e.g.
    # Adafactor→AdamW at the LoRA root fix). The checkpoint's optim state is incompatible with the new
    # optimizer ("expected BytesStorageMetadata but found TensorStorageMetadata"), and ATTEMPTING the
    # optim load raises inside dcp.load's plan reduce_scatter — a failed COLLECTIVE that does NOT recover
    # cleanly across ranks (a try/except retry desyncs the process group). So NEVER attempt it: load MODEL
    # weights ONLY and start optimizer state FRESH (momentum re-warms in a few steps; bias-correction
    # handles it; trained weights fully preserved). The driver sets this flag only for the transition
    # resume (from a pre-AdamW checkpoint); once an AdamW checkpoint saves, subsequent resumes load optim.
    if os.environ.get("RESUME_MODEL_ONLY", "0") == "1":
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        if accelerator.is_main_process:
            log.warning("RESUME_MODEL_ONLY=1: MODEL-ONLY resume — optimizer state FRESH (optimizer switch, "
                        "trained weights preserved).")
        _mo = get_model_state_dict(model)
        dcp.load({"model": _mo}, checkpoint_id=dcp_dir)
        set_model_state_dict(model, _mo)
        del _mo
    elif os.environ.get("BAKE_LORA_ONLY", "0") == "1":
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
        msd, _osd = get_state_dict(model, optimizer)
        # DCP collective plans require the exact same key set on every rank.  A local
        # shard-filter can diverge, so derive the union once via object all-gather and
        # filter every rank against that globally identical list.
        import hashlib
        _local_keys = sorted(k for k in msd if "lora_A" in k or "lora_B" in k)
        _all_keys = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(_all_keys, _local_keys)
        _keys = sorted(set().union(*map(set, _all_keys)))
        _key_hash = hashlib.sha256("\n".join(_keys).encode()).hexdigest()[:16]
        log.info("BAKE_LORA_ONLY key-set rank=%s count=%d hash=%s", accelerator.process_index, len(_keys), _key_hash)
        if any(sorted(x) != _keys for x in _all_keys):
            raise RuntimeError(f"BAKE_LORA_ONLY divergent LoRA key sets: hashes={[hashlib.sha256(chr(10).join(sorted(x)).encode()).hexdigest()[:16] for x in _all_keys]}")
        lora_msd = {k: msd[k] for k in _keys}
        if len(lora_msd) != 704:
            raise RuntimeError(f"BAKE_LORA_ONLY expected 704 LoRA tensors, found {len(lora_msd)}")
        dcp.load({"model": lora_msd}, checkpoint_id=dcp_dir)
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        set_model_state_dict(model, lora_msd, options=StateDictOptions(strict=False))
        del msd, _osd, lora_msd
    else:
        msd, osd = get_state_dict(model, optimizer)          # templates (correct sharding)
        dcp.load({"model": msd, "optim": osd}, checkpoint_id=dcp_dir)
        set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)
        # RECLAIM the load templates + allocator cache before training resumes (2026-07-11 fix): msd/osd
        # are DUPLICATE copies of the ~13.5GB model shard + optimizer state; left referenced + cached they
        # leave no headroom, so the FIRST post-resume step OOM'd (NV_ERR_NO_MEMORY, run hung at step 11).
        del msd, osd
    import gc
    gc.collect(); torch.cuda.empty_cache(); gc.collect()
    meta = torch.load(os.path.join(delta_path, "trainer_meta.pt"),
                      map_location="cpu", weights_only=False)
    if lr_scheduler is not None and meta.get("sched") is not None:
        lr_scheduler.load_state_dict(meta["sched"])
    if meta.get("rng"):
        try:
            torch.set_rng_state(meta["rng"]["torch"])
            torch.cuda.set_rng_state_all(meta["rng"]["cuda"])
        except Exception as _e:
            log.info(f"DCP resume: RNG restore skipped ({_e})")
    if accelerator.is_main_process:
        log.info(f"DCP RESUME: step={meta.get('step')} epoch={meta.get('epoch')} "
                 f"data_pos={meta.get('data_pos')} (model+optim+sched+rng restored, sharded)")
    return meta.get("step", 0), meta.get("data_pos", 0), meta.get("epoch", 0)


def _save_checkpoint(model, optimizer, lr_scheduler, tokenizer,
                     output_dir, step, epoch, keystone_layers, accelerator, final=False):
    """Save trainable weights using FSDP summon_full_params."""
    ckpt_name = "final" if final else f"checkpoint-{step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    rank = accelerator.process_index

    os.makedirs(ckpt_dir, exist_ok=True)
    mem = torch.cuda.mem_get_info()
    log.info(f"Rank {rank}: saving to {ckpt_dir} | free={mem[0]/1e9:.1f}GB")
    accelerator.wait_for_everyone()

    import gc
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    gc.collect()
    torch.cuda.empty_cache()

    trainable_state = {}
    shard_gb = 0.0
    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        # FSDP: params sharded — gather full params on rank0 to save.
        with FSDP.summon_full_params(model, rank0_only=True, writeback=False):
            if accelerator.is_main_process:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_state[name] = param.detach().cpu().clone()
                shard_gb = sum(t.numel() * t.element_size() for t in trainable_state.values()) / 1e9
                log.info(f"Gathered {len(trainable_state)} trainable tensors ({shard_gb:.2f}GB) on rank 0")
    else:
        # DDP: full model replicated on every rank — read rank0's copy directly (no summon).
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            for name, param in unwrapped.named_parameters():
                if param.requires_grad:
                    trainable_state[_clean_fsdp_name(name)] = param.detach().cpu().clone()
            shard_gb = sum(t.numel() * t.element_size() for t in trainable_state.values()) / 1e9
            log.info(f"DDP save: {len(trainable_state)} trainable tensors ({shard_gb:.2f}GB) from rank0")

    if accelerator.is_main_process:
        from safetensors.torch import save_file

        # Full-FT save: dump every trainable tensor (which is every tensor)
        # into a single safetensors file under the standard name the proven
        # resume code looks for (`trainable_weights.safetensors`). The earlier
        # v8 used `model.safetensors` which broke the RESUME_DELTA path.
        out_file = os.path.join(ckpt_dir, "trainable_weights.safetensors")
        save_file(trainable_state, out_file)
        log.info(f"Saved {len(trainable_state)} tensors ({shard_gb:.2f}GB) to {out_file}")

        # Save metadata
        meta = {
            "step": step,
            "epoch": epoch,
            "num_ranks": accelerator.num_processes,
            "max_seq": int(os.environ.get("MAX_SEQ", "8192")),
            "method": "full_ft_dense_9b_v1",
        }
        meta_file = os.path.join(ckpt_dir, "trainer_meta.pt")
        torch.save(meta, meta_file)
        _evict_page_cache(meta_file)

        tokenizer.save_pretrained(ckpt_dir)
    accelerator.wait_for_everyone()

    mem = torch.cuda.mem_get_info()
    if accelerator.is_main_process:
        log.info(f"Rank {rank}: saved {shard_gb:.2f}GB trainable checkpoint | free={mem[0]/1e9:.1f}GB")
    else:
        log.info(f"Rank {rank}: checkpoint save complete | free={mem[0]/1e9:.1f}GB")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
