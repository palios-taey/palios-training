#!/usr/bin/env python3
"""LoRA SFT for Qwen3.6-27B careers offload adapters (knowledge packs + tasks).

Single-node, single-GPU (one Spark). No FSDP, no multi-node => no wedge.
Trains on chat-messages JSONL; loss masked to ASSISTANT tokens only.
Rows with meta.frozen_regression==true are NEVER trained (held-out recall/replay set).

Data format (one JSON object per line):
  {"messages":[{"role":"system","content":...},
               {"role":"user","content":...},
               {"role":"assistant","content":...}],
   "meta":{"task":"k1_history","frozen_regression":false, ...}}
"""
import os, json, argparse, math, time
from datetime import timedelta
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict
# 4-node FSDP2 via accelerate — proven stack from dense-9b/trainers/train_fsdp_dense_9b.py
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict,
    get_state_dict, StateDictOptions)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_lora_sft")


def save_lora_only_fsdp(model, accelerator, out_dir, adapter_name="default"):
    """Save only trainable PEFT weights from an FSDP-wrapped model via DCP.

    Copied VERBATIM from dense-9b/trainers/train_fsdp_dense_9b.py (proven FSDP2 LoRA-only save):
    gathers the full (unsharded) trainable state on rank0/CPU (ignore_frozen_params=True) then
    PEFT save_pretrained's adapter-only, embeddings excluded."""
    accelerator.wait_for_everyone()

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
    )
    trainable_state = get_model_state_dict(model, options=options)

    total_gb = 0.0
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        total_gb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in trainable_state.values()
        ) / 1e9
        log.info(
            "Saving %d trainable tensors to %s (%.2f GB)",
            len(trainable_state),
            out_dir,
            total_gb,
        )
        unwrapped.save_pretrained(
            out_dir,
            selected_adapters=[adapter_name],
            state_dict=trainable_state,
            safe_serialization=True,
            save_embedding_layers=False,
            is_main_process=True,
        )

    accelerator.wait_for_everyone()
    return total_gb


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all; for smoke tests)")
    # Qwen3.6 GDN hybrid (48 linear-attn + 16 full-attn). Gaia's RECONCILED target list (binding):
    # self_attn q/k/v/o (16 full layers) + mlp (all 64) + GDN CONTENT path in_proj_qkv,out_proj
    # (48 linear layers). Deliberately EXCLUDES the GDN state-dynamics gates in_proj_a/b/z (decay/
    # write/output gate) — freezing how the recurrent memory decays/writes; adapting only what enters
    # it. See SFT_RECIPE_RECONCILE_v1.md + Gaia reconciliation 2026-07-19.
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj")
    ap.add_argument("--lane-weights", default="", help="weighted mixture, e.g. 'stage2_scorer=0.45,jesse_voice=0.35,repo_capability=0.12,values=0.08' (empty=uniform shuffle)")
    ap.add_argument("--tiny-lane-cap", type=int, default=3, help="max repeats/example for lanes below --tiny-lane-threshold (Gaia's mandatory anti-memorization cap)")
    ap.add_argument("--tiny-lane-threshold", type=int, default=500, help="lanes with fewer rows than this get the tiny-lane-cap")
    ap.add_argument("--decoder-layer-cls", default="Qwen3_5DecoderLayer", help="transformer block class name for FSDP2 auto-wrap (V4 step 0 pins this)")
    ap.add_argument("--pg-timeout-s", type=int, default=1800, help="process-group timeout. Gaia: keep it SHORT (120) for dry-runs so a desync fails in 2min, not 30")
    ap.add_argument("--max-steps", type=int, default=0, help="hard stop after N optimizer steps (0=off) — for the 2-min save dry-run")
    ap.add_argument("--gather-frozen", action="store_true", help="Fix-A (H2a): gather FULL state dict (uniform collective schedule on every rank), filter to trainable on rank0. Avoids ignore_frozen_params shaping the gather schedule.")
    ap.add_argument("--save-cpu-offload", action="store_true", default=True)
    ap.add_argument("--no-save-cpu-offload", dest="save_cpu_offload", action="store_false", help="Fix-B (H2b): drop cpu_offload in the gather (adapter is ~100M, offload buys nothing)")
    ap.add_argument("--seed", type=int, default=42, help="set_seed(device_specific=False) so the CappedMixtureSampler pool is IDENTICAL across ranks (accelerate re-shards pool[rank::world] — divergent pools corrupt the mixture)")
    ap.add_argument("--resume-adapter", default="", help="path to a CPT adapter dir to CONTINUE (recipe: CPT->SFT->DPO on one growing adapter). Empty = fresh adapter on base.")
    ap.add_argument("--resume-state", default="", help="checkpoint dir (adapter + state.pt) to RESUME a session-cycled run: restores adapter weights + optimizer + scheduler + gstep + RNG")
    ap.add_argument("--session-seconds", type=int, default=0, help="max wall-seconds of training this session; checkpoint to adapter-live + exit(0) when exceeded (respect the ~2h thermal wall). 0=unlimited")
    ap.add_argument("--eval-probes", nargs="*", default=[], help="held-out probe jsonl(s) to eval each epoch")
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--eval-max-new", type=int, default=200)
    return ap.parse_args()


import re


def _grade(gen, target, thresh=0.7):
    g, t = re.sub(r"\s+", " ", gen.strip()).lower(), re.sub(r"\s+", " ", target.strip()).lower()
    exact = (g == t) or (t in g)
    toks = [w for w in re.findall(r"[a-z0-9_\-./]+", t) if len(w) > 3]
    contain = (sum(1 for w in toks if w in g) / len(toks)) if toks else 0.0
    return exact, contain >= thresh


@torch.no_grad()
def run_probe_eval(model, tok, probe_files, max_new):
    """Offline generation eval on held-out probes each epoch (no online-loss churn)."""
    model.eval()
    out = []
    for pf in probe_files:
        rows = [json.loads(l) for l in open(pf) if l.strip()]
        rows = [r for r in rows if r.get("meta", {}).get("frozen_regression")]
        n = ex = con = 0
        for r in rows:
            n += 1
            m = r["messages"]
            prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            e, c = _grade(gen, m[-1]["content"])
            ex += int(e); con += int(c)
        out.append((os.path.basename(pf), n, ex, con))
    model.train()
    return out


class ChatSFTDataset(Dataset):
    """system+user+assistant single-turn examples; loss on assistant tokens only."""
    def __init__(self, path, tok, max_seq, limit=0):
        self.tok, self.max_seq = tok, max_seq
        self.rows = []
        self.lanes = []            # per-row lane id (from meta.lane) for the weighted sampler
        skipped_frozen = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if ex.get("meta", {}).get("frozen_regression"):
                    skipped_frozen += 1
                    continue
                if not ex.get("messages"):
                    continue
                self.rows.append(ex["messages"])
                self.lanes.append(ex.get("meta", {}).get("lane", "_unlaned"))
        if limit:
            self.rows = self.rows[:limit]
            self.lanes = self.lanes[:limit]
        import collections as _c
        print(f"[data] {len(self.rows)} train rows loaded ({skipped_frozen} frozen-regression rows held out)")
        print(f"[data] lane counts: {dict(_c.Counter(self.lanes))}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]
        # apply_chat_template return type is unreliable across versions; render TEXT
        # (tokenize=False) then tokenize the string -> plain list[int]. (proven pattern)
        prompt_text = self.tok.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, tokenize=False)
        full_text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=False)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"]
        # prompt_text is a prefix of full_text (assistant header shared) -> mask prompt
        plen = len(prompt_ids) if full_ids[:len(prompt_ids)] == prompt_ids else min(len(prompt_ids), len(full_ids))
        if len(full_ids) > self.max_seq:
            # NEVER TRUNCATE (Jesse, standing invariant).
            raise RuntimeError(
                f"row exceeds max_seq: len(full_ids)={len(full_ids)} > {self.max_seq}; "
                f"corpus must be pre-chunked or windowed — truncation is not permitted"
            )
        input_ids = full_ids
        labels = ([-100] * plen + full_ids[plen:])[: len(input_ids)]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lbl, att = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * (m - n))
        lbl.append(b["labels"] + [-100] * (m - n))
        att.append([1] * n + [0] * (m - n))
    return (torch.tensor(ids), torch.tensor(lbl), torch.tensor(att))


class CappedMixtureSampler(torch.utils.data.Sampler):
    """Single-stage weighted lane mixture with a per-example cap on tiny lanes.
    target draws per lane = weight * epoch_size, but no lane draws more than lane_size*cap
    when lane_size < threshold, and no example is repeated more than `cap` times. Deterministic
    composition; reshuffled each epoch (RNG seeded by torch global state)."""
    def __init__(self, lanes, lane_weights, cap, threshold):
        import collections as _c
        self.lanes = lanes
        self.tgt = {}
        for kv in lane_weights.split(","):
            k, v = kv.split("="); self.tgt[k.strip()] = float(v)
        counts = _c.Counter(lanes)
        missing = [l for l in counts if l not in self.tgt]
        if missing:
            raise SystemExit(f"ABORT: --lane-weights missing a weight for present lanes: {missing}")
        self.lane_idx = _c.defaultdict(list)
        for i, l in enumerate(lanes):
            self.lane_idx[l].append(i)
        epoch = len(lanes)
        self.per_lane_draws, self.cap, self.threshold = {}, cap, threshold
        for l, idxs in self.lane_idx.items():
            n = len(idxs)
            eff_cap = cap if n < threshold else 10**9
            target = round(self.tgt[l] * epoch)
            self.per_lane_draws[l] = min(target, n * eff_cap)

    def _build(self):
        pool = []
        for l, idxs in self.lane_idx.items():
            n, draws = len(idxs), self.per_lane_draws[l]
            full = draws // n
            for _ in range(full):
                pool += idxs
            rem = draws - full * n
            if rem:
                perm = torch.randperm(n).tolist()
                pool += [idxs[j] for j in perm[:rem]]
        perm = torch.randperm(len(pool)).tolist()
        return [pool[j] for j in perm]

    def __iter__(self):
        return iter(self._build())

    def __len__(self):
        return sum(self.per_lane_draws.values())


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)

    # ── 4-node FSDP2 via accelerate (proven stack: dense-9b/trainers/train_fsdp_dense_9b.py) ──
    # Bind THIS rank's CUDA device BEFORE process-group init so NCCL does not "guess device ID from
    # global rank" and hang the first collective (proven fabric fix in the reference trainer; ADDED
    # beyond the reconciled DESIGN — see report).
    _local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank)
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=a.pg_timeout_s))])
    # IDENTICAL sampler pool on every rank: accelerate re-shards pool[rank::world]; divergent pools
    # corrupt the lane mixture. device_specific=False (default) → same torch CPU-RNG on all ranks.
    set_seed(a.seed)
    main_proc = accelerator.is_main_process
    if main_proc:
        print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
              f"world={accelerator.num_processes} device={accelerator.device}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ── ORDER (non-negotiable): base → adapters → optimizer/sched → FSDP2 policy → prepare ──
    if main_proc:
        print("[model] loading base (bf16, low_cpu_mem_usage, NO .to(device) — FSDP2 shards it)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.config.use_cache = False
    # NOTE: HF gradient_checkpointing_enable() is a no-op/harmful under FSDP2 (proven in the reference
    # trainer, exp9 — it only sets a Python attr, never installs torch.utils.checkpoint, and conflicts
    # with accelerate's own AC). Omitted here. If activation memory OOMs, enable accelerate's
    # fsdp_activation_checkpointing (config/env), NOT the HF call. (Deviation from single-node — report.)

    resuming = bool(a.resume_state and os.path.exists(os.path.join(a.resume_state, "state.pt")))
    if resuming:
        # Session-cycled resume: adapter WEIGHTS reloaded here (pre-shard, is_trainable); optimizer/
        # scheduler/step/RNG restored AFTER prepare (below).
        if main_proc:
            print(f"[model] RESUMING session-cycled adapter: {a.resume_state}", flush=True)
        model = PeftModel.from_pretrained(model, a.resume_state, is_trainable=True)
    elif a.resume_adapter:
        # Recipe: CPT -> SFT -> DPO on ONE growing adapter. Continue the CPT adapter (is_trainable).
        if main_proc:
            print(f"[model] CONTINUING CPT adapter: {a.resume_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, a.resume_adapter, is_trainable=True)
    else:
        lc = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                        target_modules=a.target_modules.split(","), task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
    if main_proc:
        model.print_trainable_parameters()

    # ── data + sampler (built IDENTICALLY on every rank; sampler code UNCHANGED) ──
    ds = ChatSFTDataset(a.data, tok, a.max_seq, a.limit)
    if a.lane_weights:
        # Gaia recipe: single-stage weighted mixture (NOT row duplication in-file) with a per-example
        # CAP on tiny lanes so 21 values / 160 repo rows aren't drawn 17x and memorized verbatim.
        sampler = CappedMixtureSampler(ds.lanes, a.lane_weights, a.tiny_lane_cap, a.tiny_lane_threshold)
        dl = DataLoader(ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
        if main_proc:
            print(f"[data] CAPPED-WEIGHTED sampler: mix {sampler.tgt} "
                  f"caps<{a.tiny_lane_threshold}rows@{a.tiny_lane_cap}x -> {sampler.per_lane_draws} "
                  f"draws/epoch (total {len(sampler)})", flush=True)
    else:
        dl = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    # steps/epoch is PER-RANK: accelerate shards the (world-identical) pool across ranks, so each rank
    # iterates ~len(dl)/world batches/epoch, then grad_accum micro-batches per optimizer step. (The
    # single-node math did NOT divide by world — corrected for 4-node — see report.)
    world = max(1, accelerator.num_processes)
    per_rank_batches = math.ceil(len(dl) / world)
    steps_per_epoch = math.ceil(per_rank_batches / a.grad_accum)
    total_steps = steps_per_epoch * a.epochs

    # ── optimizer + cosine schedule BUILT BEFORE prepare (FSDP2 re-points the optimizer to the DTensor
    # params inside prepare). Keep AdamW — LoRA is ~100M params; the CPT Adafactor/DTensor stack solved
    # a full-param problem that does not apply here. ──
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    # accelerate's AcceleratedScheduler steps the wrapped scheduler num_processes× per .step() (proven
    # in the reference: internal_step = world×optimizer_step). Build the schedule in INTERNAL units
    # (optimizer-steps × world) so warmup/cosine don't run world× too fast. (Not in DESIGN — report.)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup * world, total_steps * world)
    if main_proc:
        print(f"[train] {len(dl)} pool draws → ~{per_rank_batches}/rank × {a.epochs} epochs / "
              f"grad_accum {a.grad_accum} = {total_steps} optim steps (world={world})", flush=True)

    # ── resolve the decoder-layer class by NAME from the live module tree (robust to import path) ──
    layer_cls = None
    for m in model.modules():
        if m.__class__.__name__ == a.decoder_layer_cls:
            layer_cls = m.__class__
            break
    if layer_cls is None:
        raise SystemExit(f"ABORT: decoder layer class {a.decoder_layer_cls!r} not found in model.modules()")

    # ── FSDP2 MixedPrecisionPolicy(bf16) + transformer auto-wrap for the decoder layer, set on the
    # accelerator's fsdp_plugin BEFORE prepare (mirrors reference ~1005-1018 / 1180-1207) ──
    _fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if _fsdp_plugin is not None:
        import functools
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from torch.distributed.fsdp import MixedPrecisionPolicy
        _fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
        _fsdp_plugin.auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
        if main_proc:
            print(f"[fsdp2] MixedPrecisionPolicy(bf16) + auto-wrap {a.decoder_layer_cls}", flush=True)
    elif main_proc:
        print("[fsdp2] WARNING: no fsdp_plugin on accelerator.state — running non-FSDP", flush=True)
    _fsdp_v2 = _fsdp_plugin is not None and int(getattr(_fsdp_plugin, "fsdp_version", 1)) == 2

    # ── prepare model + optimizer + dataloader + scheduler TOGETHER (FSDP2 requirement) ──
    model, opt, dl, sched = accelerator.prepare(model, opt, dl, sched)

    # ── restore optimizer/scheduler/step/RNG AFTER prepare on a session-cycled resume.
    #    (adapter WEIGHTS were already loaded pre-prepare via PeftModel.from_pretrained(is_trainable).) ──
    gstep = 0
    if resuming:
        stt = torch.load(os.path.join(a.resume_state, "state.pt"), map_location="cpu")
        set_optimizer_state_dict(model, opt, optim_state_dict=stt["opt"],
                                 options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        sched.load_state_dict(stt["sched"])
        gstep = stt["gstep"]
        torch.set_rng_state(stt["torch_rng"].to("cpu", torch.uint8))
        if main_proc:
            print(f"[resume] restored optimizer+sched+RNG at gstep={gstep}/{total_steps}", flush=True)

    def save_ckpt(tag):
        """V4-FIX (2026-07-21): the previous version called save_lora_only_fsdp (get_model_state_dict)
        and get_optimizer_state_dict as SEPARATE collective sequences. On 4-node FSDP2 that desynced
        the ranks and hung an all-gather (COALESCED 69632->278528) for the full 1800s PG timeout ->
        SIGABRT at step 30. The PROVEN reference (_save_checkpoint_dcp in train_fsdp_dense_9b.py) uses
        the COMBINED get_state_dict(model, optimizer) — model+optimizer gathered in ONE consistent
        collective sequence. Matching that. ignore_frozen_params keeps it adapter-only (~100M, tiny)."""
        d = os.path.join(a.out, tag)
        # ---- collectives FIRST, identical on every rank, ZERO rank0-gated work between them
        #      (Gaia 2026-07-21: the hang is the FIRST get_model_state_dict all-gather; the entry
        #      barrier completed on all ranks, so it is a gather-schedule divergence, not a barrier
        #      mismatch. save_lora_only_fsdp is INLINED here to drop its two internal barriers.) ----
        if a.gather_frozen:
            # Fix-A (H2a): do NOT let ignore_frozen_params shape the collective schedule — gather the
            # FULL state dict (uniform schedule on every rank), filter to trainable on rank0 only.
            msd = get_model_state_dict(model, options=StateDictOptions(
                full_state_dict=True, cpu_offload=a.save_cpu_offload))
        else:
            msd = get_model_state_dict(model, options=StateDictOptions(
                full_state_dict=True, cpu_offload=a.save_cpu_offload, ignore_frozen_params=True))
        osd = get_optimizer_state_dict(model, opt, options=StateDictOptions(
            full_state_dict=True, cpu_offload=a.save_cpu_offload))
        accelerator.wait_for_everyone()
        # ---- rank0-only I/O: NO collectives past this point ----
        if accelerator.is_main_process:
            os.makedirs(d, exist_ok=True)
            if a.gather_frozen:
                trainable = {n for n, p in accelerator.unwrap_model(model).named_parameters() if p.requires_grad}
                msd = {k: v for k, v in msd.items() if k in trainable}
            if not a.save_cpu_offload:
                msd = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in msd.items()}
            accelerator.unwrap_model(model).save_pretrained(
                d, state_dict=msd, safe_serialization=True, save_embedding_layers=False)
            # DROP cuda_rng (per DESIGN): only the CPU RNG governs the cross-rank-identical sampler pool.
            torch.save({"opt": osd, "sched": sched.state_dict(), "gstep": gstep,
                        "torch_rng": torch.get_rng_state()},
                       os.path.join(d, "state.pt"))
            tok.save_pretrained(d)
            print(f"[ckpt] saved {tag} @gstep={gstep}/{total_steps} "
                  f"({len(msd)} adapter tensors)", flush=True)
        accelerator.wait_for_everyone()

    def batch_stream():
        while True:
            for b in dl:
                yield b

    if main_proc:
        print(f"[train] {total_steps} total steps, grad_accum {a.grad_accum}; "
              f"session budget {a.session_seconds}s (0=unlimited); starting at gstep={gstep}", flush=True)
    model.train()
    stream = batch_stream()
    session_start = time.time(); accum = 0; sess_step0 = gstep
    opt.zero_grad()
    _hard_stop = a.max_steps if a.max_steps else total_steps
    while gstep < total_steps and gstep < _hard_stop:
        ids, lbl, att = next(stream)
        ids = ids.to(accelerator.device); lbl = lbl.to(accelerator.device); att = att.to(accelerator.device)
        out = model(input_ids=ids, attention_mask=att, labels=lbl)
        accelerator.backward(out.loss / a.grad_accum)
        accum += 1
        if accum % a.grad_accum == 0:
            # DESIGN asked for an unconditional accelerator.clip_grad_norm_(model.parameters(), 1.0).
            # The PROVEN reference SKIPS clip under FSDP2: DTensor grads make accelerator.clip_grad_norm_
            # build a _NormPartial DTensor then do an in-place aten.pow_ across a placement change, which
            # DTensor refuses ("in-place operations that require placement changes are not supported") →
            # crash at the first optimizer step. LoRA params live inside the wrapped decoder layers, so
            # they are DTensors too. Deviating to match the reference — clip on FSDP1/non-FSDP only.
            if not _fsdp_v2:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            gstep += 1
            if main_proc and gstep % a.log_every == 0:
                el = time.time() - session_start
                sps = el / max(gstep - sess_step0, 1)
                print(f"[step {gstep}/{total_steps}] loss={out.loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {sps:.1f}s/step sess={el/60:.0f}m", flush=True)
            # Decide the session-stop FIRST (distributed-safe: ALL ranks must AGREE before any
            # collective save, else a rank enters save_ckpt on a time-cross a peer hasn't seen).
            session_stop = False
            if a.session_seconds and gstep < total_steps:
                stop = torch.tensor(
                    1 if (time.time() - session_start) > a.session_seconds else 0,
                    device=accelerator.device)
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
                session_stop = bool(int(stop.item()))
            # V4-FIX: save AT MOST ONCE per step. Previously save_every and the session-limit could
            # BOTH fire at the same gstep (they did at step 30), running two collective save
            # sequences back-to-back. One save, one collective sequence.
            if session_stop:
                save_ckpt("adapter-live")
                if main_proc:
                    print(f"[SESSION_LIMIT] {a.session_seconds}s elapsed @gstep={gstep}/{total_steps} "
                          f"— checkpointed to adapter-live, resumable. EXIT.", flush=True)
                accelerator.wait_for_everyone()
                return
            if gstep % a.save_every == 0 and gstep < total_steps:
                save_ckpt("adapter-live")
    save_ckpt("adapter-final")
    if main_proc:
        print(f"[done] TRAINING COMPLETE @gstep={gstep}/{total_steps} -> {a.out}/adapter-final", flush=True)


if __name__ == "__main__":
    main()
