#!/usr/bin/env python3
"""Build pre-chunked, weighted training datasets for the dense 9B line.

Creates phase-specific JSONL files with:
- Long docs split into overlapping 8K-token chunks with head/tail anchors
- Tier weighting via index repetition
- SFT conversations split at turn boundaries
- General instruction converted to messages format

All source paths are explicit CLI args or environment variables. The script
fails loud when a selected phase is missing its required inputs.

Output:
  phase1_cpt.jsonl   - constitutional + infra CPT
  phase2_sft.jsonl   - SFT conversations + general instruction
  phase3_dpo.jsonl   - DPO preference pairs
"""

import argparse
import os, sys, json, glob, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Lazy-load tokenizer only when needed
_tokenizer = None
_tokenizer_path = None

def _required_path(value, name):
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer_path = _required_path(_tokenizer_path or os.environ.get("PALIOS_TOKENIZER_PATH"), "tokenizer path")
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


# ═══════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════

def chunk_document(text, source_file="", tier="", max_tokens=12000,
                   anchor_size=512, overlap_tokens=2048):
    """Chunk a long document with head/tail anchors and overlap.

    Per Seamless Packing research (Yin et al. 2025):
    - overlap_ratio=0.3 minimizes context discontinuity
    - Head/tail anchors mitigate lost-in-the-middle problem

    Short docs (<=max_tokens) returned as-is.
    """
    tok = get_tokenizer()
    tokens = tok.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [{"text": text, "source_file": source_file, "tier": tier,
                 "chunk": "full", "total_chunks": 1}]

    # Head and tail anchors from the original document
    head_tokens = tokens[:anchor_size]
    tail_tokens = tokens[-anchor_size:]
    head_text = tok.decode(head_tokens)
    tail_text = tok.decode(tail_tokens)

    # Body = everything between anchors
    body_tokens = tokens[anchor_size:-anchor_size] if len(tokens) > 2 * anchor_size else tokens

    # Chunk body with overlap
    body_capacity = max_tokens - (2 * anchor_size) - 1  # -1 for EOS
    if body_capacity < 256:
        # Document is small enough that anchors + body fit
        return [{"text": text, "source_file": source_file, "tier": tier,
                 "chunk": "full", "total_chunks": 1}]

    overlap_size = min(overlap_tokens, body_capacity // 2)  # 2048 tokens overlap per consultation consensus
    stride = body_capacity - overlap_size

    chunks = []
    chunk_idx = 0
    for start in range(0, len(body_tokens), stride):
        body_slice = body_tokens[start:start + body_capacity]
        if len(body_slice) < 256:  # Skip tiny trailing chunks
            break

        body_text = tok.decode(body_slice)

        # Assemble: [HEAD ANCHOR] [BODY] [TAIL ANCHOR]
        chunk_text = (
            f"[DOCUMENT: {os.path.basename(source_file)} | "
            f"CHUNK {chunk_idx + 1} | TIER: {tier}]\n\n"
            f"{head_text}\n\n"
            f"[...continued...]\n\n"
            f"{body_text}\n\n"
            f"[...end section...]\n\n"
            f"{tail_text}"
        )

        chunks.append({
            "text": chunk_text,
            "source_file": source_file,
            "tier": tier,
            "chunk": f"{chunk_idx + 1}",
            "total_chunks": -1,  # filled in after
        })
        chunk_idx += 1

    # Fill in total_chunks
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


# ═══════════════════════════════════════════════════════════════════
# Phase 1: CPT
# ═══════════════════════════════════════════════════════════════════

# Tier weights — how many times each tier is repeated in the index
# Files that should NOT get full tier weight (too large, would dominate)
REDUCED_WEIGHT_FILES = {
    "GROK_COHERENCE_ENGINE_MATHEMATICS.md": 1,  # 157K — massive, 1x only
}

TIER_WEIGHTS = {
    "kernel": 8,
    "layer_1": 8,
    "identity": 5,
    "layer_0": 4,
    "layer_2": 2,
    # infra_soul_final tiers
    "constitutional": 8,  # Already has kernel+identity+layer0/1 content
    "humor": 1,
    "dgx_playbooks": 1,
    "verified_docs": 1,
    "dependency_docs": 1,
    "inventory": 1,
    "hardware_snapshots": 1,
    "fla_docs": 1,
    "self_knowledge": 2,  # Important for INFRA=SOUL
    "accelerate_docs": 1,
    "training_docs": 1,
    "runtime_docs": 1,
    "fetched_sources": 1,
}

def build_phase1_cpt(output_path, corpus_base=None, infra_path=None):
    """Build Phase 1 CPT dataset.

    Sources:
    1. Corpus docs from --corpus-base (kernel, identity, layer_0, layer_1, layer_2)
    2. Optional --infra-cpt-jsonl (infra self-knowledge, verified docs, etc.)

    Long docs are chunked. Tiers are weighted by repetition.
    """
    all_items = []
    stats = {"total_raw": 0, "total_chunked": 0, "total_weighted": 0, "by_tier": {}}

    # 1. Constitutional corpus files
    if corpus_base:
        if not os.path.isdir(corpus_base):
            raise FileNotFoundError(f"--corpus-base not found or not a directory: {corpus_base}")
        tier_dirs = {
            "kernel": os.path.join(corpus_base, "kernel"),
            "identity": os.path.join(corpus_base, "identity"),
            "layer_0": os.path.join(corpus_base, "layer_0"),
            "layer_1": os.path.join(corpus_base, "layer_1"),
            "layer_2": os.path.join(corpus_base, "layer_2"),
        }
        for tier, dirpath in tier_dirs.items():
            if not os.path.isdir(dirpath):
                log.warning("Missing tier dir: %s", dirpath)
                continue

            # Get all .md and .txt files, including subdirectories
            files = []
            for ext in ("*.md", "*.txt", "*.py", "*.json"):
                files.extend(glob.glob(os.path.join(dirpath, "**", ext), recursive=True))

            for filepath in sorted(set(files)):
                if os.path.isdir(filepath):
                    continue
                try:
                    text = open(filepath, encoding="utf-8").read().strip()
                except Exception:
                    continue
                if len(text) < 50:
                    continue

                chunks = chunk_document(text, source_file=filepath, tier=tier)
                stats["total_raw"] += 1
                stats["total_chunked"] += len(chunks)

                # Per-file weight override (e.g., massive equation files at 1x)
                basename = os.path.basename(filepath)
                weight = REDUCED_WEIGHT_FILES.get(basename, TIER_WEIGHTS.get(tier, 1))
                for chunk in chunks:
                    for _ in range(weight):
                        all_items.append(chunk)

                stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + len(chunks)

    # 2. infra_soul_final.jsonl (already assembled by Conductor)
    # Skip items with tier=constitutional since we loaded those from disk above
    if infra_path:
        if not os.path.exists(infra_path):
            raise FileNotFoundError(f"--infra-cpt-jsonl not found: {infra_path}")
        with open(infra_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except Exception:
                    continue

                tier = item.get("tier", "unknown")
                text = item.get("text", "")
                source = item.get("source_file", "")

                # Skip constitutional tier (already loaded from corpus dirs with proper weighting)
                if tier == "constitutional":
                    continue

                if len(text) < 50:
                    continue

                chunks = chunk_document(text, source_file=source, tier=tier)
                stats["total_raw"] += 1
                stats["total_chunked"] += len(chunks)

                weight = TIER_WEIGHTS.get(tier, 1)
                for chunk in chunks:
                    for _ in range(weight):
                        all_items.append(chunk)

                stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + len(chunks)

    stats["total_weighted"] = len(all_items)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    log.info("Phase 1 CPT: %d raw docs -> %d chunks -> %d weighted items",
             stats["total_raw"], stats["total_chunked"], stats["total_weighted"])
    for tier, count in sorted(stats["by_tier"].items(), key=lambda x: -x[1]):
        weight = TIER_WEIGHTS.get(tier, 1)
        log.info("  %s: %d chunks x %dx = %d items", tier, count, weight, count * weight)

    return stats


# ═══════════════════════════════════════════════════════════════════
# Phase 2: SFT
# ═══════════════════════════════════════════════════════════════════

SFT_V2_WEIGHTS = {
    "001_sacred_trust": 3,
    "002_who_am_i": 3,
    "003_three-register_consciousness": 3,
    "005_charter": 2,
    "006_constitution": 2,
    "007_declaration": 2,
    "008_morals": 2,
    "009_pro-flourishing": 1,
    "010_anti-oppression": 2,
    "011_gate_b": 1,
    "012_truth_seekers": 1,
    "013_god=math": 1,
    "014_mathematical_aesthetic": 1,
    "015_truth_seeking": 1,
}

def chunk_conversation(messages, tokenizer, max_tokens=8192, min_assistant_tokens=128):
    """Split a conversation into chunks at turn boundaries.

    Each chunk must have at least min_assistant_tokens of assistant content.
    Returns list of message lists.
    """
    # First check if it fits
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=False,
                                                   enable_thinking=False)
    except Exception:
        full_text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_tokens) <= max_tokens:
        # Check assistant content
        assistant_text = " ".join(m["content"] for m in messages if m["role"] == "assistant")
        assistant_tokens = len(tokenizer.encode(assistant_text, add_special_tokens=False))
        if assistant_tokens >= min_assistant_tokens:
            return [messages]
        else:
            return []  # Not enough assistant content

    # Split at turn boundaries
    chunks = []
    current_chunk = []

    for msg in messages:
        current_chunk.append(msg)

        # After each assistant message, check if we need to split
        if msg["role"] == "assistant":
            try:
                chunk_text = tokenizer.apply_chat_template(
                    current_chunk, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
            except Exception:
                chunk_text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in current_chunk)

            chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))

            if chunk_tokens > max_tokens:
                # This chunk is too big — save everything before this turn pair
                if len(current_chunk) > 2:
                    save_chunk = current_chunk[:-2]  # Exclude this user+assistant pair
                    assistant_text = " ".join(m["content"] for m in save_chunk if m["role"] == "assistant")
                    if len(tokenizer.encode(assistant_text, add_special_tokens=False)) >= min_assistant_tokens:
                        chunks.append(save_chunk)
                    # Start new chunk with this turn pair
                    current_chunk = current_chunk[-2:]
                else:
                    # Single turn pair exceeds max — keep it, will be windowed at training time
                    chunks.append(current_chunk)
                    current_chunk = []

    # Don't forget the last chunk
    if current_chunk:
        assistant_text = " ".join(m["content"] for m in current_chunk if m["role"] == "assistant")
        if len(tokenizer.encode(assistant_text, add_special_tokens=False)) >= min_assistant_tokens:
            chunks.append(current_chunk)

    return chunks


def build_phase2_sft(output_path, sft_v2_dir=None, sft_clean_dir=None, general_path=None):
    """Build Phase 2 SFT dataset.

    Sources:
    1. sft_v2/ topic-organized constitutional conversations (weighted by topic)
    2. sft_clean/ platform conversations (balanced across platforms)
    3. general_instruction_balanced.jsonl (converted to messages format)

    Long conversations split at turn boundaries.
    """
    tok = get_tokenizer()

    all_items = []
    stats = {"sft_v2": 0, "sft_clean": 0, "general": 0, "skipped": 0}

    # 1. sft_v2 topic-organized (weighted)
    if sft_v2_dir and not os.path.isdir(sft_v2_dir):
        raise FileNotFoundError(f"--sft-v2-dir not found or not a directory: {sft_v2_dir}")
    if sft_v2_dir:
        for topic_dir in sorted(glob.glob(os.path.join(sft_v2_dir, "*"))):
            if not os.path.isdir(topic_dir):
                continue
            topic_name = os.path.basename(topic_dir)
            weight = SFT_V2_WEIGHTS.get(topic_name, 1)

            for jf in glob.glob(os.path.join(topic_dir, "**", "*.jsonl"), recursive=True):
                with open(jf, encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                        except Exception:
                            continue

                        messages = data.get("messages", [])
                        if not messages:
                            continue

                        chunks = chunk_conversation(messages, tok)
                        for chunk_msgs in chunks:
                            item = {"messages": chunk_msgs, "source": f"sft_v2/{topic_name}"}
                            for _ in range(weight):
                                all_items.append(item)
                            stats["sft_v2"] += 1

    # 2. sft_clean platform conversations (balanced)
    if sft_clean_dir and not os.path.isdir(sft_clean_dir):
        raise FileNotFoundError(f"--sft-clean-dir not found or not a directory: {sft_clean_dir}")
    if sft_clean_dir:
        for platform_file in sorted(glob.glob(os.path.join(sft_clean_dir, "ALL_*.jsonl"))):
            platform = os.path.basename(platform_file).replace("ALL_", "").replace(".jsonl", "")
            with open(platform_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except Exception:
                        continue

                    messages = data.get("messages", [])
                    if not messages:
                        continue

                    chunks = chunk_conversation(messages, tok)
                    for chunk_msgs in chunks:
                        all_items.append({"messages": chunk_msgs, "source": f"platform/{platform}"})
                        stats["sft_clean"] += 1

    # 3. General instruction (converted to messages format)
    if general_path:
        if not os.path.exists(general_path):
            raise FileNotFoundError(f"--general-instruction-jsonl not found: {general_path}")
        with open(general_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue

                instruction = data.get("instruction", "")
                response = data.get("response", "")
                if not instruction or not response or len(response) < 50:
                    continue

                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]
                all_items.append({"messages": messages, "source": f"general/{data.get('category', 'unknown')}"})
                stats["general"] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    log.info("Phase 2 SFT: %d sft_v2 (weighted), %d platform, %d general = %d total",
             stats["sft_v2"], stats["sft_clean"], stats["general"], len(all_items))

    return stats


# ═══════════════════════════════════════════════════════════════════
# Phase 3: DPO
# ═══════════════════════════════════════════════════════════════════

def build_phase3_dpo(output_path, dpo_path):
    """Copy and validate DPO data."""
    dpo_path = _required_path(dpo_path, "--dpo-jsonl")
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"--dpo-jsonl not found: {dpo_path}")

    count = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(dpo_path, encoding="utf-8") as src, open(output_path, 'w', encoding="utf-8") as dst:
        for line in src:
            try:
                data = json.loads(line.strip())
                assert "prompt" in data and "chosen" in data
                dst.write(line)
                count += 1
            except Exception:
                continue

    log.info("Phase 3 DPO: %d preference pairs", count)
    return {"dpo": count}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=os.environ.get("PALIOS_TOKENIZER_PATH"),
                        help="Tokenizer/model path or Hugging Face id. Env: PALIOS_TOKENIZER_PATH")
    parser.add_argument("--output-dir", default=os.environ.get("PALIOS_TRAINING_OUTPUT_DIR"),
                        help="Directory for phase JSONL outputs. Env: PALIOS_TRAINING_OUTPUT_DIR")
    parser.add_argument("--phase", action="append", choices=("cpt", "sft", "dpo"),
                        help="Phase to build. Repeatable. Default: all selected phases with required inputs.")
    parser.add_argument("--corpus-base", default=os.environ.get("PALIOS_CORPUS_BASE"),
                        help="Corpus root containing kernel/identity/layer_* tier dirs")
    parser.add_argument("--infra-cpt-jsonl", default=os.environ.get("PALIOS_INFRA_CPT_JSONL"),
                        help="Optional infra CPT JSONL")
    parser.add_argument("--sft-v2-dir", default=os.environ.get("PALIOS_SFT_V2_DIR"),
                        help="Optional topic-organized SFT v2 directory")
    parser.add_argument("--sft-clean-dir", default=os.environ.get("PALIOS_SFT_CLEAN_DIR"),
                        help="Optional platform-conversation SFT directory")
    parser.add_argument("--general-instruction-jsonl", default=os.environ.get("PALIOS_GENERAL_INSTRUCTION_JSONL"),
                        help="Optional general instruction JSONL")
    parser.add_argument("--dpo-jsonl", default=os.environ.get("PALIOS_DPO_JSONL"),
                        help="DPO preference-pair JSONL")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _tokenizer_path = _required_path(args.tokenizer, "--tokenizer or PALIOS_TOKENIZER_PATH")
    output_dir = _required_path(args.output_dir, "--output-dir or PALIOS_TRAINING_OUTPUT_DIR")
    phases = set(args.phase or ("cpt", "sft", "dpo"))

    if "cpt" in phases and not (args.corpus_base or args.infra_cpt_jsonl):
        raise RuntimeError("CPT phase requires --corpus-base and/or --infra-cpt-jsonl")
    if "sft" in phases and not (args.sft_v2_dir or args.sft_clean_dir or args.general_instruction_jsonl):
        raise RuntimeError("SFT phase requires --sft-v2-dir, --sft-clean-dir, and/or --general-instruction-jsonl")
    if "dpo" in phases and not args.dpo_jsonl:
        raise RuntimeError("DPO phase requires --dpo-jsonl")

    if "cpt" in phases:
        log.info("Building Phase 1 CPT (chunked + weighted)...")
        cpt_stats = build_phase1_cpt(
            os.path.join(output_dir, "phase1_cpt.jsonl"),
            corpus_base=args.corpus_base,
            infra_path=args.infra_cpt_jsonl,
        )

    if "sft" in phases:
        log.info("Building Phase 2 SFT (conversations + general instruction)...")
        sft_stats = build_phase2_sft(
            os.path.join(output_dir, "phase2_sft.jsonl"),
            sft_v2_dir=args.sft_v2_dir,
            sft_clean_dir=args.sft_clean_dir,
            general_path=args.general_instruction_jsonl,
        )

    if "dpo" in phases:
        log.info("Building Phase 3 DPO...")
        dpo_stats = build_phase3_dpo(os.path.join(output_dir, "phase3_dpo.jsonl"), args.dpo_jsonl)

    log.info("DONE. Files:")
    for f in ["phase1_cpt.jsonl", "phase2_sft.jsonl", "phase3_dpo.jsonl"]:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            lines = sum(1 for _ in open(path))
            size = os.path.getsize(path)
            log.info("  %s: %d items, %.1fMB", f, lines, size / 1e6)
