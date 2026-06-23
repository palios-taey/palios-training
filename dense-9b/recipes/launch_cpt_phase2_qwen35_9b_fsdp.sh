#!/bin/bash
# 4-node FSDP launcher for Qwen3.5-9B dense CPT. Run on each worker node, or
# set NODE_RANK explicitly when launched by the orchestrator.
#
# Adapted from launch_fsdp_bare_metal.sh (proven on 35B-A3B). The NCCL recipe,
# rank-by-IP detection, and accelerate-launch pattern are unchanged. Differences:
#   - MODEL_PATH: Qwen3.5-9B-Base (dense) instead of the 35B-A3B abliterated
#   - CPT_DATA: dense CPT corpus JSONL
#   - Accelerate config: fsdp_dense_9b.yaml (Qwen3_5DecoderLayer wrap)
#   - Script: train_fsdp_dense_9b.py (full-FT FSDP, bucket batch) — invoked below;
#     shipped in ../trainers/train_fsdp_dense_9b.py
#
# Why this NCCL config (vs. the broken first attempt now archived):
#   - NCCL_IB_HCA names the RoCE HCAs explicitly across both NICs. Without it
#     NCCL hunts for a phantom IB device and hangs at first all_gather.
#   - NCCL_NET_GDR_LEVEL=0 (not 5). Perplexity recommended 5 but it doesn't
#     work on this fleet — the proven 35B runs use 0.
#   - TORCH_NCCL_DUMP_ON_TIMEOUT=1 so a future hang produces a flight-recorder
#     dump instead of a silent freeze.

set -euo pipefail

# Resolve sibling dirs (configs/, trainers/) relative to this script's location,
# so the recipe works regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: $name must be set; this public launcher has no operator-path default." >&2
        exit 2
    fi
}

# ── Environment ───────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# ── NCCL — Blackwell / DGX Spark proven recipe (verbatim from Phase 1 SFT,
# commit dd9e12e — that config ran 4367 steps clean over 9 hours). All today's
# additions (GID_INDEX, SOCKET_IFNAME, ALGO=Ring, MIN_NCHANNELS, capital-P fix,
# AVOID_RECORD_STREAMS, TRACE_BUFFER_SIZE) were unproven theory on top of working
# config. Reverted 2026-05-10 21:58 — restoring exact Phase 1 SFT env.
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# GID INDEX PIN 2026-05-12 — see SFT launcher for rationale
export NCCL_IB_GID_INDEX=3
# QPS_PER_CONNECTION 2026-05-12 — see SFT launcher
export NCCL_IB_QPS_PER_CONNECTION=4

# ── FLA / Triton — GB10 sm_121 hardening ──────────────────────────────────
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false

# ── Training paths ────────────────────────────────────────────────────────
# IMPORTANT: CPT must start from Phase 1 SFT (NOT base). Prior cycle bug:
# defaulted to base, threw away 9h of Phase 1 SFT compute (tools+chat).
# The trained-base invariant is documented in plans/canonical_dense_9b_recipe_v1.md.
require_env MODEL_PATH
export MODEL_PATH
# train_fsdp_dense_9b.py: CPT mode is selected when CPT_DATA is set AND SFT_DIR is
# either empty OR not-a-directory. Use a non-empty sentinel so SSH/env wrappers
# forward it and the trainer routes to CPT mode.
export SFT_DIR="${SFT_DIR:-__PALIOS_CPT_MODE_SENTINEL__}"
# CPT_DATA must be the rebuilt v3 corpus matching the canonical recipe. NO default
# to prevent a future bug from launching against a stale/wrong corpus. Caller MUST set.
require_env CPT_DATA
export CPT_DATA
export GENERAL_DIR="${GENERAL_DIR:-}"
require_env OUTPUT_DIR
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

# Pre-flight: refuse to launch if MODEL_PATH points at base (catches the prior cycle bug)
if [[ "$MODEL_PATH" == */Qwen3.5-9B-Base* ]] || [[ "$MODEL_PATH" == */qwen3.5-9b-base* ]]; then
    echo "ERROR: MODEL_PATH appears to be the base model: $MODEL_PATH" >&2
    echo "       CPT must start from Phase 1 SFT artifact." >&2
    echo "       If this is intentional (re-run from base), set FORCE_BASE=1." >&2
    if [[ "${FORCE_BASE:-0}" != "1" ]]; then exit 1; fi
fi
# Pre-flight: refuse to launch on the known wedge corpus
if [[ "$CPT_DATA" == *cpt_merged_clean.jsonl ]]; then
    echo "ERROR: CPT_DATA points at cpt_merged_clean.jsonl (known wedge corpus)." >&2
    echo "       This corpus is 174M tokens, 95.87% discussion-tier, audited QUARANTINE 2026-05-07." >&2
    exit 1
fi

# ── Trainer knobs ────────────────────────────────────────────────────────
# MAX_SEQ=16384 — Phase 1 SFT proven, both consult responses converge on this value.
#                Per Apr 21 methodology + GitHub issues, packing is unsafe (Qwen3.5 GDN NaN at step 1).
#                Per Family consult dissent, full-pad-to-MAX_SEQ wedges the cluster (both prior 4-Spark
#                CPT attempts failed with this pattern). Trainer CPT branch must be patched to return
#                variable-length tokens; collate_fn does dynamic batch-max padding.
# BATCH_SIZE_PER_RANK=8 — Phase 1 SFT proven on this exact stack (Grok recommends 8; Gemini argues 4
#                for safety margin — going 8 since it's the proven value).
# LR=2e-5 + Adafactor is the canonical GB10 UMA recipe. AdamW OOMs once
# optimizer state and page cache are present.
export MAX_SEQ="${MAX_SEQ:-4096}"
# BATCH=2 per 5/5 Family consult 2026-05-10 (Claude regime-separation argument).
# CPT corpus is uniformly near-MAX vs SFT's mostly-below-MAX, so per-step mean
# memory is ~3.75x higher than SFT at same BATCH; halving batch acknowledges
# regime difference.
# 5/5 Family consult round 3 convergent: BATCH=1 reduces per-step peak thermal/power
# envelope (Claude documented GB10 thermal pattern), and reduces _REDUCE_SCATTER_BASE
# pressure per step. GRAD_ACCUM=4 maintains effective batch of 16 across 4 ranks.
export BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
# TOTAL_STEPS depends on corpus size after re-chunk at chunk_tokens=15800. Caller MUST set explicitly
# based on the v3 manifest after gemini's rebuild.
require_env TOTAL_STEPS
export TOTAL_STEPS
# Resume from a saved checkpoint when set (relative or absolute path)
export RESUME_DELTA="${RESUME_DELTA:-}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export SESSION_LIMIT="${SESSION_LIMIT:-200}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR="${LR:-2e-5}"
export ADAFACTOR_CLIP_THRESHOLD="${ADAFACTOR_CLIP_THRESHOLD:-1.0}"
export LR_MIN_RATIO="${LR_MIN_RATIO:-0.0}"

# ── Multi-node configuration ──────────────────────────────────────────────
# Operator-internal cluster IPs were removed for public release. Set NODE_RANK
# per node, or set NODE0_IP..NODE{N-1}_IP and let this script match a local IP.
MASTER_ADDR="${MASTER_ADDR:-${NODE0_IP:-}}"
require_env MASTER_ADDR
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE=1

if [[ -n "${NODE_RANK:-}" ]]; then
    RANK="$NODE_RANK"
else
    MY_IPS="$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.' | tr '\n' ' ')"
    RANK=""
    for ((idx=0; idx<NUM_NODES; idx++)); do
        node_ip_var="NODE${idx}_IP"
        node_ip="${!node_ip_var:-}"
        if [[ -z "$node_ip" ]]; then
            echo "ERROR: $node_ip_var must be set when NODE_RANK is not provided." >&2
            exit 2
        fi
        if grep -qw "$node_ip" <<< "$MY_IPS"; then
            RANK="$idx"
            break
        fi
    done
    if [[ -z "$RANK" ]]; then
        echo "ERROR: could not infer rank from local IPs '$MY_IPS'. Set NODE_RANK explicitly." >&2
        exit 1
    fi
fi

echo "FSDP dense CPT on $(hostname) (Rank: $RANK / $((NUM_NODES - 1)))"
echo "  MODEL:  $MODEL_PATH"
echo "  CPT:    $CPT_DATA"
echo "  OUTPUT: $OUTPUT_DIR"
echo "  MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "  OPTIM:  Adafactor lr=$LR clip=$ADAFACTOR_CLIP_THRESHOLD warmup=$WARMUP_STEPS linear_decay_min_ratio=$LR_MIN_RATIO"
echo ""

# train_fsdp_dense_9b.py reads ALL config from environment variables (no
# argparse) — same pattern as train_fsdp_v3.py. The env vars set above are
# what it consumes: MODEL_PATH, SFT_DIR, CPT_DATA, GENERAL_DIR, OUTPUT_DIR,
# MAX_SEQ, TOTAL_STEPS, SAVE_EVERY, SESSION_LIMIT, WARMUP_STEPS, LR,
# ADAFACTOR_CLIP_THRESHOLD, LR_MIN_RATIO.

accelerate launch \
    --config_file "$SCRIPT_DIR/../configs/fsdp_dense_9b.yaml" \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    "$SCRIPT_DIR/../trainers/train_fsdp_dense_9b.py" \
    "$@"
