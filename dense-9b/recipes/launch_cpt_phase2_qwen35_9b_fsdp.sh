#!/bin/bash
# 4-node FSDP launcher for Qwen3.5-9B dense CPT. Run on EACH Spark; the
# script detects its own fabric IP and assigns rank from it.
#
# Adapted from launch_fsdp_bare_metal.sh (proven on 35B-A3B). The NCCL recipe,
# rank-by-IP detection, and accelerate-launch pattern are unchanged. Differences:
#   - MODEL_PATH: clean Qwen3.5-9B text-derived base
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

set -eo pipefail

# Resolve sibling dirs (configs/, trainers/) relative to this script's location,
# so the recipe works regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"

# ── NCCL — Blackwell / DGX Spark proven recipe (verbatim from Phase 1 SFT,
# commit dd9e12e — that config ran 4367 steps clean over 9 hours). All today's
# additions (GID_INDEX, SOCKET_IFNAME, ALGO=Ring, MIN_NCHANNELS, capital-P fix,
# AVOID_RECORD_STREAMS, TRACE_BUFFER_SIZE) were unproven theory on top of working
# config. Reverted 2026-05-10 21:58 — restoring exact Phase 1 SFT env.
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
# RCA fix-test 2026-06-28: NCCL 2.27+ defaults NCCL_NET_GDR_C2C=1, which can override GDR_LEVEL=0 on
# C2C-attached NICs (our GB10 topology) — i.e. "GDR off" was likely FALSE. Explicitly force it off.
export NCCL_NET_GDR_C2C="${NCCL_NET_GDR_C2C:-0}"
export NCCL_NET_GDR_READ="${NCCL_NET_GDR_READ:-0}"
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# GID INDEX PIN 2026-05-12 — see SFT launcher for rationale
export NCCL_IB_GID_INDEX=3
# QPS_PER_CONNECTION 2026-05-12 — see SFT launcher
# RCA fix-test 2026-06-28: 4 QPs/conn × 3 peers × 2 rails = 24 concurrent RC QPs; under the first-collective
# burst all retry-storm at once. Drop to 1 to serialize DMA (5/5 panel). Overridable.
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"

# ── FLA / Triton — GB10 sm_121 hardening ──────────────────────────────────
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
# RCA fix-test 2026-06-28: garbage_collection_threshold:0.8 ties to the 81%-UMA flatline at the wedge
# (Gemini Deep Think: 0.8 GC churn -> SMMUv3 TLB-shootdown storm -> PCIe CTO -> SError/GIC lock). Made
# env-overridable so we can A/B drop it without losing the recipe default.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export TOKENIZERS_PARALLELISM=false

# ── Training paths ────────────────────────────────────────────────────────
# IMPORTANT: this restart does CPT from the clean text-derived base, not from
# the raw multimodal base and not from a stale SFT checkpoint lineage.
export MODEL_PATH="${MODEL_PATH:-/home/spark/models/Qwen3.5-9B-Base-Text-Derived}"
# train_fsdp_dense_9b.py: CPT mode is selected when CPT_DATA is set AND SFT_DIR is
# either empty OR not-a-directory. Orchestrator forwards env vars only when non-empty,
# AND the trainer defaults SFT_DIR to /var/spark/isma/training/sft (a real dir on the
# Sparks) when not set. So passing SFT_DIR="" gets dropped by the orchestrator and
# the trainer then thinks SFT mode is desired. Use an explicit sentinel (non-empty,
# clearly non-dir) so the orchestrator forwards it and the trainer routes to CPT mode.
export SFT_DIR="/nonexistent/cpt_mode_sentinel"
# CPT_DATA must be the rebuilt dense v3 corpus matching the canonical recipe.
export CPT_DATA="${CPT_DATA:-/var/spark/isma/training/cpt_v3_dense_9b.jsonl}"
export GENERAL_DIR="${GENERAL_DIR:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/cpt_v3_dense_9b}"
mkdir -p "$OUTPUT_DIR"

# Pre-flight: refuse the actual raw multimodal base while allowing the clean
# text-derived base path, whose name intentionally contains Qwen3.5-9B-Base.
MODEL_PATH_STRIPPED="${MODEL_PATH%/}"
if [[ "$MODEL_PATH_STRIPPED" == "/home/spark/models/Qwen3.5-9B-Base" ]] || [[ "$MODEL_PATH" == *ForConditionalGeneration* ]]; then
    echo "ERROR: MODEL_PATH points at the raw multimodal base: $MODEL_PATH" >&2
    echo "       CPT must start from /home/spark/models/Qwen3.5-9B-Base-Text-Derived." >&2
    echo "       Do not bypass this guard for the raw base or ForConditionalGeneration exports." >&2
    exit 1
fi
# Pre-flight: refuse to launch on known stale/wedge corpora.
case "$CPT_DATA" in
    *cpt_merged_clean.jsonl|*cpt_v3_v2_dense_9b.jsonl|*cpt_v3_v3_dense_9b.jsonl|*cpt_v3_v4_dense_9b.jsonl|*cpt_v3_v4_sorted_dense_9b.jsonl)
        echo "ERROR: CPT_DATA points at a stale or quarantined corpus: $CPT_DATA" >&2
        echo "       Use /var/spark/isma/training/cpt_v3_dense_9b.jsonl for this restart." >&2
        exit 1
        ;;
esac

# ── Trainer knobs (defaults from 2026-05-08 Family consult: Gemini + Grok converge) ─────────
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
export TOTAL_STEPS="${TOTAL_STEPS:?ERROR: TOTAL_STEPS must be set; depends on cpt_v3_dense_9b corpus row count}"
# Resume from a saved checkpoint when set (relative or absolute path)
export RESUME_DELTA="${RESUME_DELTA:-}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export SESSION_LIMIT="${SESSION_LIMIT:-200}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR="${LR:-2e-5}"
export ADAFACTOR_CLIP_THRESHOLD="${ADAFACTOR_CLIP_THRESHOLD:-1.0}"

# ── Multi-node configuration ──────────────────────────────────────────────
# --- Multi-node addressing ---------------------------------------------------
# Operator-internal cluster IPs were REMOVED for public release. Set these to
# YOUR cluster: NODE0_IP..NODE3_IP (NODE0 = rank 0 / master), or set NODE_RANK
# per node. MASTER_ADDR defaults to NODE0_IP.
MASTER_ADDR="${MASTER_ADDR:-${NODE0_IP}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE=1

# Detect rank from local fabric IP. Mapping is fixed by the cluster wiring:
#   ${NODE0_IP} = Spark 1 = rank 0  (master)
#   ${NODE1_IP} = Spark 2 = rank 1
#   ${NODE2_IP} = Spark 3 = rank 2
#   ${NODE3_IP} = Spark 4 = rank 3
MY_IP=$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.' | head -n 1)

case "$MY_IP" in
    "${NODE0_IP}") RANK=0 ;;
    "${NODE1_IP}") RANK=1 ;;
    "${NODE2_IP}") RANK=2 ;;
    "${NODE3_IP}") RANK=3 ;;
    *)
        echo "ERROR: Unknown fabric IP '$MY_IP' on $(hostname). Set NODE0_IP..NODE3_IP (or NODE_RANK per node)." >&2
        exit 1
        ;;
esac

echo "FSDP dense CPT on $(hostname) (IP: $MY_IP, Rank: $RANK / $((NUM_NODES - 1)))"
echo "  MODEL:  $MODEL_PATH"
echo "  CPT:    $CPT_DATA"
echo "  OUTPUT: $OUTPUT_DIR"
echo "  MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "  OPTIM:  Adafactor lr=$LR clip=$ADAFACTOR_CLIP_THRESHOLD"
echo ""

# train_fsdp_dense_9b.py reads ALL config from environment variables (no
# argparse) — same pattern as train_fsdp_v3.py. The env vars set above are
# what it consumes: MODEL_PATH, SFT_DIR, CPT_DATA, GENERAL_DIR, OUTPUT_DIR,
# MAX_SEQ, TOTAL_STEPS, SAVE_EVERY, SESSION_LIMIT, WARMUP_STEPS, LR_LORA,
# LR_ROUTER, LR_ESFT, FREEZE_CONFIG.

accelerate launch \
    --config_file "$SCRIPT_DIR/../configs/fsdp_dense_9b.yaml" \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    "$SCRIPT_DIR/../trainers/train_fsdp_dense_9b.py" \
    "$@"
