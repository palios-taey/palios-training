#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
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

# ── NCCL / env — VALIDATED fleet block, copied VERBATIM from
# careers-qwen/launch_4node.sh:6-26 (NET_PLUGIN=none proven 21.7GB/s; infra
# confirms == nccl_rescue_bank). Replaces the prior hand-tuned NCCL/FLA/alloc
# block whose deltas (heartbeat 1800 vs 120, expandable_segments True vs False,
# missing SOCKET_IFNAME, invented GDR_C2C/READ/PXN_C2C) were the wedge mechanism.
# ZERO substitutions — see dense-9b/plans/build_launcher_spec.md §1.
NCCL_LIB=${SPARK_HOME}/.local/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB:${LD_LIBRARY_PATH:-}
# --- validated NCCL block for 4× GB10 dual-rail RoCE CPU-proxy ---
export NCCL_NET_PLUGIN=none          # CRITICAL: AWS-OFI plugin fails on GB10 (proven)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_MERGE_NICS=1
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=8388608
export NCCL_TIMEOUT=1800
# heartbeat watchdog: without this a hung collective NEVER times out → blocked CUDA stream
# + memory pressure → driver OOM → kernel panic → power-cycle. With it: clean traceback+abort.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export HF_HUB_DISABLE_XET=1 HF_HOME=${SPARK_HOME}/hf_cache TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # Chat-converged: expandable_segments stresses the GB10 driver VA path under multi-node pinned-buffer load

# ── Training paths ────────────────────────────────────────────────────────
# IMPORTANT: this restart does CPT from the clean text-derived base, not from
# the raw multimodal base and not from a stale SFT checkpoint lineage.
export MODEL_PATH="${MODEL_PATH:-${SPARK_HOME}/models/Qwen3.5-9B-Base-Text-Derived}"
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
export OUTPUT_DIR="${OUTPUT_DIR:-${SPARK_HOME}/training_outputs/cpt_v3_dense_9b}"
mkdir -p "$OUTPUT_DIR"

# Pre-flight: refuse the actual raw multimodal base while allowing the clean
# text-derived base path, whose name intentionally contains Qwen3.5-9B-Base.
MODEL_PATH_STRIPPED="${MODEL_PATH%/}"
if [[ "$MODEL_PATH_STRIPPED" == "${SPARK_HOME}/models/Qwen3.5-9B-Base" ]] || [[ "$MODEL_PATH" == *ForConditionalGeneration* ]]; then
    echo "ERROR: MODEL_PATH points at the raw multimodal base: $MODEL_PATH" >&2
    echo "       CPT must start from ${SPARK_HOME}/models/Qwen3.5-9B-Base-Text-Derived." >&2
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
export MAX_SEQ="${MAX_SEQ:-16384}"
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

# ── Multi-node configuration — EXPLICIT POSITIONAL RANK + rail master ───────
# Per build_launcher_spec.md §2/§3: rank is the positional arg $1 (exactly like
# careers-qwen/launch_4node.sh:5 `RANK=$1`), NOT IP-detection. The prior recipe's
# `ip addr | head -1` returned the MANAGEMENT IP on these nodes (wrong network) —
# that addressing bug is removed. Master is the validated RAIL address
# ${SPARK_RAIL_MASTER} (Spark 1 = rank 0) — the accelerate form of launch_4node.sh's
# `--master_addr=${SPARK_RAIL_MASTER}`.
# Run once per node with its explicit rank: Spark1=0 Spark2=1 Spark3=2 Spark4=3.
RANK="${1:?ERROR: pass explicit node rank as arg 1 (Spark1=0 Spark2=1 Spark3=2 Spark4=3), like launch_4node.sh}"; shift
MASTER_ADDR="${MASTER_ADDR:-${SPARK_RAIL_MASTER}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE=1

echo "FSDP dense CPT on $(hostname) (Rank: $RANK / $((NUM_NODES - 1)))"
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
    --config_file "${ACCEL_CONFIG:-$SCRIPT_DIR/../configs/fsdp_dense_9b.yaml}" \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    "$SCRIPT_DIR/../trainers/train_fsdp_dense_9b.py" \
    "$@"
