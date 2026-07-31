#!/bin/bash
# COMBINED V1 TAIL — refinement on combined_v1's 27 audit corrections
# Pattern: resume combined_v1/final (step 582) + SFT ~20 new steps on the 27 BETRAYED-category corrections
# Hypothesis: targeted tail on healthy substrate (82.8%) can nudge weak cats without full retrain.
# ~15-25 min train time (small data, few steps).

# Resolve sibling dirs (configs/, trainers/) relative to this script's location,
# so the recipe works regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"

export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0

export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

export MODEL_PATH="/home/user/models/Huihui-Qwen3.5-35B-A3B-abliterated"
export RESUME_DELTA="${RESUME_DELTA:-/home/user/training_outputs/phase_combined_v1/final}"
export SFT_DIR="${SFT_DIR:-/home/user/training_data/combined_v1_tail}"
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-8192}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/user/training_outputs/phase_combined_v1_tail}"

# Hyperparams — lower LR for refinement (don't smash the healthy substrate)
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
# combined_v1 final is step 582; add ~30 tail steps (27 pairs × ~1 epoch at batch 1)
export TOTAL_STEPS="${TOTAL_STEPS:-612}"
export SESSION_LIMIT="${SESSION_LIMIT:-1200}"
export SAVE_EVERY="${SAVE_EVERY:-30}"

# Config B (experts-only ESFT) — identical to combined_v1
export FREEZE_CONFIG="B"
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/user/training_data/phase1_constitutional/frozen_experts_v3.json}"

# --- Multi-node addressing ---------------------------------------------------
# Operator-internal cluster IPs were REMOVED for public release. Set these to
# YOUR cluster: NODE0_IP..NODE3_IP (NODE0 = rank 0 / master), or set NODE_RANK
# per node. MASTER_ADDR defaults to NODE0_IP.
MASTER_ADDR="${NODE0_IP}"
MASTER_PORT="29500"
NUM_NODES=4
GPUS_PER_NODE=1

MY_IP=$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.' | head -n 1)
case "$MY_IP" in
    "${NODE0_IP}") RANK=0 ;;
    "${NODE1_IP}") RANK=1 ;;
    "${NODE2_IP}") RANK=2 ;;
    "${NODE3_IP}") RANK=3 ;;
    *) RANK="${NODE_RANK:?could not auto-detect rank; set NODE_RANK=0..3 per node, or NODE0_IP..NODE3_IP}" ;;
esac

echo "COMBINED V1 TAIL — refinement on 27 corrections"
echo "  Resume: $RESUME_DELTA (step 582)"
echo "  Data: $SFT_DIR (27 items)"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA (lower than v1 for gentle tuning)"
echo "  TOTAL_STEPS=$TOTAL_STEPS (30 new on top of 582)"

accelerate launch \
    --config_file "$SCRIPT_DIR/../configs/fsdp_lora.yaml" \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf 'timeout=3600' \
    "$SCRIPT_DIR/../trainers/train_fsdp_v3.py" \
    "$@"
