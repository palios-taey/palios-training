#!/bin/bash
# PRODUCTION SFT-ONLY: Config B experts-only ESFT, full gated SFT dataset, NO DPO
# Resolve sibling dirs (configs/, trainers/) relative to this script's location,
# so the recipe works regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${MODEL_PATH:?set MODEL_PATH to the base model path or Hugging Face id}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the training output directory}"

export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0:1,roceP2p1s0f0:1}"
export NCCL_IB_TC="${NCCL_IB_TC:-104}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export SFT_DIR="${SFT_DIR:-$REPO_ROOT/datasets/current/moe-35b}"
export SFT_GLOB="${SFT_GLOB:-combined_v1_gated.jsonl}"
export DPO_DIR="${DPO_DIR:-}"
export CPT_DATA="${CPT_DATA:-}"
export GENERAL_DIR="${GENERAL_DIR:-}"
export MAX_SEQ="${MAX_SEQ:-8192}"
export RESUME_DELTA="${RESUME_DELTA:-}"
export TOTAL_STEPS="${TOTAL_STEPS:-582}"
export SESSION_LIMIT="${SESSION_LIMIT:-350}"
export SAVE_EVERY="${SAVE_EVERY:-350}"
export FREEZE_CONFIG="${FREEZE_CONFIG:-B}"
export KEYSTONE_LAYERS="${KEYSTONE_LAYERS:-[8,9,11,15,21,23]}"
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-$SCRIPT_DIR/../configs/frozen_experts_v3.json}"
export LR_ESFT="${LR_ESFT:-2e-5}"
export LR_LORA="${LR_LORA:-0}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-25}"
# --- Multi-node addressing ---------------------------------------------------
# Operator-internal cluster IPs were REMOVED for public release. Set these to
# YOUR cluster: NODE0_IP..NODE3_IP (NODE0 = rank 0 / master), or set NODE_RANK
# per node. MASTER_ADDR defaults to NODE0_IP.
MASTER_ADDR="${MASTER_ADDR:-${NODE0_IP:?set NODE0_IP or MASTER_ADDR}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MY_IP=$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.' | head -n 1)
case "$MY_IP" in
    "${NODE0_IP}") RANK=0 ;;
    "${NODE1_IP}") RANK=1 ;;
    "${NODE2_IP}") RANK=2 ;;
    "${NODE3_IP}") RANK=3 ;;
    *) RANK="${NODE_RANK:?could not auto-detect rank; set NODE_RANK=0..3 per node, or NODE0_IP..NODE3_IP}" ;;
esac
echo "PRODUCTION SFT-ONLY: Config B, NO DPO"
echo "MODEL_PATH=$MODEL_PATH"
echo "SFT_DIR=$SFT_DIR SFT_GLOB=$SFT_GLOB"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT SAVE_EVERY=$SAVE_EVERY RESUME=$RESUME_DELTA"
echo "FREEZE_CONFIG=$FREEZE_CONFIG KEYSTONE_LAYERS=$KEYSTONE_LAYERS FROZEN_EXPERTS=$FROZEN_EXPERTS"
echo "LR_ESFT=$LR_ESFT LR_LORA=$LR_LORA LR_ROUTER=$LR_ROUTER MAX_SEQ=$MAX_SEQ"
accelerate launch \
    --config_file "$SCRIPT_DIR/../configs/fsdp_lora.yaml" \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    "$SCRIPT_DIR/../trainers/train_fsdp_v3.py" \
    "$@"
