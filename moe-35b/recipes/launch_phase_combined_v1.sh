#!/usr/bin/env bash
# PHASE COMBINED V1: mixed identity + infra SFT from the abliterated base.
# De-umbilicalized from the production Spark launcher: all operator paths and
# fabric addresses are provided by environment variables, and repo assets are
# referenced relative to this recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${MODEL_PATH:?set MODEL_PATH to the base model path or Hugging Face id}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the training output directory}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 fabric address}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# NCCL (documented production defaults, override per cluster as needed)
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0:1,rocep2s0f0:1}"
export NCCL_IB_TC="${NCCL_IB_TC:-104}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"

# Triton/FLA
export FLA_USE_TMA="${FLA_USE_TMA:-0}"
export TRITON_AUTOTUNE_DISABLE="${TRITON_AUTOTUNE_DISABLE:-1}"
export FLA_DISABLE_CAUSAL_CONV1D="${FLA_DISABLE_CAUSAL_CONV1D:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# Model + Data
export SFT_DIR="${SFT_DIR:-$REPO_ROOT/datasets/current/moe-35b}"
export SFT_GLOB="${SFT_GLOB:-combined_v1_gated.jsonl}"
export DPO_DIR="${DPO_DIR:-}"
export CPT_DATA="${CPT_DATA:-}"
export GENERAL_DIR="${GENERAL_DIR:-}"
export MAX_SEQ="${MAX_SEQ:-8192}"

# Fresh start from abliterated base unless explicitly resumed.
export RESUME_DELTA="${RESUME_DELTA:-}"

# CORPUS V2: 1 epoch = 582 steps (2,325 packed / 4 nodes). Single session.
export TOTAL_STEPS="${TOTAL_STEPS:-582}"
export SESSION_LIMIT="${SESSION_LIMIT:-700}"
export SAVE_EVERY="${SAVE_EVERY:-700}"

# Config B: experts-only ESFT (same as Phase 1 v3)
export FREEZE_CONFIG="${FREEZE_CONFIG:-B}"
export KEYSTONE_LAYERS="${KEYSTONE_LAYERS:-[8, 9, 11, 15, 21, 23]}"
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-$SCRIPT_DIR/../configs/frozen_experts_v3.json}"
export LR_ESFT="${LR_ESFT:-2e-5}"
export LR_LORA="${LR_LORA:-3e-4}"
export LR_ROUTER="${LR_ROUTER:-3e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-25}"

# FSDP network. Set NODE_RANK directly, or provide NODE_IPS plus FABRIC_SUBNET
# so each node can map its local fabric IP to rank without hardcoded addresses.
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
if [[ -n "${NODE_RANK:-}" ]]; then
    RANK="$NODE_RANK"
else
    : "${FABRIC_SUBNET:?set FABRIC_SUBNET prefix or NODE_RANK}"
    : "${NODE_IPS:?set NODE_IPS comma-separated rank0,rank1,... or NODE_RANK}"
    MY_IP="$(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | while read -r ip_addr; do
        case "$ip_addr" in
            "$FABRIC_SUBNET"*) printf '%s\n' "$ip_addr"; break ;;
        esac
    done)"
    if [[ -z "$MY_IP" ]]; then
        echo "ERROR: no local IPv4 address matched FABRIC_SUBNET='$FABRIC_SUBNET'; set NODE_RANK explicitly" >&2
        exit 1
    fi
    IFS=',' read -r -a NODE_IP_ARRAY <<< "$NODE_IPS"
    RANK=""
    for idx in "${!NODE_IP_ARRAY[@]}"; do
        node_ip="${NODE_IP_ARRAY[$idx]//[[:space:]]/}"
        if [[ "$MY_IP" == "$node_ip" ]]; then
            RANK="$idx"
            break
        fi
    done
    if [[ -z "$RANK" ]]; then
        echo "ERROR: local fabric IP '$MY_IP' not found in NODE_IPS; set NODE_RANK explicitly" >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR"

echo "PHASE COMBINED V1 — Mixed identity + infra (one phase)"
echo "  Base: $MODEL_PATH"
echo "  SFT_DIR: $SFT_DIR"
echo "  SFT_GLOB: $SFT_GLOB"
echo "  Output: $OUTPUT_DIR"
echo "  Keystones: $KEYSTONE_LAYERS"
echo "  Freeze config: $FREEZE_CONFIG"
echo "  TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT SAVE_EVERY=$SAVE_EVERY"
echo "  Rank: $RANK / NUM_NODES=$NUM_NODES MASTER=$MASTER_ADDR:$MASTER_PORT"

accelerate launch \
    --config_file ../configs/fsdp_lora.yaml \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --rdzv_conf 'timeout=3600' \
    ../trainers/train_fsdp_v3.py \
    "$@"
