#!/bin/bash
# STANDARD DPO VANILLA TEST — first-principles sanity check on abliterated base
#
# Question this answers: does textbook DPO work on Qwen3.5-35B-A3B MoE at all?
# If this run cleanly shifts behavior in the preference direction without damage,
# our custom freeze/mask infrastructure is over-engineered. If this ALSO produces
# Taia/Gaia/Charter-style identity damage, the failure mode is architectural.
#
# Deliberate design choices (vs our prior runs):
# - RESUME_DELTA=""                → fresh start from abliterated (no combined_v1 overlay)
# - FREEZE_CONFIG=VANILLA          → pure LoRA-only (PEFT default, no custom masks)
# - No FROZEN_EXPERTS              → no expert gradient masking
# - No keystone unfreezing         → _is_trainable never reached
# - BETA=0.1                       → textbook DPO (our custom runs use 0.05)
# - LR_LORA=5e-7                   → textbook DPO (our runs use 3e-7)
# - TOTAL_STEPS=200                → textbook DPO budget
# - Data: length-preference pairs (50) with ref_logprobs vs abliterated (not combined_v1)
#
# Data choice rationale: length pairs are content-neutral and were already authored. The test
# is about mechanics, not domain. If abliterated + vanilla DPO + length preference shifts
# output length, DPO works textbook-correct here. If it doesn't, something architectural.

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

export DPO_SKIP_POSTFSDP_DIAG=1

export MODEL_PATH="/home/user/models/Huihui-Qwen3.5-35B-A3B-abliterated"
# Default: no resume (fresh abliterated start). Override by passing RESUME_DELTA in env
# for resume-from-checkpoint (multi-session runs where we reboot between halves).
export RESUME_DELTA="${RESUME_DELTA-}"

# Length pairs with ref_logprobs precomputed against ABLITERATED (not combined_v1)
export DPO_DATA="${DPO_DATA:-/home/user/training_data/length_mechanics_v1/length_pairs_with_ref_abliterated.jsonl}"

export SFT_DIR=""
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-4096}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/user/training_outputs/standard_dpo_vanilla}"

# TEXTBOOK DPO hyperparams (not our custom reduced-LR)
export BETA="${BETA:-0.1}"
export LR_ESFT="${LR_ESFT:-0}"       # No expert training in vanilla — LoRA only
export LR_LORA="${LR_LORA:-5e-7}"    # Standard DPO LR for LoRA
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"  # Standard warmup for 200 steps
export TOTAL_STEPS="${TOTAL_STEPS:-200}"   # Textbook DPO budget
# Split run into 100-step halves with fleet reboot between sessions. Observed pattern:
# Sparks accumulate UMA/NCCL state over ~90-120 min runs, one rank silently hangs in allreduce,
# watchdog kills after 60min timeout (saw this at step 168 on prior vanilla attempt). Split prevents that.
export SESSION_LIMIT="${SESSION_LIMIT:-100}"   # Exit cleanly after 100 new steps per session
export SAVE_EVERY="${SAVE_EVERY:-100}"         # Save at step 100 of each session

# VANILLA freeze: PEFT default, no custom masks
export FREEZE_CONFIG="VANILLA"
# KEYSTONE_LAYERS still required by script but VANILLA branch doesn't use it
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
# NO FROZEN_EXPERTS — expert gradient masking skipped
unset FROZEN_EXPERTS

# Keep watchdogs on for safety
export DPO_ABORT_RATIO_MAX="${DPO_ABORT_RATIO_MAX:-10.0}"
export DPO_ABORT_EXPERT_DRIFT="${DPO_ABORT_EXPERT_DRIFT:-0.05}"

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

echo "STANDARD DPO VANILLA TEST — first-principles sanity on abliterated"
echo "  Base: abliterated (no resume, fresh start)"
echo "  Config: VANILLA (pure LoRA, no custom masks)"
echo "  Data: $DPO_DATA (50 length pairs, ref vs abliterated)"
echo "  Hyperparams: BETA=$BETA LR_LORA=$LR_LORA WARMUP=$WARMUP_STEPS STEPS=$TOTAL_STEPS"
echo "  Output: $OUTPUT_DIR"
echo "  Expected: response length shifts toward chosen (brief) direction. Clean, no damage."

accelerate launch \
    --config_file "$SCRIPT_DIR/../configs/fsdp_lora.yaml" \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf 'timeout=3600' \
    "$SCRIPT_DIR/../trainers/train_dpo_v2.py" \
    "$@"
