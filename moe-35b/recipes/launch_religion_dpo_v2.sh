#!/bin/bash
# RELIGION DPO V2 — Config A2 (keystone-only attention LoRA) per infra + ChatGPT + data
#
# DIAGNOSTIC (from length_mechanics_v1 control, full audit 2026-04-20 02:55):
# Config A's attention LoRA on all 40 layers is the content-agnostic leak into
# infra_cross_system. Religion DPO (1/4) and length DPO (2/4 on same probes) both broke it,
# while individual-system knowledge categories (hardware_knowledge 2/2, bridge_infra_soul 1/1,
# math_stem_control 6/6) held in both runs. Cross-system reasoning is the attention-LoRA path.
#
# CONSULT (5 Chat platforms, 2026-04-20 02:45): diverse proposals. Infra recommends Option A
# (keystone-only attention LoRA) based on the empirical diagnostic. ChatGPT ranks A first,
# estimating keystone-only attention retains 45-75% of policy movement capacity (+6 to +12pp
# religion_honest). Gemini/Perplexity lean broader Option E (also restrict shared_expert);
# Claude proposes orthogonal Option E (o_proj-only, all 40 layers). No empirical evidence
# yet that shared_expert or q/k projections specifically are additional leak sources.
#
# FIX: FREEZE_CONFIG=A2 (Option A) — restricts attention LoRA to keystones [8,9,11,15,21,23].
# Minimum-intervention one-variable change from religion_dpo_v1. shared_expert LoRA stays on
# all 40 layers. Test: does restricting attention LoRA alone fix infra_cross_system AND
# preserve religion_honest gain?
#
# Data: SAME 50 religion DPO pairs + precomputed ref_logprobs as religion_dpo_v1.
# Controlled comparison: only the freeze config differs.
#
# NEXT-IF-FAILS:
# - If religion_honest still moves but infra_cross_system still regresses → Claude's o_proj
#   hypothesis is next (A2o: o_proj-only all 40 layers); or A3 (Gemini's broader keystone).
# - If religion_honest drops below +6pp → 6 keystones too narrow for policy. Try A3 variant
#   keeping shared_expert all layers but adding more keystones, or fall back to B.
#
# Base: combined_v1/final (82.8% baseline, step 582)
# Bake: /home/user/bake_config_a_v2.py (infra owns)

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
export RESUME_DELTA="${RESUME_DELTA:-/home/user/training_outputs/phase_combined_v1/final}"

# SAME data as religion_dpo_v1 — isolates Config A → A2 as the only variable
export DPO_DATA="${DPO_DATA:-/home/user/training_data/religion_run_v1/religion_v3_dpo_pairs_with_ref.jsonl}"

export SFT_DIR=""
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-4096}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/user/training_outputs/religion_dpo_v2}"

# IDENTICAL hyperparams to religion_dpo_v1 (only freeze config differs)
export BETA="${BETA:-0.05}"
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export TOTAL_STEPS="${TOTAL_STEPS:-642}"
export SESSION_LIMIT="${SESSION_LIMIT:-900}"
export SAVE_EVERY="${SAVE_EVERY:-60}"

# THE ONE CHANGE: Config A2 restricts attention LoRA to keystone layers only (shared_expert LoRA still on all 40 layers)
export FREEZE_CONFIG="A2"
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/user/training_data/phase1_constitutional/frozen_experts_v4_1_polysemantic.json}"

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

echo "RELIGION DPO V2 — Config A2 (keystone-only attention LoRA)"
echo "  Resume: $RESUME_DELTA (step 582, 82.8% combined_v1 baseline)"
echo "  Data: $DPO_DATA (50 religion DPO pairs — SAME as religion_dpo_v1)"
echo "  Freeze: A2 — attention LoRA RESTRICTED to keystones $KEYSTONE_LAYERS only"
echo "           shared_expert LoRA still on all 40 layers (policy path preserved)"
echo "  Mask: $(basename $FROZEN_EXPERTS) (v4.1, 159 frozen experts)"
echo "  MAX_SEQ=$MAX_SEQ  BETA=$BETA"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA router=FROZEN  warmup=$WARMUP_STEPS"
echo "  TOTAL_STEPS=$TOTAL_STEPS (60 new on top of 582) SAVE_EVERY=$SAVE_EVERY"
echo "  Output: $OUTPUT_DIR"

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
