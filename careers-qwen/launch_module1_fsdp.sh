#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Per-node 4-node FSDP2 LoRA-SFT launcher for Module-1 (careers 27B revenue adapter).
# Run on EACH Spark:   ./launch_module1_fsdp.sh <NODE_RANK 0-3>
# rank 0 = master (${SPARK_RAIL_MASTER}) — start it FIRST so it binds the rendezvous before workers dial in.
#
# THERMAL: the graphics-clock cap (nvidia-smi -lgc 0,2000) that holds the boards below the ~94C
# hard-crash is applied by the Mira-side orchestrator (run_4node_27b_cpt.sh pattern), NOT here — it
# needs sudo and must run once per node per boot. Apply it before launching this on all 4 nodes.
set -uo pipefail
RANK=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) proven GB10 NCCL fabric + CUDA-13 toolchain block (sourced, not executed)
source "$SCRIPT_DIR/fabric_env.sh"

# 2) accelerate FSDP2 env (mirrors the 27B CPT launcher's FSDP_* block).
export ACCELERATE_USE_FSDP=true
export FSDP_VERSION=2
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_TRANSFORMER_CLS_TO_WRAP=Qwen3_5DecoderLayer
export FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT
# NOTE (DEVIATION — see report): the DESIGN text said FSDP_CPU_RAM_EFFICIENT_LOADING=true, but the
# proven 27B CPT launcher exports it FALSE on this exact stack (accelerate 1.13.0) because FSDP2 has
# NO sync_module_states — the rank0-real/workers-meta trick leaves top-level params on meta and
# fully_shard's _validate_no_meta_params raises. This trainer loads real weights on ALL ranks (like
# the reference), so it MUST be false. Flip to true only if the reviewer confirms the FSDP2 meta path.
export FSDP_CPU_RAM_EFFICIENT_LOADING=false
export FSDP_RESHARD_AFTER_FORWARD=true
# accelerate FSDP2 activation checkpointing — the HF gradient_checkpointing_enable() call is a no-op
# under FSDP2 (agent flag 5), so enable AC here. 27B forward @ seq-4096 is unchanged by LoRA; AC bounds
# the activation-memory peak (the CPT run used it). Disable (false) only if V4 shows ample headroom.
export FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"

# 3) rendezvous (copied from the CPT launcher — do NOT invent these)
MASTER_ADDR=${SPARK_RAIL_MASTER}
MASTER_PORT=29500

echo "FSDP2 LoRA-SFT Module-1 on $(hostname) (rank $RANK / 3)  master=$MASTER_ADDR:$MASTER_PORT"

python3 -m torch.distributed.run \
    --nnodes=4 --nproc_per_node=1 --node_rank="$RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train_lora_sft.py" \
    --model ${SPARK_HOME}/models/Qwen3.6-27B \
    --data ${SPARK_HOME}/module1_train.jsonl \
    --out ${SPARK_HOME}/module1_lora_out \
    --max-seq 4096 \
    --rank 16 --alpha 32 --dropout 0.05 \
    --lr 1e-4 --epochs 2 --warmup 50 --grad-accum 8 \
    --lane-weights "stage2_scorer=0.45,jesse_voice=0.35,repo_capability=0.12,values=0.08" \
    --tiny-lane-cap 3 \
    --decoder-layer-cls Qwen3_5DecoderLayer \
    --save-every "${SAVE_EVERY:-99999}" \
    --session-seconds "${SESSION_SECONDS:-5400}" \
    ${EXTRA_ARGS:-}
