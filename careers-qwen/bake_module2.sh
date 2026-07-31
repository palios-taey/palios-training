#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# bake_module2.sh — bake module-2 LoRA checkpoint-225 to a servable PEFT adapter.
# Path mirrors module-1 EXACTLY: run_4node_27b_cpt.sh (the only launcher that forwards BOTH
# LORA_MODE/LORA_R/LORA_ALPHA/SFT_DIR and BAKE_TO_HF) → trainer BAKE_TO_HF branch → 4-rank
# FSDP2 build + dcp.load → rank0 PEFT save_pretrained. bake_27b.sh is NOT usable here: its
# RUN_ENV omits the LoRA vars, so the PEFT wrapper would not exist when dcp.load runs.
# PRE-CONDITION per doctrine: reboot all 4 first (this script does it).
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
CKPT=${SPARK_HOME}/training_outputs/module2_cumulative_lora/checkpoint-225
OUT=${SPARK_HOME}/models/module2_adapter_hf
RECIPES=${REPO_ROOT}/dense-9b/recipes
LOG=${MIRA_HOME}/.taey-babysit/module2_bake.log
say(){ echo "[bake $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "verifying source checkpoint on all 4 nodes"
for n in "${NODES[@]}"; do
  r=$(timeout 10 ssh -o ConnectTimeout=6 spark@"$n" "test -f $CKPT/COMPLETE && ls $CKPT/dcp | wc -l" 2>/dev/null)
  [ -n "$r" ] && [ "$r" -gt 0 ] 2>/dev/null || { say "ABORT: checkpoint missing/incomplete on $n"; exit 1; }
  say "  .${n##*.} COMPLETE, dcp entries=$r"
done

say "rebooting all 4 (doctrine: pristine nodes before any distributed launch)"
for n in "${NODES[@]}"; do timeout 8 ssh -o ConnectTimeout=5 spark@"$n" 'sudo reboot' 2>/dev/null; done
for t in $(seq 1 60); do
  back=0
  for n in "${NODES[@]}"; do
    r=$(timeout 5 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" 'cut -d. -f1 /proc/uptime' 2>/dev/null)
    [ -n "$r" ] && [ "$r" -lt 180 ] 2>/dev/null && back=$((back+1))
  done
  [ "$back" = 4 ] && { say "all 4 back"; break; }
  sleep 10
done

say "launching 4-rank bake → $OUT"
env LORA_MODE=1 LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
  MODEL_PATH=${SPARK_HOME}/models/prod_v2_ep3_hf \
  SFT_DIR=${SPARK_HOME}/module2_sft SFT_JSONL=${SPARK_HOME}/module2_sft/module2_train.jsonl \
  BAKE_TO_HF="$OUT" RESUME_DELTA="$CKPT" \
  OUTPUT_DIR=${SPARK_HOME}/training_outputs/module2_cumulative_lora \
  MAX_SEQ=8192 BATCH_SIZE_PER_RANK=1 CHECKPOINT_DCP=1 \
  TOTAL_STEPS=225 SESSION_LIMIT=1 SAVE_EVERY=1 CLOCK_CAP=1600 \
  bash "$RECIPES/run_4node_27b_cpt.sh" >> "$LOG.launch" 2>&1

say "waiting for bake session exit"
for t in $(seq 1 80); do
  alive=0
  for n in "${NODES[@]}"; do
    r=$(timeout 6 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" 'tmux has-session -t cpt27b 2>/dev/null && echo A' 2>/dev/null)
    [ "$r" = "A" ] && alive=$((alive+1))
  done
  [ "$alive" = 0 ] && break
  sleep 30
done

say "=== bake receipts (mutual gate with codex) ==="
timeout 20 ssh spark@${SPARK_MASTER} "ls -la $OUT 2>/dev/null; echo '--- adapter sha256 ---'; sha256sum $OUT/adapter_model.safetensors 2>/dev/null; md5sum $OUT/adapter_model.safetensors 2>/dev/null; echo '--- config ---'; cat $OUT/adapter_config.json 2>/dev/null" | tee -a "$LOG"
say "receipts above — send to codex for deploy approval (NO deploy without it)"
