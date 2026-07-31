#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Module-1 SFT on RECIPE B — the stack that has actually completed a 27B LoRA run.
# Recipe B = moe-35b/trainers/train_fsdp_v3.py + launch_production_sft.sh, transformers' stock
# Adafactor, NO torch-Adafactor monkeypatch. See RECIPES.md.
# PROVEN BY: prod_sft_v1_pubrepo_chunked, 150 steps on this 27B at 4 ranks, 2026-06-18.
set -uo pipefail
NODES=($SPARK_MGMT_IPS); MASTER=${SPARK_RAIL_MASTER}
REC=${SPARK_HOME}/palios-training/moe-35b/recipes
CKPT=${SPARK_HOME}/training_outputs/module1_recipeB
TOTAL=${TOTAL:-2304}; SL=${SL:-350}
say(){ echo "[m1B $(date +%H:%M:%S)] $*"; }
latest(){ ssh -o ConnectTimeout=6 spark@"${NODES[0]}" "ls -d $CKPT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1" 2>/dev/null; }

for attempt in $(seq 1 12); do
  cur=$(latest); cur=${cur:-0}
  say "=== session $attempt: at step ${cur}/${TOTAL} ==="
  [ "$cur" -ge "$TOTAL" ] && { say "COMPLETE at $cur"; break; }

  say "reboot all 4 (ALWAYS)"
  for n in "${NODES[@]}"; do timeout 8 ssh -o ConnectTimeout=5 spark@"$n" 'sudo reboot' 2>/dev/null; done
  sleep 30
  up=0
  for t in $(seq 1 45); do
    up=0; for n in "${NODES[@]}"; do timeout 5 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" true 2>/dev/null && up=$((up+1)); done
    [ "$up" -eq 4 ] && break; sleep 10
  done
  [ "$up" -eq 4 ] || { say "ABORT: only $up/4 back"; exit 1; }
  sleep 20

  RD=""; [ "$cur" -gt 0 ] && RD="$CKPT/checkpoint-$cur"
  say "launching RECIPE B session (SESSION_LIMIT=$SL, resume=${RD:-none})"
  for i in "${!NODES[@]}"; do
    n="${NODES[$i]}"
    ssh -o ConnectTimeout=10 spark@"$n" "tmux kill-session -t m1b 2>/dev/null; tmux new-session -d -s m1b \
      'cd $REC && NODE0_IP=$MASTER MASTER_ADDR=$MASTER NODE_RANK=$i NNODES=4 \
       MODEL_PATH=${SPARK_HOME}/models/prod_v2_ep3_hf OUTPUT_DIR=$CKPT \
       SFT_DIR=${SPARK_HOME}/module1_sft SFT_GLOB=module1_train.jsonl \
       MAX_SEQ=8192 KEYSTONE_LAYERS=\"[8,9,21,25,28,38]\" \
       TOTAL_STEPS=$TOTAL SESSION_LIMIT=$SL SAVE_EVERY=$SL WARMUP_STEPS=25 \
       ${RD:+RESUME_DELTA=$RD} \
       bash launch_production_sft.sh > ${SPARK_HOME}/cpt27b_logs/m1b_r$i.log 2>&1'" 2>/dev/null
    say "  node $i ($n) launched"
  done

  say "waiting for session exit (max 2.2h)"
  for t in $(seq 1 160); do
    a=0; for n in "${NODES[@]}"; do timeout 6 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" 'tmux has-session -t m1b 2>/dev/null && echo A' 2>/dev/null | grep -q A && a=$((a+1)); done
    [ "$a" -eq 0 ] && { say "session ended"; break; }
    sleep 50
  done
  new=$(latest); say "session $attempt done: step ${cur} -> ${new:-0}"
  [ "${new:-0}" = "$cur" ] && { say "NO PROGRESS — stopping for diagnosis"; exit 1; }
done
say "MODULE-1 RECIPE-B RUN FINISHED at step $(latest)/$TOTAL"
