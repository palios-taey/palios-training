#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# run_refresh_gate.sh — 50-step LIVE DOSE GATE for cpt_refresh_v1 (session 1 of the real
# schedule: warmup 33 + cosine over TOTAL_STEPS=1565, truncated at SESSION_LIMIT=50).
# Clone of the production_v2-proven run_till_done_v2.sh loop, single target, fresh start.
# Recipe: careers-qwen/CPT_REFRESH_RECIPE_v0.9.md (provisional 3-lane synthesis).
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
MASTER=${SPARK_MASTER}
CKPT_DIR=${SPARK_HOME}/training_outputs/cpt_refresh_v1
RECIPES=${REPO_ROOT}/dense-9b/recipes
INSTR=${REPO_ROOT}/dense-9b/instrumentation
DLOG=${MIRA_HOME}/.taey-babysit/refresh_gate.log
TARGET=50
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DLOG"; }

latest_ckpt() {
  timeout 10 ssh -o ConnectTimeout=6 spark@"$MASTER" \
    "if test -f '$CKPT_DIR/final/COMPLETE'; then
       python3 -c \"import torch; print(torch.load('$CKPT_DIR/final/trainer_meta.pt', map_location='cpu', weights_only=False)['step'])\";
     else
       ls -d '$CKPT_DIR'/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1;
     fi" 2>/dev/null
}

reboot_all() {
  say "rebooting all 4"
  for n in "${NODES[@]}"; do timeout 8 ssh -o ConnectTimeout=5 spark@"$n" 'sudo reboot' 2>/dev/null; done
  for t in $(seq 1 60); do
    back=0
    for n in "${NODES[@]}"; do
      r=$(timeout 5 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" 'cut -d. -f1 /proc/uptime' 2>/dev/null)
      [ -n "$r" ] && [ "$r" -lt 180 ] 2>/dev/null && back=$((back+1))
    done
    [ "$back" = 4 ] && { say "all 4 back"; return 0; }
    sleep 10
  done
  say "WARN: not all nodes back after 10min"; return 1
}

fabric_ok() {
  timeout 12 ssh spark@"$MASTER" 'for ip in ${SPARK_RAIL_MASTER} ${SPARK_RAIL_MASTER} ${SPARK_RAIL_MASTER}; do ping -c1 -W1 $ip >/dev/null 2>&1 || exit 1; done' 2>/dev/null
}

disk_ok() {
  for n in "${NODES[@]}"; do
    g=$(timeout 8 ssh -o ConnectTimeout=5 spark@"$n" "df --output=avail -BG / | tail -1 | tr -dc 0-9" 2>/dev/null)
    [ -n "$g" ] && [ "$g" -ge 40 ] 2>/dev/null || { say "DISK GATE FAIL on $n (${g:-?}G)"; return 1; }
  done
  return 0
}

attempt=0
while true; do
  cur=$(latest_ckpt); cur=${cur:-0}
  [ "$cur" -ge "$TARGET" ] 2>/dev/null && break
  attempt=$((attempt + 1))
  [ "$attempt" -gt 3 ] && { say "3 attempts without a save — STOP for RCA (first-error-full-stop)"; taey-notify --from tutor tutor "REFRESH GATE: 3 failed attempts, stopped for RCA. See $DLOG" >/dev/null 2>&1; exit 1; }
  SL=$((TARGET - cur))
  say "=== attempt $attempt: checkpoint-$cur -> gate target $TARGET (SESSION_LIMIT=$SL) ==="

  reboot_all
  fabric_ok || { say "fabric FAIL — retrying cycle"; sleep 30; continue; }
  disk_ok || exit 1
  # sync recipe scripts to the node-local trees (nodes run THEIR copy — a Mira-side
  # fix is not live until shipped; corpus-gate stale-launcher failure 2026-07-24)
  for n in "${NODES[@]}"; do
    scp -o ConnectTimeout=8 -q "$RECIPES/launch_cpt_qwen36_27b_fsdp.sh" "$RECIPES/run_4node_27b_cpt.sh" \
      spark@"$n":${SPARK_HOME}/palios-training/dense-9b/recipes/ 2>/dev/null || say "WARN: recipe sync failed on $n"
    scp -o ConnectTimeout=8 -q "$RECIPES/../trainers/train_fsdp_dense_9b.py" \
      spark@"$n":${SPARK_HOME}/palios-training/dense-9b/trainers/ 2>/dev/null || say "WARN: trainer sync failed on $n"
  done
  rm -f /tmp/thermal_pulloff.flag

  for pid in $(ps -eo pid,args | awk '/^ *[0-9]+ bash thermal_watchdog.sh/{print $1}'); do kill "$pid" 2>/dev/null; done
  ( cd "$INSTR" && setsid bash -c "PULL_OFF=90 PERSIST=3 INTERVAL=8 LOG=watchdog_gate_a${attempt}.csv exec bash thermal_watchdog.sh" >> "$DLOG.watchdog" 2>&1 < /dev/null & )

  RESUME=""
  [ "$cur" -gt 0 ] 2>/dev/null && RESUME="RESUME_DELTA=$CKPT_DIR/checkpoint-$cur"

  env $RESUME \
  CPT_DATA=/var/spark/isma/training/refresh_v1/MERGED_cpt_refresh_v1_train.jsonl \
  CPT_PACKED=1 MAX_SEQ=2560 BATCH_SIZE_PER_RANK=4 \
  TOTAL_STEPS=1565 SESSION_LIMIT=$SL SAVE_EVERY=$SL CHECKPOINT_DCP=1 \
  LR=4e-6 WARMUP_STEPS=32 \
  ADAFACTOR_ALPHA_MODE=absolute ADAFACTOR_EPS1=fp32 ADAFACTOR_DOSE_LOG=1 \
  MODEL_PATH=${SPARK_HOME}/models/prod_v2_ep3_hf \
  OUTPUT_DIR=$CKPT_DIR \
  CLOCK_CAP=1600 \
  bash "$RECIPES/run_4node_27b_cpt.sh" >> "$DLOG.launch_a$attempt" 2>&1

  say "session launched; waiting for exit"
  for t in $(seq 1 420); do
    alive=0
    for n in "${NODES[@]}"; do
      r=$(timeout 6 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" 'tmux has-session -t cpt27b 2>/dev/null && echo A' 2>/dev/null)
      [ "$r" = "A" ] && alive=$((alive+1))
    done
    [ "$alive" = 0 ] && break
    sleep 30
  done
  new=$(latest_ckpt); new=${new:-0}
  if [ "$new" -gt "$cur" ] 2>/dev/null; then
    say "PROGRESS: checkpoint-$new saved"
  else
    say "NO PROGRESS (pull-off/crash before save) — retry (pulloff=$(cat /tmp/thermal_pulloff.flag 2>/dev/null || echo none))"
  fi
done

# Gate readout: AF-DOSE + SR lines from master r0 log
say "=== GATE TARGET REACHED: checkpoint-$TARGET — extracting dose readout ==="
timeout 20 ssh spark@"$MASTER" "grep -hE 'AF-DOSE|SR-DELTA|floor_frac|U_hat' ${SPARK_HOME}/cpt27b_logs/r0.log 2>/dev/null | tail -30" | tee -a "$DLOG"
taey-notify --from tutor tutor "REFRESH GATE: checkpoint-50 SAVED. AF-DOSE readout appended to $DLOG — read bands now (floor_frac<0.05, RMS(U_hat) 0.3-1.5, SR live)." >/dev/null 2>&1
