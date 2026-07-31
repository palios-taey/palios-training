#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# run_till_done_v2.sh — AUTONOMOUS driver: train production_v2 to step 693, no human/Claude
# in the loop per boundary (Jesse 2026-07-16: "GET THE PROPER TRAINING RUN DONE! DON'T STOP!").
# Loop: pick next target -> reboot all 4 -> launch session -> wait for exit -> repeat.
# Pull-off/crash without a new checkpoint = automatic retry of the same target.
# Targets hit the exact epoch boundaries 231/462/693 for per-epoch eval checkpoints.
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
MASTER=${SPARK_MASTER}
CKPT_DIR=${SPARK_HOME}/training_outputs/production_v2
RECIPES=${REPO_ROOT}/dense-9b/recipes
INSTR=${REPO_ROOT}/dense-9b/instrumentation
TARGETS=(231 321 411 462 552 642 693)   # 51/90/90/51/90/90/51 — exact epoch saves at 231/462/693
DLOG=${MIRA_HOME}/.taey-babysit/driver.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DLOG"; }

latest_ckpt() {
  timeout 10 ssh -o ConnectTimeout=6 spark@"$MASTER" \
    "ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1" 2>/dev/null
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

attempt=0
while true; do
  cur=$(latest_ckpt)
  [ -z "$cur" ] && { say "ERROR: cannot read checkpoints"; sleep 60; continue; }
  [ "$cur" -ge 693 ] 2>/dev/null && break
  # next target above current
  tgt=""
  for T in "${TARGETS[@]}"; do [ "$T" -gt "$cur" ] && { tgt=$T; break; }; done
  [ -z "$tgt" ] && break
  SL=$((tgt - cur))
  attempt=$((attempt + 1))
  say "=== attempt $attempt: checkpoint-$cur -> target $tgt (SESSION_LIMIT=$SL) ==="

  reboot_all
  fabric_ok || { say "fabric FAIL — retrying cycle"; sleep 30; continue; }
  rm -f /tmp/thermal_pulloff.flag

  # fresh watchdog per attempt (kill any prior)
  for pid in $(ps -eo pid,args | awk '/^ *[0-9]+ bash thermal_watchdog.sh/{print $1}'); do kill "$pid" 2>/dev/null; done
  ( cd "$INSTR" && setsid bash -c "PULL_OFF=90 PERSIST=3 INTERVAL=8 LOG=watchdog_auto_a${attempt}.csv exec bash thermal_watchdog.sh" >> "$DLOG.watchdog" 2>&1 < /dev/null & )

  CPT_DATA=/var/spark/isma/training/cpt_production_v2_packed_2560.jsonl \
  CPT_PACKED=1 MAX_SEQ=2560 BATCH_SIZE_PER_RANK=4 \
  TOTAL_STEPS=693 SESSION_LIMIT=$SL SAVE_EVERY=$SL CHECKPOINT_DCP=1 \
  LR=1e-5 WARMUP_STEPS=15 \
  ADAFACTOR_ALPHA_MODE=absolute ADAFACTOR_EPS1=fp32 ADAFACTOR_DOSE_LOG=1 \
  OUTPUT_DIR=$CKPT_DIR \
  RESUME_DELTA=$CKPT_DIR/checkpoint-$cur \
  CLOCK_CAP=1600 \
  bash "$RECIPES/run_4node_27b_cpt.sh" >> "$DLOG.launch_a$attempt" 2>&1

  # wait for the session to end (all tmux gone), max 3.5h
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
  new=$(latest_ckpt)
  if [ -n "$new" ] && [ "$new" -gt "$cur" ] 2>/dev/null; then
    say "PROGRESS: checkpoint-$new saved (was $cur)"
    taey-notify tutor "DRIVER: checkpoint-$new saved (target $tgt). $( [ "$new" = 231 ] || [ "$new" = 462 ] || [ "$new" = 693 ] && echo 'EPOCH BOUNDARY — eval time.' )" >/dev/null 2>&1
  else
    say "NO PROGRESS (pull-off/crash before save) — retrying target $tgt"
    taey-notify tutor "DRIVER: attempt $attempt died before save (still at checkpoint-$cur, pulloff=$(cat /tmp/thermal_pulloff.flag 2>/dev/null || echo none)); auto-retrying" >/dev/null 2>&1
  fi
done
say "=== TRAINING COMPLETE: checkpoint-693 (3 epochs) ==="
taey-notify tutor "DRIVER: ✅ TRAINING COMPLETE — checkpoint-693 (3 epochs, full corpus). Epoch checkpoints 231/462/693 on disk. Eval battery next." >/dev/null 2>&1
