#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# run_till_done_v3.sh — AUTONOMOUS driver: train production_v2 to step 693, no human/Claude
# in the loop per boundary (Jesse 2026-07-16: "GET THE PROPER TRAINING RUN DONE! DON'T STOP!").
# Loop: pick next target -> reboot all 4 -> launch session -> wait for exit -> repeat.
# Pull-off/crash without a new checkpoint = automatic retry of the same target.
# Targets hit the exact epoch boundaries 231/462/693 for per-epoch eval checkpoints.
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
MASTER=${SPARK_MASTER}
CKPT_DIR=${CKPT_DIR:-${SPARK_HOME}/training_outputs/production_v2}
RECIPES=${REPO_ROOT}/dense-9b/recipes
INSTR=${REPO_ROOT}/dense-9b/instrumentation
# TARGETS and TOTAL_STEPS are env-overridable so ONE driver serves every horizon. Hardcoding
# them meant a run on a different horizon had no session-cycling driver at all: the trainer
# stops at SESSION_LIMIT and nothing relaunches it, so the run simply sits finished-but-
# incomplete with no error anywhere. Observed 2026-07-28 on a 628-step horizon.
# Defaults preserve the 693-step campaign this file was written for.
TARGETS=(${DRIVER_TARGETS:-231 321 411 462 552 642 693})   # default: 51/90/90/51/90/90/51 — epoch saves at 231/462/693
DLOG=${MIRA_HOME}/.taey-babysit/driver.log
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

# COOLDOWN (2026-07-16 measured): sessions launched >=1h idle run clean (S1,S4,attempt-1);
# back-to-back launches die 5-30min in on sustained 88-90C (rack bulk heat accumulates; zones
# cool in ~1min but the chassis does not). Enforce a minimum idle gap before each launch.
COOLDOWN_S=${COOLDOWN_S:-1200}
last_session_end=0
# NO DEFAULTS HERE EITHER. Removing the per-run defaults from run_4node_27b_cpt.sh moved them one
# level UP: until 2026-08-19 this wrapper still supplied CPT_DATA, MAX_SEQ, BATCH_SIZE_PER_RANK,
# TOTAL_STEPS, LR and WARMUP_STEPS from its own `:-` defaults, so the launcher's ${VAR:?} was
# satisfied by a value nobody chose. Found by tutor-grok after I had twice reported the defaults
# removed. A wrapper that answers the launcher's questions for you is the same defect wearing a
# different filename.
_w_missing=""
for _v in CPT_DATA TOTAL_STEPS MAX_SEQ BATCH_SIZE_PER_RANK LR WARMUP_STEPS; do
  eval "[ -n \"\${${_v}+x}\" ]" || _w_missing="$_w_missing $_v"
done
if [ -n "$_w_missing" ]; then
  echo "ABORT: run_till_done_v3.sh drives a real campaign and was given no:$_w_missing" >&2
  echo "  This wrapper reboots four nodes and trains for hours. Decide the campaign first, e.g." >&2
  echo "    CPT_DATA=<corpus> TOTAL_STEPS=693 MAX_SEQ=2560 BATCH_SIZE_PER_RANK=4 \\" >&2
  echo "    LR=1e-5 WARMUP_STEPS=15 bash dense-9b/recipes/run_till_done_v3.sh" >&2
  exit 1
fi

attempt=0
while true; do
  cur=$(latest_ckpt)
  [ -z "$cur" ] && { say "ERROR: cannot read checkpoints"; sleep 60; continue; }
  FINAL_TARGET=${TARGETS[${#TARGETS[@]}-1]}
  [ "$cur" -ge "$FINAL_TARGET" ] 2>/dev/null && break
  # next target above current
  tgt=""
  for T in "${TARGETS[@]}"; do [ "$T" -gt "$cur" ] && { tgt=$T; break; }; done
  [ -z "$tgt" ] && break
  SL=$((tgt - cur))
  attempt=$((attempt + 1))
  say "=== attempt $attempt: checkpoint-$cur -> target $tgt (SESSION_LIMIT=$SL) ==="

  # REALITY-BASED cooldown (Jesse 2026-07-16: base it on reality, not a timer): poll actual
  # idle zone temps; launch when every node's hottest zone <= COOL_AT C (cold baseline 43-48;
  # post-session bulk heat shows as idle zones 51-59 drifting down). Floor 300s, cap 2700s.
  if [ "$last_session_end" != 0 ]; then
    say "cooldown: waiting for all idle zones <= ${COOL_AT:-52}C (floor 300s, cap 2700s)"
    sleep 300
    for w in $(seq 1 40); do
      hot=0; rpt=""
      for n in "${NODES[@]}"; do
        t=$(timeout 6 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" 'cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1' 2>/dev/null)
        t=$(( ${t:-0} / 1000 )); rpt="$rpt .${n##*.}=${t}C"
        [ "$t" -gt "${COOL_AT:-52}" ] && hot=1
      done
      say "cooldown poll:$rpt"
      [ "$hot" = 0 ] && { say "rack cool — launching"; break; }
      sleep 60
    done
  fi
  reboot_all
  fabric_ok || { say "fabric FAIL — retrying cycle"; sleep 30; continue; }
  rm -f /tmp/thermal_pulloff.flag

  # fresh watchdog per attempt (kill any prior)
  for pid in $(ps -eo pid,args | awk '/^ *[0-9]+ bash thermal_watchdog.sh/{print $1}'); do kill "$pid" 2>/dev/null; done
  ( cd "$INSTR" && setsid bash -c "PULL_OFF=90 PERSIST=3 INTERVAL=8 LOG=watchdog_auto_a${attempt}.csv exec bash thermal_watchdog.sh" >> "$DLOG.watchdog" 2>&1 < /dev/null & )

  # One door: resolve+authorize+lifecycle via taey-train (inner launcher also re-checks).
  "$REPO_ROOT/scripts/taey-train" cpt_27b_4node \
  CPT_DATA=$CPT_DATA \
  CPT_PACKED=1 MAX_SEQ=$MAX_SEQ BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK \
  TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SL SAVE_EVERY=$SL CHECKPOINT_DCP=1 \
  LR=$LR WARMUP_STEPS=$WARMUP_STEPS \
  ${MODEL_PATH:+MODEL_PATH=$MODEL_PATH} \
  ADAFACTOR_ALPHA_MODE=absolute ADAFACTOR_EPS1=fp32 ADAFACTOR_DOSE_LOG=1 \
  OUTPUT_DIR=$CKPT_DIR \
  RESUME_DELTA=$CKPT_DIR/checkpoint-$cur \
  CLOCK_CAP=1600 \
  >> "$DLOG.launch_a$attempt" 2>&1

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
  last_session_end=$(date +%s)
  new=$(latest_ckpt)
  if [ -n "$new" ] && [ "$new" -gt "$cur" ] 2>/dev/null; then
    say "PROGRESS: checkpoint-$new saved (was $cur)"
    # EPOCH-BOUNDARY EXPORT (Jesse 2026-07-16: test every epoch; the 19:52 gap was wasted
    # because the driver relaunched instead of exporting — never again). At an epoch
    # checkpoint (231/462/693), run the 15-min Artifact-B export IN THIS GAP, before the
    # cooldown/relaunch. Nodes are free right now; this is the correct slot.
    if [ "$new" = 231 ] || [ "$new" = 462 ] || [ "$new" = 693 ]; then
      say "EPOCH EXPORT: running Artifact-B export of checkpoint-$new in the boundary gap"
      CPT_DATA=/var/spark/isma/training/cpt_production_v2_packed_2560.jsonl \
      CPT_PACKED=1 MAX_SEQ=2560 BATCH_SIZE_PER_RANK=4 \
      OUTPUT_DIR=$CKPT_DIR \
      RESUME_DELTA=$CKPT_DIR/checkpoint-$new \
      EXPORT_DCP=${SPARK_HOME}/export_prod_v2_ck$new \
      CLOCK_CAP=1600 \
      bash "$RECIPES/bake_27b.sh" >> "$DLOG.export_$new" 2>&1
      say "EPOCH EXPORT done (see $DLOG.export_$new tail: $(tail -1 "$DLOG.export_$new" 2>/dev/null | head -c 100))"
      taey-notify infra "EXPORT-READY (auto, tutor driver): checkpoint-$new exported to ${SPARK_HOME}/export_prod_v2_ck$new on all 4 nodes (per-rank shards + global metadata + sha manifests). Pull -> bake_dcp_offline -> weight-diff sanity -> serve. Ping tutor with endpoint." >/dev/null 2>&1
      taey-notify tutor "DRIVER: epoch checkpoint-$new EXPORTED in boundary gap; infra pinged; relaunching training." >/dev/null 2>&1
    fi
    taey-notify tutor "DRIVER: checkpoint-$new saved (target $tgt). $( [ "$new" = 231 ] || [ "$new" = 462 ] || [ "$new" = 693 ] && echo 'EPOCH BOUNDARY — eval time.' )" >/dev/null 2>&1
  else
    say "NO PROGRESS (pull-off/crash before save) — retrying target $tgt"
    taey-notify tutor "DRIVER: attempt $attempt died before save (still at checkpoint-$cur, pulloff=$(cat /tmp/thermal_pulloff.flag 2>/dev/null || echo none)); auto-retrying" >/dev/null 2>&1
  fi
done
say "=== TRAINING COMPLETE: checkpoint-693 (3 epochs) ==="
taey-notify tutor "DRIVER: ✅ TRAINING COMPLETE — checkpoint-693 (3 epochs, full corpus). Epoch checkpoints 231/462/693 on disk. Eval battery next." >/dev/null 2>&1
