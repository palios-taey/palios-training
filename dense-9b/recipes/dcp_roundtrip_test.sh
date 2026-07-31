#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# dcp_roundtrip_test.sh — PRODUCTION validation of the resumable DCP checkpoint (enabler #1).
# Full cycle on the real 27B: reboot->train a few steps->DCP save->clean exit->collect to Mira->
# REBOOT all 4->scatter->RESUME from checkpoint->verify (DCP RESUME log + data-skip + loss continuity).
#
# NOTE (Jesse 2026-07-10): the REBOOT+RESUME automation is LAST PRIORITY and has not worked well in
# the past. In production the operator is WOKEN every ~2h to reboot + collect/scatter + resume BY HAND
# using the durable primitives: checkpoint_sync.sh (collect/scatter) + the trainer DCP save/load.
# This monolithic auto-runner is a CONVENIENCE for a one-shot validation only — the manual step-by-step
# path (drive each phase, verify the checkpoint before resuming) is the real workflow. Do not depend
# on this script's auto-reboot for the actual 2-hour cycle.
set -uo pipefail
MASTER=${SPARK_MASTER}
NODES=($SPARK_MGMT_IPS)
REPO=${REPO_ROOT}
SYNC=$REPO/dense-9b/recipes/checkpoint_sync.sh
TEST_OUT=${SPARK_HOME}/training_outputs/dcp_test          # isolated test OUTPUT_DIR (node-local)
CLOCK_CAP=2200
LOGDIR=${SPARK_HOME}/cpt27b_logs
SAVE_STEP=4                                              # session-limit clean-exit + DCP save here
S(){ ssh -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vaE "ControlSocket|mux_client"; }

reboot_all(){
  echo "=== REBOOT all 4 (discipline) $(date -u +%H:%M:%S) ==="
  for n in "${NODES[@]}"; do S "$n" 'sudo reboot' >/dev/null 2>&1 & done
  sleep 45
  for n in "${NODES[@]}"; do
    for i in $(seq 1 40); do
      up=$(S "$n" 'cut -d. -f1 /proc/uptime' 2>/dev/null | tr -dc 0-9)
      [ -n "$up" ] && [ "$up" -lt 300 ] 2>/dev/null && { echo "  .${n##*.} up (${up}s)"; break; }
      sleep 6
    done
  done
}
prep_all(){   # deploy current trainer + clock cap + earlyoom off (per run discipline)
  echo "=== PREP: deploy trainer + clock cap ${CLOCK_CAP}MHz + earlyoom off ==="
  for n in "${NODES[@]}"; do
    scp -q -o ControlMaster=no -o ControlPath=none "$REPO/dense-9b/trainers/train_fsdp_dense_9b.py" \
      spark@"$n":${SPARK_HOME}/palios-training/dense-9b/trainers/train_fsdp_dense_9b.py 2>&1 | grep -vaE "ControlSocket|mux"
    S "$n" "sudo systemctl stop earlyoom 2>/dev/null; sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP 2>&1 | head -1; mkdir -p $LOGDIR"
  done
}
launch(){     # $1=extra env (OUTPUT_DIR/RESUME_DELTA etc). Short config: session-limit exit at SAVE_STEP.
  echo "=== LAUNCH (short: SESSION_LIMIT=$SAVE_STEP SAVE_EVERY=$SAVE_STEP) $1 ==="
  env $1 OUTPUT_DIR=$TEST_OUT SESSION_LIMIT=$SAVE_STEP SAVE_EVERY=$SAVE_STEP TOTAL_STEPS=3000 \
      CHECKPOINT_DCP=1 bash "$REPO/dense-9b/recipes/run_4node_27b_cpt.sh"
}
wait_complete(){  # poll master for the checkpoint COMPLETE marker (or failure) up to ~20 min
  local ck=$1
  echo "=== WAIT for $ck/COMPLETE on master (or failure) ==="
  for t in $(seq 1 60); do
    if [ "$(S "$MASTER" "test -f $TEST_OUT/$ck/COMPLETE && echo Y")" = "Y" ]; then echo "  COMPLETE at $ck (t=$t)"; return 0; fi
    local err; err=$(S "$MASTER" "grep -aE 'OOM|out of memory|Traceback|CUDA error|NCCL' $LOGDIR/r0.log 2>/dev/null | tail -1")
    [ -n "$err" ] && { echo "  FAILURE signal: ${err:0:120}"; return 1; }
    sleep 20
  done
  echo "  TIMEOUT waiting for $ck"; return 1
}
losses(){ S "$MASTER" "grep -aoE 'step [0-9]+.*loss[= ]+[0-9.]+' $LOGDIR/r0.log 2>/dev/null | tail -12"; }

echo "############ PHASE 1: train -> DCP save -> collect ############"
reboot_all; prep_all
launch "" || true
if ! wait_complete "checkpoint-$SAVE_STEP"; then echo "PHASE1 FAILED — see $MASTER:$LOGDIR/r0.log"; exit 1; fi
echo "--- phase1 losses (baseline trajectory) ---"; losses
echo "--- phase1 DCP save log ---"; S "$MASTER" "grep -aE 'DCP save|COMPLETE' $LOGDIR/r0.log | tail -4"
bash "$SYNC" collect "$SAVE_STEP" || { echo "COLLECT FAILED"; exit 1; }

echo "############ PHASE 2: reboot -> scatter -> RESUME -> verify ############"
reboot_all; prep_all
bash "$SYNC" scatter "$SAVE_STEP" "" "$TEST_OUT" || { echo "SCATTER FAILED"; exit 1; }
# resume: train SAVE_STEP more (session-limit is relative to session start -> exits at step 2*SAVE_STEP)
launch "RESUME_DELTA=$TEST_OUT/checkpoint-$SAVE_STEP" || true
if ! wait_complete "checkpoint-$((SAVE_STEP*2))"; then echo "PHASE2 resume did not reach 2nd checkpoint"; fi
echo "=== VERIFY resume ==="
echo "--- DCP RESUME + data-skip logs (must be present) ---"
S "$MASTER" "grep -aE 'DCP RESUME|Resume skip complete|Resume: fast-forwarding|Scheduler' $LOGDIR/r0.log | tail -6"
echo "--- phase2 losses (must continue phase1 trajectory, NOT a fresh-start spike) ---"; losses
echo "############ DCP ROUND-TRIP TEST DONE $(date -u +%H:%M:%S) ############"
