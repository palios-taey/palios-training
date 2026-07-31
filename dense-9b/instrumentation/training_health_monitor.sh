#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# TRAINING HEALTH MONITOR — active stall/crash detector (Jesse 2026-07-23: "how do you not know
# when training completely stops?"). Polls the master node's r0.log every CHECK_INTERVAL and ALERTS
# the owning session IMMEDIATELY (taey-notify) when training stalls, crashes, or wedges — instead of a
# coarse 25-min wakeup poll that let a hang sit ~46 min undetected. A hung NCCL collective emits NO new
# log lines, so a stalled log mtime / non-advancing step IS the signal.
set -uo pipefail
MASTER="${MASTER:-${SPARK_MASTER}}"
LOG="${LOG:-${SPARK_HOME}/cpt27b_logs/r0.log}"
NOTIFY_TARGET="${NOTIFY_TARGET:-tutor}"
CHECK_INTERVAL="${CHECK_INTERVAL:-90}"    # seconds between checks
STALL_SECONDS="${STALL_SECONDS:-360}"     # log-mtime staleness that counts as a stall (6 min > 1 step)
TARGET_STEP="${TARGET_STEP:-2304}"
STATE=""                                   # last alert state (dedup: only notify on state change)
say(){ echo "[health $(date +%H:%M:%S)] $*"; }
alert(){ # $1=state $2=message  — notify only when the state changes (no spam)
  [ "$STATE" = "$1" ] && return
  STATE="$1"
  taey-notify "$NOTIFY_TARGET" "TRAINING-HEALTH [$1]: $2" --type escalation 2>/dev/null
  say "ALERT[$1]: $2"
}
say "monitor start: master=$MASTER log=$LOG interval=${CHECK_INTERVAL}s stall=${STALL_SECONDS}s"
while true; do
  # One ssh round-trip gathers: tmux alive?, log mtime epoch, last step, wedge/watchdog markers, latest ckpt.
  R=$(ssh -o ConnectTimeout=8 -o BatchMode=yes spark@"$MASTER" '
    tmux has-session -t cpt27b 2>/dev/null && echo "TMUX=alive" || echo "TMUX=dead"
    if [ -f '"$LOG"' ]; then echo "MTIME=$(stat -c %Y '"$LOG"')"; else echo "MTIME=0"; fi
    echo "NOW=$(date +%s)"
    echo "STEP=$(grep -aoE "\[step [0-9]+\]" '"$LOG"' 2>/dev/null | tail -1 | grep -oE "[0-9]+")"
    echo "BAD=$(grep -acE "WEDGE-GUARD|Watchdog caught|RANK-DIVERGENT|Traceback|SIGABRT" '"$LOG"' 2>/dev/null)"
    echo "CKPT=$(ls -d ${SPARK_HOME}/training_outputs/module1_lora/checkpoint-* 2>/dev/null | sed "s#.*checkpoint-##" | sort -n | tail -1)"
  ' 2>/dev/null)
  # DRIVER-OWNED WINDOWS (2026-07-23, false-alarm classes 2+3): the session driver reboots all 4
  # nodes between sessions — while it is alive, node-unreachable and tmux-gone are EXPECTED states
  # it manages itself. Only alert on those when NO driver is running (a real orphan/down).
  DRIVER_ALIVE=0
  pgrep -f "run_module1_till_done.sh" >/dev/null 2>&1 && DRIVER_ALIVE=1
  if [ -z "$R" ]; then
    if [ "$DRIVER_ALIVE" = "1" ]; then say "node unreachable but driver alive (reboot window) — no alert"
    else alert "NODE-UNREACHABLE" "cannot ssh $MASTER — node down or network (no driver running)"; fi
    sleep "$CHECK_INTERVAL"; continue
  fi
  TMUX=$(echo "$R" | sed -n 's/^TMUX=//p'); MTIME=$(echo "$R" | sed -n 's/^MTIME=//p')
  NOW=$(echo "$R" | sed -n 's/^NOW=//p'); STEP=$(echo "$R" | sed -n 's/^STEP=//p')
  BAD=$(echo "$R" | sed -n 's/^BAD=//p'); CKPT=$(echo "$R" | sed -n 's/^CKPT=//p')
  AGE=$(( NOW - ${MTIME:-0} ))
  if [ "${BAD:-0}" -gt 0 ]; then
    alert "WEDGE-OR-ERROR" "wedge/watchdog/error marker in $LOG at step ${STEP:-?} (ckpt ${CKPT:-?}). Capture py-spy + re-consult."
  elif [ "$TMUX" = "alive" ] && [ -n "$STEP" ] && [ "$AGE" -gt "$STALL_SECONDS" ]; then
    # NOTE: `-n "$STEP"` gates the stall detector to the STEPPING phase only. accelerator.prepare()
    # (FSDP2 wrapping the 27B) legitimately produces a ~6-min no-log gap during init — flagging that as
    # a stall is a false alarm (fixed 2026-07-23 after it cried wolf on the AdamW root-fix init).
    alert "STALL" "log frozen ${AGE}s at step ${STEP:-?} (tmux alive, no progress). Likely a hung collective — capture the 4-rank py-spy divergence map NOW before the 1hr watchdog SIGABRT."
  elif [ "$TMUX" = "dead" ] && [ "${CKPT:-0}" -lt "$TARGET_STEP" ]; then
    if [ "$DRIVER_ALIVE" = "1" ]; then say "session gone at ckpt ${CKPT:-?} but driver alive (between-sessions) — no alert"
    else
    alert "SESSION-DOWN" "tmux cpt27b gone at ckpt ${CKPT:-?}/$TARGET_STEP (not complete, no driver running). Session ended/crashed — check relaunch."
    fi
  else
    # Healthy (advancing, or cleanly complete). Reset state so the next fault re-alerts.
    [ -n "$STATE" ] && say "recovered/healthy: step=${STEP:-?} ckpt=${CKPT:-?} age=${AGE}s tmux=$TMUX"
    STATE=""
  fi
  sleep "$CHECK_INTERVAL"
done
