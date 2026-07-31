#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Durable 27B babysitter — SYSTEM CRONTAB every 5 min (survives Claude session death).
# Jesse 2026-07-16: "Your wake did not work... you have to do that properly." In-session
# ScheduleWakeup/CronCreate die with the session; this is the durable wake path: check the
# run's real state and taey-notify tutor ONLY when attention is needed (event-driven, quiet
# when healthy). The notification daemon injects into the tutor session regardless of
# harness wake state.
set -uo pipefail
MASTER=${SPARK_MASTER}
NODES=($SPARK_MGMT_IPS)
LOG=${SPARK_HOME}/cpt27b_logs/r0.log
CKPT_DIR=${SPARK_HOME}/training_outputs/production_v2
STATE=${MIRA_HOME}/.taey-babysit
mkdir -p "$STATE"
notify() {
  # ONE alert per INCIDENT (2026-07-16 fix: 30-min re-fire flooded 35 dupes into an unread
  # inbox). Fingerprint = condition + message; re-notify only if the state CHANGES, or as a
  # 2-hour escalation re-ping while still unhandled.
  local key="$1" msg="$2" now stampfile fpfile last fp
  stampfile="$STATE/last_$key"; fpfile="$STATE/fp_$key"
  now=$(date +%s); last=$(cat "$stampfile" 2>/dev/null || echo 0)
  fp=$(echo "$msg" | md5sum | cut -d' ' -f1)
  if [ "$(cat "$fpfile" 2>/dev/null)" = "$fp" ] && [ $((now - last)) -lt 7200 ]; then
    return 0
  fi
  echo "$now" > "$stampfile"; echo "$fp" > "$fpfile"
  # target = TUTOR (conductor receipts 2026-07-16: THIS process is registered tutor
  # (cwd=palios-training) despite living in the treasurer-NAMED tmux; tutor-inbox [NOTIFY]
  # injections demonstrably reach it — the earlier 'treasurer' retarget sent alerts to the
  # real treasurer session instead, which is why the 14:33 pull-off went unseen for 2.5h)
  /usr/local/bin/taey-notify tutor "BABYSIT-CRON: $msg" >/dev/null 2>&1
}

# 1. reachability + tmux liveness per node
dead=""; unreach=""
for n in "${NODES[@]}"; do
  r=$(timeout 8 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" \
      'tmux has-session -t cpt27b 2>/dev/null && echo ALIVE || echo EXITED' 2>/dev/null)
  case "$r" in
    ALIVE) ;;
    EXITED) dead="$dead .${n##*.}";;
    *) unreach="$unreach .${n##*.}";;
  esac
done
[ -n "$unreach" ] && notify unreach "node(s) UNREACHABLE:$unreach — check power/net"
if [ -n "$dead" ]; then
  ck=$(timeout 8 ssh -o ConnectTimeout=5 spark@"$MASTER" "ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1" 2>/dev/null)
  notify exited "tmux EXITED on$dead — session boundary or crash. Latest ckpt: ${ck:-none}. Reboot-cycle + resume needed."
fi

# 2. thermal pull-off flag (watchdog stopped the run)
[ -f /tmp/thermal_pulloff.flag ] && notify pulloff "thermal watchdog PULLED OFF ($(cat /tmp/thermal_pulloff.flag 2>/dev/null | head -c 120)) — reboot-cool + resume needed"

# 3. new failure lines in master log
errs=$(timeout 8 ssh -o ConnectTimeout=5 spark@"$MASTER" \
  "grep -acE 'Traceback|CUDA error|out of memory|NaN detected| nan,' $LOG 2>/dev/null" 2>/dev/null || echo 0)
prev_errs=$(cat "$STATE/err_count" 2>/dev/null || echo 0)
if [ -n "$errs" ] && [ "$errs" -gt "$prev_errs" ] 2>/dev/null; then
  echo "$errs" > "$STATE/err_count"
  tailerr=$(timeout 8 ssh -o ConnectTimeout=5 spark@"$MASTER" \
    "grep -aE 'Traceback|CUDA error|out of memory|NaN detected' $LOG 2>/dev/null | tail -1 | head -c 160" 2>/dev/null)
  notify logerr "NEW failure lines in r0.log (count $prev_errs→$errs): $tailerr"
fi

# 4. stall: tmux alive everywhere but log not advancing >6 min
if [ -z "$dead" ] && [ -z "$unreach" ]; then
  mt=$(timeout 8 ssh -o ConnectTimeout=5 spark@"$MASTER" "stat -c %Y $LOG 2>/dev/null" 2>/dev/null)
  now=$(date +%s)
  if [ -n "$mt" ] && [ $((now - mt)) -gt 360 ]; then
    notify stall "training log STALLED ($(( (now - mt) / 60 ))min no writes, tmux alive) — possible wedge"
  fi
fi

# 5. Mira-side safety monitors died while training runs
if [ -z "$dead" ] && [ -z "$unreach" ]; then
  wd=$(pgrep -f "bash thermal_watchdog.sh" | wc -l)
  [ "$wd" -eq 0 ] && notify nowd "thermal watchdog NOT RUNNING while training is live — restart it"
fi
exit 0
