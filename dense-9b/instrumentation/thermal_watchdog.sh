#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# thermal_watchdog.sh — "know when to pull off the track" (Jesse-directed 2026-07-11).
#
# The GB10 whole-node death is a ~94C board/SoC THERMAL shutdown (instrumented 2026-07-10). A hard
# shutdown leaves the node needing a PHYSICAL power-cycle (no BMC). This watchdog polls all 4 nodes'
# hottest gauge every INTERVAL s during training and, when ANY node crosses PULL_OFF (a margin BELOW
# 94C), it GRACEFULLY stops the run on all nodes (kills the training load) so the boards COOL instead
# of hard-crashing. The node stays operable -> a normal reboot-resume, never a physical power-cycle.
#
# It does NOT reboot — it stops the load and drops a flag file; the orchestrator/loop sees the flag
# (or the training-orchestrator task ending) and does the clean reboot+resume from the last checkpoint.
#
# Also logs every reading to a CSV (evidence + Taey substrate-proprioception corpus, SOUL=INFRA).
#
# Usage: PULL_OFF=82 INTERVAL=15 thermal_watchdog.sh   (env-overridable)
set -u
NODES=(68 80 12 19)
PULL_OFF="${PULL_OFF:-82}"      # C — pull off the track at this hottest-gauge temp (margin below 94C crash)
INTERVAL="${INTERVAL:-15}"      # s between sweeps
CRASH_TEMP=94                   # C — the known hard-shutdown point (for margin reporting)
FLAG=/tmp/thermal_pulloff.flag
LOG="${LOG:-${REPO_ROOT}/dense-9b/instrumentation/watchdog_$(date -u +%Y%m%d_%H%M%S 2>/dev/null || echo run).csv}"
rm -f "$FLAG"
echo "ts_utc,node,gpu_c,board_c,node_max_c" > "$LOG" 2>/dev/null || true
echo "=== thermal watchdog LIVE: pull-off@${PULL_OFF}C (crash~${CRASH_TEMP}C), sweep ${INTERVAL}s, log=$LOG ==="

read_temp() {  # $1=node -> prints "gpu board" (C)
  timeout 8 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@${SPARK_SUBNET}."$1" \
    'g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -dc 0-9); z=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo "${g:-0} $(( ${z:-0} / 1000 ))"' 2>/dev/null
}

graceful_pulloff() {  # $1=hottest node  $2=temp
  echo ">>> PULL OFF THE TRACK: node .$1 hit ${2}C (>= ${PULL_OFF}C, crash ~${CRASH_TEMP}C) at $(date -u +%H:%M:%S)"
  echo "PULLOFF node=.$1 temp=${2}C ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FLAG"
  # Stop the load on ALL nodes so boards cool — clean process stop, NOT a crash.
  for n in "${NODES[@]}"; do
    timeout 8 ssh -o ConnectTimeout=5 spark@${SPARK_SUBNET}."$n" \
      'tmux kill-session -t cpt27b 2>/dev/null; pkill -TERM -f train_fsdp_dense 2>/dev/null; sleep 2; pkill -9 -f train_fsdp_dense 2>/dev/null' 2>/dev/null &
  done
  wait
  taey-notify treasurer "tutor→Jesse: ⚠️ THERMAL PULL-OFF — node .$1 reached ${2}C (pull-off ${PULL_OFF}C, hard-crash ~${CRASH_TEMP}C). Gracefully STOPPED the run on all 4 nodes BEFORE a hard crash — nodes stay operable (normal reboot, NO power-cycle needed). Boards cooling now. Will reboot+resume from last checkpoint. This is the racecar pulling off the track, not slamming the rail." 2>/dev/null
  echo ">>> load stopped on all nodes; flag=$FLAG written; watchdog exiting (orchestrator will reboot+resume)"
}

# DEBOUNCE (2026-07-16, measured root cause of the false pull-offs): the ACPI zone readings
# produce single-sample transient spikes of +10-12C/8s (76→86, 75→87) that bulk board temp
# physically cannot do — thermal mass limits real climbs to ~C/minute (S3's real climb was
# 60→86 over ~60min). A single-sweep trigger killed HEALTHY runs on whichever node glitched
# (the "moving" failure). Require PERSIST consecutive sweeps >= PULL_OFF before pulling off:
# real runaways persist (S3 read 85,86 on consecutive sweeps and would still trigger);
# one-sample glitches get logged as SPIKE and ignored. Crash protection intact — at real
# climb rates, 86→94C takes minutes, and PERSIST adds only (PERSIST-1)*INTERVAL s latency.
PERSIST="${PERSIST:-3}"
consec=0
while true; do
  gmax=0; hottest=0; line=""
  for n in "${NODES[@]}"; do
    r=$(read_temp "$n"); gpu=$(echo "$r" | awk '{print $1+0}'); board=$(echo "$r" | awk '{print $2+0}')
    nmax=$gpu; [ "$board" -gt "$nmax" ] 2>/dev/null && nmax=$board
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),.$n,$gpu,$board,$nmax" >> "$LOG" 2>/dev/null || true
    line="$line .$n=${nmax}C"
    if [ "$nmax" -gt "$gmax" ] 2>/dev/null; then gmax=$nmax; hottest=$n; fi
  done
  echo "[$(date -u +%H:%M:%S)] hottest .$hottest=${gmax}C (margin $((CRASH_TEMP-gmax))C to crash) |$line"
  if [ "$gmax" -ge "$PULL_OFF" ] 2>/dev/null; then
    consec=$((consec + 1))
    if [ "$consec" -ge "$PERSIST" ]; then
      graceful_pulloff "$hottest" "$gmax"; exit 0
    fi
    echo "[$(date -u +%H:%M:%S)] SPIKE .$hottest=${gmax}C >= ${PULL_OFF}C (${consec}/${PERSIST} consecutive — transient unless it persists)"
  else
    consec=0
  fi
  sleep "$INTERVAL"
done
