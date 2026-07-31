#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Capture a training wedge across ALL FOUR ranks simultaneously, BEFORE rebooting.
#
# WHY ALL FOUR: the deadlock shape is one rank diverging from the others — rank0's stack alone
# cannot show you that. On 2026-07-21 py-spy was installed on .68 only, so a live wedge could be
# traced on rank0 but not compared against its peers. Binary is now on all 4.
#
# WHY BEFORE REBOOTING: a reboot destroys the only evidence. The standing rule is always-reboot to
# recover — capture first, then reboot.
#
# ALSO CAPTURED: the progress-vs-spin discriminators. High GPU utilisation is NOT a progress
# signal — a spinning collective busy-waits and reports ~100% util while doing no work. The honest
# signals are (a) does the step counter advance, (b) power draw (real 27B compute is not ~10W),
# (c) whether the NCCL watchdog fired.
#
# Usage: capture_wedge.sh [label]
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
LABEL="${1:-wedge}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$(dirname "$0")/wedge_captures/${LABEL}_${TS}"
mkdir -p "$OUT"
echo "capturing -> $OUT"

for i in "${!NODES[@]}"; do
  h="${NODES[$i]}"
  {
    echo "=== rank$i ($h) @ $(date -Is) ==="
    ssh -o ConnectTimeout=10 spark@"$h" '
      p=$(pgrep -f train_fsdp_dense_9b | sed -n 2p)
      echo "pid=$p  state=$(grep ^State /proc/$p/status 2>/dev/null | awk "{print \$2}")"
      echo "-- step counter (the ONLY real progress signal) --"
      ls -t ${SPARK_HOME}/cpt27b_logs/r*.log 2>/dev/null | head -1 | xargs -r tail -3
      echo "-- gpu (util alone proves NOTHING; watch power) --"
      nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | head -1
      echo "-- nccl watchdog fired? --"
      ls -t ${SPARK_HOME}/cpt27b_logs/r*.log 2>/dev/null | head -1 | xargs -r grep -ac "Watchdog\|heartbeat\|NCCL_TIMEOUT\|timed out"
      echo "-- py-spy dump --"
      sudo env "PATH=$PATH" timeout 90 ${SPARK_HOME}/.local/bin/py-spy dump --pid $p 2>&1 | head -60
    ' 2>&1
  } > "$OUT/rank${i}_${h}.txt" &
done
wait

echo
echo "=== MAIN THREAD per rank (compare these — a divergent rank is the deadlock) ==="
for i in "${!NODES[@]}"; do
  printf "rank%s: " "$i"
  grep -A2 'MainThread' "$OUT/rank${i}_${NODES[$i]}.txt" 2>/dev/null | sed -n '2p' | sed 's/^ *//' || echo "(no stack)"
done
echo
echo "captures in $OUT — commit them, then reboot to recover."
