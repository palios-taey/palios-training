#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Per-node LOAD profiler (diagnosis, OBSERVE-ONLY — does NOT pull off; the thermal
# watchdog owns safety). Jesse 2026-07-16: the crash "keeps moving" node-to-node, so it
# is NOT the hardware — it is a code-driven per-node OVERLOAD. Temp is the symptom; WATTS
# at a fixed clock is the load signal (more work at same MHz = more watts). This samples
# all 4 nodes side-by-side so the diverging (overloaded) node is visible directly.
# CSV cols: ts,node,watts,util%,clkMHz,tempC,hostUsedG,hostAvailG,gpuProcs
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
INTERVAL="${INTERVAL:-8}"
LOG="${LOG:-node_load_profile.csv}"
echo "ts,node,watts,util,clkMHz,tempC,hostUsedG,hostAvailG,gpuProcs,load1,topCpuPct,topCpuComm" > "$LOG"
echo "[profiler] logging → $LOG every ${INTERVAL}s (observe-only, no pull-off)"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  line=""
  for n in "${NODES[@]}"; do
    r=$(timeout 6 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" \
      'g=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.gr,temperature.gpu --format=csv,noheader,nounits 2>/dev/null|head -1|tr -d " "); \
       u=$(free -g|awk "/Mem:/{print \$3}"); a=$(free -g|awk "/Mem:/{print \$7}"); \
       p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -c .); \
       l=$(cut -d" " -f1 /proc/loadavg); \
       t=$(ps -eo pcpu,comm --sort=-pcpu --no-headers | head -1 | awk "{printf \"%s,%s\", \$1, \$2}"); \
       echo "$g,$u,$a,$p,$l,$t"' 2>/dev/null)
    [ -z "$r" ] && r="NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
    echo "$ts,.${n##*.},$r" >> "$LOG"
    # compact live line: node=watts@clk/util%/tempC
    w=$(echo "$r"|cut -d, -f1); util=$(echo "$r"|cut -d, -f2); clk=$(echo "$r"|cut -d, -f3); t=$(echo "$r"|cut -d, -f4)
    line="$line .${n##*.}=${w}W/${util}%/${t}C"
  done
  echo "[$ts]$line"
  sleep "$INTERVAL"
done
