#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Watch-only monitor for an already-running 27B CPT run. Does NOT launch or reboot.
# Success signal = the literal "FIRST STEP:" log line (trainer:1410), which prints ONLY
# after the first optimizer step COMPLETES. (The earlier monitor bug matched "grads="
# inside "keep_low_precision_grads=True" — fixed: match "FIRST STEP:" only.)
set -uo pipefail
MASTER=${SPARK_MASTER}
ALL=($SPARK_MGMT_IPS)
LOGDIR=${SPARK_HOME}/cpt27b_logs
for t in $(seq 1 120); do          # ~60 min @ 30s
  line=""; alive=0
  for h in "${ALL[@]}"; do
    r=$(ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$h" 'free -g|awk "/Mem:/{print \$3}"' 2>/dev/null)
    if [ -n "$r" ]; then alive=$((alive+1)); line="$line ${h##*.}=${r}G"; fi
  done
  st=$(ssh -o ConnectTimeout=5 spark@"$MASTER" \
       'grep -aE "FIRST STEP:|\[step [0-9]|evicted|Starting: steps|OOM|out of memory|Killed|Traceback" '"$LOGDIR"'/r0.log 2>/dev/null | tail -1' 2>/dev/null)
  echo "[$t $(date -u +%H:%M:%S)] alive=$alive/4 used:$line | ${st:0:88}"
  case "$st" in
    *"FIRST STEP:"*|*"[step 1"[0-9]*)
       echo ">>> FIRST OPTIMIZER STEP COMPLETED — 27B IS TRAINING. grads breakdown:"
       ssh -o ConnectTimeout=5 spark@"$MASTER" 'grep -aE "FIRST STEP:|params=.*grads=.*optim=" '"$LOGDIR"'/r0.log | tail -2' 2>/dev/null
       break;;
    *OOM*|*"out of memory"*|*Killed*|*Traceback*)
       echo ">>> FAILURE in master log — stopping"; break;;
  esac
  if [ "$alive" -lt 4 ]; then echo ">>> A NODE WENT DARK (alive=$alive/4) — stopping"; break; fi
  sleep 30
done
echo "=== watch end $(date -u +%H:%M:%S) ==="
