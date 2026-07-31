#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# spark_parity_gate.sh — MECHANICAL GATE: the 4 Sparks are ONE machine in four boxes.
# Jesse (standing, restated 2026-07-24): "ALL THE SPARKS MUST BE THE SAME. THE WORKLOAD MUST
# BE DISTRIBUTED EVENLY IN ALL PHASES." A divergent node is tutor's defect, not an infra
# ticket — the Sparks are tutor's — and it must fail the launch, not surface at step 200.
#
# READS PRODUCTION TELEMETRY ONLY (node_telemetry.py, the 1Hz 59-gauge fsync logger).
# DO NOT use `nvidia-smi --query-gpu=utilization.gpu` or `memory.used` on GB10: unified memory
# makes both meaningless (memory.used returns [N/A]; utilization.gpu is an invalid field on this
# stack). Jesse, 2026-07-24: "96% IS NOT A REAL NUMBER. YOU CANNOT USE REGULAR SMI ON SPARKS
# BECAUSE OF UMA. WE HAVE PRODUCTION MONITORING."
#
# PRECISION (tutor-codex correction, same day — do not over-read the rule): clocks_gr here reads
# the SAME nvidia-smi field, so CLOCK readings from either source are equally valid. The ban is on
# utilization.gpu and memory.used specifically. Node .80 was observed at 611MHz on two bake-time
# point samples and 1787-1820MHz after teardown; CAUSE IS UNKNOWN. Earlier comments in this file
# blamed stale procs — that was asserted from a pgrep count that was itself a self-match artifact
# and is WITHDRAWN. Resolving it needs a time-aligned logger during real training steps.
#
# Real signals on this platform: power_draw, clocks_gr, gpu_tlimit margin, throttle-reason bits,
# and /proc/meminfo (torch- and smi-blind to pinned host staging).
#
# THIS IS NOT THE LAUNCH GATE. It cannot be. (tutor 2026-07-25, after it blessed a launch
# onto a fleet containing a 3x straggler.)
#
# The clock check below reads IDLE telemetry. At idle all four GB10s sit at exactly 208MHz,
# so the spread computes to 0% and this script prints "PARITY OK: 4 Sparks equal" on a fleet
# where one node collapses to ~611MHz / 28.5 TFLOPS the moment real work starts. It is
# STRUCTURALLY BLIND to a load-only divergence — the one failure mode we have actually hit.
# I had written "idle: all four identical at 208 MHz" in a consult packet and still let this
# gate a production launch; tutor-codex called the full stop.
#
# THE LAUNCH GATE IS THE LOADED BF16 GEMM PREFLIGHT in dense-9b/recipes/run_4node_27b_cpt.sh
# (measure_gemm_tflops + GEMM_PREFLIGHT_ONLY, Jesse's commit 2c797807). It caps clocks, runs
# 5 warmups + 20 8192^3 GEMMs per node, and refuses any rank below a ratio of the fleet
# median. Run THAT before any distributed launch:
#   SPARK_NODES="<4 hosts in rank order>" MASTER_ADDR=<rank0 fabric addr> \
#   CLOCK_CAP=1600 GEMM_PREFLIGHT_MIN_PEER_RATIO=0.95 GEMM_PREFLIGHT_ONLY=1 \
#   bash dense-9b/recipes/run_4node_27b_cpt.sh
#
# What THIS script is still good for, and all it claims: reachability, telemetry liveness and
# freshness, stale trainer procs, thermal headroom, hardware power-brake. Those are real and
# worth checking. Passing here means "nothing is obviously broken at rest" — it does NOT mean
# the ranks are performance-equal, and it must never be the last check before a launch.
#
# Exit 0 = at-rest checks pass. Exit 1 = ABORT. Neither verdict speaks to loaded performance.
# Usage: bash spark_parity_gate.sh   (call AFTER reboot + telemetry start, BEFORE the GEMM preflight)
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
CLOCK_TOL_PCT=15        # fastest-vs-slowest graphics-clock spread
POWER_TOL_PCT=40        # idle power spread (wide: idle draw is noisy)
TLIMIT_MIN=20           # thermal headroom in degrees before we refuse to start a run
FAIL=0
say(){ echo "[parity $(date -u +%H:%M:%S)] $*"; }

declare -A CLK PWR TLIM PROCS
for n in "${NODES[@]}"; do
  r=$(timeout 20 ssh -o ConnectTimeout=8 -o BatchMode=yes spark@"$n" \
      "f=\$(ls -t ${SPARK_HOME}/telemetry/telem_*.csv 2>/dev/null | head -1); \
       [ -z \"\$f\" ] && { echo NOTELEM; exit 0; }; \
       echo \"AGE \$(( \$(date +%s) - \$(stat -c %Y \"\$f\") ))\"; \
       python3 -c \"
import csv
r=list(csv.DictReader(open('\$f')))[-1]
def g(k,d='0'):
    return r.get(k,d) or d
# SCHEMA-TOLERANT (2026-07-25): the v2 logger renamed every field this gate reads.
# v1 clocks_gr / power_draw / gpu_tlimit  ->  v2 gpu_clock_graphics_mhz /
# gpu_power_draw_average_w / gpu_tlimit_margin_c. Reading a v1 name against a v2 row
# returned the '0' default and the gate REFUSED A HEALTHY NODE on 'thermal margin 0'.
# A false refusal is as harmful as a false pass: it blocks production for a fiction.
def gg(*keys, d='MISSING'):
    # NEVER return a number for an absent field. A missing gauge that reads as 0 is
    # indistinguishable from a real measurement of 0, and the two demand opposite
    # responses: 0 thermal margin means STOP, a missing field means I CANNOT TELL.
    # This gate reported 'thermal margin 0' on a healthy node for exactly that reason.
    for k in keys:
        v = r.get(k)
        if v not in (None, '', '[N/A]'):
            return v
    return d
print('|'.join([gg('gpu_clock_graphics_mhz','clocks_gr'),
      gg('gpu_power_draw_average_w','gpu_power_draw_instant_w','power_draw'),
      gg('gpu_tlimit_margin_c','gpu_tlimit'),
      gg('gpu_clock_event_hw_power_brake_slowdown_active','thr_hw_power_brake_slowdown',d='?'),
      gg('gpu_clock_event_sw_power_cap_active','thr_sw_power_cap',d='?')]))
\"; pgrep -c -f '[t]rain_fsdp_dense|[t]orchrun' 2>/dev/null || echo 0" 2>/dev/null)
  [ -z "$r" ] && { say "FAIL: $n unreachable"; FAIL=1; continue; }
  case "$r" in NOTELEM*) say "FAIL: $n has NO telemetry — run start_node_telemetry.sh first (never fall back to nvidia-smi)"; FAIL=1; continue;; esac
  age=$(echo "$r" | grep "^AGE " | awk '{print $2}')
  line=$(echo "$r" | grep -v "^AGE " | sed -n 1p); PROCS[$n]=$(echo "$r" | grep -v "^AGE " | sed -n 2p | tr -dc 0-9)
  # STALENESS (added 2026-07-24): a driver reboot kills node_telemetry.py and no driver
  # re-arms it, so this gate was reading rows HOURS old and passing on them. Fresh-or-fail:
  # old gauges are worse than none, because they look like a healthy reading.
  if [ -n "$age" ] && [ "$age" -gt 300 ] 2>/dev/null; then
    say "FAIL: $n telemetry is ${age}s STALE — re-arm start_node_telemetry.sh (a reboot kills the logger)"
    FAIL=1
  fi
  raw_clk=$(echo "$line" | cut -d'|' -f1); raw_pwr=$(echo "$line" | cut -d'|' -f2); raw_tlim=$(echo "$line" | cut -d'|' -f3)
  # A MISSING gauge is not a reading. Refuse on it explicitly rather than letting it
  # become a 0 that looks measured (tutor-codex rule, and the defect that made this
  # gate refuse a 43C node on 'margin 0').
  for _f in "$raw_clk:clock" "$raw_pwr:power" "$raw_tlim:tlimit_margin"; do
    if [ "${_f%%:*}" = "MISSING" ]; then
      say "FAIL: $n gauge '${_f##*:}' MISSING from telemetry — cannot evaluate, not treating as zero"
      FAIL=1
    fi
  done
  CLK[$n]=$(echo "$raw_clk" | tr -dc 0-9)
  PWR[$n]=$(echo "$raw_pwr" | tr -d 'a-zA-Z[]/')
  TLIM[$n]=$(echo "$raw_tlim" | tr -dc 0-9)
  brake=$(echo "$line" | cut -d'|' -f4); cap=$(echo "$line" | cut -d'|' -f5)
  say "$n clock=${CLK[$n]}MHz power=${PWR[$n]}W tlimit_margin=${TLIM[$n]} brake=$brake swcap=$cap stale_procs=${PROCS[$n]}"
  [ "$brake" = "Active" ] && { say "FAIL: $n hardware power brake ACTIVE"; FAIL=1; }
done
[ "$FAIL" = 1 ] && { say "=== PARITY GATE FAILED — DO NOT LAUNCH ==="; exit 1; }

# 1. no stale trainer procs (hygiene invariant in its own right — NOT a proven cause of
#    any clock anomaly; see the precision note above)
for n in "${NODES[@]}"; do
  [ "${PROCS[$n]:-0}" -gt 0 ] 2>/dev/null && { say "FAIL: $n has ${PROCS[$n]} stale trainer proc(s) — reboot before launch"; FAIL=1; }
done

# 2. thermal headroom on every node
for n in "${NODES[@]}"; do
  [ "${TLIM[$n]:-0}" -lt "$TLIMIT_MIN" ] 2>/dev/null && { say "FAIL: $n thermal margin ${TLIM[$n]} < $TLIMIT_MIN"; FAIL=1; }
done

# 3. CLOCK PARITY — the slowest node gates every collective in a synchronous run
mx=0; mn=999999
for n in "${NODES[@]}"; do
  c=${CLK[$n]:-0}
  [ "$c" -gt "$mx" ] 2>/dev/null && mx=$c
  [ "$c" -lt "$mn" ] 2>/dev/null && mn=$c
done
if [ "$mx" -gt 0 ] 2>/dev/null; then
  spread=$(( (mx - mn) * 100 / mx ))
  say "clock spread: min=${mn} max=${mx} => ${spread}% (tol ${CLOCK_TOL_PCT}%)"
  [ "$spread" -gt "$CLOCK_TOL_PCT" ] && { say "FAIL: clocks not in parity"; FAIL=1; }
fi

[ "$FAIL" = 1 ] && { say "=== PARITY GATE FAILED — DO NOT LAUNCH ==="; exit 1; }
say "=== PARITY OK: 4 Sparks equal, safe to launch ==="
exit 0
