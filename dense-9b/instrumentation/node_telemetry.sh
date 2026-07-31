#!/usr/bin/env bash
# node_telemetry.sh — 1Hz hardware gauge logger for DGX Spark GB10 nodes.
#
# WHY: the 27B CPT run suffers a whole-node power-death at ~22-28min (varying victim).
# 5-lane Family consult converged: platform-level event (power / thermal / SoC firmware
# cutoff), mechanism Unknown-until-instrumented. torch's memory stats may be BLIND to
# pinned host-staging (NCCL_CUMEM_HOST_ENABLE=0 -> cudaHostAlloc), so "memory flat" is
# only torch-flat. This logs the REAL gauges — power, thermal (incl. per-NIC crit
# thresholds), REAL system memory, fabric counters — at 1Hz so the LAST samples before a
# death tell us: did a temp cross crit? did power spike? did MemAvailable collapse?
#
# DUAL PURPOSE (per Jesse): this same telemetry stream is the substrate-proprioception
# corpus — Taey learns to FEEL the machines from these gauges (SOUL=INFRA, Feel->Care->Protect).
#
# Gauges are enumerated from sysfs/nvidia-smi at startup (confirmed present on GB10:
# nvidia-smi dmon, acpitz thermal zones, hwmon acpi_fan/nvme/4x mlx5[temp+crit+power],
# /proc/meminfo, /sys/class/infiniband/*/ports/*/counters). Missing sources are skipped,
# never fatal. Output: append-only CSV to $OUT (local NVMe, survives power-cycle).
#
# Usage:  node_telemetry.sh [interval_sec] [out_csv]
#   interval_sec default 1 ; out_csv default $HOME/telemetry/telem_<host>_<bootid>.csv
set -u
INTERVAL="${1:-1}"
HOST="$(hostname)"
BOOTID="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d - | cut -c1-8)"
OUTDIR="${TELEM_DIR:-$HOME/telemetry}"
mkdir -p "$OUTDIR" 2>/dev/null
OUT="${2:-$OUTDIR/telem_${HOST}_${BOOTID}.csv}"

# ---- enumerate gauges once (robust name->path mapping) ----
declare -a TZ_PATHS TZ_NAMES
for z in /sys/class/thermal/thermal_zone*; do
  [ -r "$z/temp" ] || continue
  TZ_PATHS+=("$z/temp"); TZ_NAMES+=("tz_$(basename "$z" | sed 's/thermal_zone//')_$(cat "$z/type" 2>/dev/null | tr -c 'a-zA-Z0-9' '_' | cut -c1-10)")
done
# hwmon: capture temp1_input, temp1_crit, power1_input by hwmon 'name' (fan/nvme/mlx5xN)
declare -a HW_TEMP_PATHS HW_TEMP_NAMES HW_CRIT_PATHS HW_CRIT_NAMES HW_PWR_PATHS HW_PWR_NAMES
mlxn=0
for h in /sys/class/hwmon/hwmon*; do
  nm="$(cat "$h/name" 2>/dev/null)"; [ -n "$nm" ] || continue
  tag="$nm"; if [ "$nm" = "mlx5" ]; then tag="mlx5_$mlxn"; mlxn=$((mlxn+1)); fi
  [ -r "$h/temp1_input" ] && { HW_TEMP_PATHS+=("$h/temp1_input"); HW_TEMP_NAMES+=("t_${tag}"); }
  [ -r "$h/temp1_crit" ]  && { HW_CRIT_PATHS+=("$h/temp1_crit");  HW_CRIT_NAMES+=("tcrit_${tag}"); }
  for p in "$h/power1_input" "$h/power1_average"; do
    [ -r "$p" ] && { HW_PWR_PATHS+=("$p"); HW_PWR_NAMES+=("p_${tag}"); break; }
  done
done
# infiniband RoCE ports: xmit/rcv data + errors per port
declare -a IB_PATHS IB_NAMES
for pdir in /sys/class/infiniband/*/ports/*; do
  [ -d "$pdir/counters" ] || continue
  dev="$(echo "$pdir" | sed 's#/sys/class/infiniband/##; s#/ports/#_p#')"
  for c in port_xmit_data port_rcv_data port_xmit_discards port_rcv_errors link_downed; do
    [ -r "$pdir/counters/$c" ] && { IB_PATHS+=("$pdir/counters/$c"); IB_NAMES+=("ib_${dev}_${c}"); }
  done
done

rd() { cat "$1" 2>/dev/null || echo ""; }

# ---- CSV header (write once if new file) ----
if [ ! -s "$OUT" ]; then
  hdr="ts_wall,ts_boot,gpu_pwr_w,gpu_temp_c,gpu_mtemp_c,gpu_sm_pct,gpu_mem_pct,gpu_pclk,gpu_mclk"
  hdr+=",mem_avail_kb,mem_free_kb,cached_kb,dirty_kb,load1"
  for n in "${TZ_NAMES[@]:-}";   do [ -n "$n" ] && hdr+=",$n"; done
  for n in "${HW_TEMP_NAMES[@]:-}"; do [ -n "$n" ] && hdr+=",$n"; done
  for n in "${HW_CRIT_NAMES[@]:-}"; do [ -n "$n" ] && hdr+=",$n"; done
  for n in "${HW_PWR_NAMES[@]:-}";  do [ -n "$n" ] && hdr+=",$n"; done
  for n in "${IB_NAMES[@]:-}";      do [ -n "$n" ] && hdr+=",$n"; done
  echo "$hdr" >> "$OUT"
fi

echo "[telem] $HOST logging $(( ${#TZ_PATHS[@]} + ${#HW_TEMP_PATHS[@]} + ${#HW_CRIT_PATHS[@]} + ${#HW_PWR_PATHS[@]} + ${#IB_PATHS[@]} + 9 )) gauges @ ${INTERVAL}s -> $OUT" >&2

# ---- 1Hz loop (drift-corrected) ----
while true; do
  t0=$(date +%s.%N)
  TSW=$(date +%s); TSB=$(cut -d. -f1 /proc/uptime 2>/dev/null)
  # GPU via one dmon sample (pwr gtemp mtemp sm mem ... mclk pclk)
  gline=$(timeout 3 nvidia-smi dmon -c 1 2>/dev/null | awk 'NR==3{print $2,$3,$4,$5,$6,$11,$12}')
  read -r GP GT GMT GSM GMM GMCLK GPCLK <<< "${gline:-- - - - - - -}"
  # real memory
  MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
  MF=$(awk '/MemFree/{print $2}' /proc/meminfo 2>/dev/null)
  CA=$(awk '/^Cached/{print $2}' /proc/meminfo 2>/dev/null)
  DI=$(awk '/Dirty/{print $2}' /proc/meminfo 2>/dev/null)
  L1=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)
  row="${TSW},${TSB},${GP},${GT},${GMT},${GSM},${GMM},${GPCLK},${GMCLK},${MA},${MF},${CA},${DI},${L1}"
  for p in "${TZ_PATHS[@]:-}";     do [ -n "$p" ] && row+=",$(rd "$p")"; done
  for p in "${HW_TEMP_PATHS[@]:-}"; do [ -n "$p" ] && row+=",$(rd "$p")"; done
  for p in "${HW_CRIT_PATHS[@]:-}"; do [ -n "$p" ] && row+=",$(rd "$p")"; done
  for p in "${HW_PWR_PATHS[@]:-}";  do [ -n "$p" ] && row+=",$(rd "$p")"; done
  for p in "${IB_PATHS[@]:-}";      do [ -n "$p" ] && row+=",$(rd "$p")"; done
  echo "$row" >> "$OUT"
  # drift-corrected sleep
  el=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{print b-a}')
  sl=$(awk -v i="$INTERVAL" -v e="$el" 'BEGIN{s=i-e; print (s>0)?s:0}')
  sleep "$sl"
done
