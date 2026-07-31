#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# mira_capture.sh — Mira-side death-capture: UPS wall-power @1Hz + netconsole kernel-gasp listener.
# Runs on Mira (the always-up orchestrator). Timestamps (epoch) align with the per-node fsync
# telemetry so the death moment can be correlated across: node power/thermal/mem <-> UPS input
# watts <-> kernel printk. Append-only, fsync'd.
#
# Distinguishes the death mode at the moment it happens:
#   - UPS load COLLAPSES (draw -> lower) + node hw_power_brake asserts  => POWER CUT (PD/OCP)
#   - node gpu_tlimit -> 0 / mlx5 -> 105C, UPS load steady               => THERMAL cutoff
#   - netconsole captures a panic/oops trace                            => KERNEL death
#
# Usage: mira_capture.sh [capture_dir] [netconsole_port]
set -u
CAPDIR="${1:-${REPO_ROOT}/audit/capture_$(date +%Y%m%d_%H%M%S)}"
NCPORT="${2:-6666}"
mkdir -p "$CAPDIR"
echo "[mira-capture] -> $CAPDIR (UPS 1Hz + netconsole :$NCPORT)" >&2
echo "$CAPDIR" > /tmp/mira_capture_dir   # so the launcher/monitor can find it

# --- UPS 1Hz logger (fsync via python for the same power-cut-survival reason) ---
UPS_CSV="$CAPDIR/ups.csv"
python3 - "$UPS_CSV" <<'PY' &
import subprocess, time, os, sys
out=sys.argv[1]
f=open(out,"a",buffering=1)
if os.path.getsize(out)==0:
    f.write("ts_wall,ups_load_pct,ups_realpower_w,input_voltage,output_voltage,battery_charge,battery_runtime,ups_status\n")
    f.flush(); os.fsync(f.fileno())
def g(var):
    try:
        r=subprocess.run(["upsc","cyberpower@localhost",var],capture_output=True,text=True,timeout=3)
        return r.stdout.strip()
    except Exception: return ""
NOM=g("ups.realpower.nominal") or "900"
while True:
    t=int(time.time())
    load=g("ups.load")
    try: w=str(round(float(load)/100.0*float(NOM),1))
    except Exception: w=""
    row=[str(t),load,w,g("input.voltage"),g("output.voltage"),g("battery.charge"),g("battery.runtime"),g("ups.status")]
    f.write(",".join(row)+"\n"); f.flush(); os.fsync(f.fileno())
    time.sleep(1)
PY
UPS_PID=$!
echo "$UPS_PID" > "$CAPDIR/ups_logger.pid"

# --- netconsole UDP listener (captures the last kernel printk before a node goes dark) ---
NC_LOG="$CAPDIR/netconsole.log"
if command -v socat >/dev/null 2>&1; then
  socat -u UDP-RECV:$NCPORT OPEN:"$NC_LOG",creat,append & echo $! > "$CAPDIR/netconsole.pid"
elif command -v nc >/dev/null 2>&1; then
  nc -ul "$NCPORT" >> "$NC_LOG" & echo $! > "$CAPDIR/netconsole.pid"
else
  echo "[mira-capture] WARN: no socat/nc — netconsole not captured" >&2
fi

# --- live-stream each node's fsync telemetry to Mira (so the death-moment node gauges are on Mira
#     even if the node goes dark and never comes back) ---
NODES=($SPARK_MGMT_IPS)
for n in "${NODES[@]}"; do
  ( ssh -o ConnectTimeout=6 -o ServerAliveInterval=5 -o ControlMaster=no -o ControlPath=none \
      spark@"$n" 'F=$(ls -t $HOME/telemetry/telem_*.csv 2>/dev/null | head -1); [ -n "$F" ] && tail -F -n +1 "$F"' \
      >> "$CAPDIR/node_${n##*.}.csv" 2>/dev/null ) &
  echo "$!" >> "$CAPDIR/streamers.pid"
done

echo "[mira-capture] UPS pid=$UPS_PID, netconsole -> $NC_LOG, 4 node streamers live. kill via pids in $CAPDIR." >&2
echo "  capture dir: $CAPDIR"
wait
