#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# start_node_telemetry.sh — run ON each Spark node (post-reboot, before training) to start the
# fsync gauge logger + arm crash-survival. Idempotent: kills any prior logger first.
# Usage: start_node_telemetry.sh <MIRA_IP>
set -u
MIRA_IP="${1:-${ORCHESTRATOR_IP}}"
HERE="$(dirname "$(readlink -f "$0")")"
mkdir -p ${SPARK_HOME}/telemetry

# 1. (re)start the 1Hz fsync telemetry logger.
#    setsid (not just nohup): detaches into a NEW session so systemd doesn't kill it as a
#    session-scope process when this SSH connection closes (nohup alone does NOT survive that).
pkill -f node_telemetry.py 2>/dev/null; sleep 1
setsid python3 "$HERE/node_telemetry.py" --interval 1 </dev/null >${SPARK_HOME}/telemetry/logger.out 2>&1 &
sleep 2
if pgrep -f node_telemetry.py >/dev/null; then
  echo "[node] telemetry logger up (pid $(pgrep -f node_telemetry.py | head -1)) -> $(ls -t ${SPARK_HOME}/telemetry/telem_*.csv 2>/dev/null | head -1)"
else
  echo "[node] ERROR: telemetry logger failed to start"; tail -3 ${SPARK_HOME}/telemetry/logger.out 2>/dev/null
fi

# 2. arm crash-survival (panic sysctls + netconsole to Mira) — non-fatal if it partially fails
if [ -x "$HERE/setup_crash_survival.sh" ]; then
  sudo "$HERE/setup_crash_survival.sh" "$MIRA_IP" 2>&1 | sed 's/^/[node] /'
fi
echo "[node] $(hostname) instrumentation armed."
