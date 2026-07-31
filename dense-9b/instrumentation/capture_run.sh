#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# capture_run.sh — full instrumented 27B run for DEATH-SIGNATURE CAPTURE.
# Reboot all 4 (discipline) -> deploy instrumentation -> start fsync telemetry + crash-survival on
# each -> start Mira-side UPS+netconsole capture -> launch the 27B training. When a node dies at
# ~22-28min, the capture dir holds the correlated evidence (node power/thermal/mem, UPS watts,
# kernel printk) to classify POWER-CUT vs THERMAL vs KERNEL.
#
# Run from Mira. Nodes: .68(rank0/master) .80 .12 .19. Usage: capture_run.sh
set -u
MIRA_IP=${ORCHESTRATOR_IP}
NODES=($SPARK_MGMT_IPS)
REPO=${REPO_ROOT}
INSTR=$REPO/dense-9b/instrumentation
S() { ssh -o ConnectTimeout=6 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vE "ControlSocket|mux_client"; }

echo "=== [1/5] REBOOT all 4 (Jesse discipline: fresh GPUs every run) ==="
for n in "${NODES[@]}"; do S "$n" 'sudo reboot' >/dev/null 2>&1 & done
sleep 45
echo "  waiting for all 4 to come back (uptime<180s + ssh ok)..."
for n in "${NODES[@]}"; do
  for i in $(seq 1 40); do
    up=$(S "$n" 'cut -d. -f1 /proc/uptime' 2>/dev/null | tr -dc 0-9)
    if [ -n "$up" ] && [ "$up" -lt 300 ] 2>/dev/null; then echo "  .$n up (${up}s)"; break; fi
    sleep 6
  done
done

echo "=== [2/5] DEPLOY instrumentation to all 4 ==="
for n in "${NODES[@]}"; do
  S "$n" 'mkdir -p $HOME/instr' >/dev/null
  scp -q -o ControlMaster=no -o ControlPath=none \
    "$INSTR/node_telemetry.py" "$INSTR/setup_crash_survival.sh" "$INSTR/start_node_telemetry.sh" \
    spark@"$n":${SPARK_HOME}/instr/ 2>&1 | grep -vE "ControlSocket|mux" | head
  S "$n" 'chmod +x $HOME/instr/*.sh $HOME/instr/*.py'
done

echo "=== [3/5] START telemetry + crash-survival + THERMAL CLOCK CAP on each node ==="
# THERMAL FIX (2026-07-10): death is a ~94C board/SoC thermal shutdown. GB10 has no power-limit
# (-pl N/A) but supports graphics-clock lock (-lgc). Cap max graphics clock to hold the board < ~90C.
# CLOCK_CAP_MAX default 2000MHz (down from the ~2398 it ran at 94C). Raise later to reclaim throughput
# once a stable 2h run is proven. Set CLOCK_CAP_MAX=0 to disable.
CLOCK_CAP_MAX="${CLOCK_CAP_MAX:-2000}"
for n in "${NODES[@]}"; do
  echo "--- .$n ---"; S "$n" "bash ${SPARK_HOME}/instr/start_node_telemetry.sh $MIRA_IP"
  if [ "$CLOCK_CAP_MAX" != "0" ]; then
    S "$n" "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP_MAX 2>&1 | head -1"
  fi
done
echo "  clock cap: graphics <= ${CLOCK_CAP_MAX}MHz (thermal fix)"

echo "=== [4/5] START Mira-side capture (UPS + netconsole) ==="
pkill -f mira_capture.sh 2>/dev/null; pkill -f 'UDP-RECV:6666' 2>/dev/null; sleep 1
nohup bash "$INSTR/mira_capture.sh" >/tmp/mira_capture.out 2>&1 &
sleep 3
CAPDIR=$(cat /tmp/mira_capture_dir 2>/dev/null)
echo "  capture dir: $CAPDIR"
echo "  UPS sample: $(tail -1 "$CAPDIR/ups.csv" 2>/dev/null)"

echo "=== [5/5] LAUNCH 27B training (instrumented) ==="
echo "  -> $REPO/dense-9b/recipes/run_4node_27b_cpt.sh"
echo "  (monitor: node telem = ${SPARK_HOME}/telemetry/telem_*.csv ; Mira = $CAPDIR/{ups.csv,netconsole.log})"
bash "$REPO/dense-9b/recipes/run_4node_27b_cpt.sh"
