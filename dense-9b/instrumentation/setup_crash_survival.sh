#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# setup_crash_survival.sh — make the next whole-node death LEAVE EVIDENCE (run per-node, needs sudo).
#
# The nodes currently die DARK with zero trail (kernel.panic=0 = hang-forever, no netconsole).
# The 5-lane consult said the #1 move is making the next death diagnostic, because the
# candidate causes (power-cut vs kernel-panic vs thermal-cut) have OPPOSITE fixes and are
# distinguished ONLY by evidence captured at the death moment.
#
# This sets up, on the local node:
#   1. panic sysctls  — a panic/oops/lockup now AUTO-REBOOTS (panic=10s) instead of hanging
#      dark forever, and converts a silent lockup into a captured panic.
#   2. netconsole     — streams kernel printk over UDP to Mira in REAL TIME, so the last
#      kernel messages before the box goes dark are captured on Mira even though local disk
#      may not flush. THE key gasp-catcher on a machine with no BMC.
#   3. pstore check   — report whether efi-pstore/ramoops can persist a trace across cycle.
# earlyoom is handled by the launch wrapper (stop before a run, restart after), not here.
#
# Usage: sudo ./setup_crash_survival.sh <MIRA_IP> [netconsole_port]
set -u
MIRA_IP="${1:-${ORCHESTRATOR_IP}}"
NCPORT="${2:-6666}"
HOST="$(hostname)"; MYIP="$(hostname -I | awk '{print $1}')"
echo "[crash-survival] $HOST ($MYIP) -> netconsole to $MIRA_IP:$NCPORT"

# --- 1. panic sysctls (persist) ---
SYSCTL=/etc/sysctl.d/99-crash-diag.conf
sudo tee "$SYSCTL" >/dev/null <<EOF
# whole-node-death diagnostics (tutor, foundations Phase 0)
kernel.panic = 10
kernel.panic_on_oops = 1
kernel.softlockup_panic = 1
kernel.hardlockup_panic = 1
EOF
sudo sysctl -p "$SYSCTL" 2>&1 | sed 's/^/  sysctl: /'

# --- 2. netconsole to Mira (find the default-route iface, target Mira) ---
IFACE="$(ip route get "$MIRA_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
GWMAC="$(ip neigh show "$MIRA_IP" 2>/dev/null | awk '{print $5; exit}')"
# resolve Mira's L2 (same subnet -> Mira's own MAC; else gateway MAC). Try direct first.
DSTMAC="$(ip neigh show "$MIRA_IP" 2>/dev/null | awk '/lladdr/{print $5; exit}')"
if [ -z "$DSTMAC" ]; then ping -c1 -W1 "$MIRA_IP" >/dev/null 2>&1; DSTMAC="$(ip neigh show "$MIRA_IP" 2>/dev/null | awk '/lladdr/{print $5; exit}')"; fi
echo "  netconsole: iface=$IFACE dstmac=${DSTMAC:-UNKNOWN}"
if [ -n "$IFACE" ] && [ -n "$DSTMAC" ]; then
  sudo modprobe netconsole 2>/dev/null || true
  # dynamic netconsole target via configfs (survives, reconfigurable)
  CG=/sys/kernel/config/netconsole
  if [ -d "$CG" ]; then
    sudo bash -c "cd $CG && ([ -d mira ] || mkdir mira) && cd mira &&
      echo 0 > enabled 2>/dev/null;
      echo $IFACE > dev_name;
      echo $MYIP > local_ip;
      echo $NCPORT > local_port;
      echo $MIRA_IP > remote_ip;
      echo $NCPORT > remote_port;
      echo $DSTMAC > remote_mac;
      echo 1 > enabled" 2>&1 | sed 's/^/    /' || echo "    (configfs netconsole target failed — trying modprobe args)"
  fi
  # verify
  cat "$CG/mira/enabled" 2>/dev/null | sed 's/^/    netconsole enabled=/'
else
  echo "  netconsole: SKIPPED (could not resolve iface/dstmac to $MIRA_IP)"
fi

# --- 3. pstore backend report ---
echo "  pstore: mounted=$(mount | grep -c pstore) backend=$(dmesg 2>/dev/null | grep -io 'pstore:.*registered' | tail -1) entries=$(ls /sys/fs/pstore 2>/dev/null | wc -l)"
echo "  ramoops: $(lsmod | grep -q ramoops && echo loaded || echo 'not loaded (needs reserved-mem region — see DR)')"
echo "[crash-survival] $HOST done."
