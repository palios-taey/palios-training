#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# 4-node DCP node-local probe orchestrator (from Mira). gloo/CPU — no GPU, no reboot.
# Phase A: each node saves its shard to a NODE-LOCAL dir. Phase B (metadata scattered): each
# node loads from its node-local dir holding ONLY its own shard + a copy of .metadata.
set -u
MASTER=${SPARK_MASTER}
NODES=($SPARK_MGMT_IPS)
DIR=/tmp/dcp_ml/dcp
PY=${SPARK_HOME}/dcp_toy_ml.py
TORCHRUN="${SPARK_HOME}/.local/bin/torchrun"
S(){ ssh -o ConnectTimeout=6 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vaE "ControlSocket|mux_client"; }

echo "=== deploy toy + clean local dirs ==="
for i in "${!NODES[@]}"; do
  scp -q -o ControlMaster=no -o ControlPath=none \
    /tmp/claude-1000/-home-mira-palios-training/81cb4a42-0404-4c25-902d-72520dc82a7f/scratchpad/dcp_toy_ml.py \
    spark@"${NODES[$i]}":$PY 2>&1 | grep -vaE "ControlSocket|mux"
  S "${NODES[$i]}" "rm -rf /tmp/dcp_ml; mkdir -p $DIR"
done

launch(){ # $1=mode
  local mode=$1
  for i in "${!NODES[@]}"; do
    S "${NODES[$i]}" "cd ${SPARK_HOME} && GLOO_SOCKET_IFNAME=enP7s7 $TORCHRUN --nnodes=4 --node_rank=$i \
      --master_addr=$MASTER --master_port=29677 --nproc_per_node=1 \
      $PY $mode $DIR > /tmp/dcp_ml_$mode.log 2>&1" &
  done
  wait
  echo "--- $mode logs ---"
  for i in "${!NODES[@]}"; do
    echo "[node ${NODES[$i]##*.} rank$i]"; S "${NODES[$i]}" "grep -aE 'SAVE|LOAD' /tmp/dcp_ml_$mode.log"
  done
}

echo "=== PHASE A: SAVE (each node writes ONLY its own shard locally) ==="
launch save

echo "=== SCATTER: copy rank0's .metadata to every worker's node-local dir ==="
# pull .metadata from master, push to the 3 workers (they only have their own .distcp)
scp -q -o ControlMaster=no -o ControlPath=none spark@$MASTER:$DIR/.metadata /tmp/dcp_ml_metadata 2>&1 | grep -vaE "ControlSocket|mux"
for n in ${SPARK_NODE1} ${SPARK_NODE2} ${SPARK_NODE3}; do
  scp -q -o ControlMaster=no -o ControlPath=none /tmp/dcp_ml_metadata spark@"$n":$DIR/.metadata 2>&1 | grep -vaE "ControlSocket|mux"
done
echo "  scattered .metadata to workers"

echo "=== PHASE B: LOAD (each node has ONLY its own shard + the scattered .metadata) ==="
launch load
echo "=== DONE ==="
