#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Test use_collectives=0 (self-contained per-rank) vs =1 (default dedup) for only-local load
# on the REAL 4-node topology, WITH a replicated tensor. No scatter — pure node-local.
set -u
MASTER=${SPARK_MASTER}; NODES=($SPARK_MGMT_IPS)
PY=${SPARK_HOME}/dcp_toy_repl.py; TR=${SPARK_HOME}/.local/bin/torchrun
S(){ ssh -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vaE "ControlSocket|mux_client"; }
for n in "${NODES[@]}"; do
  scp -q -o ControlMaster=no -o ControlPath=none /tmp/claude-1000/-home-mira-palios-training/81cb4a42-0404-4c25-902d-72520dc82a7f/scratchpad/dcp_toy_repl.py spark@"$n":$PY 2>&1 | grep -vaE "ControlSocket|mux"
done
run(){ local mode=$1 uc=$2 dir=$3
  for i in "${!NODES[@]}"; do
    S "${NODES[$i]}" "cd ${SPARK_HOME} && USE_COLLECTIVES=$uc GLOO_SOCKET_IFNAME=enP7s7 $TR --nnodes=4 --node_rank=$i --master_addr=$MASTER --master_port=29699 --nproc_per_node=1 $PY $mode $dir > /tmp/repl_${mode}_${uc}.log 2>&1" &
  done; wait
  for i in "${!NODES[@]}"; do echo -n "  .${NODES[$i]##*.} "; S "${NODES[$i]}" "grep -aE 'SAVE|LOAD' /tmp/repl_${mode}_${uc}.log | tail -1"; done
}
for UC in 0 1; do
  DIR=/tmp/repl_uc$UC/dcp
  echo "############ use_collectives=$UC ############"
  for n in "${NODES[@]}"; do S "$n" "rm -rf /tmp/repl_uc$UC; mkdir -p $DIR"; done
  echo "-- SAVE (node-local, NO scatter) --"; run save $UC $DIR
  echo "-- LOAD (only-local) --"; run load $UC $DIR
done
