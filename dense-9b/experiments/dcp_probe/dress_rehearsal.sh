#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Full dress rehearsal of the DCP checkpoint CYCLE using the gloo toy (no 27B):
#   toy-save into the REAL checkpoint layout -> checkpoint_sync collect -> WIPE nodes
#   (simulate reboot) -> checkpoint_sync scatter -> toy-load -> verify all ranks load.
# Proves the Mira collect/scatter orchestration end-to-end.
set -u
MASTER=${SPARK_MASTER}
NODES=($SPARK_MGMT_IPS)
OUT=${SPARK_HOME}/training_outputs/cpt_27b_full_ft
STEP=1; CK="checkpoint-$STEP"
DCPDIR="$OUT/$CK/dcp"
PY=${SPARK_HOME}/dcp_toy_ml.py
TR=${SPARK_HOME}/.local/bin/torchrun
SYNC=${REPO_ROOT}/dense-9b/recipes/checkpoint_sync.sh
S(){ ssh -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vaE "ControlSocket|mux_client"; }

launch(){ local mode=$1
  for i in "${!NODES[@]}"; do
    S "${NODES[$i]}" "cd ${SPARK_HOME} && GLOO_SOCKET_IFNAME=enP7s7 $TR --nnodes=4 --node_rank=$i \
      --master_addr=$MASTER --master_port=29688 --nproc_per_node=1 $PY $mode $DCPDIR > /tmp/dr_$mode.log 2>&1" &
  done; wait
}

echo "=== [1] clean + toy-SAVE into real layout ($DCPDIR) ==="
for n in "${NODES[@]}"; do S "$n" "rm -rf $OUT/$CK; mkdir -p $DCPDIR"; done
launch save
# mimic the trainer's rank0 side-writes (trainer_meta.pt + atomic COMPLETE)
S "$MASTER" "python3 -c \"import torch; torch.save({'format':'dcp_v1','step':$STEP,'epoch':0,'data_pos':$STEP,'sched':None,'rng':None}, '$OUT/$CK/trainer_meta.pt')\"; echo 'step=$STEP epoch=0 data_pos=$STEP' > $OUT/$CK/COMPLETE"
echo "  save done; per-node dcp contents:"
for i in "${!NODES[@]}"; do echo -n "   .${NODES[$i]##*.} rank$i: "; S "${NODES[$i]}" "ls $DCPDIR/ $DCPDIR/.metadata 2>/dev/null | tr '\n' ' '"; echo; done

echo "=== [2] checkpoint_sync COLLECT -> Mira ==="
bash "$SYNC" collect "$STEP"

echo "=== [3] WIPE all node-local checkpoints (simulate the reboot wipe) ==="
for n in "${NODES[@]}"; do S "$n" "rm -rf $OUT/$CK"; done
echo "  wiped $OUT/$CK on all 4 nodes (nodes now have NOTHING locally)"

echo "=== [4] checkpoint_sync SCATTER Mira -> nodes ==="
bash "$SYNC" scatter "$STEP"

echo "=== [5] toy-LOAD from scattered node-local checkpoints ==="
launch load
echo "  load results:"
for i in "${!NODES[@]}"; do echo -n "   .${NODES[$i]##*.} rank$i: "; S "${NODES[$i]}" "grep -aE 'LOAD rank' /tmp/dr_load.log | tail -1"; done
echo "=== DRESS REHEARSAL DONE ==="
