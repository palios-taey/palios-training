#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# V4 validation driver — 4-node FSDP2 LoRA-SFT module-1, the pre-commit gate.
# Mirrors run_till_done_v3.sh's proven node-prep, then launches the module-1 launcher with SAVE_EVERY=30
# so the save->resume round-trip (Gaia V4 seam) is testable. Runs prep+launch ATOMICALLY (one bg unit).
# Does NOT auto-kill — tutor watches ${SPARK_HOME}/module1_dryrun.log for the V4 criteria, then decides.
set -uo pipefail
NODES=($SPARK_MGMT_IPS)
MASTER=${SPARK_MASTER}
CLOCK_CAP=1600
SRC=${REPO_ROOT}/careers-qwen
say(){ echo "[v4 $(date +%H:%M:%S)] $*"; }

say "STEP 1: reboot all 4 (clean slate — reboot-after-every-run)"
for n in "${NODES[@]}"; do timeout 8 ssh -o ConnectTimeout=5 spark@"$n" 'sudo reboot' 2>/dev/null; done
sleep 30
up=0
for t in $(seq 1 45); do
  up=0; for n in "${NODES[@]}"; do timeout 5 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" true 2>/dev/null && up=$((up+1)); done
  say "nodes back: $up/4"; [ "$up" -eq 4 ] && break; sleep 10
done
[ "$up" -eq 4 ] || { say "ABORT: only $up/4 nodes back"; exit 1; }
sleep 20  # services settle

say "STEP 2: fabric check (master pings the 3 rail IPs)"
if ! timeout 15 ssh spark@"$MASTER" 'for ip in ${SPARK_RAIL_IP} ${SPARK_RAIL_IP} ${SPARK_RAIL_IP}; do ping -c1 -W2 $ip >/dev/null 2>&1 || { echo "FAIL $ip"; exit 1; }; done; echo OK' 2>/dev/null | grep -q OK; then
  say "ABORT: fabric ping failed"; exit 1
fi
say "fabric OK"

say "STEP 3: thermal clock cap ${CLOCK_CAP}MHz on all 4 (prevents ~94C hard-crash)"
for n in "${NODES[@]}"; do
  timeout 12 ssh -o ConnectTimeout=6 spark@"$n" "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP >/dev/null 2>&1 && echo capped" 2>/dev/null | grep -q capped \
    && say ".${n##*.} capped @${CLOCK_CAP}" || { say "ABORT: .${n##*.} clock cap FAILED (thermal unsafe)"; exit 1; }
done

say "STEP 4: sync latest harness+launcher+fabric to all 4 (post-reboot, disk preserved but ensure current)"
for n in "${NODES[@]}"; do scp -q "$SRC"/train_lora_sft.py "$SRC"/launch_module1_fsdp.sh "$SRC"/fabric_env.sh spark@"$n":${SPARK_HOME}/ 2>/dev/null; done

say "STEP 5: launch V4 via detached tmux (PROVEN mechanism from run_4node_27b_cpt.sh) — rank 0 FIRST to bind :29500"
# tmux new-session -d = fully detached, survives ssh close (setsid-over-ssh was the divergence that failed rendezvous)
ssh -o ConnectTimeout=8 spark@"$MASTER" "cd ${SPARK_HOME} && tmux kill-session -t module1 2>/dev/null; tmux new-session -d -s module1 \"SAVE_EVERY=1 SESSION_SECONDS=0 EXTRA_ARGS="--max-steps 2 --pg-timeout-s 120" TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_FR_DUMP_TEMP_FILE=${SPARK_HOME}/fr_rank bash launch_module1_fsdp.sh 0 > ${SPARK_HOME}/module1_dryrun.log 2>&1\"" 2>/dev/null
say "rank 0 (master) launched in tmux — waiting 20s for it to bind the rendezvous store"
sleep 20
r=1
for n in ${SPARK_NODE1} ${SPARK_NODE2} ${SPARK_NODE3}; do
  ssh -o ConnectTimeout=8 spark@"$n" "cd ${SPARK_HOME} && tmux kill-session -t module1 2>/dev/null; tmux new-session -d -s module1 \"SAVE_EVERY=1 SESSION_SECONDS=0 EXTRA_ARGS="--max-steps 2 --pg-timeout-s 120" TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_FR_DUMP_TEMP_FILE=${SPARK_HOME}/fr_rank bash launch_module1_fsdp.sh $r > ${SPARK_HOME}/module1_dryrun.log 2>&1\"" 2>/dev/null
  say "rank $r launched in tmux on .${n##*.}"; r=$((r+1))
done
say "V4 LAUNCHED on all 4. Watch: ssh spark@$MASTER 'tail -f $HOME/module1_dryrun.log'"
say "V4 criteria: [fsdp2] param count ~100M | loss tracks smoke (6-7 region early) | step-time | no OOM | save at gstep 30"
