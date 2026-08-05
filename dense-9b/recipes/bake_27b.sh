#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# bake_27b.sh — production Artifact-B exporter for a use_collectives=False per-rank resume checkpoint.
# EXPORT_DCP reuses the trainer's proven 4-rank FSDP2 build + dcp.load path, then writes a coordinated
# model-only DCP with global metadata. BAKE_TO_HF remains only as an explicitly named legacy mode.
#
# PRE-CONDITION: all 4 nodes freshly rebooted + pristine (same as a training launch). This script
# does NOT reboot. Deploy the trainer first (scp the edited train_fsdp_dense_9b.py to all 4 nodes).
#
# Required env:
#   RESUME_DELTA=<abs path to checkpoint-NNN>   the DCP checkpoint to bake (has dcp/ + COMPLETE)
#   BAKE_TO_HF=<abs out dir on .68>             where the servable HF model dir is written (rank0)
# Optional: CPT_DATA (a corpus path the trainer opens during setup — a tiny/any packed jsonl is fine),
#   MAX_SEQ, BATCH_SIZE_PER_RANK, CPT_PACKED, OUTPUT_DIR, CLOCK_CAP.
set -euo pipefail

: "${SPARK_HOME:?fleet.env must define SPARK_HOME}"
: "${SPARK_MASTER:?fleet.env must define SPARK_MASTER}"
: "${SPARK_MGMT_IPS:?fleet.env must define SPARK_MGMT_IPS}"
: "${SPARK_RAIL_MASTER:?fleet.env must define SPARK_RAIL_MASTER}"
: "${SPARK_USER:?fleet.env must define SPARK_USER}"
: "${NCCL_IB_HCA:?run manifest must define NCCL_IB_HCA}"
: "${NCCL_NET_GDR_LEVEL:?run manifest must define NCCL_NET_GDR_LEVEL}"
: "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:?run manifest must define TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"

MASTER=${SPARK_MASTER}
WORKERS=(${SPARK_MGMT_IPS#* })
ALL=("$MASTER" "${WORKERS[@]}")

: "${RESUME_DELTA:?set RESUME_DELTA to the checkpoint dir to bake}"
# Mode: EXACTLY ONE of BAKE_TO_HF (legacy full-gather bake) or EXPORT_DCP (Artifact-B coordinated
# export, the production path). BAKE_TO_HF writes HF on rank0; EXPORT_DCP writes per-node sharded DCP.
if { [ -z "${BAKE_TO_HF:-}" ] && [ -z "${EXPORT_DCP:-}" ]; } ||
   { [ -n "${BAKE_TO_HF:-}" ] && [ -n "${EXPORT_DCP:-}" ]; }; then
  echo "ERROR: set exactly one of BAKE_TO_HF=<hf-dir> (legacy) or EXPORT_DCP=<dcp-dir> (production)" >&2
  exit 1
fi
: "${BAKE_TO_HF:=}"
: "${CPT_DATA:=/var/spark/isma/training/comprehensive_v1_packed_2560.jsonl}"
: "${MAX_SEQ:=2560}"; : "${BATCH_SIZE_PER_RANK:=1}"; : "${CPT_PACKED:=1}"
: "${CHECKPOINT_DCP:=1}"; : "${TOTAL_STEPS:=1}"; : "${SESSION_LIMIT:=1}"; : "${SAVE_EVERY:=100000}"
: "${OUTPUT_DIR:=${SPARK_HOME}/training_outputs/base_27b_full_fc}"
: "${MODEL_PATH:=${BASE_MODEL:-${SPARK_HOME}/models/Qwen3.6-27B}}"
: "${CLOCK_CAP:=${SPARK_CLOCK_CAP:-2000}}"

RUN_ENV="BAKE_TO_HF=$BAKE_TO_HF RESUME_DELTA=$RESUME_DELTA OUTPUT_DIR=$OUTPUT_DIR \
CPT_DATA=$CPT_DATA MAX_SEQ=$MAX_SEQ BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK CPT_PACKED=$CPT_PACKED \
CHECKPOINT_DCP=$CHECKPOINT_DCP TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT SAVE_EVERY=$SAVE_EVERY \
MODEL_PATH=$MODEL_PATH"
# EXPORT_DCP: Artifact-B model-only coordinated (gloo) sharded export — the production bake path
# (BAKE_ARCHITECTURE_27b.md). No full-state gather → the wedge class cannot occur. Convert offline
# with bake_dcp_offline.py. Set EXPORT_DCP=<dir> + RESUME_DELTA=<ckpt> (mutually exclusive w/ BAKE_TO_HF).
# TOPOLOGY MUST BE FORWARDED — same defect and same reason as run_4node_27b_cpt.sh. fleet.env is
# sourced HERE on Mira but is gitignored and absent from the nodes, while
# launch_cpt_qwen36_27b_fsdp.sh:399 dereferences SPARK_RAIL_MASTER under `set -u`. RUN_ENV is an
# allowlist, so an unnamed var never reaches the node. Fixed in both launchers together: they are
# two consumers of one requirement, and fixing only the one that happened to fail would leave the
# bake path to fail identically the next time it runs.
RUN_ENV="$RUN_ENV SPARK_RAIL_MASTER=$SPARK_RAIL_MASTER"
# SPARK_HOME: same defect as the training launcher, second file. RUN_ENV is an ALLOWLIST and
# launch_cpt_qwen36_27b_fsdp.sh:13 dereferences ${SPARK_HOME} under set -u, so without this the
# bake dies instantly on every rank AFTER printing "launched bake rank N" four times. Fixed in
# run_4node_27b_cpt.sh 2026-07-28; this file has its OWN RUN_ENV and was missed.
RUN_ENV="$RUN_ENV SPARK_HOME=$SPARK_HOME"
RUN_ENV="$RUN_ENV NCCL_IB_HCA=$NCCL_IB_HCA NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
[ -n "${EXPORT_DCP:-}" ] && RUN_ENV="$RUN_ENV EXPORT_DCP=$EXPORT_DCP"
# Q2 Flight-Recorder discriminator env (both lanes): a wedge on ANY collective now aborts-with-stack
# + dumps per-rank traces instead of the 1-hour silent sit. Harmless on the healthy gloo export path.
RUN_ENV="$RUN_ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_TRACE_CPP_STACK=1 TORCH_NCCL_DESYNC_DEBUG=1"
[ -n "${NCCL_DEBUG:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG=$NCCL_DEBUG"
# Deployment recipe root: honor REMOTE_PALIOS_TRAINING_ROOT the same way
# export_exact_checkpoint_artifact_b.sh does for inspection. Hardcoding
# ${SPARK_HOME}/palios-training forced a dirty/live checkout path on nodes.
REMOTE_PALIOS_TRAINING_ROOT="${REMOTE_PALIOS_TRAINING_ROOT:-${SPARK_HOME}/palios-training}"
case "$REMOTE_PALIOS_TRAINING_ROOT" in
  /*) ;;
  *)
    echo "ERROR: REMOTE_PALIOS_TRAINING_ROOT must be an absolute path (got: $REMOTE_PALIOS_TRAINING_ROOT)" >&2
    exit 1
    ;;
esac
if [ "$REMOTE_PALIOS_TRAINING_ROOT" = "/" ]; then
  echo "ERROR: REMOTE_PALIOS_TRAINING_ROOT must not be '/'" >&2
  exit 1
fi
RECIPE_DIR="${REMOTE_PALIOS_TRAINING_ROOT%/}/dense-9b/recipes"
LOGDIR=${SPARK_HOME}/cpt27b_logs
EXPORT_FREE_MARGIN_BYTES=${EXPORT_FREE_MARGIN_BYTES:-10737418240}

if [ -n "${EXPORT_DCP:-}" ]; then
  case "$EXPORT_DCP" in
    "${SPARK_HOME}"/exports/*_artifactB) ;;
    *) echo "ERROR: EXPORT_DCP must be ${SPARK_HOME}/exports/*_artifactB" >&2; exit 1;;
  esac
  echo "=== EXPORT_DCP disk preflight ==="
  existing=0
  for i in 0 1 2 3; do
    host=${ALL[$i]}
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${host}" "test -e '$EXPORT_DCP'"; then
      existing=$((existing + 1))
      ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${host}" \
        "test ! -f '$EXPORT_DCP/READY.rank${i}'" || {
        echo "ERROR: refusing to reset an export with READY.rank${i} on $host" >&2
        exit 1
      }
    fi
  done
  if [ "$existing" -gt 0 ]; then
    if [ "${RESET_INCOMPLETE_EXPORT:-0}" != 1 ]; then
      echo "ERROR: incomplete EXPORT_DCP exists on $existing node(s); set RESET_INCOMPLETE_EXPORT=1 only after inspecting it" >&2
      exit 1
    fi
    for host in "${ALL[@]}"; do
      ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${host}" \
        "rm -rf -- '$EXPORT_DCP'"
    done
    echo "  reset known-incomplete Artifact B on $existing node(s)"
  fi
  for i in 0 1 2 3; do
    host=${ALL[$i]}
    read -r shard_bytes available_bytes < <(
      ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${host}" \
        "printf '%s %s\n' \"\$(stat -c %s '$RESUME_DELTA/dcp/__${i}_0.distcp')\" \"\$(df -B1 --output=avail '$(dirname "$EXPORT_DCP")' | tail -1 | tr -d ' ')\""
    )
    required_bytes=$((shard_bytes + EXPORT_FREE_MARGIN_BYTES))
    echo "  .$host rank$i free=$available_bytes required=$required_bytes shard=$shard_bytes"
    if [ "$available_bytes" -lt "$required_bytes" ]; then
      echo "ERROR: rank$i has insufficient disk for Artifact B ($available_bytes < $required_bytes)" >&2
      exit 1
    fi
  done
fi

launch_rank () {
  local host=$1 rank=$2
  ssh -o ConnectTimeout=8 "${SPARK_USER}@${host}" \
    "mkdir -p $LOGDIR; cd $RECIPE_DIR && tmux new-session -d -s cpt27b \
       \"$RUN_ENV ./launch_cpt_qwen36_27b_fsdp.sh $rank > $LOGDIR/bake_r$rank.log 2>&1\"" \
    && echo "  launched bake rank $rank on $host"
}

echo "=== 27B BAKE launch $(date -u +%H:%M:%S) — $RESUME_DELTA → $BAKE_TO_HF ==="
CLOCK_CAP="${CLOCK_CAP:-2000}"
if [ "$CLOCK_CAP" != "0" ]; then
  for h in "${ALL[@]}"; do
    ssh -o ConnectTimeout=6 "${SPARK_USER}@${h}" "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP >/dev/null 2>&1" \
      && echo "    .${h##*.} capped @${CLOCK_CAP}" || echo "    .${h##*.} cap FAILED"
  done
fi

echo "  master $MASTER = rank 0 (binds :29500 first)"
launch_rank "$MASTER" 0
sleep 12
for i in 0 1 2; do launch_rank "${WORKERS[$i]}" $((i+1)); done

echo ""
# MATCH BOTH MODES. The loop previously matched only "BAKE COMPLETE", so an EXPORT_DCP run --
# which emits "EXPORT_DCP COMPLETE" (train_fsdp_dense_9b.py:2244) -- could never satisfy it and
# spun the full 60x30s = 30 MINUTES after a SUCCESSFUL export before falling through. A monitor
# that cannot recognise its own success condition reports every success as a timeout.
echo "=== waiting for completion on master (BAKE COMPLETE or EXPORT_DCP COMPLETE) ==="
complete=0
for t in $(seq 1 60); do   # ~30 min @ 30s
  st=$(ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" \
       'grep -aE "BAKE COMPLETE|EXPORT_DCP COMPLETE|EXPORT_DCP: |BAKE: save_pretrained|BAKE: gathering|OOM|out of memory|No space left|unexpected pos|CheckpointException|ChildFailedError|Traceback|Error|RuntimeError" '"$LOGDIR"'/bake_r0.log 2>/dev/null | tail -1' 2>/dev/null)
  echo "[$t] ${st:0:130}"
  case "$st" in
    *"BAKE COMPLETE"*) echo ">>> BAKE DONE → $BAKE_TO_HF"; complete=1; break;;
    *"EXPORT_DCP COMPLETE"*) echo ">>> EXPORT DONE → ${EXPORT_DCP:-?}"; echo "    $st"; complete=1; break;;
    *OOM*|*"out of memory"*|*"No space left"*|*"unexpected pos"*|*CheckpointException*|*ChildFailedError*|*Traceback*|*RuntimeError*)
      echo ">>> FAILURE in bake log" >&2
      exit 1
      ;;
  esac
  alive=$(ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" \
    'tmux has-session -t cpt27b 2>/dev/null && echo 1 || echo 0' 2>/dev/null)
  if [ "$alive" = 0 ]; then
    echo ">>> FAILURE: rank0 session exited without a completion marker" >&2
    ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" "tail -n 80 '$LOGDIR/bake_r0.log'" >&2
    exit 1
  fi
  sleep 30
done
[ "$complete" = 1 ] || {
  echo ">>> FAILURE: bake/export monitor timed out without completion" >&2
  exit 1
}

if [ -n "${EXPORT_DCP:-}" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${MASTER}" "test -f '$EXPORT_DCP/.metadata'"
  for i in 0 1 2 3; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${SPARK_USER}@${ALL[$i]}" \
      "test -f '$EXPORT_DCP/READY.rank${i}' && test -f '$EXPORT_DCP/manifest.rank${i}.json'"
  done
  echo "=== Artifact B acceptance: global .metadata + 4/4 READY + 4/4 manifests ==="
fi
echo "=== bake monitor end $(date -u +%H:%M:%S) ==="
