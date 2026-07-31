#!/usr/bin/env bash
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${RUN_TAG:?set RUN_TAG to the completed CPT run tag}"
: "${SFT_CORPUS:?set SFT_CORPUS to the absolute sanctioned corpus path on every Spark}"
: "${EXPECTED_SFT_SAMPLES:?set EXPECTED_SFT_SAMPLES from sft_dataset_receipt.py}"
: "${MAX_SEQ:?set MAX_SEQ explicitly for production SFT}"
BATCH_SIZE_PER_RANK=${BATCH_SIZE_PER_RANK:-1}
case "$BATCH_SIZE_PER_RANK:$MAX_SEQ:$EXPECTED_SFT_SAMPLES" in
  *[!0-9:]*|*::*|:*|*:)
    echo "REFUSE: batch, max sequence, and expected samples must be positive integers." >&2
    exit 1
    ;;
esac
[ "$BATCH_SIZE_PER_RANK" -gt 0 ] &&
[ "$MAX_SEQ" -gt 256 ] &&
[ "$EXPECTED_SFT_SAMPLES" -gt 0 ] || {
  echo "REFUSE: invalid production SFT sizing contract." >&2
  exit 1
}

case "$RUN_TAG:$SFT_CORPUS" in
  *[!A-Za-z0-9._-]*:*)
    echo "REFUSE: RUN_TAG contains unsafe characters: $RUN_TAG" >&2
    exit 1
    ;;
  *:/*) ;;
  *)
    echo "REFUSE: SFT_CORPUS must be an absolute path on every Spark." >&2
    exit 1
    ;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: production SFT requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}

BASE_MODEL="${SPARK_HOME%/}/models/${RUN_TAG}_servable"
OUTPUT_DIR=${OUTPUT_DIR:-${SPARK_HOME%/}/training_outputs/${RUN_TAG}_stage2_all_rows}
DEPLOY_REF=${DEPLOY_SHA:-HEAD}
DEPLOY_SHA=$(git rev-parse --verify "${DEPLOY_REF}^{commit}")
FRESH_UPTIME_MAX=${FRESH_UPTIME_MAX:-180}
MEM_AVAILABLE_MIN_BYTES=${MEM_AVAILABLE_MIN_BYTES:-100000000000}
DISK_AVAILABLE_MIN_BYTES=${DISK_AVAILABLE_MIN_BYTES:-42949672960}
STEP10_TIMEOUT_SECONDS=${STEP10_TIMEOUT_SECONDS:-3600}
LOGDIR="${SPARK_HOME%/}/cpt27b_logs"
RUNTIME_FILES=(
  dense-9b/trainers/train_fsdp_dense_9b.py
  dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh
  dense-9b/configs/fsdp_dense_27b.yaml
)
case "$DEPLOY_SHA" in
  *[!0-9a-f]*|'') echo "REFUSE: deployment ref did not resolve to a commit SHA." >&2; exit 1;;
esac
[ "${#DEPLOY_SHA}" = 40 ] || {
  echo "REFUSE: deployment ref did not resolve to a full 40-character commit SHA." >&2
  exit 1
}

RESUME_STEP=0
if [ -n "${RESUME_DELTA:-}" ]; then
  case "$RESUME_DELTA" in
    "$OUTPUT_DIR"/checkpoint-[0-9]*)
      RESUME_STEP=${RESUME_DELTA##*-}
      ;;
    *)
      echo "REFUSE: RESUME_DELTA must be a checkpoint under $OUTPUT_DIR." >&2
      exit 1
      ;;
  esac
fi
REQUIRE_STEP10_GATE=${REQUIRE_STEP10_GATE:-$([ "$RESUME_STEP" = 0 ] && echo 1 || echo 0)}
case "$REQUIRE_STEP10_GATE:${SKIP_BASE_STAGE:-0}" in
  [01]:[01]) ;;
  *)
    echo "REFUSE: REQUIRE_STEP10_GATE and SKIP_BASE_STAGE must be 0 or 1." >&2
    exit 1
    ;;
esac

if [ "${SKIP_BASE_STAGE:-0}" = 1 ]; then
  echo "=== PRODUCTION SFT 0/5 — base transfer skipped by the session driver; four-rank verification remains mandatory ==="
else
  echo "=== PRODUCTION SFT 0/5 — stage the exact baked base on all four Sparks ==="
  RUN_TAG=$RUN_TAG bash careers-qwen/stage_sft_base.sh
fi

echo
echo "=== PRODUCTION SFT 1/5 — refuse to reboot over a live trainer ==="
declare -a BOOT_IDS
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "trainers=\$(ps -eo args= | awk '/[t]orchrun|[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}');
     printf '%s %s\\n' \"\$(cat /proc/sys/kernel/random/boot_id)\" \"\$trainers\"")
  read -r boot_before trainers <<<"$receipt"
  BOOT_IDS[$rank]=$boot_before
  [ "$trainers" = 0 ] || {
    echo "REFUSE: rank$rank .$node has $trainers trainer process(es); no reboot was issued." >&2
    exit 1
  }
  echo "  rank$rank .$node trainers=0 boot=${BOOT_IDS[$rank]}"
done

echo
echo "=== PRODUCTION SFT 2/5 — reboot all four and prove clean new boots ==="
for node in "${NODES[@]}"; do
  (
    timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 spark@"$node" \
      "sudo systemctl reboot" >/dev/null 2>&1 || true
  ) &
done
wait

ready=0
for attempt in $(seq 1 60); do
  ready=0
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=4 spark@"$node" \
      "printf '%s %s\\n' \"\$(cat /proc/sys/kernel/random/boot_id)\" \"\$(cut -d. -f1 /proc/uptime)\"" \
      2>/dev/null || true)
    read -r boot_id uptime_seconds <<<"$receipt"
    if [ -n "${boot_id:-}" ] &&
       [ "$boot_id" != "${BOOT_IDS[$rank]}" ] &&
       [ "${uptime_seconds:-999999}" -lt "$FRESH_UPTIME_MAX" ] 2>/dev/null; then
      ready=$((ready + 1))
    fi
  done
  [ "$ready" = 4 ] && break
  if [ $((attempt % 3)) = 0 ]; then
    echo "  waiting for clean reboot receipts: $ready/4 ready"
  fi
  sleep 10
done
[ "$ready" = 4 ] || {
  echo "ABORT: not all four Sparks proved changed boot IDs and low uptime within 10 minutes." >&2
  exit 1
}

CORPUS_RECEIPT=
BASE_MANIFEST_SHA=
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$SFT_CORPUS" "$BASE_MODEL" "${RESUME_DELTA:-}" "$rank" "$OUTPUT_DIR")" <<'REMOTE'
set -euo pipefail
corpus=$1
base=$2
resume=$3
rank=$4
output=$5
uptime_seconds=$(cut -d. -f1 /proc/uptime)
trainers=$(ps -eo args= | awk '/[t]orchrun|[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}')
nccl_shm=$(find /dev/shm -maxdepth 1 -name 'nccl*' -print | wc -l)
mem_available=$(awk '/MemAvailable:/{print $2*1024}' /proc/meminfo)
disk_available=$(df -B1 --output=avail "$base" | tail -1 | tr -d ' ')
test -f "$corpus"
test -f "$base/GRAFT_COMPLETE"
test -f "$base/weight_diff.json"
test -f "$base/training_provenance.json"
test -f "$base/SOURCE_SHA256SUMS"
if [ -n "$resume" ]; then
  test -f "$resume/COMPLETE"
  test -f "$resume/dcp/__${rank}_0.distcp"
else
  existing=
  if [ -d "$output" ]; then
    existing=$(find "$output" -mindepth 1 -print -quit)
  fi
  [ -z "$existing" ] || {
    echo "refusing fresh launch over non-empty output directory $output" >&2
    exit 1
  }
fi
corpus_bytes=$(stat -c %s "$corpus")
corpus_sha=$(sha256sum "$corpus" | cut -d' ' -f1)
corpus_rows=$(wc -l <"$corpus")
base_manifest=$(sha256sum "$base/SOURCE_SHA256SUMS" | cut -d' ' -f1)
printf '%s %s %s %s %s %s %s %s %s\n' \
  "$uptime_seconds" "$trainers" "$nccl_shm" "$mem_available" "$disk_available" \
  "$corpus_bytes" "$corpus_sha" "$corpus_rows" "$base_manifest"
REMOTE
)
  read -r uptime_seconds trainers nccl_shm mem_available disk_available \
    corpus_bytes corpus_sha corpus_rows base_manifest <<<"$receipt"
  [ "$uptime_seconds" -lt "$FRESH_UPTIME_MAX" ]
  [ "$trainers" = 0 ]
  [ "$nccl_shm" = 0 ]
  [ "$mem_available" -ge "$MEM_AVAILABLE_MIN_BYTES" ] || {
    echo "ABORT: rank$rank .$node has only $mem_available bytes available after reboot." >&2
    exit 1
  }
  [ "$disk_available" -ge "$DISK_AVAILABLE_MIN_BYTES" ] || {
    echo "ABORT: rank$rank .$node has only $disk_available bytes free for SFT outputs." >&2
    exit 1
  }
  if [ "$rank" = 0 ]; then
    CORPUS_RECEIPT="$corpus_bytes $corpus_sha $corpus_rows"
    BASE_MANIFEST_SHA=$base_manifest
  else
    [ "$corpus_bytes $corpus_sha $corpus_rows" = "$CORPUS_RECEIPT" ] || {
      echo "ABORT: rank$rank corpus receipt differs from rank0." >&2
      exit 1
    }
    [ "$base_manifest" = "$BASE_MANIFEST_SHA" ] || {
      echo "ABORT: rank$rank base manifest differs from rank0." >&2
      exit 1
    }
  fi
  echo "  rank$rank .$node uptime=${uptime_seconds}s trainers=0 nccl_shm=0 mem=$mem_available disk=$disk_available"
done

echo
echo "=== PRODUCTION SFT 3/5 — deploy immutable training-runtime bytes from $DEPLOY_SHA ==="
runtime_stage=$(mktemp -d)
cleanup_runtime_stage(){
  rm -rf -- "$runtime_stage"
}
trap cleanup_runtime_stage EXIT
for file in "${RUNTIME_FILES[@]}"; do
  mkdir -p "$runtime_stage/$(dirname "$file")"
  git show "$DEPLOY_SHA:$file" >"$runtime_stage/$file"
done
for node in "${NODES[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "mkdir -p '$SPARK_HOME/palios-training/dense-9b/trainers' \
              '$SPARK_HOME/palios-training/dense-9b/recipes' \
              '$SPARK_HOME/palios-training/dense-9b/configs'"
  for file in "${RUNTIME_FILES[@]}"; do
    scp -q -o BatchMode=yes -o ConnectTimeout=10 \
      "$runtime_stage/$file" "spark@$node:$SPARK_HOME/palios-training/$file"
    local_sha=$(sha256sum "$runtime_stage/$file" | cut -d' ' -f1)
    remote_sha=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "sha256sum '$SPARK_HOME/palios-training/$file' | cut -d' ' -f1")
    [ "$local_sha" = "$remote_sha" ] || {
      echo "ABORT: runtime hash mismatch for $file on rank .$node." >&2
      exit 1
    }
  done
  echo "  .$node runtime byte-exact"
done

echo
echo "=== PRODUCTION SFT 4/5 — launch the canonical four-rank recipe ==="
export BASE_MODEL SFT_CORPUS
export BATCH_SIZE_PER_RANK MAX_SEQ EXPECTED_SFT_SAMPLES
export LR=${LR:-1e-4}
read -r sft_bytes sft_sha sft_rows <<<"$CORPUS_RECEIPT"
export TOTAL_STEPS=$(( (EXPECTED_SFT_SAMPLES + BATCH_SIZE_PER_RANK * 4 - 1) / (BATCH_SIZE_PER_RANK * 4) ))
export WARMUP_STEPS=$(( TOTAL_STEPS / 10 ))
export SESSION_LIMIT=${SESSION_LIMIT:-250}
export SAVE_EVERY=${SAVE_EVERY:-$SESSION_LIMIT}
export OUTPUT_DIR
bash careers-qwen/launch_stage2_sft.sh

echo
echo "=== PRODUCTION SFT 5/5 — verify live env and watch the step-10 dose gauge ==="
deadline=$((SECONDS + STEP10_TIMEOUT_SECONDS))
env_verified=0
while [ "$SECONDS" -lt "$deadline" ]; do
  alive=0
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    trainers=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "ps -eo args= | awk '/[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}'" 2>/dev/null || true)
    [ "${trainers:-0}" -gt 0 ] 2>/dev/null && alive=$((alive + 1))
  done
  latest=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
    "grep -aE '\\[step [0-9]+\\]|\\[SR-DELTA\\]' '$LOGDIR/r0.log' 2>/dev/null | tail -3" \
    2>/dev/null || true)
  failure=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
    "grep -aE 'Traceback|CUDA out of memory|RuntimeError|ERROR:|ABORT:' '$LOGDIR/r0.log' 2>/dev/null | tail -1" \
    2>/dev/null || true)
  printf '[%(%Y-%m-%dT%H:%M:%SZ)T] alive=%s/4 %s\n' -1 "$alive" \
    "$(tail -1 <<<"$latest" | tr '\n' ' ')"
  if [ -n "$failure" ]; then
    echo "ABORT: failure signal in the rank-0 production log:" >&2
    printf '%s\n' "$failure" >&2
    exit 1
  fi
  [ "$alive" = 4 ] || {
    echo "ABORT: only $alive/4 ranks have a live trainer before the step-10 gate." >&2
    exit 1
  }
  if [ "$env_verified" = 0 ]; then
    for rank in 0 1 2 3; do
      node=${NODES[$rank]}
      live_env=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
        "pid=\$(ps -eo pid=,args= | awk '/[t]rain_fsdp_dense_9b.py/{print \$1; exit}');
         tr '\\0' '\\n' < /proc/\$pid/environ")
      for expected in \
        "LORA_MODE=1" \
        "LR=$LR" \
        "MODEL_PATH=$BASE_MODEL" \
        "SFT_JSONL=$SFT_CORPUS" \
        "MAX_SEQ=$MAX_SEQ" \
        "EXPECTED_SFT_SAMPLES=$EXPECTED_SFT_SAMPLES" \
        "BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK" \
        "CPT_PACKED=0" \
        "EPOCHS=1" \
        "EXACT_SFT_EPOCH=0" \
        "RESUME_MODEL_ONLY=0" \
        "CHECKPOINT_DCP=1" \
        "LORA_R=16" \
        "LORA_ALPHA=32" \
        "LORA_DROPOUT=0.05" \
        "LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj" \
        "TOTAL_STEPS=$TOTAL_STEPS" \
        "WARMUP_STEPS=$WARMUP_STEPS" \
        "SESSION_LIMIT=$SESSION_LIMIT" \
        "SAVE_EVERY=$SAVE_EVERY" \
        "OUTPUT_DIR=$OUTPUT_DIR" \
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8" \
        "PYTORCH_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8" \
        "NCCL_NET_PLUGIN=none" \
        "NCCL_IB_DISABLE=0" \
        "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0" \
        "NCCL_IB_MERGE_NICS=0" \
        "NCCL_CROSS_NIC=1" \
        "NCCL_SOCKET_IFNAME=enp1s0f0np0" \
        "GLOO_SOCKET_IFNAME=enp1s0f0np0" \
        "NCCL_NET_GDR_C2C=0" \
        "NCCL_NET_GDR_LEVEL=LOC" \
        "NCCL_DMABUF_ENABLE=0" \
        "NCCL_LOCAL_REGISTER=0" \
        "NCCL_GRAPH_REGISTER=0" \
        "NCCL_WIN_ENABLE=0" \
        "NCCL_NVLS_ENABLE=0" \
        "NCCL_CUMEM_HOST_ENABLE=0" \
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120" \
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" \
        "NCCL_DEBUG=WARN" \
        "NCCL_DEBUG_SUBSYS=INIT" \
        "TORCH_NCCL_TRACE_BUFFER_SIZE=20000"; do
        grep -qxF "$expected" <<<"$live_env" || {
          echo "ABORT: rank$rank .$node live environment lacks: $expected" >&2
          exit 1
        }
      done
      if [ -n "${RESUME_DELTA:-}" ]; then
        grep -qxF "RESUME_DELTA=$RESUME_DELTA" <<<"$live_env" || {
          echo "ABORT: rank$rank .$node live resume path differs from $RESUME_DELTA." >&2
          exit 1
        }
      fi
    done
    env_verified=1
    echo "  live env exact on all four: LoRA, base, corpus, schedule, DCP, allocator, dual-rail NCCL"
    if [ "$REQUIRE_STEP10_GATE" = 0 ]; then
      echo "  resumed session is live at checkpoint-$RESUME_STEP; step-10 dose was gated in session 1"
      exit 0
    fi
  fi
  step_seen=$(grep -aoE '\[step [0-9]+\]' <<<"$latest" | tail -1 | tr -cd '0-9')
  if [ "${step_seen:-0}" -ge 10 ] 2>/dev/null; then
    dose=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
      "grep -aE '\\[SR-DELTA\\].*mean\\|dW\\|' '$LOGDIR/r0.log' | tail -1")
    [ -n "$dose" ] || {
      echo "ABORT: optimizer reached step $step_seen without the required SR-DELTA gauge." >&2
      exit 1
    }
    echo "STEP-10 PRODUCTION GATE"
    echo "  $dose"
    echo "  rank0 log: spark@$SPARK_MASTER:$LOGDIR/r0.log"
    exit 0
  fi
  sleep 20
done

echo "ABORT: production SFT did not reach the step-10 gate within ${STEP10_TIMEOUT_SECONDS}s." >&2
exit 1
