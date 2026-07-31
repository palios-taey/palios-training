#!/usr/bin/env bash
# Bounded four-Spark DDP-LoRA qualification. It never launches the full SFT.
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] || {
  echo "REFUSE: set FLEET_ENV to the private production environment file." >&2
  exit 1
}
. "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"
: "${SPARK_RAIL_MASTER:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${POST_CPT_ARTIFACT_STORE:?fleet.env did not load}"
: "${RUN_TAG:?set RUN_TAG to the completed CPT run tag}"
: "${SFT_CORPUS:?set SFT_CORPUS to the sanctioned corpus path on every Spark}"
: "${EXPECTED_SFT_SAMPLES:?set EXPECTED_SFT_SAMPLES from sft_dataset_receipt.py}"
: "${QUAL_TAG:?set a unique QUAL_TAG for this bounded qualification}"

case "$RUN_TAG:$QUAL_TAG:$SFT_CORPUS" in
  *[!A-Za-z0-9._-]*:*:*)
    echo "REFUSE: RUN_TAG contains unsafe characters: $RUN_TAG" >&2
    exit 1
    ;;
  *:*[!A-Za-z0-9._-]*:*)
    echo "REFUSE: QUAL_TAG contains unsafe characters: $QUAL_TAG" >&2
    exit 1
    ;;
  *:*:/*) ;;
  *)
    echo "REFUSE: SFT_CORPUS must be an absolute path on every Spark." >&2
    exit 1
    ;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: DDP qualification requires four rank-ordered Sparks." >&2
  exit 1
}
SSH_OPTIONS=(-o ControlMaster=no -o ControlPath=none)

MAX_SEQ=${MAX_SEQ:-1792}
PAD_TO_MULTIPLE=${PAD_TO_MULTIPLE:-16}
SHORT_MAX=${SHORT_MAX:-160}
MID_MAX=${MID_MAX:-512}
SHORT_BATCH=${SHORT_BATCH:-8}
MID_BATCH=${MID_BATCH:-2}
LONG_BATCH=${LONG_BATCH:-1}
QUAL_STEPS_PER_BUCKET=${QUAL_STEPS_PER_BUCKET:-64}
QUAL_WARMUP_STEPS=${QUAL_WARMUP_STEPS:-2}
MIN_USEFUL_TPS=${MIN_USEFUL_TPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-96}
LR=${LR:-1e-4}
CLOCK_CAP=${CLOCK_CAP:-2000}
START_BOARD_MAX_C=${START_BOARD_MAX_C:-65}
MAX_BOARD_CELSIUS=${MAX_BOARD_CELSIUS:-90}
THERMAL_PERSIST_STEPS=${THERMAL_PERSIST_STEPS:-3}
SELECTIVE_AC_START=${SELECTIVE_AC_START:-1536}
SELECTIVE_AC_BUDGET=${SELECTIVE_AC_BUDGET:-1472}
FRESH_UPTIME_MAX=${FRESH_UPTIME_MAX:-240}
MEM_AVAILABLE_MIN_BYTES=${MEM_AVAILABLE_MIN_BYTES:-100000000000}
DISK_AVAILABLE_MIN_BYTES=${DISK_AVAILABLE_MIN_BYTES:-30000000000}
MIN_TRAIN_MEM_AVAILABLE_BYTES=${MIN_TRAIN_MEM_AVAILABLE_BYTES:-40000000000}
CACHE_RELEASE_INTERVAL_STEPS=${CACHE_RELEASE_INTERVAL_STEPS:-24}
CACHE_RELEASE_BELOW_AVAILABLE_BYTES=${CACHE_RELEASE_BELOW_AVAILABLE_BYTES:-8000000000}
MIN_AFTER_CACHE_RELEASE_BYTES=${MIN_AFTER_CACHE_RELEASE_BYTES:-40000000000}
MAX_SWAP_USED_BYTES=${MAX_SWAP_USED_BYTES:-134217728}
RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-3600}
MAX_STEP_STALL_SECONDS=${MAX_STEP_STALL_SECONDS:-180}
EXIT_GRACE_SECONDS=${EXIT_GRACE_SECONDS:-300}
BASE_MODEL="${SPARK_HOME%/}/models/${RUN_TAG}_servable"
OUTPUT_DIR="${SPARK_HOME%/}/training_outputs/${RUN_TAG}_stage2_ddp_qual_${QUAL_TAG}"
STATE_DIR=${STATE_DIR:-${POST_CPT_ARTIFACT_STORE%/}/runs/${RUN_TAG}/stage2_ddp_qual/${QUAL_TAG}}
LOGDIR="${SPARK_HOME%/}/cpt27b_logs"
SESSION=stage2-ddp

QUALIFIED_RUN_TAG=cpt_v7_eps1fix
QUALIFIED_CORPUS_SHA=cdb345826b6d6b11d7a4e25f26c0342fab720877f555b4541fbbb1740d2357b6
QUALIFIED_CORPUS_BYTES=20757633
QUALIFIED_CORPUS_ROWS=9887
QUALIFIED_BASE_MANIFEST_SHA=2406fff54148dd44d9c7a4824d43aaace0c450d3e6b174bfe1268565a9512c5d
QUALIFIED_SAMPLES=10033
QUALIFIED_INPUT_TOKENS=3888967
QUALIFIED_LABEL_TOKENS=1107992
QUALIFIED_DATASET_SHAPE_SHA=c65815f4c91d9e4e2d5e810ae3af3e8008a738bf98b8c7a77c47bec3bdc2d96a
PRODUCTION_PLAN_SHA=63780ca438a6a2e5f362b4c4a9a9b13e1ff6b030aaa0c7d1bc16f9032df35550
QUALIFIED_PLAN_STEPS=979

[ "$RUN_TAG" = "$QUALIFIED_RUN_TAG" ] || {
  echo "REFUSE: RUN_TAG=$RUN_TAG is not the production-qualified CPT base." >&2
  exit 1
}
[ "$EXPECTED_SFT_SAMPLES" = "$QUALIFIED_SAMPLES" ] || {
  echo "REFUSE: expected sample count differs from the production corpus." >&2
  exit 1
}

for numeric in "$EXPECTED_SFT_SAMPLES" "$MAX_SEQ" "$PAD_TO_MULTIPLE" \
  "$SHORT_MAX" "$MID_MAX" \
  "$SHORT_BATCH" "$MID_BATCH" "$LONG_BATCH" "$QUAL_STEPS_PER_BUCKET" \
  "$QUAL_WARMUP_STEPS" "$SAVE_EVERY" \
  "$CLOCK_CAP" "$START_BOARD_MAX_C" \
  "$MAX_BOARD_CELSIUS" "$THERMAL_PERSIST_STEPS" "$FRESH_UPTIME_MAX" \
  "$SELECTIVE_AC_START" "$SELECTIVE_AC_BUDGET" \
  "$MEM_AVAILABLE_MIN_BYTES" "$DISK_AVAILABLE_MIN_BYTES" \
  "$MIN_TRAIN_MEM_AVAILABLE_BYTES" "$CACHE_RELEASE_INTERVAL_STEPS" \
  "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" \
  "$MIN_AFTER_CACHE_RELEASE_BYTES" "$MAX_SWAP_USED_BYTES" \
  "$RUN_TIMEOUT_SECONDS" "$MAX_STEP_STALL_SECONDS" "$EXIT_GRACE_SECONDS"; do
  case "$numeric" in *[!0-9]*|'')
    echo "REFUSE: integer setting is malformed: $numeric" >&2
    exit 1
    ;;
  esac
done
[ "$EXPECTED_SFT_SAMPLES" -gt 0 ] &&
[ "$MAX_SEQ" -eq 1792 ] &&
[ "$PAD_TO_MULTIPLE" -eq 16 ] &&
[ "$SHORT_MAX" -eq 160 ] &&
[ "$MID_MAX" -eq 512 ] &&
[ "$MID_MAX" -lt "$MAX_SEQ" ] &&
[ "$SHORT_BATCH" -eq 8 ] &&
[ "$MID_BATCH" -eq 2 ] &&
[ "$LONG_BATCH" -eq 1 ] &&
[ "$QUAL_STEPS_PER_BUCKET" -eq 64 ] &&
[ "$QUAL_WARMUP_STEPS" -eq 2 ] &&
[ "$SAVE_EVERY" -eq 96 ] &&
[ "$CLOCK_CAP" -eq 2000 ] &&
[ "$START_BOARD_MAX_C" -gt 0 ] &&
[ "$START_BOARD_MAX_C" -lt "$MAX_BOARD_CELSIUS" ] &&
[ "$MAX_BOARD_CELSIUS" -ge 50 ] &&
[ "$THERMAL_PERSIST_STEPS" -gt 0 ] &&
[ "$SELECTIVE_AC_START" -eq 1536 ] &&
[ "$SELECTIVE_AC_BUDGET" -eq 1472 ] &&
[ "$MIN_TRAIN_MEM_AVAILABLE_BYTES" -eq 40000000000 ] &&
[ "$CACHE_RELEASE_INTERVAL_STEPS" -eq 24 ] &&
[ "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" -eq 8000000000 ] &&
[ "$MIN_AFTER_CACHE_RELEASE_BYTES" -eq 40000000000 ] &&
[ "$MAX_SWAP_USED_BYTES" -eq 134217728 ] &&
[ "$MAX_BOARD_CELSIUS" -lt 94 ] || {
  echo "REFUSE: invalid qualification sizing contract." >&2
  exit 1
}
awk -v value="$LR" 'BEGIN { exit !(value == 0.0001) }' || {
  echo "REFUSE: bounded qualification requires LR=1e-4." >&2
  exit 1
}
EXPECTED_QUAL_STEPS=$((QUAL_STEPS_PER_BUCKET * 3))
[ "$EXPECTED_QUAL_STEPS" -eq 192 ] || {
  echo "REFUSE: bounded qualification must contain exactly 192 steps." >&2
  exit 1
}
awk -v value="$MIN_USEFUL_TPS" 'BEGIN { exit !(value >= 1000) }' || {
  echo "REFUSE: MIN_USEFUL_TPS cannot be below Jesse's 1000 tok/s gate." >&2
  exit 1
}
[ "${GRADIENT_CHECKPOINTING:-0}" = 0 ] || {
  echo "REFUSE: global activation checkpointing cannot be combined with the selective production schedule." >&2
  exit 1
}
[ "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" -gt 0 ] &&
[ "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" -lt "$MIN_AFTER_CACHE_RELEASE_BYTES" ] &&
[ "$MIN_AFTER_CACHE_RELEASE_BYTES" -le "$MIN_TRAIN_MEM_AVAILABLE_BYTES" ] || {
  echo "REFUSE: invalid receipt-faithful system-memory policy." >&2
  exit 1
}

DEPLOY_REF=${DEPLOY_SHA:-HEAD}
DEPLOY_SHA=$(git rev-parse --verify "${DEPLOY_REF}^{commit}")
case "$DEPLOY_SHA" in *[!0-9a-f]*|'')
  echo "REFUSE: deployment ref did not resolve to a commit SHA." >&2
  exit 1
  ;;
esac
[ "${#DEPLOY_SHA}" = 40 ] || {
  echo "REFUSE: deployment SHA is not 40 characters." >&2
  exit 1
}
RUNTIME_FILES=(
  careers-qwen/qualify_stage2_sft_ddp.sh
  careers-qwen/launch_4node.sh
  careers-qwen/train_ddp_lora.py
  dense-9b/trainers/train_fsdp_dense_9b.py
)
for file in "${RUNTIME_FILES[@]}"; do
  working_blob=$(git hash-object -- "$file")
  committed_blob=$(git rev-parse "${DEPLOY_SHA}:$file")
  [ "$working_blob" = "$committed_blob" ] || {
    echo "REFUSE: production file differs from $DEPLOY_SHA: $file" >&2
    exit 1
  }
done

mkdir -p "$STATE_DIR/logs"
exec 9>"$STATE_DIR/driver.lock"
flock -n 9 || {
  echo "REFUSE: another qualification holds $STATE_DIR/driver.lock." >&2
  exit 1
}
DRIVER_LOG="$STATE_DIR/driver.log"
say(){
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

launched=0
cleanup_complete=0
emergency_cleanup_active=0
runtime_stage=

retire_measurement_outputs(){
  local requirement=${1:-required}
  local rank node retired_bytes failed=0
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    if ! retired_bytes=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes \
      -o ConnectTimeout=15 spark@"$node" \
      "$(printf '%q ' bash -s -- "${SPARK_HOME%/}/training_outputs" "$OUTPUT_DIR" "$requirement")" <<'REMOTE'
set -euo pipefail
root=$1
output=$2
requirement=$3
case "$output" in
  "$root"/*_stage2_ddp_qual_*) ;;
  *)
    echo "REFUSE: qualification output escaped the exact training-output namespace." >&2
    exit 1
    ;;
esac
if [ -d "$output" ]; then
  bytes=$(du -sb "$output" | cut -f1)
  rm -rf -- "$output"
  [ ! -e "$output" ]
else
  [ "$requirement" = optional ]
  bytes=0
fi
printf '%s\n' "$bytes"
REMOTE
    ); then
      say "ERROR: rank$rank .$node measurement-output retirement failed"
      failed=1
      continue
    fi
    case "$retired_bytes" in *[!0-9]*|'')
      say "ERROR: rank$rank .$node returned a malformed retirement receipt"
      failed=1
      continue
      ;;
    esac
    say "retired measurement output rank$rank .$node bytes=$retired_bytes"
  done
  [ "$failed" = 0 ]
}

cleanup_on_exit(){
  local rc=$?
  trap - EXIT
  set +e
  if [ -n "${runtime_stage:-}" ] && [ -d "$runtime_stage" ]; then
    rm -rf -- "$runtime_stage"
  fi
  if [ "$launched" = 1 ] &&
     [ "$cleanup_complete" != 1 ] &&
     [ "$emergency_cleanup_active" != 1 ]; then
    emergency_cleanup_active=1
    say "EMERGENCY CLEANUP: preserving logs, rebooting all four Sparks, and retiring the exact measurement namespace"
    local rank node receipt boot uptime trainers nccl_shm
    local ready attempt cleanup_failed=0
    local -a boot_ids
    boot_ids=()
    for rank in 0 1 2 3; do
      node=${NODES[$rank]}
      scp "${SSH_OPTIONS[@]}" -q -o BatchMode=yes -o ConnectTimeout=8 \
        "spark@$node:$LOGDIR/ddp_r${rank}.log" \
        "$STATE_DIR/logs/rank${rank}.emergency.log" 2>/dev/null
      boot_ids[$rank]=$(timeout -k 2 8 ssh "${SSH_OPTIONS[@]}" \
        -o BatchMode=yes -o ConnectTimeout=5 spark@"$node" \
        "cat /proc/sys/kernel/random/boot_id" 2>/dev/null)
      [ -n "${boot_ids[$rank]}" ] ||
        boot_ids[$rank]="unreachable-before-cleanup-$rank"
    done
    for node in "${NODES[@]}"; do
      (
        timeout -k 2 8 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes \
          -o ConnectTimeout=5 spark@"$node" \
          "sudo systemctl reboot" >/dev/null 2>&1 || true
      ) &
    done
    wait
    ready=0
    for attempt in $(seq 1 60); do
      ready=0
      for rank in 0 1 2 3; do
        node=${NODES[$rank]}
        receipt=$(timeout -k 2 6 ssh "${SSH_OPTIONS[@]}" \
          -o BatchMode=yes -o ConnectTimeout=4 spark@"$node" \
          "printf '%s %s\n' \
            \"\$(cat /proc/sys/kernel/random/boot_id)\" \
            \"\$(cut -d. -f1 /proc/uptime)\"" 2>/dev/null)
        read -r boot uptime <<<"$receipt"
        if [ -n "${boot:-}" ] &&
           [ "$boot" != "${boot_ids[$rank]}" ] &&
           [ "${uptime:-999999}" -lt "$FRESH_UPTIME_MAX" ] 2>/dev/null; then
          ready=$((ready + 1))
        fi
      done
      [ "$ready" = 4 ] && break
      sleep 10
    done
    if [ "$ready" != 4 ]; then
      say "ERROR: emergency cleanup proved only $ready/4 fresh reboots"
      cleanup_failed=1
    else
      for rank in 0 1 2 3; do
        node=${NODES[$rank]}
        receipt=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" \
          -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
          "printf '%s %s\n' \
            \"\$(ps -eo args= | awk \
              '/[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++} END{print n+0}')\" \
            \"\$(find /dev/shm -maxdepth 1 -name 'nccl*' -print | wc -l)\"" \
          2>/dev/null)
        read -r trainers nccl_shm <<<"$receipt"
        if [ "$trainers" != 0 ] || [ "$nccl_shm" != 0 ]; then
          say "ERROR: emergency cleanup rank$rank .$node trainers=${trainers:-unknown} nccl_shm=${nccl_shm:-unknown}"
          cleanup_failed=1
        fi
      done
    fi
    if [ "$cleanup_failed" = 0 ]; then
      printf 'mode=emergency ranks=4 trainers=0 nccl_shm=0\n' \
        >"$STATE_DIR/POST_RUN_CLEAN"
    fi
    if retire_measurement_outputs optional; then
      printf 'retired_output=%s mode=emergency\n' "$OUTPUT_DIR" \
        >"$STATE_DIR/OUTPUTS_RETIRED"
    else
      cleanup_failed=1
    fi
    if [ "$cleanup_failed" = 0 ]; then
      cleanup_complete=1
      say "EMERGENCY CLEANUP COMPLETE: all four Sparks are fresh and the measurement namespace is absent"
    else
      say "EMERGENCY CLEANUP INCOMPLETE: manual fleet recovery is required"
      [ "$rc" != 0 ] || rc=1
    fi
  fi
  exit "$rc"
}
trap cleanup_on_exit EXIT

trainer_count(){
  local node=$1
  ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "ps -eo args= | awk '
       /[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++}
       END{print n+0}'"
}

capture_boot_ids(){
  local rank node trainers session_state
  BOOT_IDS=()
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    trainers=$(trainer_count "$node")
    session_state=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "tmux has-session -t '$SESSION' 2>/dev/null && echo present || echo absent")
    [ "$trainers" = 0 ] && [ "$session_state" = absent ] || {
      echo "REFUSE: rank$rank .$node is not idle: trainers=$trainers session=$session_state" >&2
      exit 1
    }
    BOOT_IDS[$rank]=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "cat /proc/sys/kernel/random/boot_id")
    say "pre-reboot rank$rank .$node boot=${BOOT_IDS[$rank]} trainers=0 session=absent"
  done
}

reboot_and_verify(){
  local label=$1 rank node ready attempt receipt boot_id uptime
  say "$label reboot issued to all four Sparks"
  for node in "${NODES[@]}"; do
    (
      timeout -k 2 8 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=5 spark@"$node" \
        "sudo systemctl reboot" >/dev/null 2>&1 || true
    ) &
  done
  wait

  ready=0
  for attempt in $(seq 1 60); do
    ready=0
    for rank in 0 1 2 3; do
      node=${NODES[$rank]}
      receipt=$(timeout -k 2 6 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=4 spark@"$node" \
        "printf '%s %s\n' \
          \"\$(cat /proc/sys/kernel/random/boot_id)\" \
          \"\$(cut -d. -f1 /proc/uptime)\"" 2>/dev/null || true)
      read -r boot_id uptime <<<"$receipt"
      if [ -n "${boot_id:-}" ] &&
         [ "$boot_id" != "${BOOT_IDS[$rank]}" ] &&
         [ "${uptime:-999999}" -lt "$FRESH_UPTIME_MAX" ] 2>/dev/null; then
        ready=$((ready + 1))
      fi
    done
    [ "$ready" = 4 ] && break
    if [ $((attempt % 3)) = 0 ]; then
      say "$label reboot wait: $ready/4 ready"
    fi
    sleep 10
  done
  [ "$ready" = 4 ] || {
    echo "ABORT: $label reboot did not produce four changed low-uptime boot IDs." >&2
    exit 1
  }

  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "$(printf '%q ' bash -s -- "$SFT_CORPUS" "$BASE_MODEL" "$OUTPUT_DIR" "$label" "$SAVE_EVERY" "$EXPECTED_QUAL_STEPS")" <<'REMOTE'
set -euo pipefail
corpus=$1
base=$2
output=$3
label=$4
save_every=$5
expected_steps=$6
boot=$(cat /proc/sys/kernel/random/boot_id)
uptime=$(cut -d. -f1 /proc/uptime)
trainers=$(ps -eo args= | awk \
  '/[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++} END{print n+0}')
nccl_shm=$(find /dev/shm -maxdepth 1 -name 'nccl*' -print | wc -l)
mem=$(awk '/MemAvailable:/{print $2*1024}' /proc/meminfo)
disk=$(df -B1 --output=avail "$base" | tail -1 | tr -d ' ')
test -f "$corpus"
test -f "$base/GRAFT_COMPLETE"
test -f "$base/weight_diff.json"
test -f "$base/training_provenance.json"
test -f "$base/SOURCE_SHA256SUMS"
(
  cd "$base"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
if [ "$label" = pre-run ]; then
  [ ! -e "$output" ]
  checkpoint_step=none
  adapter_sha=none
else
  latest=0
  for candidate in "$output"/checkpoint-*; do
    [ -d "$candidate" ] || continue
    step=${candidate##*-}
    case "$step" in *[!0-9]*|'') continue;; esac
    [ "$step" -le "$latest" ] || latest=$step
  done
  [ "$latest" = "$expected_steps" ]
  adapter_receipts=
  for required_step in "$save_every" "$expected_steps"; do
    checkpoint="$output/checkpoint-$required_step"
    test -f "$checkpoint/COMPLETE"
    test -f "$checkpoint/adapter_model.safetensors"
    test -f "$checkpoint/adapter_config.json"
    test -f "$checkpoint/trainer_state.pt"
    test -f "$checkpoint/training_manifest.json"
    test -f "$checkpoint/SHA256SUMS.json"
    adapter_sha=$(python3 - "$checkpoint" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
checksums = json.loads(
    (checkpoint / "SHA256SUMS.json").read_text(encoding="utf-8")
)
required = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "trainer_state.pt",
    "training_manifest.json",
}
if not required.issubset(checksums):
    raise SystemExit("incomplete checkpoint checksum receipt")
for name in sorted(required):
    digest = hashlib.sha256()
    with (checkpoint / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checksums[name]:
        raise SystemExit(f"checkpoint checksum mismatch: {name}")
print(checksums["adapter_model.safetensors"])
PY
)
    adapter_receipts="${adapter_receipts}${adapter_receipts:+,}${required_step}:${adapter_sha}"
  done
  checkpoint_step=$latest
  adapter_sha=$adapter_receipts
fi
printf '%s %s %s %s %s %s %s %s %s %s %s %s\n' \
  "$boot" "$uptime" "$trainers" "$nccl_shm" "$mem" "$disk" \
  "$(stat -c %s "$corpus")" "$(wc -l <"$corpus")" \
  "$(sha256sum "$corpus" | cut -d' ' -f1)" \
  "$(sha256sum "$base/SOURCE_SHA256SUMS" | cut -d' ' -f1)" \
  "$checkpoint_step" "$adapter_sha"
REMOTE
)
    read -r boot_id uptime trainers nccl_shm mem disk \
      corpus_bytes corpus_rows corpus_sha base_manifest_sha \
      checkpoint_step adapter_sha <<<"$receipt"
    [ "$trainers" = 0 ] && [ "$nccl_shm" = 0 ]
    [ "$mem" -ge "$MEM_AVAILABLE_MIN_BYTES" ] || {
      echo "ABORT: rank$rank .$node post-reboot memory $mem is below gate." >&2
      exit 1
    }
    [ "$disk" -ge "$DISK_AVAILABLE_MIN_BYTES" ] || {
      echo "ABORT: rank$rank .$node disk $disk is below gate." >&2
      exit 1
    }
    if [ "$rank" = 0 ]; then
      CLUSTER_ARTIFACT_RECEIPT="$corpus_bytes $corpus_rows $corpus_sha $base_manifest_sha"
    else
      [ "$corpus_bytes $corpus_rows $corpus_sha $base_manifest_sha" = "$CLUSTER_ARTIFACT_RECEIPT" ] || {
        echo "ABORT: rank$rank corpus/base receipt differs from rank0." >&2
        exit 1
      }
    fi
    if [ "$label" = post-run ]; then
      if [ "$rank" = 0 ]; then
        CLUSTER_CHECKPOINT_RECEIPT="$checkpoint_step $adapter_sha"
      else
        [ "$checkpoint_step $adapter_sha" = "$CLUSTER_CHECKPOINT_RECEIPT" ] || {
          echo "ABORT: rank$rank post-run checkpoint receipt differs from rank0." >&2
          exit 1
        }
      fi
    fi
    BOOT_IDS[$rank]=$boot_id
    say "$label rank$rank .$node boot=$boot_id uptime=${uptime}s trainers=0 nccl_shm=0 mem=$mem disk=$disk corpus_rows=$corpus_rows checkpoint=$checkpoint_step adapter_sha=$adapter_sha"
  done
}

capture_boot_ids
reboot_and_verify pre-run
[ "$CLUSTER_ARTIFACT_RECEIPT" = \
  "$QUALIFIED_CORPUS_BYTES $QUALIFIED_CORPUS_ROWS $QUALIFIED_CORPUS_SHA $QUALIFIED_BASE_MANIFEST_SHA" ] || {
  echo "ABORT: cluster corpus/base receipt differs from production." >&2
  exit 1
}

say "measuring fresh board/SoC headroom before the clock decision"
for node in "${NODES[@]}"; do
  board_milli=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1")
  case "$board_milli" in *[!0-9]*|'')
    echo "ABORT: no numeric board/SoC receipt from .$node." >&2
    exit 1
    ;;
  esac
  board_c=$((board_milli / 1000))
  [ "$board_c" -le "$START_BOARD_MAX_C" ] || {
    echo "ABORT: .$node board/SoC ${board_c}C exceeds the ${START_BOARD_MAX_C}C launch gate." >&2
    exit 1
  }
  say "pre-clock .$node board=${board_c}C"
done
say "fresh thermal headroom admits the committed ${CLOCK_CAP}MHz production cap; pull-off=${MAX_BOARD_CELSIUS}C x${THERMAL_PERSIST_STEPS}"

say "deploying immutable runtime bytes from $DEPLOY_SHA"
runtime_stage=$(mktemp -d)
for file in "${RUNTIME_FILES[@]}"; do
  mkdir -p "$runtime_stage/$(dirname "$file")"
  git show "$DEPLOY_SHA:$file" >"$runtime_stage/$file"
done
for node in "${NODES[@]}"; do
  for file in "${RUNTIME_FILES[@]}"; do
    ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "mkdir -p '$SPARK_HOME/palios-training/$(dirname "$file")'"
    scp "${SSH_OPTIONS[@]}" -q -o BatchMode=yes -o ConnectTimeout=10 \
      "$runtime_stage/$file" "spark@$node:$SPARK_HOME/palios-training/$file"
    local_sha=$(sha256sum "$runtime_stage/$file" | cut -d' ' -f1)
    remote_sha=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "sha256sum '$SPARK_HOME/palios-training/$file' | cut -d' ' -f1")
    [ "$local_sha" = "$remote_sha" ] || {
      echo "ABORT: runtime hash mismatch on .$node for $file." >&2
      exit 1
    }
  done
  say "runtime byte-exact on .$node"
done

say "pinning every Spark to ${CLOCK_CAP}MHz and running BF16 GEMM peer gate"
GEMM_TFLOPS=()
for node in "${NODES[@]}"; do
  ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "sudo nvidia-smi -pm 1 >/dev/null 2>&1;
     sudo nvidia-smi -lgc 0,'$CLOCK_CAP' >/dev/null 2>&1"
  result=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "PYTHONWARNINGS=ignore python3 - <<'PY'
import time
import torch
n = 8192
a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
started = time.perf_counter()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
seconds = (time.perf_counter() - started) / 20
print(f'{2*n**3/seconds/1e12:.3f}')
PY")
  case "$result" in *[!0-9.]*|'')
    echo "ABORT: invalid GEMM receipt from .$node: $result" >&2
    exit 1
    ;;
  esac
  GEMM_TFLOPS+=("$result")
  say "GEMM .$node ${result}TFLOPS"
done
readarray -t SORTED_TFLOPS < <(printf '%s\n' "${GEMM_TFLOPS[@]}" | sort -n)
GEMM_MEDIAN=$(awk -v a="${SORTED_TFLOPS[1]}" -v b="${SORTED_TFLOPS[2]}" \
  'BEGIN { printf "%.3f", (a+b)/2 }')
for rank in 0 1 2 3; do
  awk -v value="${GEMM_TFLOPS[$rank]}" -v median="$GEMM_MEDIAN" \
    'BEGIN { exit !(value >= median*0.80) }' || {
    echo "ABORT: rank$rank GEMM ${GEMM_TFLOPS[$rank]} is below 0.80x median $GEMM_MEDIAN." >&2
    exit 1
  }
done

TRAIN_ARGS=(
  --model "$BASE_MODEL"
  --data "$SFT_CORPUS"
  --canonical-trainer "$SPARK_HOME/palios-training/dense-9b/trainers/train_fsdp_dense_9b.py"
  --out "$OUTPUT_DIR"
  --expected-samples "$EXPECTED_SFT_SAMPLES"
  --max-seq "$MAX_SEQ"
  --pad-to-multiple "$PAD_TO_MULTIPLE"
  --short-max "$SHORT_MAX"
  --mid-max "$MID_MAX"
  --short-batch "$SHORT_BATCH"
  --mid-batch "$MID_BATCH"
  --long-batch "$LONG_BATCH"
  --qualification-steps-per-bucket "$QUAL_STEPS_PER_BUCKET"
  --qualification-warmup-steps "$QUAL_WARMUP_STEPS"
  --min-useful-tps "$MIN_USEFUL_TPS"
  --save-every "$SAVE_EVERY"
  --min-mem-available-bytes "$MIN_TRAIN_MEM_AVAILABLE_BYTES"
  --cache-release-interval-steps "$CACHE_RELEASE_INTERVAL_STEPS"
  --cache-release-below-available-bytes "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES"
  --min-after-cache-release-bytes "$MIN_AFTER_CACHE_RELEASE_BYTES"
  --max-swap-used-bytes "$MAX_SWAP_USED_BYTES"
  --max-board-celsius "$MAX_BOARD_CELSIUS"
  --thermal-persist-steps "$THERMAL_PERSIST_STEPS"
  --selective-ac-start "$SELECTIVE_AC_START"
  --selective-ac-budget "$SELECTIVE_AC_BUDGET"
  --lr "$LR"
  --log-every 1
)

launch_rank(){
  local node=$1 rank=$2
  ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$rank" "$SESSION" "$LOGDIR" "$SPARK_HOME" "$SPARK_RAIL_MASTER" "${TRAIN_ARGS[@]}")" <<'REMOTE'
set -euo pipefail
rank=$1
session=$2
logdir=$3
spark_home=$4
rail_master=$5
shift 5
mkdir -p "$logdir"
log="$logdir/ddp_r${rank}.log"
exit_file="$logdir/ddp_r${rank}.exit"
if [ -s "$log" ]; then
  mv "$log" "$logdir/ddp_r${rank}.$(date -r "$log" +%Y%m%dT%H%M%S).log"
fi
rm -f -- "$exit_file"
printf -v command '%q ' \
  env "SPARK_HOME=$spark_home" "SPARK_RAIL_MASTER=$rail_master" \
  bash "$spark_home/palios-training/careers-qwen/launch_4node.sh" "$rank" "$@"
command+=" >$(printf '%q' "$log") 2>&1; "
command+="rc=\$?; printf '%s\\n' \"\$rc\" >$(printf '%q' "$exit_file"); exit \"\$rc\""
tmux new-session -d -s "$session" "$command"
REMOTE
  say "launched rank$rank on .$node"
}

say "launching bounded 192-step DDP qualification release_interval=$CACHE_RELEASE_INTERVAL_STEPS emergency_below=$CACHE_RELEASE_BELOW_AVAILABLE_BYTES post_release_floor=$MIN_AFTER_CACHE_RELEASE_BYTES allocator_gc=0.8 python_gc=off; full SFT continuation is impossible from this script"
launched=1
launch_rank "${NODES[0]}" 0
sleep 12
launch_rank "${NODES[1]}" 1
launch_rank "${NODES[2]}" 2
launch_rank "${NODES[3]}" 3

deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
partial_since=0
last_progress_line=
last_progress_at=0
while [ "$SECONDS" -lt "$deadline" ]; do
  alive=0
  exited=0
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    state=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "if [ -f '$LOGDIR/ddp_r${rank}.exit' ]; then echo EXITED;
       elif tmux has-session -t '$SESSION' 2>/dev/null; then echo ALIVE;
       else echo MISSING; fi" 2>/dev/null || true)
    [ "$state" = ALIVE ] && alive=$((alive + 1))
    [ "$state" = EXITED ] && exited=$((exited + 1))
  done
  latest=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"${NODES[0]}" \
    "grep -aE '\\[step [0-9]+/|QUALIFICATION_|CHECKPOINT_COMPLETE|Traceback|RuntimeError|CUDA out of memory' \
       '$LOGDIR/ddp_r0.log' 2>/dev/null | tail -1" 2>/dev/null || true)
  if [ -n "$latest" ] && [ "$latest" != "$last_progress_line" ]; then
    last_progress_line=$latest
    last_progress_at=$SECONDS
  elif [ "$last_progress_at" -gt 0 ] &&
    [ $((SECONDS - last_progress_at)) -ge "$MAX_STEP_STALL_SECONDS" ]; then
    say "qualification made no logged progress for ${MAX_STEP_STALL_SECONDS}s"
    break
  fi
  say "qualification alive=$alive/4 exited=$exited/4 ${latest:-loading-model}"
  [ "$exited" = 4 ] && break
  if [ "$alive" -lt 4 ] && [ "$exited" -gt 0 ]; then
    [ "$partial_since" -ne 0 ] || partial_since=$SECONDS
    [ $((SECONDS - partial_since)) -lt "$EXIT_GRACE_SECONDS" ] || {
      say "distributed ranks did not converge to exit within ${EXIT_GRACE_SECONDS}s"
      break
    }
  else
    partial_since=0
  fi
  sleep 20
done

EXIT_CODES=()
all_exited=1
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  code=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "cat '$LOGDIR/ddp_r${rank}.exit' 2>/dev/null || true")
  case "$code" in
    *[!0-9]*|'') code=missing; all_exited=0;;
  esac
  EXIT_CODES[$rank]=$code
  scp "${SSH_OPTIONS[@]}" -q -o BatchMode=yes -o ConnectTimeout=10 \
    "spark@$node:$LOGDIR/ddp_r${rank}.log" \
    "$STATE_DIR/logs/rank${rank}.log"
done

qualification_marker=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"${NODES[0]}" \
  "if [ -f '$OUTPUT_DIR/QUALIFICATION_PASSED.json' ]; then echo PASSED;
   elif [ -f '$OUTPUT_DIR/QUALIFICATION_REJECTED.json' ]; then echo REJECTED;
   else echo MISSING; fi" 2>/dev/null || true)
if [ "$qualification_marker" != MISSING ]; then
  scp "${SSH_OPTIONS[@]}" -q -o BatchMode=yes -o ConnectTimeout=10 \
    "spark@${NODES[0]}:$OUTPUT_DIR/QUALIFICATION_${qualification_marker}.json" \
    "$STATE_DIR/"
fi
say "pre-clean verdict marker=$qualification_marker exits=${EXIT_CODES[*]}"

for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  BOOT_IDS[$rank]=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "cat /proc/sys/kernel/random/boot_id" 2>/dev/null || printf 'unreachable-rank%s' "$rank")
done
reboot_and_verify post-run
printf 'checkpoint_receipt=%s\n' \
  "${CLUSTER_CHECKPOINT_RECEIPT:-none none}" >"$STATE_DIR/POST_RUN_CLEAN"

[ "$all_exited" = 1 ] || {
  echo "REJECTED: not every rank wrote an exit receipt; post-run reboot completed." >&2
  exit 1
}
[ "${CLUSTER_CHECKPOINT_RECEIPT:-none none}" != "none none" ] || {
  echo "REJECTED: no verified checkpoint receipt after post-run reboot." >&2
  exit 1
}

case "$qualification_marker:${EXIT_CODES[*]}" in
  "PASSED:0 0 0 0")
    verdict_path="$STATE_DIR/QUALIFICATION_PASSED.json"
    ;;
  "REJECTED:1 1 1 1")
    verdict_path="$STATE_DIR/QUALIFICATION_REJECTED.json"
    ;;
  *)
    echo "REJECTED: marker=$qualification_marker exits=${EXIT_CODES[*]} is not a coherent terminal verdict." >&2
    exit 1
    ;;
esac
[ -s "$verdict_path" ] || {
  echo "REJECTED: terminal verdict was not retained off-cluster." >&2
  exit 1
}

if [ "$qualification_marker" = PASSED ]; then
python3 - "$verdict_path" \
  "$PRODUCTION_PLAN_SHA" "$QUALIFIED_SAMPLES" "$QUALIFIED_INPUT_TOKENS" \
  "$QUALIFIED_LABEL_TOKENS" "$QUALIFIED_DATASET_SHAPE_SHA" \
  "$EXPECTED_QUAL_STEPS" "$MIN_USEFUL_TPS" \
  "$CACHE_RELEASE_INTERVAL_STEPS" "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" \
  "$MIN_AFTER_CACHE_RELEASE_BYTES" "$MAX_SWAP_USED_BYTES" <<'PY'
import json
import math
from pathlib import Path
import sys

(
    path,
    production_plan_sha,
    samples,
    input_tokens,
    label_tokens,
    dataset_shape_sha,
    expected_steps,
    minimum_tps,
    release_interval,
    no_release_floor,
    post_release_floor,
    maximum_swap,
) = sys.argv[1:]
value = json.loads(Path(path).read_text(encoding="utf-8"))
dataset = value.get("dataset_receipt", {})
expected_dataset = {
    "samples": int(samples),
    "input_tokens": int(input_tokens),
    "label_tokens": int(label_tokens),
    "shape_sha256": dataset_shape_sha,
}
if {key: dataset.get(key) for key in expected_dataset} != expected_dataset:
    raise SystemExit("qualification dataset receipt differs from production")
if value.get("production_plan_sha256") != production_plan_sha:
    raise SystemExit("qualification production plan hash differs")
if value.get("completed_steps") != int(expected_steps):
    raise SystemExit("qualification did not complete 192 steps")
checkpoints = value.get("checkpoint_receipts", [])
if [receipt.get("step") for receipt in checkpoints] != [96, 192]:
    raise SystemExit("qualification checkpoint cadence differs")
for receipt in checkpoints:
    fingerprint = receipt.get("lora_fingerprint", {})
    if (
        fingerprint.get("tensors") != 704
        or fingerprint.get("bytes") != 400556032
        or fingerprint.get("finite") is not True
    ):
        raise SystemExit("qualification full-adapter receipt differs")
if value.get("final_lora_fingerprint") != checkpoints[-1].get(
    "lora_fingerprint"
):
    raise SystemExit("qualification final adapter receipt differs")
delta = value.get("final_lora_delta", {})
if (
    delta.get("tensors") != 704
    or delta.get("elements") != 100139008
    or delta.get("finite") is not True
    or delta.get("changed_elements", 0) <= 0
    or delta.get("changed_fraction", 0) <= 0
    or delta.get("mean_abs", 0) <= 0
):
    raise SystemExit("qualification full-adapter delta receipt is incomplete")
if value.get("reclaim_python_gc") is not False:
    raise SystemExit("qualification unexpectedly ran Python cyclic GC")
topology = value.get("topology", {})
expected_topology = {
    "world_size": 4,
    "max_seq": 1792,
    "pad_to_multiple": 16,
    "thresholds": [160, 512],
    "local_batches": [8, 2, 1],
    "schedule_steps": 979,
}
if {key: topology.get(key) for key in expected_topology} != expected_topology:
    raise SystemExit("qualification topology differs from production")
activation = topology.get("activation_checkpointing", {})
if (
    activation.get("selective_start_padded_tokens") != 1536
    or activation.get("selective_activation_budget_tokens") != 1472
):
    raise SystemExit("qualification activation schedule differs")
memory = topology.get("memory", {})
expected_memory = {
    "cache_release_interval_steps": int(release_interval),
    "cache_release_below_available_bytes": int(no_release_floor),
    "minimum_after_cache_release_bytes": int(post_release_floor),
    "maximum_swap_used_bytes": int(maximum_swap),
    "reclaim_python_gc": False,
    "pass_fail_signal": "system_MemAvailable",
    "cuda_free": "telemetry_only",
}
if {key: memory.get(key) for key in expected_memory} != expected_memory:
    raise SystemExit("qualification memory policy differs")
allocator = topology.get("allocator_configuration", {})
expected_allocator = (
    "expandable_segments:False,garbage_collection_threshold:0.8"
)
if (
    allocator.get("PYTORCH_CUDA_ALLOC_CONF") != expected_allocator
    or allocator.get("PYTORCH_ALLOC_CONF") != expected_allocator
):
    raise SystemExit("qualification allocator policy differs")
thermal = value.get("thermal", {})
if (
    thermal.get("observations") != int(expected_steps) * 4
    or thermal.get("pull_off_celsius") != 90
    or thermal.get("maximum_board_celsius", 999) >= 90
    or thermal.get("minimum_graphics_clock_mhz", 0) < 1900
    or thermal.get("maximum_graphics_clock_mhz", 9999) > 2050
):
    raise SystemExit("qualification board/clock receipt differs")
plan = value.get("qualification_plan", {})
buckets = plan.get("buckets", {})
for bucket, partial_steps in (("short", 1), ("mid", 0), ("long", 1)):
    receipt = buckets.get(bucket, {})
    if (
        receipt.get("steps") != 64
        or receipt.get("partial_steps") != partial_steps
        or receipt.get("late_maximum_steps", 0) < 2
    ):
        raise SystemExit(f"qualification coverage differs for {bucket}")
expected_transitions = {
    f"{source}->{target}"
    for source in ("short", "mid", "long")
    for target in ("short", "mid", "long")
    if source != target
}
if set(plan.get("transitions", {})) != expected_transitions:
    raise SystemExit("qualification transition receipt is incomplete")
minimum_tps = float(minimum_tps)
lower = value.get("one_sided_throughput_lower_bounds", {})
lower_metrics = (
    "measured_useful_input_tok_s",
    "projected_full_pass_useful_input_tok_s",
)
if any(
    isinstance(lower.get(name), bool)
    or not isinstance(lower.get(name), (int, float))
    or not math.isfinite(lower[name])
    or lower[name] <= 0
    for name in lower_metrics
):
    raise SystemExit("qualification throughput uncertainty receipt is incomplete")
if value.get("throughput_contract") != {
    "hard_gate": "point_estimates_at_or_above_minimum",
    "bootstrap_lower_bounds": "diagnostic_only",
}:
    raise SystemExit("qualification throughput contract differs")
point_estimates_passed = (
    value.get("measured_useful_input_tok_s", 0) >= minimum_tps
    and value.get("projected_full_pass_useful_input_tok_s", 0) >= minimum_tps
)
if not point_estimates_passed:
    raise SystemExit("qualification throughput receipt is below gate")
bootstrap_lower_bounds_passed = all(
    lower[name] >= minimum_tps for name in lower_metrics
)
gates = value.get("gates", {})
if not all(
    gates.get(name) is True
    for name in (
        "measured_and_projected_throughput",
        "point_estimates_at_or_above_minimum",
        "receipt_faithful_system_memory",
        "zero_swap_growth",
        "zero_allocator_retry_growth",
        "zero_oom_growth",
    )
):
    raise SystemExit("qualification terminal gates are incomplete")
if (
    gates.get("system_no_release_floor_bytes") != int(no_release_floor)
    or gates.get("system_post_release_floor_bytes") != int(post_release_floor)
    or gates.get("maximum_swap_used_bytes") != int(maximum_swap)
    or gates.get("cuda_free_hard_gate") is not False
):
    raise SystemExit("qualification terminal memory thresholds differ")
uma = value.get("uma", {})
if (
    uma.get("minimum_system_available_after_bytes", 0)
    < int(no_release_floor)
    or uma.get("minimum_system_available_after_reclaim_bytes", 0)
    < int(post_release_floor)
    or uma.get("maximum_swap_used_bytes", int(maximum_swap) + 1)
    > int(maximum_swap)
    or uma.get("maximum_swap_growth_bytes") != 0
    or uma.get("maximum_allocator_retry_growth") != 0
    or uma.get("maximum_oom_growth") != 0
    or uma.get("memory_guard_exit_rank_events") != 0
):
    raise SystemExit("qualification UMA receipt violates the production gate")
baselines = value.get("uma_baseline", [])
if (
    len(baselines) != 4
    or any(
        receipt.get("swap_used_bytes", int(maximum_swap) + 1)
        > int(maximum_swap)
        for receipt in baselines
    )
):
    raise SystemExit("qualification swap baseline receipt differs")
projection = value.get("session_memory_projection", {})
if projection.get("hard_gate") != "none_diagnostic_only":
    raise SystemExit("qualification memory projection is not diagnostic-only")
if (
    gates.get("one_sided_95pct_lower_bounds_at_or_above_minimum")
    is not bootstrap_lower_bounds_passed
):
    raise SystemExit("qualification throughput uncertainty gate is inconsistent")
PY
fi

retire_measurement_outputs required
printf 'retired_output=%s\n' "$OUTPUT_DIR" >"$STATE_DIR/OUTPUTS_RETIRED"
cleanup_complete=1

result=$(cat "$verdict_path")
if [ "$qualification_marker" = REJECTED ]; then
  say "QUALIFICATION REJECTED: $result"
  say "cluster post-run clean and measurement outputs retired; full SFT was not launched"
  exit 2
fi
say "QUALIFICATION PASSED: $result"
say "cluster post-run clean and measurement outputs retired; full SFT was not launched"
