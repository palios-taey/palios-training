#!/usr/bin/env bash
# Production Stage-2 campaign for the throughput-qualified replicated DDP LoRA topology.
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
  echo "REFUSE: production DDP SFT requires four rank-ordered Sparks." >&2
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
FIRST_SESSION_STEPS=${FIRST_SESSION_STEPS:-50}
MAX_SESSION_STEPS=${MAX_SESSION_STEPS:-250}
MAX_SESSIONS=${MAX_SESSIONS:-8}
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
SAVE_EVERY=${SAVE_EVERY:-50}
SESSION_BASE_TIMEOUT_SECONDS=${SESSION_BASE_TIMEOUT_SECONDS:-1200}
STEP_TIMEOUT_SECONDS=${STEP_TIMEOUT_SECONDS:-10}
MAX_STEP_STALL_SECONDS=${MAX_STEP_STALL_SECONDS:-180}
EXIT_GRACE_SECONDS=${EXIT_GRACE_SECONDS:-180}
OUTPUT_DIR=${OUTPUT_DIR:-${SPARK_HOME%/}/training_outputs/${RUN_TAG}_stage2_ddp_all_rows}
STATE_DIR=${STATE_DIR:-${POST_CPT_ARTIFACT_STORE%/}/runs/${RUN_TAG}/stage2_sft_ddp}
LOGDIR="${SPARK_HOME%/}/cpt27b_logs/stage2_sft_ddp_${RUN_TAG}"
SESSION=stage2-ddp-sft
BASE_MODEL="${SPARK_HOME%/}/models/${RUN_TAG}_servable"

QUALIFIED_RUN_TAG=cpt_v7_eps1fix
# ── QUARANTINE NOTICE (2026-08-02) — READ BEFORE ACTING ON A REFUSAL BELOW ──
# QUALIFIED_CORPUS_SHA below is cdb345826b6d6b11..., which is the corpus treasurer's OPEN
# fl-cred-corpus-quarantine covers: it carries plaintext host credentials in the ASSISTANT role,
# i.e. as training TARGETS — the model is taught to PRODUCE them, not merely to read past them.
#
# CONSEQUENCE: a Stage-2 SFT DDP run against these bytes is BLOCKED BY CONSTRUCTION. The trainer's
# loader refuses this sha by CONTENT (train_fsdp_dense_9b.py, digest gate), so the run cannot
# start no matter what this qualification block says.
#
# THAT IS THE CORRECT OUTCOME, AND THE QUALIFICATION BELOW IS THE STALE HALF, NOT THE GATE.
# This notice exists because the alternative reading is dangerous and cheap: a qualification block
# carrying an authorization sha, a plan sha and a dataset-shape sha LOOKS like the authority, and
# a single loader gate looks like an obstacle. Under run pressure the tempting fix is to disable
# the quarantine or add an exception "just to unblock". That would train credentials into a
# production artifact while feeling like unsticking a stuck run.
#
# THE CORRECT PATH FORWARD, none of which is disabling anything:
#   1. treasurer sanctions replacement bytes with the credential rows resolved;
#   2. re-run qualification against those bytes to produce a NEW QUALIFIED_CORPUS_SHA;
#   3. update this block to the new sha.
# Flagged by treasurer 2026-08-02; the contradiction itself was the hazard, independent of intent.
QUALIFIED_CORPUS_SHA=cdb345826b6d6b11d7a4e25f26c0342fab720877f555b4541fbbb1740d2357b6
QUALIFIED_CORPUS_BYTES=20757633
QUALIFIED_CORPUS_ROWS=9887

# ACTIVE REFUSAL, not just the notice above. A comment informs a reader; this stops a runner.
# Self-retiring by construction: it fires only while QUALIFIED_CORPUS_SHA names quarantined bytes,
# so when treasurer sanctions replacements and this pin is updated to the new sha, the check
# passes with no edit here. Nothing to remember to remove, and no exception flag to reach for --
# an override env var would be the escape hatch this exists to prevent.
_QUARANTINED_SHAS="cdb345826b6d6b11d7a4e25f26c0342fab720877f555b4541fbbb1740d2357b6"
for _q in $_QUARANTINED_SHAS; do
  if [ "$QUALIFIED_CORPUS_SHA" = "$_q" ]; then
    echo "REFUSE: QUALIFIED_CORPUS_SHA names a QUARANTINED corpus (${_q:0:16}...)." >&2
    echo "        treasurer fl-cred-corpus-quarantine, OPEN: plaintext host credentials in the" >&2
    echo "        ASSISTANT role, i.e. as training TARGETS. The trainer's loader refuses these" >&2
    echo "        bytes by content, so this run cannot start regardless of this qualification." >&2
    echo "        DO NOT disable the quarantine or add an exception to unblock. Correct path:" >&2
    echo "          1. treasurer sanctions replacement bytes with the credential rows resolved" >&2
    echo "          2. re-run qualification to produce a NEW QUALIFIED_CORPUS_SHA" >&2
    echo "          3. update this pin to that sha -- this check then passes by itself" >&2
    exit 1
  fi
done
QUALIFIED_BASE_MANIFEST_SHA=2406fff54148dd44d9c7a4824d43aaace0c450d3e6b174bfe1268565a9512c5d
QUALIFIED_SAMPLES=10033
QUALIFIED_INPUT_TOKENS=3888967
QUALIFIED_LABEL_TOKENS=1107992
QUALIFIED_DATASET_SHAPE_SHA=c65815f4c91d9e4e2d5e810ae3af3e8008a738bf98b8c7a77c47bec3bdc2d96a
PRODUCTION_PLAN_SHA=63780ca438a6a2e5f362b4c4a9a9b13e1ff6b030aaa0c7d1bc16f9032df35550
QUALIFIED_PLAN_STEPS=979
QUALIFIED_LORA_SHA=e4a071dadbbbd06e890fee84fd4b1d791a9fa6c75d26818a2d8a37ced9a82eb5
QUALIFIED_MIN_USEFUL_TPS=1000
QUALIFIED_QUALIFICATION_AUTHORIZATION_SHA=ccef0cfcd37db816465a8b88420386885383d3aab194211d46eff38bb991ed42

[ "$QUALIFIED_QUALIFICATION_AUTHORIZATION_SHA" != UNQUALIFIED ] || {
  echo "REFUSE: this deployment is qualifier-only; promote the exact passed qualification authorization in a new commit before full SFT." >&2
  exit 1
}

[ "$RUN_TAG" = "$QUALIFIED_RUN_TAG" ] || {
  echo "REFUSE: RUN_TAG=$RUN_TAG has not qualified this production topology." >&2
  exit 1
}
[ "$MAX_SEQ:$PAD_TO_MULTIPLE:$SHORT_MAX:$MID_MAX:$SHORT_BATCH:$MID_BATCH:$LONG_BATCH:$CLOCK_CAP" = \
  "1792:16:160:512:8:2:1:2000" ] || {
  echo "REFUSE: requested topology differs from the >=1000 tok/s qualified topology." >&2
  exit 1
}
[ "$SELECTIVE_AC_START:$SELECTIVE_AC_BUDGET" = "1536:1472" ] || {
  echo "REFUSE: activation-checkpoint schedule differs from qualification." >&2
  exit 1
}
[ "$CACHE_RELEASE_INTERVAL_STEPS:$CACHE_RELEASE_BELOW_AVAILABLE_BYTES:$MIN_AFTER_CACHE_RELEASE_BYTES:$SAVE_EVERY:$MAX_SWAP_USED_BYTES:$MAX_STEP_STALL_SECONDS" = \
  "24:8000000000:40000000000:50:134217728:180" ] || {
  echo "REFUSE: memory recovery/checkpoint policy differs from qualification." >&2
  exit 1
}
case "$OUTPUT_DIR" in
  "${SPARK_HOME%/}"/training_outputs/*) ;;
  *)
    echo "REFUSE: OUTPUT_DIR escaped the Spark training-output root: $OUTPUT_DIR" >&2
    exit 1
    ;;
esac
for numeric in "$FIRST_SESSION_STEPS" "$MAX_SESSION_STEPS" "$MAX_SESSIONS" \
  "$PAD_TO_MULTIPLE" \
  "$START_BOARD_MAX_C" "$MAX_BOARD_CELSIUS" "$THERMAL_PERSIST_STEPS" \
  "$FRESH_UPTIME_MAX" "$MEM_AVAILABLE_MIN_BYTES" \
  "$DISK_AVAILABLE_MIN_BYTES" "$MIN_TRAIN_MEM_AVAILABLE_BYTES" \
  "$CACHE_RELEASE_INTERVAL_STEPS" \
  "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" "$MIN_AFTER_CACHE_RELEASE_BYTES" \
  "$MAX_SWAP_USED_BYTES" "$SAVE_EVERY" "$SESSION_BASE_TIMEOUT_SECONDS" \
  "$STEP_TIMEOUT_SECONDS" "$MAX_STEP_STALL_SECONDS" \
  "$EXIT_GRACE_SECONDS"; do
  case "$numeric" in *[!0-9]*|'')
    echo "REFUSE: malformed integer production setting: $numeric" >&2
    exit 1
    ;;
  esac
  [ "$numeric" -gt 0 ]
done
[ "$FIRST_SESSION_STEPS" -le "$MAX_SESSION_STEPS" ] &&
[ "$PAD_TO_MULTIPLE" -ge 8 ] &&
[ $((PAD_TO_MULTIPLE % 8)) -eq 0 ] &&
[ "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" -lt "$MIN_AFTER_CACHE_RELEASE_BYTES" ] &&
[ "$MIN_AFTER_CACHE_RELEASE_BYTES" -le "$MIN_TRAIN_MEM_AVAILABLE_BYTES" ] &&
[ "$SAVE_EVERY" -le "$FIRST_SESSION_STEPS" ] &&
[ "$MAX_BOARD_CELSIUS" -ge 50 ] &&
[ "$MAX_BOARD_CELSIUS" -lt 94 ] &&
[ "$START_BOARD_MAX_C" -lt "$MAX_BOARD_CELSIUS" ] || {
  echo "REFUSE: invalid session or thermal production contract." >&2
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
IMMUTABLE_FILES=(
  careers-qwen/run_stage2_sft_ddp_till_done.sh
  careers-qwen/sft_ddp_plan_receipt.py
  careers-qwen/stage_sft_base.sh
  careers-qwen/launch_4node.sh
  careers-qwen/train_ddp_lora.py
  dense-9b/trainers/train_fsdp_dense_9b.py
)
for file in "${IMMUTABLE_FILES[@]}"; do
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
  echo "REFUSE: another DDP SFT campaign holds $STATE_DIR/driver.lock." >&2
  exit 1
}
DRIVER_LOG="$STATE_DIR/driver.log"
say(){
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

trainer_count(){
  local node=$1
  ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
    "ps -eo args= | awk '
       /[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++}
       END{print n+0}'"
}

latest_complete(){
  local rank node value reference_step= reference_sha=
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    value=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=15 spark@"$node" \
      "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "$QUALIFIED_PLAN_STEPS" "$PRODUCTION_PLAN_SHA")" <<'REMOTE'
set -euo pipefail
output=$1
horizon=$2
expected_plan=$3
latest=0
latest_sha=none
if [ -d "$output" ]; then
  temporary=$(find "$output" -maxdepth 1 -name '.checkpoint-*' -print -quit)
  [ -z "$temporary" ] || {
    echo "incomplete temporary checkpoint: $temporary" >&2
    exit 1
  }
  for candidate in "$output"/checkpoint-*; do
    [ -d "$candidate" ] || continue
    step=${candidate##*-}
    case "$step" in *[!0-9]*|'')
      echo "malformed checkpoint path: $candidate" >&2
      exit 1
      ;;
    esac
    [ "$step" -le "$horizon" ] || {
      echo "checkpoint-$step exceeds horizon $horizon" >&2
      exit 1
    }
    for file in COMPLETE adapter_config.json adapter_model.safetensors \
      trainer_state.pt training_manifest.json SHA256SUMS.json; do
      [ -s "$candidate/$file" ] || {
        echo "incomplete checkpoint-$step: missing $file" >&2
        exit 1
      }
    done
    complete_sha=$(sed -n 's/^adapter_sha256=//p' "$candidate/COMPLETE")
    case "$complete_sha" in *[!0-9a-f]*|'')
      echo "malformed COMPLETE digest in checkpoint-$step" >&2
      exit 1
      ;;
    esac
    [ "${#complete_sha}" = 64 ]
    read -r manifest_step manifest_plan listed_sha < <(
      python3 - "$candidate/training_manifest.json" "$candidate/SHA256SUMS.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
sums = json.load(open(sys.argv[2]))
print(
    manifest["completed_steps"],
    manifest["plan_sha256"],
    sums["adapter_model.safetensors"],
)
PY
    )
    [ "$manifest_step" = "$step" ]
    [ "$manifest_plan" = "$expected_plan" ]
    [ "$listed_sha" = "$complete_sha" ]
    actual_sha=$(sha256sum "$candidate/adapter_model.safetensors" | cut -d' ' -f1)
    [ "$actual_sha" = "$complete_sha" ]
    if [ "$step" -gt "$latest" ]; then
      latest=$step
      latest_sha=$complete_sha
    fi
  done
fi
printf '%s %s\n' "$latest" "$latest_sha"
REMOTE
)
    read -r step adapter_sha <<<"$value"
    case "$step" in *[!0-9]*|'')
      echo "REFUSE: malformed checkpoint step from rank$rank .$node: $value" >&2
      return 1
      ;;
    esac
    if [ "$step" = 0 ]; then
      [ "$adapter_sha" = none ] || {
        echo "REFUSE: rank$rank reported a digest without a checkpoint: $value" >&2
        return 1
      }
    else
      case "$adapter_sha" in *[!0-9a-f]*|'')
        echo "REFUSE: malformed checkpoint digest from rank$rank .$node: $value" >&2
        return 1
        ;;
      esac
      [ "${#adapter_sha}" = 64 ] || {
        echo "REFUSE: short checkpoint digest from rank$rank .$node: $value" >&2
        return 1
      }
    fi
    if [ "$rank" = 0 ]; then
      reference_step=$step
      reference_sha=$adapter_sha
    elif [ "$step $adapter_sha" != "$reference_step $reference_sha" ]; then
      echo "REFUSE: DDP checkpoint split-brain: rank0=$reference_step/$reference_sha rank$rank=$step/$adapter_sha." >&2
      return 1
    fi
  done
  printf '%s %s\n' "$reference_step" "$reference_sha"
}

verify_protected_checkpoint(){
  local rank node expected_metadata receipt
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    case "$rank" in
      0) expected_metadata=1127670 ;;
      1) expected_metadata=1130940 ;;
      2|3) expected_metadata=1130948 ;;
    esac
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "$(printf '%q ' bash -s -- "$rank" "$expected_metadata")" <<'REMOTE'
set -euo pipefail
rank=$1
expected_metadata=$2
checkpoint=${STAGE2_CHECKPOINT:?set STAGE2_CHECKPOINT to the stage2 checkpoint path}
[ "$(cat "$checkpoint/COMPLETE")" = "step=800 epoch=0 data_pos=800 rank=$rank" ]
[ "$(stat -c %s "$checkpoint/trainer_meta.pt")" = 7245 ]
[ "$(stat -c %s "$checkpoint/dcp/__${rank}.metadata")" = "$expected_metadata" ]
[ "$(stat -c %s "$checkpoint/dcp/__${rank}_0.distcp")" = 13754046548 ]
printf 'rank=%s meta=%s shard=%s\n' \
  "$rank" "$expected_metadata" 13754046548
REMOTE
)
    say "protected checkpoint-800 .$node $receipt"
  done
}

reboot_and_verify(){
  local label expected_step expected_sha rank node trainers session_state
  local ready attempt receipt boot_id uptime nccl_shm mem disk board checkpoint_sha
  label=$1
  expected_step=$2
  expected_sha=$3
  local -a boot_ids
  boot_ids=()
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    trainers=$(trainer_count "$node")
    session_state=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "tmux has-session -t '$SESSION' 2>/dev/null && echo present || echo absent")
    [ "$trainers" = 0 ] && [ "$session_state" = absent ] || {
      echo "REFUSE: rank$rank .$node is not idle before $label reboot: trainers=$trainers session=$session_state" >&2
      exit 1
    }
    boot_ids[$rank]=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "cat /proc/sys/kernel/random/boot_id")
    say "$label pre-reboot rank$rank .$node boot=${boot_ids[$rank]} trainers=0 session=absent"
  done

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
         [ "$boot_id" != "${boot_ids[$rank]}" ] &&
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
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=15 spark@"$node" \
      "$(printf '%q ' bash -s -- "$SFT_CORPUS" "$BASE_MODEL" "$OUTPUT_DIR" "$expected_step")" <<'REMOTE'
set -euo pipefail
corpus=$1
base=$2
output=$3
expected_step=$4
boot=$(cat /proc/sys/kernel/random/boot_id)
uptime=$(cut -d. -f1 /proc/uptime)
trainers=$(ps -eo args= | awk \
  '/[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++} END{print n+0}')
nccl_shm=$(find /dev/shm -maxdepth 1 -name 'nccl*' -print | wc -l)
mem=$(awk '/MemAvailable:/{print $2*1024}' /proc/meminfo)
disk=$(df -B1 --output=avail "$base" | tail -1 | tr -d ' ')
board=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
test -f "$corpus"
test -f "$base/GRAFT_COMPLETE"
test -f "$base/weight_diff.json"
test -f "$base/training_provenance.json"
test -f "$base/SOURCE_SHA256SUMS"
(
  cd "$base"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
checkpoint_sha=none
if [ "$expected_step" -gt 0 ]; then
  checkpoint="$output/checkpoint-$expected_step"
  test -s "$checkpoint/COMPLETE"
  test -s "$checkpoint/adapter_model.safetensors"
  test -s "$checkpoint/trainer_state.pt"
  checkpoint_sha=$(sed -n 's/^adapter_sha256=//p' "$checkpoint/COMPLETE")
  actual_sha=$(sha256sum "$checkpoint/adapter_model.safetensors" | cut -d' ' -f1)
  [ "$checkpoint_sha" = "$actual_sha" ]
fi
printf '%s %s %s %s %s %s %s %s %s %s %s\n' \
  "$boot" "$uptime" "$trainers" "$nccl_shm" "$mem" "$disk" "$board" \
  "$(stat -c %s "$corpus")" "$(sha256sum "$corpus" | cut -d' ' -f1)" \
  "$(sha256sum "$base/SOURCE_SHA256SUMS" | cut -d' ' -f1)" "$checkpoint_sha"
REMOTE
)
    read -r boot_id uptime trainers nccl_shm mem disk board \
      corpus_bytes corpus_sha base_manifest_sha checkpoint_sha <<<"$receipt"
    [ "$trainers" = 0 ] && [ "$nccl_shm" = 0 ]
    [ "$uptime" -lt "$FRESH_UPTIME_MAX" ]
    case "$board" in *[!0-9]*|'')
      echo "ABORT: rank$rank .$node has no numeric board-temperature receipt." >&2
      exit 1
      ;;
    esac
    [ "$mem" -ge "$MEM_AVAILABLE_MIN_BYTES" ] || {
      echo "ABORT: rank$rank .$node has only $mem available after reboot." >&2
      exit 1
    }
    [ "$disk" -ge "$DISK_AVAILABLE_MIN_BYTES" ] || {
      echo "ABORT: rank$rank .$node has only $disk free after reboot." >&2
      exit 1
    }
    [ $((board / 1000)) -le "$START_BOARD_MAX_C" ] || {
      echo "ABORT: rank$rank .$node board $((board / 1000))C exceeds launch gate." >&2
      exit 1
    }
    [ "$corpus_bytes:$corpus_sha" = \
      "$QUALIFIED_CORPUS_BYTES:$QUALIFIED_CORPUS_SHA" ] || {
      echo "ABORT: rank$rank corpus changed after qualification." >&2
      exit 1
    }
    [ "$base_manifest_sha" = "$QUALIFIED_BASE_MANIFEST_SHA" ] || {
      echo "ABORT: rank$rank baked base manifest changed after qualification." >&2
      exit 1
    }
    if [ "$expected_step" -gt 0 ]; then
      [ "$checkpoint_sha" = "$expected_sha" ] || {
        echo "ABORT: rank$rank checkpoint-$expected_step SHA changed across reboot." >&2
        exit 1
      }
    fi
    say "$label rank$rank .$node boot=$boot_id uptime=${uptime}s trainers=0 nccl_shm=0 mem=$mem disk=$disk board=$((board / 1000))C checkpoint=$expected_step adapter_sha=$checkpoint_sha base_manifest=$base_manifest_sha"
  done
  verify_protected_checkpoint
}

deploy_runtime(){
  local runtime_stage file node local_sha remote_sha
  runtime_stage=$(mktemp -d)
  for file in "${IMMUTABLE_FILES[@]}"; do
    mkdir -p "$runtime_stage/$(dirname "$file")"
    git show "$DEPLOY_SHA:$file" >"$runtime_stage/$file"
  done
  for node in "${NODES[@]}"; do
    for file in "${IMMUTABLE_FILES[@]}"; do
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
    say "runtime byte-exact on .$node at $DEPLOY_SHA"
  done
  rm -rf -- "$runtime_stage"
}

plan_receipt(){
  local rank node receipt reference=
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=120 spark@"$node" \
      "python3 '$SPARK_HOME/palios-training/careers-qwen/sft_ddp_plan_receipt.py' \
        --trainer '$SPARK_HOME/palios-training/dense-9b/trainers/train_fsdp_dense_9b.py' \
        --ddp-trainer '$SPARK_HOME/palios-training/careers-qwen/train_ddp_lora.py' \
        --corpus '$SFT_CORPUS' \
        --model '$BASE_MODEL' \
        --max-seq '$MAX_SEQ' \
        --short-max '$SHORT_MAX' \
        --mid-max '$MID_MAX' \
        --short-batch '$SHORT_BATCH' \
        --mid-batch '$MID_BATCH' \
        --long-batch '$LONG_BATCH'")
    if [ "$rank" = 0 ]; then
      reference=$receipt
    elif [ "$receipt" != "$reference" ]; then
      echo "ABORT: rank$rank DDP plan receipt differs from rank0." >&2
      exit 1
    fi
    say "plan rank$rank .$node $receipt" >&2
  done
  printf '%s\n' "$reference"
}

validate_plan_receipt(){
  local receipt=$1 fields
  fields=$(python3 - "$receipt" <<'PY'
import json
import sys
prefix = "SFT_DDP_PLAN_RECEIPT "
value = sys.argv[1]
if not value.startswith(prefix):
    raise SystemExit("missing SFT_DDP_PLAN_RECEIPT prefix")
receipt = json.loads(value[len(prefix):])
corpus = receipt["corpus"]
dataset = receipt["dataset"]
print(
    corpus["bytes"],
    corpus["rows"],
    corpus["sha256"],
    dataset["samples"],
    dataset["input_tokens"],
    dataset["label_tokens"],
    dataset["shape_sha256"],
    receipt["full_schedule_steps"],
    receipt["plan_sha256"],
)
PY
)
  read -r corpus_bytes corpus_rows corpus_sha samples input_tokens label_tokens \
    dataset_shape plan_steps plan_sha <<<"$fields"
  [ "$corpus_bytes:$corpus_rows:$corpus_sha" = \
    "$QUALIFIED_CORPUS_BYTES:$QUALIFIED_CORPUS_ROWS:$QUALIFIED_CORPUS_SHA" ] &&
  [ "$samples:$input_tokens:$label_tokens:$dataset_shape" = \
    "$QUALIFIED_SAMPLES:$QUALIFIED_INPUT_TOKENS:$QUALIFIED_LABEL_TOKENS:$QUALIFIED_DATASET_SHAPE_SHA" ] &&
  [ "$plan_steps:$plan_sha" = \
    "$QUALIFIED_PLAN_STEPS:$PRODUCTION_PLAN_SHA" ] || {
    echo "ABORT: production corpus/plan differs from the throughput-qualified receipt." >&2
    exit 1
  }
  say "qualified plan bound: samples=$samples tokens=$input_tokens labels=$label_tokens steps=$plan_steps plan_sha=$plan_sha"
}

pin_and_peer_gate(){
  local node result median rank
  local -a gemm_tflops sorted_tflops
  gemm_tflops=()
  for node in "${NODES[@]}"; do
    ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
      "sudo nvidia-smi -pm 1 >/dev/null 2>&1;
       sudo nvidia-smi -lgc 0,'$CLOCK_CAP' >/dev/null 2>&1"
    result=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=30 spark@"$node" \
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
    gemm_tflops+=("$result")
    say "GEMM .$node ${result}TFLOPS at ${CLOCK_CAP}MHz"
  done
  readarray -t sorted_tflops < <(printf '%s\n' "${gemm_tflops[@]}" | sort -n)
  median=$(awk -v a="${sorted_tflops[1]}" -v b="${sorted_tflops[2]}" \
    'BEGIN { printf "%.3f", (a+b)/2 }')
  for rank in 0 1 2 3; do
    awk -v value="${gemm_tflops[$rank]}" -v center="$median" \
      'BEGIN { exit !(value >= center*0.80) }' || {
      echo "ABORT: rank$rank GEMM ${gemm_tflops[$rank]} is below 0.80x median $median." >&2
      exit 1
    }
  done
}

archive_logs(){
  local label=$1 rank node remote_log
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    remote_log="$LOGDIR/${label}_r${rank}.log"
    scp "${SSH_OPTIONS[@]}" -q -o BatchMode=yes -o ConnectTimeout=15 \
      "spark@$node:$remote_log" \
      "$STATE_DIR/logs/${label}_r${rank}.log" ||
      say "WARN: could not archive $label rank$rank log from .$node"
  done
}

rotate_checkpoints(){
  local expected_step=$1 rank node receipt
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=15 spark@"$node" \
      "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "${SPARK_HOME%/}/training_outputs" "$expected_step")" <<'REMOTE'
set -euo pipefail
output=$1
training_root=$2
expected=$3
case "$output" in "$training_root"/*) ;; *) exit 1;; esac
steps=()
for candidate in "$output"/checkpoint-*; do
  [ -d "$candidate" ] || continue
  step=${candidate##*-}
  case "$step" in *[!0-9]*|'') exit 1;; esac
  [ -s "$candidate/COMPLETE" ] &&
  [ -s "$candidate/adapter_model.safetensors" ] &&
  [ -s "$candidate/trainer_state.pt" ]
  steps+=("$step")
done
[ "${#steps[@]}" -gt 0 ]
mapfile -t steps < <(printf '%s\n' "${steps[@]}" | sort -n)
latest=${steps[$((${#steps[@]} - 1))]}
[ "$latest" = "$expected" ]
remove_count=$((${#steps[@]} - 2))
[ "$remove_count" -gt 0 ] || remove_count=0
removed=
for ((index=0; index<remove_count; index++)); do
  step=${steps[$index]}
  target="$output/checkpoint-$step"
  case "$target" in "$output"/checkpoint-[0-9]*) ;; *) exit 1;; esac
  [ -s "$target/COMPLETE" ] &&
  [ -s "$target/adapter_model.safetensors" ] &&
  [ -s "$target/trainer_state.pt" ]
  rm -rf -- "$target"
  removed="${removed}${removed:+,}$step"
done
kept=$(printf '%s\n' "${steps[@]}" | tail -2 | paste -sd, -)
printf 'kept=%s removed=%s\n' "$kept" "${removed:-none}"
REMOTE
)
    say "checkpoint rotation rank$rank .$node $receipt"
  done
}

stop_remote_session(){
  local label=$1 reason=$2 node
  local failed=0
  say "stopping exact remote session=$SESSION label=$label reason=$reason"
  for node in "${NODES[@]}"; do
    if ! timeout -k 2 15 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes \
      -o ConnectTimeout=8 spark@"$node" \
      "if tmux has-session -t '$SESSION' 2>/dev/null;
       then tmux send-keys -t '$SESSION' C-c; fi"; then
      say "ERROR: exact C-c failed on .$node"
      failed=1
    fi
  done
  sleep 30
  for node in "${NODES[@]}"; do
    if ! timeout -k 2 15 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes \
      -o ConnectTimeout=8 spark@"$node" \
      "if tmux has-session -t '$SESSION' 2>/dev/null;
       then tmux kill-session -t '$SESSION'; fi;
       sleep 1;
       trainers=\$(ps -eo args= | awk '
         /[t]rain_ddp_lora.py|[t]orch.distributed.run/{n++}
         END{print n+0}');
       [ \"\$trainers\" = 0 ]"; then
      say "ERROR: exact tmux cleanup failed on .$node"
      failed=1
    fi
  done
  [ "$failed" = 0 ] || {
    echo "ABORT: exact remote-session cleanup was incomplete." >&2
    return 1
  }
}

launch_session(){
  local current=$1 target=$2 label rank node resume remote_log exit_file
  local pass_line measured_tps min_system_after min_system_after_reclaim
  local max_swap max_swap_growth allocator_retry_growth oom_growth
  local memory_exit_events
  local reported_steps requested_steps memory_guard_exit new_step new_sha
  local -a train_args
  label="session_${current}_to_${target}_$(date -u +%Y%m%dT%H%M%SZ)"
  resume=
  if [ "$current" -gt 0 ]; then
    resume="$OUTPUT_DIR/checkpoint-$current"
  fi
  train_args=(
    --model "$BASE_MODEL"
    --data "$SFT_CORPUS"
    --canonical-trainer "$SPARK_HOME/palios-training/dense-9b/trainers/train_fsdp_dense_9b.py"
    --out "$OUTPUT_DIR"
    --expected-samples "$QUALIFIED_SAMPLES"
    --max-seq "$MAX_SEQ"
    --pad-to-multiple "$PAD_TO_MULTIPLE"
    --short-max "$SHORT_MAX"
    --mid-max "$MID_MAX"
    --short-batch "$SHORT_BATCH"
    --mid-batch "$MID_BATCH"
    --long-batch "$LONG_BATCH"
    --max-steps "$target"
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
    --save-every "$SAVE_EVERY"
  )
  [ -z "$resume" ] || train_args+=(--resume "$resume")

  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    remote_log="$LOGDIR/${label}_r${rank}.log"
    exit_file="$LOGDIR/${label}_r${rank}.exit"
    ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "$(printf '%q ' bash -s -- "$rank" "$SESSION" "$remote_log" "$exit_file" "$SPARK_HOME" "$SPARK_RAIL_MASTER" "${train_args[@]}")" <<'REMOTE'
set -euo pipefail
rank=$1
session=$2
log=$3
exit_file=$4
spark_home=$5
rail_master=$6
shift 6
mkdir -p "$(dirname "$log")"
[ ! -e "$log" ] && [ ! -e "$exit_file" ]
printf -v command '%q ' \
  env "SPARK_HOME=$spark_home" "SPARK_RAIL_MASTER=$rail_master" \
  bash "$spark_home/palios-training/careers-qwen/launch_4node.sh" "$rank" "$@"
command+=" >$(printf '%q' "$log") 2>&1; "
command+="rc=\$?; printf '%s\\n' \"\$rc\" >$(printf '%q' "$exit_file"); exit \"\$rc\""
tmux new-session -d -s "$session" "$command"
REMOTE
    say "launched $label rank$rank on .$node"
    [ "$rank" = 0 ] && sleep 12
  done

  local deadline alive exited missing partial_since failure step_line state
  local last_progress_line= last_progress_at=0
  deadline=$((SECONDS + SESSION_BASE_TIMEOUT_SECONDS + (target - current) * STEP_TIMEOUT_SECONDS))
  partial_since=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    alive=0
    exited=0
    missing=0
    for rank in 0 1 2 3; do
      node=${NODES[$rank]}
      exit_file="$LOGDIR/${label}_r${rank}.exit"
      state=$(timeout -k 2 10 ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
        "if [ -f '$exit_file' ]; then echo EXITED;
         elif tmux has-session -t '$SESSION' 2>/dev/null; then echo ALIVE;
         else echo MISSING; fi" 2>/dev/null || true)
      [ "$state" = ALIVE ] && alive=$((alive + 1))
      [ "$state" = EXITED ] && exited=$((exited + 1))
      [ "$state" = MISSING ] && missing=$((missing + 1))
    done
    [ "$missing" = 0 ] || {
      stop_remote_session "$label" "missing-rank"
      archive_logs "$label"
      echo "ABORT: $label has $missing rank(s) without a tmux session or exit receipt." >&2
      exit 1
    }
    failure=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
      "grep -aE 'Traceback|CUDA out of memory|RuntimeError|ValueError|FileNotFoundError|ChildFailedError|NCCL.*(error|failed)' \
        '$LOGDIR/${label}_r0.log' 2>/dev/null | tail -1" 2>/dev/null || true)
    [ -z "$failure" ] || {
      stop_remote_session "$label" "rank0-failure-signal"
      archive_logs "$label"
      echo "ABORT: rank-0 failure signal: $failure" >&2
      exit 1
    }
    step_line=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
      "grep -aE '\\[step [0-9]+/|CHECKPOINT_COMPLETE|TRAINING_PASS_COMPLETE' \
        '$LOGDIR/${label}_r0.log' 2>/dev/null | tail -1" 2>/dev/null || true)
    if [ -n "$step_line" ] && [ "$step_line" != "$last_progress_line" ]; then
      last_progress_line=$step_line
      last_progress_at=$SECONDS
    elif [ "$last_progress_at" -gt 0 ] &&
      [ $((SECONDS - last_progress_at)) -ge "$MAX_STEP_STALL_SECONDS" ]; then
      stop_remote_session "$label" \
        "no-step-progress-${MAX_STEP_STALL_SECONDS}s"
      archive_logs "$label"
      echo "ABORT: $label made no logged step progress for ${MAX_STEP_STALL_SECONDS}s." >&2
      exit 1
    fi
    say "$label alive=$alive/4 exited=$exited/4 ${step_line:-loading-model}"
    [ "$exited" = 4 ] && break
    if [ "$alive" -gt 0 ] && [ "$alive" -lt 4 ]; then
      [ "$partial_since" -ne 0 ] || partial_since=$SECONDS
      [ $((SECONDS - partial_since)) -lt "$EXIT_GRACE_SECONDS" ] || {
        stop_remote_session "$label" "split-ranks-${EXIT_GRACE_SECONDS}s"
        archive_logs "$label"
        echo "ABORT: $label ranks remained split for ${EXIT_GRACE_SECONDS}s." >&2
        exit 1
      }
    else
      partial_since=0
    fi
    sleep 30
  done
  [ "${exited:-0}" = 4 ] || {
    stop_remote_session "$label" "production-timeout"
    archive_logs "$label"
    echo "ABORT: $label exceeded its production timeout." >&2
    exit 1
  }

  local -a exits
  exits=()
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    exit_file="$LOGDIR/${label}_r${rank}.exit"
    exits[$rank]=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "cat '$exit_file'")
    [ "${exits[$rank]}" = 0 ] || {
      archive_logs "$label"
      echo "ABORT: $label rank$rank exited ${exits[$rank]}." >&2
      exit 1
    }
  done
  archive_logs "$label"

  local rank0_log="$STATE_DIR/logs/${label}_r0.log"
  grep -aF "\"full_schedule_steps\":$QUALIFIED_PLAN_STEPS" "$rank0_log" >/dev/null
  grep -aF "\"sha256\":\"$PRODUCTION_PLAN_SHA\"" "$rank0_log" >/dev/null
  grep -aF "\"pad_to_multiple\":$PAD_TO_MULTIPLE" "$rank0_log" >/dev/null
  grep -aF '"bytes":400556032' "$rank0_log" >/dev/null
  grep -aF '"dtypes":["torch.float32"]' "$rank0_log" >/dev/null
  grep -aF '"tensors":704' "$rank0_log" >/dev/null
  if [ "$current" = 0 ]; then
    grep -aF "\"sha256\":\"$QUALIFIED_LORA_SHA\"" "$rank0_log" >/dev/null
  fi
  grep -aF '"rms_norm_modules":129' "$rank0_log" >/dev/null
  grep -aF '"swiglu_modules":64' "$rank0_log" >/dev/null
  grep -aF '"projected_tokens":"assistant_labels_only"' "$rank0_log" >/dev/null
  pass_line=$(grep -aF "TRAINING_PASS_COMPLETE " "$rank0_log" | tail -1)
  read -r reported_steps requested_steps memory_guard_exit measured_tps \
    min_system_after min_system_after_reclaim max_swap max_swap_growth \
    allocator_retry_growth oom_growth memory_exit_events <<<"$(
    python3 - "$pass_line" <<'PY'
import re
import sys

line = sys.argv[1]
names = (
    "steps",
    "requested_steps",
    "memory_guard_exit",
    "measured_useful_input_tok_s",
    "uma_min_system_available_after",
    "uma_min_system_available_after_reclaim",
    "uma_max_swap",
    "uma_max_swap_growth",
    "uma_max_allocator_retry_growth",
    "uma_max_oom_growth",
    "memory_guard_exit_rank_events",
)
values = []
for name in names:
    match = re.search(rf"(?:^| ){name}=([^ ]+)", line)
    if match is None:
        raise SystemExit(f"missing {name} in TRAINING_PASS_COMPLETE")
    values.append(match.group(1))
print(*values)
PY
  )"
  read -r new_step new_sha <<<"$(latest_complete)"
  [ "$new_step" -gt "$current" ] &&
  [ "$new_step" -le "$target" ] &&
  [ "$reported_steps" = "$new_step" ] &&
  [ "$requested_steps" = "$target" ] || {
    echo "ABORT: $label checkpoint/marker mismatch: previous=$current latest=$new_step reported=$reported_steps requested=$requested_steps target=$target." >&2
    exit 1
  }
  awk -v value="$measured_tps" -v minimum="$QUALIFIED_MIN_USEFUL_TPS" \
    'BEGIN { exit !(value >= minimum) }' || {
    echo "ABORT: $label measured ${measured_tps} useful tok/s; production minimum is ${QUALIFIED_MIN_USEFUL_TPS}." >&2
    exit 1
  }
  [ "$max_swap" -le "$MAX_SWAP_USED_BYTES" ] || {
    echo "ABORT: $label used $max_swap swap bytes; production maximum is $MAX_SWAP_USED_BYTES." >&2
    exit 1
  }
  [ "$memory_guard_exit" = 0 ] &&
  [ "$new_step" = "$target" ] &&
  [ "$min_system_after" -ge "$CACHE_RELEASE_BELOW_AVAILABLE_BYTES" ] &&
  [ "$min_system_after_reclaim" -ge "$MIN_AFTER_CACHE_RELEASE_BYTES" ] &&
  [ "$max_swap_growth" = 0 ] &&
  [ "$allocator_retry_growth" = 0 ] &&
  [ "$oom_growth" = 0 ] &&
  [ "$memory_exit_events" = 0 ] || {
    echo "ABORT: $label violated the receipt-faithful memory gate: checkpoint=$new_step target=$target memory_guard_exit=$memory_guard_exit min_system_after=$min_system_after min_system_after_reclaim=$min_system_after_reclaim max_swap_growth=$max_swap_growth allocator_retry_growth=$allocator_retry_growth oom_growth=$oom_growth memory_exit_events=$memory_exit_events." >&2
    exit 1
  }
  if grep -aE 'missing_grads_sum=[1-9][0-9]*' "$rank0_log" >/dev/null; then
    echo "ABORT: $label observed missing LoRA gradients." >&2
    exit 1
  fi

  say "$label COMPLETE checkpoint-$new_step/$target adapter_sha=$new_sha measured_useful_tps=$measured_tps min_system_after=$min_system_after min_system_after_reclaim=$min_system_after_reclaim memory_guard_exit=$memory_guard_exit exits=${exits[*]}"
  if command -v taey-notify >/dev/null; then
    taey-notify tutor \
      "SFT DDP PROGRESS: ${RUN_TAG} checkpoint-${new_step}/${QUALIFIED_PLAN_STEPS} COMPLETE on all 4; adapter_sha=${new_sha}; logs=${STATE_DIR}/logs" \
      --type response_ready || say "WARN: taey-notify tutor failed"
  fi
}

say "staging exact baked CPT base for run=$RUN_TAG"
RUN_TAG=$RUN_TAG bash careers-qwen/stage_sft_base.sh
say "deploying immutable DDP production runtime from $DEPLOY_SHA"
deploy_runtime
PLAN_RECEIPT=$(plan_receipt)
validate_plan_receipt "$PLAN_RECEIPT"
printf '%s\n' "$PLAN_RECEIPT" >"$STATE_DIR/plan_receipt.txt"
verify_protected_checkpoint

read -r current adapter_sha <<<"$(latest_complete)"
if [ "$current" = 0 ]; then
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    contents=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "if [ -d '$OUTPUT_DIR' ]; then find '$OUTPUT_DIR' -mindepth 1 -maxdepth 1 -print -quit; fi")
    [ -z "$contents" ] || {
      echo "REFUSE: fresh production output is non-empty on rank$rank: $contents" >&2
      exit 1
    }
  done
fi
say "campaign start checkpoint=$current adapter_sha=$adapter_sha output=$OUTPUT_DIR"
reboot_and_verify pre-campaign "$current" "$adapter_sha"

session_number=0
while [ "$current" -lt "$QUALIFIED_PLAN_STEPS" ]; do
  session_number=$((session_number + 1))
  [ "$session_number" -le "$MAX_SESSIONS" ] || {
    echo "ABORT: reached MAX_SESSIONS=$MAX_SESSIONS at checkpoint-$current." >&2
    exit 1
  }
  remaining=$((QUALIFIED_PLAN_STEPS - current))
  if [ "$current" = 0 ] && [ "$remaining" -gt "$FIRST_SESSION_STEPS" ]; then
    session_steps=$FIRST_SESSION_STEPS
  elif [ "$remaining" -lt "$MAX_SESSION_STEPS" ]; then
    session_steps=$remaining
  else
    session_steps=$MAX_SESSION_STEPS
  fi
  target=$((current + session_steps))
  say "session=$session_number checkpoint-$current -> checkpoint-$target"
  pin_and_peer_gate
  launch_session "$current" "$target"
  read -r current adapter_sha <<<"$(latest_complete)"
  reboot_and_verify "post-session-$session_number" "$current" "$adapter_sha"
  rotate_checkpoints "$current"
done

completion=$(python3 - "$RUN_TAG" "$OUTPUT_DIR" "$QUALIFIED_PLAN_STEPS" \
  "$adapter_sha" "$PRODUCTION_PLAN_SHA" "$DEPLOY_SHA" <<'PY'
import json
import sys
run_tag, output, steps, adapter_sha, plan_sha, deploy_sha = sys.argv[1:]
print(json.dumps({
    "adapter_sha256": adapter_sha,
    "deploy_sha": deploy_sha,
    "format": "stage2-ddp-lora-complete-v1",
    "output_dir": output,
    "plan_sha256": plan_sha,
    "run_tag": run_tag,
    "steps": int(steps),
}, sort_keys=True, separators=(",", ":")))
PY
)
printf '%s\n' "$completion" >"$STATE_DIR/TRAINING_COMPLETE.json"
for node in "${NODES[@]}"; do
  ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "$completion")" <<'REMOTE'
set -euo pipefail
output=$1
completion=$2
printf '%s\n' "$completion" >"$output/TRAINING_COMPLETE.json"
REMOTE
done
say "TRAINING COMPLETE: checkpoint-$QUALIFIED_PLAN_STEPS adapter_sha=$adapter_sha plan_sha=$PRODUCTION_PLAN_SHA"
if command -v taey-notify >/dev/null; then
  taey-notify tutor \
    "SFT DDP COMPLETE: ${RUN_TAG} checkpoint-${QUALIFIED_PLAN_STEPS} survived final reboot on all 4; adapter_sha=${adapter_sha}; output=${OUTPUT_DIR}; logs=${STATE_DIR}/logs" \
    --type response_ready || say "WARN: taey-notify tutor failed"
fi
