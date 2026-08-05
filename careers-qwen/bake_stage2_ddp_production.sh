#!/usr/bin/env bash
# Controller-side production lifecycle for a completed four-node DDP LoRA bake.
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${POST_CPT_ARTIFACT_STORE:?fleet.env did not load}"
: "${POST_CPT_CONVERT_SSH:?fleet.env did not load}"
: "${POST_CPT_CONVERT_ROOT:?fleet.env did not load}"
: "${POST_CPT_CONVERT_IMAGE:?fleet.env did not load}"
: "${RUN_TAG:?set RUN_TAG to the completed, qualified CPT run tag}"

case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|"")
    echo "REFUSE: unsafe RUN_TAG: $RUN_TAG" >&2
    exit 1
    ;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: production bake requires four rank-ordered Sparks." >&2
  exit 1
}
SSH_OPTIONS=(-o ControlMaster=no -o ControlPath=none)

QUALIFIED_RUN_TAG=cpt_v7_eps1fix
PLAN_SHA=63780ca438a6a2e5f362b4c4a9a9b13e1ff6b030aaa0c7d1bc16f9032df35550
BASE_MANIFEST_SHA=2406fff54148dd44d9c7a4824d43aaace0c450d3e6b174bfe1268565a9512c5d
FINAL_STEP=979
OUTPUT_DIR="${SPARK_HOME%/}/training_outputs/${RUN_TAG}_stage2_ddp_all_rows"
CHECKPOINT="$OUTPUT_DIR/checkpoint-$FINAL_STEP"
STATE_DIR="${POST_CPT_ARTIFACT_STORE%/}/runs/${RUN_TAG}/stage2_sft_ddp"
TRAINING_COMPLETION="$STATE_DIR/TRAINING_COMPLETE.json"
BASE="${POST_CPT_CONVERT_ROOT%/}/${RUN_TAG}_servable"
OUTPUT="${POST_CPT_CONVERT_ROOT%/}/${RUN_TAG}_stage2_ddp_servable"
REMOTE_TOOL_ROOT="${POST_CPT_CONVERT_ROOT%/}/tools/stage2-ddp"
REMOTE_ADAPTER_ROOT="${POST_CPT_CONVERT_ROOT%/}/adapters"
REMOTE_LOG_ROOT="${POST_CPT_CONVERT_ROOT%/}/logs"
SPACE_MARGIN_BYTES=${SPACE_MARGIN_BYTES:-10737418240}
BAKE_TIMEOUT_SECONDS=${BAKE_TIMEOUT_SECONDS:-10800}
NOTIFY_TRAINING_TARGET=${SFT_BAKE_TRAINING_NOTIFY_TARGET:-tutor}
NOTIFY_SERVING_TARGET=${SFT_BAKE_SERVING_NOTIFY_TARGET:-infra}

[ "$RUN_TAG" = "$QUALIFIED_RUN_TAG" ] || {
  echo "REFUSE: RUN_TAG=$RUN_TAG has not qualified this bake lifecycle." >&2
  exit 1
}
for path in "$OUTPUT_DIR" "$STATE_DIR" "$TRAINING_COMPLETION" "$BASE" "$OUTPUT" \
  "$REMOTE_TOOL_ROOT" "$REMOTE_ADAPTER_ROOT" "$REMOTE_LOG_ROOT"; do
  case "$path" in
    /*) ;;
    *)
      echo "REFUSE: production path is not absolute: $path" >&2
      exit 1
      ;;
  esac
done
for numeric in "$SPACE_MARGIN_BYTES" "$BAKE_TIMEOUT_SECONDS"; do
  case "$numeric" in
    *[!0-9]*|"")
      echo "REFUSE: malformed integer setting: $numeric" >&2
      exit 1
      ;;
  esac
  [ "$numeric" -gt 0 ]
done

DEPLOY_REF=${DEPLOY_SHA:-HEAD}
DEPLOY_SHA=$(git rev-parse --verify "${DEPLOY_REF}^{commit}")
case "$DEPLOY_SHA" in
  *[!0-9a-f]*|"")
    echo "REFUSE: deployment ref did not resolve to a commit." >&2
    exit 1
    ;;
esac
[ "${#DEPLOY_SHA}" = 40 ]
PRODUCTION_FILES=(
  careers-qwen/bake_stage2_ddp_production.sh
  careers-qwen/bake_stage2_ddp_worker.sh
  careers-qwen/bake_lora_nopeft.py
  careers-qwen/finalize_stage2_ddp_bake.py
)
for file in "${PRODUCTION_FILES[@]}"; do
  working_blob=$(git hash-object -- "$file")
  committed_blob=$(git rev-parse "${DEPLOY_SHA}:$file")
  [ "$working_blob" = "$committed_blob" ] || {
    echo "REFUSE: production file differs from $DEPLOY_SHA: $file" >&2
    exit 1
  }
done

mkdir -p "$STATE_DIR/bake-inputs" "$STATE_DIR/bake-logs"
exec 9>"$STATE_DIR/bake.lock"
flock -n 9 || {
  echo "REFUSE: another bake lifecycle holds $STATE_DIR/bake.lock." >&2
  exit 1
}
DRIVER_LOG="$STATE_DIR/bake.log"
say(){
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

sha256_file(){
  sha256sum "$1" | cut -d' ' -f1
}

verify_adapter_hold(){
  local hold=$1 adapter_sha=$2 config_sha=$3
  python3 - "$hold" "$adapter_sha" "$config_sha" "$RUN_TAG" "$FINAL_STEP" \
    "$PLAN_SHA" "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import os
import sys

hold, adapter_sha, config_sha, run_tag, steps, plan_sha, output = sys.argv[1:]
expected_files = {
    "ADAPTER_INPUT_SHA256SUMS",
    "COMPLETE",
    "SHA256SUMS.json",
    "TRAINING_COMPLETE.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_manifest.json",
}
actual_files = set(os.listdir(hold))
if actual_files not in (
    expected_files,
    expected_files - {"ADAPTER_INPUT_SHA256SUMS"},
):
    raise SystemExit(
        f"REFUSE: adapter hold files differ: {sorted(actual_files)}"
    )
def digest(name):
    value = hashlib.sha256()
    with open(os.path.join(hold, name), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
if digest("adapter_model.safetensors") != adapter_sha:
    raise SystemExit("REFUSE: held adapter digest differs")
if digest("adapter_config.json") != config_sha:
    raise SystemExit("REFUSE: held adapter config digest differs")
if open(os.path.join(hold, "COMPLETE")).read().strip() != (
    f"adapter_sha256={adapter_sha}"
):
    raise SystemExit("REFUSE: held COMPLETE differs")
sums = json.load(open(os.path.join(hold, "SHA256SUMS.json")))
manifest = json.load(open(os.path.join(hold, "training_manifest.json")))
completion = json.load(open(os.path.join(hold, "TRAINING_COMPLETE.json")))
if sums.get("adapter_model.safetensors") != adapter_sha:
    raise SystemExit("REFUSE: held SHA256SUMS differs")
if (
    manifest.get("completed_steps") != int(steps)
    or manifest.get("plan_sha256") != plan_sha
):
    raise SystemExit("REFUSE: held training manifest differs")
expected_completion = {
    "adapter_sha256": adapter_sha,
    "format": "stage2-ddp-lora-complete-v1",
    "output_dir": output,
    "plan_sha256": plan_sha,
    "run_tag": run_tag,
    "steps": int(steps),
}
actual_completion = {
    key: completion.get(key) for key in expected_completion
}
if actual_completion != expected_completion:
    raise SystemExit("REFUSE: held training completion differs")
PY
  if [ -s "$hold/ADAPTER_INPUT_SHA256SUMS" ]; then
    (
      cd "$hold"
      sha256sum -c ADAPTER_INPUT_SHA256SUMS >/dev/null
    )
  else
    (
      cd "$hold"
      sha256sum COMPLETE SHA256SUMS.json TRAINING_COMPLETE.json \
        adapter_config.json adapter_model.safetensors training_manifest.json \
        >ADAPTER_INPUT_SHA256SUMS
      sha256sum -c ADAPTER_INPUT_SHA256SUMS >/dev/null
    )
  fi
  sha256_file "$hold/ADAPTER_INPUT_SHA256SUMS"
}

read_completion(){
  python3 - "$1" "$RUN_TAG" "$OUTPUT_DIR" "$FINAL_STEP" "$PLAN_SHA" <<'PY'
import json
import sys

path, run_tag, output, steps, plan_sha = sys.argv[1:]
record = json.load(open(path))
expected = {
    "format": "stage2-ddp-lora-complete-v1",
    "output_dir": output,
    "plan_sha256": plan_sha,
    "run_tag": run_tag,
    "steps": int(steps),
}
actual = {key: record.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"REFUSE: completion mismatch: {actual} != {expected}")
adapter_sha = record.get("adapter_sha256", "")
deploy_sha = record.get("deploy_sha", "")
if len(adapter_sha) != 64 or any(value not in "0123456789abcdef" for value in adapter_sha):
    raise SystemExit("REFUSE: malformed final adapter digest")
if len(deploy_sha) != 40 or any(value not in "0123456789abcdef" for value in deploy_sha):
    raise SystemExit("REFUSE: malformed training deployment commit")
print(adapter_sha, deploy_sha)
PY
}

verify_final_fleet(){
  local rank node receipt
  local reference=
  local expected_completion_sha
  local remote_adapter remote_config remote_completion boot_id uptime_seconds
  local expected_metadata_bytes
  expected_completion_sha=$(sha256_file "$TRAINING_COMPLETION")
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh "${SSH_OPTIONS[@]}" -o BatchMode=yes -o ConnectTimeout=15 \
      spark@"$node" \
      "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "$CHECKPOINT" "$RUN_TAG" \
        "$FINAL_STEP" "$PLAN_SHA" "$expected_completion_sha" "$rank")" <<'REMOTE'
set -euo pipefail
output=$1
checkpoint=$2
run_tag=$3
steps=$4
plan_sha=$5
expected_completion_sha=$6
rank=$7
for file in COMPLETE adapter_config.json adapter_model.safetensors \
  trainer_state.pt training_manifest.json SHA256SUMS.json; do
  [ -s "$checkpoint/$file" ]
done
[ -s "$output/TRAINING_COMPLETE.json" ]
completion_sha=$(sha256sum "$output/TRAINING_COMPLETE.json" | cut -d' ' -f1)
[ "$completion_sha" = "$expected_completion_sha" ]
adapter_sha=$(sha256sum "$checkpoint/adapter_model.safetensors" | cut -d' ' -f1)
config_sha=$(sha256sum "$checkpoint/adapter_config.json" | cut -d' ' -f1)
complete_sha=$(sed -n 's/^adapter_sha256=//p' "$checkpoint/COMPLETE")
read -r receipt_adapter receipt_run receipt_steps receipt_plan manifest_step \
  manifest_plan listed_sha < <(
  python3 - "$output/TRAINING_COMPLETE.json" \
    "$checkpoint/training_manifest.json" "$checkpoint/SHA256SUMS.json" <<'PY'
import json
import sys
completion = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
sums = json.load(open(sys.argv[3]))
print(
    completion["adapter_sha256"],
    completion["run_tag"],
    completion["steps"],
    completion["plan_sha256"],
    manifest["completed_steps"],
    manifest["plan_sha256"],
    sums["adapter_model.safetensors"],
)
PY
)
[ "$receipt_adapter" = "$adapter_sha" ]
[ "$receipt_run" = "$run_tag" ]
[ "$receipt_steps" = "$steps" ]
[ "$receipt_plan" = "$plan_sha" ]
[ "$manifest_step" = "$steps" ]
[ "$manifest_plan" = "$plan_sha" ]
[ "$listed_sha" = "$adapter_sha" ]
[ "$complete_sha" = "$adapter_sha" ]
trainers=$(ps -eo args= | awk '
  /[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++}
  END{print n+0}')
[ "$trainers" = 0 ]
if tmux has-session -t stage2-ddp-sft 2>/dev/null; then
  echo "REFUSE: training tmux still exists on rank$rank" >&2
  exit 1
fi
protected=${PROTECTED_CHECKPOINT:?set PROTECTED_CHECKPOINT to the protected stage2 checkpoint path}
[ -s "$protected/COMPLETE" ]
[ -s "$protected/trainer_meta.pt" ]
[ -s "$protected/dcp/__${rank}_0.distcp" ]
protected_bytes=$(stat -c %s "$protected/dcp/__${rank}_0.distcp")
[ "$protected_bytes" = 13754046548 ]
case "$rank" in
  0) expected_metadata_bytes=1127670 ;;
  1) expected_metadata_bytes=1130940 ;;
  2|3) expected_metadata_bytes=1130948 ;;
  *) exit 1 ;;
esac
[ "$(stat -c %s "$protected/dcp/__${rank}.metadata")" = \
  "$expected_metadata_bytes" ]
[ "$(stat -c %s "$protected/trainer_meta.pt")" = 7245 ]
printf '%s %s %s %s %s\n' \
  "$adapter_sha" "$config_sha" "$completion_sha" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$(cut -d. -f1 /proc/uptime)"
REMOTE
)
    read -r remote_adapter remote_config remote_completion boot_id \
      uptime_seconds <<<"$receipt"
    say "final fleet rank$rank .$node adapter=$remote_adapter config=$remote_config completion=$remote_completion boot=$boot_id uptime=${uptime_seconds}s trainers=0 protected-checkpoint-800=exact" >&2
    if [ "$rank" = 0 ]; then
      reference="$remote_adapter $remote_config $remote_completion"
    else
      [ "$remote_adapter $remote_config $remote_completion" = "$reference" ] || {
        echo "REFUSE: final checkpoint receipts differ across ranks." >&2
        exit 1
      }
    fi
  done
  printf '%s\n' "$reference"
}

stage_controller_adapter(){
  local hold=$1 adapter_sha=$2 config_sha=$3
  if [ -d "$hold" ]; then
    verify_adapter_hold "$hold" "$adapter_sha" "$config_sha" >/dev/null
    say "controller adapter hold already verified: $hold"
    return
  fi
  local stage
  stage=$(mktemp -d "$STATE_DIR/bake-inputs/.checkpoint-${FINAL_STEP}.staging.XXXXXX")
  rsync -a --protect-args \
    -e "ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes" \
    --include=/adapter_model.safetensors \
    --include=/adapter_config.json \
    --include=/training_manifest.json \
    --include=/SHA256SUMS.json \
    --include=/COMPLETE \
    --exclude='*' \
    "spark@${SPARK_MASTER}:$CHECKPOINT/" "$stage/"
  cp "$TRAINING_COMPLETION" "$stage/TRAINING_COMPLETE.json"
  verify_adapter_hold "$stage" "$adapter_sha" "$config_sha" >/dev/null
  mv "$stage" "$hold"
  say "controller adapter hold staged and hash-verified: $hold"
}

deploy_tools(){
  local tool local_sha remote_sha
  local remote_dir=$1
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "mkdir -p '$(dirname "$remote_dir")' '$REMOTE_ADAPTER_ROOT' '$REMOTE_LOG_ROOT'"
  if ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "test -d '$remote_dir'"; then
    for tool in "${PRODUCTION_FILES[@]:1}"; do
      local_sha=$(sha256_file "$tool")
      remote_sha=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
        "sha256sum '$remote_dir/$(basename "$tool")' | cut -d' ' -f1")
      [ "$local_sha" = "$remote_sha" ] || {
        echo "REFUSE: immutable bake tool differs: $tool" >&2
        exit 1
      }
    done
    say "existing immutable bake tools verified: $remote_dir"
    return
  fi
  local stage="${remote_dir}.staging.$$"
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "test ! -e '$stage'; mkdir '$stage'"
  for tool in "${PRODUCTION_FILES[@]:1}"; do
    scp -q -o BatchMode=yes "$tool" \
      "$POST_CPT_CONVERT_SSH:$stage/$(basename "$tool")"
    local_sha=$(sha256_file "$tool")
    remote_sha=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
      "sha256sum '$stage/$(basename "$tool")' | cut -d' ' -f1")
    [ "$local_sha" = "$remote_sha" ] || {
      echo "REFUSE: deployed bake tool hash mismatch: $tool" >&2
      exit 1
    }
  done
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "mv '$stage' '$remote_dir'"
  say "immutable bake tools deployed from commit=$DEPLOY_SHA dir=$remote_dir"
}

stage_remote_adapter(){
  local hold=$1 remote_adapter=$2 expected_manifest=$3
  if ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "test -d '$remote_adapter'"; then
    receipt=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
      "cd '$remote_adapter';
       sha256sum -c ADAPTER_INPUT_SHA256SUMS >/dev/null;
       sha256sum ADAPTER_INPUT_SHA256SUMS | cut -d' ' -f1")
    [ "$receipt" = "$expected_manifest" ]
    say "conversion-host adapter hold already verified: $remote_adapter"
    return
  fi
  local stage="${remote_adapter}.staging.$$"
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "test ! -e '$stage'; mkdir '$stage'"
  rsync -a --protect-args -e "ssh -o BatchMode=yes" \
    "$hold/" "$POST_CPT_CONVERT_SSH:$stage/"
  receipt=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "cd '$stage';
     sha256sum -c ADAPTER_INPUT_SHA256SUMS >/dev/null;
     sha256sum ADAPTER_INPUT_SHA256SUMS | cut -d' ' -f1")
  [ "$receipt" = "$expected_manifest" ]
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "mv '$stage' '$remote_adapter'"
  say "conversion-host adapter staged and hash-verified: $remote_adapter"
}

launch_and_monitor(){
  local remote_worker=$1 remote_adapter=$2 adapter_sha=$3 config_sha=$4
  local stage="${OUTPUT}.staging.${adapter_sha:0:12}"
  local label="stage2_ddp_bake_${RUN_TAG}_${adapter_sha:0:12}"
  local log="$REMOTE_LOG_ROOT/${label}.log"
  local exit_file="$REMOTE_LOG_ROOT/${label}.exit"
  local pid_file="$REMOTE_LOG_ROOT/${label}.pid"
  local deadline state failure
  local -a worker_args=(
    bash
    "$remote_worker"
    "$POST_CPT_CONVERT_ROOT"
    "$POST_CPT_CONVERT_IMAGE"
    "$(dirname "$remote_worker")"
    "$BASE"
    "$remote_adapter"
    "$stage"
    "$OUTPUT"
    "$remote_adapter/TRAINING_COMPLETE.json"
    "$adapter_sha"
    "$config_sha"
    "$RUN_TAG"
    "$PLAN_SHA"
    "$BASE_MANIFEST_SHA"
    "$DEPLOY_SHA"
  )
  local command
  printf -v command '%q ' "${worker_args[@]}"
  command+=" >$(printf '%q' "$log") 2>&1; "
  command+="rc=\$?; printf '%s\\n' \"\$rc\" >$(printf '%q' "$exit_file"); exit \"\$rc\""
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "$(printf '%q ' bash -s -- "$label" "$log" "$exit_file" "$pid_file" \
      "$stage" "$OUTPUT" "$command")" <<'REMOTE'
set -euo pipefail
label=$1
log=$2
exit_file=$3
pid_file=$4
stage=$5
output=$6
command=$7
[ ! -e "$log" ] && [ ! -e "$exit_file" ] && [ ! -e "$pid_file" ]
[ ! -e "$stage" ] && [ ! -e "$output" ]
command -v nohup >/dev/null
command -v setsid >/dev/null
nohup setsid bash -c "$command" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
sleep 1
kill -0 "$pid" 2>/dev/null || [ -f "$exit_file" ]
REMOTE
  say "bake worker launched detached label=$label log=$log pid_receipt=$pid_file"

  deadline=$((SECONDS + BAKE_TIMEOUT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    state=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
      "if [ -f '$exit_file' ]; then printf 'EXIT %s\\n' \"\$(cat '$exit_file')\";
       elif [ -f '$pid_file' ] && kill -0 \"\$(cat '$pid_file')\" 2>/dev/null;
         then echo ALIVE;
       else echo MISSING; fi")
    failure=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
      "test -f '$log';
       awk '/Traceback|RuntimeError|REFUSE:|No space left|Killed/{line=\$0}
            END{print line}' '$log'")
    [ -z "$failure" ] || {
      echo "ABORT: bake failure signal: $failure" >&2
      exit 1
    }
    case "$state" in
      ALIVE)
        progress=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
          "test -f '$log';
           awk '/Processing model-|Total LoRA|SFT_MERGE_VALIDATED|SFT_BAKE_PROMOTED/{line=\$0}
                END{print line}' '$log'")
        say "bake worker alive ${progress:-initializing}"
        ;;
      "EXIT 0")
        say "bake worker exit=0"
        return
        ;;
      "EXIT "*)
        echo "ABORT: bake worker ${state}." >&2
        exit 1
        ;;
      *)
        echo "ABORT: bake worker disappeared without an exit receipt." >&2
        exit 1
        ;;
    esac
    sleep 30
  done
  echo "ABORT: bake worker exceeded ${BAKE_TIMEOUT_SECONDS}s." >&2
  exit 1
}

verify_promoted(){
  local adapter_sha=$1
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "$(printf '%q ' bash -s -- "$OUTPUT" "$RUN_TAG" "$FINAL_STEP" "$PLAN_SHA" \
      "$adapter_sha")" <<'REMOTE'
set -euo pipefail
output=$1
run_tag=$2
steps=$3
plan_sha=$4
adapter_sha=$5
[ -s "$output/SFT_MERGE_COMPLETE.json" ]
[ -s "$output/training_provenance.json" ]
[ -s "$output/merged_weight_diff.json" ]
[ -s "$output/lora_merge_receipt.json" ]
[ -s "$output/SOURCE_SHA256SUMS" ]
(
  cd "$output"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
python3 - "$output" "$run_tag" "$steps" "$plan_sha" "$adapter_sha" <<'PY'
import json
import os
import sys
output, run_tag, steps, plan_sha, adapter_sha = sys.argv[1:]
index = json.load(open(os.path.join(output, "model.safetensors.index.json")))
complete = json.load(open(os.path.join(output, "SFT_MERGE_COMPLETE.json")))
diff = json.load(open(os.path.join(output, "merged_weight_diff.json")))
expected = {
    "adapter_sha256": adapter_sha,
    "artifact_tensors": 1199,
    "format": "stage2-ddp-sft-merge-complete-v1",
    "plan_sha256": plan_sha,
    "run_tag": run_tag,
    "steps": int(steps),
    "target_matrices": 352,
}
actual = {key: complete.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"REFUSE: promoted completion differs: {actual} != {expected}")
if len(index["weight_map"]) != 1199:
    raise SystemExit("REFUSE: promoted model does not contain 1199 tensors")
print(
    f"{diff['abs_mean_dW']:.17g}",
    f"{diff['abs_max_dW']:.17g}",
    len(index["weight_map"]),
)
PY
REMOTE
}

command -v taey-notify >/dev/null || {
  echo "REFUSE: taey-notify is required for the serving handoff." >&2
  exit 1
}
[ -s "$TRAINING_COMPLETION" ] || {
  echo "REFUSE: final training completion receipt is absent: $TRAINING_COMPLETION" >&2
  exit 1
}
completion_receipt=$(read_completion "$TRAINING_COMPLETION")
read -r ADAPTER_SHA TRAINING_DEPLOY_SHA <<<"$completion_receipt"
say "training completion bound adapter=$ADAPTER_SHA training_commit=$TRAINING_DEPLOY_SHA"
fleet_receipt=$(verify_final_fleet)
read -r FLEET_ADAPTER_SHA ADAPTER_CONFIG_SHA COMPLETION_SHA <<<"$fleet_receipt"
[ "$FLEET_ADAPTER_SHA" = "$ADAPTER_SHA" ] || {
  echo "REFUSE: fleet adapter does not match controller completion." >&2
  exit 1
}
[ "$COMPLETION_SHA" = "$(sha256_file "$TRAINING_COMPLETION")" ] || {
  echo "REFUSE: fleet completion does not match controller completion." >&2
  exit 1
}

HOLD="$STATE_DIR/bake-inputs/checkpoint-${FINAL_STEP}_${ADAPTER_SHA}"
REMOTE_TOOLS="$REMOTE_TOOL_ROOT/$DEPLOY_SHA"
REMOTE_ADAPTER="$REMOTE_ADAPTER_ROOT/${RUN_TAG}_stage2_ddp_checkpoint-${FINAL_STEP}_${ADAPTER_SHA}"
stage_controller_adapter "$HOLD" "$ADAPTER_SHA" "$ADAPTER_CONFIG_SHA"
HOLD_MANIFEST_SHA=$(verify_adapter_hold "$HOLD" "$ADAPTER_SHA" \
  "$ADAPTER_CONFIG_SHA")
deploy_tools "$REMOTE_TOOLS"
stage_remote_adapter "$HOLD" "$REMOTE_ADAPTER" "$HOLD_MANIFEST_SHA"

base_receipt=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "$(printf '%q ' bash -s -- "$BASE" "$BASE_MANIFEST_SHA" \
    "$POST_CPT_CONVERT_ROOT" "$SPACE_MARGIN_BYTES" "$POST_CPT_CONVERT_IMAGE" \
    "$OUTPUT")" <<'REMOTE'
set -euo pipefail
base=$1
expected_manifest=$2
root=$3
margin=$4
image=$5
output=$6
[ -s "$base/model.safetensors.index.json" ]
[ -z "$(find "$base" -type l -print -quit)" ]
manifest_tmp=$(mktemp)
trap 'rm -f -- "$manifest_tmp"' EXIT
(
  cd "$base"
  find . -type f ! -name SOURCE_SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 -r sha256sum >"$manifest_tmp"
)
[ "$(sha256sum "$manifest_tmp" | cut -d' ' -f1)" = "$expected_manifest" ]
tensor_count=$(python3 - "$base/model.safetensors.index.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))["weight_map"]))
PY
)
[ "$tensor_count" = 1199 ]
base_bytes=$(du -sb "$base" | cut -f1)
free_bytes=$(df -B1 --output=avail "$root" | tail -1 | tr -d ' ')
if [ ! -e "$output" ]; then
  [ "$free_bytes" -ge $((base_bytes + margin)) ]
fi
sudo docker image inspect "$image" >/dev/null
printf '%s %s %s\n' "$tensor_count" "$base_bytes" "$free_bytes"
REMOTE
)
say "conversion preflight base_tensors/base_bytes/free_bytes=$base_receipt image=$POST_CPT_CONVERT_IMAGE"

if ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "test -e '$OUTPUT'"; then
  say "promoted output already exists; validating idempotent result"
else
  launch_and_monitor "$REMOTE_TOOLS/bake_stage2_ddp_worker.sh" \
    "$REMOTE_ADAPTER" "$ADAPTER_SHA" "$ADAPTER_CONFIG_SHA"
fi
promoted_receipt=$(verify_promoted "$ADAPTER_SHA")
read -r MERGED_MEAN MERGED_MAX TENSORS <<<"$promoted_receipt"
MANIFEST_SHA=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "sha256sum '$OUTPUT/SOURCE_SHA256SUMS' | cut -d' ' -f1")
say "SFT BAKE COMPLETE output=$OUTPUT tensors=$TENSORS mean_abs_dW=$MERGED_MEAN max_abs_dW=$MERGED_MAX manifest_sha=$MANIFEST_SHA"

taey-notify "$NOTIFY_TRAINING_TARGET" \
  "SFT BAKE COMPLETE: ${RUN_TAG} checkpoint-${FINAL_STEP} adapter=${ADAPTER_SHA}; candidate=${POST_CPT_CONVERT_SSH}:${OUTPUT}; tensors=${TENSORS}; merged_mean_abs_dW=${MERGED_MEAN}; manifest_sha256=${MANIFEST_SHA}; serving remains infra-owned." \
  --type response_ready
taey-notify "$NOTIFY_SERVING_TARGET" \
  "SFT BAKED READY: ${RUN_TAG} checkpoint-${FINAL_STEP}; candidate=${POST_CPT_CONVERT_SSH}:${OUTPUT}; tensors=${TENSORS}; adapter_sha256=${ADAPTER_SHA}; manifest_sha256=${MANIFEST_SHA}; training+bake validation complete, serving handoff is infra-owned." \
  --type response_ready

echo "DONE"
echo "  candidate          $POST_CPT_CONVERT_SSH:$OUTPUT"
echo "  adapter            $ADAPTER_SHA"
echo "  tensors            $TENSORS"
echo "  merged mean |dW|   $MERGED_MEAN"
echo "  merged max  |dW|   $MERGED_MAX"
echo "  manifest           $MANIFEST_SHA"
echo "  next               infra owns serving"
