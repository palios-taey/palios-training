#!/usr/bin/env bash
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${POST_CPT_ARTIFACT_STORE:?fleet.env did not define POST_CPT_ARTIFACT_STORE}"
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
  echo "REFUSE: production SFT requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}

BATCH_SIZE_PER_RANK=${BATCH_SIZE_PER_RANK:-1}
MAX_SEQ=${MAX_SEQ:-4096}
FIRST_SESSION_STEPS=${FIRST_SESSION_STEPS:-50}
MAX_SESSION_STEPS=${MAX_SESSION_STEPS:-250}
MAX_SESSIONS=${MAX_SESSIONS:-12}
STEP_TIMEOUT_SECONDS=${STEP_TIMEOUT_SECONDS:-75}
CHECKPOINT_MARGIN_SECONDS=${CHECKPOINT_MARGIN_SECONDS:-1800}
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-}
COOL_AT_C=${COOL_AT_C:-52}
CLOCK_CAP=${CLOCK_CAP:-1600}
OUTPUT_DIR=${OUTPUT_DIR:-${SPARK_HOME%/}/training_outputs/${RUN_TAG}_stage2_all_rows}
case "$OUTPUT_DIR" in
  "${SPARK_HOME%/}"/training_outputs/*) ;;
  *)
    echo "REFUSE: OUTPUT_DIR escaped the Spark training-output root: $OUTPUT_DIR" >&2
    exit 1
    ;;
esac
DEPLOY_REF=${DEPLOY_SHA:-HEAD}
DEPLOY_SHA=$(git rev-parse --verify "${DEPLOY_REF}^{commit}")
case "$DEPLOY_SHA" in
  *[!0-9a-f]*|'') echo "REFUSE: deployment ref did not resolve to a full commit SHA." >&2; exit 1;;
esac
[ "${#DEPLOY_SHA}" = 40 ] || {
  echo "REFUSE: deployment ref did not resolve to a 40-character commit SHA." >&2
  exit 1
}
IMMUTABLE_FILES=(
  careers-qwen/run_stage2_sft_till_done.sh
  careers-qwen/stage_sft_base.sh
  careers-qwen/sft_dataset_receipt.py
  careers-qwen/run_stage2_sft_production.sh
  careers-qwen/launch_stage2_sft.sh
  careers-qwen/gate_negative_controls.py
  dense-9b/trainers/train_fsdp_dense_9b.py
  dense-9b/recipes/run_4node_27b_cpt.sh
  dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh
  dense-9b/configs/fsdp_dense_27b.yaml
)
for file in "${IMMUTABLE_FILES[@]}"; do
  committed_blob=$(git rev-parse "${DEPLOY_SHA}:$file")
  working_blob=$(git hash-object -- "$file")
  [ "$working_blob" = "$committed_blob" ] || {
    echo "REFUSE: production file differs from $DEPLOY_SHA: $file" >&2
    exit 1
  }
done
STATE_DIR=${STATE_DIR:-${POST_CPT_ARTIFACT_STORE%/}/runs/${RUN_TAG}/stage2_sft}
LOGDIR="${SPARK_HOME%/}/cpt27b_logs"
for numeric in "$BATCH_SIZE_PER_RANK" "$MAX_SEQ" "$FIRST_SESSION_STEPS" \
  "$MAX_SESSION_STEPS" "$MAX_SESSIONS" \
  "$STEP_TIMEOUT_SECONDS" "$CHECKPOINT_MARGIN_SECONDS" "$COOL_AT_C" \
  "$CLOCK_CAP"; do
  case "$numeric" in *[!0-9]*|'')
    echo "REFUSE: numeric production settings must be positive integers." >&2
    exit 1
    ;;
  esac
  [ "$numeric" -gt 0 ]
done
[ "$MAX_SEQ" -gt 256 ] || {
  echo "REFUSE: MAX_SEQ must exceed the 256-token overlap." >&2
  exit 1
}
[ "$FIRST_SESSION_STEPS" -ge 40 ] || {
  echo "REFUSE: FIRST_SESSION_STEPS must reach the mandatory step-40 update gate." >&2
  exit 1
}
if [ -n "$SESSION_TIMEOUT_SECONDS" ]; then
  case "$SESSION_TIMEOUT_SECONDS" in *[!0-9]*|'')
    echo "REFUSE: SESSION_TIMEOUT_SECONDS must be a positive integer when set." >&2
    exit 1
    ;;
  esac
  [ "$SESSION_TIMEOUT_SECONDS" -gt 0 ]
fi

mkdir -p "$STATE_DIR/logs"
exec 9>"$STATE_DIR/driver.lock"
flock -n 9 || {
  echo "REFUSE: another production SFT driver holds $STATE_DIR/driver.lock." >&2
  exit 1
}
DRIVER_LOG="$STATE_DIR/driver.log"
say(){
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

latest_complete(){
  local rank node value reference=
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    value=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "$rank" "$TOTAL_STEPS")" <<'REMOTE'
set -euo pipefail
output=$1
rank=$2
total=$3
latest=0
if [ -d "$output" ]; then
  for candidate in "$output"/checkpoint-*; do
    [ -d "$candidate" ] || continue
    step=${candidate##*-}
    case "$step" in *[!0-9]*|'') continue;; esac
    [ -f "$candidate/COMPLETE" ] &&
    [ -f "$candidate/trainer_meta.pt" ] &&
    [ -f "$candidate/dcp/__${rank}_0.distcp" ] || {
      echo "incomplete checkpoint directory: $candidate" >&2
      exit 1
    }
    [ "$step" -le "$total" ] || {
      echo "checkpoint-$step exceeds horizon $total" >&2
      exit 1
    }
    [ "$step" -le "$latest" ] || latest=$step
  done
  if [ -e "$output/final" ] && [ "$latest" != "$total" ]; then
    echo "final artifact exists before checkpoint-$total" >&2
    exit 1
  fi
fi
printf '%s\n' "$latest"
REMOTE
)
    case "$value" in *[!0-9]*|'')
      echo "REFUSE: invalid latest-checkpoint receipt from rank$rank .$node: $value" >&2
      return 1
      ;;
    esac
    if [ "$rank" = 0 ]; then
      reference=$value
    elif [ "$value" != "$reference" ]; then
      echo "REFUSE: checkpoint split-brain: rank0=$reference rank$rank=$value." >&2
      return 1
    fi
  done
  printf '%s\n' "$reference"
}

echo "=== SFT CORPUS RECEIPT — four ranks must name identical bytes ==="
CORPUS_RECEIPT=
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "test -f '$SFT_CORPUS' &&
     printf '%s %s %s\\n' \
       \"\$(stat -c %s '$SFT_CORPUS')\" \
       \"\$(sha256sum '$SFT_CORPUS' | cut -d' ' -f1)\" \
       \"\$(wc -l < '$SFT_CORPUS')\"")
  if [ "$rank" = 0 ]; then
    CORPUS_RECEIPT=$receipt
  else
    [ "$receipt" = "$CORPUS_RECEIPT" ] || {
      echo "REFUSE: rank$rank .$node corpus receipt differs from rank0." >&2
      exit 1
    }
  fi
  echo "  rank$rank .$node $receipt"
done
read -r corpus_bytes corpus_sha corpus_rows <<<"$CORPUS_RECEIPT"

say "staging the exact baked base before tokenizer-derived sizing"
RUN_TAG=$RUN_TAG bash careers-qwen/stage_sft_base.sh
base_stage_done=1
BASE_MODEL="${SPARK_HOME%/}/models/${RUN_TAG}_servable"
receipt_stage=$(mktemp -d)
cleanup_receipt_stage(){
  rm -rf -- "$receipt_stage"
}
trap cleanup_receipt_stage EXIT
git show "$DEPLOY_SHA:dense-9b/trainers/train_fsdp_dense_9b.py" \
  >"$receipt_stage/train_fsdp_dense_9b.py"
git show "$DEPLOY_SHA:careers-qwen/sft_dataset_receipt.py" \
  >"$receipt_stage/sft_dataset_receipt.py"
DATASET_RECEIPT=
remote_receipt_dir="/tmp/stage2-sft-receipt-${DEPLOY_SHA}"
echo "=== SFT DATASET RECEIPT — exact tokenizer/trainer semantics on all four ranks ==="
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "mkdir -p '$remote_receipt_dir'"
  scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$receipt_stage/train_fsdp_dense_9b.py" \
    "$receipt_stage/sft_dataset_receipt.py" \
    "spark@$node:$remote_receipt_dir/"
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=30 spark@"$node" \
    "python3 '$remote_receipt_dir/sft_dataset_receipt.py' \
       --trainer '$remote_receipt_dir/train_fsdp_dense_9b.py' \
       --corpus '$SFT_CORPUS' \
       --model '$BASE_MODEL' \
       --max-seq '$MAX_SEQ'")
  if [ "$rank" = 0 ]; then
    DATASET_RECEIPT=$receipt
  else
    [ "$receipt" = "$DATASET_RECEIPT" ] || {
      echo "REFUSE: rank$rank .$node tokenizer-derived dataset receipt differs from rank0." >&2
      exit 1
    }
  fi
  echo "  rank$rank .$node $receipt"
done
read -r receipt_tag receipt_corpus_sha receipt_rows EXPECTED_SFT_SAMPLES \
  extra_samples over_max_rows max_tokens assistant_tokens zero_assistant_samples \
  min_assistant_tokens sample_shape_sha \
  <<<"$DATASET_RECEIPT"
field_count=$(wc -w <<<"$DATASET_RECEIPT")
[ "$field_count" = 11 ] || {
  echo "REFUSE: tokenizer-derived receipt has $field_count fields; expected 11." >&2
  exit 1
}
for numeric in "$receipt_rows" "$EXPECTED_SFT_SAMPLES" "$extra_samples" \
  "$over_max_rows" "$max_tokens" "$assistant_tokens" \
  "$zero_assistant_samples" "$min_assistant_tokens"; do
  case "$numeric" in *[!0-9]*|'')
    echo "REFUSE: tokenizer-derived receipt contains a non-integer field." >&2
    exit 1
    ;;
  esac
done
case "$receipt_corpus_sha:$sample_shape_sha" in
  *[!0-9a-f:]*|*:*:*)
    echo "REFUSE: tokenizer-derived receipt contains a malformed digest." >&2
    exit 1
    ;;
esac
[ "${#receipt_corpus_sha}" = 64 ] &&
[ "${#sample_shape_sha}" = 64 ] || {
  echo "REFUSE: tokenizer-derived receipt digests must be full SHA-256 values." >&2
  exit 1
}
[ "$receipt_tag" = SFT_DATASET_RECEIPT ] &&
[ "$receipt_corpus_sha" = "$corpus_sha" ] &&
[ "$receipt_rows" = "$corpus_rows" ] &&
[ "$EXPECTED_SFT_SAMPLES" -ge "$corpus_rows" ] &&
[ "$extra_samples" = $((EXPECTED_SFT_SAMPLES - corpus_rows)) ] &&
[ "$over_max_rows" -le "$corpus_rows" ] &&
{ [ "$over_max_rows" = 0 ] || [ "$max_tokens" -gt "$MAX_SEQ" ]; } &&
[ "$assistant_tokens" -gt 0 ] &&
[ "$zero_assistant_samples" = 0 ] &&
[ "$min_assistant_tokens" -gt 0 ] || {
  echo "REFUSE: tokenizer-derived receipt does not bind the corpus in hand." >&2
  exit 1
}
for node in "${NODES[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "rm -rf -- '$remote_receipt_dir'"
done
GLOBAL_BATCH=$((BATCH_SIZE_PER_RANK * 4))
TOTAL_STEPS=$(( (EXPECTED_SFT_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
[ "$TOTAL_STEPS" -gt 0 ]
say "campaign run=$RUN_TAG rows=$corpus_rows samples=$EXPECTED_SFT_SAMPLES extra_windows=$extra_samples over_max_rows=$over_max_rows max_tokens=$max_tokens assistant_tokens=$assistant_tokens max_seq=$MAX_SEQ sample_shape_sha256=$sample_shape_sha global_batch=$GLOBAL_BATCH total_steps=$TOTAL_STEPS output=$OUTPUT_DIR"

archive_logs(){
  local label=$1 rank node
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    scp -q -o BatchMode=yes -o ConnectTimeout=10 \
      "spark@$node:$LOGDIR/r$rank.log" \
      "$STATE_DIR/logs/rank${rank}_${label}.log" 2>/dev/null ||
      say "WARN: could not archive rank$rank log for $label"
  done
}

rotate_checkpoints(){
  local expected_step=$1 rank node receipt
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "$(printf '%q ' bash -s -- "$OUTPUT_DIR" "${SPARK_HOME%/}/training_outputs" "$rank" "$expected_step")" <<'REMOTE'
set -euo pipefail
output=$1
training_root=$2
rank=$3
expected=$4
case "$output" in
  "$training_root"/*) ;;
  *) echo "checkpoint output escaped training root" >&2; exit 1;;
esac
complete_steps=()
for candidate in "$output"/checkpoint-*; do
  [ -d "$candidate" ] || continue
  step=${candidate##*-}
  case "$step" in *[!0-9]*|'') echo "malformed checkpoint path: $candidate" >&2; exit 1;; esac
  [ -f "$candidate/COMPLETE" ] &&
  [ -f "$candidate/trainer_meta.pt" ] &&
  [ -f "$candidate/dcp/__${rank}_0.distcp" ] || {
    echo "refusing rotation with incomplete checkpoint: $candidate" >&2
    exit 1
  }
  complete_steps+=("$step")
done
[ "${#complete_steps[@]}" -gt 0 ]
mapfile -t complete_steps < <(printf '%s\n' "${complete_steps[@]}" | sort -n)
latest=${complete_steps[$((${#complete_steps[@]} - 1))]}
[ "$latest" = "$expected" ] || {
  echo "latest checkpoint-$latest differs from expected checkpoint-$expected" >&2
  exit 1
}
remove_count=$((${#complete_steps[@]} - 2))
[ "$remove_count" -gt 0 ] || remove_count=0
removed=
for ((index=0; index<remove_count; index++)); do
  step=${complete_steps[$index]}
  target="$output/checkpoint-$step"
  case "$target" in "$output"/checkpoint-[0-9]*) ;; *) exit 1;; esac
  [ -f "$target/COMPLETE" ] &&
  [ -f "$target/dcp/__${rank}_0.distcp" ]
  rm -rf -- "$target"
  removed="${removed}${removed:+,}$step"
done
kept=$(printf '%s\n' "${complete_steps[@]}" | tail -2 | paste -sd, -)
printf 'kept=%s removed=%s\n' "$kept" "${removed:-none}"
REMOTE
)
    say "checkpoint rotation rank$rank .$node $receipt"
  done
}

session=0
while true; do
  current=$(latest_complete)
  [ "$current" -le "$TOTAL_STEPS" ] || {
    say "ABORT: latest checkpoint-$current exceeds derived horizon $TOTAL_STEPS"
    exit 1
  }
  if [ "$current" = "$TOTAL_STEPS" ]; then
    break
  fi
  session=$((session + 1))
  [ "$session" -le "$MAX_SESSIONS" ] || {
    say "ABORT: reached MAX_SESSIONS=$MAX_SESSIONS at checkpoint-$current"
    exit 1
  }
  remaining=$((TOTAL_STEPS - current))
  if [ "$current" = 0 ] && [ "$remaining" -gt "$FIRST_SESSION_STEPS" ]; then
    session_steps=$FIRST_SESSION_STEPS
  elif [ "$remaining" -lt "$MAX_SESSION_STEPS" ]; then
    session_steps=$remaining
  else
    session_steps=$MAX_SESSION_STEPS
  fi
  if [ -n "$SESSION_TIMEOUT_SECONDS" ]; then
    session_timeout=$SESSION_TIMEOUT_SECONDS
  else
    session_timeout=$((session_steps * STEP_TIMEOUT_SECONDS + CHECKPOINT_MARGIN_SECONDS))
  fi
  target=$((current + session_steps))
  resume=
  require_gate=1
  if [ "$current" -gt 0 ]; then
    resume="$OUTPUT_DIR/checkpoint-$current"
    require_gate=0
  fi
  say "session=$session checkpoint-$current -> target=$target session_steps=$session_steps timeout=${session_timeout}s"

  launch_stamp=$(date -u +%Y%m%dT%H%M%SZ)
  launch_log="$STATE_DIR/logs/controller_session_${current}_to_${target}_${launch_stamp}.log"
  launch_env=(
    env
    "RUN_TAG=$RUN_TAG"
    "SFT_CORPUS=$SFT_CORPUS"
    "BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK"
    "MAX_SEQ=$MAX_SEQ"
    "EXPECTED_SFT_SAMPLES=$EXPECTED_SFT_SAMPLES"
    "CLOCK_CAP=$CLOCK_CAP"
    "OUTPUT_DIR=$OUTPUT_DIR"
    "DEPLOY_SHA=$DEPLOY_SHA"
    "SESSION_LIMIT=$session_steps"
    "SAVE_EVERY=$session_steps"
    "REQUIRE_STEP10_GATE=$require_gate"
    "SKIP_BASE_STAGE=$base_stage_done"
  )
  [ -z "$resume" ] || launch_env+=("RESUME_DELTA=$resume")
  if ! "${launch_env[@]}" bash careers-qwen/run_stage2_sft_production.sh \
      2>&1 | tee "$launch_log"; then
    archive_logs "failed_launch_${current}_to_${target}_${launch_stamp}"
    say "ABORT: production launch or live gate failed; see $launch_log"
    exit 1
  fi
  base_stage_done=1
  if [ "$require_gate" = 1 ]; then
    dose=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$SPARK_MASTER" \
      "grep -aE '\\[SR-DELTA\\].*mean\\|dW\\|' '$LOGDIR/r0.log' | tail -1")
    say "step-10 gate: $dose"
    if command -v taey-notify >/dev/null; then
      taey-notify tutor \
        "SFT LIVE: ${RUN_TAG} reached step-10 on 4 Sparks; ${dose}; output=${OUTPUT_DIR}" \
        --type response_ready || say "WARN: taey-notify tutor failed"
    fi
  else
    say "resumed four-rank session is live from checkpoint-$current"
  fi

  say "watching remote tmux sessions until checkpoint target $target"
  deadline=$((SECONDS + session_timeout))
  partial_since=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    alive=0
    for node in "${NODES[@]}"; do
      state=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
        "tmux has-session -t cpt27b 2>/dev/null && echo ALIVE || echo EXITED" \
        2>/dev/null || true)
      [ "$state" = ALIVE ] && alive=$((alive + 1))
    done
    failure=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
      "grep -aE 'Traceback|CUDA out of memory|RuntimeError|ERROR:|ABORT:' '$LOGDIR/r0.log' 2>/dev/null | tail -1" \
      2>/dev/null || true)
    [ -z "$failure" ] || {
      archive_logs "failed_runtime_${current}_to_${target}_${launch_stamp}"
      say "ABORT: rank-0 failure signal: $failure"
      exit 1
    }
    step_line=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$SPARK_MASTER" \
      "grep -aE '\\[step [0-9]+\\]|DCP save COMPLETE|FRAGMENTATION EXIT' '$LOGDIR/r0.log' 2>/dev/null | tail -1" \
      2>/dev/null || true)
    say "session=$session alive=$alive/4 ${step_line:-no-step-receipt}"
    [ "$alive" -gt 0 ] || break
    if [ "$alive" -lt 4 ]; then
      [ "$partial_since" -ne 0 ] || partial_since=$SECONDS
      [ $((SECONDS - partial_since)) -lt 120 ] || {
        archive_logs "split_ranks_${current}_to_${target}_${launch_stamp}"
        say "ABORT: ranks remained split ($alive/4 alive) for 120 seconds"
        exit 1
      }
    else
      partial_since=0
    fi
    sleep 60
  done
  [ "$alive" = 0 ] || {
    archive_logs "timeout_${current}_to_${target}_${launch_stamp}"
    say "ABORT: session exceeded ${session_timeout}s with $alive/4 tmux sessions alive"
    exit 1
  }

  archive_logs "session_${current}_to_${target}_${launch_stamp}"
  new=$(latest_complete)
  [ "$new" -gt "$current" ] || {
    archive_logs "no_progress_${current}_to_${target}_${launch_stamp}"
    say "ABORT: session exited without a new four-rank COMPLETE checkpoint (still $current)"
    exit 1
  }
  [ "$new" -le "$target" ] || {
    say "ABORT: session advanced to checkpoint-$new beyond declared target $target"
    exit 1
  }
  if [ "$current" = 0 ] && [ "$new" != "$target" ]; then
    say "ABORT: first production session stopped at checkpoint-$new before its declared batch/memory gate checkpoint-$target"
    exit 1
  fi
  say "checkpoint-$new COMPLETE on all four; logs archived under $STATE_DIR/logs"
  if [ "$current" = 0 ]; then
    dose40=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$SPARK_MASTER" \
      "grep -aE '\\[SR-DELTA\\].*PASS \\(in Gaia band' '$LOGDIR/r0.log' | tail -1")
    [ -n "$dose40" ] || {
      say "ABORT: first production session reached checkpoint-$new without the required step-40 SR-DELTA PASS"
      exit 1
    }
    say "step-40 gate: $dose40"
  fi
  rotate_checkpoints "$new"
  if command -v taey-notify >/dev/null; then
    taey-notify tutor \
      "SFT PROGRESS: ${RUN_TAG} checkpoint-${new}/${TOTAL_STEPS} COMPLETE on all 4; logs=${STATE_DIR}/logs" \
      --type response_ready || say "WARN: taey-notify tutor failed"
  fi

  [ "$new" = "$TOTAL_STEPS" ] && continue
  say "cooldown: minimum 300s, then all hottest thermal zones must be <= ${COOL_AT_C}C"
  sleep 300
  cool=0
  for poll in $(seq 1 40); do
    cool=1
    temperatures=
    for node in "${NODES[@]}"; do
      milli=$(ssh -o BatchMode=yes -o ConnectTimeout=8 spark@"$node" \
        "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1")
      case "$milli" in *[!0-9]*|'')
        say "ABORT: no numeric thermal-zone receipt from .$node"
        exit 1
        ;;
      esac
      temp=$((milli / 1000))
      temperatures="$temperatures .${node##*.}=${temp}C"
      [ "$temp" -le "$COOL_AT_C" ] || cool=0
    done
    say "cooldown:$temperatures"
    [ "$cool" = 1 ] && break
    sleep 60
  done
  [ "$cool" = 1 ] || {
    say "ABORT: fleet did not cool to ${COOL_AT_C}C within the production cooldown window"
    exit 1
  }
done

for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "test -f '$OUTPUT_DIR/checkpoint-$TOTAL_STEPS/COMPLETE' &&
     test -f '$OUTPUT_DIR/checkpoint-$TOTAL_STEPS/dcp/__${rank}_0.distcp' &&
     test -f '$OUTPUT_DIR/final/COMPLETE'"
done
say "TRAINING COMPLETE: checkpoint-$TOTAL_STEPS and final are present on all four ranks"
if command -v taey-notify >/dev/null; then
  taey-notify tutor \
    "SFT COMPLETE: ${RUN_TAG} checkpoint-${TOTAL_STEPS} + final present on all 4; output=${OUTPUT_DIR}; logs=${STATE_DIR}/logs" \
    --type response_ready || say "WARN: taey-notify tutor failed"
fi
