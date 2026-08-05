#!/usr/bin/env bash
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${DCP_DIR:?set DCP_DIR to the completed CPT output directory}"
: "${POST_CPT_CONVERT_SSH:?fleet.env did not define POST_CPT_CONVERT_SSH}"
: "${POST_CPT_CONVERT_ROOT:?fleet.env did not define POST_CPT_CONVERT_ROOT}"
: "${POST_CPT_GRAFT_BASE:?fleet.env did not define POST_CPT_GRAFT_BASE}"
: "${POST_CPT_CONVERT_IMAGE:?fleet.env did not define POST_CPT_CONVERT_IMAGE}"

case "$DCP_DIR" in
  /*) ;;
  *) echo "REFUSE: DCP_DIR must be absolute." >&2; exit 1;;
esac

RUN_TAG=$(basename "$DCP_DIR")
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*)
    echo "REFUSE: RUN_TAG contains unsafe characters: $RUN_TAG" >&2
    exit 1
    ;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: candidate finalization requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}

RC=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "cat '$DCP_DIR/run_config.env' 2>/dev/null" || true)
[ -n "$RC" ] || {
  echo "REFUSE: no run_config.env under $DCP_DIR on $SPARK_MASTER." >&2
  exit 1
}
rc_get(){ sed -n "s|^$1=||p" <<<"$RC" | tail -1; }

CORPUS=$(rc_get CPT_PATH_FROM_LOG)
TOTAL_STEPS=$(rc_get TOTAL_STEPS)
WARMUP_STEPS=$(rc_get WARMUP_STEPS)
LR=$(rc_get LR)
TRAIN_BASE=$(rc_get TRAIN_BASE)
CAPTURED_CORPUS_INPUTS=$(rc_get CORPUS_INPUTS)
# LR is required for the same reason the others are, with more history behind it:
# run_4node_27b_cpt.sh:58-60 records that LR and WARMUP_STEPS were silently NOT forwarded
# until 2026-07-13, so runs before that trained at the trainer default regardless of what
# the operator set. WARMUP_STEPS was guarded here; LR was not — leaving the one value whose
# divergence is already documented as the unguarded one.
for required_name in CORPUS TOTAL_STEPS WARMUP_STEPS LR TRAIN_BASE; do
  [ -n "${!required_name}" ] || {
    echo "REFUSE: $required_name is absent from run_config.env." >&2
    exit 1
  }
done
CORPUS_MANIFEST="${CORPUS}.manifest.json"
corpus_receipt_text=
if ! corpus_receipt_text=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
  "$(printf '%q ' /usr/bin/python3 - verify --corpus "$CORPUS" \
    --manifest "$CORPUS_MANIFEST" --receipt-lines)" \
  <careers-qwen/corpus_manifest.py); then
  echo "REFUSE: packed corpus has no valid artifact-bound input manifest." >&2
  exit 1
fi
mapfile -t CORPUS_RECEIPT <<<"$corpus_receipt_text"
[ "${#CORPUS_RECEIPT[@]}" = 3 ] || {
  echo "REFUSE: corpus manifest verifier returned an incomplete receipt." >&2
  exit 1
}
CORPUS_MANIFEST_SHA=${CORPUS_RECEIPT[1]}
CORPUS_INPUTS=${CORPUS_RECEIPT[2]}
[ -z "$CAPTURED_CORPUS_INPUTS" ] || [ "$CAPTURED_CORPUS_INPUTS" = "$CORPUS_INPUTS" ] || {
  echo "REFUSE: run_config CORPUS_INPUTS disagrees with the packed artifact manifest." >&2
  exit 1
}

CKPT=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "ls -d '$DCP_DIR'/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1")
case "$CKPT:$TOTAL_STEPS:$WARMUP_STEPS" in
  *[!0-9:]*|"")
    echo "REFUSE: checkpoint and schedule fields must be integers: $CKPT/$TOTAL_STEPS/$WARMUP_STEPS" >&2
    exit 1
    ;;
esac
[ "$CKPT" -le "$TOTAL_STEPS" ] || {
  echo "REFUSE: completed checkpoint $CKPT exceeds schedule horizon $TOTAL_STEPS." >&2
  exit 1
}

CONVERT_ROOT=${POST_CPT_CONVERT_ROOT%/}
CONVERT_TOOLS=${POST_CPT_CONVERT_TOOLS:-$CONVERT_ROOT/tools}
CONVERT_BASE="$CONVERT_ROOT/${RUN_TAG}_training_base"
HF_OUT="$CONVERT_ROOT/${RUN_TAG}_hf"
SERVABLE_OUT="$CONVERT_ROOT/${RUN_TAG}_servable"
SERVABLE_STAGE="${SERVABLE_OUT}.staging.$$"
CONVERT_CORPUS="$CONVERT_ROOT/corpora/$(basename "$CORPUS")"
CONVERT_CORPUS_MANIFEST="${CONVERT_CORPUS}.manifest.json"
GRAFT_BASE=$POST_CPT_GRAFT_BASE
CONVERT_IMAGE=$POST_CPT_CONVERT_IMAGE
SANCTION=${SANCTION:-${POST_CPT_SANCTION:-"treasurer task-dfa3fd75 2026-07-28"}}
SPACE_MARGIN_BYTES=${SPACE_MARGIN_BYTES:-10737418240}
NOTIFY_TARGET=${POST_CPT_NOTIFY_TARGET:-infra}
TOOLING_COMMIT=$(git rev-parse HEAD)
LOCAL_TOOLS=(
  careers-qwen/graft_cpt_into_servable.py
  careers-qwen/corpus_manifest.py
  ${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT}/sources/palios-training-c164d35/tree/careers-qwen/emit_training_provenance.py
)
PRODUCTION_FILES=(
  careers-qwen/finalize_post_cpt_candidate.sh
  "${LOCAL_TOOLS[@]}"
)
for production_file in "${PRODUCTION_FILES[@]}"; do
  git ls-files --error-unmatch "$production_file" >/dev/null 2>&1 || {
    echo "REFUSE: finalization runtime is not tracked at tooling commit: $production_file" >&2
    exit 1
  }
done
git diff --quiet "$TOOLING_COMMIT" -- "${PRODUCTION_FILES[@]}" || {
  echo "REFUSE: finalization runtime differs from tooling commit $TOOLING_COMMIT." >&2
  exit 1
}

for absolute_path in "$CONVERT_ROOT" "$CONVERT_TOOLS" "$CONVERT_BASE" "$HF_OUT" \
  "$SERVABLE_OUT" "$CONVERT_CORPUS" "$CONVERT_CORPUS_MANIFEST" "$GRAFT_BASE"; do
  case "$absolute_path" in
    /*) ;;
    *) echo "REFUSE: production path is not absolute: $absolute_path" >&2; exit 1;;
  esac
done

remote_tensor_count(){
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "/usr/bin/python3 - '$1'" <<'PY'
import json
import os
import sys

index_path = os.path.join(sys.argv[1], "model.safetensors.index.json")
print(len(json.load(open(index_path))["weight_map"]) if os.path.isfile(index_path) else 0)
PY
}

remote_container_python(){
  local -a command=(
    sudo docker run --rm --network none
    -v "$CONVERT_ROOT:$CONVERT_ROOT"
    -v "$GRAFT_BASE:$GRAFT_BASE:ro"
    --entrypoint /opt/venv/bin/python
    "$CONVERT_IMAGE"
    "$@"
  )
  local quoted
  printf -v quoted '%q ' "${command[@]}"
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "$quoted"
}

echo "=== FINALIZE 0/5 — bind to the completed checkpoint and exact conversion inputs ==="
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "test -f '$DCP_DIR/checkpoint-$CKPT/COMPLETE' &&
     test -f '$DCP_DIR/checkpoint-$CKPT/dcp/__${rank}_0.distcp' &&
     trainers=\$(ps -eo args= | awk '/[t]orchrun|[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}') &&
     printf '%s %s\\n' \"\$(stat -c %s '$DCP_DIR/checkpoint-$CKPT/dcp/__${rank}_0.distcp')\" \"\$trainers\"")
  read -r shard_bytes trainers <<<"$receipt"
  [ "$trainers" = 0 ] || {
    echo "REFUSE: rank$rank .$node has $trainers trainer process(es)." >&2
    exit 1
  }
  echo "  rank$rank .$node checkpoint-$CKPT COMPLETE shard=$shard_bytes trainers=0"
done

python3 -m py_compile "${LOCAL_TOOLS[@]}"
for local_tool in "${LOCAL_TOOLS[@]}"; do
  rsync -a -e "ssh -o BatchMode=yes" "$local_tool" \
    "$POST_CPT_CONVERT_SSH:$CONVERT_TOOLS/"
  local_sha=$(sha256sum "$local_tool" | cut -d' ' -f1)
  remote_sha=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "sha256sum '$CONVERT_TOOLS/$(basename "$local_tool")' | cut -d' ' -f1")
  [ "$local_sha" = "$remote_sha" ] || {
    echo "REFUSE: deployed tool hash mismatch for $local_tool." >&2
    exit 1
  }
done
versions=$(remote_container_python -c \
  'import torch,transformers; print(torch.__version__); print(transformers.__version__)')
printf '%s\n' "$versions"
grep -qx '2.10.0' <<<"$versions"
grep -qx '5.3.0' <<<"$versions"

[ "$(remote_tensor_count "$CONVERT_BASE")" = 851 ] || {
  echo "REFUSE: exact training base is absent or is not 851 tensors." >&2
  exit 1
}
[ "$(remote_tensor_count "$HF_OUT")" = 851 ] || {
  echo "REFUSE: converted CPT output is absent or is not 851 tensors." >&2
  exit 1
}
[ "$(remote_tensor_count "$GRAFT_BASE")" = 1199 ] || {
  echo "REFUSE: serving donor is absent or is not 1199 tensors." >&2
  exit 1
}
ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "test -f '$HF_OUT/CONVERT_COMPLETE' &&
   test -f '$HF_OUT/weight_diff.json' &&
   test -f '$CONVERT_CORPUS' &&
   test -f '$CONVERT_CORPUS_MANIFEST' &&
   test \"\$(sha256sum '$CONVERT_CORPUS_MANIFEST' | cut -d' ' -f1)\" = '$CORPUS_MANIFEST_SHA'"

echo
echo "=== FINALIZE 1/5 — classify the exact weight diff ==="
diff_receipt=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "/usr/bin/python3 - '$HF_OUT/weight_diff.json'" <<'PY'
import hashlib
import json
import math
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
value = float(json.loads(raw)["abs_mean_dW"])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"REFUSE: invalid abs_mean_dW: {value}")
band = "IN_BAND" if 5e-5 <= value <= 8e-4 else "BELOW" if value < 5e-5 else "ABOVE"
print(f"{value:.17g} {band} {hashlib.sha256(raw).hexdigest()}")
PY
)
read -r DIFF_ABS DIFF_GATE DIFF_SHA <<<"$diff_receipt"
printf 'WEIGHT_DIFF %.9e %s sha256=%s\n' "$DIFF_ABS" "$DIFF_GATE" "$DIFF_SHA"
[ "$DIFF_GATE" != ABOVE ] || {
  echo "FULL STOP: above-band candidates require a separate explicit production disposition." >&2
  exit 3
}

REVIEW_SHA=
REVIEW_REMOTE=
if [ "$DIFF_GATE" = BELOW ]; then
  : "${BELOW_BAND_REVIEW_RECEIPT:?below-band finalization requires BELOW_BAND_REVIEW_RECEIPT}"
  case "$BELOW_BAND_REVIEW_RECEIPT" in
    /*) ;;
    *) echo "REFUSE: BELOW_BAND_REVIEW_RECEIPT must be absolute." >&2; exit 1;;
  esac
  REVIEW_SHA=$(python3 - "$BELOW_BAND_REVIEW_RECEIPT" "$RUN_TAG" "$DIFF_ABS" "$DIFF_SHA" <<'PY'
import hashlib
import json
import math
import os
import sys

receipt_path, run_tag, expected_diff, expected_diff_sha = sys.argv[1:]
raw = open(receipt_path, "rb").read()
record = json.loads(raw)
if record.get("schema") != "palios.post_cpt_below_band_review.v1":
    raise SystemExit("REFUSE: wrong below-band review schema")
if record.get("run_tag") != run_tag:
    raise SystemExit("REFUSE: review run_tag does not match the candidate")
if record.get("weight_diff_sha256") != expected_diff_sha:
    raise SystemExit("REFUSE: review is not bound to this weight_diff.json")
if not math.isclose(float(record.get("abs_mean_dW")), float(expected_diff),
                    rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit("REFUSE: review abs_mean_dW does not match the candidate")
if record.get("disposition") != "trial_after_chat_review":
    raise SystemExit("REFUSE: review disposition does not authorize a trial")
if record.get("authorized_by") != "Jesse":
    raise SystemExit("REFUSE: below-band trial was not bound to Jesse's instruction")

references = [record.get("review_packet"), record.get("peer_review")]
consultations = record.get("consultations")
if not isinstance(consultations, list) or len(consultations) < 2:
    raise SystemExit("REFUSE: at least two approved Chat consultation outputs are required")
references.extend(consultations)
for reference in references:
    if not isinstance(reference, dict):
        raise SystemExit("REFUSE: review evidence entry is missing")
    path = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(path, str) or not os.path.isabs(path) or not os.path.isfile(path):
        raise SystemExit(f"REFUSE: review evidence path is unavailable: {path}")
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual != expected:
        raise SystemExit(f"REFUSE: review evidence hash mismatch: {path}")
print(hashlib.sha256(raw).hexdigest())
PY
)
  REVIEW_REMOTE="$CONVERT_ROOT/${RUN_TAG}_below_band_review.json"
  review_stage="${REVIEW_REMOTE}.staging.$$"
  scp -q -o BatchMode=yes "$BELOW_BAND_REVIEW_RECEIPT" \
    "$POST_CPT_CONVERT_SSH:$review_stage"
  remote_review_sha=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "sha256sum '$review_stage' | cut -d' ' -f1")
  [ "$remote_review_sha" = "$REVIEW_SHA" ] || {
    echo "REFUSE: review receipt changed during transfer." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "mv '$review_stage' '$REVIEW_REMOTE'"
  echo "  below-band trial review verified and bound: $REVIEW_SHA"
else
  echo "  in-band candidate requires no below-band review override"
fi

echo
echo "=== FINALIZE 2/5 — graft the 851-tensor CPT result into the 1199-tensor serving donor ==="
servable_tensors=$(remote_tensor_count "$SERVABLE_OUT")
if ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "test -e '$SERVABLE_OUT'"; then
  [ "$servable_tensors" = 1199 ] || {
    echo "REFUSE: existing servable output has $servable_tensors tensors." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "test -f '$SERVABLE_OUT/GRAFT_COMPLETE'"
  echo "  SKIP graft — verified 1199-tensor candidate already exists"
else
  hf_bytes=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "du -sb '$HF_OUT' | cut -f1")
  disk_free=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "df -B1 --output=avail '$CONVERT_ROOT' | tail -1 | tr -d ' '")
  [ "$disk_free" -ge $((hf_bytes + SPACE_MARGIN_BYTES)) ] || {
    echo "REFUSE: conversion host lacks space for graft plus safety margin." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" "test ! -e '$SERVABLE_STAGE'"
  remote_container_python "$CONVERT_TOOLS/graft_cpt_into_servable.py" \
    --base "$GRAFT_BASE" --cpt "$HF_OUT" --out "$SERVABLE_STAGE"
  [ "$(remote_tensor_count "$SERVABLE_STAGE")" = 1199 ] || {
    echo "REFUSE: staged graft did not produce 1199 tensors." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "test -f '$SERVABLE_STAGE/GRAFT_COMPLETE' &&
     sudo mv '$SERVABLE_STAGE' '$SERVABLE_OUT'"
fi
ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "sudo cp '$HF_OUT/weight_diff.json' '$SERVABLE_OUT/weight_diff.json'"
if [ "$DIFF_GATE" = BELOW ]; then
  ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "sudo cp '$REVIEW_REMOTE' '$SERVABLE_OUT/below_band_review.json'"
fi

echo
echo "=== FINALIZE 3/5 — write provenance with schedule horizon and completed step separated ==="
NOTE="weight_diff=$DIFF_ABS gate=$DIFF_GATE"
[ -z "$REVIEW_SHA" ] || NOTE="$NOTE below_band_review_sha256=$REVIEW_SHA"
for artifact_stage in "$HF_OUT:cpt:851" "$SERVABLE_OUT:graft:1199"; do
  artifact=${artifact_stage%%:*}
  remainder=${artifact_stage#*:}
  stage=${remainder%%:*}
  expected_tensors=${remainder##*:}
  remote_container_python "${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT}/sources/palios-training-c164d35/tree/careers-qwen/emit_training_provenance.py" \
    --artifact "$artifact" --stage "$stage" --base "$CONVERT_BASE" \
    --corpus "$CONVERT_CORPUS" --total-steps "$TOTAL_STEPS" \
    --completed-step "$CKPT" --warmup-steps "$WARMUP_STEPS" \
    --resumed-step 0 --tooling-commit "$TOOLING_COMMIT" \
    --sanction "$SANCTION" --corpus-manifest "$CONVERT_CORPUS_MANIFEST" \
    --note "$NOTE"
  remote_container_python -c '
import json, os, sys
path, expected_stage, expected_tensors, total, completed = sys.argv[1:]
record = json.load(open(os.path.join(path, "training_provenance.json")))
schedule = record["schedule"]
expected = {
    "stage": expected_stage,
    "artifact_tensors": int(expected_tensors),
    "base_tensors": 851,
}
actual = {name: record.get(name) for name in expected}
if actual != expected:
    raise SystemExit(f"REFUSE: provenance artifact fields differ: {actual} != {expected}")
expected_schedule = {
    "total_steps": int(total),
    "completed_step": int(completed),
    "resumed_step": 0,
    "steps_executed": int(completed),
}
actual_schedule = {name: schedule.get(name) for name in expected_schedule}
if actual_schedule != expected_schedule:
    raise SystemExit(
        f"REFUSE: provenance schedule differs: {actual_schedule} != {expected_schedule}"
    )
if len(record.get("tooling_commit", "")) != 40:
    raise SystemExit("REFUSE: provenance tooling_commit is malformed")
if len(record.get("tooling_sha256", "")) != 64:
    raise SystemExit("REFUSE: provenance tooling_sha256 is malformed")
print(f"PROVENANCE PASS {path} total={total} completed={completed} executed={completed}")
' "$artifact" "$stage" "$expected_tensors" "$TOTAL_STEPS" "$CKPT"
done

echo
echo "=== FINALIZE 4/5 — verify candidate receipts, then retire reproducible transients ==="
[ "$(remote_tensor_count "$SERVABLE_OUT")" = 1199 ]
ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
  "test -f '$SERVABLE_OUT/GRAFT_COMPLETE' &&
   test -f '$SERVABLE_OUT/weight_diff.json' &&
   test -f '$SERVABLE_OUT/training_provenance.json'"
if [ "$DIFF_GATE" = BELOW ]; then
  candidate_review_sha=$(ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
    "sha256sum '$SERVABLE_OUT/below_band_review.json' | cut -d' ' -f1")
  [ "$candidate_review_sha" = "$REVIEW_SHA" ]
fi
case "$HF_OUT:$CONVERT_BASE" in
  "$CONVERT_ROOT"/*_hf:"$CONVERT_ROOT"/*_training_base)
    ssh -o BatchMode=yes "$POST_CPT_CONVERT_SSH" \
      "sudo rm -rf -- '$HF_OUT' '$CONVERT_BASE'"
    ;;
  *) echo "REFUSE: unsafe transient cleanup targets." >&2; exit 1;;
esac
echo "  retired converted HF and conversion-host base; controller Artifact B/base remain reproducible holds"

echo
echo "=== FINALIZE 5/5 — handoff receipt ==="
command -v taey-notify >/dev/null || {
  echo "REFUSE: candidate is complete, but taey-notify is unavailable." >&2
  exit 1
}
taey-notify "$NOTIFY_TARGET" \
  "POST-CPT READY: ${RUN_TAG} weight-diff=${DIFF_ABS} ${DIFF_GATE}; candidate=${POST_CPT_CONVERT_SSH}:${SERVABLE_OUT}; tensors=1199; completed-step=${CKPT}/${TOTAL_STEPS}; training owns SFT staging, serving remains infra-owned." \
  --type response_ready

echo "DONE"
echo "  weight-diff       $DIFF_ABS $DIFF_GATE"
echo "  servable          $POST_CPT_CONVERT_SSH:$SERVABLE_OUT"
echo "  tensors           1199"
echo "  schedule          completed=$CKPT horizon=$TOTAL_STEPS"
[ -z "$REVIEW_SHA" ] || echo "  below-band review $REVIEW_SHA"
echo "  next              stage exact candidate and launch the production SFT driver"
