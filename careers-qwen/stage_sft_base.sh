#!/usr/bin/env bash
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${POST_CPT_ARTIFACT_STORE:?fleet.env did not define POST_CPT_ARTIFACT_STORE}"
: "${POST_CPT_CONVERT_SSH:?fleet.env did not define POST_CPT_CONVERT_SSH}"
: "${POST_CPT_CONVERT_ROOT:?fleet.env did not define POST_CPT_CONVERT_ROOT}"

[ "$#" -le 1 ] || {
  echo "usage: RUN_TAG=<completed CPT run> $0 [run-tag]" >&2
  exit 2
}
RUN_TAG=${RUN_TAG:-${1:-}}
: "${RUN_TAG:?set RUN_TAG to the completed CPT run tag}"
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*)
    echo "REFUSE: RUN_TAG contains unsafe characters: $RUN_TAG" >&2
    exit 1
    ;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: SFT base staging requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}

TRANSFER_MARGIN_BYTES=${TRANSFER_MARGIN_BYTES:-10737418240}
SOURCE="${POST_CPT_CONVERT_ROOT%/}/${RUN_TAG}_servable"
CONTROLLER="${POST_CPT_ARTIFACT_STORE%/}/candidates/${RUN_TAG}_servable"
TARGET="${SPARK_HOME%/}/models/${RUN_TAG}_servable"
CONTROLLER_STAGE="${CONTROLLER}.staging"
TRANSFER_MARKER=.TRANSFER_SOURCE_MANIFEST_SHA256

case "$SOURCE:$CONTROLLER:$TARGET" in
  "${POST_CPT_CONVERT_ROOT%/}"/*_servable:"${POST_CPT_ARTIFACT_STORE%/}"/candidates/*_servable:"${SPARK_HOME%/}"/models/*_servable) ;;
  *)
    echo "REFUSE: candidate paths escaped their production roots." >&2
    exit 1
    ;;
esac

verify_candidate(){
  local root=$1
  [ -f "$root/GRAFT_COMPLETE" ]
  [ -f "$root/weight_diff.json" ]
  [ -f "$root/training_provenance.json" ]
  [ -f "$root/SOURCE_SHA256SUMS" ]
  (
    cd "$root"
    sha256sum -c SOURCE_SHA256SUMS >/dev/null
  )
  python3 - "$root" <<'PY'
import hashlib
import json
import math
import os
import sys

root = sys.argv[1]
index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
weight_map = index["weight_map"]
if len(weight_map) != 1199:
    raise SystemExit(f"REFUSE: {root} has {len(weight_map)} tensors; expected 1199")
missing = sorted({
    shard for shard in weight_map.values()
    if not os.path.isfile(os.path.join(root, shard))
})
if missing:
    raise SystemExit(f"REFUSE: {root} is missing indexed shards: {missing}")
diff = json.load(open(os.path.join(root, "weight_diff.json")))
value = float(diff["abs_mean_dW"])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"REFUSE: invalid abs_mean_dW in {root}: {value}")
if value > 8e-4:
    raise SystemExit(f"REFUSE: above-band candidate cannot be staged: {value}")
provenance = json.load(open(os.path.join(root, "training_provenance.json")))
schedule = provenance.get("schedule", {})
fields = [schedule.get(name) for name in
          ("total_steps", "completed_step", "resumed_step", "steps_executed")]
if any(type(field) is not int for field in fields):
    raise SystemExit("REFUSE: provenance lacks integer schedule completion fields")
total, completed, resumed, executed = fields
if not 0 <= resumed <= completed <= total or executed != completed - resumed:
    raise SystemExit("REFUSE: provenance schedule completion fields are inconsistent")
if provenance.get("stage") != "graft" or provenance.get("artifact_tensors") != 1199:
    raise SystemExit("REFUSE: candidate provenance is not a 1199-tensor graft")
if provenance.get("base_tensors") != 851:
    raise SystemExit("REFUSE: candidate provenance does not identify an 851-tensor CPT base")
commit = provenance.get("tooling_commit", "")
tool_sha = provenance.get("tooling_sha256", "")
if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
    raise SystemExit("REFUSE: provenance tooling_commit is absent or malformed")
if len(tool_sha) != 64 or any(c not in "0123456789abcdef" for c in tool_sha):
    raise SystemExit("REFUSE: provenance tooling_sha256 is absent or malformed")
if value < 5e-5:
    review_path = os.path.join(root, "below_band_review.json")
    if not os.path.isfile(review_path):
        raise SystemExit("REFUSE: below-band candidate has no bound Chats review")
    review = json.load(open(review_path))
    diff_sha = hashlib.sha256(
        open(os.path.join(root, "weight_diff.json"), "rb").read()
    ).hexdigest()
    run_tag = os.path.basename(root).removesuffix("_servable")
    if (review.get("schema") != "palios.post_cpt_below_band_review.v1"
            or review.get("run_tag") != run_tag
            or review.get("weight_diff_sha256") != diff_sha
            or not math.isclose(float(review.get("abs_mean_dW", -1)), value,
                                rel_tol=0.0, abs_tol=1e-15)
            or review.get("disposition") != "trial_after_chat_review"
            or review.get("authorized_by") != "Jesse"):
        raise SystemExit("REFUSE: below-band review is not bound to this candidate")
print(f"candidate=1199 diff={value:.9e}")
PY
}

remote_verify_command(){
  local root=$1 expected_manifest=$2
  printf '%q ' bash -s -- "$root" "$expected_manifest"
}

remote_verify(){
  local ssh_target=$1 root=$2 expected_manifest=$3
  local command
  command=$(remote_verify_command "$root" "$expected_manifest")
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$command" <<'REMOTE'
set -euo pipefail
root=$1
expected_manifest=$2
test -f "$root/GRAFT_COMPLETE"
test -f "$root/weight_diff.json"
test -f "$root/training_provenance.json"
test -f "$root/SOURCE_SHA256SUMS"
actual_manifest=$(sha256sum "$root/SOURCE_SHA256SUMS" | cut -d' ' -f1)
[ "$actual_manifest" = "$expected_manifest" ]
(
  cd "$root"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
python3 - "$root" <<'PY'
import hashlib
import json
import math
import os
import sys

root = sys.argv[1]
weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
if len(weight_map) != 1199:
    raise SystemExit(f"candidate has {len(weight_map)} tensors; expected 1199")
if any(not os.path.isfile(os.path.join(root, shard)) for shard in set(weight_map.values())):
    raise SystemExit("candidate is missing an indexed shard")
value = float(json.load(open(os.path.join(root, "weight_diff.json")))["abs_mean_dW"])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"invalid abs_mean_dW: {value}")
if value > 8e-4:
    raise SystemExit(f"above-band candidate cannot be staged: {value}")
provenance = json.load(open(os.path.join(root, "training_provenance.json")))
schedule = provenance.get("schedule", {})
fields = [schedule.get(name) for name in
          ("total_steps", "completed_step", "resumed_step", "steps_executed")]
if any(type(field) is not int for field in fields):
    raise SystemExit("provenance lacks integer schedule completion fields")
total, completed, resumed, executed = fields
if not 0 <= resumed <= completed <= total or executed != completed - resumed:
    raise SystemExit("provenance schedule completion fields are inconsistent")
if (provenance.get("stage") != "graft"
        or provenance.get("artifact_tensors") != 1199
        or provenance.get("base_tensors") != 851):
    raise SystemExit("candidate provenance does not describe an 851→1199 graft")
commit = provenance.get("tooling_commit", "")
tool_sha = provenance.get("tooling_sha256", "")
if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
    raise SystemExit("provenance tooling_commit is absent or malformed")
if len(tool_sha) != 64 or any(c not in "0123456789abcdef" for c in tool_sha):
    raise SystemExit("provenance tooling_sha256 is absent or malformed")
if value < 5e-5:
    review_path = os.path.join(root, "below_band_review.json")
    if not os.path.isfile(review_path):
        raise SystemExit("below-band candidate has no bound Chats review")
    review = json.load(open(review_path))
    diff_sha = hashlib.sha256(
        open(os.path.join(root, "weight_diff.json"), "rb").read()
    ).hexdigest()
    run_tag = os.path.basename(root).removesuffix("_servable")
    if (review.get("schema") != "palios.post_cpt_below_band_review.v1"
            or review.get("run_tag") != run_tag
            or review.get("weight_diff_sha256") != diff_sha
            or not math.isclose(float(review.get("abs_mean_dW", -1)), value,
                                rel_tol=0.0, abs_tol=1e-15)
            or review.get("disposition") != "trial_after_chat_review"
            or review.get("authorized_by") != "Jesse"):
        raise SystemExit("below-band review is not bound to this candidate")
print(f"1199 {value:.9e}")
PY
REMOTE
}

echo "=== SFT BASE SOURCE — verify baked candidate on the conversion host ==="
source_receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$POST_CPT_CONVERT_SSH" \
  "$(printf '%q ' bash -s -- "$SOURCE")" <<'REMOTE'
set -euo pipefail
root=$1
test -d "$root"
test -f "$root/GRAFT_COMPLETE"
test -f "$root/weight_diff.json"
test -f "$root/training_provenance.json"
python3 - "$root" <<'PY'
import hashlib
import json
import math
import os
import sys

root = sys.argv[1]
weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
if len(weight_map) != 1199:
    raise SystemExit(f"candidate has {len(weight_map)} tensors; expected 1199")
missing = sorted({
    shard for shard in weight_map.values()
    if not os.path.isfile(os.path.join(root, shard))
})
if missing:
    raise SystemExit(f"candidate is missing indexed shards: {missing}")
value = float(json.load(open(os.path.join(root, "weight_diff.json")))["abs_mean_dW"])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"invalid abs_mean_dW: {value}")
if value > 8e-4:
    raise SystemExit(f"above-band candidate cannot be staged: {value}")
provenance = json.load(open(os.path.join(root, "training_provenance.json")))
schedule = provenance.get("schedule", {})
fields = [schedule.get(name) for name in
          ("total_steps", "completed_step", "resumed_step", "steps_executed")]
if any(type(field) is not int for field in fields):
    raise SystemExit("provenance lacks integer schedule completion fields")
total, completed, resumed, executed = fields
if not 0 <= resumed <= completed <= total or executed != completed - resumed:
    raise SystemExit("provenance schedule completion fields are inconsistent")
if (provenance.get("stage") != "graft"
        or provenance.get("artifact_tensors") != 1199
        or provenance.get("base_tensors") != 851):
    raise SystemExit("candidate provenance does not describe an 851→1199 graft")
commit = provenance.get("tooling_commit", "")
tool_sha = provenance.get("tooling_sha256", "")
if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
    raise SystemExit("provenance tooling_commit is absent or malformed")
if len(tool_sha) != 64 or any(c not in "0123456789abcdef" for c in tool_sha):
    raise SystemExit("provenance tooling_sha256 is absent or malformed")
if value < 5e-5:
    review_path = os.path.join(root, "below_band_review.json")
    if not os.path.isfile(review_path):
        raise SystemExit("below-band candidate has no bound Chats review")
    review = json.load(open(review_path))
    diff_sha = hashlib.sha256(
        open(os.path.join(root, "weight_diff.json"), "rb").read()
    ).hexdigest()
    run_tag = os.path.basename(root).removesuffix("_servable")
    if (review.get("schema") != "palios.post_cpt_below_band_review.v1"
            or review.get("run_tag") != run_tag
            or review.get("weight_diff_sha256") != diff_sha
            or not math.isclose(float(review.get("abs_mean_dW", -1)), value,
                                rel_tol=0.0, abs_tol=1e-15)
            or review.get("disposition") != "trial_after_chat_review"
            or review.get("authorized_by") != "Jesse"):
        raise SystemExit("below-band review is not bound to this candidate")
print(os.path.getsize(os.path.join(root, "model.safetensors.index.json")), len(weight_map), f"{value:.9e}")
PY
REMOTE
)
read -r index_bytes source_tensors diff_abs <<<"$source_receipt"
[ "$source_tensors" = 1199 ]
echo "  source=$POST_CPT_CONVERT_SSH:$SOURCE tensors=$source_tensors diff=$diff_abs index_bytes=$index_bytes"

manifest_tmp=$(mktemp)
cleanup_manifest(){
  rm -f -- "$manifest_tmp"
}
trap cleanup_manifest EXIT
ssh -o BatchMode=yes -o ConnectTimeout=10 "$POST_CPT_CONVERT_SSH" \
  "$(printf '%q ' bash -s -- "$SOURCE")" <<'REMOTE' >"$manifest_tmp"
set -euo pipefail
root=$1
cd "$root"
find . -type l -print -quit | grep -q . && {
  echo "REFUSE: source candidate contains a symbolic link" >&2
  exit 1
}
find . -type f ! -name SOURCE_SHA256SUMS -printf '%P\0' |
  LC_ALL=C sort -z |
  xargs -0 -r sha256sum
REMOTE
[ -s "$manifest_tmp" ] || {
  echo "REFUSE: source candidate checksum manifest is empty." >&2
  exit 1
}
MANIFEST_SHA=$(sha256sum "$manifest_tmp" | cut -d' ' -f1)
SOURCE_BYTES=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$POST_CPT_CONVERT_SSH" \
  "du -sb '$SOURCE' | cut -f1")
echo "  source_bytes=$SOURCE_BYTES source_manifest_sha256=$MANIFEST_SHA"

echo
echo "=== CONTROLLER HOLD — resumable copy, source hashes, atomic promotion ==="
mkdir -p "$(dirname "$CONTROLLER")"
if [ -e "$CONTROLLER" ]; then
  verify_candidate "$CONTROLLER"
  [ "$(sha256sum "$CONTROLLER/SOURCE_SHA256SUMS" | cut -d' ' -f1)" = "$MANIFEST_SHA" ] || {
    echo "REFUSE: existing controller candidate is valid but differs from the conversion source." >&2
    exit 1
  }
  echo "  SKIP controller copy — exact verified hold already exists"
else
  if [ -e "$CONTROLLER_STAGE" ]; then
    [ -f "$CONTROLLER_STAGE/$TRANSFER_MARKER" ]
    [ "$(cat "$CONTROLLER_STAGE/$TRANSFER_MARKER")" = "$MANIFEST_SHA" ] || {
      echo "REFUSE: retained controller stage belongs to a different source manifest." >&2
      exit 1
    }
  else
    mkdir "$CONTROLLER_STAGE"
    printf '%s\n' "$MANIFEST_SHA" >"$CONTROLLER_STAGE/$TRANSFER_MARKER"
  fi
  stage_bytes=$(du -sb "$CONTROLLER_STAGE" | cut -f1)
  remaining=$((SOURCE_BYTES > stage_bytes ? SOURCE_BYTES - stage_bytes : 0))
  available=$(df -B1 --output=avail "$(dirname "$CONTROLLER")" | tail -1 | tr -d ' ')
  required=$((remaining + TRANSFER_MARGIN_BYTES))
  [ "$available" -ge "$required" ] || {
    echo "REFUSE: controller has $available bytes free; remaining copy plus margin requires $required." >&2
    exit 1
  }
  cp "$manifest_tmp" "$CONTROLLER_STAGE/SOURCE_SHA256SUMS"
  if ! rsync -rt --partial --no-perms --no-owner --no-group --exclude SOURCE_SHA256SUMS \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$POST_CPT_CONVERT_SSH:$SOURCE/" "$CONTROLLER_STAGE/"; then
    echo "COPY FAILED: resumable controller stage retained at $CONTROLLER_STAGE." >&2
    exit 1
  fi
  verify_candidate "$CONTROLLER_STAGE"
  rm -- "$CONTROLLER_STAGE/$TRANSFER_MARKER"
  mv "$CONTROLLER_STAGE" "$CONTROLLER"
  echo "  controller hold promoted atomically: $CONTROLLER"
fi
verify_candidate "$CONTROLLER"

echo
echo "=== FOUR-SPARK FANOUT — disk preflight, resumable parallel copy ==="
declare -a COPY_RANK
declare -a REMOTE_STAGE
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  stage="${TARGET}.staging"
  REMOTE_STAGE[$rank]=$stage
  status=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$TARGET" "$stage" "$MANIFEST_SHA" "$SOURCE_BYTES" "$TRANSFER_MARGIN_BYTES" "$TRANSFER_MARKER")" <<'REMOTE'
set -euo pipefail
target=$1
stage=$2
expected_manifest=$3
source_bytes=$4
margin=$5
marker=$6
mkdir -p "$(dirname "$target")"
if [ -e "$target" ]; then
  echo EXISTING
  exit 0
fi
if [ -e "$stage" ]; then
  test -f "$stage/$marker"
  [ "$(cat "$stage/$marker")" = "$expected_manifest" ] || {
    echo "retained stage belongs to another source manifest" >&2
    exit 1
  }
else
  mkdir "$stage"
  printf '%s\n' "$expected_manifest" >"$stage/$marker"
fi
stage_bytes=$(du -sb "$stage" | cut -f1)
remaining=$((source_bytes > stage_bytes ? source_bytes - stage_bytes : 0))
available=$(df -B1 --output=avail "$(dirname "$target")" | tail -1 | tr -d ' ')
required=$((remaining + margin))
[ "$available" -ge "$required" ] || {
  echo "only $available bytes free; remaining copy plus margin requires $required" >&2
  exit 1
}
printf 'COPY %s %s\n' "$stage_bytes" "$available"
REMOTE
)
  if [ "$status" = EXISTING ]; then
    remote_verify "spark@$node" "$TARGET" "$MANIFEST_SHA" >/dev/null || {
      echo "REFUSE: rank$rank .$node has an existing but non-matching SFT base." >&2
      exit 1
    }
    COPY_RANK[$rank]=0
    echo "  rank$rank .$node SKIP — exact candidate already present"
  else
    COPY_RANK[$rank]=1
    echo "  rank$rank .$node $status"
  fi
done

declare -a COPY_PID
copy_failed=0
for rank in 0 1 2 3; do
  [ "${COPY_RANK[$rank]}" = 1 ] || continue
  node=${NODES[$rank]}
  (
    rsync -rt --partial --no-perms --no-owner --no-group \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$CONTROLLER/" "spark@$node:${REMOTE_STAGE[$rank]}/"
  ) &
  COPY_PID[$rank]=$!
done
for rank in 0 1 2 3; do
  [ "${COPY_RANK[$rank]}" = 1 ] || continue
  if ! wait "${COPY_PID[$rank]}"; then
    echo "  rank$rank copy failed; resumable stage retained at spark@${NODES[$rank]}:${REMOTE_STAGE[$rank]}" >&2
    copy_failed=1
  fi
done
[ "$copy_failed" = 0 ] || exit 1

echo
echo "=== ATOMIC SPARK PROMOTION + FINAL RECEIPTS ==="
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  if [ "${COPY_RANK[$rank]}" = 1 ]; then
    receipt=$(remote_verify "spark@$node" "${REMOTE_STAGE[$rank]}" "$MANIFEST_SHA")
    ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "test \"\$(cat '${REMOTE_STAGE[$rank]}/$TRANSFER_MARKER')\" = '$MANIFEST_SHA' &&
       rm -- '${REMOTE_STAGE[$rank]}/$TRANSFER_MARKER' &&
       mv '${REMOTE_STAGE[$rank]}' '$TARGET'"
  else
    receipt=$(remote_verify "spark@$node" "$TARGET" "$MANIFEST_SHA")
  fi
  read -r tensors rank_diff <<<"$receipt"
  [ "$tensors" = 1199 ] && [ "$rank_diff" = "$diff_abs" ] || {
    echo "REFUSE: rank$rank final receipt differs from the source candidate." >&2
    exit 1
  }
  echo "  rank$rank .$node tensors=$tensors diff=$rank_diff manifest=$MANIFEST_SHA"
done

echo
echo "SFT BASE STAGING COMPLETE"
echo "  source      $POST_CPT_CONVERT_SSH:$SOURCE"
echo "  controller  $CONTROLLER"
echo "  sparks      $TARGET"
echo "  tensors     1199 on all four"
echo "  weight-diff $diff_abs"
echo "  manifest    $MANIFEST_SHA"
