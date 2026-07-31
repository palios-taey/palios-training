#!/usr/bin/env bash
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FLEET_ENV=${FLEET_ENV:-"$REPO_ROOT/fleet.env"}
[ -f "$FLEET_ENV" ] && . "$FLEET_ENV"
set -euo pipefail
cd "$REPO_ROOT"

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"
: "${POST_CPT_ARTIFACT_STORE:?fleet.env did not define POST_CPT_ARTIFACT_STORE}"

RUN_TAG=cpt_v7_eps1fix
CONTROLLER="${POST_CPT_ARTIFACT_STORE%/}/candidates/${RUN_TAG}_servable"
CORPUS="${POST_CPT_ARTIFACT_STORE%/}/corpora/cpt_v6_novoice_packed_2560.jsonl"
CORPUS_MANIFEST=careers-qwen/receipts/cpt_v7_eps1fix_corpus.manifest.json
TARGET="${SPARK_HOME%/}/models/${RUN_TAG}_servable"
DCP_DIR="${SPARK_HOME%/}/training_outputs/${RUN_TAG}"
AUDIT_ROOT="${POST_CPT_ARTIFACT_STORE%/}/reviews/${RUN_TAG}/provenance-repair"
EXPECTED_OLD_PROVENANCE_SHA=87b8c691deb8eee0ff03bdbcf5249142b9b32beebe4e2870346d4b53e858c5be
EXPECTED_OLD_MANIFEST_SHA=4bcda23eff184b688ba8098bbd64d93e0b13fd6aa6043fd6df2ab6008b187d3b
EXPECTED_LEAF_PROVENANCE_SHA=f47bef809ed802f8f914a12686e6ca42089c1d915add1d162ee3f929c210f843
EXPECTED_LEAF_MANIFEST_SHA=55714e2a0630f3901bfde6071b96bfef22b974a26a1162a9c39bfc3d4ac4a772
CORRECTION_REASON="CPT v7 trained packed corpus 6f1eca from pre-scrub inputs at packer commit aa40705; the original record read later post-scrub source state."

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "REFUSE: provenance repair requires four rank-ordered Spark nodes." >&2
  exit 1
}
for absolute_path in "$CONTROLLER" "$CORPUS" "$TARGET" "$DCP_DIR" "$AUDIT_ROOT"; do
  case "$absolute_path" in
    /*) ;;
    *) echo "REFUSE: repair path is not absolute: $absolute_path" >&2; exit 1;;
  esac
done

TOOLING_COMMIT=$(git rev-parse HEAD)
PRODUCTION_FILES=(
  careers-qwen/corpus_manifest.py
  careers-qwen/repair_training_provenance.py
  careers-qwen/repair_cpt_v7_provenance.sh
  "$CORPUS_MANIFEST"
)
for production_file in "${PRODUCTION_FILES[@]}"; do
  git ls-files --error-unmatch "$production_file" >/dev/null 2>&1 || {
    echo "REFUSE: provenance-repair runtime is not tracked at tooling commit: $production_file" >&2
    exit 1
  }
done
git diff --quiet "$TOOLING_COMMIT" -- "${PRODUCTION_FILES[@]}" || {
  echo "REFUSE: provenance-repair runtime differs from tooling commit $TOOLING_COMMIT." >&2
  exit 1
}

verify_tree(){
  local root=$1
  (
    cd "$root"
    sha256sum -c SOURCE_SHA256SUMS >/dev/null
  )
}

manifest_sha(){
  sha256sum "$1/SOURCE_SHA256SUMS" | cut -d' ' -f1
}

echo "=== CONTROLLER SOURCE — verify and repair once ==="
python3 careers-qwen/corpus_manifest.py verify \
  --corpus "$CORPUS" --manifest "$CORPUS_MANIFEST" >/dev/null
mkdir -p "$AUDIT_ROOT"
controller_manifest=$(manifest_sha "$CONTROLLER")
if [ "$controller_manifest" = "$EXPECTED_OLD_MANIFEST_SHA" ]; then
  verify_tree "$CONTROLLER"
  audit_manifest="$AUDIT_ROOT/SOURCE_SHA256SUMS.pre-lineage-repair"
  if [ -f "$audit_manifest" ]; then
    [ "$(sha256sum "$audit_manifest" | cut -d' ' -f1)" = "$EXPECTED_OLD_MANIFEST_SHA" ] || {
      echo "REFUSE: controller audit manifest already exists with different bytes." >&2
      exit 1
    }
  else
    cp "$CONTROLLER/SOURCE_SHA256SUMS" "$audit_manifest"
  fi
  python3 careers-qwen/repair_training_provenance.py \
    --artifact "$CONTROLLER" \
    --corpus "$CORPUS" \
    --corpus-manifest "$CORPUS_MANIFEST" \
    --expected-current-provenance-sha256 "$EXPECTED_OLD_PROVENANCE_SHA" \
    --audit-copy "$AUDIT_ROOT/training_provenance.pre-lineage-repair.json" \
    --date 2026-07-30 \
    --reason "$CORRECTION_REASON"
  controller_manifest_stage="$CONTROLLER/SOURCE_SHA256SUMS.lineage-repair.tmp"
  (
    cd "$CONTROLLER"
    find . -type f \
      ! -name SOURCE_SHA256SUMS \
      ! -name 'SOURCE_SHA256SUMS.lineage-repair.tmp' \
      -printf '%P\0' |
      sort -z |
      xargs -0 -r sha256sum >"$controller_manifest_stage"
  )
  mv "$controller_manifest_stage" "$CONTROLLER/SOURCE_SHA256SUMS"
else
  python3 careers-qwen/repair_training_provenance.py \
    --artifact "$CONTROLLER" \
    --corpus "$CORPUS" \
    --corpus-manifest "$CORPUS_MANIFEST" \
    --expected-current-provenance-sha256 "$EXPECTED_OLD_PROVENANCE_SHA" \
    --audit-copy "$AUDIT_ROOT/training_provenance.pre-lineage-repair.json" \
    --date 2026-07-30 \
    --reason "$CORRECTION_REASON"
fi
verify_tree "$CONTROLLER"
NEW_MANIFEST_SHA=$(manifest_sha "$CONTROLLER")
NEW_PROVENANCE_SHA=$(sha256sum "$CONTROLLER/training_provenance.json" | cut -d' ' -f1)

python3 - "$AUDIT_ROOT/SOURCE_SHA256SUMS.pre-lineage-repair" \
  "$CONTROLLER/SOURCE_SHA256SUMS" <<'PY'
import sys

def read_manifest(path):
    rows = {}
    for line in open(path):
        digest, name = line.rstrip("\n").split("  ", 1)
        rows[name] = digest
    return rows

before = read_manifest(sys.argv[1])
after = read_manifest(sys.argv[2])
if set(before) != set(after):
    raise SystemExit("REFUSE: provenance repair changed the controller artifact file set")
changed = sorted(name for name in before if before[name] != after[name])
if changed != ["training_provenance.json"]:
    raise SystemExit(f"REFUSE: provenance repair changed unexpected files: {changed}")
print("controller_delta=training_provenance.json_only")
PY
echo "  controller manifest=$NEW_MANIFEST_SHA provenance=$NEW_PROVENANCE_SHA"

echo
echo "=== FOUR-SPARK METADATA FANOUT — no weight transfer ==="
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$TARGET" "$EXPECTED_LEAF_MANIFEST_SHA" \
      "$EXPECTED_LEAF_PROVENANCE_SHA" "$EXPECTED_OLD_PROVENANCE_SHA" \
      "$NEW_MANIFEST_SHA")" <<'REMOTE'
set -euo pipefail
root=$1
expected_leaf_manifest=$2
expected_leaf_provenance=$3
expected_original_provenance=$4
new_manifest=$5
trainers=$(ps -eo args= | awk \
  '/[t]rain_ddp_lora.py|[t]rain_fsdp_dense_9b.py|[t]orch.distributed.run/{n++} END{print n+0}')
[ "$trainers" = 0 ]
current_manifest=$(sha256sum "$root/SOURCE_SHA256SUMS" | cut -d' ' -f1)
if [ "$current_manifest" = "$new_manifest" ]; then
  (
    cd "$root"
    sha256sum -c SOURCE_SHA256SUMS >/dev/null
  )
  echo READY
  exit 0
fi
[ "$current_manifest" = "$expected_leaf_manifest" ]
[ "$(sha256sum "$root/training_provenance.json" | cut -d' ' -f1)" = \
  "$expected_leaf_provenance" ]
[ "$(sha256sum "$root/training_provenance.json.pre-lineage-fix" | cut -d' ' -f1)" = \
  "$expected_original_provenance" ]
(
  cd "$root"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
echo REPAIR
REMOTE
)
  case "$receipt" in
    READY)
      echo "  rank$rank .$node SKIP — already exact"
      continue
      ;;
    REPAIR) ;;
    *) echo "REFUSE: rank$rank .$node returned unexpected preflight: $receipt" >&2; exit 1;;
  esac

  provenance_stage="$TARGET/training_provenance.json.lineage-repair"
  manifest_stage="$TARGET/SOURCE_SHA256SUMS.lineage-repair"
  scp -q -o BatchMode=yes "$CONTROLLER/training_provenance.json" \
    "spark@$node:$provenance_stage"
  scp -q -o BatchMode=yes "$CONTROLLER/SOURCE_SHA256SUMS" \
    "spark@$node:$manifest_stage"
  final_receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "$(printf '%q ' bash -s -- "$TARGET" "$provenance_stage" "$manifest_stage" \
      "$NEW_PROVENANCE_SHA" "$NEW_MANIFEST_SHA" "$DCP_DIR")" <<'REMOTE'
set -euo pipefail
root=$1
provenance_stage=$2
manifest_stage=$3
expected_provenance=$4
expected_manifest=$5
dcp_dir=$6
[ "$(sha256sum "$provenance_stage" | cut -d' ' -f1)" = "$expected_provenance" ]
[ "$(sha256sum "$manifest_stage" | cut -d' ' -f1)" = "$expected_manifest" ]
audit="$dcp_dir/provenance-repair"
mkdir -p "$audit"
if [ -f "$root/training_provenance.json.pre-lineage-fix" ]; then
  mv "$root/training_provenance.json.pre-lineage-fix" \
    "$audit/training_provenance.pre-lineage-repair.json"
fi
mv "$provenance_stage" "$root/training_provenance.json"
mv "$manifest_stage" "$root/SOURCE_SHA256SUMS"
(
  cd "$root"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
printf '%s %s\n' \
  "$(sha256sum "$root/SOURCE_SHA256SUMS" | cut -d' ' -f1)" \
  "$(sha256sum "$root/training_provenance.json" | cut -d' ' -f1)"
REMOTE
)
  [ "$final_receipt" = "$NEW_MANIFEST_SHA $NEW_PROVENANCE_SHA" ] || {
    echo "REFUSE: rank$rank .$node final receipt differs: $final_receipt" >&2
    exit 1
  }
  echo "  rank$rank .$node manifest=$NEW_MANIFEST_SHA provenance=$NEW_PROVENANCE_SHA"
done

echo
echo "CPT V7 TRAINING PROVENANCE REPAIR COMPLETE"
echo "  controller_manifest $NEW_MANIFEST_SHA"
echo "  provenance_sha256   $NEW_PROVENANCE_SHA"
echo "  changed_content     training_provenance.json only"
