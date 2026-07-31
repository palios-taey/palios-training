#!/usr/bin/env bash
# Immutable off-cluster worker for merging and validating a completed DDP LoRA.
set -euo pipefail

[ "$#" = 14 ] || {
  echo "REFUSE: bake worker requires 14 positional arguments." >&2
  exit 1
}

CONVERT_ROOT=$1
CONVERT_IMAGE=$2
TOOLS_DIR=$3
BASE=$4
ADAPTER=$5
STAGE=$6
OUTPUT=$7
TRAINING_COMPLETION=$8
ADAPTER_SHA=$9
ADAPTER_CONFIG_SHA=${10}
RUN_TAG=${11}
PLAN_SHA=${12}
BASE_MANIFEST_SHA=${13}
TOOLING_COMMIT=${14}
STEPS=979

for path in "$CONVERT_ROOT" "$TOOLS_DIR" "$BASE" "$ADAPTER" "$STAGE" \
  "$OUTPUT" "$TRAINING_COMPLETION"; do
  case "$path" in
    /*) ;;
    *)
      echo "REFUSE: worker path is not absolute: $path" >&2
      exit 1
      ;;
  esac
done
case "$STAGE:$OUTPUT" in
  "$CONVERT_ROOT"/*.staging.*:"$CONVERT_ROOT"/*_stage2_ddp_servable) ;;
  *)
    echo "REFUSE: worker output paths escaped the production bake contract." >&2
    exit 1
    ;;
esac
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|"")
    echo "REFUSE: unsafe run tag: $RUN_TAG" >&2
    exit 1
    ;;
esac
for digest in "$ADAPTER_SHA" "$ADAPTER_CONFIG_SHA" "$PLAN_SHA" \
  "$BASE_MANIFEST_SHA"; do
  case "$digest" in
    *[!0-9a-f]*|"")
      echo "REFUSE: malformed SHA-256 digest." >&2
      exit 1
      ;;
  esac
  [ "${#digest}" = 64 ]
done
case "$TOOLING_COMMIT" in
  *[!0-9a-f]*|"")
    echo "REFUSE: malformed tooling commit." >&2
    exit 1
    ;;
esac
[ "${#TOOLING_COMMIT}" = 40 ]

[ -d "$BASE" ] &&
[ -d "$ADAPTER" ] &&
[ -s "$TRAINING_COMPLETION" ] &&
[ ! -e "$STAGE" ] &&
[ ! -e "$OUTPUT" ] || {
  echo "REFUSE: bake inputs are absent or an output path already exists." >&2
  exit 1
}

container_python(){
  sudo docker run --rm --network none \
    -v "$CONVERT_ROOT:$CONVERT_ROOT" \
    --entrypoint /opt/venv/bin/python \
    "$CONVERT_IMAGE" "$@"
}

mkdir "$STAGE"
echo "=== MERGE — apply the exact checkpoint-$STEPS adapter to the CPT base ==="
sudo docker run --rm --network none \
  -v "$CONVERT_ROOT:$CONVERT_ROOT" \
  -e "BASE_MODEL=$BASE" \
  -e "LORA_PATH=$ADAPTER" \
  -e "OUTPUT_PATH=$STAGE" \
  -e "BASE_MANIFEST_SHA256=$BASE_MANIFEST_SHA" \
  --entrypoint /opt/venv/bin/python \
  "$CONVERT_IMAGE" "$TOOLS_DIR/bake_lora_nopeft.py"

echo
echo "=== VALIDATE — prove 352 targets changed and 847 tensors did not ==="
container_python "$TOOLS_DIR/finalize_stage2_ddp_bake.py" \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --output "$STAGE" \
  --training-completion "$TRAINING_COMPLETION" \
  --adapter-sha "$ADAPTER_SHA" \
  --adapter-config-sha "$ADAPTER_CONFIG_SHA" \
  --base-manifest-sha "$BASE_MANIFEST_SHA" \
  --run-tag "$RUN_TAG" \
  --plan-sha "$PLAN_SHA" \
  --steps "$STEPS" \
  --tooling-commit "$TOOLING_COMMIT"

echo
echo "=== MANIFEST — hash every promoted artifact byte ==="
[ -z "$(find "$STAGE" -type l -print -quit)" ] || {
  echo "REFUSE: staged artifact contains a symbolic link." >&2
  exit 1
}
(
  cd "$STAGE"
  find . -type f ! -name SOURCE_SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 -r sha256sum >SOURCE_SHA256SUMS
  sha256sum -c SOURCE_SHA256SUMS
)
sync
mv "$STAGE" "$OUTPUT"
echo "SFT_BAKE_PROMOTED output=$OUTPUT adapter_sha=$ADAPTER_SHA steps=$STEPS"
