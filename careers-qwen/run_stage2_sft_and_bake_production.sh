#!/usr/bin/env bash
# One production entrypoint: clean four-node SFT campaign, then validated bake.
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
set -euo pipefail
cd "$REPO_ROOT"

: "${RUN_TAG:?set RUN_TAG to the qualified CPT run tag}"
: "${SFT_CORPUS:?set SFT_CORPUS to the sanctioned SFT corpus on every Spark}"

DEPLOY_REF=${DEPLOY_SHA:-HEAD}
DEPLOY_SHA=$(git rev-parse --verify "${DEPLOY_REF}^{commit}")
PRODUCTION_FILES=(
  careers-qwen/run_stage2_sft_and_bake_production.sh
  careers-qwen/run_stage2_sft_ddp_till_done.sh
  careers-qwen/bake_stage2_ddp_production.sh
)
for file in "${PRODUCTION_FILES[@]}"; do
  [ "$(git hash-object -- "$file")" = "$(git rev-parse "${DEPLOY_SHA}:$file")" ] || {
    echo "REFUSE: integrated lifecycle file differs from $DEPLOY_SHA: $file" >&2
    exit 1
  }
done

export DEPLOY_SHA RUN_TAG SFT_CORPUS
bash careers-qwen/run_stage2_sft_ddp_till_done.sh
bash careers-qwen/bake_stage2_ddp_production.sh
