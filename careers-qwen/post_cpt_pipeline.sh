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

ARTIFACT_STORE=${ARTIFACT_STORE:-${POST_CPT_ARTIFACT_STORE:-}}
CONVERT_SSH=${CONVERT_SSH:-${POST_CPT_CONVERT_SSH:-}}
CONVERT_ROOT=${CONVERT_ROOT:-${POST_CPT_CONVERT_ROOT:-}}
CONVERT_GRAFT_BASE=${CONVERT_GRAFT_BASE:-${POST_CPT_GRAFT_BASE:-}}
CONVERT_IMAGE=${CONVERT_IMAGE:-${POST_CPT_CONVERT_IMAGE:-}}
CONVERT_TOOLS=${CONVERT_TOOLS:-${POST_CPT_CONVERT_TOOLS:-${CONVERT_ROOT%/}/tools}}

: "${ARTIFACT_STORE:?set ARTIFACT_STORE or POST_CPT_ARTIFACT_STORE to durable controller storage}"
: "${CONVERT_SSH:?set CONVERT_SSH or POST_CPT_CONVERT_SSH to the off-cluster conversion SSH target}"
: "${CONVERT_ROOT:?set CONVERT_ROOT or POST_CPT_CONVERT_ROOT to the off-cluster artifact root}"
: "${CONVERT_GRAFT_BASE:?set CONVERT_GRAFT_BASE or POST_CPT_GRAFT_BASE to the 1199-tensor serving donor}"
: "${CONVERT_IMAGE:?set CONVERT_IMAGE or POST_CPT_CONVERT_IMAGE to the pinned conversion image digest}"

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "ABORT: post-CPT requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}
for absolute_path in "$DCP_DIR" "$ARTIFACT_STORE" "$CONVERT_ROOT" "$CONVERT_GRAFT_BASE" "$CONVERT_TOOLS"; do
  case "$absolute_path" in
    /*) ;;
    *) echo "ABORT: production paths must be absolute: $absolute_path" >&2; exit 1;;
  esac
done

RC=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" "cat '$DCP_DIR/run_config.env' 2>/dev/null" || true)
[ -n "$RC" ] || {
  echo "ABORT: no run_config.env under $DCP_DIR on ${SPARK_MASTER}." >&2
  echo "The live trainer is the only authoritative source; this configuration cannot be reconstructed." >&2
  exit 1
}
rc_get(){ sed -n "s|^$1=||p" <<<"$RC" | tail -1; }

CORPUS=$(rc_get CPT_PATH_FROM_LOG)
PROV_TOTAL_STEPS=$(rc_get TOTAL_STEPS)
PROV_WARMUP_STEPS=$(rc_get WARMUP_STEPS)
PROV_LR=$(rc_get LR)
TRAIN_BASE=$(rc_get TRAIN_BASE)
CAPTURED_CORPUS_INPUTS=$(rc_get CORPUS_INPUTS)
# LR is required for the same reason the others are, and with more history behind it:
# run_4node_27b_cpt.sh:58-60 records that LR and WARMUP_STEPS were silently NOT forwarded
# until 2026-07-13, so every run before that trained at the trainer default regardless of
# what the operator set. WARMUP_STEPS was guarded here; LR was not — leaving the one value
# whose divergence is already documented as the unguarded one. A run whose learning rate
# cannot be named from its own capture cannot have its result explained later.
for required_name in CORPUS PROV_TOTAL_STEPS PROV_WARMUP_STEPS PROV_LR TRAIN_BASE; do
  [ -n "${!required_name}" ] || {
    echo "ABORT: $required_name absent from run_config.env — refusing to default it." >&2
    exit 1
  }
done
CORPUS_MANIFEST="${CORPUS}.manifest.json"
corpus_receipt_text=
if ! corpus_receipt_text=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
  "$(printf '%q ' /usr/bin/python3 - verify --corpus "$CORPUS" \
    --manifest "$CORPUS_MANIFEST" --receipt-lines)" \
  <careers-qwen/corpus_manifest.py); then
  echo "ABORT: packed corpus has no valid artifact-bound input manifest." >&2
  exit 1
fi
mapfile -t CORPUS_RECEIPT <<<"$corpus_receipt_text"
[ "${#CORPUS_RECEIPT[@]}" = 3 ] || {
  echo "ABORT: corpus manifest verifier returned an incomplete receipt." >&2
  exit 1
}
CORPUS_SHA=${CORPUS_RECEIPT[0]}
CORPUS_MANIFEST_SHA=${CORPUS_RECEIPT[1]}
CORPUS_INPUTS=${CORPUS_RECEIPT[2]}
[ -z "$CAPTURED_CORPUS_INPUTS" ] || [ "$CAPTURED_CORPUS_INPUTS" = "$CORPUS_INPUTS" ] || {
  echo "ABORT: run_config CORPUS_INPUTS disagrees with the packed artifact manifest." >&2
  exit 1
}
LOGSTEPS=$(rc_get TOTAL_STEPS_FROM_LOG)
[ -z "$LOGSTEPS" ] || [ "$LOGSTEPS" = "$PROV_TOTAL_STEPS" ] || {
  echo "ABORT: TOTAL_STEPS disagrees between process env ($PROV_TOTAL_STEPS) and log ($LOGSTEPS)." >&2
  exit 1
}
echo "$CORPUS_INPUTS" | grep -q voice && {
  echo "ABORT: a voice slice is named in CORPUS_INPUTS; it is out of production." >&2
  exit 1
}

RUN_TAG=$(basename "$DCP_DIR")
EXPORT_DIR="${SPARK_HOME}/exports/${RUN_TAG}_artifactB"
LOCAL_ARTIFACT="${ARTIFACT_STORE%/}/${RUN_TAG}_artifactB"
LOCAL_BASE="${ARTIFACT_STORE%/}/bases/${RUN_TAG}_training_base"
REMOTE_ARTIFACT="${CONVERT_ROOT%/}/${RUN_TAG}_artifactB"
CONVERT_BASE="${CONVERT_ROOT%/}/${RUN_TAG}_training_base"
HF_OUT="${CONVERT_ROOT%/}/${RUN_TAG}_hf"
SERVABLE_OUT="${CONVERT_ROOT%/}/${RUN_TAG}_servable"
HF_STAGE="${HF_OUT}.staging.$$"
SERVABLE_STAGE="${SERVABLE_OUT}.staging.$$"
GRAFT_BASE=$CONVERT_GRAFT_BASE
CONVERT_CORPUS="${CONVERT_ROOT%/}/corpora/$(basename "$CORPUS")"
CONVERT_CORPUS_MANIFEST="${CONVERT_CORPUS}.manifest.json"
SANCTION=${SANCTION:-${POST_CPT_SANCTION:-"treasurer task-dfa3fd75 2026-07-28"}}
SPACE_MARGIN_BYTES=${SPACE_MARGIN_BYTES:-10737418240}
FRESH_UPTIME_MAX=${FRESH_UPTIME_MAX:-180}
TOOLING_COMMIT=$(git rev-parse HEAD)

SYNC_TOOL=dense-9b/recipes/artifact_b_sync.sh
MODEL_SYNC_TOOL=dense-9b/recipes/model_snapshot_sync.sh
EXPORT_TOOL=dense-9b/recipes/bake_27b.sh
REMOTE_PY_TOOLS=(
  dense-9b/recipes/bake_dcp_offline.py
  careers-qwen/graft_cpt_into_servable.py
  careers-qwen/measure_cpt_delta.py
  careers-qwen/corpus_manifest.py
  ${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT}/sources/palios-training-c164d35/tree/careers-qwen/emit_training_provenance.py
)
EXPORT_RUNTIME=(
  dense-9b/trainers/train_fsdp_dense_9b.py
  dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh
  dense-9b/configs/fsdp_dense_27b.yaml
)
PRODUCTION_FILES=(
  careers-qwen/post_cpt_pipeline.sh
  "$SYNC_TOOL"
  "$MODEL_SYNC_TOOL"
  "$EXPORT_TOOL"
  "${REMOTE_PY_TOOLS[@]}"
  "${EXPORT_RUNTIME[@]}"
)
for production_file in "${PRODUCTION_FILES[@]}"; do
  git ls-files --error-unmatch "$production_file" >/dev/null 2>&1 || {
    echo "ABORT: production runtime is not tracked at tooling commit: $production_file" >&2
    exit 1
  }
done
git diff --quiet "$TOOLING_COMMIT" -- "${PRODUCTION_FILES[@]}" || {
  echo "ABORT: post-CPT runtime differs from tooling commit $TOOLING_COMMIT." >&2
  exit 1
}

remote_tensor_count(){
  ssh -o BatchMode=yes "$CONVERT_SSH" "/usr/bin/python3 - '$1'" <<'PY'
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
  ssh -o BatchMode=yes "$CONVERT_SSH" "$quoted"
}

declare -a SOURCE_SHARD_BYTES
declare -a BOOT_IDS

verify_checkpoint(){
  local phase=$1 rank node receipt shard_bytes trainers free_bytes
  echo "=== CHECKPOINT $phase ==="
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "test -f '$DCP_DIR/checkpoint-$CKPT/COMPLETE' &&
       test -f '$DCP_DIR/checkpoint-$CKPT/dcp/__${rank}_0.distcp' &&
       shard_bytes=\$(stat -c %s '$DCP_DIR/checkpoint-$CKPT/dcp/__${rank}_0.distcp') &&
       trainers=\$(ps -eo args= | awk '/[t]orchrun|[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}') &&
       free_bytes=\$(df -B1 --output=avail '$DCP_DIR' | tail -1 | tr -d ' ') &&
       printf '%s %s %s\\n' \"\$shard_bytes\" \"\$trainers\" \"\$free_bytes\"")
    read -r shard_bytes trainers free_bytes <<<"$receipt"
    [ "$trainers" = 0 ] || {
      echo "ABORT: rank$rank on $node still has $trainers trainer process(es)." >&2
      exit 1
    }
    if [ "$phase" = before-reboot ]; then
      SOURCE_SHARD_BYTES[$rank]=$shard_bytes
    else
      [ "$shard_bytes" = "${SOURCE_SHARD_BYTES[$rank]}" ] || {
        echo "ABORT: checkpoint shard changed across reboot on rank$rank." >&2
        exit 1
      }
    fi
    echo "  rank$rank .$node COMPLETE shard=$shard_bytes trainers=0 free=$free_bytes"
  done
}

reboot_cluster(){
  local rank node result boot_id uptime_seconds back
  echo
  echo "=== CLUSTER RESET — reboot all four, prove new boot IDs ==="
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    BOOT_IDS[$rank]=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "cat /proc/sys/kernel/random/boot_id")
  done
  for node in "${NODES[@]}"; do
    (
      timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 spark@"$node" \
        "sudo systemctl reboot" >/dev/null 2>&1 || true
    ) &
  done
  wait

  for _ in $(seq 1 60); do
    back=0
    for rank in 0 1 2 3; do
      node=${NODES[$rank]}
      result=$(timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=4 spark@"$node" \
        "printf '%s %s\\n' \"\$(cat /proc/sys/kernel/random/boot_id)\" \"\$(cut -d. -f1 /proc/uptime)\"" \
        2>/dev/null || true)
      read -r boot_id uptime_seconds <<<"$result"
      if [ -n "${boot_id:-}" ] &&
         [ "$boot_id" != "${BOOT_IDS[$rank]}" ] &&
         [ "${uptime_seconds:-999999}" -lt "$FRESH_UPTIME_MAX" ] 2>/dev/null; then
        back=$((back + 1))
      fi
    done
    [ "$back" = 4 ] && {
      echo "  all four nodes returned with changed boot IDs and uptime < ${FRESH_UPTIME_MAX}s"
      verify_checkpoint after-reboot
      return 0
    }
    sleep 10
  done
  echo "ABORT: not all four nodes proved a fresh reboot within 10 minutes." >&2
  exit 1
}

deploy_export_runtime(){
  local node file local_sha remote_sha
  echo
  echo "=== EXPORT RUNTIME DEPLOY — exact controller bytes to every rank ==="
  for node in "${NODES[@]}"; do
    ssh -o BatchMode=yes spark@"$node" \
      "mkdir -p '$SPARK_HOME/palios-training/dense-9b/trainers' \
                '$SPARK_HOME/palios-training/dense-9b/recipes' \
                '$SPARK_HOME/palios-training/dense-9b/configs'"
    for file in "${EXPORT_RUNTIME[@]}"; do
      scp -q -o BatchMode=yes "$file" "spark@$node:$SPARK_HOME/palios-training/$file"
      local_sha=$(sha256sum "$file" | cut -d' ' -f1)
      remote_sha=$(ssh -o BatchMode=yes spark@"$node" \
        "sha256sum '$SPARK_HOME/palios-training/$file' | cut -d' ' -f1")
      [ "$local_sha" = "$remote_sha" ] || {
        echo "ABORT: deployed runtime hash mismatch for $file on $node." >&2
        exit 1
      }
    done
    echo "  .$node runtime hash-verified"
  done
}

CKPT=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
  "ls -d '$DCP_DIR'/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1")
: "${CKPT:?no checkpoint found under $DCP_DIR}"

echo "=== RUN CONFIG — captured from the live trainer ==="
printf '  %-18s %s\n' training-base "$TRAIN_BASE" corpus "$CORPUS" checkpoint "$CKPT" \
  total-steps "$PROV_TOTAL_STEPS" warmup "$PROV_WARMUP_STEPS" local-artifact "$LOCAL_ARTIFACT" \
  local-base "$LOCAL_BASE" convert-root "$CONVERT_ROOT" graft-base "$GRAFT_BASE"
echo "  slices: $(tr ',' ' ' <<<"$CORPUS_INPUTS" | wc -w) (0 voice)"

echo
echo "=== 0. PRODUCTION INFRASTRUCTURE PREFLIGHT ==="
bash -n "$SYNC_TOOL"
bash -n "$MODEL_SYNC_TOOL"
bash -n "$EXPORT_TOOL"
python3 -m py_compile "${REMOTE_PY_TOOLS[@]}"
ssh -o BatchMode=yes "$CONVERT_SSH" \
  "test -f '$GRAFT_BASE/model.safetensors.index.json';
   mkdir -p '$CONVERT_ROOT' '$CONVERT_TOOLS' '$(dirname "$CONVERT_CORPUS")'"
graft_tensors=$(remote_tensor_count "$GRAFT_BASE")
[ "$graft_tensors" = 1199 ] || {
  echo "ABORT: serving donor has $graft_tensors tensors; expected 1199." >&2
  exit 1
}
convert_versions=$(remote_container_python -c \
  'import torch,transformers; print(torch.__version__); print(transformers.__version__)')
printf '%s\n' "$convert_versions"
grep -qx '2.10.0' <<<"$convert_versions"
grep -qx '5.3.0' <<<"$convert_versions"
echo "  pinned conversion image present; torch=2.10.0 transformers=5.3.0; donor=1199 tensors"

for local_tool in "${REMOTE_PY_TOOLS[@]}"; do
  rsync -a -e "ssh -o BatchMode=yes" "$local_tool" "$CONVERT_SSH:$CONVERT_TOOLS/"
  local_sha=$(sha256sum "$local_tool" | cut -d' ' -f1)
  remote_sha=$(ssh -o BatchMode=yes "$CONVERT_SSH" \
    "sha256sum '$CONVERT_TOOLS/$(basename "$local_tool")' | cut -d' ' -f1")
  [ "$local_sha" = "$remote_sha" ] || {
    echo "ABORT: deployed tool hash mismatch for $local_tool." >&2
    exit 1
  }
done
echo "  conversion tools deployed byte-exact"

verify_checkpoint before-reboot
reboot_cluster
deploy_export_runtime

echo
echo "=== 1. ARTIFACT-B EXPORT — canonical coordinated DCP path ==="
if [ -d "$LOCAL_ARTIFACT" ] &&
   python3 dense-9b/recipes/bake_dcp_offline.py \
     --assembled "$LOCAL_ARTIFACT" --verify-manifests --verify-only >/dev/null; then
  echo "  SKIP export — verified portable Artifact B already exists at $LOCAL_ARTIFACT"
else
  metadata_count=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
    "test -f '$EXPORT_DIR/.metadata' && echo 1 || echo 0")
  ready_count=0
  for rank in 0 1 2 3; do
    ready=$(ssh -o BatchMode=yes spark@"${NODES[$rank]}" \
      "test -f '$EXPORT_DIR/READY.rank${rank}' && echo 1 || echo 0")
    ready_count=$((ready_count + ready))
  done
  if [ "$metadata_count" = 1 ] && [ "$ready_count" = 4 ]; then
    echo "  SKIP export — accepted Artifact B already exists on the Sparks (.metadata + 4/4 READY)"
  else
    (
      cd dense-9b/recipes
      RESET_INCOMPLETE_EXPORT=1 \
      EXPORT_DCP="$EXPORT_DIR" \
      RESUME_DELTA="$DCP_DIR/checkpoint-$CKPT" \
      OUTPUT_DIR="$DCP_DIR" \
      BASE_MODEL="$TRAIN_BASE" \
      CPT_DATA="$CORPUS" \
      bash bake_27b.sh
    )
  fi
fi

echo
echo "=== 2. COLLECT OFF THE SPARKS — atomic + source-hash-verified ==="
bash "$SYNC_TOOL" collect "$EXPORT_DIR" "$LOCAL_ARTIFACT"
bash "$SYNC_TOOL" retire-sparks "$EXPORT_DIR" "$LOCAL_ARTIFACT"
bash "$MODEL_SYNC_TOOL" collect "$TRAIN_BASE" "$LOCAL_BASE"

corpus_local="${ARTIFACT_STORE%/}/corpora/$(basename "$CORPUS")"
corpus_manifest_local="${corpus_local}.manifest.json"
mkdir -p "$(dirname "$corpus_local")"
corpus_sha=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" "sha256sum '$CORPUS' | cut -d' ' -f1")
[ "$corpus_sha" = "$CORPUS_SHA" ] || {
  echo "ABORT: corpus changed after its input manifest was verified." >&2
  exit 1
}
if [ ! -f "$corpus_local" ] || [ "$(sha256sum "$corpus_local" | cut -d' ' -f1)" != "$corpus_sha" ]; then
  corpus_stage="${corpus_local}.staging.$$"
  scp -q -o BatchMode=yes "spark@${SPARK_MASTER}:$CORPUS" "$corpus_stage"
  [ "$(sha256sum "$corpus_stage" | cut -d' ' -f1)" = "$corpus_sha" ] || {
    echo "ABORT: controller corpus copy failed its source hash." >&2
    exit 1
  }
  mv "$corpus_stage" "$corpus_local"
fi
if [ ! -f "$corpus_manifest_local" ] ||
   [ "$(sha256sum "$corpus_manifest_local" | cut -d' ' -f1)" != "$CORPUS_MANIFEST_SHA" ]; then
  corpus_manifest_stage="${corpus_manifest_local}.staging.$$"
  scp -q -o BatchMode=yes "spark@${SPARK_MASTER}:$CORPUS_MANIFEST" "$corpus_manifest_stage"
  [ "$(sha256sum "$corpus_manifest_stage" | cut -d' ' -f1)" = "$CORPUS_MANIFEST_SHA" ] || {
    echo "ABORT: controller corpus-manifest copy failed its source hash." >&2
    exit 1
  }
  mv "$corpus_manifest_stage" "$corpus_manifest_local"
fi
python3 careers-qwen/corpus_manifest.py verify \
  --corpus "$corpus_local" --manifest "$corpus_manifest_local" >/dev/null

echo
echo "=== 3. STAGE TO THE OFF-CLUSTER CONVERSION HOST ==="
bash "$MODEL_SYNC_TOOL" push "$LOCAL_BASE" "$CONVERT_SSH" "$CONVERT_BASE"
bash "$SYNC_TOOL" push "$LOCAL_ARTIFACT" "$CONVERT_SSH" "$REMOTE_ARTIFACT"
rsync -a -e "ssh -o BatchMode=yes" "$corpus_local" "$CONVERT_SSH:$CONVERT_CORPUS"
rsync -a -e "ssh -o BatchMode=yes" \
  "$corpus_manifest_local" "$CONVERT_SSH:$CONVERT_CORPUS_MANIFEST"
[ "$(ssh -o BatchMode=yes "$CONVERT_SSH" "sha256sum '$CONVERT_CORPUS' | cut -d' ' -f1")" = "$corpus_sha" ] || {
  echo "ABORT: conversion-host corpus copy failed its source hash." >&2
  exit 1
}
[ "$(ssh -o BatchMode=yes "$CONVERT_SSH" \
  "sha256sum '$CONVERT_CORPUS_MANIFEST' | cut -d' ' -f1")" = "$CORPUS_MANIFEST_SHA" ] || {
  echo "ABORT: conversion-host corpus manifest failed its source hash." >&2
  exit 1
}
base_tensors=$(remote_tensor_count "$CONVERT_BASE")
[ "$base_tensors" = 851 ] || {
  echo "ABORT: exact training-base snapshot has $base_tensors tensors; expected 851." >&2
  exit 1
}
echo "  exact 851-tensor training base and portable Artifact B staged off-cluster"

echo
echo "=== 4. OFFLINE CONVERT — pinned container, no Spark consolidation, no full gather ==="
hf_tensors=0
if ssh -o BatchMode=yes "$CONVERT_SSH" "test -e '$HF_OUT'"; then
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "test -f '$HF_OUT/CONVERT_COMPLETE' && test -f '$HF_OUT/model.safetensors.index.json'" || {
    echo "ABORT: existing HF output has no completion marker: $HF_OUT" >&2
    exit 1
  }
  hf_tensors=$(remote_tensor_count "$HF_OUT")
  [ "$hf_tensors" = 851 ] || {
    echo "ABORT: existing HF output is incomplete or wrong ($hf_tensors tensors): $HF_OUT" >&2
    exit 1
  }
fi
if [ "$hf_tensors" = 851 ]; then
  echo "  SKIP convert — verified 851-tensor HF output already exists"
else
  base_bytes=$(du -sb "$LOCAL_BASE" | cut -f1)
  convert_disk_free=$(ssh -o BatchMode=yes "$CONVERT_SSH" \
    "df -B1 --output=avail '$CONVERT_ROOT' | tail -1 | tr -d ' '")
  convert_mem_free=$(ssh -o BatchMode=yes "$CONVERT_SSH" \
    "awk '/MemAvailable:/{print \$2*1024}' /proc/meminfo")
  [ "$convert_disk_free" -ge $((base_bytes + SPACE_MARGIN_BYTES)) ] || {
    echo "ABORT: conversion host lacks disk for HF output ($convert_disk_free free)." >&2
    exit 1
  }
  [ "$convert_mem_free" -ge $((base_bytes + SPACE_MARGIN_BYTES)) ] || {
    echo "ABORT: conversion host lacks idle memory for the production converter ($convert_mem_free free)." >&2
    echo "Do not co-run conversion beside a live 27B service; use the redundant-node maintenance path." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$CONVERT_SSH" "test ! -e '$HF_STAGE'"
  remote_container_python "$CONVERT_TOOLS/bake_dcp_offline.py" \
    --assembled "$REMOTE_ARTIFACT" --base "$CONVERT_BASE" --out "$HF_STAGE" --verify-manifests
  [ "$(remote_tensor_count "$HF_STAGE")" = 851 ] || {
    echo "ABORT: staged conversion did not produce 851 tensors: $HF_STAGE" >&2
    exit 1
  }
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "test -f '$HF_STAGE/CONVERT_COMPLETE' && sudo mv '$HF_STAGE' '$HF_OUT'"
fi

hf_tensors=$(remote_tensor_count "$HF_OUT")
[ "$hf_tensors" = 851 ] || {
  echo "ABORT: expected 851 tensors after conversion, got $hf_tensors." >&2
  exit 1
}
ssh -o BatchMode=yes "$CONVERT_SSH" "test -f '$HF_OUT/CONVERT_COMPLETE'"
case "$REMOTE_ARTIFACT" in
  "${CONVERT_ROOT%/}"/*_artifactB)
    ssh -o BatchMode=yes "$CONVERT_SSH" "sudo rm -rf -- '$REMOTE_ARTIFACT'"
    ;;
  *) echo "ABORT: unsafe conversion-host Artifact-B cleanup target: $REMOTE_ARTIFACT" >&2; exit 1;;
esac
echo "  conversion complete; remote transient retired; verified controller Artifact B remains"

echo
echo "=== 5. WEIGHT-DIFF HARD GATE ==="
diff_json=$(remote_container_python "$CONVERT_TOOLS/measure_cpt_delta.py" \
  --base "$CONVERT_BASE" --cand "$HF_OUT" --json)
echo "$diff_json"
printf '%s\n' "$diff_json" | ssh -o BatchMode=yes "$CONVERT_SSH" \
  "sudo tee '$HF_OUT/weight_diff.json' >/dev/null"
diff_abs=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["abs_mean_dW"])' <<<"$diff_json")
gate=$(python3 - "$diff_abs" <<'PY'
import sys

value = float(sys.argv[1])
print("IN_BAND" if 5e-5 <= value <= 8e-4 else "BELOW" if value < 5e-5 else "ABOVE")
PY
)
printf 'WEIGHT_DIFF %.9e %s\n' "$diff_abs" "$gate"
[ "$gate" = IN_BAND ] || {
  echo "FULL STOP: weight-diff is $gate; no unreviewed graft, handoff, or SFT." >&2
  if [ "$gate" = BELOW ]; then
    echo "After the approved Chats review, bind its receipt and continue without repeating export:" >&2
    echo "  DCP_DIR='$DCP_DIR' BELOW_BAND_REVIEW_RECEIPT=<absolute-review.json> bash careers-qwen/finalize_post_cpt_candidate.sh" >&2
  fi
  exit 3
}

echo
echo "=== 6. GRAFT — only after the weight-diff gate passes ==="
servable_tensors=0
if ssh -o BatchMode=yes "$CONVERT_SSH" "test -e '$SERVABLE_OUT'"; then
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "test -f '$SERVABLE_OUT/GRAFT_COMPLETE' && test -f '$SERVABLE_OUT/model.safetensors.index.json'" || {
    echo "ABORT: existing servable output has no completion marker: $SERVABLE_OUT" >&2
    exit 1
  }
  servable_tensors=$(remote_tensor_count "$SERVABLE_OUT")
  [ "$servable_tensors" = 1199 ] || {
    echo "ABORT: existing servable output is incomplete or wrong ($servable_tensors tensors)." >&2
    exit 1
  }
fi
if [ "$servable_tensors" != 1199 ]; then
  hf_bytes=$(ssh -o BatchMode=yes "$CONVERT_SSH" "du -sb '$HF_OUT' | cut -f1")
  convert_disk_free=$(ssh -o BatchMode=yes "$CONVERT_SSH" \
    "df -B1 --output=avail '$CONVERT_ROOT' | tail -1 | tr -d ' '")
  [ "$convert_disk_free" -ge $((hf_bytes + SPACE_MARGIN_BYTES)) ] || {
    echo "ABORT: conversion host lacks disk for the servable graft." >&2
    exit 1
  }
  ssh -o BatchMode=yes "$CONVERT_SSH" "test ! -e '$SERVABLE_STAGE'"
  remote_container_python "$CONVERT_TOOLS/graft_cpt_into_servable.py" \
    --base "$GRAFT_BASE" --cpt "$HF_OUT" --out "$SERVABLE_STAGE"
  [ "$(remote_tensor_count "$SERVABLE_STAGE")" = 1199 ] || {
    echo "ABORT: staged graft did not produce 1199 tensors: $SERVABLE_STAGE" >&2
    exit 1
  }
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "test -f '$SERVABLE_STAGE/GRAFT_COMPLETE' && sudo mv '$SERVABLE_STAGE' '$SERVABLE_OUT'"
fi

servable_tensors=$(remote_tensor_count "$SERVABLE_OUT")
[ "$servable_tensors" = 1199 ] || {
  echo "ABORT: graft produced $servable_tensors tensors, expected 1199." >&2
  exit 1
}
ssh -o BatchMode=yes "$CONVERT_SSH" "test -f '$SERVABLE_OUT/GRAFT_COMPLETE'"
ssh -o BatchMode=yes "$CONVERT_SSH" \
  "sudo cp '$HF_OUT/weight_diff.json' '$SERVABLE_OUT/weight_diff.json'"

echo
echo "=== 7. PROVENANCE + TRANSIENT RETIREMENT ==="
for artifact_stage in "$HF_OUT:cpt" "$SERVABLE_OUT:graft"; do
  artifact=${artifact_stage%%:*}
  stage=${artifact_stage##*:}
  remote_container_python "${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT}/sources/palios-training-c164d35/tree/careers-qwen/emit_training_provenance.py" \
    --artifact "$artifact" --stage "$stage" --base "$CONVERT_BASE" --corpus "$CONVERT_CORPUS" \
    --total-steps "$PROV_TOTAL_STEPS" --completed-step "$CKPT" \
    --warmup-steps "$PROV_WARMUP_STEPS" \
    --resumed-step 0 --tooling-commit "$TOOLING_COMMIT" \
    --sanction "$SANCTION" --corpus-manifest "$CONVERT_CORPUS_MANIFEST"
done
case "$HF_OUT:$CONVERT_BASE" in
  "${CONVERT_ROOT%/}"/*_hf:"${CONVERT_ROOT%/}"/*_training_base)
    ssh -o BatchMode=yes "$CONVERT_SSH" "sudo rm -rf -- '$HF_OUT' '$CONVERT_BASE'"
    ;;
  *) echo "ABORT: unsafe successful-run transient cleanup targets." >&2; exit 1;;
esac

command -v taey-notify >/dev/null || {
  echo "ABORT: in-band artifact is complete, but taey-notify is unavailable for the required infra handoff." >&2
  exit 1
}
taey-notify infra \
  "POST-CPT READY: ${RUN_TAG} weight-diff=${diff_abs} IN_BAND; 1199-tensor candidate at ${CONVERT_SSH}:${SERVABLE_OUT}; corpus_sha256=${corpus_sha}; training owns SFT staging, serving remains infra-owned." \
  --type response_ready

echo
echo "DONE"
echo "  weight-diff       $diff_abs IN BAND"
echo "  servable          $CONVERT_SSH:$SERVABLE_OUT"
echo "  tensors           1199"
echo "  corpus-sha256     $corpus_sha"
echo "  portable hold     $LOCAL_ARTIFACT"
echo "  exact base hold   $LOCAL_BASE"
echo "  next               stage exact candidate and launch the production SFT driver"
