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
CONVERT_IMAGE=${CONVERT_IMAGE:-${POST_CPT_CONVERT_IMAGE:-}}

: "${CONVERT_SSH:?set CONVERT_SSH or POST_CPT_CONVERT_SSH to the Thor delivery SSH target}"
: "${CONVERT_ROOT:?set CONVERT_ROOT or POST_CPT_CONVERT_ROOT to the Thor model root}"
: "${CONVERT_IMAGE:?set CONVERT_IMAGE or POST_CPT_CONVERT_IMAGE to the pinned conversion image digest}"

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "ABORT: post-CPT requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}
for absolute_path in "$DCP_DIR" "$CONVERT_ROOT"; do
  case "$absolute_path" in
    /*) ;;
    *) echo "ABORT: production paths must be absolute: $absolute_path" >&2; exit 1;;
  esac
done
if [ -n "$ARTIFACT_STORE" ]; then
  case "$ARTIFACT_STORE" in
    /*) ;;
    *) echo "ABORT: durable ARTIFACT_STORE must be absolute when set: $ARTIFACT_STORE" >&2; exit 1;;
  esac
fi

RC=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" "cat '$DCP_DIR/run_config.env' 2>/dev/null" || true)
[ -n "$RC" ] || {
  echo "ABORT: no run_config.env under $DCP_DIR on ${SPARK_MASTER}." >&2
  echo "The live trainer is the only authoritative source; this configuration cannot be reconstructed." >&2
  exit 1
}
rc_get(){ sed -n "s|^$1=||p" <<<"$RC" | tail -1; }

CORPUS=$(rc_get CPT_PATH_FROM_LOG)
PROV_TOTAL_STEPS=$(rc_get TOTAL_STEPS)
PROV_SESSION_LIMIT=$(rc_get SESSION_LIMIT)
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
for required_name in CORPUS PROV_TOTAL_STEPS PROV_SESSION_LIMIT PROV_WARMUP_STEPS PROV_LR TRAIN_BASE; do
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
BAKE_ROOT="${SPARK_HOME%/}/post_cpt/${RUN_TAG}"
REMOTE_ARTIFACT="${BAKE_ROOT}/${RUN_TAG}_artifactB"
CONVERT_BASE="$TRAIN_BASE"
HF_OUT="${BAKE_ROOT}/${RUN_TAG}_hf"
SERVABLE_OUT="${BAKE_ROOT}/${RUN_TAG}_servable"
HF_STAGE="${HF_OUT}.staging.$$"
SERVABLE_STAGE="${SERVABLE_OUT}.staging.$$"
DELIVERY_OUT="${CONVERT_ROOT%/}/${RUN_TAG}_servable"
CONVERT_TOOLS="${BAKE_ROOT}/tools"
CONVERT_CORPUS="$CORPUS"
CONVERT_CORPUS_MANIFEST="$CORPUS_MANIFEST"
DURABLE_ARTIFACT="${ARTIFACT_STORE:+${ARTIFACT_STORE%/}/${RUN_TAG}_artifactB}"
DURABLE_BASE="${ARTIFACT_STORE:+${ARTIFACT_STORE%/}/bases/${RUN_TAG}_training_base}"
DURABLE_SERVABLE="${ARTIFACT_STORE:+${ARTIFACT_STORE%/}/${RUN_TAG}_servable}"
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
  careers-qwen/emit_training_provenance.py
  careers-qwen/post_cpt_contract.py
  careers-qwen/verify_artifact_b.py
  scripts/training_lifecycle.py
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

bake_tensor_count(){
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "/usr/bin/python3 - '$1'" <<'PY'
import json
import os
import sys

index_path = os.path.join(sys.argv[1], "model.safetensors.index.json")
print(len(json.load(open(index_path))["weight_map"]) if os.path.isfile(index_path) else 0)
PY
}

# THE CONTAINER AND THE HOST MUST AGREE ABOUT THE FILESYSTEM. Both halves below are
# required, and until 2026-08-19 neither was present, which is why NO bake had ever
# completed on this node: `find $SPARK_HOME/post_cpt -name weight_diff.json` returned 0
# and there were no bake roots at all.
bake_container_python(){
  local -a command=(
    sudo docker run --rm --network none
    -v "$SPARK_HOME:$SPARK_HOME"
  )
  # VISIBILITY. The corpus the trainer recorded may live OUTSIDE $SPARK_HOME (production
  # keeps it under /var/spark). emit_training_provenance.py runs INSIDE this container and
  # reads it, so a path the host can see and the container cannot is a hard abort:
  #   ABORT: corpus not found: /var/spark/isma/training/cpt_prod_v4_repos_packed_8192.jsonl
  # while that file was present and byte-identical on all four nodes. Mount its DIRECTORY
  # read-only, derived from the value — the location comes from run_config.env and is not
  # ours to hardcode. Skipped when already covered by the $SPARK_HOME mount.
  local corpus_dir
  if [ -n "${CONVERT_CORPUS:-}" ]; then
    corpus_dir=$(dirname "$CONVERT_CORPUS")
    case "$corpus_dir" in
      "$SPARK_HOME"|"$SPARK_HOME"/*) : ;;
      # NEVER mount host root. A corpus at "/x.jsonl" makes dirname "/", which the /*
      # branch below matches, and the mount would become -v /:/:ro — the entire host
      # filesystem inside the container. Bounded by --network none, :ro and the pinned
      # image, and no production corpus lives at root, but a defence that depends on the
      # input never taking one specific shape is not a defence. Found by gatekeeper's R5
      # on this PR, who verified the glob does match "/"; excluded here rather than
      # deferred, because a known hole with a follow-up ticket is a shipped hole.
      /) : ;;
      /*) command+=(-v "$corpus_dir:$corpus_dir:ro") ;;
    esac
  fi
  command+=(--entrypoint /opt/venv/bin/python "$CONVERT_IMAGE" "$@")
  local quoted
  printf -v quoted '%q ' "${command[@]}"
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "$quoted"
  local rc=$?
  # OWNERSHIP. The image REQUIRES root — verified by execution: `--user 1000:1000` crashes
  # it outright, so running unprivileged is not available. Every artifact it writes
  # therefore lands root-owned, and the host side of this pipeline runs as spark and cannot
  # write beside it. That killed the run twice in sequence, once per container stage:
  #   tee: .../cpt_prod_v5_repos_1ep_hf/weight_diff.json: Permission denied
  #   cp:  .../cpt_prod_v5_repos_1ep_servable/weight_diff.json: Permission denied
  # Reclaim ownership after EVERY container step, not once at the end — each stage creates
  # a new directory. Metadata-only, so it costs nothing on a 52G tree. Never masks rc.
  if [ -n "${BAKE_ROOT:-}" ]; then
    ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
      "test -d '$BAKE_ROOT' && sudo chown -R spark:spark '$BAKE_ROOT'" || true
  fi
  return $rc
}

bake_python(){
  local quoted
  printf -v quoted '%q ' /usr/bin/python3 "$@"
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "$quoted"
}

declare -a SOURCE_SHARD_BYTES
declare -a BOOT_IDS

verify_checkpoint(){
  local phase=$1 rank node receipt shard_bytes trainers free_bytes
  echo "=== CHECKPOINT $phase ==="
  for rank in 0 1 2 3; do
    node=${NODES[$rank]}
    receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
      "test -f '$DCP_DIR/$CKPT_NAME/COMPLETE' &&
       test -f '$DCP_DIR/$CKPT_NAME/dcp/__${rank}_0.distcp' &&
       shard_bytes=\$(stat -c %s '$DCP_DIR/$CKPT_NAME/dcp/__${rank}_0.distcp') &&
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

# CHECKPOINT SELECTION — the run's FINAL artifact wins, and a glob cannot find it.
# train_fsdp_dense_9b.py:3643 names a COMPLETED run's save `final`, not `checkpoint-<step>`:
#     ckpt_name = "final" if final else f"checkpoint-{step}"
# So `ls -d checkpoint-*` is blind to the finished model and silently selects the last
# INTERMEDIATE checkpoint — baking a partially-trained model and shipping it as the product,
# with every downstream gate passing because the artifact it was handed is internally consistent.
# OBSERVED 2026-08-18: cpt_qwen38_v3 held checkpoint-73, checkpoint-146 and final (step=218).
# The glob chose 146 and the bake launched on two-thirds of the run. Caught only because the
# launch line printed the path. This is why the run that COMPLETES is the one that breaks:
# every prior bake ran on an interrupted run whose last save really was checkpoint-<N>.
# CKPT_NAME is the directory to read; CKPT stays the completed-STEP number provenance records.
if ssh -o BatchMode=yes spark@"${SPARK_MASTER}" "test -f '$DCP_DIR/final/COMPLETE'" 2>/dev/null; then
  CKPT_NAME=final
  CKPT=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
    "python3 -c \"import torch; print(torch.load('$DCP_DIR/final/trainer_meta.pt', map_location='cpu', weights_only=False)['step'])\"" 2>/dev/null)
  [ -n "$CKPT" ] || {
    echo "ABORT: $DCP_DIR/final exists but its trainer_meta.pt yields no step." >&2
    echo "The completed-step is provenance, not decoration — refusing to guess it." >&2
    exit 1
  }
  echo "  checkpoint selection: final/ (completed run, step $CKPT)"
else
  CKPT=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" \
    "ls -d '$DCP_DIR'/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1")
  : "${CKPT:?no checkpoint found under $DCP_DIR}"
  CKPT_NAME="checkpoint-$CKPT"
  echo "  checkpoint selection: $CKPT_NAME (no final/ present — run did not reach its horizon)"
fi

echo "=== RUN CONFIG — captured from the live trainer ==="
printf '  %-18s %s\n' training-base "$TRAIN_BASE" corpus "$CORPUS" checkpoint "$CKPT_NAME (step $CKPT)" \
  total-steps "$PROV_TOTAL_STEPS" session-limit "$PROV_SESSION_LIMIT" warmup "$PROV_WARMUP_STEPS" \
  bake-host "spark@$SPARK_MASTER" bake-root "$BAKE_ROOT" thor "$CONVERT_SSH:$DELIVERY_OUT" \
  durable-store "${ARTIFACT_STORE:-<disabled>}"
echo "  slices: $(tr ',' ' ' <<<"$CORPUS_INPUTS" | wc -w) (0 voice)"

echo
echo "=== 0. PRODUCTION INFRASTRUCTURE PREFLIGHT ==="
bash -n "$SYNC_TOOL"
bash -n "$MODEL_SYNC_TOOL"
bash -n "$EXPORT_TOOL"
python3 -m py_compile "${REMOTE_PY_TOOLS[@]}"
ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "test -f '$TRAIN_BASE/model.safetensors.index.json'; mkdir -p '$BAKE_ROOT' '$CONVERT_TOOLS'"
ssh -o BatchMode=yes "$CONVERT_SSH" "mkdir -p '$CONVERT_ROOT'"
convert_versions=$(bake_container_python -c \
  'import torch,transformers; print(torch.__version__); print(transformers.__version__)')
printf '%s\n' "$convert_versions"
grep -qx '2.10.0' <<<"$convert_versions"
grep -qx '5.3.0' <<<"$convert_versions"
echo "  pinned conversion image present on bake Spark; torch=2.10.0 transformers=5.3.0"

for local_tool in "${REMOTE_PY_TOOLS[@]}"; do
  rsync -a -e "ssh -o BatchMode=yes" "$local_tool" "spark@$SPARK_MASTER:$CONVERT_TOOLS/"
  local_sha=$(sha256sum "$local_tool" | cut -d' ' -f1)
  remote_sha=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "sha256sum '$CONVERT_TOOLS/$(basename "$local_tool")' | cut -d' ' -f1")
  [ "$local_sha" = "$remote_sha" ] || {
    echo "ABORT: deployed tool hash mismatch for $local_tool." >&2
    exit 1
  }
done
echo "  conversion tools deployed byte-exact to bake Spark"

DONOR_RECEIPT=$(bake_python "$CONVERT_TOOLS/post_cpt_contract.py" resolve-donor \
  --model-path-loaded "$TRAIN_BASE")
echo "$DONOR_RECEIPT"
GRAFT_BASE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["donor"])' <<<"$DONOR_RECEIPT")
[ -n "$GRAFT_BASE" ] || { echo "ABORT: donor resolver returned no donor." >&2; exit 1; }

ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "python3 - validate --log '$DCP_DIR/lifecycle_events.jsonl'" \
  < scripts/training_lifecycle.py
LAST_EVENT=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "python3 - last --log '$DCP_DIR/lifecycle_events.jsonl' --json" < scripts/training_lifecycle.py)
LAST_STATE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' <<<"$LAST_EVENT")
case "$LAST_STATE" in
  CHECKPOINT_SAVED|BAKE_COMPLETE) ;;
  *)
    echo "ABORT: post-CPT requires CHECKPOINT_SAVED or resumable BAKE_COMPLETE, got $LAST_STATE." >&2
    exit 1
    ;;
esac

verify_checkpoint before-reboot
reboot_cluster
deploy_export_runtime

echo
echo "=== 1. ARTIFACT-B EXPORT — canonical coordinated DCP path ==="
if bash "$SYNC_TOOL" verify-sparks "$EXPORT_DIR" >/tmp/post_cpt_export_verify.$$ 2>&1; then
  cat /tmp/post_cpt_export_verify.$$
  rm -f /tmp/post_cpt_export_verify.$$
  echo "  SKIP export — every Spark shard matches its rank manifest"
else
  cat /tmp/post_cpt_export_verify.$$ >&2
  rm -f /tmp/post_cpt_export_verify.$$
  (
    cd dense-9b/recipes
    RESET_INCOMPLETE_EXPORT=1 \
    EXPORT_DCP="$EXPORT_DIR" \
    RESUME_DELTA="$DCP_DIR/$CKPT_NAME" \
    OUTPUT_DIR="$DCP_DIR" \
    BASE_MODEL="$TRAIN_BASE" \
    CPT_DATA="$CORPUS" \
    bash bake_27b.sh
  )
fi
bash "$SYNC_TOOL" verify-sparks "$EXPORT_DIR"

echo
echo "=== 2. ASSEMBLE ON ONE SPARK — no controller-storage staging ==="
bash "$SYNC_TOOL" assemble-spark "$EXPORT_DIR" "$REMOTE_ARTIFACT"
corpus_sha=$(ssh -o BatchMode=yes spark@"${SPARK_MASTER}" "sha256sum '$CORPUS' | cut -d' ' -f1")
[ "$corpus_sha" = "$CORPUS_SHA" ] || {
  echo "ABORT: corpus changed after its input manifest was verified." >&2
  exit 1
}
base_tensors=$(bake_tensor_count "$CONVERT_BASE")
[ "$base_tensors" = 851 ] || {
  echo "ABORT: exact training-base snapshot has $base_tensors tensors; expected 851." >&2
  exit 1
}
echo "  exact 851-tensor training base, corpus, and portable Artifact B are node-local"

echo
echo "=== 3. OFFLINE CONVERT — pinned container on the bake Spark, no full gather ==="
hf_tensors=0
if ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test -e '$HF_OUT'"; then
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "test -f '$HF_OUT/CONVERT_COMPLETE' && test -f '$HF_OUT/model.safetensors.index.json'" || {
    echo "ABORT: existing HF output has no completion marker: $HF_OUT" >&2
    exit 1
  }
  hf_tensors=$(bake_tensor_count "$HF_OUT")
  [ "$hf_tensors" = 851 ] || {
    echo "ABORT: existing HF output is incomplete or wrong ($hf_tensors tensors): $HF_OUT" >&2
    exit 1
  }
fi
if [ "$hf_tensors" = 851 ]; then
  echo "  SKIP convert — verified 851-tensor HF output already exists"
else
  base_bytes=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" "du -sb '$CONVERT_BASE' | cut -f1")
  convert_disk_free=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "df -B1 --output=avail '$BAKE_ROOT' | tail -1 | tr -d ' '")
  convert_mem_free=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
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
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test ! -e '$HF_STAGE'"
  bake_container_python "$CONVERT_TOOLS/bake_dcp_offline.py" \
    --assembled "$REMOTE_ARTIFACT" --base "$CONVERT_BASE" --out "$HF_STAGE" --verify-manifests
  [ "$(bake_tensor_count "$HF_STAGE")" = 851 ] || {
    echo "ABORT: staged conversion did not produce 851 tensors: $HF_STAGE" >&2
    exit 1
  }
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "test -f '$HF_STAGE/CONVERT_COMPLETE' && mv '$HF_STAGE' '$HF_OUT'"
fi

hf_tensors=$(bake_tensor_count "$HF_OUT")
[ "$hf_tensors" = 851 ] || {
  echo "ABORT: expected 851 tensors after conversion, got $hf_tensors." >&2
  exit 1
}
ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test -f '$HF_OUT/CONVERT_COMPLETE'"
echo "  conversion complete on bake Spark; Artifact B retained until delivery verifies"

echo
echo "=== 4. WEIGHT-DIFF HARD GATE ==="
diff_json=$(bake_container_python "$CONVERT_TOOLS/measure_cpt_delta.py" \
  --base "$GRAFT_BASE" --cand "$HF_OUT" --json)
echo "$diff_json"
printf '%s\n' "$diff_json" | ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "tee '$HF_OUT/weight_diff.json' >/dev/null"
diff_abs=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["abs_mean_dW"])' <<<"$diff_json")
bake_python "$CONVERT_TOOLS/post_cpt_contract.py" weight-gate \
  --weight-diff "$HF_OUT/weight_diff.json" || {
  echo "FULL STOP: weight-diff is out of band; no unreviewed graft, handoff, or SFT." >&2
  if python3 -c 'import sys; sys.exit(not (float(sys.argv[1]) < 5e-5))' "$diff_abs"; then
    echo "After the approved Chats review, bind its receipt and continue without repeating export:" >&2
    echo "  DCP_DIR='$DCP_DIR' BELOW_BAND_REVIEW_RECEIPT=<absolute-review.json> bash careers-qwen/finalize_post_cpt_candidate.sh" >&2
  fi
  exit 3
}
printf 'WEIGHT_DIFF %.9e IN_BAND\n' "$diff_abs"

echo
echo "=== 5. GRAFT — per-run donor only, after the weight-diff gate passes ==="
servable_tensors=0
if ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test -e '$SERVABLE_OUT'"; then
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "test -f '$SERVABLE_OUT/GRAFT_COMPLETE' && test -f '$SERVABLE_OUT/model.safetensors.index.json'" || {
    echo "ABORT: existing servable output has no completion marker: $SERVABLE_OUT" >&2
    exit 1
  }
  servable_tensors=$(bake_tensor_count "$SERVABLE_OUT")
  [ "$servable_tensors" = 1199 ] || {
    echo "ABORT: existing servable output is incomplete or wrong ($servable_tensors tensors)." >&2
    exit 1
  }
fi
if [ "$servable_tensors" != 1199 ]; then
  hf_bytes=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" "du -sb '$HF_OUT' | cut -f1")
  convert_disk_free=$(ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "df -B1 --output=avail '$BAKE_ROOT' | tail -1 | tr -d ' '")
  [ "$convert_disk_free" -ge $((hf_bytes + SPACE_MARGIN_BYTES)) ] || {
    echo "ABORT: conversion host lacks disk for the servable graft." >&2
    exit 1
  }
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test ! -e '$SERVABLE_STAGE'"
  bake_container_python "$CONVERT_TOOLS/graft_cpt_into_servable.py" \
    --base "$GRAFT_BASE" --cpt "$HF_OUT" --out "$SERVABLE_STAGE"
  [ "$(bake_tensor_count "$SERVABLE_STAGE")" = 1199 ] || {
    echo "ABORT: staged graft did not produce 1199 tensors: $SERVABLE_STAGE" >&2
    exit 1
  }
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "test -f '$SERVABLE_STAGE/GRAFT_COMPLETE' && mv '$SERVABLE_STAGE' '$SERVABLE_OUT'"
fi

servable_tensors=$(bake_tensor_count "$SERVABLE_OUT")
[ "$servable_tensors" = 1199 ] || {
  echo "ABORT: graft produced $servable_tensors tensors, expected 1199." >&2
  exit 1
}
ssh -o BatchMode=yes spark@"$SPARK_MASTER" "test -f '$SERVABLE_OUT/GRAFT_COMPLETE'"
ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "cp '$HF_OUT/weight_diff.json' '$SERVABLE_OUT/weight_diff.json'"

echo
echo "=== 6. PROVENANCE + FINAL MODEL CONTRACT ==="
for artifact_stage in "$HF_OUT:cpt" "$SERVABLE_OUT:graft"; do
  artifact=${artifact_stage%%:*}
  stage=${artifact_stage##*:}
  bake_container_python "$CONVERT_TOOLS/emit_training_provenance.py" \
    --artifact "$artifact" --stage "$stage" --base "$CONVERT_BASE" --corpus "$CONVERT_CORPUS" \
    --total-steps "$PROV_TOTAL_STEPS" --completed-step "$CKPT" \
    --warmup-steps "$PROV_WARMUP_STEPS" \
    --resumed-step 0 --tooling-commit "$TOOLING_COMMIT" \
    --sanction "$SANCTION" --corpus-manifest "$CONVERT_CORPUS_MANIFEST"
done
bake_python "$CONVERT_TOOLS/post_cpt_contract.py" final-gate \
  --training-base "$CONVERT_BASE" --converted "$HF_OUT" --servable "$SERVABLE_OUT" \
  --weight-diff "$HF_OUT/weight_diff.json"
seal_json=$(bake_python "$CONVERT_TOOLS/post_cpt_contract.py" seal-artifact --root "$SERVABLE_OUT")
echo "$seal_json"
artifact_sha=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["artifact_sha256"])' <<<"$seal_json")
if [ "$LAST_STATE" = CHECKPOINT_SAVED ]; then
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
    "python3 - append --log '$DCP_DIR/lifecycle_events.jsonl' --state BAKE_COMPLETE \
       --run-tag '$RUN_TAG' --artifact-sha256 '$artifact_sha'" < scripts/training_lifecycle.py
else
  recorded_artifact_sha=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["artifact_sha256"])' \
    <<<"$LAST_EVENT")
  [ "$artifact_sha" = "$recorded_artifact_sha" ] || {
    echo "ABORT: resumed bake artifact digest=$artifact_sha lifecycle=$recorded_artifact_sha." >&2
    exit 1
  }
  echo "  resume: BAKE_COMPLETE already durable; artifact digest remeasured as $artifact_sha"
fi

echo
echo "=== 7. DIRECT DELIVERY — bake Spark to Thor, content-verified ==="
DELIVERY_STAGE="${DELIVERY_OUT}.staging.$$"
if ssh -o BatchMode=yes "$CONVERT_SSH" "test -e '$DELIVERY_OUT'"; then
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "python3 - verify-artifact --root '$DELIVERY_OUT' --expected-sha256 '$artifact_sha'" \
    < careers-qwen/post_cpt_contract.py || {
    echo "ABORT: existing Thor destination does not match this run: $CONVERT_SSH:$DELIVERY_OUT" >&2
    exit 1
  }
  echo "  SKIP delivery — exact content already verified on Thor"
else
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "test ! -e '$DELIVERY_STAGE'; mkdir -p '$DELIVERY_STAGE'"
  printf -v direct_rsync '%q ' rsync -a --partial -e 'ssh -o BatchMode=yes' \
    "$SERVABLE_OUT/" "$CONVERT_SSH:$DELIVERY_STAGE/"
  ssh -o BatchMode=yes spark@"$SPARK_MASTER" "$direct_rsync"
  ssh -o BatchMode=yes "$CONVERT_SSH" \
    "python3 - verify-artifact --root '$DELIVERY_STAGE' --expected-sha256 '$artifact_sha'" \
    < careers-qwen/post_cpt_contract.py
  ssh -o BatchMode=yes "$CONVERT_SSH" "mv '$DELIVERY_STAGE' '$DELIVERY_OUT'"
fi
ssh -o BatchMode=yes "$CONVERT_SSH" \
  "python3 - verify-artifact --root '$DELIVERY_OUT' --expected-sha256 '$artifact_sha'" \
  < careers-qwen/post_cpt_contract.py
ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "python3 - append --log '$DCP_DIR/lifecycle_events.jsonl' --state THOR_DELIVERED \
     --run-tag '$RUN_TAG' --artifact-sha256 '$artifact_sha' --served-root '$CONVERT_SSH:$DELIVERY_OUT'" \
  < scripts/training_lifecycle.py

echo
echo "=== 8. DURABLE COPY LAST — outside the delivery critical path ==="
if [ -n "$ARTIFACT_STORE" ]; then
  bash "$SYNC_TOOL" collect "$EXPORT_DIR" "$DURABLE_ARTIFACT"
  bash "$MODEL_SYNC_TOOL" collect "$TRAIN_BASE" "$DURABLE_BASE"
  if [ -e "$DURABLE_SERVABLE" ]; then
    python3 careers-qwen/post_cpt_contract.py verify-artifact \
      --root "$DURABLE_SERVABLE" --expected-sha256 "$artifact_sha"
  else
    durable_stage="${DURABLE_SERVABLE}.staging.$$"
    mkdir -p "$durable_stage"
    rsync -a --partial -e "ssh -o BatchMode=yes" \
      "spark@$SPARK_MASTER:$SERVABLE_OUT/" "$durable_stage/"
    python3 careers-qwen/post_cpt_contract.py verify-artifact \
      --root "$durable_stage" --expected-sha256 "$artifact_sha"
    mv "$durable_stage" "$DURABLE_SERVABLE"
  fi
  echo "  durable Artifact B, training base, and servable copy verified at $ARTIFACT_STORE"
else
  echo "  durable copy disabled — ARTIFACT_STORE is not required for delivery"
fi

for rank in 0 1 2 3; do
  ssh -o BatchMode=yes spark@"${NODES[$rank]}" \
    "case '$EXPORT_DIR' in '${SPARK_HOME}'/exports/*_artifactB) rm -rf -- '$EXPORT_DIR';; *) exit 1;; esac"
done
ssh -o BatchMode=yes spark@"$SPARK_MASTER" \
  "case '$BAKE_ROOT' in '${SPARK_HOME}'/post_cpt/'$RUN_TAG') rm -rf -- '$BAKE_ROOT';; *) exit 1;; esac"
echo "  verified delivery complete; exact Spark export and bake transients retired"

command -v taey-notify >/dev/null || {
  echo "ABORT: in-band artifact is complete, but taey-notify is unavailable for the required control handoff." >&2
  exit 1
}
: "${POST_CPT_NOTIFY_TARGET:?set POST_CPT_NOTIFY_TARGET in fleet.env to the live control session that owns serving cutover}"
taey-notify "$POST_CPT_NOTIFY_TARGET" \
  "POST-CPT READY: ${RUN_TAG} weight-diff=${diff_abs} IN_BAND; 1199-tensor candidate at ${CONVERT_SSH}:${DELIVERY_OUT}; artifact_sha256=${artifact_sha}; corpus_sha256=${corpus_sha}; training owns SFT staging, serving remains infra-owned." \
  --type response_ready

echo
echo "DONE"
echo "  weight-diff       $diff_abs IN BAND"
echo "  servable          $CONVERT_SSH:$DELIVERY_OUT"
echo "  artifact-sha256   $artifact_sha"
echo "  tensors           1199"
echo "  corpus-sha256     $corpus_sha"
echo "  durable hold      ${ARTIFACT_STORE:-disabled}"
echo "  next               stage exact candidate and launch the production SFT driver"
