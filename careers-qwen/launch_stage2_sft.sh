#!/bin/bash
# STAGE 2 — the SFT/LoRA module carrying every sanctioned row.
#
# WHY ONE MODULE AND NOT FIVE
# Modules 1-4 are ABSENT from the served model. cpt_refresh_v3 resumed directly from
# prod_v2_ep3_hf and dropped everything trained after it; module5_adapter then declared
# cpt_refresh_v3_servable as its base rather than module4_merged, so 4 and 5 are siblings
# rather than a chain. Every module's rows live in the canonical set, so training them as ONE
# adapter recovers modules 1-5 and removes the orphaning hazard entirely — there is only one
# adapter, so there is nothing left to orphan.
#
# SIZING IS DERIVED FROM THIS CORPUS, NEVER FROM ANOTHER RUN'S LOG.
# That distinction is not pedantic: the CPT stage was first launched with TOTAL_STEPS=157
# because global_batch=16 was read off an older run whose BATCH_SIZE_PER_RANK was 4. This
# config runs at 1, so global_batch is 4 and the real epoch is 628 steps — the 157-step run
# would have consumed 25% of the corpus and stopped, reporting success. The COVERAGE PROOF
# line caught it. The formula below is cross-checked against module4's real run: 101 rows at
# gb=4 logged steps/epoch=26, and ceil(101/4)=26.
# fleet.env is sourced BEFORE `set -u`, exactly as run_4node_27b_cpt.sh:4 does it, and for a
# concrete reason: fleet.env line 6 (THOR1_ENDPOINT) references THOR1_HOST, which is undefined
# at that point. Sourcing it under `set -u` therefore kills the script inside fleet.env itself
# with "fleet.env: line 6: THOR1_HOST: unbound variable" — an error that points at the topology
# file rather than at the script that mis-ordered two lines. Caught by dry-running this launcher
# while Stage 1 was still training; it would otherwise have died at Stage 2 launch.
cd "$(dirname "$0")/.."
[ -f fleet.env ] && . ./fleet.env
set -euo pipefail
# FAIL HERE, NOT 47 LINES LATER. `[ -f fleet.env ] && .` continues silently when the file is
# absent or the script is invoked from elsewhere, and the first use of ${SPARK_HOME} then dies
# under `set -u` with "line 68: SPARK_HOME: unbound variable" — an error that names a line
# nowhere near the cause. That is the same shape as the defect that killed every rank earlier
# today: a topology variable missing at the point of use, reported as a deployment problem
# rather than a config one. Observed while dry-checking this script 2026-07-28.
: "${SPARK_HOME:?fleet.env did not load (run this from the repo, or export SPARK_HOME)}"
: "${SPARK_MASTER:?fleet.env did not load (run this from the repo, or export SPARK_MASTER)}"
: "${SPARK_MGMT_IPS:?fleet.env did not define SPARK_MGMT_IPS}"

: "${SFT_CORPUS:?set SFT_CORPUS to the materialized sanctioned rows}"
: "${BASE_MODEL:?set BASE_MODEL to the Stage-1 CPT servable artifact}"
: "${EXPECTED_SFT_SAMPLES:?set EXPECTED_SFT_SAMPLES from sft_dataset_receipt.py}"
case "$SFT_CORPUS:$BASE_MODEL" in
  /*:/*) ;;
  *) echo "ABORT: SFT_CORPUS and BASE_MODEL must be absolute paths on every Spark." >&2; exit 1;;
esac

NODES=(${SPARK_MGMT_IPS})
[ "${#NODES[@]}" = 4 ] || {
  echo "ABORT: Stage-2 SFT requires four rank-ordered Spark nodes; got ${#NODES[@]}." >&2
  exit 1
}

CORPUS_BYTES=
CORPUS_SHA=
ROWS=
BASE_MANIFEST_SHA=
for rank in 0 1 2 3; do
  node=${NODES[$rank]}
  receipt=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$node" \
    "bash -s -- '$SFT_CORPUS' '$BASE_MODEL'" <<'REMOTE'
set -euo pipefail
corpus=$1
base=$2
test -f "$corpus"
test -f "$base/GRAFT_COMPLETE"
test -f "$base/weight_diff.json"
test -f "$base/training_provenance.json"
test -f "$base/SOURCE_SHA256SUMS"
(
  cd "$base"
  sha256sum -c SOURCE_SHA256SUMS >/dev/null
)
python3 - "$corpus" "$base" <<'PY'
import hashlib
import json
import os
import sys

corpus, base = sys.argv[1:]
digest = hashlib.sha256()
rows = 0
with open(corpus, "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        rows += chunk.count(b"\n")
index_path = os.path.join(base, "model.safetensors.index.json")
weight_map = json.load(open(index_path))["weight_map"]
missing = sorted({
    shard for shard in weight_map.values()
    if not os.path.isfile(os.path.join(base, shard))
})
if missing:
    raise SystemExit(f"missing indexed base shards: {missing}")
diff = float(json.load(open(os.path.join(base, "weight_diff.json")))["abs_mean_dW"])
json.load(open(os.path.join(base, "training_provenance.json")))
manifest = hashlib.sha256(open(os.path.join(base, "SOURCE_SHA256SUMS"), "rb").read()).hexdigest()
print(os.path.getsize(corpus), digest.hexdigest(), rows, len(weight_map), manifest, f"{diff:.9e}")
PY
REMOTE
)
  read -r bytes sha rows base_tensors manifest_sha base_diff <<<"$receipt"
  [ "$base_tensors" = 1199 ] || {
    echo "ABORT: rank$rank .$node base has $base_tensors tensors; expected 1199." >&2
    exit 1
  }
  if [ "$rank" = 0 ]; then
    CORPUS_BYTES=$bytes
    CORPUS_SHA=$sha
    ROWS=$rows
    BASE_MANIFEST_SHA=$manifest_sha
  else
    [ "$bytes $sha $rows $manifest_sha" = \
      "$CORPUS_BYTES $CORPUS_SHA $ROWS $BASE_MANIFEST_SHA" ] || {
      echo "ABORT: rank$rank .$node corpus/base receipt differs from rank0." >&2
      exit 1
    }
  fi
  echo "  rank$rank .$node corpus=$bytes/$rows/$sha base=1199/$manifest_sha diff=$base_diff"
done

# The production SFT evidence is dynamic-padding LoRA at batch 1 and MAX_SEQ=4096.
# The 741/737 tok/s batch-4 receipts are fixed-length PackedCPTDataset runs at 2560,
# whose memory/throughput shape does not establish a safe SFT batch.
# GPU CLOCK — set EXPLICITLY, never inherited. nvidia-smi -lgc persists across jobs, so a run
# that sets nothing takes whatever the previous job left. A bake leaves 1000MHz; an SFT launched
# after one therefore ran its whole length at 33% of the 3003MHz ceiling, silently. One
# definition lives in fleet.env; this reads it rather than carrying a seventh copy.
# EXPORTED, not merely set. `exec` at the end of this script replaces the process and the child
# inherits only EXPORTED variables, so a decided-but-unexported CLOCK_CAP arrived unset in the
# launcher and its ${CLOCK_CAP:?} killed the run. The value was always decided here; it just
# never travelled.
export CLOCK_CAP="${CLOCK_CAP:-1600}"
if [ "$CLOCK_CAP" != "0" ]; then
  echo "  clock: pinning graphics <= ${CLOCK_CAP}MHz on all 4 (ceiling ${SPARK_CLOCK_MAX_MHZ:-3003}MHz)"
  for h in ${SPARK_MGMT_IPS:?fleet.env did not define SPARK_MGMT_IPS}; do
    ssh -o ConnectTimeout=6 spark@"$h" \
      "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP >/dev/null 2>&1" \
      && echo "    .${h##*.} @${CLOCK_CAP}" || {
        echo "ABORT: .${h##*.} CLOCK SET FAILED" >&2
        exit 1
      }
  done
fi

PER_RANK=${BATCH_SIZE_PER_RANK:-1}
N_RANKS=4
MAX_SEQ=${MAX_SEQ:-4096}
case "$PER_RANK:$MAX_SEQ:$EXPECTED_SFT_SAMPLES" in
  *[!0-9:]*|*::*|:*|*:)
    echo "ABORT: batch, max sequence, and expected samples must be positive integers." >&2
    exit 1
    ;;
esac
[ "$PER_RANK" -gt 0 ] && [ "$MAX_SEQ" -gt 256 ] &&
[ "$EXPECTED_SFT_SAMPLES" -ge "$ROWS" ] || {
  echo "ABORT: invalid SFT sizing: rows=$ROWS samples=$EXPECTED_SFT_SAMPLES max_seq=$MAX_SEQ." >&2
  exit 1
}
GLOBAL_BATCH=$(( PER_RANK * N_RANKS ))
STEPS=$(( (EXPECTED_SFT_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
WARM=$(( STEPS / 10 ))
RESUMED_STEP=0
if [ -n "${RESUME_DELTA:-}" ]; then
  case "$RESUME_DELTA" in
    */checkpoint-[0-9]*) RESUMED_STEP=${RESUME_DELTA##*-};;
    *) echo "ABORT: RESUME_DELTA is not a numbered DCP checkpoint: $RESUME_DELTA" >&2; exit 1;;
  esac
fi

echo "=== STAGE 2 SIZING (derived from the corpus in hand) ==="
echo "  corpus      $SFT_CORPUS"
echo "  rows        $ROWS"
echo "  samples     $EXPECTED_SFT_SAMPLES ($((EXPECTED_SFT_SAMPLES - ROWS)) extra windows)"
echo "  bytes       $CORPUS_BYTES"
echo "  sha256      $CORPUS_SHA"
echo "  base        $BASE_MODEL"
echo "  base sha    $BASE_MANIFEST_SHA"
echo "  max_seq     $MAX_SEQ (packing OFF; assistant labels occur exactly once across supervised windows)"
echo "  per_rank=$PER_RANK x $N_RANKS ranks -> global_batch $GLOBAL_BATCH  ->  1 epoch = $STEPS steps"
echo "  TOTAL_STEPS=$STEPS WARMUP_STEPS=$WARM resumed_step=$RESUMED_STEP"
echo "  coverage    $((STEPS * GLOBAL_BATCH)) slots vs $EXPECTED_SFT_SAMPLES tokenizer samples"
python3 - "$STEPS" "$WARM" <<'PY'
import sys, math
t, w = int(sys.argv[1]), int(sys.argv[2])
f = lambda s: s/w if s < w else 0.1 + 0.45*(1 + math.cos(math.pi*(s-w)/max(1, t-w)))
peak = max(f(s) for s in range(1, t+1))
print(f"  peak mult   {peak:.4f}   {'(full LR reached)' if peak > 0.99 else '*** ANNEALED TAIL — DO NOT LAUNCH ***'}")
print(f"  dose sum-f  {sum(f(s) for s in range(1, t+1)):.1f}")
PY
echo

export SFT_JSONL="$SFT_CORPUS"
export SFT_DIR
SFT_DIR=$(dirname "$SFT_CORPUS")
unset BAKE_TO_HF DISABLE_FLA FP8 LANE_WEIGHTS TINY_LANE_CAP TINY_LANE_THRESHOLD
export MODEL_PATH="$BASE_MODEL"
export LORA_MODE=1 LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05
export CPT_PACKED=0 EPOCHS=1 EXACT_SFT_EPOCH=0 RESUME_MODEL_ONLY=0
export GATE_PREFLIGHT=1
export NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=INIT TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export LR=${LR:-1e-4}
export TOTAL_STEPS=$STEPS WARMUP_STEPS=$WARM
export SESSION_LIMIT=${SESSION_LIMIT:-250}
export SAVE_EVERY=${SAVE_EVERY:-$SESSION_LIMIT}
export CHECKPOINT_DCP=1
export BATCH_SIZE_PER_RANK=$PER_RANK
export MAX_SEQ EXPECTED_SFT_SAMPLES
export OUTPUT_DIR=${OUTPUT_DIR:-${SPARK_HOME}/training_outputs/stage2_all_rows}

# Match the targets the production module runs actually used (from module4's log), rather
# than the trainer default — a narrower target set silently trains a different model.
export LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj"

echo "=== launching 4-node ==="
exec bash dense-9b/recipes/run_4node_27b_cpt.sh
