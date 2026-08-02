#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Mira-side 4-node orchestrator for Qwen3.6-27B full-parameter CPT.
# Launches the per-node recipe (launch_cpt_qwen36_27b_fsdp.sh) on each Spark in a
# detached tmux, master (.68=rank0) FIRST so it binds :29500 before workers dial in.
#
# PRE-CONDITION (caller's responsibility, per reboot-after-every-run): all 4 nodes
# freshly REBOOTED + pristine (avail~108G, /dev/shm/nccl*=0, no torch procs). This
# script does NOT reboot — it assumes clean nodes and refuses to kill-and-relaunch.
set -uo pipefail

MASTER=${SPARK_MASTER}            # rank 0 — torchrun rendezvous host (${SPARK_RAIL_MASTER} rail)
WORKERS=(${SPARK_MGMT_IPS#* })   # ranks 1,2,3
ALL=("$MASTER" "${WORKERS[@]}")

# Run knobs — bf16 + keep_low_precision_grads fit (commit c63bc2d); chunked2560 corpus.
export CPT_DATA="${CPT_DATA:-/var/spark/isma/training/cpt_raw_corpus_train_no_superseded.chunked2560.jsonl}"
# Collective-reduction v2 (GAIA power/thermal fix, memory-safe): micro_bsz 32/12/2 ballooned memory
# 80->114G and HUNG. Better lever: keep micro-batches SMALL (16/4/1 = known-good 80G) and cut
# TOKEN_BUDGET_PER_STEP 262144->65536 -> ~4x FEWER micro-batches/collectives per optimizer step
# -> lower CX-7/proxy duty per step + opt-step barriers give the fabric periodic recovery -> power
# relief WITHOUT touching the solved 80G memory. Smaller effective batch (fine for CPT lr=1e-5).
# Knobs are env-overridable (caller may export any of these; production defaults below).
# Added for the DCP checkpoint round-trip test: OUTPUT_DIR / RESUME_DELTA / CHECKPOINT_DCP passthrough.
: "${MAX_SEQ:=2560}"; : "${TOKEN_BUDGET_PER_STEP:=65536}"; : "${TOTAL_STEPS:=3000}"
: "${SESSION_LIMIT:=200}"; : "${SAVE_EVERY:=66}"; : "${CHECKPOINT_DCP:=1}"
: "${CPT_SHORT_BATCH:=8}"; : "${CPT_MID_BATCH:=4}"; : "${CPT_LONG_BATCH:=1}"
# PACKED mode (fixed-length, uniform shape → no fragmentation): set CPT_DATA to a *packed* corpus +
# BATCH_SIZE_PER_RANK (bucket vars are then ignored by the trainer's else-branch dataloader).
: "${BATCH_SIZE_PER_RANK:=1}"; : "${CPT_PACKED:=0}"
RUN_ENV="CPT_DATA=$CPT_DATA MAX_SEQ=$MAX_SEQ BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK CPT_PACKED=$CPT_PACKED CPT_SHORT_BATCH=$CPT_SHORT_BATCH CPT_MID_BATCH=$CPT_MID_BATCH CPT_LONG_BATCH=$CPT_LONG_BATCH TOKEN_BUDGET_PER_STEP=$TOKEN_BUDGET_PER_STEP TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT SAVE_EVERY=$SAVE_EVERY CHECKPOINT_DCP=$CHECKPOINT_DCP"
# TOPOLOGY MUST BE FORWARDED — the nodes do not have fleet.env.
# This script sources fleet.env on MIRA, but fleet.env is gitignored and is NOT deployed to the
# Sparks. launch_cpt_qwen36_27b_fsdp.sh:399 dereferences SPARK_RAIL_MASTER under `set -u`, so
# without this line every remote rank dies instantly with
#     launch_cpt_qwen36_27b_fsdp.sh: line 399: SPARK_RAIL_MASTER: unbound variable
# after the corpus check has already printed OK — which reads like a healthy start. Observed
# 2026-07-27 on the CPT re-dose launch, and independently on a module-5 launch an hour earlier.
# RUN_ENV is an ALLOWLIST: a var not named here does not reach the node, however it was exported.
RUN_ENV="$RUN_ENV SPARK_RAIL_MASTER=$SPARK_RAIL_MASTER"
# SPARK_HOME has the SAME defect and it is worse, because it fails EARLIER and reads as a
# deployment problem rather than a config one. ce60925 ("de-umbilical: remove every hardcoded
# operator path — repo is going PUBLIC") replaced <SPARK_HOME> with ${SPARK_HOME} throughout
# launch_cpt_qwen36_27b_fsdp.sh, whose line 13 dereferences it under `set -u`. fleet.env is
# gitignored and NOT deployed to the Sparks, and SPARK_HOME was never added to this allowlist —
# so the public-safe node script cannot run on a node at all:
#     launch_cpt_qwen36_27b_fsdp.sh: line 13: SPARK_HOME: unbound variable
# It stayed invisible only because the Sparks still held a PRE-de-umbilical copy with the path
# hardcoded. Observed 2026-07-28 the moment a current copy was deployed to the nodes: every rank
# died instantly, after the launcher had already printed a healthy GEMM preflight and
# "launched rank N" for all four — which reads exactly like a successful start.
RUN_ENV="$RUN_ENV SPARK_HOME=$SPARK_HOME"
[ -n "${OUTPUT_DIR:-}" ] && RUN_ENV="$RUN_ENV OUTPUT_DIR=$OUTPUT_DIR"
[ -n "${RESUME_DELTA:-}" ] && RUN_ENV="$RUN_ENV RESUME_DELTA=$RESUME_DELTA"
[ -n "${RESUME_MODEL_ONLY:-}" ] && RUN_ENV="$RUN_ENV RESUME_MODEL_ONLY=$RESUME_MODEL_ONLY"
# LR + WARMUP_STEPS passthrough (2026-07-13 fix: these were NOT forwarded, so every run silently used
# the trainer default lr=1e-5 regardless of the LR env set on the Mira shell — the under-dose bug).
[ -n "${LR:-}" ] && RUN_ENV="$RUN_ENV LR=$LR"
# LoRA module-training passthrough (LORA_MODE=1 -> the launcher takes the SFT_DIR path and the
# trainer attaches PEFT before prepare; unset = unchanged legacy CPT behavior).
[ -n "${LORA_MODE:-}" ] && RUN_ENV="$RUN_ENV LORA_MODE=$LORA_MODE"
[ -n "${LORA_R:-}" ] && RUN_ENV="$RUN_ENV LORA_R=$LORA_R"
[ -n "${LORA_ALPHA:-}" ] && RUN_ENV="$RUN_ENV LORA_ALPHA=$LORA_ALPHA"
[ -n "${LORA_DROPOUT:-}" ] && RUN_ENV="$RUN_ENV LORA_DROPOUT=$LORA_DROPOUT"
[ -n "${DISABLE_FLA:-}" ] && RUN_ENV="$RUN_ENV DISABLE_FLA=$DISABLE_FLA"
# infra host-side diagnostic passthroughs (NCCL flight recorder + proxy-thread debug)
[ -n "${NCCL_DEBUG:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG=$NCCL_DEBUG"
[ -n "${NCCL_DEBUG_SUBSYS:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
[ -n "${TORCH_NCCL_TRACE_BUFFER_SIZE:-}" ] && RUN_ENV="$RUN_ENV TORCH_NCCL_TRACE_BUFFER_SIZE=$TORCH_NCCL_TRACE_BUFFER_SIZE"
[ -n "${LORA_TARGET_MODULES:-}" ] && RUN_ENV="$RUN_ENV LORA_TARGET_MODULES=$LORA_TARGET_MODULES"
[ -n "${SFT_DIR:-}" ] && RUN_ENV="$RUN_ENV SFT_DIR=$SFT_DIR"
[ -n "${SFT_JSONL:-}" ] && RUN_ENV="$RUN_ENV SFT_JSONL=$SFT_JSONL"
[ -n "${EXPECTED_SFT_SAMPLES:-}" ] && RUN_ENV="$RUN_ENV EXPECTED_SFT_SAMPLES=$EXPECTED_SFT_SAMPLES"
[ -n "${EPOCHS:-}" ] && RUN_ENV="$RUN_ENV EPOCHS=$EPOCHS"
[ -n "${EXACT_SFT_EPOCH:-}" ] && RUN_ENV="$RUN_ENV EXACT_SFT_EPOCH=$EXACT_SFT_EPOCH"
[ -n "${LANE_WEIGHTS:-}" ] && RUN_ENV="$RUN_ENV LANE_WEIGHTS=$LANE_WEIGHTS"
[ -n "${TINY_LANE_CAP:-}" ] && RUN_ENV="$RUN_ENV TINY_LANE_CAP=$TINY_LANE_CAP"
[ -n "${TINY_LANE_THRESHOLD:-}" ] && RUN_ENV="$RUN_ENV TINY_LANE_THRESHOLD=$TINY_LANE_THRESHOLD"
[ -n "${MODEL_PATH:-}" ] && RUN_ENV="$RUN_ENV MODEL_PATH=$MODEL_PATH"
[ -n "${WARMUP_STEPS:-}" ] && RUN_ENV="$RUN_ENV WARMUP_STEPS=$WARMUP_STEPS"
# HORIZON_PARTIAL: declares a deliberately short packed-CPT run (throughput probe, smoke test)
# to the trainer's CPT horizon contract, which otherwise requires TOTAL_STEPS to cover the whole
# corpus. MUST be named here: RUN_ENV is an ALLOWLIST, and dropped silently it would not weaken
# the gate but strand it — the probe would be refused on the node with a declaration the operator
# believes they made, which reads as a broken gate and invites bypassing it. The trainer requires
# this to EQUAL TOTAL_STEPS, so a stale value fails closed rather than open.
[ -n "${HORIZON_PARTIAL:-}" ] && RUN_ENV="$RUN_ENV HORIZON_PARTIAL=$HORIZON_PARTIAL"
# NSYS_* must be named here or the profile silently never arms: RUN_ENV is an ALLOWLIST, and a var
# absent from it does not reach the node. That is the same mechanism as the 2026-07-13 LR/WARMUP
# non-forwarding bug recorded above — the run would look normal and produce no trace.
# RUN_ENV is an ALLOWLIST — an unnamed var never reaches the node. LIGER/TORCH_COMPILE are
# throughput levers whose whole purpose is A/B, so a silently-dropped flag would produce a
# null result that looks like a real measurement.
[ -n "${AC_LAYER_GRANULAR:-}" ] && RUN_ENV="$RUN_ENV AC_LAYER_GRANULAR=$AC_LAYER_GRANULAR"
[ -n "${AC_LAYER_CLS:-}" ] && RUN_ENV="$RUN_ENV AC_LAYER_CLS=$AC_LAYER_CLS"
# QUARANTINE_DIGESTS: the trainer DEFAULTS this to the in-repo registry and refuses to start
# without it, so forwarding is only needed to override. Named here so an override actually
# reaches the node -- RUN_ENV is an allowlist and an unnamed var is silently dropped.
[ -n "${QUARANTINE_DIGESTS:-}" ] && RUN_ENV="$RUN_ENV QUARANTINE_DIGESTS=$QUARANTINE_DIGESTS"
[ -n "${LIGER:-}" ] && RUN_ENV="$RUN_ENV LIGER=$LIGER"
[ -n "${TORCH_COMPILE:-}" ] && RUN_ENV="$RUN_ENV TORCH_COMPILE=$TORCH_COMPILE"
[ -n "${TORCH_COMPILE_MODE:-}" ] && RUN_ENV="$RUN_ENV TORCH_COMPILE_MODE=$TORCH_COMPILE_MODE"
[ -n "${NSYS_PROFILE_STEP:-}" ] && RUN_ENV="$RUN_ENV NSYS_PROFILE_STEP=$NSYS_PROFILE_STEP"
[ -n "${NSYS_PROFILE_ALL_RANKS:-}" ] && RUN_ENV="$RUN_ENV NSYS_PROFILE_ALL_RANKS=$NSYS_PROFILE_ALL_RANKS"
[ -n "${NSYS_OUT_DIR:-}" ] && RUN_ENV="$RUN_ENV NSYS_OUT_DIR=$NSYS_OUT_DIR"
[ -n "${FP32_MASTER:-}" ] && RUN_ENV="$RUN_ENV FP32_MASTER=$FP32_MASTER"
[ -n "${ADAFACTOR_ALPHA_MODE:-}" ] && RUN_ENV="$RUN_ENV ADAFACTOR_ALPHA_MODE=$ADAFACTOR_ALPHA_MODE"
# ADAFACTOR_EPS1 — DEFAULTED HERE, not merely forwarded. 2026-07-29 root cause of a 148-step
# 27B CPT that completed clean and landed 1.284e-05, BELOW our own under-dosed reference.
# Unset, the trainer falls back to finfo(bf16).eps = 7.8125e-03. The second-moment estimate on
# this model sits entirely under eps1^2, so floor_frac measured 1.000 on ALL 25 logged samples:
# rsqrt() becomes the CONSTANT 1/eps1 = 128 and the update degenerates to 128*grad — linear in
# gradient, Adafactor's scale-invariance gone. Measured RMS(U_hat) 0.0002-0.0130 against the
# author's own bands (healthy ~O(1), starved <0.05). Per-step delta 2.5e-09 vs a bf16 ULP of
# 6.2e-05 = 0.00004 ULP, so stochastic rounding had essentially nothing to accumulate.
# Restoring normalization is ~5000x on the update field.
# WHY IT WAS MISSED: run_till_done_v2/v3 and run_refresh_gate all set ADAFACTOR_EPS1=fp32 and
# then call THIS launcher. Invoked directly, the launcher only forwarded a value nobody had set,
# so the correct recipe depended on the ENTRY POINT rather than on the launcher. A run started
# the "wrong" way looks identical in every log: same tok/s, same memory, same loss, same clean
# save. Defaulting it here makes the launcher correct by itself and the wrappers redundant
# rather than load-bearing.
# Horizon recommended fp32 on 2026-07-14 and the code gated it "unchanged until Chats rule";
# the 5-lane consult of 2026-07-29 ruled. This is that ruling applied.
: "${ADAFACTOR_EPS1:=fp32}"
RUN_ENV="$RUN_ENV ADAFACTOR_EPS1=$ADAFACTOR_EPS1"
# The dose gauge is what caught this. It must never be optional again -- it reported FAIL-LOW at
# step 20 of a run that then executed 128 more steps unchallenged.
: "${ADAFACTOR_DOSE_LOG:=1}"
[ -n "${ADAFACTOR_DOSE_LOG:-}" ] && RUN_ENV="$RUN_ENV ADAFACTOR_DOSE_LOG=$ADAFACTOR_DOSE_LOG"
[ -n "${BAKE_TO_HF:-}" ] && RUN_ENV="$RUN_ENV BAKE_TO_HF=$BAKE_TO_HF"
# Forward NCCL debug capture to the per-node sessions (the orchestrator only passes an allowlist, so
# NCCL_DEBUG set on the Mira shell would NOT reach the remote tmux without this). Enables the fabric
# bus-bandwidth/topology capture (NCCL_DEBUG=INFO NCCL_DEBUG_FILE=...).
[ -n "${NCCL_DEBUG:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG=$NCCL_DEBUG"
[ -n "${NCCL_DEBUG_FILE:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE"
[ -n "${NCCL_DEBUG_SUBSYS:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
[ -n "${FP8:-}" ] && RUN_ENV="$RUN_ENV FP8=$FP8"
RECIPE_DIR=${SPARK_HOME}/palios-training/dense-9b/recipes
LOGDIR=${SPARK_HOME}/cpt27b_logs

launch_rank () {  # $1=host  $2=rank
  local host=$1 rank=$2
  # LOG ROTATION (2026-07-27). The redirect below is `>`, so every launch TRUNCATED the previous
  # run's log. On 2026-07-26 that destroyed the forensics of a completed null run: a later launch
  # that was killed ~60s in overwrote rank 0's log with a 10-line stub, and the AF-DOSE /
  # monkeypatch-install evidence needed to explain the null was gone. Worse, the stub was then
  # read as if it were the null run's log and produced two opposite wrong conclusions before the
  # mtimes gave it away. Ranks 1-3 survived only because that launch died before reaching them.
  # Rotate-then-write: the read path ($LOGDIR/r$rank.log) is unchanged for the monitor below and
  # for every caller, and no prior run's evidence is ever destroyed again. Cost is disk, which is
  # the cheapest thing we have (478G free) against a forensic dead end, which cost a whole evening.
  ssh -o ConnectTimeout=8 spark@"$host" \
    "mkdir -p $LOGDIR; \
     if [ -s $LOGDIR/r$rank.log ]; then \
       mv $LOGDIR/r$rank.log $LOGDIR/r$rank.\$(date -r $LOGDIR/r$rank.log +%Y%m%dT%H%M%S).log; \
     fi; \
     cd $RECIPE_DIR && tmux new-session -d -s cpt27b \
       \"$RUN_ENV ./launch_cpt_qwen36_27b_fsdp.sh $rank > $LOGDIR/r$rank.log 2>&1\"" \
    && echo "  launched rank $rank on $host" || echo "  FAILED to launch rank $rank on $host"
}

measure_gemm_tflops() {
  local host=$1
  ssh -o ConnectTimeout=8 -o BatchMode=yes spark@"$host" \
    "PYTHONWARNINGS=ignore python3 -c '
import subprocess
import time
import torch

n = 8192
a = torch.randn(n, n, device=\"cuda\", dtype=torch.bfloat16)
b = torch.randn(n, n, device=\"cuda\", dtype=torch.bfloat16)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
iterations = 20
started = time.perf_counter()
for _ in range(iterations):
    c = a @ b
torch.cuda.synchronize()
seconds = (time.perf_counter() - started) / iterations
query = subprocess.run(
    [
        \"nvidia-smi\",
        \"--query-gpu=clocks.gr,power.draw,temperature.gpu\",
        \"--format=csv,noheader\",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().split(\",\")
metrics = [value.strip().split()[0] for value in query]
print(f\"{2 * n ** 3 / seconds / 1e12:.3f}|{metrics[0]}|{metrics[1]}|{metrics[2]}\")
'"
}

echo "=== 4-node 27B CPT launch $(date -u +%H:%M:%S) ==="
echo "  fix: reduce_dtype=bf16 + keep_low_precision_grads=True (c63bc2d)"

# ── PRE-LAUNCH GATE INTEGRITY (whole-operation review, all 5 seats, 2026-07-28) ──
# Clarity named the gap precisely: "the packet doesn't state whether the fixes are PROACTIVE
# PRE-LAUNCH checks or still reactive." They were reactive — a harness someone had to remember to
# run. This makes them pre-launch, which is the whole point: a blind gate discovered at step 5 has
# already cost ~8 min of setup plus a reboot, and module 5 paid that three times.
#
# The check is cheap (seconds, local, no cluster) and it verifies that each gate can still go RED
# against a known-bad fixture. A gate that cannot fail its own control is not passing — it is not
# looking, and every PASS it emits afterwards is inadmissible.
#
# GATE_PREFLIGHT=0 skips it. That exists for the case where the controls themselves are being
# repaired; it is not a routine escape hatch, and skipping is announced rather than silent.
if [ "${GATE_PREFLIGHT:-1}" = "1" ]; then
  _gnc="$(git rev-parse --show-toplevel 2>/dev/null)/careers-qwen/gate_negative_controls.py"
  if [ -f "$_gnc" ]; then
    echo "  gate integrity: verifying every gate can still go RED..."
    if python3 "$_gnc" >/tmp/gate_preflight.$$ 2>&1; then
      echo "    $(grep -oE '[0-9]+/[0-9]+ gates have demonstrated' /tmp/gate_preflight.$$ | head -1)"
    else
      echo "  ABORT: a gate could not demonstrate it can fail — its PASS is inadmissible." >&2
      grep -E "NOT-LOOKING" -A1 /tmp/gate_preflight.$$ >&2
      echo "  Fix the gate, or set GATE_PREFLIGHT=0 if you are deliberately repairing the controls." >&2
      rm -f /tmp/gate_preflight.$$
      exit 1
    fi
    rm -f /tmp/gate_preflight.$$
  else
    echo "  gate integrity: harness not found at $_gnc — SKIPPED (announced, not silent)"
  fi
else
  echo "  gate integrity: SKIPPED by GATE_PREFLIGHT=0"
fi

# THERMAL PROTECTION (Jesse 2026-07-11: "stop crashing these machines"). The whole-node death is a
# ~94C board/SoC THERMAL shutdown (instrumented 2026-07-10), NOT bad nodes. GB10 exposes no power
# limit (-pl N/A) but supports graphics-clock lock (-lgc); cap the max graphics clock to hold boards
# below the shutdown point. The cap does NOT persist across reboot, so it MUST be applied here on
# every launch (the production launcher previously omitted it -> the crashes). CLOCK_CAP=0 disables.
CLOCK_CAP="${CLOCK_CAP:-2000}"
if [ "$CLOCK_CAP" != "0" ]; then
  echo "  thermal: capping graphics clock <= ${CLOCK_CAP}MHz on all 4 nodes (prevents ~94C hard-crash)"
  for h in "${ALL[@]}"; do
    ssh -o ConnectTimeout=6 spark@"$h" "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP >/dev/null 2>&1 && nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader 2>/dev/null | head -1" \
      && echo "    .${h##*.} capped @${CLOCK_CAP}" || echo "    .${h##*.} CAP FAILED (check sudo/nvidia-smi)"
  done
fi

GEMM_PREFLIGHT_MIN_PEER_RATIO="${GEMM_PREFLIGHT_MIN_PEER_RATIO:-0.80}"
GEMM_PREFLIGHT_ONLY="${GEMM_PREFLIGHT_ONLY:-0}"
if ! awk -v ratio="$GEMM_PREFLIGHT_MIN_PEER_RATIO" \
    'BEGIN { exit !(ratio >= 0.50 && ratio <= 1.00) }'; then
  echo "ERROR: GEMM_PREFLIGHT_MIN_PEER_RATIO must be between 0.50 and 1.00" >&2
  exit 1
fi
[[ "$GEMM_PREFLIGHT_ONLY" =~ ^[01]$ ]] || {
  echo "ERROR: GEMM_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 1
}
echo "  performance: BF16 GEMM peer-homogeneity preflight (minimum ${GEMM_PREFLIGHT_MIN_PEER_RATIO}x median)"
GEMM_TFLOPS=()
for h in "${ALL[@]}"; do
  result=$(measure_gemm_tflops "$h") || {
    echo "ERROR: GEMM preflight failed to execute on $h" >&2
    exit 1
  }
  IFS='|' read -r tflops clock power temp <<< "$result"
  [[ "$tflops" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "ERROR: invalid GEMM preflight result from $h: $result" >&2
    exit 1
  }
  GEMM_TFLOPS+=("$tflops")
  echo "    .${h##*.}: ${tflops} TFLOPS clock=${clock}MHz power=${power}W temp=${temp}C"
done
readarray -t SORTED_TFLOPS < <(printf '%s\n' "${GEMM_TFLOPS[@]}" | sort -n)
GEMM_MEDIAN=$(awk -v lower="${SORTED_TFLOPS[1]}" -v upper="${SORTED_TFLOPS[2]}" \
  'BEGIN { printf "%.3f", (lower + upper) / 2 }')
GEMM_FAILURES=0
for i in 0 1 2 3; do
  if ! awk -v observed="${GEMM_TFLOPS[$i]}" -v median="$GEMM_MEDIAN" \
      -v ratio="$GEMM_PREFLIGHT_MIN_PEER_RATIO" \
      'BEGIN { exit !(observed >= median * ratio) }'; then
    echo "ERROR: ${ALL[$i]} GEMM ${GEMM_TFLOPS[$i]} TFLOPS is below ${GEMM_PREFLIGHT_MIN_PEER_RATIO}x fleet median ${GEMM_MEDIAN}" >&2
    GEMM_FAILURES=$((GEMM_FAILURES + 1))
  fi
done
if [ "$GEMM_FAILURES" -ne 0 ]; then
  echo "ERROR: refusing distributed launch with a rank-performance straggler" >&2
  exit 1
fi
echo "    fleet median=${GEMM_MEDIAN} TFLOPS — all ranks within peer threshold"
if [ "$GEMM_PREFLIGHT_ONLY" = "1" ]; then
  echo "GEMM PREFLIGHT COMPLETE"
  exit 0
fi

echo "  master $MASTER = rank 0 (binding :29500 first)"
launch_rank "$MASTER" 0
echo "  (master settling 12s...)"
sleep 12
for i in 0 1 2; do launch_rank "${WORKERS[$i]}" $((i+1)); done

echo ""
echo "=== monitor: peakUsed/node + trainer liveness + master optimizer-step progression ==="
STEP_SEEN=0
for t in $(seq 1 80); do   # ~40 min @ 30s
  line=""; alive=0
  for h in "${ALL[@]}"; do
    r=$(ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$h" \
        'u=$(free -g|awk "/Mem:/{print \$3}"); p=$(ps -eo comm=,args= | awk "/^python3[[:space:]].*[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}"); printf "%s %s" "$u" "$p"' \
        2>/dev/null)
    used=""; trainers=""; extra=""
    read -r used trainers extra <<<"$r"
    if [[ "$used" =~ ^[0-9]+$ && "$trainers" =~ ^[0-9]+$ && -z "$extra" ]]; then
      [ "$trainers" -gt 0 ] && alive=$((alive+1))
      line="$line ${h##*.}=${used}G/${trainers}t"
    else
      line="$line ${h##*.}=unreachable"
    fi
  done
  # master's latest step / status line. "FIRST STEP:" prints ONLY after the first
  # optimizer step COMPLETES (trainer:1410) — the fp32 run never reached it (OOM-died
  # at the step). That line + its grads= breakdown is the definitive fit signal.
  st=$(ssh -o ConnectTimeout=5 spark@"$MASTER" \
       'grep -aE "FIRST STEP:|params=.*grads=|\[step [0-9]|Starting: steps|OOM|out of memory|Traceback|Error" '"$LOGDIR"'/r0.log 2>/dev/null | tail -1' 2>/dev/null)
  step_num=$(ssh -o ConnectTimeout=5 spark@"$MASTER" \
       'grep -aoE "\[step [0-9]+\]" '"$LOGDIR"'/r0.log 2>/dev/null | tail -1 | tr -cd "0-9"' \
       2>/dev/null)
  echo "[$t] alive=$alive/4 used:$line | ${st:0:100}"
  case "$st" in
    *OOM*|*"out of memory"*|*Traceback*|*ERROR:*)
       echo ">>> FAILURE signal in master log — stopping monitor"; break;;
  esac
  if [[ "$step_num" =~ ^[0-9]+$ ]]; then
    STEP_SEEN=1
    echo ">>> OPTIMIZER STEP $step_num COMPLETED — 27B IS TRAINING (fit confirmed). Full grads= breakdown:"
    ssh -o ConnectTimeout=5 spark@"$MASTER" 'grep -aE "FIRST STEP:|params=.*grads=.*optim=" '"$LOGDIR"'/r0.log | tail -2' 2>/dev/null
    break
  fi
  sleep 30
done
echo "=== monitor end $(date -u +%H:%M:%S) step_seen=$STEP_SEEN ==="
[ "$STEP_SEEN" -eq 1 ] || {
  echo "ERROR: no completed optimizer step was observed; launch is not admissible." >&2
  exit 1
}
