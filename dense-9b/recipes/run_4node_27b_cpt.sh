#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Mira-side 4-node orchestrator for Qwen3.6-27B full-parameter CPT.
# Launches the per-node recipe (launch_cpt_qwen36_27b_fsdp.sh) under a systemd
# service on each Spark, master rank 0 FIRST so it binds :29500 before workers dial in.
#
# PRE-CONDITION (caller's responsibility, per reboot-after-every-run): all 4 nodes
# freshly REBOOTED + pristine (avail~108G, /dev/shm/nccl*=0, no torch procs). This
# script does NOT reboot — it assumes clean nodes and refuses to kill-and-relaunch.
set -uo pipefail

: "${SPARK_HOME:?fleet.env must define SPARK_HOME}"
: "${SPARK_MASTER:?fleet.env must define SPARK_MASTER}"
: "${SPARK_MGMT_IPS:?fleet.env must define SPARK_MGMT_IPS}"
: "${SPARK_RAIL_MASTER:?fleet.env must define SPARK_RAIL_MASTER}"
: "${SPARK_USER:?fleet.env must define SPARK_USER}"

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
: "${BATCH_SIZE_PER_RANK:=1}"; : "${CPT_PACKED:=0}"; : "${CPT_BUCKETING:=1}"
RUN_ENV="CPT_DATA=$CPT_DATA MAX_SEQ=$MAX_SEQ BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK CPT_PACKED=$CPT_PACKED CPT_BUCKETING=$CPT_BUCKETING CPT_SHORT_BATCH=$CPT_SHORT_BATCH CPT_MID_BATCH=$CPT_MID_BATCH CPT_LONG_BATCH=$CPT_LONG_BATCH TOKEN_BUDGET_PER_STEP=$TOKEN_BUDGET_PER_STEP TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT SAVE_EVERY=$SAVE_EVERY CHECKPOINT_DCP=$CHECKPOINT_DCP"
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
# EXACT_SFT_EPOCH's OWN PREREQUISITES. The trainer REFUSES to start under EXACT_SFT_EPOCH=1
# without EXPECTED_REAL_SAMPLES>0 and REQUIRE_LORA_INIT_PARITY=1 — and RUN_ENV is an ALLOWLIST, so
# both were being dropped silently and the mode could never actually run through this driver. The
# failure would read as "EXACT_SFT_EPOCH requires EXPECTED_REAL_SAMPLES > 0" on the node while the
# operator could see it set in their own shell: a gate stranded from its own precondition, which is
# exactly the HORIZON_PARTIAL defect recorded a few lines below, in a second mode nobody had run
# through this driver yet. Found by auditing the allowlist BEFORE launching rather than by watching
# a run die.
[ -n "${EXPECTED_REAL_SAMPLES:-}" ] && RUN_ENV="$RUN_ENV EXPECTED_REAL_SAMPLES=$EXPECTED_REAL_SAMPLES"
[ -n "${REQUIRE_LORA_INIT_PARITY:-}" ] && RUN_ENV="$RUN_ENV REQUIRE_LORA_INIT_PARITY=$REQUIRE_LORA_INIT_PARITY"
# LR_LORA: the adapter learning rate. Distinct from LR (which the CPT path uses); dropping it means
# a LoRA run silently trains at the trainer's default instead of the rate the operator chose — the
# 2026-07-13 LR non-forwarding bug, repeated for the adapter path.
[ -n "${LR_LORA:-}" ] && RUN_ENV="$RUN_ENV LR_LORA=$LR_LORA"

# HORIZON_PARTIAL: declares a deliberately short packed-CPT run (throughput probe, smoke test)
# to the trainer's CPT horizon contract, which otherwise requires TOTAL_STEPS to cover the whole
# corpus. MUST be named here: RUN_ENV is an ALLOWLIST, and dropped silently it would not weaken
# the gate but strand it — the probe would be refused on the node with a declaration the operator
# believes they made, which reads as a broken gate and invites bypassing it. The trainer requires
# this to EQUAL TOTAL_STEPS, so a stale value fails closed rather than open.
[ -n "${HORIZON_PARTIAL:-}" ] && RUN_ENV="$RUN_ENV HORIZON_PARTIAL=$HORIZON_PARTIAL"
[ -n "${MEMORY_GATE_LOG_EVERY_STEP:-}" ] && RUN_ENV="$RUN_ENV MEMORY_GATE_LOG_EVERY_STEP=$MEMORY_GATE_LOG_EVERY_STEP"
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
# EXPORT_DCP is the coordinated sharded export path for full-parameter CPT.
# It is intentionally distinct from adapter-only export.
[ -n "${EXPORT_DCP:-}" ] && RUN_ENV="$RUN_ENV EXPORT_DCP=$EXPORT_DCP"
# Forward NCCL debug capture to the per-node sessions (the orchestrator only passes an allowlist, so
# NCCL_DEBUG set on the Mira shell would NOT reach the remote tmux without this). Enables the fabric
# bus-bandwidth/topology capture (NCCL_DEBUG=INFO NCCL_DEBUG_FILE=...).
[ -n "${NCCL_DEBUG:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG=$NCCL_DEBUG"
[ -n "${NCCL_DEBUG_FILE:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG_FILE=$NCCL_DEBUG_FILE"
[ -n "${NCCL_DEBUG_SUBSYS:-}" ] && RUN_ENV="$RUN_ENV NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
# NCCL/torch TUNING — not debug. Only the four DEBUG vars above were forwarded, so the GB10/Blackwell
# settings that actually change fabric behaviour were unreachable through this launcher: values
# exported by the controller were silently dropped before reaching the remote process, so the run
# proceeded on defaults while the operator believed the explicit manifest was active.
#
# WHY THE ALLOWLIST GATE DID NOT CATCH THIS, recorded so the gate is not over-trusted:
# allowlist_completeness_gate.py diffs the trainer's own os.environ.get calls against what the
# launcher forwards. NCCL_* and TORCH_NCCL_* are consumed by the NCCL C library and by torch's
# distributed layer, NEVER read by the trainer through os.environ — so they are invisible to that
# scan by construction. The gate covers trainer-read config only; library-read config needs this
# explicit list.
: "${NCCL_IB_HCA:?ERROR: NCCL_IB_HCA must state the canonical HCA and port selection; it is not defaulted}"
: "${NCCL_NET_GDR_LEVEL:?ERROR: NCCL_NET_GDR_LEVEL must state the canonical GDR selection; it is not defaulted}"
: "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:?ERROR: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must state the canonical heartbeat; it is not defaulted}"
RUN_ENV="$RUN_ENV NCCL_IB_HCA=$NCCL_IB_HCA NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
[ -n "${NCCL_P2P_DISABLE:-}" ] && RUN_ENV="$RUN_ENV NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
[ -n "${NCCL_SHM_DISABLE:-}" ] && RUN_ENV="$RUN_ENV NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
[ -n "${NCCL_IB_DISABLE:-}" ] && RUN_ENV="$RUN_ENV NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && RUN_ENV="$RUN_ENV NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
[ -n "${NCCL_ALGO:-}" ] && RUN_ENV="$RUN_ENV NCCL_ALGO=$NCCL_ALGO"
[ -n "${NCCL_PROTO:-}" ] && RUN_ENV="$RUN_ENV NCCL_PROTO=$NCCL_PROTO"
[ -n "${NCCL_TIMEOUT:-}" ] && RUN_ENV="$RUN_ENV NCCL_TIMEOUT=$NCCL_TIMEOUT"
[ -n "${TORCH_NCCL_DUMP_ON_TIMEOUT:-}" ] && RUN_ENV="$RUN_ENV TORCH_NCCL_DUMP_ON_TIMEOUT=$TORCH_NCCL_DUMP_ON_TIMEOUT"
[ -n "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}" ] && RUN_ENV="$RUN_ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
[ -n "${PG_TIMEOUT_SEC:-}" ] && RUN_ENV="$RUN_ENV PG_TIMEOUT_SEC=$PG_TIMEOUT_SEC"
[ -n "${FP8:-}" ] && RUN_ENV="$RUN_ENV FP8=$FP8"
RECIPE_DIR=${SPARK_HOME}/palios-training/dense-9b/recipes
LOGDIR=${SPARK_HOME}/cpt27b_logs
SYSTEMD_STARTER=${RECIPE_DIR}/systemd/start_cpt_rank_service.sh
SYSTEMD_VERIFIER=${RECIPE_DIR}/systemd/verify_cpt_rank_process_env.sh
ATTEMPTED_HOSTS=()
ATTEMPTED_RANKS=()

launch_rank () {  # $1=host  $2=rank
  local host=$1 rank=$2
  local entry remote_command
  local -a run_env_entries=()

  read -r -a run_env_entries <<< "$RUN_ENV"
  for entry in "${run_env_entries[@]}"; do
    if [[ ! "$entry" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]]; then
      echo "ERROR: RUN_ENV cannot be serialized safely for rank $rank: $entry" >&2
      return 1
    fi
  done

  printf -v remote_command \
    'SPARK_HOME=%q PALIOS_TRAINING_USER=%q /usr/bin/bash %q %q' \
    "$SPARK_HOME" "$SPARK_USER" "$SYSTEMD_STARTER" "$rank"

  # Record the attempt before SSH. A lost start receipt is ambiguous: the unit may
  # already be running remotely even though this client observes a transport failure.
  ATTEMPTED_HOSTS+=("$host")
  ATTEMPTED_RANKS+=("$rank")

  if printf '%s\n' "${run_env_entries[@]}" | \
      ssh -o ConnectTimeout=8 -o BatchMode=yes "${SPARK_USER}@${host}" "$remote_command"; then
    echo "  systemd owns rank $rank on $host"
  else
    echo "  FAILED to start supervised rank $rank on $host" >&2
    return 1
  fi
}

rollback_attempted_ranks() {
  local idx host rank unit remote_command
  local rollback_status=0

  echo "  rolling back ${#ATTEMPTED_HOSTS[@]} attempted rank service(s)" >&2
  for ((idx=${#ATTEMPTED_HOSTS[@]} - 1; idx >= 0; idx--)); do
    host=${ATTEMPTED_HOSTS[$idx]}
    rank=${ATTEMPTED_RANKS[$idx]}
    unit="palios-cpt-rank@${rank}.service"
    printf -v remote_command \
      'unit=%q; load_state=$(sudo systemctl show --property LoadState --value "$unit") || exit 1; if [ "$load_state" = not-found ]; then exit 0; fi; sudo systemctl stop "$unit"; active_state=$(sudo systemctl show --property ActiveState --value "$unit") || exit 1; [ "$active_state" = inactive ] || [ "$active_state" = failed ]' \
      "$unit"

    if ssh -o ConnectTimeout=8 -o BatchMode=yes "${SPARK_USER}@${host}" "$remote_command"; then
      echo "  rollback confirmed rank $rank on $host" >&2
    else
      echo "  ROLLBACK FAILED rank $rank on $host" >&2
      rollback_status=1
    fi
  done

  return "$rollback_status"
}

launch_rank_or_rollback() {
  local host=$1 rank=$2 launch_status

  launch_rank "$host" "$rank"
  launch_status=$?
  if [ "$launch_status" -eq 0 ]; then
    return 0
  fi

  echo "ERROR: rank $rank failed to start on $host; stopping the whole attempted job" >&2
  if ! rollback_attempted_ranks; then
    echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
  fi
  return "$launch_status"
}

verify_all_rank_environments() {
  local rank host remote_command
  local verification_failures=0

  for rank in 0 1 2 3; do
    host=${ALL[$rank]}
    printf -v remote_command '/usr/bin/bash %q %q' "$SYSTEMD_VERIFIER" "$rank"
    if ssh -o ConnectTimeout=8 -o BatchMode=yes "${SPARK_USER}@${host}" "$remote_command"; then
      echo "  live environment confirmed rank $rank on $host"
    else
      echo "  LIVE ENVIRONMENT FAILED rank $rank on $host" >&2
      verification_failures=$((verification_failures + 1))
    fi
  done

  [ "$verification_failures" -eq 0 ]
}

verify_all_rank_environments_or_rollback() {
  if verify_all_rank_environments; then
    return 0
  fi

  echo "ERROR: live environment verification failed; stopping the whole attempted job" >&2
  if ! rollback_attempted_ranks; then
    echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
  fi
  return 1
}

measure_gemm_tflops() {
  local host=$1
  ssh -o ConnectTimeout=8 -o BatchMode=yes "${SPARK_USER}@${host}" \
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
    ssh -o ConnectTimeout=6 "${SPARK_USER}@${h}" "sudo nvidia-smi -pm 1 >/dev/null 2>&1; sudo nvidia-smi -lgc 0,$CLOCK_CAP >/dev/null 2>&1 && nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader 2>/dev/null | head -1" \
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
launch_rank_or_rollback "$MASTER" 0 || exit 1
echo "  (master settling 12s...)"
sleep 12
for i in 0 1 2; do launch_rank_or_rollback "${WORKERS[$i]}" $((i+1)) || exit 1; done

echo ""
echo "=== monitor: peakUsed/node + trainer liveness + master optimizer-step progression ==="
STEP_SEEN=0
ENV_VERIFIED=0
for t in $(seq 1 80); do   # ~40 min @ 30s
  line=""; alive=0; failed_units=0
  for rank in 0 1 2 3; do
    h=${ALL[$rank]}
    r=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${SPARK_USER}@${h}" \
        'u=$(free -g|awk "/Mem:/{print \$3}"); p=$(ps -eo comm=,args= | awk "/^python3[[:space:]].*[t]rain_fsdp_dense_9b.py/{n++} END{print n+0}"); s=$(systemctl show --property ActiveState --value palios-cpt-rank@'"$rank"'.service); printf "%s %s %s" "$u" "$p" "$s"' \
        2>/dev/null)
    used=""; trainers=""; unit_state=""; extra=""
    read -r used trainers unit_state extra <<<"$r"
    if [[ "$used" =~ ^[0-9]+$ && "$trainers" =~ ^[0-9]+$ && "$unit_state" =~ ^[a-z-]+$ && -z "$extra" ]]; then
      [ "$trainers" -gt 0 ] && alive=$((alive+1))
      [[ "$unit_state" =~ ^(active|activating)$ ]] || failed_units=$((failed_units+1))
      line="$line ${h##*.}=${used}G/${trainers}t/${unit_state}"
    else
      line="$line ${h##*.}=unreachable"
    fi
  done
  if [ "$failed_units" -ne 0 ]; then
    echo ">>> $failed_units systemd rank unit(s) failed — stopping ALL RANKS" >&2
    if ! rollback_attempted_ranks; then
      echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
    fi
    exit 1
  fi
  if [ "$alive" -eq 4 ] && [ "$ENV_VERIFIED" -eq 0 ]; then
    verify_all_rank_environments_or_rollback || exit 1
    ENV_VERIFIED=1
    echo ">>> LIVE ENVIRONMENT VERIFIED ON ALL 4 RANKS"
  fi
  # master's latest step / status line. "FIRST STEP:" prints ONLY after the first
  # optimizer step COMPLETES (trainer:1410) — the fp32 run never reached it (OOM-died
  # at the step). That line + its grads= breakdown is the definitive fit signal.
  st=$(ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" \
       'grep -aE "FIRST STEP:|params=.*grads=|\[step [0-9]|Starting: steps|OOM|out of memory|Traceback|Error" '"$LOGDIR"'/r0.log 2>/dev/null | tail -1' 2>/dev/null)
  step_num=$(ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" \
       'grep -aoE "\[step [0-9]+\]" '"$LOGDIR"'/r0.log 2>/dev/null | tail -1 | tr -cd "0-9"' \
       2>/dev/null)
  echo "[$t] alive=$alive/4 used:$line | ${st:0:100}"
  case "$st" in
    *OOM*|*"out of memory"*|*Traceback*|*ERROR:*)
      echo ">>> FAILURE signal in master log — stopping ALL RANKS" >&2
      if ! rollback_attempted_ranks; then
        echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
      fi
      exit 1
      ;;
  esac
  if [[ "$step_num" =~ ^[0-9]+$ ]]; then
    if [ "$ENV_VERIFIED" -ne 1 ]; then
      echo "ERROR: optimizer step appeared before the all-rank live environment gate" >&2
      if ! rollback_attempted_ranks; then
        echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
      fi
      exit 1
    fi
    STEP_SEEN=1
    echo ">>> OPTIMIZER STEP $step_num COMPLETED — 27B IS TRAINING (fit confirmed). Full grads= breakdown:"
    ssh -o ConnectTimeout=5 "${SPARK_USER}@${MASTER}" 'grep -aE "FIRST STEP:|params=.*grads=.*optim=" '"$LOGDIR"'/r0.log | tail -2' 2>/dev/null
    break
  fi
  sleep 30
done
echo "=== monitor end $(date -u +%H:%M:%S) step_seen=$STEP_SEEN ==="
if [ "$STEP_SEEN" -ne 1 ]; then
  echo "ERROR: no completed optimizer step was observed; launch is not admissible." >&2
  if ! rollback_attempted_ranks; then
    echo "FATAL: whole-job rollback was incomplete; manual intervention is required" >&2
  fi
  exit 1
fi
