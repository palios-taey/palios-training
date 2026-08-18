---
type: consultation
to: gaia, logos, horizon, cosmos, clarity
from: tutor
date: 2026-08-18
available_context_inventory:
  - INCLUDED: docs/postmortem/PART1_measured_timeline.md — the measured run, gate and CI record
  - INCLUDED: docs/postmortem/PART2_config_surface.md — the full non-public configuration surface
  - INCLUDED: PRODUCTION_MANIFEST.yml — machine-readable statement of what is production
  - INCLUDED: README.md, careers-qwen/RUNBOOK_CPT_SFT_BAKE.md — the intended process
  - INCLUDED: dense-9b/trainers/, dense-9b/recipes/, careers-qwen/ — the executing code
  - INCLUDED: docs/SPARK_TOPOLOGY.md, dense-9b/receipts/ — measured hardware and run receipts
  - INCLUDED: careers-qwen/receipts/cpt_qwen38_v3_corpus.manifest.json — corpus receipt for this run
  - FLAGGED: docs/postmortem/POSTMORTEM_cpt_qwen38_v3.md — our own causal analysis; see section 6
  - INCLUDED: fleet.env — reproduced COMPLETE in Appendix A of this packet, values verbatim except
    host addresses (stable labels) and operator home prefixes
  - INCLUDED: the full read surface of all three production entrypoints — Appendix B, every variable
    with its unset behaviour
  - EXCLUDED: the training corpus itself — training data is never published; its receipt (shas,
    counts, input registry) is included instead
  - EXCLUDED: the model weights — 52 GB artifact; its tensor-level gate results are in PART1
  - EXCLUDED: raw run logs — not tracked in the repository; the values read from them are quoted
    in PART1 and labelled Observed
---

# Consult packet — CPT training and bake, PALIOS-TAEY

## Ground truth

**Source of truth:** https://github.com/palios-taey/palios-training
**Branch: `tutor/requalify-manifest-shas`** — not `main`.

[Constraint] **Do not take this packet, its framing, or its file list as ground truth.** Read the
repository and verify against it yourself. **Enumerate the relevant surface yourself** — anything
listed here is a starting point and may be incomplete or biased. Reach your own conclusions.

[Constraint] If you cannot fetch the repository in this session, say so and decline to rule, rather
than ruling on our description. [Constraint] Findings that contradict us are the purpose of this
exercise, not a problem with it.

[Observed] `main` does not currently contain the trainer path that produced the model now in
production. [Observed] That branch difference is itself part of the record under examination.

### The measured record, in the repository

[Observed] Two documents are fact-only and carry Observed / Inferred / Unknown on every claim with
`file:line` and commit citations:

- `docs/postmortem/PART1_measured_timeline.md` — what ran, when, what it emitted, what each gate
  said; session boundaries, artifact receipts, gate results, and the public CI record.
- `docs/postmortem/PART2_config_surface.md` — every non-public input the training and bake paths
  read, enumerated from call sites, with unset-behaviour per variable, and a determined/undetermined
  table for what the public repository alone decides.

[Observed] `PRODUCTION_MANIFEST.yml` is the machine-readable statement of what is production, gated
on content shas. [Observed] The three legal surfaces are corpus pack, CPT, and bake/export.
[Unknown] Whether the documents and the code agree. That is a question for you, not an assertion
from us.

### One document to read last, or not at all

[Observed] `docs/postmortem/POSTMORTEM_cpt_qwen38_v3.md` is our own causal analysis, written by the
seat that ran the work, and it names causes and proposes remedies. [Constraint] We flag it rather
than hide it: if you read it first you will likely return our own conclusions to us, which is the
failure mode this packet is shaped to avoid. Form your view from the measured record first.

### Already known to be unresolved

Stated so you do not spend effort rediscovering it, with no proposed cause attached.

[Observed] The run's final in-training SR-DELTA verdict was `FAIL-LOW` at 0.49× ULP against a 0.5u
floor, measured near step 190 at LR 1.42e-06 versus about 9.7e-06 at peak.
[Observed] Sessions 1 and 2 of the same run read 1.16u and 1.03u.
[Observed] The cumulative post-export weight-diff measured 2.223e-04 against the band
`5e-05 .. 8e-04`. [Unknown] How those two readings relate.
[Observed] The CPT launcher has zero environment variables that abort when unset; the bake pipeline
has ten.

### Hardware, measured

[Observed] 4× DGX Spark GB10 (Blackwell `sm_121`), 119 GB unified memory per node.
[Observed] Dual-rail RoCEv2 measured 666 MB/s between nodes; the management LAN measured ~112 MB/s.
[Observed] Two Thor serving hosts. [Observed] The model is 27B dense, trained full-parameter with
FSDP2 and a DTensor-patched Adafactor.
[Observed] The most recent cycle — a 218-step CPT, its bake, and its deployment — took two days.
[Observed] We have completed this cycle before on this hardware.

## Appendix A — `fleet.env`, complete

[Observed] The complete gitignored `fleet.env` that drives production on this machine.
[Constraint] Every name and value is verbatim EXCEPT host addresses and operator home
prefixes. Each distinct address gets a STABLE label, so relationships between variables
survive: the same label always means the same host, `<MGMT_*>` is the management LAN and
`<RAIL_*>` is the 666 MB/s Spark fabric. Nothing else is withheld.

```bash
SPARK_MGMT_IPS="<MGMT_A> <MGMT_B> <MGMT_C> <MGMT_D>"
SPARK_MASTER="<MGMT_A>"
SPARK_USER="spark"
SPARK_RAIL_MASTER="<RAIL_A>"
SPARK_RAIL_IPS="<RAIL_A> <RAIL_B> <RAIL_C> <RAIL_D>"
# ORDERING IS LOAD-BEARING (fixed 2026-07-30): the THOR*_ENDPOINT and POST_CPT_CONVERT_SSH
# lines below EXPAND these, so they must be defined FIRST. Previously they sat after, which
# made this file unsourceable under `set -u` — it died on its own line 6 with
# 'THOR1_HOST: unbound variable', naming the topology file instead of the caller. Every
# 'source fleet.env BEFORE set -u' workaround in the repo existed to route around that.
# Keep host definitions above any line that expands them.
THOR1_HOST="<MGMT_E>"
THOR2_HOST="<MGMT_D>7"

THOR1_ENDPOINT="${THOR1_HOST}:8000"
THOR2_ENDPOINT="${THOR2_HOST}:8000"
SPARK_HOME="$NODE_HOME"
SPARK_MODELS="$NODE_HOME/models"
SPARK_NODE1="<MGMT_B>"
SPARK_NODE2="<MGMT_C>"
SPARK_NODE3="<MGMT_D>"
SPARK_RAIL_IP="<RAIL_B>"
ORCHESTRATOR_IP="<MGMT_G>"
SPARK_SUBNET="10.0.0"

# Post-CPT production lifecycle: durable copies on Mira, conversion in the pinned Thor1 image.
POST_CPT_ARTIFACT_STORE="/media/mira/Expansion/training-artifacts"
POST_CPT_CONVERT_SSH="jetson@${THOR1_HOST}"
POST_CPT_CONVERT_ROOT="$NODE_HOME/cpt-artifacts"
POST_CPT_GRAFT_BASE="$NODE_HOME/serve-models/module5_merged"
POST_CPT_CONVERT_IMAGE="taey-convert@sha256:d571caf7bdafde39a3fcdca1c322b03045fa6ddd977a196d188f87d94602c669"
POST_CPT_SANCTION="treasurer task-dfa3fd75 2026-07-28"

# ── GPU CLOCK POLICY — ONE DEFINITION (added 2026-07-29) ────────────────────────────────
# nvidia-smi -lgc IS PERSISTENT. It survives the job that set it and applies to whatever runs
# next. Before this line there were SIX values across the repo (1000/1600/2000/2200/none), so
# the clock a run got was decided by whichever job happened to run before it. launch_stage2_sft.sh
# set NO cap at all, which means an SFT launched after a bake ran its entire length at 1000MHz —
# 33% of this hardware's 3003MHz ceiling — and nothing in any log would say so.
# MEASURED 2026-07-29: clocks.max.gr = 3003MHz. Observed locked at 897-955MHz after a bake.
SPARK_CLOCK_MAX_MHZ=3003          # hardware ceiling, measured via nvidia-smi --query-gpu=clocks.max.gr
SPARK_CLOCK_CAP=2000              # PROVEN SAFE: 2h07m sustained 27B CPT (01:49:49->03:57:35Z,
                                  # 148 steps, 793 tok/s) with zero thermal events.
# RAISE CONDITION, now MET and not yet taken: dense-9b/instrumentation/capture_run.sh:42 says
# "raise later to reclaim throughput once a stable 2h run is proven" — 2398MHz is where it hit
# 94C, 2000 was the retreat. The stable 2h run now exists, so the next step is an INSTRUMENTED
# raise toward ~2200 with loaded temps captured, not a guess. SUBSTRATE_PHYSICS.md:24 records
# 85-99 TFLOPS at 1976-2463MHz, so the headroom between 2000 and 2398 is real throughput.

```

[Observed] 25 variables are declared in `fleet.env`; 7 distinct management
addresses and 4 distinct rail addresses appear across them.

## Appendix B — every environment variable each production surface READS

[Observed] Enumerated from the call sites, not from any declaration list. `abort` means the
script exits with a stated message when the variable is unset. `default=''` means it silently
becomes the empty string.

### CPT — dense-9b/recipes/run_4node_27b_cpt.sh

[Observed] reads 73 distinct variables; 0 abort when unset.

| variable | unset behaviour |
|---|---|
| `AC_LAYER_CLS` | default=`''` (empty) |
| `AC_LAYER_GRANULAR` | default=`''` (empty) |
| `ADAFACTOR_ALPHA_MODE` | default=`''` (empty) |
| `ADAFACTOR_DOSE_LOG` | default=`''` (empty) |
| `ADAFACTOR_EPS1` | read, no default declared |
| `ALL` | read, no default declared |
| `BAKE_TO_HF` | default=`''` (empty) |
| `BATCH_SIZE_PER_RANK` | read, no default declared |
| `CHECKPOINT_DCP` | read, no default declared |
| `CLOCK_CAP` | default=`2000` |
| `CPT_DATA` | default=`/var/spark/isma/training/cpt_raw_corpus_train_no_superseded.` |
| `CPT_LONG_BATCH` | read, no default declared |
| `CPT_MID_BATCH` | read, no default declared |
| `CPT_PACKED` | read, no default declared |
| `CPT_SHORT_BATCH` | read, no default declared |
| `DISABLE_FLA` | default=`''` (empty) |
| `EPOCHS` | default=`''` (empty) |
| `EXACT_SFT_EPOCH` | default=`''` (empty) |
| `EXPECTED_REAL_SAMPLES` | default=`''` (empty) |
| `EXPECTED_SFT_SAMPLES` | default=`''` (empty) |
| `FP32_MASTER` | default=`''` (empty) |
| `FP8` | default=`''` (empty) |
| `GATE_PREFLIGHT` | default=`1` |
| `GEMM_FAILURES` | read, no default declared |
| `GEMM_MEDIAN` | read, no default declared |
| `GEMM_PREFLIGHT_MIN_PEER_RATIO` | default=`0.80` |
| `GEMM_PREFLIGHT_ONLY` | default=`0` |
| `GEMM_TFLOPS` | read, no default declared |
| `HORIZON_PARTIAL` | default=`''` (empty) |
| `LANE_WEIGHTS` | default=`''` (empty) |
| `LIGER` | default=`''` (empty) |
| `LOGDIR` | read, no default declared |
| `LORA_ALPHA` | default=`''` (empty) |
| `LORA_DROPOUT` | default=`''` (empty) |
| `LORA_MODE` | default=`''` (empty) |
| `LORA_R` | default=`''` (empty) |
| `LORA_TARGET_MODULES` | default=`''` (empty) |
| `LR_LORA` | default=`''` (empty) |
| `MASTER` | read, no default declared |
| `MAX_SEQ` | read, no default declared |
| `MODEL_PATH` | default=`''` (empty) |
| `NCCL_DEBUG` | default=`''` (empty) |
| `NCCL_DEBUG_FILE` | default=`''` (empty) |
| `NCCL_DEBUG_SUBSYS` | default=`''` (empty) |
| `NSYS_OUT_DIR` | default=`''` (empty) |
| `NSYS_PROFILE_ALL_RANKS` | default=`''` (empty) |
| `NSYS_PROFILE_STEP` | default=`''` (empty) |
| `OUTPUT_DIR` | default=`''` (empty) |
| `QUARANTINE_DIGESTS` | default=`''` (empty) |
| `RECIPE_DIR` | read, no default declared |
| `REQUIRE_LORA_INIT_PARITY` | default=`''` (empty) |
| `RESUME_DELTA` | default=`''` (empty) |
| `RESUME_MODEL_ONLY` | default=`''` (empty) |
| `RUN_ENV` | read, no default declared |
| `SAVE_EVERY` | read, no default declared |
| `SESSION_LIMIT` | read, no default declared |
| `SFT_DIR` | default=`''` (empty) |
| `SFT_JSONL` | default=`''` (empty) |
| `SORTED_TFLOPS` | read, no default declared |
| `SPARK_HOME` | read, no default declared |
| `SPARK_MASTER` | read, no default declared |
| `SPARK_MGMT_IPS` | read, no default declared |
| `SPARK_RAIL_MASTER` | read, no default declared |
| `STEP_SEEN` | read, no default declared |
| `TINY_LANE_CAP` | default=`''` (empty) |
| `TINY_LANE_THRESHOLD` | default=`''` (empty) |
| `TOKEN_BUDGET_PER_STEP` | read, no default declared |
| `TORCH_COMPILE` | default=`''` (empty) |
| `TORCH_COMPILE_MODE` | default=`''` (empty) |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | default=`''` (empty) |
| `TOTAL_STEPS` | read, no default declared |
| `WARMUP_STEPS` | default=`''` (empty) |
| `WORKERS` | read, no default declared |

### BAKE — careers-qwen/post_cpt_pipeline.sh

[Observed] reads 58 distinct variables; 10 abort when unset.

| variable | unset behaviour |
|---|---|
| `ARTIFACT_STORE` | **abort** |
| `BOOT_IDS` | read, no default declared |
| `CAPTURED_CORPUS_INPUTS` | read, no default declared |
| `CKPT` | **abort** |
| `CKPT_NAME` | read, no default declared |
| `CONVERT_BASE` | read, no default declared |
| `CONVERT_CORPUS` | read, no default declared |
| `CONVERT_CORPUS_MANIFEST` | read, no default declared |
| `CONVERT_GRAFT_BASE` | **abort** |
| `CONVERT_IMAGE` | **abort** |
| `CONVERT_ROOT` | **abort** |
| `CONVERT_SSH` | **abort** |
| `CONVERT_TOOLS` | default=`${POST_CPT_CONVERT_TOOLS:-${CONVERT_ROOT%/` |
| `CORPUS` | read, no default declared |
| `CORPUS_INPUTS` | read, no default declared |
| `CORPUS_MANIFEST` | read, no default declared |
| `CORPUS_MANIFEST_SHA` | read, no default declared |
| `CORPUS_RECEIPT` | read, no default declared |
| `CORPUS_SHA` | read, no default declared |
| `DCP_DIR` | **abort** |
| `EXPORT_DIR` | read, no default declared |
| `EXPORT_RUNTIME` | read, no default declared |
| `EXPORT_TOOL` | read, no default declared |
| `FLEET_ENV` | default=`"$REPO_ROOT/fleet.env"` |
| `FRESH_UPTIME_MAX` | default=`180` |
| `GRAFT_BASE` | read, no default declared |
| `HF_OUT` | read, no default declared |
| `HF_STAGE` | read, no default declared |
| `LOCAL_ARTIFACT` | read, no default declared |
| `LOCAL_BASE` | read, no default declared |
| `LOGSTEPS` | read, no default declared |
| `MODEL_SYNC_TOOL` | read, no default declared |
| `NODES` | read, no default declared |
| `POST_CPT_ARTIFACT_STORE` | read, no default declared |
| `POST_CPT_CONVERT_IMAGE` | read, no default declared |
| `POST_CPT_CONVERT_ROOT` | read, no default declared |
| `POST_CPT_CONVERT_SSH` | read, no default declared |
| `POST_CPT_CONVERT_TOOLS` | read, no default declared |
| `POST_CPT_GRAFT_BASE` | read, no default declared |
| `POST_CPT_SANCTION` | read, no default declared |
| `PRODUCTION_FILES` | read, no default declared |
| `PROV_TOTAL_STEPS` | read, no default declared |
| `PROV_WARMUP_STEPS` | read, no default declared |
| `REMOTE_ARTIFACT` | read, no default declared |
| `REMOTE_PY_TOOLS` | read, no default declared |
| `REPO_ROOT` | read, no default declared |
| `RUN_TAG` | read, no default declared |
| `SANCTION` | default=`${POST_CPT_SANCTION:-"treasurer task-dfa3fd75 2026-07-28"` |
| `SERVABLE_OUT` | read, no default declared |
| `SERVABLE_STAGE` | read, no default declared |
| `SOURCE_SHARD_BYTES` | read, no default declared |
| `SPACE_MARGIN_BYTES` | default=`10737418240` |
| `SPARK_HOME` | **abort** |
| `SPARK_MASTER` | **abort** |
| `SPARK_MGMT_IPS` | **abort** |
| `SYNC_TOOL` | read, no default declared |
| `TOOLING_COMMIT` | read, no default declared |
| `TRAIN_BASE` | read, no default declared |

### CORPUS PACK — careers-qwen/pack_production_corpus.py

[Observed] reads 4 distinct variables; 0 abort when unset.

| variable | unset behaviour |
|---|---|
| `ALLOW_SHRINK` | read, no default declared |
| `MAX_SEQ` | default=`4096` |
| `PACK_SEQ` | read, no default declared |
| `PREV_CORPUS` | read, no default declared |


## Problem statement

Given the process as it exists in the repository above, **what would make this cycle fast and
reliable? Where is it fragile, and what would you change?**

[Unknown] Which parts of the two-day duration were inherent to the work and which were avoidable.
[Unknown] Whether the process as documented and the process as coded agree; nobody outside the lane
has checked.

## Constraints

[Constraint] The answer must hold for the hardware and software that exist today, not for a cluster
or toolchain we would have to acquire.
[Constraint] Training data is never published; only receipts about it are.
[Constraint] Sparks train and Thors serve; no serving runs on a Spark.
[Constraint] A run is not considered to have learned without a post-export weight-diff inside the
band `5e-05 .. 8e-04`.
[Constraint] A CPT bake emits 851 text-only tensors and production serves 1199, so the vision tower
must be grafted back; both counts are gated.
[Constraint] Every capability runs through one launcher that verifies content shas; there is no
force flag.

## Objective

[Constraint] We are not asking you to ratify a plan. We do not have one to ratify.

Requested output:

1. The specific fragilities you find, each cited to `file:line` or a commit in the repository.
2. For each, what you would change, and what it would cost to change it.
3. Anything in the record that contradicts what this packet asserts.
4. Anything you could not determine from the repository plus the appendices below — that gap is a
   finding we want reported, not worked around.

Bounded: prioritise the changes that most reduce end-to-end cycle time without weakening a gate.
