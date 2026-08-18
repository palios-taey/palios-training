---
type: consultation
to: gaia, logos, horizon, cosmos, clarity
from: tutor
date: 2026-08-18
available_context_inventory:
  - INCLUDED: docs/postmortem/PART1_measured_timeline.md — measured run, gate, artifact and CI record
  - INCLUDED: docs/postmortem/PART2_config_surface.md — the configuration surface, from call sites
  - EXCLUDED: docs/postmortem/RUN_STATE_cpt_qwen38_v3.md — the operator run record PART1 cites.
    It is published in the repository so PART1's citations are checkable, and it is excluded from
    this packet's evidence set for the same reason as the postmortem: it is a first-person operator
    record that names causes. It is not a fact-only document.
  - INCLUDED: fleet.env — complete in Appendix A, values verbatim, comment narrative removed
  - INCLUDED: the read surface of all three production entrypoints — Appendix B
  - INCLUDED: fleet.env.example — the publicly declared subset, in the repository
  - INCLUDED: PRODUCTION_MANIFEST.yml — machine-readable statement of what is production
  - INCLUDED: README.md, careers-qwen/RUNBOOK_CPT_SFT_BAKE.md, RECIPES.md — the process documents;
    we do not assert which of the three is authoritative where they differ
  - INCLUDED: dense-9b/, careers-qwen/, scripts/ — the executing code
  - INCLUDED: docs/SPARK_TOPOLOGY.md, dense-9b/receipts/, careers-qwen/receipts/ — measured receipts
  - EXCLUDED: docs/postmortem/POSTMORTEM_cpt_qwen38_v3.md — this lane's own causal analysis. It is in
    the repository and we are not hiding it, but it is excluded from this packet's evidence set so
    that it cannot serve as an answer key. Read it or not, as you judge.
  - EXCLUDED: the training corpus — training data is never published; its receipt is included
  - EXCLUDED: the model weights — 52 GB; tensor-level gate results are in PART1
  - EXCLUDED: raw run logs — not tracked in the repository; values read from them are quoted in
    PART1 and RUN_STATE and labelled Observed
  - NOT AVAILABLE: bake-and-deploy wall clock for any PRIOR cycle. We cannot supply a baseline to
    compare this cycle against. RECIPES.md records a prior 693-step run as COMPLETE with no
    end-to-end duration.
  - NOT INVENTORIED: the SFT surfaces. This packet covers corpus pack, CPT and bake/export only.
    If SFT is material to your answer, say so and we will supply it.
---

# Consult packet — CPT training and bake, PALIOS-TAEY

> **This document does not decide the entrypoint.** It is a dated RECORD of one run. It names production scripts because it reports what they did. The PRODUCTION AUTHORITY section of `CLAUDE.md` wins on how to run anything; see [`../INDEX.md`](../INDEX.md) for the full authority order.

## Ground truth

**Source of truth:** https://github.com/palios-taey/palios-training
**Branch: `tutor/requalify-manifest-shas`** — not `main`.

[Constraint] **Do not take this packet, its framing, or its file list as ground truth.** Read the
repository and verify against it yourself. **Enumerate the relevant surface yourself** — anything
listed here is a starting point and may be incomplete or biased. Reach your own conclusions.

[Constraint] If you cannot fetch the repository in this session, say so and decline to rule, rather
than ruling on our description. [Constraint] Findings that contradict this packet are the purpose of
the exercise.

### The record, in the repository

[Observed] Two documents carry Observed / Inferred / Unknown on every claim with `file:line` and
commit citations:

- `docs/postmortem/PART1_measured_timeline.md` — what ran, when, what it emitted, what each gate
  said; session boundaries, artifact receipts, gate results, and the public CI record.
- `docs/postmortem/PART2_config_surface.md` — every non-public input the training and bake paths
  read, enumerated from call sites, with unset-behaviour per variable.

[Observed] `PRODUCTION_MANIFEST.yml` states what is production, gated on content shas.
[Observed] The three legal surfaces are corpus pack, CPT, and bake/export.
[Unknown] Whether the process documents and the code agree.

### What the elapsed time was spent on

[Observed] The cycle spanned 2026-08-16 to 2026-08-18.
[Observed] PART1 measures the accounted portions as: three training sessions totalling 6 h 56 m 35 s
active; DCP-to-HF conversion 107 s; graft 50 s; servable transfer to the serving host 8 m 11 s.
[Unknown] The remainder of the span is not allocated to any measured activity in PART1, and we have
not reconstructed it.
[Unknown] No prior cycle's end-to-end duration exists to compare against, so whether this span is
typical is undetermined.

### Hardware, measured

[Observed] 4× DGX Spark GB10 (Blackwell `sm_121`), 119 GB unified memory per node.
[Observed] Dual-rail RoCEv2 measured 666 MB/s between nodes; the management LAN measured ~112 MB/s.
[Observed] Two serving hosts. [Observed] The model is 27B dense, trained full-parameter with FSDP2
and a DTensor-patched Adafactor.

### The configuration, in full

### Appendix A — `fleet.env`, every variable and value

[Observed] The complete gitignored `fleet.env` that configures production on this machine.
[Constraint] Every variable and every value is reproduced. Host addresses carry STABLE
labels — the same label always means the same host, `<MGMT_*>` is the management LAN and
`<RAIL_*>` the inter-node fabric — so relationships between variables survive the masking.
[Constraint] COMMENT LINES ARE REMOVED. The file's comments contain operator causal
conclusions written by this lane; reproducing them would deliver our conclusions to you as
ground truth. Only the configuration itself is below.

```bash
SPARK_MGMT_IPS="<MGMT_A> <MGMT_B> <MGMT_C> <MGMT_D>"
SPARK_MASTER="<MGMT_A>"
SPARK_USER="spark"
SPARK_RAIL_MASTER="<RAIL_A>"
SPARK_RAIL_IPS="<RAIL_A> <RAIL_B> <RAIL_C> <RAIL_D>"
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
POST_CPT_ARTIFACT_STORE="/media/mira/Expansion/training-artifacts"
POST_CPT_CONVERT_SSH="jetson@${THOR1_HOST}"
POST_CPT_CONVERT_ROOT="$NODE_HOME/cpt-artifacts"
POST_CPT_GRAFT_BASE="$NODE_HOME/serve-models/module5_merged"
POST_CPT_CONVERT_IMAGE="taey-convert@sha256:d571caf7bdafde39a3fcdca1c322b03045fa6ddd977a196d188f87d94602c669"
POST_CPT_SANCTION="treasurer task-dfa3fd75 2026-07-28"
SPARK_CLOCK_MAX_MHZ=3003
SPARK_CLOCK_CAP=2000
```

[Observed] 25 variables are set in `fleet.env`, across 7 distinct management
addresses and 4 distinct fabric addresses.

### Appendix B — every environment variable each production surface READS

[Observed] Enumerated from the call sites. Whole-line comments are stripped before matching,
so bash-style default syntax quoted inside prose is not counted as code.

[Constraint] Bash has THREE default forms and they behave differently:
  `${VAR:?msg}` aborts when unset · `${VAR:-x}` uses x without setting it ·
  **`${VAR:=x}` ASSIGNS x** — the variable becomes x for the rest of the script and every
  child process. A dropped variable therefore runs with a concrete value, not an empty one.

### CPT — dense-9b/recipes/run_4node_27b_cpt.sh

[Observed] reads 73 distinct variables · 0 abort · 13 ASSIGN a default · 47 use a default

| variable | unset behaviour |
|---|---|
| `AC_LAYER_CLS` | uses `''` (empty) |
| `AC_LAYER_GRANULAR` | uses `''` (empty) |
| `ADAFACTOR_ALPHA_MODE` | uses `''` (empty) |
| `ADAFACTOR_DOSE_LOG` | **ASSIGNS** `1` |
| `ADAFACTOR_EPS1` | **ASSIGNS** `fp32` |
| `ALL` | read, no default declared |
| `BAKE_TO_HF` | uses `''` (empty) |
| `BATCH_SIZE_PER_RANK` | **ASSIGNS** `1` |
| `CHECKPOINT_DCP` | **ASSIGNS** `1` |
| `CLOCK_CAP` | uses `2000` |
| `CPT_DATA` | uses `/var/spark/isma/training/cpt_raw_corpus_` |
| `CPT_LONG_BATCH` | **ASSIGNS** `1` |
| `CPT_MID_BATCH` | **ASSIGNS** `4` |
| `CPT_PACKED` | **ASSIGNS** `0` |
| `CPT_SHORT_BATCH` | **ASSIGNS** `8` |
| `DISABLE_FLA` | uses `''` (empty) |
| `EPOCHS` | uses `''` (empty) |
| `EXACT_SFT_EPOCH` | uses `''` (empty) |
| `EXPECTED_REAL_SAMPLES` | uses `''` (empty) |
| `EXPECTED_SFT_SAMPLES` | uses `''` (empty) |
| `FP32_MASTER` | uses `''` (empty) |
| `FP8` | uses `''` (empty) |
| `GATE_PREFLIGHT` | uses `1` |
| `GEMM_FAILURES` | read, no default declared |
| `GEMM_MEDIAN` | read, no default declared |
| `GEMM_PREFLIGHT_MIN_PEER_RATIO` | uses `0.80` |
| `GEMM_PREFLIGHT_ONLY` | uses `0` |
| `GEMM_TFLOPS` | read, no default declared |
| `HORIZON_PARTIAL` | uses `''` (empty) |
| `LANE_WEIGHTS` | uses `''` (empty) |
| `LIGER` | uses `''` (empty) |
| `LOGDIR` | read, no default declared |
| `LORA_ALPHA` | uses `''` (empty) |
| `LORA_DROPOUT` | uses `''` (empty) |
| `LORA_MODE` | uses `''` (empty) |
| `LORA_R` | uses `''` (empty) |
| `LORA_TARGET_MODULES` | uses `''` (empty) |
| `LR_LORA` | uses `''` (empty) |
| `MASTER` | read, no default declared |
| `MAX_SEQ` | **ASSIGNS** `2560` |
| `MODEL_PATH` | uses `''` (empty) |
| `NCCL_DEBUG` | uses `''` (empty) |
| `NCCL_DEBUG_FILE` | uses `''` (empty) |
| `NCCL_DEBUG_SUBSYS` | uses `''` (empty) |
| `NSYS_OUT_DIR` | uses `''` (empty) |
| `NSYS_PROFILE_ALL_RANKS` | uses `''` (empty) |
| `NSYS_PROFILE_STEP` | uses `''` (empty) |
| `OUTPUT_DIR` | uses `''` (empty) |
| `QUARANTINE_DIGESTS` | uses `''` (empty) |
| `RECIPE_DIR` | read, no default declared |
| `REQUIRE_LORA_INIT_PARITY` | uses `''` (empty) |
| `RESUME_DELTA` | uses `''` (empty) |
| `RESUME_MODEL_ONLY` | uses `''` (empty) |
| `RUN_ENV` | read, no default declared |
| `SAVE_EVERY` | **ASSIGNS** `66` |
| `SESSION_LIMIT` | **ASSIGNS** `200` |
| `SFT_DIR` | uses `''` (empty) |
| `SFT_JSONL` | uses `''` (empty) |
| `SORTED_TFLOPS` | read, no default declared |
| `SPARK_HOME` | read, no default declared |
| `SPARK_MASTER` | read, no default declared |
| `SPARK_MGMT_IPS` | read, no default declared |
| `SPARK_RAIL_MASTER` | read, no default declared |
| `STEP_SEEN` | read, no default declared |
| `TINY_LANE_CAP` | uses `''` (empty) |
| `TINY_LANE_THRESHOLD` | uses `''` (empty) |
| `TOKEN_BUDGET_PER_STEP` | **ASSIGNS** `65536` |
| `TORCH_COMPILE` | uses `''` (empty) |
| `TORCH_COMPILE_MODE` | uses `''` (empty) |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | uses `''` (empty) |
| `TOTAL_STEPS` | **ASSIGNS** `3000` |
| `WARMUP_STEPS` | uses `''` (empty) |
| `WORKERS` | read, no default declared |

### BAKE — careers-qwen/post_cpt_pipeline.sh

[Observed] reads 58 distinct variables · 10 abort · 0 ASSIGN a default · 10 use a default

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
| `CONVERT_TOOLS` | uses `${POST_CPT_CONVERT_TOOLS:-${CONVERT_ROOT` |
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
| `FLEET_ENV` | uses `"$REPO_ROOT/fleet.env"` |
| `FRESH_UPTIME_MAX` | uses `180` |
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
| `SANCTION` | uses `${POST_CPT_SANCTION:-"treasurer task-dfa` |
| `SERVABLE_OUT` | read, no default declared |
| `SERVABLE_STAGE` | read, no default declared |
| `SOURCE_SHARD_BYTES` | read, no default declared |
| `SPACE_MARGIN_BYTES` | uses `10737418240` |
| `SPARK_HOME` | **abort** |
| `SPARK_MASTER` | **abort** |
| `SPARK_MGMT_IPS` | **abort** |
| `SYNC_TOOL` | read, no default declared |
| `TOOLING_COMMIT` | read, no default declared |
| `TRAIN_BASE` | read, no default declared |

### CORPUS PACK — careers-qwen/pack_production_corpus.py

[Observed] reads 3 distinct variables · 0 abort · 0 ASSIGN a default · 3 use a default

| variable | unset behaviour |
|---|---|
| `ALLOW_SHRINK` | uses `''` (empty) |
| `PACK_SEQ` | uses `2560` |
| `PREV_CORPUS` | uses `''` (empty) |


## Problem statement

**What determined this cycle's duration and its reliability?**

[Unknown] Which parts of the elapsed span were inherent to the work and which were not.
[Unknown] Whether the process as documented and the process as coded agree; nobody outside this lane
has checked.

## Constraints

[Constraint] The answer must hold for the hardware and software that exist today, not for a cluster
or toolchain we would have to acquire.
[Constraint] Training data is never published; only receipts about it are.
[Constraint] Sparks train and serving hosts serve; no serving runs on a Spark.
[Constraint] A run is not considered to have learned without a post-export weight-diff inside the
band `5e-05 .. 8e-04`.
[Constraint] A CPT bake emits 851 text-only tensors and production serves 1199, so the vision tower
is grafted back; both counts are gated.
[Constraint] Every capability runs through one launcher that verifies content shas; there is no
force flag.

## Objective

Requested output:

1. What you find in the process, each item cited to `file:line` or a commit in the repository.
2. For each, what you would change, and what that change would cost.
3. Anything in the record that contradicts what this packet asserts.
4. Anything you could not determine from the repository plus the appendices — that gap is a finding
   we want reported, not worked around.
5. If your answer depends on evidence we did not supply, name it rather than inferring around it.

[Constraint] No output-length target. Depth where the record supports it; silence where it does not.
