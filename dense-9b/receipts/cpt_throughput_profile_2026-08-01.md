# Per-rank intra-step profile — 27B dense CPT, 4× GB10, MAX_SEQ=8192

**Date**: 2026-08-01
**Seat**: tutor
**Why**: four Family lanes fetched the repo at `6de4a11` and independently converged on the same
instruction — *measure before intervening*, with per-rank timelines, because rank-0-only
instrumentation cannot identify a straggler. This is that measurement.

## How it was produced

```
run_4node_27b_cpt.sh
  MAX_SEQ=8192 CPT_PACKED=1 BATCH_SIZE_PER_RANK=1
  TOTAL_STEPS=10 WARMUP_STEPS=2 CHECKPOINT_DCP=0
  MODEL_PATH=<SPARK_HOME>/models/Qwen3.6-27B     # same base as the 4096/8192 throughput legs
  NSYS_PROFILE_STEP=6 NSYS_PROFILE_ALL_RANKS=1
```

nsys `--capture-range=cudaProfilerApi` recording exactly optimizer step 6 on **every** rank
(4 × ~4.3 MB `.nsys-rep`). Cluster reboot-verified beforehand (4/4 changed boot IDs).
Throughput during the profiled run was 715–740 tok/s against 737 un-profiled, so the
instrumentation did not distort the thing being measured.

## Result — Observed

| rank | node | total GPU | AllGather | ReduceScatter | collectives | coll % |
|---|---|---|---|---|---|---|
| 0 | .68 | 51.36 s | 7.13 s | 3.03 s | 10.16 s | 19.8% |
| 1 | .80 | 50.55 s | 6.83 s | **2.33 s** | 9.16 s | 18.1% |
| 2 | .12 | 51.87 s | 7.28 s | 3.36 s | 10.64 s | 20.5% |
| 3 | .19 | 52.10 s | 7.21 s | **3.70 s** | 10.91 s | 20.9% |

All four ranks execute the same 65 ReduceScatter and 1176 AllGather instances — identical work,
different wait.

## The two findings

### 1. The run is NO LONGER communication-bound. The 62% figure is stale.

| | collectives | step |
|---|---|---|
| July profile, MAX_SEQ=**2560** | **61.8%** (RS 40.4% + AG 21.4%) | 103.0 s |
| This profile, MAX_SEQ=**8192** | **19.9%** (mean across ranks) | 44.4 s |

**Inferred** — enlarging the window amortized the fixed per-step collective cost, and that alone
moved the run out of the communication-bound regime. Compute now dominates: the three nvjet/cutlass
GEMM kernels account for ~41% of kernel time (21.5% + 11.3% + 8.7% on rank0).

**This inverts the standing priority list.** Every lane — and this seat — reasoned from the 62%
number, which described a configuration we no longer run. Further collective-reduction work has a
much smaller ceiling than assumed; precision and kernel efficiency move up.

### 2. `.80` is the straggler — Observed, no longer inferred.

Under a collective every rank blocks until the slowest arrives, so the straggler is the rank that
waits **least**. `.80` spends 2.33 s in ReduceScatter against 3.70 s on `.19` — the lowest of the
four on both collectives. Spread: **1.37 s per step**, roughly 3% of a 44.4 s step.

Real, reproducible, and *minor* relative to its prior billing. Worth fixing; not the main lever.

## What this does NOT establish — Unknown

- **Why** `.80` arrives late. Clock, thermal, memory-bandwidth, or host-side. Not measured here.
- Whether the ~41% GEMM share is efficient for this shape on sm_121, or leaves headroom.
- Whether FP8 would convert into end-to-end gain. No FP8 path exists in the trainer today
  (confirmed independently by two lanes: zero `fp8` matches in the trainer).
- Whether `CHECKPOINT_DCP=1` changes the picture; this profile disabled it.

## Correction this profile forces on the consult packet

**`TOKEN_BUDGET_PER_STEP` is inert in packed mode** — found by Horizon, verified here directly:
`train_fsdp_dense_9b.py:1523-1529` sets `cpt_bucket_mode = False` whenever `CPT_PACKED=1`, and the
variable is only read at line 1659, **inside** `if cpt_bucket_mode:`. Every run we have made is
packed, so the trainer never consults it.

This retires the single highest-leverage recommendation from the July round, and retires this
seat's own "we deliver 32,768 tokens against a 65,536 budget" observation — that was a comparison
against a number nothing reads.

## Artifacts

`nsys_rank{0,1,2,3}_step6.nsys-rep` on `.68` / `.80` / `.12` / `.19` under
`${SPARK_HOME}/cpt27b_logs/`, with matching `.sqlite` extractions.
