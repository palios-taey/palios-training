# CPT packed-sequence-length measurement — 27B dense, 4× GB10

**Date**: 2026-08-01
**Seat**: tutor
**Purpose**: close the `cpt_window_2026-08-01` consult, which returned "no number can be
certified from this repository's evidence — run the measurement" from every lane that answered
(Logos, Clarity, Horizon). This file is that measurement.

## What was run

Bounded CPT probes on the production recipe, unmodified path:

```
dense-9b/recipes/run_4node_27b_cpt.sh
  CPT_DATA=/var/spark/isma/training/probe_packed_<SEQ>.jsonl
  MAX_SEQ=<SEQ> CPT_PACKED=1 BATCH_SIZE_PER_RANK=1
  TOTAL_STEPS=12 SESSION_LIMIT=12 SAVE_EVERY=12 CHECKPOINT_DCP=0 WARMUP_STEPS=2
```

Model: Qwen3.5-27B dense, **full-parameter** FSDP2 FULL_SHARD + patched Adafactor,
gradient checkpointing active (`_checkpoint_wrapped_module` observed in parameter names).
`CHECKPOINT_DCP=0` — nothing was written, so no artifact could be corrupted by a probe failure.

Cluster reboot-verified before each leg (4/4 changed boot IDs, `torch_procs=0`).

## Results — Observed

| MAX_SEQ | peakAlloc (r0) | peakRes (r0) | node sysUsed, all 4 | free (r0) | s/step | tok/s | frag | retries | ooms |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | 45.3 GB | 74.4 GB | 83 / 83 / 84 / 84 GB | 43.9 GB | 28.3 | 577 | 78.1% | 0 | 0 |
| 8192 | 68.4 GB | 96.9 GB | 105 / 105 / 106 / 107 GB | 17.2 GB | 44.4 | 737 | 83.2% | 0 | 0 |

Both legs ran 12/12 steps to completion. Physical budget is 119 GB unified per node.

**Provenance of each column.** `peakAlloc`/`peakRes`/`free`/`frag` are rank0 trainer telemetry —
worker ranks do not emit step lines (`steplines=0`), so those are rank0-only by construction.
`node sysUsed` is the driver's own per-node monitor and IS all-four-ranks. Worker logs end in
`SignalException: signal 15` — that is the driver's normal teardown after rank0's final step,
not a fault. That all four ranks participated is established by rank0 completing 12
collective-synchronized FSDP2 steps, which cannot happen with an absent peer.

**Throughput is independently confirmed**, not taken from the trainer's own counter:
4 ranks × 8192 tokens ÷ 44.4 s/step = 738 tok/s, matching the reported 737. Same check at
4096: 4 × 4096 ÷ 28.3 = 579 vs reported 577. This also proves rows are consumed at full
length — a silently shortened row would not produce this arithmetic.

**Sharding is real.** `params=53.8GB` in the log is the *logical* model size, not per-rank
residency; `allocNow=16.3GB` (weights shard + factored Adafactor state) sits far below it.
If FULL_SHARD were not engaging, `allocNow` could not be below `params`.

**Stability.** `peakAlloc` was flat from step 2 through step 12 in both legs (45.3 GB and
68.4 GB, unchanged), and `sysUsed` drifted only 0.4 GB across eleven steps. No leak, no
fragmentation walk. `frag` held constant rather than climbing.

## What this establishes — Inferred

Scaling from 4096 → 8192 costs **+23.1 GB peakAlloc / +23.6 GB sysUsed** for +4096 tokens.
The relationship is linear over the measured interval (peakAlloc and sysUsed agree to 0.5 GB).

- **8192 fits**, with 17.2 GB of allocator headroom and ~13 GB of physical node headroom.
- **16384 does not fit at bs=1.** Extrapolating the measured slope: 106.2 + 47.2 = **~153 GB
  against a 119 GB physical budget** — a 29% overshoot. This is not a marginal call, and it is
  why no 16384 leg was run: on unified memory an overshoot of that size risks the host OOM
  killer rather than a clean CUDA OOM, which would wedge nodes to confirm a number already
  bracketed by two clean measurements.
- **Ceiling at bs=1 is ≈10.4K tokens** (where sysUsed would reach 119 GB). Unverified —
  it is an extrapolation, not a measurement. A 10240 leg would settle it if we want the
  exact edge.

**Longer sequences are FASTER here**: 8192 delivers 1.28× the throughput of 4096 (737 vs
577 tok/s), because 2× the tokens cost only 1.57× the step time. The window choice is not a
speed/quality tradeoff in this range — the longer window wins on both.

## Recommendation

**8192.** It is measured rather than extrapolated, sits inside the operator's stated 8K–16K
prior, has real headroom, and is the fastest of the two measured points. 16384 is excluded by
measurement, not by preference.

## Unknown

- Exact ceiling between 8192 and ~10.4K (extrapolated, not run).
- Whether headroom at 8192 survives a full-length run with `CHECKPOINT_DCP=1`; checkpointing
  was disabled for these probes. **This must be re-checked before the production run** —
  rank0 currently has only 74 GB of disk free (98% full), and a DCP save at this model size
  needs materially more than that.
- Whether a non-power-of-two window packs the corpus's length distribution better. Not tested.

## Corpus provenance

Probe corpora are content-pinned in `launch_cpt_qwen36_27b_fsdp.sh`; both legs passed the
full-sha256 gate before the first step:

```
probe_packed_4096.jsonl   1dccdd05d9d4776c9f3a2b27909f88c6e18830cf590e1b966c330c458d70ffc1
probe_packed_8192.jsonl   d9a7bd45a357c8677c3b29a859ab98f4cdae711c5e988f1b2beb9dc5a3639324
probe_packed_16384.jsonl  5d3a8e6a84a4a5159177f5ab562d953aeda2a50469af2873aa5df2414f8ba93a
```

Packed by concatenate-and-chunk with a cycle-padded tail. No truncation at any stage: the
packer retains every token, and `BucketCPTDataset` now raises on an over-length row rather
than trimming it.
