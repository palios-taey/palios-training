# restart-9b CPT — performance experiments log
Baseline (FULL_SHARD, seq4096, batch1, grad_accum4): **9.2 s/step, ~50W, 96% util(spin-wait), 27% token-eff, UMA ~60GB**. Goal: full safe util + max effectiveness. Every experiment = its own PR. Consult: plans/restart9b_consult/PERF_synthesis_first_experiment.md.

| # | PR | Change (delta vs baseline) | Predicted | Measured (power / step / loss / UMA) | Verdict |
|---|----|----|----|----|----|
| 1 | #5 | FULL_SHARD→SHARD_GRAD_OP + no_sync on non-terminal grad-accum microsteps | 50W→>100-150W, 9.2s→~3-4s, loss overlay 7.23, UMA ~30GB | _running @ seq16384_ | _pending_ |

### exp1 run notes (diagnostic — full Git record)
- **1st launch FAILED at dataset-load (NOT a comm result, never trained a step):** launched with `MAX_SEQ=4096` to isolate the comm lever cheaply, but the deployed corpus is the **16K re-chunked** `cpt_v3_dense_9b.jsonl` (125,748 rows, max 16,328). Trainer's no-truncate guard (`train_fsdp_dense_9b.py:653`) correctly asserted `CPT row exceeds max_seq: 4563>4096; corpus must be pre-chunked`. Rank 2 reported first → ChildFailedError; ranks reached finite POST-FSDP loss (7.2266) so substrate is sound. (The `Pre-FSDP forward FAILED: x.is_cuda` on rank 0 was a caught diagnostic, not the fatal path.) Clean exit, no UMA leak, no reboot.
- **Root cause = launch param, not the SHARD_GRAD_OP+no_sync change.** Fix: relaunch at `MAX_SEQ=16384` — the production seq matching the deployed corpus. At batch1 the comm lever is still isolated (no intra-batch padding), and 16K-FULL_SHARD is exactly the config that wedged at work-882, so this probe also tests whether SHARD_GRAD_OP+no_sync clears that wedge. **Probe seq is 16384, not the baseline's 4096** — absolute step-time not directly comparable; power (50W→?) is the seq-independent comm signal.

## Option space to work through (from 4/4 deep-think) before re-consulting
- A. sharding strategy: SHARD_GRAD_OP (chosen exp1) — vs NO_SHARD (risk: 16K OOM)
- B. batch_per_rank ↑ (comm-amortization, freed UMA)
- C. no_sync on grad-accum (in exp1)
- D. length-sorted seq tiers (padding tax — Step 2)
- E. allocator: drop garbage_collection_threshold:0.8 (16K GC-cliff deadlock)
- F. 16K SDPA O(N²) attention memory (needs padding/seq work first)
- G. prefetch tuning (UMA bus contention)
- H. bucketing exact per-rank micro-batch sync (if revisited)
