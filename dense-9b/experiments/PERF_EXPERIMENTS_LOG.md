# restart-9b CPT — performance experiments log
Baseline (FULL_SHARD, seq4096, batch1, grad_accum4): **9.2 s/step, ~50W, 96% util(spin-wait), 27% token-eff, UMA ~60GB**. Goal: full safe util + max effectiveness. Every experiment = its own PR. Consult: plans/restart9b_consult/PERF_synthesis_first_experiment.md.

| # | PR | Change (delta vs baseline) | Predicted | Measured (power / step / loss / UMA) | Verdict |
|---|----|----|----|----|----|
| 1 | (pending) | FULL_SHARD→SHARD_GRAD_OP + no_sync on non-terminal grad-accum microsteps | 50W→>100-150W, 9.2s→~3-4s, loss overlay 7.23, UMA ~30GB | _running_ | _pending_ |

## Option space to work through (from 4/4 deep-think) before re-consulting
- A. sharding strategy: SHARD_GRAD_OP (chosen exp1) — vs NO_SHARD (risk: 16K OOM)
- B. batch_per_rank ↑ (comm-amortization, freed UMA)
- C. no_sync on grad-accum (in exp1)
- D. length-sorted seq tiers (padding tax — Step 2)
- E. allocator: drop garbage_collection_threshold:0.8 (16K GC-cliff deadlock)
- F. 16K SDPA O(N²) attention memory (needs padding/seq work first)
- G. prefetch tuning (UMA bus contention)
- H. bucketing exact per-rank micro-batch sync (if revisited)
