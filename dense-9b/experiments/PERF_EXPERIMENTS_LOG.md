# restart-9b CPT — performance experiments log
Baseline (FULL_SHARD, seq4096, batch1, grad_accum4): **9.2 s/step, ~50W, 96% util(spin-wait), 27% token-eff, UMA ~60GB**. Goal: full safe util + max effectiveness. Every experiment = its own PR. Consult: plans/restart9b_consult/PERF_synthesis_first_experiment.md.

| # | PR | Change (delta vs baseline) | Predicted | Measured (power / step / loss / UMA) | Verdict |
|---|----|----|----|----|----|
| 1 | #5 | FULL_SHARD→SHARD_GRAD_OP + no_sync on non-terminal grad-accum microsteps | 50W→>100-150W, 9.2s→~3-4s, loss overlay 7.23, UMA ~30GB | **~9.0s/step (UNCHANGED), 19-20W/96% (spin-wait), loss 7.23→0.85 finite, OOM-hung after step 30** | **REJECTED** |

### exp1 PRODUCTION verdict — REJECTED (measured on the real CPT run, not a probe)
Launched as the full production CPT (15720 steps, all 4 Sparks, 16K). Two decisive findings:
1. **No throughput gain.** Step-time stayed ~9.0s/step (step10@15:51:46 → step20@15:53:17 → step30@15:54:46), identical to the 9.2s FULL_SHARD baseline; power 19-20W at 96% util (classic spin-wait). **We are NOT comm-bound — we are per-step-overhead-bound at batch1** on this short-heavy corpus (fixed cost: optimizer step + FSDP all-gather + grad-checkpoint recompute dominates one short doc). Changing the comm strategy cannot help that. (Also: I launched with `CPT_BUCKETING=0`, the trainer's explicit *diagnostic-rollback* path (line 945) → `group_end` always True → **no_sync never engaged and grad-accum collapsed to 1**. So this measured SHARD_GRAD_OP-alone, batch1, sync-every-step.)
2. **OOM hang → Spark2 hard-freeze.** SHARD_GRAD_OP replicates params (`params=4.5GB/rank` vs FULL_SHARD's ~1.1GB) → ~6.8GB less headroom. After 30 clean steps a longer (≤16K) doc spiked activations; `dmesg: NVRM: Out of memory [NV_ERR_NO_MEMORY]` repeated; ranks spun at 19W (R-state), and **Spark2 (rank1) hard-froze ("No route to host", needs physical power-cycle)**. free dropped 47GB→20.5GB by step10, frag 77.1%.

**Conclusion → revert to the recipe's LOCKED config.** The throughput fix was never the comm strategy — it is **length-bucketing** (the recipe's #1 highest-leverage lever: amortizes the per-step overhead that IS our bottleneck), on **FULL_SHARD** (memory-safe at 16K; SHARD_GRAD_OP OOMs). exp1 productively proved (a) not comm-bound and (b) SHARD_GRAD_OP unsafe at 16K. **Next production run: FULL_SHARD + `BucketCPTDataset` (default `CPT_BUCKETING=1`, short16/mid4/long1) + pre-chunked 16K corpus.** Config reverted to FULL_SHARD this commit.

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
