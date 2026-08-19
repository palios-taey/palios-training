# 27B Bake / Export Architecture — UNANIMOUS Chats ruling (2026-07-14)

> **This document does not decide the entrypoint.** It names production scripts because it records or explains them. The PRODUCTION AUTHORITY section of `CLAUDE.md` wins on how to run anything; see [`docs/INDEX.md`](../../docs/INDEX.md) for the full authority order.

Source: `treasurer/consultations/consult_bake_architecture.md` (+2 addenda) → responses
`bake_architecture_{gaia,horizon}.md`. Both lanes CONVERGE. This is the production export design.
Jesse directives it satisfies: (1) bakes off the training cluster / no Spark downtime; (2) 1 epoch
at a time + retention test each epoch; (3) zero truncation; (4) all production infra used.

## Root cause of the "offline reassembly is broken" finding (Gaia + Horizon agree)
`use_collectives=False` (our proven no-shared-FS resume format) writes a per-rank `__<rank>.metadata`
and **NO single global `.metadata`**. The shards are FINE; the metadata describing how they compose
is missing. Offline `dcp.load`/`dcp_to_torch_save` reads shards under the wrong global layout →
`model.norm` came back 0.23 vs true 0.96. Not a data bug — a metadata bug at SAVE time.

## The production architecture (both lanes, unanimous)
At each epoch boundary, write TWO checkpoints:

- **Artifact A — authoritative resume ckpt (UNCHANGED):** exactly today's `_save_checkpoint_dcp`
  (`use_collectives=False`, per-rank self-contained, model+optim+sched+RNG+data_pos). Proven 15s
  same-world resume. Written FIRST. Never delete the prior accepted A until the new one resumes OK.
- **Artifact B — portable model-only coordinated DCP (NEW):** `get_model_state_dict(full_state_dict=
  **False**)` (SHARDED — no 51GB gather, no wedge) → `dcp.save({"model": msd}, process_group=
  <GLOO pg>, use_collectives=True)`. The gloo PG exchanges only KB-scale plan/metadata over TCP
  (sidesteps the RoCE fabric that's the prime wedge suspect); each rank writes its OWN shard
  locally; a single global `.metadata` lands on the coordinator. **NO full_state_dict, NO
  cpu_offload gather, NO save_pretrained on the Sparks.**

Then: release all 4 Sparks → Mira durably collects (rank0 `.metadata` + every `*.distcp` + per-rank
manifests w/ sha256) and snapshots the exact training base → the pinned off-cluster conversion image
runs CPU `dcp.load(no_dist=True)` into a CPU-allocated **bf16** model (no fp32 masters) → strict
`load_state_dict` → integrity checks → `save_pretrained(safe_serialization=True)` with pinned
config/tokenizer/chat-template/generation-config. **All consolidation + HF + vLLM + eval happen
OFF the Spark cluster.** The executable lifecycle is `careers-qwen/post_cpt_pipeline.sh`.

### MANDATORY golden parity test BEFORE epoch-1 production (Horizon, hard gate)
Pin exactly torch 2.10.0 + the training transformers/model revision. Compare a B-path HF export
against a known-good reference tensor-by-tensor: identical keys/shapes/dtypes, `torch.equal` every
tensor (bf16 must be BIT-EXACT), no NaN/Inf, tied-weights identical, deterministic-logit parity on
a fixed smoke set, cold-load through the exact Thor vLLM command. ANY tensor mismatch BLOCKS — do
not stitch metadata or accept tolerance-based comparison.
- Practical golden reference for us (no prior full-gather HF exists — the gather wedged): (i)
  base-model round-trip (save base as B-format, reload via the converter, torch.equal = proves the
  converter is lossless independent of training); (ii) convert gate3 ckpt-50 via B-path → run
  gate_diff.py vs base → coherence check (model.norm ≈ 0.96 NOT scrambled-0.23; decoder meanabs in
  the AF-DOSE-predicted 1e-5–8e-4 band). (ii) SIMULTANEOUSLY delivers the deferred gate-3 bake-diff
  verdict. Scrambled norm → path wrong → back to Chats.

## Q2 — wedge root cause + the ONE discriminator (both lanes)
Leading cause: RoCE/CX-7 transport gray-failure on the FIRST cross-rank collective post-reboot
(resume is uncoordinated, so the full-state gather is the first time the ranks talk); QP/GID/ARP/PFC
not settled after cold boot → op posted, zero bytes, indefinite wait. Why nothing aborted 35 min:
our PG timeout is `timedelta(hours=1)` and heartbeat monitors watchdog-liveness not collective
progress. The B-path ELIMINATES this by not running the full gather at all; for the fallback/golden
full-gather runs, the discriminator env (both lanes):
```
TORCH_NCCL_ASYNC_ERROR_HANDLING=1  TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_TRACE_BUFFER_SIZE=2000  TORCH_NCCL_TRACE_CPP_STACK=1
TORCH_NCCL_DESYNC_DEBUG=1  NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```
+ a SHORT PG timeout (300s) so a wedge aborts-with-stack in 5 min, + a gloo `monitored_barrier(
wait_all_ranks=True)` immediately before any suspect collective (reports which rank failed to
arrive), + a 1-element NCCL all-reduce **fabric preflight** after init to force QP establishment
early and fail fast on a cold rail. Horizon adds: capture NCCL RAS (`echo verbose status | nc
localhost 28028`) twice 30-60s apart BEFORE rebooting a wedged run (op-count deltas discriminate
desync vs progress-failure).

## Q3 — LR schedule (both lanes, unanimous)
ONE warmup+cosine over the FULL 3-epoch optimizer-step horizon; warm up ONCE at global step 0;
scheduler state carried in Artifact A across the per-epoch runs; resume continuously. NOT per-epoch
sawtooth. Compute TOTAL_STEPS from the EXACT step count (not the ~120 approx). Peak lr = 1e-5
absolute (validated). Our trainer already does LambdaLR warmup+cosine over TOTAL_STEPS with DCP
scheduler resume — **already correct, no change** beyond pinning TOTAL_STEPS to the exact 3-epoch
count. Assert expected-LR == loaded-LR after each resume.

## Q4 — retention probe (mechanical verdict)
Base-vs-base noise floor FIRST (σ per category; Horizon: 4 base evals across both Thors + both
thinking modes). Then epoch-N vs untouched-base back-to-back, same serving config, greedy/temp=0.
Verdict = AND of per-category inequalities (no aggregate average hiding a category):
- Acquisition floor per target category: score(N) − base ≥ δ > σ.
- Regression guard per protected category: score(N) ≥ base − σ.
- Monotonic (N≥2): score(N) ≥ score(N−1) − σ on targets.
Report every delta as count-next-to-σ. Horizon's rigorous form adds T+/P/Z category classes,
bootstrap CIs (10k resamples), Holm-adjustment, host-swap, two thinking-mode strata, PASS/HOLD/FAIL
state machine — that's the target; Gaia's simpler AND-of-inequalities is the minimum viable gate.
Thor serves bf16 (infra: control-precision parity), thinking-mode fixed+frozen.

## Q5 — tail handling (both lanes: mask the pad)
Current cycle-pad is zero-DROP but not zero-DISTORTION (683 head tokens = 0.0139% get 2× grad/epoch).
Correct fix = **masked padding**: keep the 1926×2560 shape, keep the 683 head tokens as physical
filler, but mask every filler TARGET with `ignore_index=-100` (`labels[last_block, 1877:] = -100`)
so they contribute ZERO gradient — plus loss normalization by global valid-target-token count
(`reduction="sum"` / Σ valid tokens, all-reduced over DP ranks). Causal model → masked right-tail
filler can't leak into real-token loss. Interim: current cycle-pad is ACCEPTABLE at 0.0139% if
logged as a known head-upweight (Gaia option 3); masked padding is the ship-target.

## Divergence (minor, reconciled)
Gaia offered (a) in-run full-state consolidation as a hardened BRIDGE while (c) stands up; Horizon
REJECTS (a) outright (retains the wedge-prone gather). Reconciliation: skip (a) — go straight to
Artifact-B model-only coordinated DCP (Gaia's own "Form 1" == Horizon's "modified (c)", identical
mechanism). The full-state gather never runs in production. Both lanes support this exact end-state.
