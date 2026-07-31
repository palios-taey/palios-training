# PRODUCTION RECIPE — Qwen3.6-27B revenue-model CPT (validated 2026-07-14)

Everything below is EVIDENCE-VALIDATED, not proposed. Optimizer: gate-3 weight-diff PASS +
in-run AF-DOSE PASS. Bake: Artifact-B path proven (coherent offline convert). Corpus: zero-truncation
packed from treasurer-registered slices. This is the recipe to run once treasurer registers the
packed-corpus row.

## Objective (Jesse, scope-locked)
Train the careers/revenue 27B to KNOW in-weights, zero-lookup: all repos as sellable capabilities,
Jesse's career background + voice, platform strategies/processes/best-practices (Upwork/LinkedIn),
revenue research. WORK-KNOWLEDGE + VOICE. NOT constitutional/personality training. NOT RAG.

## Corpus (BLOCKED on treasurer REGISTRY row)
- `cpt_production_v1_packed_2560.jsonl`, sha256 `4bfb2a57cc9feb786cc6042a39614e9704e4f73305a8004a3ae63f611395e68c`
- 1926 blocks × 2560 tok = 4,930,560 tok, zero-truncation (cycle-pad tail; masked-pad is the ship-refinement, logged 0.0139% head-upweight interim)
- 6 sha-gated registered inputs: raw_corpus_v4 946r@6e44bd0c, public_repos_v1 890r@948eed2c,
  careers_kb_v1 316r@ac151e02, db_worldmodel_v1 33r@bb80a36f, consultations_v1 330r@1a405a83,
  recaps_v1 16r@a1079b6f. Method: `careers-qwen/pack_production_corpus.py` (committed de6b2c9).
- Deployed sha-identical on all 4 nodes at `/var/spark/isma/training/`.

## Optimizer (VALIDATED — de5d78e, gate-3 PASS)
- `ADAFACTOR_ALPHA_MODE=absolute` (param_scale=1.0 → alpha=lr), `ADAFACTOR_EPS1=fp32` (1.192e-7 —
  correctness fix, carries unconditionally), `LR=1e-5`, `WARMUP_STEPS=15` (≈4% of 363), wd=0.01, d=1.0.
- bf16 params + stochastic-rounding write-back (validated: null signature broken, SR-DELTA PASS).
- Gate-3 evidence: floor_frac=0.000, RMS(U_hat) 0.88–1.0 (in [0.3,1.5]), decoder bake-diff meanabs
  8.7e-5 (in band, ~1% of |w|). lr=1e-5 kept (gate-3's U_hat≈1 at d=1 clip → effective dose = lr;
  meanabs healthy, no lr recompute needed). If epoch-1 decoder movement reads low, use Gaia's
  lr_prod = 1e-5 × (M_target/meanabs) — but gate-3 says 1e-5 already imprints.

## Schedule (UNANIMOUS Chats: single cosine over full 3-epoch horizon)
- TOTAL_STEPS = exact 3-epoch optimizer-step count. COVERAGE PROOF at launch: steps/epoch × 16
  blocks/step ≈ 1926 → steps/epoch = ceil(1926/16) = 121; ×3 epochs = **363**.
  (16 = global batch = BATCH_SIZE_PER_RANK 4 × 4 ranks.)
- ONE warmup (15 steps) at global step 0, then cosine decay to ~5–10% floor over the remaining
  steps through end of epoch 3. LambdaLR (trainer already does warmup+cosine over TOTAL_STEPS);
  scheduler state carried in DCP (Artifact A) and resumed continuously across the per-epoch runs.
- Do NOT restart warmup per epoch. After resume, assert expected-LR == loaded-LR.


## ★ SAVE/REBOOT ARCHITECTURE (Jesse directive 2026-07-15, BINDING — I violated this and killed the cluster)

**ONE save per session. The save IS the session end. The session end IS the reboot boundary.
NEVER save-and-continue. NEVER stop/restart without a reboot.**

- Therefore **`SAVE_EVERY` MUST EQUAL `SESSION_LIMIT`**, sized to the ~2h cycle (~90 steps @77s/step).
- Sequence per cycle: train N steps → save (the ONE save) → process exits → **reboot all 4** →
  resume from that checkpoint. This is the only exercised, proven path (every gate ran 50/50).
- **What a mid-session save does (measured 2026-07-15 12:40Z):** `SESSION_LIMIT=90 SAVE_EVERY=40`
  saved at step 40 and tried to keep training → host RAM to 0 of 122566 MiB → **all 4 nodes
  hard-died**. The DCP save host-stages ~13.4GB/rank from the SAME unified pool the allocator is
  already holding (~94GB reserved); resuming training on top of that = node death. Thermal was
  excluded (83C peak, watchdog never fired).
- To land a session exactly on an epoch boundary, size SESSION_LIMIT to the remaining steps
  (e.g. resume@40 + SESSION_LIMIT=SAVE_EVERY=81 → save+exit exactly at step 121 = epoch-1).

## Epoch-at-a-time loop (Jesse directive; per-epoch, ONE epoch then verdict gate)
For epoch N (SESSION_LIMIT == SAVE_EVERY, sized to a ~2h cycle ~90 steps, or to land exactly on the
epoch boundary; a 121-step epoch = e.g. 90 then 31, or 40 then 81):
1. Reboot all 4 fresh → fabric verify → ONE watchdog. Launch/resume training on the production corpus:
   ```
   CLOCK_CAP=1000 ADAFACTOR_ALPHA_MODE=absolute ADAFACTOR_EPS1=fp32 LR=1e-5 WARMUP_STEPS=15 \
   CPT_DATA=/var/spark/isma/training/cpt_production_v1_packed_2560.jsonl CPT_PACKED=1 \
   BATCH_SIZE_PER_RANK=4 MAX_SEQ=2560 TOTAL_STEPS=363 SESSION_LIMIT=<N> SAVE_EVERY=<N>  # MUST BE EQUAL \
   OUTPUT_DIR=<SPARK_HOME>/training_outputs/production_v1 bash run_4node_27b_cpt.sh
   ```
   Verify env in the live proc. SESSION-1 IS THE SMOKE TEST (Gaia mandate): watch [AF-DOSE] live
   (floor_frac<0.05, U_hat 0.3–1.5, eps1_actual=1.192e-7) — halt if out-of-band before burning the run.
2. At epoch-N boundary: write Artifact A (resume ckpt, unchanged) FIRST. Then run the production
   post-CPT wrapper; it fresh-reboots all four and writes Artifact B via `EXPORT_DCP`
   (gloo-coordinated, no full gather).
3. `DCP_DIR=<completed-run> bash careers-qwen/post_cpt_pipeline.sh` atomically collects Artifact B
   plus the exact training base to durable controller storage, retires the transient Spark export,
   stages both to Thor1, and converts in the immutable torch 2.10 / transformers 5.3 image named in
   `fleet.env`. The weight-diff gate runs before graft or handoff. The wrapper never launches SFT.
4. Retention probe on Thor vLLM (bf16, thinking-mode frozen): epoch-N vs untouched-base,
   base-vs-base σ floor first; verdict = AND of per-category inequalities (Q4 protocol, Gaia
   minimal / Horizon rigorous). PASS → launch epoch N+1. FAIL/HOLD → stop, numbers to Chats.
5. **Jesse surface point = the epoch-1 retention probe result** (demonstrated retention = the report).

## Hardware / thermal (unchanged, binding)
- 4× DGX Spark GB10, 128GB UMA. CLOCK_CAP=1000 (nvidia-smi -lgc 0,1000) every launch. ONE thermal
  watchdog (PULL_OFF=86). Reboot-after-every-session. ~77s/step → 363 steps ≈ 8h train + per-epoch
  export/convert/probe. Never kill-and-relaunch on dirty GPUs.

## Bake instrumentation (validated, committed 83807ae)
- EXPORT_DCP fabric preflight (localizes any RoCE wedge) + Flight-Recorder env (wedge aborts-with-
  stack not 1hr silent). Offline convert aborts on missing .metadata / NaN / sha-mismatch.

## Launch gates (all must hold before epoch-1)
1. [DONE] optimizer validated (gate-3 weight-diff + AF-DOSE).
2. [DONE] bake path golden-validated (coherent offline convert).
3. [PENDING] treasurer registers `cpt_production_v1_packed_2560` (4bfb2a57) REGISTRY row.
4. [DONE 2026-07-29] Thor1 conversion image pinned by immutable digest; verified torch 2.10.0 and
   transformers 5.3.0. Conversion requires the redundant-node maintenance path, never co-runs with
   a live 27B service.
Then: launch epoch-1 session-1 as smoke.
