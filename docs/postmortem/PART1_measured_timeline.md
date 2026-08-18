# Qwen3.8-27B CPT and bake: measured timeline

> **This document does not decide the entrypoint.** It is a dated RECORD of one run. It names production scripts because it reports what they did. The PRODUCTION AUTHORITY section of `CLAUDE.md` wins on how to run anything; see [`../INDEX.md`](../INDEX.md) for the full authority order.

Part 1 of 2. This section is limited to the measured run, artifact, gate, repository, and CI record. The separate configuration analysis is outside its scope.

All timestamps are UTC. `Observed` means a value was read from an artifact, log, repository object, or GitHub Actions receipt. `Inferred` means arithmetic over observed values. `Unknown` means the source needed for a fresh measurement was not present in the authorized local evidence set.

## Evidence boundary

- **Observed:** The run and production-change evidence in this section is pinned to `palios-taey/palios-training` commit `fa5ff1e89c650ac2a0f76148b042517aef659be5` on PR #11. The delivery base for this file is the current `tutor/requalify-manifest-shas` head, `c89fce50632313a89ed81d828727926158147c4d`; the current public `main` head is `129de416a9ca14a163c1b0d96decd8c25dd0601a`.
- **Observed:** The preserved same-day run receipt is `QWEN38_RUN_STATE.md`, SHA-256 `6a697a73a0051335509eecb5eda43f660105ebd33bc94cc4f374f066348dbddf`. Citations to that filename refer to the local receipt, not to a tracked public-repository file.
- **Observed:** The durable corpus is `cpt_qwen38_v2_nopack_8192.jsonl`, SHA-256 `3973c2af608974191c7db2568c008510aa1711bdb714eede31a33fe414576e97`. Its manifest SHA-256 is `1a46a259d5eea32a54a320fef673160f6c91f2f681d2f71bc03492630140579b`.
- **Observed:** The durable exported Artifact B contains global `.metadata`, four `READY.rank<N>` files, four `manifest.rank<N>.json` files, and four equal `13,449,338,955`-byte DCP shards. Its `SHA256SUMS` file has SHA-256 `d39e5d77998762cf9abade694525c1cf6cc253cfe090ec17b56155d212cdd644`.
- **Unknown:** A fresh read of the original `final/trainer_meta.pt` and rank-0 log was not possible from the authorized local evidence set. The metadata and log values below are the values captured from those sources on 2026-08-18 while they were available. The trainer defines the emitted metadata fields at `dense-9b/trainers/train_fsdp_dense_9b.py:3666-3674` and writes `COMPLETE` last at `:3678-3680`.

## Run and session timeline

### Corpus and horizon

- **Observed:** The corpus manifest reports `2,717` rows, `24,542,925` bytes, and `5,334,849` tokens. The committed builder recomputes the corpus hash, byte count, and row count at `careers-qwen/build_cpt_nopack_corpus.py:120-140` and emits the same canonical fields on a full build at `:213-238` (`e164ebd5a849ed687637af04c7339c9db2357c31`).
- **Observed:** The first launch attempt emitted `real_unique=2717`, `omitted=0`, `duplicates=0`, `expected_padding=15`, `emitted_padding=15`, bucket counts `short=1637`, `mid=1057`, `long=23`, and `groups_by_epoch=[72, 73, 73]`. It then emitted `CPT BUCKET HORIZON FAILED: TOTAL_STEPS=213, expected optimizer groups=218, remaining=5`. The accepted launch emitted `CPT BUCKET HORIZON PASS: TOTAL_STEPS=218 groups_by_epoch=[72, 73, 73]` (`QWEN38_RUN_STATE.md:167-179`).
- **Observed:** The code constructs the coverage receipt at `dense-9b/trainers/train_fsdp_dense_9b.py:923-937`, rejects nonzero omissions or duplicates at `:1896-1907`, sums the per-epoch group counts at `:2493-2494`, and emits the failed or passed horizon verdict at `:2509-2524` (`4f935235a04013064d455659a02b5d4155433636`).

### Final metadata captured from `trainer_meta.pt`

| Field | Captured value | Register |
|---|---:|---|
| `step` | `218` | Observed |
| `epoch` | `2` (zero-indexed third epoch) | Observed |
| `num_ranks` | `4` | Observed |
| `max_seq` | `8192` | Observed |
| `sched.base_lrs` | `[1e-05]` | Observed |
| `sched._last_lr` | `[1.0000000000000002e-06]` | Observed |
| `sched._step_count` | `873` | Observed |
| `data_pos` | `73` | Observed |

The final directory name is also mechanical: a completed DCP save uses `final`, while an intermediate save uses `checkpoint-<step>` (`dense-9b/trainers/train_fsdp_dense_9b.py:3643-3644`).

### Three training sessions

| Session | Observed log boundary | Emitted artifact / terminal log line | Wall clock |
|---|---|---|---:|
| 1, steps `0 -> 73` | `02:55:10.738` start; `05:14:43.803` DCP complete | `checkpoint-73`; `05:14:44.239 [step 73] FRAGMENTATION EXIT` | `2:19:34` (Inferred, rounded) |
| 2, steps `73 -> 146` | `05:26:49.967` DCP resume; `05:26:57.381` start; `07:46:11.105` DCP complete | `checkpoint-146`; `07:46:11.593 [step 146] FRAGMENTATION EXIT` | `2:19:14` (Inferred) |
| 3, steps `146 -> 218` | `07:56:58.880` DCP resume; `07:57:06.607` start; `10:14:54.350` DCP complete | `final`; the completion log names `checkpoint-218` | `2:17:48` (Inferred) |

- **Observed:** Session 1's checkpoint was `13G` with `COMPLETE` on all four nodes; its save and exit lines are preserved at `QWEN38_RUN_STATE.md:215-225`.
- **Observed:** Session 2 resumed model, optimizer, scheduler, and RNG at step `73`, with scheduler internal step `292` and LR `8.31e-06`; the preserved receipt is `QWEN38_RUN_STATE.md:243-250`.
- **Observed:** Session 3 resumed the same four state classes at step `146`, scheduler internal step `584`, and LR `3.52e-06`.
- **Observed:** The first two sessions ended through the checkpoint-and-exit path that prints `FRAGMENTATION EXIT` at `dense-9b/trainers/train_fsdp_dense_9b.py:3481-3509`. The third reached the final-save path and emitted `final`.
- **Inferred:** Using the unrounded session boundaries, the active-session sum is `6:56:35`. The first start through final DCP completion spans `7:19:44`; this span includes the intervals between sessions.

## Gate record

The in-training and post-export reads are reported together:

| Gate | Measured result | Gate statement | Register |
|---|---:|---|---|
| Session 1 SR-DELTA | `1.16x ULP` | `PASS (in Gaia band [0.5u,20u])` | Observed |
| Session 2 SR-DELTA | `1.03x ULP` | `PASS (in Gaia band [0.5u,20u])` | Observed |
| Session 3 SR-DELTA, binding read near step `190` | `0.49x ULP`; LR `1.42e-06`, versus observed peak about `9.7e-06` | `FAIL-LOW (<0.5u)` | Observed |
| Post-export absolute weight diff | `2.223e-04` against `5e-05 .. 8e-04` | `IN BAND` | Observed |
| Post-export changed fraction | `0.881` | reported; not the script's pass/fail axis | Observed |

The trainer captures an SR reference and reads it at session-relative steps `1`, `10`, `20`, and `40`; its `0.5u .. 20u` binding band and emitted tags are at `dense-9b/trainers/train_fsdp_dense_9b.py:3313-3365`. The post-export tool defines the absolute band at `careers-qwen/measure_cpt_delta.py:33-51`, computes absolute, relative, and changed-fraction statistics at `:95-118`, and prints the absolute-axis verdict at `:131-148`. The paired run receipt is preserved at `QWEN38_RUN_STATE.md:287-300`.

## Bake and artifact movement

- **Observed:** DCP-to-HF conversion of the `851`-tensor language artifact ran node-local on one Spark from `20:46:41` to `20:48:28`: `107 s`.
- **Observed:** The graft ran on the same Spark from `20:56:20` to `20:57:10`: `50 s`. It emitted `patched 851`, `kept 348`, `total 1199`; the resulting directory had `18` shards and five named sidecars.
- **Observed:** The completed Spark-to-Thor1 transfer receipt records `55,586,114,895` bytes in `8 min 11 s`, approximately `108 MB/s`, with `32` source and destination files matching in byte total. The tracked runbook records the transfer as `112 MB/s, ~8 min`; these are the observed completion-average and the tracked rounded throughput record, respectively (`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md:162-169`, `5629834054ada77d934ed2c2e5f63918d717d5bc`).
- **Observed:** The same runbook records the controller artifact store at `19 MB/s`, approximately `46 min` each way for the same `52G` class of artifact (`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md:168-169`, `5629834054ada77d934ed2c2e5f63918d717d5bc`).
- **Observed:** The current pipeline requires `ARTIFACT_STORE` at `careers-qwen/post_cpt_pipeline.sh:13-24`, derives `LOCAL_ARTIFACT` and `LOCAL_BASE` beneath it at `:97-98`, copies the corpus and manifest into that store at `:378-406`, and then stages the stored inputs to the conversion host at `:408-420` (`fa5ff1e89c650ac2a0f76148b042517aef659be5`).

## Recorded production mismatches and committed changes

This list records each pre-change state, its observed manifestation, and the commit that changed the repository. It does not rank the entries.

1. **Completed checkpoint selection.** **Observed:** Before `6b63716`, the pipeline selected only the numerically last `checkpoint-*` directory (`6b63716^:careers-qwen/post_cpt_pipeline.sh:269-274`). The run held `checkpoint-73`, `checkpoint-146`, and `final`; that selector returned `146`. The trainer names a completed save `final` at `dense-9b/trainers/train_fsdp_dense_9b.py:3643-3644`. Commit `6b63716d82d4b9caf65076e37919e9c81479af4f` added the `final/COMPLETE` branch, reads its step from `trainer_meta.pt`, and retains the numeric fallback when no final directory is present (`careers-qwen/post_cpt_pipeline.sh:270-295`).

2. **851-tensor causal-LM base derivation.** **Observed:** The parent tree of `fc3c7fb` contained no `careers-qwen/derive_training_base_851.py`. Commit `fc3c7fb21d3de83d5028b01c3d4391b64bb22155` added it. The committed mapping keeps `lm_head.weight`, maps `model.language_model.<X>` to `model.<X>`, and expects exactly `851` tensors (`careers-qwen/derive_training_base_851.py:46-59`). It rejects wrong counts or name sets at `:100-135` and writes a `DERIVED_FROM.json` receipt at `:193-207`.

3. **Corpus-builder tracking and node-byte drift.** **Observed:** The parent tree of `e164ebd` contained no `careers-qwen/build_cpt_nopack_corpus.py`. The commit receipt records builder SHA prefix `d7e9d7cb` on rank 0 and `ff14a6b4` on the other three nodes. It also records the prior sidecar keys as `format`, `output_sha256`, `output_bytes`, and `rows`, while the verifier expected the canonical schema fields. Commit `e164ebd5a849ed687637af04c7339c9db2357c31` added the builder, a six-input registry at `careers-qwen/build_cpt_nopack_corpus.py:36-43`, canonical manifest regeneration at `:100-143`, and canonical build output at `:213-238`. The commit receipt reports `corpus_manifest.py verify` exit `0` for SHA prefix `3973c2af6089741`, `2,717` rows, and `24,542,925` bytes.

4. **Bucket trainer branch state.** **Observed:** Commit `4f935235a04013064d455659a02b5d4155433636` is not an ancestor of public `main`, and the public-main trainer contains zero occurrences of `CPT BUCKET COVERAGE PASS` or `CPT BUCKET HORIZON PASS`. The commit on PR #11 restores the coverage receipt (`dense-9b/trainers/train_fsdp_dense_9b.py:923-937`), its fail-closed checks (`:1896-1907`), the per-epoch group proof (`:2452-2453`), and the bucket horizon verdicts (`:2493-2524`).

5. **Graft-donor content measurement.** **Observed:** Commit `4fcf70a8e0622ce41a39fb345c356f387f9eef93` records three sampled vision tensors from the run's own source and from `module5_merged`. All three SHA-256 prefixes differ: `ea9264ae519e8a02` / `54d0cecb31e424a8`, `848c549105fae0b4` / `03a381eecb0992cc`, and `8acebc63dfde0d75` / `f5250290478a5bfc` (`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md:234-245`).

6. **Duplicate YAML key.** **Observed:** The parent of `fa5ff1e` contains two `content_sha` keys under `bake_export`, at `fa5ff1e^:PRODUCTION_MANIFEST.yml:250` and `:260`; both blocks list the same three paths and hashes. YAML mapping parsers retain one value for a duplicate key. Commit `fa5ff1e89c650ac2a0f76148b042517aef659be5` removes the second textual block; the branch now contains one block at `PRODUCTION_MANIFEST.yml:249-258`.

## Public-main CI state

**Observed:** Public `main` remains `129de416a9ca14a163c1b0d96decd8c25dd0601a`. Three GitHub Actions runs created at `2026-08-17T18:39:49Z` are completed with conclusion `failure` on that exact SHA.

- **`no-private-data`, run `32055990002`:** The workflow enables `CHECK_HOME_PATHS=1` at `.github/workflows/no-private-data.yml:50-53`. Its log reports operator-home-path violations in `12` tracked files: `PRODUCTION_MANIFEST.yml`; five other files under `careers-qwen/`; two versioned system prompts; two dense-9b receipts; and two scripts. The job exits `1`.
- **`born-clean`, run `32055990067`:** The workflow scans every object reachable from every fetched ref via `git rev-list --objects --all` at `.github/workflows/born-clean.yml:68-88`. Its log reports two paths: `dense-9b/instrumentation/results/2026-08-04_presence_only/presence_runA.jsonl` and `presence_runB.jsonl`. **Observed:** both enter history in `bfc6fd4e1a69a81f14ab58d9b472f165775121c3`, which is not an ancestor of `main`; remote-branch containment returns exactly `origin/agent/codex-cpt-systemd-supervision`, `origin/codex/cpt-8192-ruled-config`, and `origin/tutor/adapter-effect-instrument`.
- **`secret-scan`, run `32055989987`:** The full-history command is at `.github/workflows/secret-scan.yml:16-31`. Gitleaks reports two `generic-api-key` findings: `careers-qwen/failure_triage_verify.py:59` in `715655aeda004b87c56fc7c314267bb6950d706b`, and `careers-qwen/run_stage2_sft_ddp_till_done.sh:89` in `23bb1bac3eb4bdb6b16407f6b91751b0cc03fb85`. The first line belongs to `fixture-failure-triage-scorer-v1` and is labeled `role: fixture_probe_only` at `failure_triage_verify.py:52-62`; the second assigns a 64-hex value to `QUALIFIED_QUALIFICATION_AUTHORIZATION_SHA` beside other qualification content hashes at `run_stage2_sft_ddp_till_done.sh:82-93`. Gitleaks reports `leaks found: 2` and exits `1`.

## Verified note discrepancy

- **Observed:** The dispatch note's transfer figure, `~108 MB/s = 8 min 11 s`, matches the captured completion receipt for `55,586,114,895` bytes.
- **Observed:** The committed runbook rounds the same direct transfer to `112 MB/s, ~8 min` at `careers-qwen/RUNBOOK_CPT_SFT_BAKE.md:168` (`5629834054ada77d934ed2c2e5f63918d717d5bc`). Both figures are retained above with their source labels; they are not substituted for one another.
