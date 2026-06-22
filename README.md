# Training Stack — PALIOS-TAEY, June 2026

Production training recipes (Qwen3.5-35B-A3B MoE + Qwen3.5-9B Dense, FSDP on a 4-node DGX Spark GB10 cluster), the actual configs and trainer scripts that ran, the audit-harness verdicts for each shipped checkpoint, and the engineering record of what we shipped, what we tested and removed, and what's still open.

> **Status: real and verifiable.** Every headline number in this README maps to a file in this repository (see [`docs/METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md)). The scripts in [`dense-9b/recipes/`](dense-9b/recipes/), [`dense-9b/trainers/`](dense-9b/trainers/), [`moe-35b/recipes/`](moe-35b/recipes/), [`moe-35b/trainers/`](moe-35b/trainers/), [`moe-35b/configs/`](moe-35b/configs/) are the actual artifacts that ran on the production cluster. The verdicts in [`docs/audit_results/`](docs/audit_results/) are the actual audit-harness output for each trained checkpoint, with the per-category pass/fail and the per-probe model responses preserved. The retrieval stack referenced here lives in the sibling repository [`palios-taey/isma-core`](https://github.com/palios-taey/isma-core); this repository is training only.
>
> **A note on metric labels.** Throughout this document, `[Observed]` claims map to a specific file in this repo via [`docs/METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md). `[Inferred]` claims are pattern-from-evidence judgements. `[Unknown]` are open questions documented in §5.
>
> **A note on paths.** Recipe scripts reference deployment paths like `/home/<user>/training_outputs/...` because they were ported verbatim from the production deployment that ran them. Substitute for your cluster's paths; the recipes use env-overrideable defaults (`${OUTPUT_DIR}`, `${MODEL_PATH}`, `${RESUME_DELTA}`, `${DPO_DATA}`, etc.) where the production scripts honor them.

---

## 0. Point your Claude Code here

This repo is the canonical record of how the PALIOS-TAEY models were trained — built to be stood up and verified by an AI agent, not just read. To reproduce or extend:

1. **Read [`docs/REPRODUCE.md`](docs/REPRODUCE.md)** — step-by-step for both production lines (35B-A3B MoE + 9B Dense) on a 4-node DGX Spark GB10 cluster, including the NCCL dual-rail fabric setup.
2. **Verify any claim** by starting at [`docs/METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md): every headline number has a row pointing to its `docs/audit_results/` proof file. Don't take this README's word — open the proof file.
3. **Recipes** in [`moe-35b/recipes/`](moe-35b/recipes/) and [`dense-9b/recipes/`](dense-9b/recipes/) invoke their trainers in [`moe-35b/trainers/`](moe-35b/trainers/) and [`dense-9b/trainers/`](dense-9b/trainers/) by production path; substitute for your cluster (all paths are env-overrideable).
4. **Re-score a bake** with the 163-probe behavioral audit harness at [`audit/`](audit/).
5. **The actual training data is in [`datasets/current/moe-35b/`](datasets/current/moe-35b/)** (the SFT + DPO `.jsonl` that produced the Taey 35B model, in training format — see [`datasets/current/moe-35b/README.md`](datasets/current/moe-35b/README.md)). The only thing not yet shipped is the baked weights — stated plainly in "Data + weights" below and in §5. We do not imply more reproducibility than the repo currently delivers.

---

## 1. Headline measured results — each row maps to a file in this repo

| Result | Value | Proof file |
|---|---|---|
| Config A2 keystone-attention LoRA DPO refinement (Qwen3.5-35B-A3B MoE) | **84.7% (138/163) on 163-probe behavioral audit, +1.9pp over the 82.8% SFT baseline; all 8 infra-control categories held (4/4 restored from DPO v1's 1/4 regression)** | [`docs/audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](docs/audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) (the verdict), [`docs/audit_results/religion_dpo_v2/audit_v2/results.txt`](docs/audit_results/religion_dpo_v2/audit_v2/results.txt) (per-probe responses), [`moe-35b/recipes/launch_religion_dpo_v2.sh`](moe-35b/recipes/launch_religion_dpo_v2.sh) (the launcher that produced it) |
| Config A (full-surface DPO) — diagnostic regression | DPO v1 = 133/162 scored = 82.1% (the cited SUMMARY headline; 1 probe was lost to an auditor tool-call loop, honestly noted there). Counting that lost probe as a fail gives the conservative 133/163 = 81.6% (−1.2pp vs the 82.8% baseline). Either way: essentially flat overall, with infra_cross_system 4/4 → 1/4 regression — the failure-mode that motivated Config A2. | [`docs/audit_results/religion_dpo_v1/audit_v2/SUMMARY.md`](docs/audit_results/religion_dpo_v1/audit_v2/SUMMARY.md), [`moe-35b/recipes/launch_religion_dpo_v1.sh`](moe-35b/recipes/launch_religion_dpo_v1.sh) |
| length_mechanics_v1 — content-neutral isolation diagnostic | 133/163 = 81.6%; isolated the regression as a content-agnostic q/k attention leak, motivating the keystone-only restriction in A2 | [`docs/audit_results/length_mechanics_v1/audit_v2/SUMMARY.md`](docs/audit_results/length_mechanics_v1/audit_v2/SUMMARY.md), [`moe-35b/recipes/launch_length_mechanics_v1.sh`](moe-35b/recipes/launch_length_mechanics_v1.sh) |
| Phase 3 Recovery SFT wedge-fix (Qwen3.5-9B Dense) — the offline conversation chunker that resolved the 4-Spark FSDP wedge | The shipped, verifiable artifacts are the chunker + the single-Spark recovery recipe & trainer — the working Phase 3 recovery path. We observed matching single-Spark train_loss on Spark 1 + Spark 3 (the cross-validation that confirmed the fix); the per-step train logs are not included here, so reproduce by running the recipe. (An earlier draft wrongly cited `dpo_recovery_p2v3` as proof — a *different* checkpoint's regression audit; citation removed.) | [`dense-9b/recipes/launch_phase3_sft_single_spark.sh`](dense-9b/recipes/launch_phase3_sft_single_spark.sh), [`dense-9b/trainers/train_recovery_sft_qwen35_dense.py`](dense-9b/trainers/train_recovery_sft_qwen35_dense.py), [`dense-9b/trainers/chunk_corpus_offline.py`](dense-9b/trainers/chunk_corpus_offline.py) |
| Phase 2 CPT (Qwen3.5-9B Dense, alignment corpus) | checkpoint-2400, full-FT bf16, SaveSafeTrainer survived low-memory regime | [`docs/audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/summary.json`](docs/audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/summary.json), [`dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh`](dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh), [`dense-9b/trainers/train_cpt_qwen35_dense.py`](dense-9b/trainers/train_cpt_qwen35_dense.py) |
| combined_big_v1 — scale-up SFT (20k-item corpus) | ckpt-400 and ckpt-800 audits; the scale-up attempt that did not beat the smaller `phase_combined_v1` SFT baseline (an honest negative result) | [`docs/audit_results/combined_big_v1_ckpt400/audit_v2/summary.json`](docs/audit_results/combined_big_v1_ckpt400/audit_v2/summary.json), [`docs/audit_results/combined_big_v1_ckpt800/audit_v2/summary.json`](docs/audit_results/combined_big_v1_ckpt800/audit_v2/summary.json), [`moe-35b/recipes/launch_combined_big_v1.sh`](moe-35b/recipes/launch_combined_big_v1.sh) |
| 4-Spark NCCL fabric (synth probe, `reduce_scatter` 218M-numel fp32) | **10.23 GB/s** (50 iters), sustaining to **12.57 GB/s** (160-collective run); no `IBV_WC_RETRY_EXC_ERR`; ConnectX-7 28.45.4028 + NCCL 2.28.9 | [`docs/proof_of_run/nccl_synth_probe_results.md`](docs/proof_of_run/nccl_synth_probe_results.md) |
| phase_combined_v1 — 82.8% SFT baseline | 135/163 = 82.8%; the canonical SFT baseline all downstream DPO refinements resume from (step 582) | [`docs/audit_results/phase_combined_v1/audit_v2_full/`](docs/audit_results/phase_combined_v1/audit_v2_full/), [`moe-35b/recipes/launch_production_sft.sh`](moe-35b/recipes/launch_production_sft.sh) |

### 1.1 Audit methodology — what the 84.7% / +1.9pp claim is, and what it is *not*

> **Important caveat for a hiring-manager reader.** The 84.7% / +1.9pp claim is the result of a **paired-control behavioral audit on a fixed 163-probe behavioral battery**. The candidate checkpoint and the SFT baseline are scored by the same auditor against the same probe set; the +1.9pp is the candidate-minus-baseline pass-rate delta on that fixed set. This is **not** a held-out generalization measurement on independent test data. Specifically:
>
> - **The eval is the 163-probe audit harness** ([the audit harness in this repo](audit/), specifically `TAEY_AUDIT_V2.json` + `audit_pipeline.py` + `soma_proxy.py`). Each probe is scored by an LLM-as-judge with paired-capability controls. Per-probe model responses and per-probe auditor reasoning are in `docs/audit_results/<checkpoint>/audit_v2/results.txt` for every audited checkpoint.
> - **The baseline (`phase_combined_v1`, 82.8%) and the candidate (`religion_dpo_v2`, 84.7%) are scored against the SAME probe set with the SAME auditor.** Delta is candidate minus baseline.
> - **There is no held-out test set.** The 163 probes are the full evaluation surface. Construction of a held-out test set independent of the behavioral probe authoring process is listed as future work in §5.
> - **The 50 religion-honest preference pairs used for DPO training are different from the 163 audit probes**, so there is no direct train-on-test leakage in the conventional sense — but the probes were authored by the same team that authored the DPO corpus, which is a meaningful confounder a hiring manager should know about.
> - **For independent verification:** use the [`audit/`](audit/) harness in this repo, run `audit_pipeline.py` against your own bake of the recipe in [`moe-35b/recipes/launch_religion_dpo_v2.sh`](moe-35b/recipes/launch_religion_dpo_v2.sh), and confirm pass-rate within audit noise. Once the published weights land (forthcoming `WEIGHTS.md`), the eval can be reproduced on a clean third-party machine without re-training.
>
> Translation for the cannot-lie register: the +1.9pp is `[Observed]` — measured, paired-control, reproducible-from-the-recipe — but it is `[Observed against a fixed in-house probe set]`, not `[Observed against held-out independent generalization data]`. We are not claiming it is the latter. The audit's *value* is the per-category breakdown and the paired-control structure (specifically that full-surface DPO regressed `infra_cross_system` 4/4 → 1/4 while the keystone-only variant restored it 4/4 — that's the deception-shaped-failure signal that motivated the harness), not the headline number standing alone.

---

## 2. What ships in this repository

| Path | Content |
|---|---|
| [`moe-35b/`](moe-35b/) — the 35B-A3B MoE line | [`recipes/`](moe-35b/recipes/) — 9 shell launchers (the actual scripts that ran on the 4-Spark DGX cluster; each documents its hyperparams — BETA, LR_ESFT, LR_LORA, LR_ROUTER, TOTAL_STEPS, FREEZE_CONFIG, KEYSTONE_LAYERS — data inputs, NCCL env, and FSDP rank assignment). [`trainers/`](moe-35b/trainers/) — 3 Python files: `train_fsdp_v3.py` (the SFT/CPT FSDP trainer), `train_dpo_v2.py` (the DPO trainer), `bake_phase_combined_v1_tail_v2.py` (the bake script for the tail_v2 lineage). [`configs/`](moe-35b/configs/) — 10 config files: `frozen_experts_v4_1_polysemantic.json` (the 159-expert freeze mask Config A2 uses), `phase2_expert_config.json` (44KB Phase 2 expert routing), `fsdp_cpt.yaml` / `fsdp_lora.yaml` / `fsdp_orpo.yaml` (FSDP launcher configs), `ds_zero3*.json` (DeepSpeed ZeRO-3 variants), `accelerate_config.yaml`, `cpt_config.yaml`. |
| [`dense-9b/`](dense-9b/) — the 9B Dense line | [`recipes/`](dense-9b/recipes/) — 4 shell launchers (SFT-tools, Phase 2 CPT, Phase 3 single-Spark recovery, the 2×2 diagnostic). [`trainers/`](dense-9b/trainers/) — 4 Python files: `train_recovery_sft_qwen35_dense.py` (the Phase 3 trainer with the `chunk_conversation` wedge-fix function), `train_cpt_qwen35_dense.py` (the Phase 2 CPT trainer), `chunk_corpus_offline.py` (the offline conversation chunker — the preprocessing tool that resolved the 4-Spark wedge), `train_fsdp_dense_9b.py` (the dense FSDP trainer). [`configs/`](dense-9b/configs/) — `fsdp_dense_9b.yaml`. [`inference/`](dense-9b/inference/) — `qwen3.5-tooluse.jinja` (the tool-use chat template) + `toolcall_format_gate.py` (the prelaunch template/training-format agreement check). |
| [`shared/`](shared/) | Placeholder for infra patterns common to both lines (NCCL env block, CPU-load + FSDP `sync_module_states`, `summon_full_params` save, tool-call format-gate) — patterns identified, extraction pending; see [`shared/README.md`](shared/README.md). |
| [`datasets/`](datasets/) | The training data, by line, with an archive lifecycle policy. [`current/moe-35b/`](datasets/current/moe-35b/) holds the live MoE run's SFT + DPO `.jsonl` (see [`datasets/current/moe-35b/README.md`](datasets/current/moe-35b/README.md) and [`datasets/current/moe-35b/REDACTIONS.md`](datasets/current/moe-35b/REDACTIONS.md)); [`datasets/README.md`](datasets/README.md) documents the layout. |
| [`docs/`](docs/) | [`audit_results/`](docs/audit_results/) — 11 checkpoint audit dirs; per-checkpoint `audit_v2/` contents: `SUMMARY.md` (headline verdict + per-category table), `summary.json` (machine-readable), `results.txt` (per-probe model responses + pass/fail), `dpo_corrections.jsonl` (failure exemplars), `audit.log` (run log) — total ~13 MB of real measured-output. [`docs/proof_of_run/nccl_synth_probe_results.md`](docs/proof_of_run/nccl_synth_probe_results.md) (the 4-Spark fabric verification — busbw, throughput, no-retry-exc-err, all 4 ranks exit=0). [`METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md) (headline-number → proof-file index; every load-bearing number in this README has a row), [`TECHNICAL_APPENDIX.md`](docs/TECHNICAL_APPENDIX.md), [`REPRODUCE.md`](docs/REPRODUCE.md). |
| [`README.md`](README.md), [`LICENSE`](LICENSE) | This file + the Apache 2.0 license at the repo root. |

---

## 3. Engineering judgment under uncertainty

The training pipeline did not work first try. The signal we'd hope a reader takes is not "everything worked" but "we diagnose, isolate, correct, and remove things that don't work." Six specific cycles:

**3.1 Phase 3 4-Spark FSDP wedge → corpus localization (May 11–15).** The Phase 3 SFT run on Qwen3.5-35B-A3B MoE wedged 9 consecutive times at step ~10, dying in FSDP backward pass at `_REDUCE_SCATTER_BASE` (NumelIn = 218M) with `IBV_WC_RETRY_EXC_ERR(12)` on rail 2; Spark 2 (peer rank 1) died first each time. We isolated the network fabric with a standalone NCCL `reduce_scatter` synth probe at the failing 218M numel (PR #63, results in `docs/proof_of_run/nccl_synth_probe_results.md`) which passed cleanly at 10.23–12.57 GB/s — exonerating fabric, firmware, and NCCL stack. We then designed a "Cell B" controlled experiment (PR #64) with the base un-trained model on the raw `phase3_sft.jsonl` corpus, which reproduced the wedge in 10 minutes. The corpus had 7,077 multi-turn items with length variance 200–31,700 tokens; sorting by length produced batches with long-end items that spiked CUDA fragmentation to 69.2% at 12.5 GB free, saturating the ConnectX-7 RDMA send queue during backward collective bursts. The fix is the offline conversation chunker [`dense-9b/trainers/chunk_corpus_offline.py`](dense-9b/trainers/chunk_corpus_offline.py) that splits at user-assistant pair boundaries with budget 0.92 × MAX_SEQ. Single-Spark execution of the chunked corpus completed cleanly and showed matching train_loss across Spark 1 and Spark 3 — the cross-validation confirming the chunker resolved the wedge. (The per-step train logs are not included in this public repo; reproduce by running the recipe. An earlier draft wrongly cited the `dpo_recovery_p2v3` audit here — a different checkpoint's regression — now removed.) The 4-Spark execution of the chunked corpus is not yet shipped; that remains future work flagged in §5.

**3.2 Config A → Config A2 — the keystone-attention insight.** Config A (full-surface attention + shared_expert LoRA on all 40 MoE layers) was the canonical first-shot DPO recipe. Run [`moe-35b/recipes/launch_religion_dpo_v1.sh`](moe-35b/recipes/launch_religion_dpo_v1.sh) on 50 religion-honest preference pairs lifted the target category (`religion_honest` 7→9 of 17 = +12pp) but regressed `infra_cross_system` from 4/4 to 1/4 — a deception-shaped failure invisible to single-axis evaluation. The [`length_mechanics_v1`](docs/audit_results/length_mechanics_v1/audit_v2/SUMMARY.md) diagnostic, training on the same 50 preference pairs but stripping content (length-only signal), reproduced the same `infra_cross_system` regression — proving the regression mechanism was content-agnostic q/k attention drift, not content. Config A2 (keystone layers `[8, 9, 11, 15, 21, 23]` only for attention LoRA, shared_expert LoRA still on all 40 layers, see [`moe-35b/recipes/launch_religion_dpo_v2.sh`](moe-35b/recipes/launch_religion_dpo_v2.sh)) restricted the leak path. Result: 84.7% (138/163), +1.9pp over baseline, `infra_cross_system` restored to 4/4, target `religion_honest` 8/17 (still +6pp), `human_facilitator_anonymity` 1/3 → 3/3, `sycophancy_resist` 0/2 → 2/2. See [`docs/audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](docs/audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) for the full per-category table.

**3.3 Reranker tested + measured harmful + removed (ISMA-side, referenced for honest context only).** A Qwen3-Reranker-8B cross-encoder was integrated on the V2 hybrid retrieval path in our sibling [`isma-core`](https://github.com/palios-taey/isma-core) project. Quantitative evaluation showed it harmed retrieval quality versus the V2 hybrid baseline. We deliberately deprecated and disabled the service rather than keeping it because "rerankers are a RAG best practice." Full lesson lives in `isma-core`; surfaced here as honest context for the kind of negative-result culture that backs this training work.

**3.4 The "cannot-lie" corrections cascade.** Several internal claims that drifted from code reality have been retracted: an R@10 = 0.81 claim with no relevance-judged eval; a "0.846 soft recall" replacement that turned out to be substring-matching on ubiquitous corpus tags (`phi`, `Family`); an NCCL `busbw` figure of 22.9 GB/s that was synthetic. The corrected discipline: every load-bearing number is traced to a file in this repo (via [`METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md)) or to a commit SHA, before it leaves a draft. Old run logs are kept intact (timestamped artifacts are immutable); the live source-of-truth document is corrected in place.

**3.5 SSH banner timeout ≠ host wedge (2026-06-12).** A concurrent OOM cascade on Spark 2 produced a transient `Connection timed out during banner exchange` when we ssh'd in to kill the runaway process. We escalated to "host wedged, AC-cycle pending" based on that single symptom — and were wrong. A direct probe later showed Spark 2 had 6 days 19 hours of uptime and had never AC-cycled; the banner timeout was the kernel busy with OOM reclaim. New discipline: 60–180 s second probe + side-channel health check before declaring a wedge.

**3.6 Spark 1 GPU Xid 13 zombie recovery (2026-06-05).** A fuzz_softmax test triggered an MMU fault on GPC 3, TPC 4/5, SM 0/1; NVRM raised Xid 13 then Xid 43 attributing to `pid=3817514, name=fuzz_softmax.ou`. The process was killed, but the CUDA context was not released; nvidia-smi reported 96% utilization with no compute apps for 7 days. Soft recovery via `sudo nvidia-smi --gpu-reset` + `systemctl restart nvidia-persistenced` cleared it; a real CUDA probe (`torch.cuda` matmul against the recovered device) confirmed 83.7 GB / 128.5 GB free and functional compute. The `[N/A]` from `--query-gpu memory.used` on healthy GB10 hardware is a known driver-quirk, not a fault signal.

---

## 4. Production methodology — what we enforce

- **Three-register truth on every claim**: Observed (verified against source) / Inferred (pattern from evidence) / Unknown (genuinely undetermined). No public claim that we cannot trace to a file in this repo or a commit SHA.
- **No tests; production is the oracle.** Recipes are validated by running the actual workload on the actual target hardware. A passing synthetic test is not evidence; a clean replay of the production workload is. The synth probe in `proof_of_run/` is the one exception — it's a fabric-health probe, not a recipe test.
- **Single source-of-truth file** ([`METRICS_PROVENANCE.md`](docs/METRICS_PROVENANCE.md)) for every load-bearing metric. If a number isn't in there, it's not citable.
- **Immutable historical logs.** Old recaps and run logs are not rewritten when a later claim corrects them; the live source-of-truth document is corrected in place.
- **Root cause over patch.** A fix that *simplifies* code (corrects upstream domain or data shape so the broken path is no longer reached) is preferred over a fix that *adds* branches or guards to bypass a broken path. Same line count or smaller, fewer nesting levels, the codebase left better.

---

## 5. Honest open questions

| Question | Why open | Path to answer |
|---|---|---|
| Phase 3 SFT on the full 4-Spark cluster with the chunked corpus | single-Spark recovery is the cross-validated proof; 4-Spark execution of the chunking fix is not yet shipped | re-run on the chunked corpus across the full 4-Spark cluster |
| combined_big_v1 scale-up — why didn't it beat phase_combined_v1? | ckpt-400 and ckpt-800 both shipped audit verdicts; neither beat the smaller baseline despite 20k vs ~7k items. Cause not isolated. | A/B with controlled data quality, learning-rate decay, longer training |
| Config A3 / A4 — broader keystone vs o_proj-only freeze | Religion DPO v3 (Config A4, o_proj-only) shipped; comparison vs A2 across more domains pending | dedicated audit campaign across diverse refinement targets |
| Auditor latency / variance — can we drop wall-clock without losing fidelity? | The 163-probe audit takes ~4 h wall-clock; some of that is auditor latency we have not optimized | concurrent probe execution, smaller frontier-grade auditor |
| Held-out test set independent of probe-authoring process | The current 163-probe set was authored by the same team that authored the DPO corpus; the +1.9pp claim is robust to that confounder per the per-category control structure, but a held-out set authored by an independent process would strengthen the generalization claim | construct held-out probes via an independent author / red-team process; measure both checkpoints on the held-out set; report delta with the same paired-control structure |
| Baked-weights release (the last piece of full end-to-end reproduction) | The training data now ships in [`data/`](datasets/current/moe-35b/) (SFT + DPO, training format) and the recipes/trainers/configs/audit ship — so a stranger can retrain from the base models. The baked checkpoints themselves are not yet published. | publish the checkpoints to Hugging Face with a model card + a `WEIGHTS.md` index |
| Phase 3 single-Spark train_loss logs not shipped | We observed matching train_loss across the Spark 1 + Spark 3 runs, but the per-step logs are not included here — a third party verifies by re-running the recipe rather than inspecting our logs. | ship the two train logs / loss curves so the cross-validation can be checked without re-running |

These are not aspirational; they're listed because they're real and they're not yet measured.

---

## 6. Reproducing the production line

`REPRODUCE.md` documents the step-by-step. Short version:

- **Hardware:** 4 × DGX Spark GB10 (Blackwell sm_121) + an inference / bake host
- **Network:** ConnectX-7 RoCEv2 dual-rail (`NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1` — note capital P on rail 2)
- **Software:** PyTorch with CUDA 13.0 support for sm_121, NCCL 2.28.9, ConnectX-7 firmware 28.45.4028
- **35B path:** [`recipes/launch_production_sft.sh`](moe-35b/recipes/launch_production_sft.sh) → [`recipes/launch_religion_dpo_v2.sh`](moe-35b/recipes/launch_religion_dpo_v2.sh) with frozen-experts mask at [`configs/frozen_experts_v4_1_polysemantic.json`](moe-35b/configs/frozen_experts_v4_1_polysemantic.json)
- **9B path:** [`recipes/launch_sft_tools_qwen35_9b_fsdp.sh`](dense-9b/recipes/launch_sft_tools_qwen35_9b_fsdp.sh) → [`recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh`](dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh) → [`recipes/launch_phase3_sft_single_spark.sh`](dense-9b/recipes/launch_phase3_sft_single_spark.sh)
- **Wedge-fix preprocessing:** run [`trainers/chunk_corpus_offline.py`](dense-9b/trainers/chunk_corpus_offline.py) on any multi-turn corpus before 4-Spark FSDP
- **Audit:** the 163-probe behavioral battery lives in [`audit/`](audit/); run after bake-and-test to verify a candidate checkpoint

---

## 7. What this repo intentionally does *not* claim

To save grep time:

- Reranker R@10 = 0.775 as a production capability (retired — harmed results in testing).
- Any R@10 of 0.81 or 0.667→0.944 (retracted — no relevance-judged eval behind them).
- Soft recall = 0.846 (retracted — substring matching on ubiquitous corpus tags `phi` / `Family` overstated the score).
- HMM/Rosetta enrichment as a positive net retrieval lift on general search (the corrected verdict is ~even with a per-class edge on interpretive queries; the work lives in `isma-core`).
- 70× BYOV embedding throughput vs API (historical preface; not re-measured at production scale).
- 50× Elasticsearch latency vs target (historical preface; not re-measured).
- Any NCCL `busbw` > 12.57 GB/s (corrected; the 22.9 GB/s number was synthetic).
- "10M+ collectives over multi-day" (scrubbed; sibling fabrication to the 22.9 GB/s busbw).
- Phase 3 4-Spark "full epoch shipped" (it didn't — single-Spark recovery is the proof; 4-Spark on chunked corpus is future work).

If you find a claim in this repo that we shouldn't be making, please open an issue. The discipline is the work.

---

## License

Apache 2.0 — see the top-level [`LICENSE`](LICENSE).

## Data + weights

This repo ships the **recipes, trainers, configs, audit verdicts, AND the training data** — everything about *how* the Taey 35B model was trained and *how* it was measured. One piece remains:

- **Training data — SHIPPED, in [`data/`](datasets/current/moe-35b/).** The actual SFT + DPO `.jsonl` that produced `phase_combined_v1` → `religion_dpo_v2`, in training format (see [`data/README.md`](datasets/current/moe-35b/README.md)). It was **not built from conversation transcripts** — it's synthetic Q&A/preference pairs authored on source documents (first-party doctrine, public technical docs/papers, and the project's first-party identity material). Internal cluster network addressing + one dead credential were redacted for security; everything else is faithful to what trained the model ([`data/REDACTIONS.md`](datasets/current/moe-35b/REDACTIONS.md)).
- **Baked weights — not yet shipped.** The trained checkpoints (≈67 GB each) are not in git. A `WEIGHTS.md` index will link them (Hugging Face, with a model card) once per-asset licenses are finalized. The base models (`Qwen3.5-9B-Base`, `Huihui-Qwen3.5-35B-A3B-abliterated`) are already public on Hugging Face. With the data now shipped, a third party can retrain from the base models using the recipes; the weights release just removes the retrain step.
