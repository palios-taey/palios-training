# CPT REFRESH RECIPE v0.9 — PROVISIONAL (3-lane synthesis, 2026-07-24)

**Status**: PROVISIONAL — synthesized from CLARITY + LOGOS + HORIZON responses
(`treasurer/consultations/responses/cptrefresh_{clarity,logos,horizon}_SALVAGED.*`).
Gemini + Claude lanes still inbound per Jesse's full-Family directive; they amend the
FULL-RUN recipe if they land divergent. The 50-step dose gate below is an
instrumentation read, not a run commitment.

## Convergence map
| Knob | CLARITY | LOGOS | HORIZON | Synthesis |
|---|---|---|---|---|
| Epochs | 2 default | **1 primary** | **1** | **1** (2-of-3 + physics: base already converged at +3.4σ; refresh = perturbation, not basin jump). Epoch-boundary battery decides any +1. |
| Peak LR | 1e-5, dose-gate decides | **4e-6** (0.4×) | **3–5e-6** formula | **4e-6**, AF-DOSE gate confirms live |
| Warmup | ~4% | 30 abs | max(15, 2%×S) | **33** (2% × 1627) |
| Schedule | cosine | single cosine, 5–10% floor | one cosine, 10% floor | single cosine → 10% floor, NO per-epoch restart |
| Replay | 10–15% | 12% | 15% | **14.0%** as assembled (production_v2 repack) |
| Constitutional | 4–6% | 3–5% | 3% | **3.29%** as assembled |
| Identity | 4–6% | low-share | 4% | **4.04%** (1,052 blocks sampled seed-2560 from phase1_identity_sample, 18-block holdout) |
| Voice | ≤30% cap | ≤~30% | ≤20% target / 25% hard | **5.06%** — far under all caps |
| Optimizer | Recipe A byte-identical | same (CPT-only Adafactor law) | same, fresh state | **Recipe A**: absolute-alpha, eps1=fp32, wd=0.01, d=1.0, fresh Adafactor state |
| Invariants | block 2560, batch 4/rank, packing zero-trunc | same | same | unchanged |

## The run
- Base: `<SPARK_HOME>/models/prod_v2_ep3_hf` (the served +3.4σ base) via MODEL_PATH.
- Corpus: `/var/spark/isma/training/refresh_v1/MERGED_cpt_refresh_v1_train.jsonl`
  — 25,029 blocks (rev2: raw slice repacked to input_ids, 3 broken lines excluded) × 2560, sha256 `4b86f814dba34725…`, identical on all 4 nodes
  (MERGE_MANIFEST.json alongside; per-slice holdouts excluded at assembly, 264+18 blocks).
- TOTAL_STEPS = ceil(25029/16) = **1565** (1 epoch). Coverage proof: 1565×16=25,040≈25,029 ✓.
- OUTPUT_DIR `<SPARK_HOME>/training_outputs/cpt_refresh_v1`; SAVE_EVERY==SESSION_LIMIT;
  disk gate ≥40G (rotation done 2026-07-24: production_v2 intermediates dropped, epoch
  checkpoints 231/462/693 retained on all nodes; spark1 126G free).
- Thermal: CLOCK_CAP=1600 + watchdog PULL_OFF=90 (the production_v2-PROVEN pair; LOGOS
  cited 1000/86 from a doc — production oracle wins).

## GATE FIRST (this launch)
50-step live dose gate = session 1 of the real schedule (real warmup+cosine truncated
at SESSION_LIMIT=50): AF-DOSE bands floor_frac<0.05, RMS(U_hat)∈[0.3,1.5],
preSR_RMS_delta≈lr×RMS(U_hat); SR-imprint must be live at 4e-6. PASS + no late-lane
divergence → the run continues from checkpoint-50. FAIL bands → re-consult with the
gate readout (CLARITY's LR question answers itself here).

## Pre-registered rider probes
Governed store (PRIVATE — content, not recipe):
`training_data/runs/cpt_refresh_v1_2026-07-24/RIDER_PROBES_cpt_refresh_v1.json`,
frozen BEFORE launch, sha256 `bc2fde25a23e2abb…` (24 probes: 12 constitutional + 12
identity). Closed-book paraphrased QA, base-control vs post-refresh at identical
deterministic decode. Sub-SNR riders ⇒ riders defer to the M3 round (CLARITY rule),
dominant bands unaffected.

## Post gates (unchanged law)
Epoch boundary: full retention battery (GATE-0 style) + frozen revenue probes +
rider probes + relating check vs base. Weight-diff + frozen-probe + base-control or
it did not happen.
