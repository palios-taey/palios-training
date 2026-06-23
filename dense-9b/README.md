# dense-9b — Qwen3.5-9B Dense training line

The dense line: continued pre-training (CPT) and recovery SFT on **Qwen3.5-9B (dense, not MoE)**, targeting a tool-calling-capable aligned model on the 4-node DGX Spark GB10 cluster.

> **Status: IN PROGRESS — not yet production-validated.** Unlike the MoE line (which has a shipped production checkpoint — see [`../moe-35b/`](../moe-35b/)), the 9B dense tool-calling model has **no clean end-to-end run that has shipped.** The prior tool-use SFT runs were compromised (a chat template asserted the XML wire format at inference time, overriding any training-time format, and the tools-grounding was off). The fixes are in place — the canonical Hermes tool template, the tools-grounding corrections, and a prelaunch format gate — but we have **not** observed a clean, audited end-to-end tool-calling run. Do not read the artifacts here as a validated capability claim; they are the working recipe surface, honestly labeled as not-yet-proven.

## Layout

| Dir | Contents |
|---|---|
| `trainers/` | `train_fsdp_dense_9b.py` (full-FT FSDP, env-driven, bucket batching; CPT vs SFT mode auto-selected by env), `build_training_data.py` (explicit-input dataset builder), `train_cpt_qwen35_dense.py`, `train_recovery_sft_qwen35_dense.py` (Phase 3 recovery SFT, pre-composes + chunks its own corpus), `chunk_corpus_offline.py` (offline conversation chunker — the wedge-fix for long multi-turn items on 4-Spark FSDP) |
| `recipes/` | `launch_cpt_phase2_qwen35_9b_fsdp.sh`, `launch_fsdp_orchestrator_cpt_v0.sh`, `launch_sft_tools_qwen35_9b_fsdp.sh`, `launch_phase3_sft_single_spark.sh`, `launch_diagnostic_2x2.sh` |
| `configs/` | `fsdp_dense_9b.yaml` (per-`Qwen3_5DecoderLayer` FSDP wrap), `cpt_cluster.env.example` (public-safe sample env) |
| `inference/` | `qwen3.5-tooluse.jinja` (canonical tool template), `toolcall_format_gate.py` (prelaunch format gate) |

## Running

Recipes resolve their trainer and config by `$SCRIPT_DIR/../{trainers,configs}/`, so they work from any working directory. The CPT path is fail-loud: `MODEL_PATH`, `CPT_DATA`, `OUTPUT_DIR`, `TOTAL_STEPS`, and node addressing must be set explicitly. Start from [`REPRODUCE.md`](REPRODUCE.md) and a local copy of [`configs/cpt_cluster.env.example`](configs/cpt_cluster.env.example):

```bash
source /path/to/filled-cpt-cluster.env
bash recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh
```

The canonical dense CPT optimizer is Adafactor at `LR=2e-5` with `scale_parameter=False`, `relative_step=False`, `warmup_init=False`, `clip_threshold=1.0`, and manual `LambdaLR` linear warmup then linear decay. AdamW is intentionally not the default because it OOMs on the GB10 UMA page-cache regime.

## What is real here vs not

- `[Observed]` Phase 2 CPT produced `checkpoint-2400` and `SaveSafeTrainer` survived the low-memory regime — see [`../docs/audit_results/cpt_qwen35_9b_v1_epoch1/`](../docs/audit_results/cpt_qwen35_9b_v1_epoch1/).
- `[Observed]` The offline chunker resolved the 4-Spark FSDP wedge on long multi-turn corpora (the mechanism and cross-validation are documented in [`../docs/`](../docs/) and the project record).
- `[Unknown / In progress]` A clean, audited, end-to-end tool-calling 9B checkpoint. The recipe + template + gate are in place; no validated run has shipped.

See [`../docs/METRICS_PROVENANCE.md`](../docs/METRICS_PROVENANCE.md) for the claim→file mapping and [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) for the step-by-step.
