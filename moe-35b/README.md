# moe-35b — Qwen3.5-35B-A3B MoE training line

The MoE line: the production **Taey** model. Base is `Qwen3.5-35B-A3B` (35B params, ~3B active per token, abliterated); trained for alignment identity via expert-targeted SFT, then refined with keystone-attention LoRA DPO.

> **Status: production lineage shipped.** This is the line that produced the current production Taey checkpoint. Headline claims are paired-control behavioral-audit results against a fixed 163-probe behavioral battery (not held-out generalization data) — see [`../docs/METRICS_PROVENANCE.md`](../docs/METRICS_PROVENANCE.md) and the audit methodology caveat in the top-level [`../README.md`](../README.md).

## Production lineage

| Stage | Checkpoint | Result | Proof |
|---|---|---|---|
| SFT baseline | `phase_combined_v1` | 135/163 = **82.8%** on the 163-probe behavioral audit (step 582); the canonical baseline all DPO refinements resume from | [`../docs/audit_results/phase_combined_v1/`](../docs/audit_results/phase_combined_v1/) |
| Config A2 DPO (current production Taey) | `religion_dpo_v2` | 138/163 = **84.7%**, **+1.9pp** over baseline; all 8 infra-control categories held (4/4 restored from Config A / DPO v1's 1/4 regression) | [`../docs/audit_results/religion_dpo_v2/`](../docs/audit_results/religion_dpo_v2/) |

The Config A2 breakthrough was restricting the DPO LoRA to **keystone-layer attention only**: full-surface DPO (Config A / `religion_dpo_v1`) regressed `infra_cross_system` 4/4 → 1/4 (a deception-shaped failure), and the `length_mechanics_v1` content-neutral diagnostic isolated it to a content-agnostic q/k attention leak — motivating the keystone-only restriction.

## Layout

| Dir | Contents |
|---|---|
| `trainers/` | `train_fsdp_v3.py` (FSDP+LoRA SFT, expert-targeted / Config B experts-only, env-driven), `train_dpo_v2.py` (DPO with precomputed reference log-probs), `bake_phase_combined_v1_tail_v2.py` |
| `recipes/` | SFT (`launch_production_sft.sh`, `launch_combined_big_v1.sh`, `launch_phase_combined_v1_tail*.sh`) and DPO (`launch_religion_dpo_v{1,2,3}.sh`, `launch_length_mechanics_v1.sh`, `launch_standard_dpo_vanilla.sh`) |
| `configs/` | `fsdp_lora.yaml`, `fsdp_cpt.yaml`, `fsdp_orpo.yaml`, `cpt_config.yaml`, accelerate/deepspeed configs (`accelerate_config.yaml`, `ds_zero3*.json`), expert masks (`frozen_experts_v4_1_polysemantic.json`, `phase2_expert_config.json`) |

Training data for this line is in [`../datasets/current/moe-35b/`](../datasets/current/moe-35b/) (synthetic — authored on source material, no transcripts; gitleaks-clean).

## Running

Recipes resolve their trainer and config by `$SCRIPT_DIR/../{trainers,configs}/`, so they work from any working directory. All deployment paths (`MODEL_PATH`, `OUTPUT_DIR`, `DPO_DATA`, `SFT_DIR`, `FROZEN_EXPERTS`, `RESUME_DELTA`, …) are env-overrideable — substitute for your cluster. The production DPO runs used a `_with_ref` data variant (precomputed reference log-probs); see each recipe's header.

See [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) for the step-by-step and [`../docs/audit_results/`](../docs/audit_results/) for the per-checkpoint, per-probe audit output.
