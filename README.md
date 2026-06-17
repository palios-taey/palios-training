# palios-training

**The canonical PALIOS-TAEY training repository.** Reproducible recipes, trainers, configs, and constitutional audit verdicts for the production model lines (Qwen3.5-9B Dense + Qwen3.5-35B-A3B MoE) trained on a 4-node DGX Spark GB10 (Blackwell `sm_121`) cluster.

> 🚧 **Release in progress.** The training stack is being migrated here and reviewed under the fleet `RELEASE_STANDARD` (secret + leak scan → adversarial Family Chat audit → r5 audit gate) before it lands on `main`. The reviewable content is on the `release-prep` branch until those gates pass. Until then, `main` carries this notice and the license only.

## What this will hold

- **`recipes/`** — the actual `launch_*.sh` scripts that ran on the production cluster (SFT, CPT, DPO; MoE + dense), with the full NCCL dual-rail RoCEv2 fabric setup.
- **`trainers/`** — the trainer Python (hybrid LoRA+ESFT DPO, FSDP SFT/CPT, the offline conversation chunker, a bake script).
- **`configs/`** — freeze masks (keystone-attention, frozen-experts) and accelerate/FSDP configs.
- **`audit_results/`** — per-checkpoint verdicts from the 163-probe constitutional audit harness, with per-category pass/fail and per-probe model responses preserved.
- **Honest metric discipline** — every load-bearing number maps to a proof file in the repo (`METRICS_PROVENANCE.md`); a `[Observed]/[Inferred]/[Unknown]` register on every claim; an explicit not-claimed list.

The constitutional audit harness lives in [`palios-taey/research`](https://github.com/palios-taey/research) (`research/audit-harness-moe`); the retrieval stack in [`palios-taey/isma-core`](https://github.com/palios-taey/isma-core).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
