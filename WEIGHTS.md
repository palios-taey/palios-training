# Baked weights

The trained checkpoint for the canonical `phase_combined_v1` SFT bake is published on Hugging Face:

**→ [`palios-taey/Taey-35B-A3B`](https://huggingface.co/palios-taey/Taey-35B-A3B)** (Apache-2.0)

## What it is

Expert-selective SFT (Config-B ESFT, keystone layers `[8,9,11,15,21,23]`) of
[`huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated)
(an abliterated Qwen3.5-35B-A3B MoE, ~3B active). This is the SFT baseline (step 582 over
`combined_v1_gated.jsonl`) that the recipes in this repo reproduce. A from-only-this-repo
reproduction lands at the same keystone-expert weight-depth (≈0.36 deviation from base).

## Serving (read this — it matters)

```bash
vllm serve palios-taey/Taey-35B-A3B --trust-remote-code --max-model-len 16384
```
- **Do NOT pass `--reasoning-parser`** — the model emits reasoning inline in `content`
  (`<think>…</think>`); a reasoning-parser empties the content field. Strip `<think>` before display.
- **Sampling: `temperature≈1.0, top_k=20, top_p=0.95`** (the model's recommended config).
  Temperature-only sampling can cause repetition loops / language drift on long generations.

## Evaluation

**135/163 = 82.8%** on the in-house 163-probe behavioral battery (`audit/`). This is a
**self-graded, in-house** audit (probes and training corpus authored by the same team;
LLM-as-judge), **not** a held-out generalization benchmark — read it as a methodology, not a
transferable score. The complete per-probe results (every response + score, including the 27
BETRAYED — identity-drift and religion-miracle hedging among them, disclosed not hidden) are in
[`docs/audit_results/phase_combined_v1/`](docs/audit_results/phase_combined_v1/). An independent
re-judge of the published responses is stricter than the in-house auditor, especially on those
weak categories.

## Connecting it to the ecosystem

- **Serving + persona/tool layer:** [`palios-taey/taey-presence`](https://github.com/palios-taey/taey-presence)
  (the `soma_proxy` wraps the bare model with the system prompt + tool routing).
- **Retrieval:** [`palios-taey/isma-core`](https://github.com/palios-taey/isma-core).

The base model (`Qwen3.5-35B-A3B`) is already public; with the data + recipes in this repo, a
third party can retrain from base — the weights release just removes the retrain step.
