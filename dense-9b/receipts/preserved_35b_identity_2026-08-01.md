# 35B identity training — found, and the single-copy risk closed

**Date**: 2026-08-01
**Seat**: tutor
**Found while**: diagnosing why rank0 (`.68`) had only 74 GB free while its peers had 852–1347 GB.

## What was found — Observed

Five complete 35B identity/constitutional LoRA runs living **only** on `.68`, under `$SPARK_HOME/`.
All five target the same base, `Huihui-Qwen3.5-35B-A3B-abliterated`:

| Run | rank | alpha | total | adapter | merged_model | checkpoints |
|---|---|---|---|---|---|---|
| `training_outputs_identity` | 16 | 16 | 78 G | 3.5 G | 67 G | 7.1 G |
| `training_outputs_identity2` | 16 | 16 | 85 G | 3.5 G | 67 G | 15 G |
| `training_outputs_direct_id` | 16 | 16 | 74 G | 3.5 G | 67 G | 3.6 G |
| `training_outputs_direct_id_r32` | 32 | 32 | 88 G | 7.0 G | 67 G | 14 G |
| `training_outputs_thinking_id` | 32 | 32 | 88 G | 7.0 G | 67 G | 14 G |

**413 GB total. Zero manifest coverage** — none of the five appears anywhere in
`treasurer/foundations/careers/training_data/PRODUCTION_TRAINED`.

**They were single-copy.** The `training_outputs_archive` directories on `.80`/`.12`/`.19`
contain only `20260731_cpt_v7_eps1fix_stage2_ddp_ckpt50` (1.2 G, the recent 27B stage-2
checkpoint) — checked directly rather than assumed from the directory name. No copy existed
on Mira.

The host filesystem was at **98% (74 G free of 3.7 T)**, with `/home` accounting for 2.4 T.

## The irreplaceable part is small

`merged_model` is base + adapter and is **regenerable** — 335 GB of the 413 GB is reproducible.
The irreplaceable artifacts are the five adapters: **25 GB total**.

The auto-generated `README.md` in each run is a stock HuggingFace TRL model card carrying **no
provenance** — base model renders as `[None]`, and no dataset or hyperparameters are recorded.
So `adapter_config.json` (base / r / alpha, above) is the only surviving run metadata. Anything
beyond that — which corpus each run consumed, what distinguishes `identity` from `identity2`,
what `direct` vs `thinking` denoted — is **Unknown from the artifacts** and would have to come
from whoever ran them or from git/plan history.

## Action taken

Replicated the 25 GB of adapters + READMEs to `.19:$SPARK_HOME/preserved_35b_identity/`
(1347 G free), preserving per-run directory structure.

**Verified by content, not by size**: `adapter_model.safetensors` sha256 MATCH on all five,
plus file-count parity (7/7 each).

```
training_outputs_identity        d29fb2f747563714…
training_outputs_identity2       7889e44cef33ddb6…
training_outputs_direct_id       f08605a5ec653d92…
training_outputs_direct_id_r32   565680ca754e5a7c…
training_outputs_thinking_id     f8c49b61a22a2502…
```

The five digests are **mutually distinct**, which is the control that makes the check meaningful:
it establishes these are five different trained states rather than five copies of one, so a
matching digest is evidence of a faithful copy and not of a degenerate comparison.

## Still open — NOT done

- **Checkpoints not replicated** (53.7 G across the five). These hold intermediate adapter
  states (`checkpoint-50` … `checkpoint-400`) and optimizer state. Secondary to the final
  adapters but not worthless; flagged rather than silently dropped.
- **Merged models not replicated** (335 G) — deliberate, they are regenerable from base +
  adapter. This is only true while the base `Huihui-Qwen3.5-35B-A3B-abliterated` remains
  available; that has not been verified here.
- **Nothing deleted.** `.68` is still at 98%. Reclaiming the 335 GB of regenerable merged
  models is the obvious relief, but deletion is not tutor's unilateral call and the base-model
  availability above must be confirmed first.
- **Not yet in the manifest.** Needs a treasurer entry — these are training *outputs*, whereas
  `PRODUCTION_TRAINED` currently indexes *corpora*, so the right home is a decision for the
  manifest's owner.

## Why this matters operationally

`.68` at 98% full is not only a preservation risk. It **blocks the 27B production CPT run**:
the window probes ran with `CHECKPOINT_DCP=0`, but a real run must checkpoint, and a DCP save
at this model size needs materially more than 74 GB. Disk on rank0 is a hard precondition for
the production run, independent of the sequence-length decision.
