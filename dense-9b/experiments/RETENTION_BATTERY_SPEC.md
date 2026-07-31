# 27B Retention Battery — SPEC (draft v0.1, tutor)

**Status:** DRAFT for treasurer co-own + registry sign-off + Family consult. Replaces the VOID named
battery (the "332 frozen_regression rows" was a boolean-field miscount — all False; real inventories
K1=8/K2=12/K3=11 too thin). This spec defines the real battery. Governed/provenance-clean, same
discipline as the stage-2 replay benchmark.

## Purpose
Mechanically answer: does the trained 27B KNOW the corpus content **in-weights, zero-lookup**, better
than the untrained base? Per slice-type (repos / voice / careers-KB+strategy / consultations+research).
This is the epoch-boundary gate (BAKE_ARCHITECTURE_27b.md Q4) that decides PASS→next-epoch vs FAIL→stop.

## ★ Provenance-clean probes (the anti-fabrication gate)
Probes are DRAWN FROM the registered corpus via a **sha-pinned holdout manifest**, NOT hand-authored.
Each probe traces to a corpus row + its slice sha. This kills the fabrication/miscount class (the void
battery + the stage-2 memorization trap): the battery cannot contain a probe that isn't provenance-linked.

## ★ Held-out discipline (per slice — the validity gate)
"Knows it in weights zero-lookup" has two honest readings; the battery uses BOTH, labeled:
1. **Absorption (held-out generalization):** reserve a deterministic K% of rows PER SLICE (by stable
   hash) EXCLUDED from the training pack. Probe the model on the held-out content it never directly
   saw → tests whether it generalized the knowledge/style, not verbatim memorization. This is the
   honest "does it KNOW the domain" test. Requires the packer to exclude the holdout manifest (a
   one-line sha-gated exclusion — I wire it).
2. **Recall (trained-content memorization):** probe on IN-corpus content → does it reproduce specific
   facts it was trained on. Legitimate for "zero-lookup capability recall" (repos, career facts).
Report the two separately; do NOT average them.

## Per-slice-type probes (retention MEANS different things per slice — Family consult refines)
| slice type | what "retention" means | probe form | scored vs |
|---|---|---|---|
| repos (8→22) | capability recall | "what does <repo> do / how do you <capability>?" | held-out repo doc facts |
| **voice** | register/style match | generate to a prompt; does it read as Jesse? | held-out voice samples — **METRIC IS THE OPEN QUESTION (Family consult): style-similarity vs a judge** |
| careers KB / strategy | process/strategy recall | "what's the process for <X>?" | held-out KB/strategy rows |
| consultations / research | finding recall | "what did we find about <X>?" | held-out consult/recap rows |

## Verdict (Q4 mechanical — same shape as stage-2)
- **Preflight: base-vs-base** (untouched base twice, same serving stack) → per-category noise floor σ.
  A threshold below σ measures nothing.
- **Per epoch:** trained-epoch-N vs untouched-base, deterministic (greedy, thinking-frozen, bf16, same
  stack). PASS = AND of per-category inequalities:
  - **Acquisition** (target slices): score(N) − base ≥ δ, with δ > σ, on the Wilson-CI LOWER bound.
  - **No-regression** (general capability / protected): score(N) ≥ base − σ.
  - **Monotonic (N≥2):** score(N) ≥ score(N−1) − σ on targets.
- Report every delta as count-next-to-σ (e.g. "+7/40, σ≈2.1"). A noise-sized move reads as noise.

## Scoring generative recall (the hard part — Family consult)
Exact-match is too strict for generative recall; a judge introduces a scorer dependency (and we just
voided a scorer). Options to resolve with the Family: (a) key-fact containment (deterministic, cheap,
what careers-qwen/eval_probes.py already does — check salient tokens present); (b) embedding-similarity
to the reference (deterministic-ish); (c) a judge with its OWN held-out validation. Default to (a)
key-fact containment for the mechanical gate; (b)/(c) as diagnostic. Voice-style scoring specifically
needs the Family consult (what does "sounds like Jesse" mean, measurably).

## Sizing
Per slice, sized to make the acquisition estimate meaningful above σ (≥~30 probes/target-category on
the Wilson lower bound). Larger slices (repos, voice) get more.

## Adversarial verification (codex)
1. Confirm the held-out manifest was actually EXCLUDED from the training pack (the validity gate).
2. Confirm no probe is hand-authored/un-provenanced (every probe → corpus row + sha).
3. Recompute the per-category deltas + CIs independently.

## Open (Family consult, per treasurer)
- Voice retention metric (the load-bearing open question).
- Held-out fraction K% per slice (generalization test size vs training data spent).
- Judge-vs-deterministic scoring for generative recall.
- Does "retention" require beating base by δ, or also an absolute floor (e.g. repo-capability accuracy ≥ X%)?

## Dependencies (why this is a FRAMEWORK now, not a runnable battery yet)
- The **corpus v2** must be finalized first (repos 8→22, voice packed in) — the battery draws from it.
- The **held-out manifest** is built at pack time (I wire the sha-gated exclusion into pack_production_corpus.py).
- So: treasurer finalizes v2 slices → I build the holdout manifest + wire the packer → battery instantiates
  against the frozen corpus → Family consult resolves the voice metric + scoring → treasurer registers.
- Until then this is the agreed STRUCTURE; it becomes runnable when the corpus is frozen.
