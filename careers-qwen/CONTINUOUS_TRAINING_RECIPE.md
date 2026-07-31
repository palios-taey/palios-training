# Continuous small-batch training — the system and the recipe

**Status:** derived from a 5/5 Family consult (2026-07-28), every seat code-grounded with file:line
citations against the public repo. Supersedes ad-hoc per-run decisions.

**Operating model this serves (Jesse):** continuous small batches of BOTH CPT and LoRA. Modules land
often, refreshes land periodically, nothing is a one-off campaign.

---

## 1. THE SCHEDULE — one self-contained cycle per batch

**Rule: every batch gets its OWN `(WARMUP_STEPS, TOTAL_STEPS)`, sized to that batch's own step
count. A batch is a complete cosine cycle, never a slice of a shared campaign horizon.**

The old model — one long horizon consumed by many bursts (`run_till_done_v3.sh:90-91`,
`TOTAL_STEPS=693` fixed, `SESSION_LIMIT` as the burst) — is correct for a *campaign* and wrong for a
*cadence*. Under continuous operation the horizon is eventually spent, and every batch after that
runs in the annealed tail at the `0.1` floor (`_lr_lambda`, `train_fsdp_dense_9b.py:1951-1955`).
That is not hypothetical: it is what produced `cpt_refresh_v3`.

```
per batch:
  TOTAL_STEPS   = the steps THIS batch will run
  WARMUP_STEPS  < TOTAL_STEPS         (existing guard, :1942-1949)
  resumed_step  = 0                    ← a fresh cycle, not a continuation
```

**The missing guard.** The trainer blocks *warmup-too-long* and does **not** block
*horizon-already-spent*. A run resuming at a step at or past `TOTAL_STEPS` gets no ramp and no
decay — it silently executes at the floor. That case needs the symmetric check:

```python
if resumed_step >= total_steps:   # horizon spent — this batch gets no cycle
    raise RuntimeError(...)
```

**Pre-launch dose proof.** Compute `Σ f(step)` over the steps the run will actually execute, from
the same `_warmup_i`/`_total_i` the run will use, and log it before launch. A batch whose Σf is a
small fraction of a full cycle is starved *by construction* and should be caught before the 8 minutes
of setup, not after.

---

## 2. CPT AND LoRA — retrain modules forward, never merge them forward

**A LoRA adapter is a delta against ONE exact base** (`adapter_config.json:base_model_name_or_path`).
A CPT refresh therefore orphans every module trained since the previous refresh. Verified 2026-07-28:
modules 4 and 5 both declare `cpt_refresh_v3_servable`, while `cpt_refresh_v4` trained from
`prod_v2_ep3_hf` — divergent branches.

**Rule, two parts:**

1. **A production CPT refresh starts FROM the current serving merged model**, not from an older
   ancestor. Starting from an ancestor is legitimate ONLY for a controlled measurement, and such a
   run is a measurement artifact that must never be promoted.
2. **After a refresh lands, RE-TRAIN each active module against the new base** rather than merging
   the old module weights forward. A module is cheap — ~7 s/step, ~4 minutes for a 60-row 2-epoch
   module — and re-deriving is far cheaper than reasoning about whether a stale delta still applies.

**Adoption gate:** a refresh is "ready to adopt" only when every currently-served module has a
re-baked counterpart against it. Until then the refresh sits parked. `cpt_refresh_v3` parked
correctly; that was the right behaviour for the wrong reason.

---

## 3. MEASUREMENT — what the weight-diff does and does not say

**`measure_cpt_delta.py:103` computes `d = (x - y).abs()` — DISPLACEMENT, not path length.**

This resolves a discrepancy that cost hours: the schedule integral `Σ f` predicts *path length*
(total distance travelled); the weight-diff measures *displacement* (straight-line start-to-end).
They are equal only if the optimizer never changes direction. Predicted 39.1× vs measured 8.79× was
never a contradiction — it was a category error in the comparison.

**Never compare a weight-diff to an LR integral again.** If path length is wanted, it must be
accumulated during the run, not inferred from the endpoints.

Also settled: `rho_t = min(lr, 1/√step)` (`:1805`) does **not** vary at our LR — at `lr=1e-5`,
`1e-5 < 1/√step` for every realistic step, so `alpha = lr` constantly, exactly as the
`ADAFACTOR_ALPHA_MODE=absolute` comment states.

---

## 4. THE GATE — production is the oracle

**Standing rule (Jesse): production is the only oracle. No eval battery, no synthetic assessment.**
This section previously said quality was "unmeasurable" pending a battery. That was wrong, and it
was wrong in a specific way worth recording: five consulted seats independently concluded a battery
was needed, and their agreement was adopted over a standing directive. Consensus among advisors is
not authority over doctrine.

**The oracle already runs.** For every batch, the question is whether the model does the job on real
work, and that is observed on the careers lane:

| signal | who observes it | what it establishes |
|---|---|---|
| tool election on a real unit — right tool, valid arguments, no tool where none is warranted | infra behavioural gate | the model acts correctly on the surface it was trained for |
| ATS submit walk — does it advance, where does it stop | apply-machine bundles | end-to-end capability on the actual task |
| LinkedIn judgments — shape-valid, exact accessible-name targets, malformed count, latency | linkedin lane | the action surface in production |
| compose / stage-2 scoring against live jobs | treasurer | the revenue path |
| served requests and their status | infra | reachability, not quality — never confuse the two |

**A batch ships and the lane is watched.** A regression is a real unit that used to advance and now
does not, or a judgment that used to be shape-valid and now is not — observed on production traffic,
not on a probe set.

**What the weight-diff is for, and only this:** confirming a run actually trained. `FAIL_LOW` catches
a null run; the band catches an over-dose. It says weights moved, never that behaviour improved. It
is a launch check, not a quality verdict.

**The hole that remains real:** nothing *enforces* the weight-diff verdict — it prints, no pipeline
halts on FAIL. That is worth closing regardless of what the quality oracle is.

## 4b. SIZING — derive the step count from THIS run, never from another run's log

**`steps_per_epoch = ceil(dataset_size / (BATCH_SIZE_PER_RANK × n_ranks))`. Compute it from the
corpus and config in hand. Reading it off a previous run's log is how a run trains a fraction of
its corpus and reports success.**

Measured 2026-07-28. A CPT stage was launched with `TOTAL_STEPS=157` because `global_batch=16` was
read from an earlier log. That run used `BATCH_SIZE_PER_RANK=4`; this one used `1`, so the real
global batch was `4` and the real epoch was **628** steps. The 157-step run would have consumed
`157 × 4 = 628` of 2,511 blocks — **25% of the corpus** — then completed its cosine cycle, saved,
and reported success. Nothing in the loss curve reveals it.

The trainer's own `COVERAGE PROOF` line catches this at startup and did. Read it every launch:

```
COVERAGE PROOF: steps/epoch=628 global_batch=4 blocks/epoch=2512 dataset_blocks=2511  -> FULL
```

`steps/epoch` must equal your `TOTAL_STEPS` for a one-epoch run. If it does not, the run is
mis-sized — stop it. **Confirm the formula against a known-good run before trusting it on a new
one:** module4's log records `steps/epoch=26` for 101 rows at `global_batch=4`, and
`ceil(101/4) = 26`. That cross-check takes seconds and is what was skipped above.

## 4c. LINEAGE — how to prove what is actually in a model

CPT artifacts record no ancestry, so when the question arises it must be measured. The decisive
test needs no statistics:

**A vector cannot contain a component larger than its own norm.** If `‖refresh − base‖` is smaller
than `‖adapter_delta‖` on the tensors the adapter targets, that adapter is not inside the refresh.
Measured 2026-07-28: CPT delta norms 0.33–0.51 against module deltas 1.01–3.01, so modules 1 and 3
were provably absent from the served model.

**Two methods that did NOT work, recorded so they are not repeated.** A cosine probe over 9 tensors
with a `max > 4 × max_control` threshold flipped its verdict between two adapters on noise alone.
A paired-random control then scored BOTH adapters at ~2× the noise floor with identical 17/24 win
rates — despite one carrying 4,599 rows and the other 159. **Identical signal from unequal inputs
means the metric is reading a generic correlation between fine-tuning directions, not presence.**
Prefer the norm bound; it is a bound, not a statistic.

## 4d. CORPUS INTEGRITY — admit by content, never by count or name

- **Digest, not row count.** An in-place edit changes bytes while leaving the count identical, so a
  count check passes exactly when content has changed. Measured: a slice matched its registered
  31,920 rows while its sha did not, and 7 live credentials sat behind that mismatch.
- **Digest, not filename.** Naming a file to match an allowlist glob is certification by
  coincidence. Add an explicit entry carrying the provenance instead.
- **Verify on every rank before launch.** A distributed run reads a node-local copy; a partial
  distribution trains different data on different ranks. `sha256` on all 4, expect 4/4.
- **Exclude credential shapes BEFORE tokenization.** A secret written into parameters cannot be
  edited out — only retrained away. Confirm a scrub by having two operators do it independently
  and comparing bytes; identical output from independent passes is far stronger than either alone.
- Exclude bare `sk-` from credential patterns: it false-positives inside URLs (`zendesk-`).

## 5. THE RUN ARTIFACT — one record, so lineage cannot drift

Every run emits one record carrying, at minimum:

- **base identity** — the exact checkpoint/model resumed from. Its absence is what produced the
  lineage collision.
- **horizon position** — `TOTAL_STEPS`, `WARMUP_STEPS`, **and the resumed step count**. Not
  `TOTAL_STEPS` alone: it is the *combination* with the resume point that determines `f(step)`.
- **corpus identity** — path, sha, `CPT_PACKED`, `MAX_SEQ`.
- **dose proof** — Σf over the executed steps.
- **outcome** — abs / rel / changed from the weight-diff, against the named base.

One record, derived once, read by everything downstream. The recurring defect class this closes is
two places deriving one quantity and only one being updated — which happened five times in a single
session.

---

## WHAT IS STILL UNKNOWN, recorded rather than papered over

- **Whether merged LoRA survives a subsequent CPT pass intact.** Never measured. The lineage is
  designed to be cumulative and the artifacts chain correctly, but no measurement confirms prior
  modules are preserved through a refresh.
- **How a batch's effect shows up in the lane, and how fast.** Production is the oracle and it is
  already running; what is not yet written down is the expected lag between a batch shipping and a
  change being visible in walk outcomes, judgment shape, or scoring.
- **What governs weight displacement**, now that it is known not to track the LR integral.
