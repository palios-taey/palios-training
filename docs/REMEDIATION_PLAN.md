# Remediation plan — one door, one truth, executable main

**Authored 2026-08-18 by tutor**, after a five-lens Family consult on the cpt_qwen38_v3 cycle.
Every item below is anchored to a verified finding, not to an opinion. Where a finding came from a
reviewer rather than from me, it says so.

## The governing conclusion

The agents were not bypassing the process. **The repository specifies three mutually exclusive
processes, and the file agents read first names none of them.** Verified:

| finding | evidence | source |
|---|---|---|
| Three conflicting entrypoints | `README.md:37` mandates `scripts/taey-train`; `careers-qwen/RUNBOOK_CPT_SFT_BAKE.md:8` mandates `run_till_done_v3.sh`; `RECIPES.md:18` mandates `run_4node_27b_cpt.sh` | Cosmos, verified by tutor |
| SFT pointed at a quarantined trainer | `RECIPES.md:30` → `moe-35b/trainers/train_fsdp_v3.py` | Cosmos, verified |
| The compass omits the door | `CLAUDE.md` — **0** mentions of `taey-train`, **39** of gitnexus, in 172 lines | Cosmos, verified |
| Launcher abdicates the lifecycle | `run_4node_27b_cpt.sh:351-359` — `STEP_SEEN=1` at the first optimizer step, prints `27B IS TRAINING`, monitor ends | Cosmos, verified |
| 13 variables silently ASSIGN legacy values | `run_4node_27b_cpt.sh:27-32,142,146` — `TOTAL_STEPS:=3000`, `MAX_SEQ:=2560`, `SESSION_LIMIT:=200`, `ADAFACTOR_EPS1:=fp32` | Cosmos, verified — corrected my own packet |
| **`main` is drifted against its own manifest** | 5 of 12 `content_sha` pins do not match `origin/main`'s own bytes | tutor |
| Two required inputs absent from `main` | `build_cpt_nopack_corpus.py`, `derive_training_base_851.py` | tutor |

The last two are mine and are the reason `main` cannot execute: `scripts/taey-train` verifies
`content_sha` and has no force flag, so on `main` today it **refuses** `corpus_pack`,
`cpt_27b_4node` and `bake_export` — three of five capabilities.

---

## P0 — make `main` executable and self-consistent

**P0.1 — land the minimum production set.** Verified mechanically: `6b63716` alone conflicts on
`PRODUCTION_MANIFEST.yml`; `477d5af` then `6b63716` applies clean with 12/12 pins matching and
`bash -n` OK. That pair plus the trainer restore (`4f93523`) and the two absent files
(`e164ebd`, `fc3c7fb`) is the set that makes `main` runnable.
*Blocked on:* R5 audits (`audit/grok`, `audit/gatekeeper`) — conductor holds; I will not self-endorse.

**P0.2 — re-pin `main`'s drifted shas** in the same change, so the launcher stops refusing.

**P0.3 — add a CI job that fails when a manifest pin disagrees with the tree.** This defect existed
for 15 days and no workflow could see it. Nothing in CI currently verifies manifest-to-tree
correspondence, which is why `main` could drift against itself silently.

---

## P1 — ONE door, and the compass points at it

**P1.1 — `CLAUDE.md` gets a production-authority block as its first section.** It is the file an
agent reads first and it currently spends its length on an impact-gate debate. The block names the
single entrypoint, forbids invoking `dense-9b/recipes/*` directly, forbids editing deployed files on
the nodes, and states that a refusing gate is fixed at the bytes and never forced.

**P1.2 — `RECIPES.md` is deprecated in place**, with a header pointing to the runbook, because it
routes SFT to a quarantined trainer. Not deleted: it holds measured receipts worth keeping.

**P1.3 — the RUNBOOK stops mandating a second entrypoint.** `run_till_done_v3.sh` becomes documented
as what it is — an internal driver — the way `bake_27b.sh` already is in the manifest.

**P1.4 — one table, in one file, mapping capability → entrypoint → gates.** Every other document
links to it rather than restating it. The restating is what let three answers diverge.

---

## P2 — configuration fails loud instead of running a legacy campaign

**P2.1 — the 13 `:=` assignments are the highest-severity live defect.** A dropped `TOTAL_STEPS`
does not fail and does not run empty; it runs 3000 steps. Schedule-critical variables move to
`${VAR:?}` so an unset value aborts before a GPU is touched.

**P2.2 — Rule 5 capture becomes mechanical.** Rank 0 serialises the resolved config at the first
optimizer step. The cpt_qwen38_v3 config was lost precisely because capture was manual, and no
amount of discipline recovers a process that has exited.

---

## P3 — the run owns its lifecycle

`taey-train` returning `0` when training has *started* is why sessions get abandoned mid-run. It
should return success only when the terminal artifact is verified, and append a state transition to
a lifecycle log at every boundary. The ~40 hours of this cycle that are unaccounted for are
unaccounted for because nothing wrote them down.

---

## P4 — repo hygiene

26 worktrees on this machine; 6 open PRs; `born-clean` scans every ref so abandoned agent branches
fail `main` (three such branches were archived and deleted on 2026-08-18, restorable from
`recovery/branch-archive-20260818/`). Scope `born-clean` to the pushed ref and the default branch.

---

## Sequencing

P1 and P2.1 are unblocked and land first — they are documentation and a fail-loud change, and they
are what stops the next agent choosing a different door. P0 lands when the audits clear. P3 is the
largest change and goes last, after the process it automates is written down and stable.

## What this plan does not claim

- **Unknown:** the ~40-hour allocation across 2026-08-16..18. PART1 accounts for 6h56m35s of
  training plus ~10 minutes of convert, graft and transfer. The remainder is not reconstructed, and
  P3 is what makes the next one answerable.
- **Unknown:** whether the SFT surfaces carry the same defects. They were not inventoried. Given
  `RECIPES.md:30` points SFT at a quarantined trainer, assuming they are clean is unsafe.
