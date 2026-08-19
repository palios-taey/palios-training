# Remediation plan — one door, one truth, executable main

**Authored 2026-08-18 by tutor.** Every item is anchored to a verified finding. Findings from
Cosmos (Gemini) and from tutor-grok's adversarial audit are attributed to them.

Status legend: **DONE** (shipped + verified) · **PARTIAL** · **NOT STARTED**

---

## The governing conclusion

The agents were not bypassing the process. **The repository specified three mutually exclusive
processes, and the file agents read first named none of them.** Verified:

| finding | evidence | found by |
|---|---|---|
| Three conflicting entrypoints | `README:37` / `RUNBOOK:8` / `RECIPES:18` | Cosmos, verified |
| SFT pointed at a quarantined trainer | `RECIPES.md:30` → `moe-35b/trainers/train_fsdp_v3.py` | Cosmos, verified |
| The compass omits the door | `CLAUDE.md` — 0 mentions of `taey-train`, 39 of gitnexus | Cosmos, verified |
| Launcher abdicates the lifecycle | `run_4node_27b_cpt.sh` STEP_SEEN=1 at first step | Cosmos, verified |
| 13 variables silently ASSIGN legacy values | `:27-32,142,146` | Cosmos — corrected my packet |
| `main` drifted against its own manifest | 5 of 12 pins | tutor |
| My gate had a bypass flag | `BAKE_TO_HF`/`EXPORT_DCP` skipped it | tutor-grok |
| My gate broke a live caller | `capture_run.sh:64` | tutor-grok |
| My two new checkers can pass while wrong | pinned-subset only; substring-only | tutor-grok |

---

## W1 — one door, and the compass points at it · **DONE**

`CLAUDE.md` PRODUCTION AUTHORITY section (`a6c35be`); `RECIPES.md` deprecated header AND body row
(`372f766`); RUNBOOK entrypoint note; README/manifest SFT status reconciled. `docs/INDEX.md` +
`scripts/check_docs_index.py` + CI make un-indexed or unattributed docs fail (`8a24349`).

**Residual:** the docs checker tests substring presence, not correctness — tutor-grok, confirmed.
Tracked in W6.

## W2 — the env maze · **PARTIAL → this cycle completes it**

Done: every per-run default deleted; `${VAR:?}` refuses at bash level; solved hardware shape set
unconditionally in one place; mode announced not inferred; export mode needs `RESUME_DELTA`
(`372f766`).

**This cycle:** a **VARIABLE REGISTRY** — the answer to "don't lose track of them". One file
declares every variable the production path reads, each classified `dynamic-required`,
`invariant`, or `optional`. A checker reads the LAUNCHER'S ACTUAL VARIABLE READS and fails when
the code and the registry disagree in either direction. That is a correspondence check, not a
consistency check: a new variable added to the launcher fails CI until someone classifies it, and
a registry entry for a variable nobody reads fails too.

Still open after this cycle: Rule 5 capture remains manual. Rank-0 serialisation at first step is
W3 work.

## W3 — the run owns its lifecycle · **NOT STARTED**

`taey-train` returns 0 when training has *started*. `scripts/taey-train:95` execs and returns its
status; the launcher ends its monitor at the first optimizer step. Nothing records state
transitions, which is why ~40 hours of this cycle cannot be accounted for.

Shape: states `SPEC_VALID → NODE_DEPLOY → TRAINING → FRAGMENTATION_EXIT* → CHECKPOINT_SAVED →
BAKE_COMPLETE → THOR_DELIVERED`, one JSON line per transition to `lifecycle_events.jsonl`, and
success returned only at `THOR_DELIVERED`. **This is the largest change and the one that answers
"you never know when anything fails."**

## W4 — mainline · **PARTIAL**

Branches 20 → 2 (`main` + PR #11); 13 branches and 5 stale PRs archived as verified bundles in
`recovery/branch-archive-20260818/` and deleted. `secret-scan`, `no-private-data`, `born-clean`,
`manifest-pins`, `docs-index` green on the branch.

**Blocked:** PR #11 needs `audit/grok` + `audit/gatekeeper`. I will not self-endorse 24 commits I
authored. **Open:** `born-clean.yml` still scans `--all` rather than the pushed ref; no
deploy-from-SHA; `main` still 5-way drifted against itself until #11 lands.

## W5 — bake path matches the measured topology · **DOCUMENTED, NOT IN CODE**

`post_cpt_pipeline.sh:13,20` still require `ARTIFACT_STORE`; `:97-98` still derive under it;
`:16,23` still take the donor from the `fleet.env` pin rather than the run's own source. The
runbook records the correct order and the measured numbers; the code does not implement it.

## W6 — the checkers must fail for the right reason · **DONE on grok/w6-correspondence**

`check_manifest_pins.py` hashed only the pinned subset — an unpinned production file could be
arbitrarily wrong and it still exited 0, and it did not compare NODE-DEPLOYED bytes to the pin.
`check_docs_index.py` tested index membership and substring presence, so a document could contain
the words "PRODUCTION AUTHORITY" and still instruct `bash run_4node_27b_cpt.sh`.

Both were the consistency-not-correspondence shape, in gates written to prevent it.

**Landed:**
- ADJUDICATED structural paths (entrypoint / trainer / per_node / config / runner / qualifier /
  path_family_winner / stage entrypoints) must be in that capability's `content_sha`. Self-test
  drops the CPT launcher pin and requires UNPINNED; mutates bytes of a pinned file and requires
  DRIFT; mutates README (not a production path) and requires PASS.
- `--deployed` hashes `$SPARK_HOME/palios-training/<path>` on each Spark. Fails closed without
  `fleet.env`. `scripts/taey-train` invokes it when `fleet.env` is present; CI cannot SSH and
  does not pass the flag. Self-test injects a mismatched remote hash; no live SSH in CI.
- Docs checker fails on an affirmative inner launch (`bash <inner>` or `LAUNCH: ... <inner>`)
  even when the AUTHORITY banner is present. Prohibitions and citations are not instructions.
  Three documents that carried the banner and still instructed the inner script were corrected
  to `scripts/taey-train` in the same commit.
- Both checkers `--self-test`, and both CI workflows run it. Crash is distinguished from
  detection: a Traceback or an unrendered `{m.group(...)}` is a SELFTEST FAIL even if exit 1.

**Not in this commit (still W5 / W3):** Rule 5 rank-0 serialisation; bake path order; lifecycle
ownership. Node-deployed hashing is the hook; it cannot be live-verified from a laptop.


---

## Sequencing

W2 registry (landed on PR #11, still under audit) → **W6 checker correspondence (this branch)** → W5 bake path → W3 lifecycle. W4 lands when the audits clear. Grok validates each against Cosmos's spec rather than against tutor's description.

## Unknowns, stated

- The ~40-hour allocation across 2026-08-16..18. W3 makes the next one answerable.
- Whether the SFT surfaces carry the same defects. Never inventoried; `RECIPES.md:30` is reason to
  assume they might.
