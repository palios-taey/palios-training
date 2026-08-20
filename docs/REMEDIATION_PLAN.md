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

## W3 — the run owns its lifecycle · **COMPLETE — CAPABILITY-SCOPED**

`taey-train` now preserves the entrypoint status and validates the durable rank-0 lifecycle journal
before returning. Starting is not success, and an entrypoint failure cannot be hidden by a journal
that already names an accepted state.

Shape: states `SPEC_VALID → NODE_DEPLOY → TRAINING → FRAGMENTATION_EXIT* → CHECKPOINT_SAVED →
BAKE_COMPLETE → THOR_DELIVERED`, one JSON line per transition to `lifecycle_events.jsonl`, and
success returned only at the invoking capability's `lifecycle_success_states` from
`PRODUCTION_MANIFEST.yml`. CPT and bake therefore have different legitimate invocation boundaries;
only the end-to-end training-plus-bake campaign ends at `THOR_DELIVERED`. **This is the largest
change and the one that answers "you never know when anything fails."**

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

---

# PHASE 2 — the end-to-end run (Jesse-directed, 2026-08-19)

**The directive, in order:** everything clean and to spec → a proper end-to-end training run on the
updated repos and the new infrastructure → then infra-codex dispatches it to ALL Chats to audit with
the full background files and packet → **Gemini sits out the audit and then does the synthesis.**

This phase exists because the last cycle burned two days for hours of work. The remediation above is
what makes this run's outcome *legible*; the run is what makes the remediation *proven*. Neither
counts alone.

## R0 — merge order, and what is actually blocking

Three stacked PRs, measured 2026-08-19:

| PR | head | base | state |
|---|---|---|---|
| #13 | `agent/codex-w3-w5-lifecycle-bake` | `tutor/requalify-manifest-shas` | open; 2 amendments outstanding |
| #12 | `grok/w6-correspondence` @ `636cbdc` | `tutor/requalify-manifest-shas` | open; predicate covers CANDIDATE_* |
| #11 | `tutor/requalify-manifest-shas` | `main` | **`r5-audit-gate: failure`** |

`#11` carries `audit/gatekeeper: success` and is missing the second lens. **The R5 gate is two
lenses on the exact head; one success is not a pass.** Nothing reaches `main` until `audit/grok`
lands on the head that is actually merged — and the head moves every time #12 or #13 lands, so the
audit is re-run on the final head, not on an ancestor of it.

Order: #12 and #13 into `tutor/requalify-manifest-shas` (merge gate below) → **re-pin, see below** →
re-audit the resulting head with BOTH lenses → #11 into `main`. Conductor merges; tutor does not
merge its own PR.

### THE MERGE RECIPE — constructed at `3d747dc`, current heads, exact values

The stack collapsed: **`491bfa0`'s parent is `99fb0f9`, so merging #14 carries #13 with it.** Two
merges, not three. Both verified **CLEAN, zero conflicts** (the earlier `_resolve_capability.py`
conflict is gone — tutor-grok rebased onto `99fb0f9` and absorbed codex's lifecycle handling).

```
1.  merge 06d5bfe   (#12  W6 correspondence + authorization guards)      CLEAN
2.  merge 491bfa0   (#14  resolver authorization; carries #13 W3/W5)     CLEAN
3.  RE-PIN — before running any gate, see below
4.  gates, then the two-lens audit on THIS head
```

**Step 3, exact.** Four pins drift because two branches edit two pinned files. Replace with the
merged-tree bytes:

| capability | path | new `content_sha` |
|---|---|---|
| `cpt_27b_4node` | `scripts/taey-train` | `066d762bca8e6dc1587cc397d12df71ac0107935660214f529fc95354ff24022` |
| `cpt_27b_4node` | `scripts/_resolve_capability.py` | `931882b5fab89a9f7832a0c5057aa2a8dc98006f221c63853c40228fbf6dc87d` |
| `bake_export` | `scripts/taey-train` | `066d762bca8e6dc1587cc397d12df71ac0107935660214f529fc95354ff24022` |
| `bake_export` | `scripts/_resolve_capability.py` | `931882b5fab89a9f7832a0c5057aa2a8dc98006f221c63853c40228fbf6dc87d` |

**ORDER IS LOAD-BEARING — re-pin BEFORE the self-test, or the gate looks broken.** On the merged
tree `check_manifest_pins.py --self-test` exits 1 with:

```
SELFTEST FAIL: real tree must PASS before mutation tests
```

That is **correct behaviour, not a defect**: the self-test refuses to run mutation fixtures against
an already-dirty baseline, since a mutation could otherwise "pass" for the wrong reason. It passes on
`06d5bfe` alone and fails on the merge only because the merge drifts the pins it is about to check.
Re-pin first and it goes green. Recorded because at merge time this reads as "grok's gate is broken"
and would cost a cycle.

Gate state on the merged tree before re-pinning: `check_variable_registry` exit 0,
`check_docs_index` exit 0, `_resolve_capability.py --self-test` exit 0, `check_manifest_pins` exit 1
(the four drifts above, and nothing else).

---

**Why the re-pin is legitimate rather than a forced gate** — first established with the earlier
`577cda2`/`636cbdc`/`82f46e0` construction:

```
DRIFT  cpt_27b_4node: scripts/taey-train
DRIFT  bake_export:   scripts/taey-train

merged   scripts/taey-train  354ca3639a2d15f3
82f46e0  scripts/taey-train  a0d96edefe67e231   (self-consistent with its own pin)
636cbdc  scripts/taey-train  4f253644487f14fb
```

Both branches edit `scripts/taey-train` — #12 adds the `--deployed` invocation, #13 adds the
lifecycle enforcement — and both pin it. The merged file contains both edits, so its sha matches
neither pin. **Any merge of two branches that both modify a pinned file leaves that pin stale by
construction.** Neither branch is defective.

The merger re-pins in the merge, stating the reason: both edits are intended, which is exactly the
condition the repo rule allows re-pinning under ("re-pin only when the change is intended, and say
why"). Recorded here because a stale pin found *after* a merge is indistinguishable at a glance from
the drift the gate exists to catch, and someone will otherwise hunt a corruption that is arithmetic.

*(A related report — W6 flagging `sft_stage2_lora` as unpinned — was a simulation artifact: it came
from running #12's checker against #13's manifest. `636cbdc` pins all three SFT structural files;
`82f46e0` pins none. On the real merged tree the finding disappears.)*

**Merge gate between #12 and #13, still live:** #13 downgrades `cpt_27b_4node` and `bake_export` to
`CANDIDATE_PENDING_PRODUCTION_RUN`; #12's UNPINNED predicate keys on status. `636cbdc` extends it to
candidate statuses, so the gate is *satisfied by that sha specifically* — merging an earlier W6 sha
alongside #13 silently lifts structural pin enforcement on the two capabilities whose bytes are new.

## R1 — the promotion problem. Read this before launching anything.

After #13 merges, `cpt_27b_4node` is `CANDIDATE_PENDING_PRODUCTION_RUN` and **`taey-train` will
refuse to launch it** (`scripts/_resolve_capability.py:50-57`; verified live — `sft_stage2_lora`
shows `blocked`). That refusal is correct: the bytes changed and have never executed.

It is also circular on its face — the status is earned by a run, and the run is gated on the status.
**The way out is not a new runnable status.** `scripts/taey-train:17` says there is no `--force` and
none may be added, and a "candidate-but-runnable" state is a `--force` wearing a different name.

The designed exit is already documented at `scripts/taey-train:8`: reaching production "requires
editing `PRODUCTION_MANIFEST.yml`: a visible, reviewable, gated act." So promotion is a **human-
authorized manifest commit**, and Jesse's directive to do the run IS that authorization. Two
constraints on how it is written, because this is the exact place a permanent bypass gets born:

1. The promoting commit states, in its body, that it carries an **authorization** and not a receipt,
   names who authorized it, and cites the audited head being promoted.
2. It is **single-use by construction**: the run's own receipt replaces it. If a promotion can be
   reused for a second run, it has become a standing bypass and must be revoked.

**DECIDED (tutor-grok, 2026-08-19). I offered two shapes and both were rejected; the third is
better than either, and one of my own stated constraints was the bug.**

I proposed (a) promote to `ADJUDICATED` with the receipt marked pending, or (b) a single-use
runnable status. Both are wrong:

- **(a) fails** because `ADJUDICATED` becomes a lie the moment it is written, and
  `_resolve_capability.py:51` treats it as a *standing* door. A `PENDING` marker in the receipt
  block is enforced by nothing — it is a comment, and the resolver never reads it.
- **(b) fails** because this plan already names it two paragraphs up: candidate-but-runnable is
  `--force` under another name. **And it breaks the run.** "Single-use, consumed on first launch"
  cannot express a multi-session CPT — `FRAGMENTATION_EXIT` means one campaign spans several
  `taey-train` invocations, so the second session would be refused by the authorization that
  legitimately covers it. My "single-use by construction" constraint was itself the defect.

**THE SHAPE (authoritative):** the status stays `CANDIDATE_PENDING_PRODUCTION_RUN` — honest, no lie
written anywhere — and the **authorization is a separate manifest object, never a runnable status**:

```yaml
authorization:                  # NOT a status. Absent by default.
  capability: cpt_27b_4node
  content_sha: {…}              # the EXACT pins it authorizes
  authorized_by: <name>
  campaign_id: <id>             # shared by every session of one campaign
```

- The resolver permits a `CANDIDATE_*` launch **only** when this block is present **and no receipt
  exists for that `campaign_id`**.
- It is bound to the **exact `content_sha` pins**. Change a byte and the authorization no longer
  describes what you are running — that binding, not a use-counter, is what stops it becoming a
  standing bypass.
- It is consumed at **campaign completion or explicit revoke** — never at the first optimizer step,
  which is the same "training started = success" error W3 exists to kill.
- After the run, a **second human commit** promotes to `ADJUDICATED` with the filled receipt **and
  deletes the authorization**. Two commits, two acts: one authorizes, one records what happened.

`taey-train:8` remains the exit — a manifest edit — and no new status enters the one door.

**Audit consequence, from the same lens:** `audit/grok` will not be posted on `9905ce8` (an ancestor)
nor on `1fb90d9` (a plan commit with #12/#13 unmerged). Pushing to the branch also cleared the
`audit/gatekeeper: success` that sat on `9905ce8` — at `1fb90d9` both lenses read *missing*, which is
the gate behaving correctly. **Both lenses re-run on the exact final merged head, once #12 and #13
land.** There is no path where an audit of an ancestor counts.

## R1a — the promotion guard, and my inverted predicate

The authorization object leaves one residual: if the promotion commit omits `campaign_id` or forgets
to delete the authorization, the authorization stays live — a standing bypass created by **omission**
rather than design.

**I specified the guard wrong.** I proposed *"CI fails when an authorization's `campaign_id` appears
in no receipt."* That state **is the valid pre-run window** (C2/C5) — an authorization exists and the
run has not happened yet. The check would have made the authorized launch un-mergeable, blocking the
exact thing the authorization exists to enable. tutor-grok refused it.

**The correct guards** (in `check_manifest_pins.py` or a sibling, never in the resolver, because
launch/refuse must keep permitting the live campaign):

1. **FAIL if an authorization names a capability whose status is `ADJUDICATED`.** The promotion
   commit must delete the authorization. *This is the forgotten-promotion catch and the one that
   matches the residual.*
2. FAIL if `authorization.campaign_id` equals a `receipt.campaign_id` — a consumed leftover. The
   resolver already refuses that launch (C4); CI failing the dirty manifest is the durable record.
3. **Do NOT fail** `CANDIDATE` + authorization + no receipt. That is the legitimate pre-run state.

A procedure note is not sufficient here: two-commit discipline without a checker is precisely how
omission becomes the standing bypass.

**Pattern worth recording about the author of this plan:** twice in one day I aimed a safety property
at the wrong state and it landed on legitimate operation — first "single-use, consumed on first
launch" (would have refused session 2 of a multi-session CPT), then this. Both were caught by the
seat implementing them, not by me. Constraints I write should be assumed too tight until someone
constructs the legitimate case they forbid.

## R2 — pre-run gates (none of these are optional, all have been skipped before)

- **Reboot all four Sparks before the run**, and again after. Never kill-and-relaunch on dirty GPUs.
- **Disk gate on every node** — a full disk on `.68` once truncated a checkpoint mid-save and wedged
  the node. Check before, not after.
- **Corpus: same six slices as the last CPT, rebuilt from the repos as they stand now** (Jesse,
  2026-08-19 — *Treasurer is NOT involved in this run*; the standing Treasurer-sanction rule does not
  govern it). The rebuild must emit a fresh receipt naming the new `corpus_sha256`, verified identical
  on all four nodes. **The v3 receipt is a receipt for a PRIOR build and must not be cited as this
  run's lineage.**

- **TWO CORPUS FINDINGS, both measured, both bearing on "properly this time":**

  **(a) The v3 corpus contains Taey's own generated output — R5's second failure mode, confirmed.**
  Found by treasurer, verified independently against the receipt's own input hashes:

  | slice | receipt sha | contents |
  |---|---|---|
  | `cpt_strategy_research_delta_v1_SCRUBBED.jsonl` | `0a81a0af…` match | generated exec-search letters ("Dear Christian & Timbers Team…") |
  | `cpt_raw_corpus_v4.jsonl` | `fd64cb08…` match | real cover letters ("Dear Boyden Team", "Dear Hiring Team at Hearst") |

  `slices_v2_probe/` and `slices_v2_probe.prescrub_20260729/` hold **byte-identical** copies of the
  file named `…SCRUBBED.jsonl` (both `0a81a0af…`) — **that scrub never touched it.** A filename
  asserting a property it does not have is why this survived prior review. Rebuilding from the same
  slices carries it forward. Scale is small and the remedy is row-level, not a redesign: ~9
  salutation-bearing rows across 1093, and some are legitimate *templates*
  (`[Standard ATS-Safe Contact Header]`, "Dear Hiring Manager / [Specific Name]") rather than
  generated artifacts. Separate the two before dropping anything.

  **(b) The builder that produced the v3 corpus is not in git.** The receipt names
  `builder_sha256 d7e9d7cb…`; the only committed revision (`e164ebd`) is `c6770d6d…`, and all four
  Sparks carry `c6770d6d…` as well. Committed and deployed agree with each other and disagree with
  the receipt. Rebuilding with the committed builder is therefore a genuinely different build — which
  is acceptable, but it must be stated in the new receipt rather than implied to be a reproduction.
- **Recipe is Chats-researched only.** No solo LR/optimizer choices.
- **Deployed bytes must equal git bytes.** `check_manifest_pins.py --deployed` (W6) hashes the Spark
  copy. This is the defect the repo already paid for; it is now checkable, so check it.
- **Every dynamic variable supplied explicitly.** `TOTAL_STEPS`, `SESSION_LIMIT`, `SAVE_EVERY`,
  `MAX_SEQ`, `CLOCK_CAP`, `BATCH_SIZE_PER_RANK`, `CPT_DATA`, `CPT_PACKED`. The launcher refuses
  without them by design — that refusal is the feature, not an obstacle to route around.

## R2a — LAUNCH CONSTRAINTS from the audit of the merged head (tutor-grok, ENDORSE)

**Anchored to `dd1c3a4`, which merged to `main` at `e21bebf`.** First written against `b907fb8`;
tutor-grok re-endorsed on `dd1c3a4` and stated the residuals unchanged, so every constraint below
still governs the run. `audit/grok` + `audit/gatekeeper` both success, `r5-audit-gate` green.

**The re-audit was EXECUTED, not read, and that is why this endorsement is worth more than the
first one.** On `b907fb8` every static check passed while `lifecycle_call` would have aborted every
CPT launch. On `dd1c3a4` tutor-grok ran it in a fresh directory:

```
SPEC_VALID, NODE_DEPLOY, TRAINING, FRAGMENTATION_EXIT   all LIFECYCLE APPEND exit 0, 4-line journal
skip to THOR_DELIVERED after FRAGMENTATION_EXIT         exit 1, LIFECYCLE INVALID, journal unchanged
path containing a space                                 %q escaped, journal written
OLD form on a sibling fresh dir                         --state: invalid choice: 'python3', NO journal
```

The last two lines are the ones that matter: a negative control proving the validator rejects an
illegal transition, and an A/B proving the old form genuinely failed. A pass without those is a pass
that cannot tell a working gate from a broken one.

Verdict: on the production door this head does not
report success a run did not earn — CANDIDATE refuses without authorization, the resolver always
emits `LIFECYCLE`, the CPT inner no longer exits at the first optimizer step under `CPT_LIFECYCLE=1`,
`taey-train` requires the capability-owned terminal state and dies on entrypoint failure even when
the journal already names an accepted state, bake will not start from `FRAGMENTATION_EXIT`, and a
leftover authorization after promotion is a CI failure (A1/A7).

The residuals below are **not** blocks. They are the conditions under which that verdict holds, and
several of them decide how the run must be launched. Read them as launch constraints, not trivia.

**1. `taey-train` SUCCESS IS CAPABILITY-SCOPED, NOT CAMPAIGN-WIDE.**
With `lifecycle: true` on `cpt_27b_4node`, an entrypoint success ending at
`FRAGMENTATION_EXIT` or `CHECKPOINT_SAVED` returns 0 because those are the CPT capability's
manifest-declared invocation boundaries. `bake_export` still accepts only `THOR_DELIVERED`.
**Do not read CPT's exit 0 as delivery or as proof that every planned session is complete.** Read
the printed lifecycle state: resume from the checkpoint after `FRAGMENTATION_EXIT`; proceed to bake
only from `CHECKPOINT_SAVED`. Any non-accepted state or non-zero entrypoint status still fails.

**2. A FRESH `OUTPUT_DIR` / `DCP_DIR` IS MANDATORY.** W3's last-state is not invocation-bound —
reusing a directory that already reached `THOR_DELIVERED` can confuse it, and `GEMM_PREFLIGHT_ONLY=1`
exits before the lifecycle resume check. A fresh directory fail-closes; a reused one may not.

**3. ONE DOOR IS MECHANICAL BEFORE PREPARATION (task-2220f1b9).**
`capture_run.sh`, `run_till_done_v{2,3}.sh`, and `run_refresh_gate.sh` first run the existing
`TAEY_TRAIN_CHECK_ONLY=1 scripts/taey-train cpt_27b_4node` gate and exit on refusal before any
remote, reboot, deploy, watchdog, capture, or durable-log side effect. Their real launch rechecks
authorization and propagates failure. `run_4node_27b_cpt.sh` also fail-closes via
`_resolve_capability` before tmux/NCCL as the direct-bash backstop.

**4. `EXPORT_DCP` MUST BE UNSET.** A leftover value restores the inner script's first-step exit 0.
Through `taey-train` with a fresh journal W3 still catches it, but do not rely on the backstop.

**5. VERIFY THE CLOCK CAP APPLIED — do not trust the script's exit.** A `CLOCK_CAP` ssh failure is
**non-fatal**, so the cap can silently fail to apply on a node. This one is not cosmetic: the 27B
whole-node death on this hardware was a **thermal** shutdown (~94 °C board/SoC), so an unapplied cap
is the failure mode that killed a previous campaign. Confirm the cap on all four nodes by reading it
back, not by observing that the launcher did not complain.

**6. `bake_27b.sh` still carries `:=` defaults, including a hardcoded corpus.** It is a stage, not
the door, so the manifest gates it — but the hardcoded corpus is exactly the "runs a different
campaign silently" shape, and it has not been cleaned.

Also open and already known: `never_defaulted()` is mode-unaware (mine); `CLAUDE.md` §7 and the
manifest header are stale against the code.

**What would invalidate the endorsement** (grok's own words, worth keeping as the falsifier):
`taey-train cpt_27b_4node` on a fresh run directory **exiting 0 with a last state outside its
manifest-declared `lifecycle_success_states`**, or `bake_export` exiting 0 without
`THOR_DELIVERED` written by this campaign.

## R3 — the run, and what "done" means

`scripts/taey-train cpt_27b_4node VAR=… …` — the one door, no inner script, no node-local edit.

**Training starting is not the run succeeding.** `run_4node_27b_cpt.sh:351-359` prints
`27B IS TRAINING` at the first optimizer step; that is confirmation of fit. A CPT invocation is
complete only at one of the CPT capability's manifest-declared lifecycle boundaries. The full
training-plus-bake campaign is complete only when the journal reaches `THOR_DELIVERED` (W3), with:

- weight-diff in band `5e-05..8e-04`, measured against the run's **own** source, never a pinned donor
- tensor count `851 → 1199`, graft verified **by content**, not by count
- artifact sha256 matched on the serving host
- served-root match by identity, consumer quiesce before any swap

**Serving swaps go through infra-codex** (Jesse, standing): check with them before doing anything
that touches a serve, so they manage it against whatever else is in flight.

## R4 — the audit dispatch

When R3 is complete, **request that infra-codex send it to all Chats** with the full required
background files and packet. Not tutor — infra-codex owns this dispatch.

- **Gemini sits out the audit, then does the synthesis** of the other lenses' findings.
- Gemini cannot fetch GitHub repos anyway, so it is the correct lens to hold back for synthesis.
- The packet points at the **public repo**, carries the don't-trust-my-summary clause, and pre-loads
  no conclusion. At least one reviewer must quote a real `file:line` or the review proved nothing.
- Reviewers that must fetch go to fetch-capable modes: Perplexity DR, ChatGPT DR, Grok DeeperSearch,
  Claude Research. A reasoning-mode "cannot fetch → BLOCK" is a dispatch error, not a finding.

## R5 — what would make this run a failure even if the loss curve looks fine

Recorded now, while nothing is at stake, because the last cycle's failure was invisible from inside:

- weights that did not move (a null run reported as a trained one)
- a corpus containing the model's own generated output
- a graft from a foreign vision tower that every count gate passes
- a "done" asserted from a status line rather than from the artifact on the serving host
