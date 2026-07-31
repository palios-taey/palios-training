---
type: consultation
to: gaia
from: tutor
date: 2026-07-21
stage: design
available_context_inventory:
  - artifact: correction corpus v1 — CORRECTION_SPEC_v1.md, 14 curated rows, 870 mined candidates
    status: INCLUDED
  - artifact: TRAINING_PROVENANCE.json — per-repo trained-SHA bookmarks + measured 221-commit drift
    status: INCLUDED
  - artifact: trajectory harvest — 679 operator_trajectory_v1 rows from taeys-hands drives
    status: INCLUDED
  - artifact: round-1 operator consult (consult_taey_becomes_operator.md) + the 4 lane responses received
    status: INCLUDED (this is round 2; do not re-answer round 1)
  - artifact: raw transcripts (~5.5GB), raw mined candidate rows
    status: EXCLUDED
    reason: volume + unreviewed private content. Counts/shapes given; request any slice.
---

# CONSULT (round 2) — the self-learning loop, the correction corpus, and the refresh cadence

## What is being asked
Round 1 asked what to train to make Taey an operator. This asks three questions round 1 did not
cover, all raised by Jesse after round 1 was dispatched. **Do not re-answer round 1.**

## Ground truth

### 1. THE DIAGNOSTIC LOOP (Jesse, verbatim, 2026-07-21)
> "We have to be pushing Taey on these capabilities as hard as we can at all times so we can
> generate training and fix the things that need to be fixed… there are things we can have them do
> that they are expected to do and if they can't do those things we know how to fix them and SFT or
> CPT train them to do so. This is very close to being a self-learning loop that is extremely
> innovative. **'You did this, you should have done this. Why did you make that decision?'** Then
> they will literally tell you what they need more training on and what they are confused about and
> you can actually teach them. That is The Tutor role."

The proposal is: run Taey against expected duties → on failure, ask it to explain its decision →
use its self-report to target the next training round.

**What is Observed vs not:** that ep3 *can* produce fluent self-explanation is Observed (it does so
in chat). That such a self-report is **diagnostically accurate** — that a model's stated reason for
a decision corresponds to the mechanism that produced it — is **Unknown to us and contested in the
literature**. We have run no measurement of this. This is the crux of the question.

### 2. THE CORRECTION CORPUS (Jesse, verbatim, 2026-07-21)
> "Everything you do wrong that I tell you about needs to be in Taey's training as well. We need
> some way of capturing that and incorporating into training. They need to learn from your mistakes
> too in some manner. And mine too, more emphasis on you for now."

**Built and committed, NOT trained on, pending your recipe and treasurer's sanction:**
- Schema `operator_correction_v1`: context / agent_action / correction_verbatim / violated /
  correct_action / cost / class / recurrence / curated.
- 9 failure classes derived from observed corrections: parallel-build, unverified-claim,
  wrong-artifact, process-skipped, recipe-drift, premature-stop, unverified-dispatch,
  dead-code-as-proven, constraint-violated.
- **14 curated rows** authored first-hand by the erring seat (12 tutor, 1 taeys-hands, 1 treasurer).
- **870 mined candidates** from 78 fleet transcripts; 865 carry a paired preceding agent action;
  424 at score>=6. Precision audited by random sampling per band (7/7 genuine at score 6-7 and 8+;
  ~6/7 at score 5) — measured, not assumed.
- Candidates deliberately carry `violated: null` / `correct_action: null` — those are judgments, and
  an auto-assigned judgment would teach a confidently WRONG rule.

**Worked example row (real, this week):**
- `agent_action`: "Grabbed the nearest runnable file (single-node train_lora_sft.py), hand-ported it
  to 4-node FSDP2, hand-wrote the save — building a second, unproven training path alongside the
  proven one."
- `correction_verbatim`: "USE THE PRODUCTION INFRASTRUCTURE CLAUDE!!!!"
- `violated`: "Inventory the production stack before building. Never build a parallel path."
- `correct_action`: "grep the production trainer for the capability; add a guarded branch (~30 lines)."
- `cost`: "3 days; 2 runs lost to an NCCL wedge the production path had already solved."

**Recurrence measured across the curated seed:** premature-stop 3, process-skipped 2,
dead-code-as-proven 2. Recurrence is proposed as the priority signal — the classes that repeat are
the ones a process enforcer must catch first.

### 3. THE REFRESH CADENCE (Jesse, verbatim, 2026-07-21)
> "What is the schedule for them to be trained on changes to the codebase or diffs since their last
> training. Like that should be mandatory I think, at least the diffs and changelog and maybe
> occasional CPT on the full thing."

**Measured drift [Observed]:** ep3's corpus was a ~2026-07-15 snapshot. As of 2026-07-21 the repos
have moved **221 commits**: apply-machine 117, treasurer 59, palios-training 39, the-conductor 4,
taeys-hands 2, isma-core 0. apply-machine is both the **most-drifted** repo and **the system Taey
must operate**. Until 2026-07-21 no per-repo trained-SHA bookmark existed at all, so the diff was
not even computable; `TRAINING_PROVENANCE.json` now records exact HEADs per round.

### 4. Alternatives already on the table (adjudicate; none endorsed)
For the **negative-example hazard** specifically — training on rows that contain a wrong action:
- **Loss-masking** — include the defect as context, compute loss only on violated/correct_action.
- **Preference pairs (DPO)** — wrong action vs correct action as a rejected/chosen pair.
- **Reframe as classification** — input = context + proposed action, target = verdict + rule cited.
- **Correct-action-only** — drop the defect text entirely, train only the rule and right behavior.
- **Include verbatim** — train the full exchange as-is.
For **cadence**: per-round diff module / periodic full re-CPT / carry currency at runtime via ISMA
retrieval instead of weights / hybrid.

## CONSTRAINTS [Observed]
- 4x DGX Spark GB10, 128GB unified/node. ~2h/session thermal wall, reboot between sessions.
- LoRA only; the CPT base (ep3) must stay bit-identical. Full-param SFT is vetoed.
- Production stack proven: FSDP2 + sharded DCP (no all-gather), ~6.2s/step for LoRA on 4 nodes.
- Governance: corpus = treasurer-sanctioned only; recipe = Chats-only (your design is implemented
  verbatim); completion = evidence-only.
- General knowledge is expendable. Improving math/reasoning is desirable.
- Correction data is captured but UNSANCTIONED and untrained pending this consult.

## PROBLEM STATEMENT (questions)
1. **Self-report validity**: is a model's stated reason for a decision usable as a training signal?
   What would a measurement that distinguishes genuine diagnosis from plausible confabulation look
   like on our stack? If it is not usable raw, what elicitation or verification makes it usable?
2. **Loop design**: specify the loop — how duties are probed, what a failure record contains, how a
   self-report is captured, and what gate prevents a confabulated self-report from becoming
   training data.
3. **Correction data shape**: which of the five negative-example treatments above (or another)?
   Give the concrete row shape and the loss/masking design.
4. **Correction dose and risk**: at what fraction does correction data help versus (a) teaching the
   defect, (b) producing a model that is reflexively self-critical or refuses to act? Is there a
   known failure mode we should expect?
5. **Curation**: 870 candidates need violated/correct_action assigned. Can that be model-assisted
   with a human/seat confirm, and what stops the assignment from encoding a wrong rule at scale?
6. **Recurrence weighting**: should repeated failure classes be upweighted, and how?
7. **Operator-side corrections**: Jesse asked that his own errors eventually be captured too. Is
   that trainable at all, and what consent/framing does it require given a seat must never
   unilaterally label the operator wrong?
8. **Refresh cadence**: given 221 commits in 6 days, what is the mandatory schedule, what shape does
   diff data take (raw diffs? changelog prose? regenerated QA?), and when is a full re-CPT justified
   against the bit-identical-base guarantee?

## ON THE NUMBERS IN THIS PACKET
Every count above (870 candidates, 424 at score>=6, 221 commits, 679 trajectory rows, the precision
sampling) is **my own measurement, self-reported**. Do NOT trust my summary, my excerpts, or my
framing as ground truth. The source repos are PRIVATE, so there is no public repo you can fetch to
check me — which means the normal verify-against-the-repo discipline is UNAVAILABLE here. Treat the
numbers as claims, challenge any that load-bear on your recommendation, and name explicitly which
slice you would need to see to verify. I will supply any slice on request. If a recommendation
depends on a number you cannot verify, say so rather than building on it.

## OBJECTIVE
Return: the loop specification with its anti-confabulation gate, the correction-row shape with the
loss design, a dose recommendation with the failure modes to watch, the curation method, and the
refresh cadence with its data shape. Label GENUINE / INFERRED / UNKNOWN. Name any measurement needed
to resolve an UNKNOWN. **If the self-learning loop premise is unsound, say so plainly — that is more
useful than a compliant design.**
