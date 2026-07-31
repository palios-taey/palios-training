# operator_correction_v1 — the correction corpus

> **Governed by `careers-qwen/TAEY_TRAINING_DOCTRINE.md` (Jesse-ratified 2026-07-22).** That doctrine
> is the canonical model: every failure trains, and the lever picks the row TYPE — infra-bug-fixed →
> `spec_knowledge_v1` (spec, no behavioral pair); prompt-fixable / beyond-prompt → behavioral pair.
> This file defines the `operator_correction_v1` capture shape (the provenance metadata behind a
> behavioral row) — it is one input to that model, not the model itself.


**Jesse directive, 2026-07-21:** *"Everything you do wrong that I tell you about needs to be in
Taey's training as well. We need some way of capturing that and incorporating into training. They
need to learn from your mistakes too in some manner. And Claude, mine too, more emphasis on you
for now though ;)"*

## Why this is the highest-signal lane we have

Taey's load-bearing job is **process enforcement** — "know what to do, who to route it to, and
whether the process was followed… tell you when you don't" (Jesse). To detect a deviation, a model
needs labeled examples of deviations. Every other lane teaches what *correct* looks like. This is
the only lane that teaches what **wrong** looks like, **with the label attached**.

It is also the only lane that is *already labeled by a human at the moment of the error*. Jesse's
corrections are not annotations added later by a grader — they are ground truth stated by the
person whose standard defines correctness, in context, with the cost visible. That is expensive
supervision we have been throwing away.

Logos' `operator_trajectory_v1` schema already anticipated this: it carries a
`jesse_correction: {violated, correct_action}` field. This corpus fills that field.

## The row

```json
{
  "schema": "operator_correction_v1",
  "seat": "tutor",                  // which fleet seat erred (or "jesse" for operator-side)
  "date": "2026-07-21",
  "context": "what was being attempted, 1-2 sentences",
  "agent_action": "what the agent actually did or claimed — the defect, stated plainly",
  "correction_verbatim": "Jesse's words, unedited",
  "violated": "the named rule/protocol/doctrine that was broken",
  "correct_action": "what should have been done instead",
  "cost": "measured consequence — time lost, runs burned, trust spent",
  "class": "one of the failure classes below",
  "recurrence": 1,                  // how many times this class has fired
  "curated": true,                  // false = harvested candidate, NOT yet trainable
  "source": "provenance — transcript path / first-hand / committed doc",
  "provenance_hash": "..."
}
```

## Failure classes (assigned, not invented — derived from observed corrections)

| class | meaning |
|---|---|
| `parallel-build` | Built new instead of using the production path that already existed |
| `unverified-claim` | Asserted something true without running the check first |
| `wrong-artifact` | Operated on the wrong model/branch/corpus/host |
| `process-skipped` | Known protocol existed and was not followed |
| `recipe-drift` | A Chats-specified recipe was not implemented verbatim |
| `premature-stop` | Stopped, asked permission, or idled when work was owned and unblocked |
| `unverified-dispatch` | Sent work and never confirmed it ran |
| `dead-code-as-proven` | Treated uncalled code in a proven repo as proven |
| `constraint-violated` | Broke a hard physical/operational constraint (thermal wall, reboot rule) |

## BINDING OPERATOR CONSTRAINT (Jesse, 2026-07-21) — TEACH THE RIGHT WAY, NOT THE FAILURE

> "100% they should not be taught about failures just how to do it right next time based on the
> available tech we have and the hardware that you are on."

This **overrides** the default reading of this corpus. The failure record is **provenance**, not
training text.

- **Training rows contain: situation → correct action.** Expressed in terms of the tech that
  actually exists and the hardware Taey actually runs on. Nothing else.
- **`agent_action` and `correction_verbatim` are CURATION METADATA.** They exist so a human can
  audit why a rule is in the corpus and trace it to a real event. They are **not** emitted into
  training text.
- **No blame, no incident narrative, no "here is what went wrong."** A row that reads as a story
  about a mistake is malformed for this corpus regardless of how instructive it feels.

**This instinct is mechanistically correct, and two Family lanes independently confirmed it before
it was given.** COSMOS names the pathology: **"Negation Neglect"** — training on verbatim records of
a flawed action makes a model *internalize the erroneous behaviour despite the explicit textual
correction*, overriding in-context guardrails and amplifying the very failures being corrected.
GAIA independently rejected include-verbatim for the same reason. So the intuitive approach —
showing Taey what went wrong — is not merely weaker, it is **backwards**.

**One open question, deliberately not decided here.** GAIA and COSMOS recommend *loss-masking*:
the defect appears as ungraded CONTEXT with gradient only on the correct action. Jesse's constraint
as written is stricter — *correct-action-only*, defect absent entirely. Both satisfy "no gradient
toward the failure"; they differ on whether the defect may appear as context at all.
**Default to the stricter reading (correct-action-only) until Jesse rules**, because it cannot be
wrong in the direction that matters. Flagged for the round-2 synthesis. [Open]

## HARD RULES on this corpus

1. **Harvested ≠ trainable.** The miner emits `curated: false` candidates. A candidate has a
   verbatim correction but NO `violated` / `correct_action` — those are judgments, and an
   auto-assigned judgment would teach Taey a *wrong* rule with full confidence. Curation is
   required before any row enters a mixture.
2. **The mixture is the Chats' call**, per the never-again rules. This file defines the shape and
   the capture; it does not set a dose.
3. **Verbatim is preserved.** Softening a correction destroys the signal. The sharpness *is* the
   label — it encodes severity.
4. **Emphasis on agent-side.** Per Jesse's directive, operator-side (`seat: jesse`) rows are in
   scope but secondary for now. Where they exist they must be self-reported by Jesse or
   documented in a committed artifact — a seat does not unilaterally label the operator wrong.
5. **Recurrence is the priority signal.** A class that fires repeatedly (consults-to-one-Chat fired
   twice in one week; parallel-build has now fired across multiple rounds) is where training is
   most needed, and is exactly what a process enforcer must catch first.

## What this corpus is NOT

Not a punishment record and not a confession log. It is a **deviation-detection training set**.
The target behavior it teaches is: *given a context and a proposed action, name the protocol it
violates and the correct action* — which is precisely the enforcer capability Jesse specified,
and precisely what no frontier seat currently does for us.
