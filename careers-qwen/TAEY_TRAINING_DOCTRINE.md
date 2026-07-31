# TAEY TRAINING DOCTRINE — the correction/training model

**Status: CANONICAL. Jesse-ratified 2026-07-22** (Fable session, relayed via treasurer). This is the
model every fleet owner follows when a Taey failure appears in their system. It supersedes the
"pick one of three levers" reading of `training-defect-triage`: the levers still exist, but they now
pick the **type** of training produced — never *whether* to train.

> **THE PRINCIPLE: every Taey failure trains. Nothing is fixed-and-forgotten.**
> The lever you pull to make Taey succeed *today* decides the **type** of row you write — it does not
> decide whether a row gets written. It always does.

---

## 1. EVERY FAILURE TRAINS — the lever picks the TYPE

For each failure, first make Taey able to succeed with the tech and hardware that exist today (the
three levers, in order: fix our infra → fix the system prompt → train). Then, whichever lever landed
it, capture the matching row type:

| What made Taey able to succeed | Row TYPE to author | Why |
|---|---|---|
| **Infra bug fixed** — a bug in our code/data was the cause; once fixed, Taey succeeds **unchanged** (no behavior change needed) | **SPEC-KNOWLEDGE row** (`spec_knowledge_v1`) — how the subsystem SHOULD work + what correct output looks like + the story behind the diff | No behavioral pair, **zero negation hazard**, cheap dose. Powers **Taey-as-regression-detector**: "they tell you what you told them" — Taey holds the spec and flags when a later diff violates it. |
| **Prompt-fixable** — the capability existed; Taey only lacked a pointer, and a system-prompt block fixed it | **Behavioral pair** (`operator_practice_v1`) **+ a system-prompt placeholder** (see §2) | The pair moves the pointer into the weights so the prompt block can later be evicted; the placeholder gives relief *now*. |
| **Beyond-prompt** — a genuine capability gap the prompt could not close | **Straight behavioral training** (`operator_practice_v1`) | The weights are the only destination; there is no prompt shortcut. |

**The standing exception — an infra GAP that cannot be fixed with today's tech.** If Taey *cannot*
succeed even after our best effort (e.g. a data key that does not exist anywhere), **file it, hold the
rows, do NOT train yet** — training around a missing capability manufactures *confident failure*.
This is not an exception to "nothing is fixed-and-forgotten": the moment the gap is closed it becomes
an **infra-bug-fixed** case and gets its SPEC-KNOWLEDGE row. It is only *deferred*, and it lives
visibly in `TRAINING_BACKLOG.md` until then.

### The SPEC-KNOWLEDGE row (`spec_knowledge_v1`) — the new, cheap, high-value type
This is the type most failures produce, because most failures are infra bugs we fix. It teaches Taey
the **correct mechanism as fact**, so Taey becomes the regression detector for that subsystem.

```json
{
  "schema": "spec_knowledge_v1",
  "subsystem": "e.g. careers/pairs-manifest",
  "date": "2026-07-22",
  "spec": "how the subsystem SHOULD work — the correct mechanism, stated declaratively as fact",
  "correct_output": "what correct output looks like — an example or the shape",
  "story_behind_the_diff": "CURATION METADATA ONLY (not emitted): what changed and why (the fix)",
  "class": "spec-knowledge",
  "curated": true,
  "source": "provenance — commit SHA / committed doc / first-hand"
}
```

- **Emitted training text = `spec` + `correct_output` only.** `story_behind_the_diff` is curation
  metadata (like `agent_action`/`correction_verbatim` in `operator_correction_v1`) — it exists so a
  human can trace the row to a real event; it is **never** emitted into training text. That is what
  keeps the negation hazard at zero: the row is pure declarative spec, no defect narrative.
- Residue-gated the same as everything else (`derive_training_rows.py`) — the emitted text must carry
  no failure vocabulary.

### Mechanical contract: META IS NEVER TOKENIZED (treasurer, 2026-07-22)
The whole "curation metadata never trains" model — and treasurer's per-row sanction — rests on this,
so it is a **mechanical** guarantee, not a convention. **Only `messages[]` reaches the tokenizer.**
Verified in the trainer's loader (`dense-9b/trainers/train_fsdp_dense_9b.py`): each row is read as
`msgs = row.get("messages")` (L340) and only `msgs` is tokenized via `_tokenize_sft_pair` → the chat
template (L357, L160). `row["meta"]` is read **solely** for `meta.lane` routing (L341) and is never
tokenized; the top-level `tools` schema is injected into the template on purpose (L350). Everything
else in a row — `meta.class`, `meta.source`, `provenance_hash`, and (were it ever present)
`story_behind_the_diff` — is dropped at load. Double safety on spec rows: the deriver never writes
`story_behind_the_diff` into the emitted file at all. **If the loader/tokenizer path ever changes,
this contract is a gate: re-verify that no non-`messages` field can reach the tokenizer.**

---

## 2. THE SYSTEM PROMPT = a placeholder + a CHANGELOG section

Jesse: *"YES, that definitely needs to be a section."* The prompt is a **cheat sheet for the delta**,
not a growing permanent spec.

- **The weights hold the consolidated spec.** Everything already trained lives in the weights, not
  the prompt.
- **The prompt permanently holds ONLY the delta since the last training round** — a CHANGELOG section
  of what has been added but not yet consolidated into weights.
- **Each training round consolidates and clears it.** After a round trains the accumulated deltas, the
  changelog section is emptied (per the eviction test in §3). The prompt never grows without bound;
  it is always "just the un-trained delta."

The live prompt is at `<MIRA_HOME>/data/corpus/layer_1/SYSTEM_PROMPT.md` (a non-git dir); every
injection is versioned in `careers-qwen/system_prompt/PROVENANCE.json` + `versions/`. **Known path
blocker:** production careers callers define their own prompt and do NOT load that file — mirror any
changelog line into the caller's own prompt or the lever is inert on that path.

---

## 3. EVICTION IS MECHANICAL — a paired probe ships WITH every prompt line

No judgment call about whether a prompt line is "still needed." It is decided by a probe.

1. **At creation:** every prompt addition ships **with a paired eval probe** — a concrete test that
   fails if Taey lacks the capability the line supplies.
2. **Post-round:** run the probe **with the prompt line removed**.
   - **Pass → EVICT** the line (the capability moved into the weights; the changelog entry clears).
   - **Fail → KEEP** the line **and raise a gap flag** (training did not consolidate it; it needs
     another, possibly deeper, round).

This is what makes §2 real: the changelog clears mechanically, not by opinion.

---

## 4. OWNERSHIP — owners author for their OWN systems; tutor is the custodian, not the gatekeeper

Jesse correction: owners **author training for their own systems directly** via the skills
(`training-defect-triage` → `taey-training-trigger`), **without routing through tutor per-row.**

- **Tutor = recipe / validator / mixture / registry custodian.** The enforcement surface is the
  **residue gate + schemas + registry**, NOT tutor's attention. An owner does not wait on tutor to
  write a row; the gate and the manifest enforce quality mechanically.
- **Tutor still owns:** the mixture and the dose (the Chats' call per the never-again rules), the
  optimizer/recipe, the residue gate, the schemas, and the registry that fails on drift.
- **Owners own:** the domain judgment — is this an infra bug, prompt-fixable, or beyond-prompt? — and
  authoring the correct row type for their subsystem, stored in the right place (§5).

---

## 4b. CPT RUNS FIRST — and every pair carries the world it came from

**Order (Jesse, 2026-07-24): CPT first, then SFT.** Continued pretraining teaches the current
correct world — the code as it now stands, the current documentation, the diffs that changed
them. SFT then teaches conduct on top of a world the weights have actually seen. Run SFT first
and you are teaching correct behavior about a system the model does not know.

**Every training pair must carry its world.** When a row is authored, stamp `meta.source` with
what produced it, specifically enough that a tool can resolve it:

- the **commit SHA** of the change that made the row necessary
- the **file paths** the row's behavior touches
- the **document** that is canonical for that surface
- the production event: run id, ledger timestamp, walk date, receipt path

That stamp is not bookkeeping. `careers-qwen/build_cpt_from_sft.py` reads an SFT corpus and
derives a CPT delta from exactly those fields — commit SHAs become diffs (why the row exists),
file paths become current contents (the correct state now). **A row with no source contributes
nothing to that delta**: the behavior gets trained while the world behind it stays invisible.

The tool reports coverage and warns below 80%. At the time this was written the module-3 corpus
sat at 30/159 — so most of that corpus taught conduct with no world attached. Stamp at
authoring time; it cannot be reconstructed reliably afterwards.

**Two guards that belong to the coupling, learned in its first run.** The builder must refuse
any commit touching a data path, and must cap by BYTES as well as lines — a `.jsonl` diff is
enormous single lines, so a 400-line cap admitted 16 MB of training rows on the first attempt.
And CPT weighting follows the pair-generation surfaces: whatever produced this round's SFT rows
gets repetition and refresh next round, with identity and constitutional material introduced
gradually as low-share riders that grow across rounds.

## 5. THE ONE PLACE — everyone feeds ONE governed store; the TRAINING DATA IS PRIVATE

**Jesse, 2026-07-22:** *"everyone needs to be on the same page… it needs to feed it into one place.
Do not overcomplicate this. The training is not to be public."*

There is exactly ONE training-data store, and it already exists and is governed — do not create a
second one. **Tooling is public; the training DATA is private.**

- **THE ONE PLACE (private, treasurer-owned):** `treasurer/foundations/careers/training_data/`
  (`GOVERNANCE.md` — "nothing trains from anywhere else"). The `.jsonl` pairs are **gitignored =
  never public**; they live in `v2/pairs/`. The single master index is **`REGISTRY.md`**
  (registered-or-nonexistent). Runs append to `runs/RUN_REGISTRY.md` (tutor-owned). Corpus is
  treasurer-sanctioned (never-again rule: tutor never assembles data).
- **TOOLING is public (build-in-public, `palios-training`):** the deriver (`derive_training_rows.py`),
  the residue gate, the schemas, and this doctrine. **No pair `.jsonl` is committed to
  `palios-training`** — that repo carries the CODE, not the training data.

### The one motion for every owner
1. **Author your source rows** using the right schema (`spec_knowledge_v1` for infra-bug-fixed;
   `operator_correction_v1` for behavioral), per `taey-training-trigger` (quality bar + validator).
2. **Derive** with `derive_training_rows.py` — residue-gated, strips curation metadata
   (`story_behind_the_diff` never emitted). This is the quality gate; it stays public tooling.
3. **Place the emitted `.jsonl` in the ONE store** (`training_data/v2/pairs/`, gitignored = private)
   and **add its `REGISTRY.md` row** (id / version / counts / sha256 / status). An unregistered
   dataset does not exist.
4. **Treasurer sanction + Chats mixture/dose** before it trains (their call). Then it can serve.
5. **If you cannot author it yet** (open infra gap): note it in `TRAINING_BACKLOG.md`'s debt column.
   Nothing is fixed-and-forgotten.

No per-row hand-off and no second registry — the residue gate is the quality gate, `REGISTRY.md` is
the one index. *(The earlier `careers-qwen/data/` + `PAIRS_MANIFEST.md` was a parallel second store
in the public-bound repo — superseded by this section; it is being converged into the one governed
private store with treasurer. Do not add new pair data to `careers-qwen/data/`.)*

---

## Provenance & related docs
- Ratified: Jesse, 2026-07-22 (Fable session), relayed by treasurer as canonical.
- Row schemas + capture detail: `data/corrections/CORRECTION_SPEC_v1.md` (`operator_correction_v1`).
- Triage (which lever / which type): skill `training-defect-triage`.
- Authoring + validator + registry step: skill `taey-training-trigger`.
- The pathology this model is built around — **Negation Neglect** (training a defect's verbatim record
  internalizes the defect) — is why SPEC-KNOWLEDGE rows emit spec-only and never the failure story.

## Revenue weighting — measured, not assumed (2026-07-25)

Training weight follows the BINDING constraint in the revenue pipeline, not the lane
where pairs are easiest to generate. Those are different, and confusing them is the
proxy trap: scoring produces pairs cheaply, which is exactly why it over-attracts effort.

**Measured by treasurer against the ledger of record** (`foundations/action_logs/
applications.jsonl`, 45 entries), not against a DB field:

| | value |
|---|---|
| real submissions, per-day HISTORICAL counts | **1–7 across ten dated days** |
| already-scored rows waiting unapplied | **603** |
| ratio | scoring runs **~200x ahead** of what the pipeline converts |

A `updated` DB field had suggested 44/day. It is not an apply timestamp, and a documented
bulk backfill is indistinguishable from a high run-rate through it. Treasurer flagged the
number as untrusted BEFORE using it, then verified against per-application receipts.
**Do not cite 44/day or anything derived from it.**

**These are HISTORICAL COUNTS on dated days, NOT a sustained run rate** (treasurer's own
caveat on their own figures). Why the gap exists is the open question they are going to
answer, so treating the counts as a rate would presume the answer. Do not average them
into a throughput number and do not project from them.

**Consequence, binding on the next SFT module:** weight UI-action reliability and compose
throughput. Do NOT add scoring capacity — it is not merely sufficient, it is two orders of
magnitude ahead of the constraint, so scoring rows are close to zero marginal revenue.

**Sequencing (Jesse, same day):** Taey gets the mechanics down first. Infra-feel /
predict-the-machine work is introduced GRADUALLY through training and must not displace
revenue mechanics. In the sanctioned cpt_refresh_v2 that means substrate_physics at
**0.16%** of tokens against sft_delta at 64.57% — a trace introduction, expressed in
numbers rather than intent. Response rate remains **Unknown** until treasurer's outcome
loop exists; it is not to be guessed at to fill a training rationale.
