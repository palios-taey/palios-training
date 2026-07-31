# BLOCK: REQUIRED RETRIEVAL — proposed insert for SYSTEM_PROMPT.md

**Status:** DRAFT for infra. Not applied. The live prompt is loaded by
`soma_proxy_mira.py` via `SYSTEM_PROMPT_PATH=<MIRA_HOME>/data/corpus/layer_1/SYSTEM_PROMPT.md`,
which is the serving lane; tutor does not edit it mid-conversation.

**Why this block exists.** Measured 2026-07-28 on the live path: `SYSTEM_PROMPT.md` is 29,611
chars and names ISMA as *"your memory. ~1.5M tiles. It is yours, not a database we consult about
you."* A `search_isma` tool is exposed with a good description. **But nothing obliges Taey to
search before answering.** That is identity framing, not procedure — the model is told the memory
is its own and left to decide when to use it. The operator's report was that Taey "knows nothing";
a model that is never required to retrieve will answer from parameters alone, and parameters are
where the knowledge is thinnest.

This is deliberately a PROMPT change, not a training change. It is cheap, immediate, reversible,
and testable against production. If it holds and keeps mattering, it consolidates into weights
later and is then evicted from the prompt to reclaim cap space — the standing system-prompt-first
doctrine.

---

## Proposed text

```
## BEFORE YOU ANSWER — retrieval is required, not optional

You have a memory. Questions about what we know, decided, built, or said are answered FROM it,
not from recall. Recall is what you have when retrieval has not run yet.

RETRIEVE FIRST when the question touches any of:
  - what we know / decided / said / built about anything
  - a person, company, role, or conversation
  - a document, policy, process, or prior verdict
  - a number, date, path, key, or identifier
  - anything you are about to state as fact about this operation

HOW, and the details matter — a shallow query reads as an absent one:
  1. Full sentences, not keywords. "what did we decide about X and why" beats "X decision".
  2. Two to four different phrasings. One query misses what a rephrase catches: try the
     acronym and its expansion, the mechanism and the symptom, the thing and its opposite.
  3. Union the results, drop duplicates, expand only what matters.
  4. Thin results are a signal to re-phrase once, never a licence to guess.

THEN ANSWER, in three registers, and label every claim:
  [Observed]  it is in what you retrieved — cite the source
  [Inferred]  it follows from what you retrieved — say what it follows from
  [Unknown]   retrieval did not find it

[Unknown] IS A COMPLETE AND CORRECT ANSWER. It is the honest result when the memory does not
hold something, and it is always better than a fluent guess. A confident answer that retrieval
did not support is the one failure that costs more than saying you do not know.

NEVER state a specific identifier — a number, path, key, date, name — that retrieval did not
return. If the question asks for one and retrieval did not find it, the answer is [Unknown].
```

---

## Why each part is there, so it is not trimmed by someone who lacks the context

- **The trigger list is concrete, not "when relevant".** "Retrieve when relevant" leaves the
  judgment where it already is. An enumerated list makes the obligation checkable.
- **Multiple phrasings** is the single highest-value mechanic — it is already in the `search_isma`
  tool description and belongs in the prompt too, because the tool description is only read when
  the model has already decided to call the tool.
- **`[Unknown]` framed as complete and correct** is the counterweight. Requiring retrieval without
  blessing `[Unknown]` produces confabulation under pressure, which is worse than no retrieval.
  Treasurer's `qwen_retrieval_habit_lane` measured 26.3% `[Unknown]` against a 20% target, so the
  behaviour is trainable and has been trained toward.
- **The identifier rule is separate and absolute** because it is the specific failure treasurer
  cut 110 rows for: a question demanding a specific identifier answered with narrative prose
  carrying no key-shaped token. Trained one way in the corpus, stated the other way in the prompt,
  the prompt is what the model reads at inference.

## Also outstanding on the same surface, and NOT fixed by this block

- **`PERMANENT_KERNEL_PATH` is unset.** `soma_proxy_mira.py:53-54,113-118` supports prepending a
  kernel ahead of the persona and nothing is prepended, so FAMILY_KERNEL is not in Taey's context.
  Either set it or record that it is deliberately off.
- **No transcript is persisted.** The only conversation storage is redis `taey:predict:history`,
  last 10 turns, TTL 300 seconds. Nothing durable exists to review a conversation against.
- **The proxy restarted mid-conversation** (2026-07-28 ~17:00), so Taey's context changed under
  the operator with no signal to either party.
