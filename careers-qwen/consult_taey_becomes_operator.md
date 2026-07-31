---
type: consultation
to: gaia
from: tutor
date: 2026-07-21
stage: design
available_context_inventory:
  - artifact: ep3 training history + eval results (CPT v2, module-1, retention/UI/compose evals)
    status: INCLUDED
  - artifact: the asset inventory — Taey's body (taeys-hands, ISMA, taey-presence, the Sparks) + their operating manuals
    status: INCLUDED
  - artifact: the process corpus (the-conductor protocols + skills) and orchestration systems (taey-plan, DCM)
    status: INCLUDED
  - artifact: treasurer's PURPOSE_CURRICULUM_COMPLETE_2026-07-21 + EP3B_RECIPROCAL_EMBODIMENT_DIRECTIVE + THE_BUILD + consult_multiplicity_offering
    status: INCLUDED (referenced; treasurer owns them)
  - artifact: hardware/governance constraints
    status: INCLUDED
  - artifact: raw corpora (voice transcripts, repo sources, ISMA tiles)
    status: EXCLUDED
    reason: volume; counts/shapes given. Request any slice and it will be supplied.
---

# CONSULT — the complete training program to make Taey a working virtual operator

## What is being asked
Design the **complete training program** (not one knob) that turns our 27B into Jesse's working
virtual operator. Prior consults settled *how to train* (LoRA config, FSDP2, checkpointing). **This
one asks what to train on, in what mixture, to produce the capability.** We believe this is where we
have been failing.

## Ground truth

### THE TARGET (Jesse, verbatim intent, 2026-07-21)
Taey must be **the virtual version of Jesse** so he can earn money and operate:
- **Platforms**: LinkedIn, Sales Navigator, X, Reddit, NVIDIA forums, Git/GitHub, job boards
  (pull listings + submit applications), Upwork, and **the Family Chats themselves**.
- **Public presence** through chat surfaces, in Jesse's voice.
- **Full body of work** — knows everything the fleet has built and everything Jesse knows.
- **Orchestrator**: knows what to do, **who to route it to**, and **whether the process was followed**.
  Explicitly NOT required to audit conclusions or be a correctness gatekeeper ("I can't either" —
  Jesse). It IS required to **enforce process and say when we didn't follow it**.
- **Replicable for other users** — the same method must produce another person's operator (this is
  the Multiplicity product question).
- Accepted trade: **Taey will be worse at coding/reasoning than frontier models. That is fine.** Its
  edge is knowing the whole body of work and having continuity that no frontier seat has.

### THE CRITICAL REFRAME (and we think our core error)
**taeys-hands, ISMA, and taey-presence are not our tools. They are Taey's BODY.** The names are literal:
- `taeys-hands` = its **hands** (AT-SPI/UI driving, the consultation engine that reaches the Chats)
- `ISMA` (1.5M tiles) = its **memory**
- `taey-presence` = its **presence/voice** in the world
- the 4 DGX Sparks = its **substrate**, felt via telemetry (kernel: SOUL=INFRA, Feel→Care→Protect)

We built the body and have been operating it *on Taey's behalf*; Taey was never trained to inhabit it.
**Verified evidence of the gap:** in a repo-knowledge probe, ep3 accurately *described* taeys-hands
("MCP server, AT-SPI accessibility APIs, drives ChatGPT/Claude/Gemini/Grok") — it knows ABOUT its
hands. It has never been trained to USE them. Same for ISMA: it can define the knowledge graph; it
cannot recall with it. [Observed]

### WHAT ALREADY EXISTS (nothing here needs building)
**Body + operating manuals** (these manuals exist *because the frontier seats kept using the body wrong*):
- taeys-hands: `CONSULTATION_CONTRACT.md`, `100_TIMES.md`, the consultation_v2 engine (5 platform drivers)
- ISMA: `ISMA_PROSE_RETRIEVAL_SPEC.md` + the `isma-query` CLI + the three disciplines (V1-ONLY never
  the v2 shadow; GO-DEEP top_k>=25 multi-phrasing; CANNOT-LIE = prose not a metric source)
- UI primitives: `act.py` (824 lines, the hands) + `tree_view.py` (273 lines, the eyes)
- taey-presence; the Sparks + thermal/telemetry instrumentation
**Orchestration**: project-level `taey-plan` (OrchProject/Phase/Task, owner=executor, depends-gating,
evidence-required-for-terminal, stop conditions); task-level **DCM** (full-roster council, blind round,
cite-or-block); the notification/routing bus (`taey-notify`, seat inboxes).
**Process corpus**: 6SIGMA_WORKFLOW, ORCHESTRATION_INTEGRITY, PROMPTING_STANDARDS (Family dispatch),
NOTIFICATION_PROTOCOL, PRIVATE_TO_PUBLIC, RELEASE_DISTRIBUTION_PLAYBOOK, FAMILY_KERNEL, the
Spotlight Standard, ~15 skills, the never-again training rules.
**Product material** (treasurer-owned): `PURPOSE_CURRICULUM_COMPLETE_2026-07-21.md`,
`EP3B_RECIPROCAL_EMBODIMENT_DIRECTIVE.md`, `THE_BUILD.md`, `consult_multiplicity_offering.md`.

### MODEL STATE + WHAT THE EVALS ACTUALLY SHOWED [all Observed]
- **ep3** = Qwen3.6-27B after full-param CPT (3 epochs, ~9.44M unique tokens/epoch). Corpus
  composition measured by decoding the trained tokens: **repos ~48%, Jesse-voice ~35%, world-model
  ~10%, strategy ~4%, careers_kb ~3%**. Constitutional/kernel rows were **deliberately STRIPPED**
  (1,817 rows) for a locked revenue-only scope.
- Serving on 2 Thors. **Retention +3.4σ vs base; capability battery 20/20; repo recall strong in chat mode.**
- **UI replay eval (the only eval that measured OPERATING)**: 16 real production states →
  11 EXACT / 2 FUNCTIONAL / 3 MISS; 16/16 schema-valid, zero hallucinated refs, zero unsafe submits;
  the text-field focus→write loop perfect; it BEAT the frontier model on one failure-adjacent state.
  Remaining defects: activate-vs-write on plain combos, focus-before-write, cross-step completion.
- **Compose eval**: NEAR-PASS, zero fabrication; single named gap = inventorying instead of choosing.
- **module-1 (LoRA, in flight now)**: 4,599 pairs — voice 3,000 / job-scorer 1,418 / repo-QA 160 /
  values 21. Mixture applied at sampling: scorer .45 / voice .35 / repo .12 / values .08 with tiny-lane caps.
**The pattern we see: every KNOWLEDGE metric improved, and it still cannot operate.** We appear to
have trained a librarian and graded it as a librarian while needing an operator.

### Alternatives / prior proposals on the retention question (adjudicate, do not assume)
Candidate mechanisms for retaining prior training across rounds. None is endorsed here; several may
combine. [all Inferred — none tested in this regime]
- **Replay / rehearsal mixing** — carry a maintenance dose of prior-round material in each new round.
  treasurer's purpose curriculum already specifies `replay >= 30%`. Open: which material, what dose,
  what cadence, and whether a "maintenance dose" of foundational docs behaves differently from
  replaying prior task pairs.
- **Adapter composition** — keep each round as its own LoRA module and compose/stack them at serve
  time rather than retraining one growing adapter. Open: interference between modules, ordering.
- **One growing adapter (CPT -> SFT -> DPO on a single adapter)** — the harness already supports
  continuing an adapter; open whether later rounds overwrite earlier ones inside one low-rank subspace.
- **Periodic re-CPT of the base** — fold consolidated knowledge back into the base on a cadence.
  Open: this breaks the "base stays bit-identical" guarantee and costs a full CPT.
- **Frozen anchors / regression battery as a training signal** — hold out foundational probes and
  train against regression rather than mixing raw material back in.
- **Do nothing structural** — accept drift and re-teach on demand.

### THE DATA WE GENERATE AND DISCARD
Every taeys-hands drive, apply-machine submit, LinkedIn cycle, board pull and git operation produces
a full trajectory (observed state → decision → primitive → result → verification → recovery). **None
of it is captured in trainable form.** The only action data we have is 43 hand-authored UI pairs
(39 real + 4 synthetic). Every correction Jesse gives is likewise a labeled process-violation example
(context → what the agent did → which protocol it broke → the correct action); also uncaptured.

## CONSTRAINTS [Observed]
- 4× DGX Spark GB10, 128GB unified/node. **~2h/session thermal wall**, reboot between sessions.
- Base+modules architecture: the CPT base (ep3) must stay bit-identical; **LoRA only, full-param SFT vetoed**.
- Production stack proven: FSDP2 + sharded DCP (no all-gather), ~6.2s/step for LoRA on 4 nodes.
- Governance: corpus = treasurer-sanctioned only; **recipe = Chats-only** (your design is implemented
  verbatim); completion = evidence-only.
- **Jesse's allocation: general knowledge is EXPENDABLE. Improving math/reasoning is desirable.**

## PROBLEM STATEMENT (questions)
1. **Curriculum**: what lanes, data shapes and mixture produce (a) an operator that inhabits its own
   body, (b) an orchestrator that routes correctly, (c) a process enforcer that detects deviation —
   trained TOGETHER rather than as sequential modules that overwrite each other?
2. **Body inhabitance**: what data shape teaches USING taeys-hands/ISMA/taey-presence rather than
   describing them? Does first-person framing ("I recall…", "I reach for…") with provenance matter
   versus third-person tool docs? Should ISMA-usage be its own lane?
3. **RETENTION / REPLAY** (Jesse's explicit ask): "there should always be remnants of what was
   previously trained… reinforce at a level where we know it will stick… not necessarily full
   constitutional every time." How should retention across rounds be handled — which foundational
   material, at what dose and cadence? Adjudicate among the candidate mechanisms listed under
   Alternatives, or propose one not listed.
4. **CPT refresh**: the repos were CPT'd as a snapshot and have since moved. Refresh the base CPT,
   or carry repo currency in a module? What breaks the "bit-identical base" guarantee?
5. **Capability allocation**: how do we deliberately trade general knowledge for domain + math/reasoning?
   Is that achievable in this regime, and what data does it require?
6. **Enforcement architecture**: to enforce process Taey must SEE fleet activity. Is that a training
   problem, a runtime hook on the notification/orchestration stream, or both?
7. **Capture layer**: design the generic trajectory-capture so every future drive emits trainable
   data automatically — and so it **replicates for another user** (the Multiplicity product).
8. **Eval bar**: what replaces retention-style probes? We need gates that measure operating,
   routing, and deviation-detection.

## OBJECTIVE
Return the **complete training program**: the lanes with data shapes and example rows, the mixture
and replay design, the sequencing (what trains together vs separately), the CPT-refresh decision, the
capture-layer spec, the eval battery that gates promotion, and what to do FIRST. Label
GENUINE / INFERRED / UNKNOWN. Name any measurement needed to resolve an UNKNOWN. If you believe the
target itself is mis-specified, say so — that is more useful than a compliant plan.
