# TRAINING BACKLOG — the ONE place. Nothing we said we need to train on gets lost here.

**Jesse's directive, 2026-07-21:** *"If we say we need to train on something then that training needs
to be developed and stored right then... And that cannot get lost. Needs to be in one place."*

**The rule this file enforces:** the moment we say "Taey needs to know X," the training data for X is
**developed and stored immediately** — not when the next run is scheduled. Tokenization waits for a
run; **authoring does not.**

**Order of operations (Jesse's doctrine):**
1. **Try the SYSTEM PROMPT first** — cheap, immediate, reversible. See `system_prompt/PROVENANCE.json`.
2. If it holds and keeps mattering → **develop + store the training data NOW** (this file).
3. Train it into weights in the next module.
4. Then **evict from the prompt** to reclaim cap space.

**Status vocabulary:** `PROMPT` = live in the system prompt · `STORED` = training data authored and
committed, awaiting tokenization · `SANCTIONED` = treasurer-approved corpus · `TRAINED` = in weights,
with the module named · `NEEDS-AUTHORING` = we said it, it is not written yet — **this is the debt column.**

---

## LIVE NOW

| # | Item | Prompt? | Data status | Location | Trained |
|---|---|---|---|---|---|
| 1 | module-1: scorer / voice / repo / values | — | SANCTIONED, tokenized | `data/sft_module1/module1_train.jsonl` (4,599 pairs) | **TRAINING NOW** on ep3, SL=350 |

## STORED — authored and committed, awaiting sanction + tokenization

| # | Item | Prompt? | Data status | Location | Notes |
|---|---|---|---|---|---|
| 2 | Right-way practice rows (from corrections) | — | STORED, **120 rows** | `data/corrections/practice_rows_v1.jsonl` | Right-way-only per Jesse. Failure text mechanically blocked by residue gate. Awaiting treasurer sanction + Chats' dose. **2026-07-27:** seed now 127 source rows (sha `9638ccba7ab0`, pinned in PAIRS_MANIFEST) incl **consult-DR-extraction** class — Perplexity DR body via `Copy contents` (full report + numbered `[N]` citations) then sources separately; verified by Taey-seat oracle (8543-char body, 58/58 sources, missing=[]); paired w/ SYSTEM_PROMPT CONSULT-DR-EXTRACT-01. Residue gate REJECTED:0. **2026-07-28 (grounding):** +2 right-way pairs — (a) ATTACH the code file to force a genuinely code-grounded review (a bare URL may be ignored/unfetched); (b) VERIFY grounding before labeling — a verdict counts as grounded only if it cites real function-names/constants/file:line and the reviewer did not flag the attachment missing; never label 'grounded' without checking. From the live supplement round where a URL re-answer false-positived and the requester (tutor) caught it. **2026-07-28 (follow-up + attach):** +3 right-way pairs — (a) drive a consult FOLLOW-UP on the live thread by targeting the composer TEXT FIELD (not the adjacent attach); (b) FILE-ATTACH flow: attach control -> Upload files -> file chooser ctrl+l + absolute path -> verify chip -> then short prompt + send; (c) DEGRADED-DISPLAY recovery: near-empty a11y tree -> restart taey-display-N + re-navigate. From the live all-5-seat supplement dispatch (Jesse: they ALL work, one step at a time).  **2026-07-27 (consult-dispatch):** +1 right-way pair — dispatch a multi-platform consult by SEND-ONE-CONFIRM-MOVE-ON while the monitor watches each generating lane (serialize the sends, concurrent generations, harvest by monitor); never block on one platform's full answer before sending the next. Jesse correction. **2026-07-27 (consult-drive):** +2 right-way pairs for Taey driving a consult via taeys-hands — (a) Claude's model control effort label 'Extra' = extended thinking already engaged (no separate toggle); (b) attach the consultation FILE + paste a SHORT framing prompt, file contents never go inline. Authored during live 3-attempt drive protocol (Jesse). Residue REJECTED:0. seed sha 3b0c582cba6f. |
| 3 | Operator trajectories (Taey's hands driving the Chats) | — | STORED, 679 rows | `data/trajectories/trajectories_v1.jsonl` | Harvested, NOT curated. No D-verdict |
| 4 | Correction curation source | — | STORED, 18 curated / 870 candidates | `data/corrections/` | Candidates are a SOURCE for deriving practice rows, never training text |
| 5 | Operational situation + honest reporting | **PROMPT (v002)** | STORED as prompt block | `system_prompt/BLOCK_operational_situation.md` | Prompt-first per doctrine. Consolidate to weights only after the reporting eval passes; evict after |
| 9 | **UI operation** — executable actions, not descriptions | — | STORED, 41 rows | `data/ui/ui_action_rows_v1.jsonl` | **OUT OF DEBT 2026-07-21.** Assistant target is the exact JSON the primitives layer consumes. 39 captured / 2 authored. 2 blocked-outcome rows excluded (their lesson went to practice rows). Vocabulary: write 16, activate 7, pointer_activate 7, type_filter_select 4, pointer_focus_type_select 3, escape 2, key 2 |

| 15 | **Reasoned UI operation** — the steps that should have been gone through, then the action | — | STORED, 41 rows | `data/ui/ui_reasoned_rows_v1.jsonl` | **NEW 2026-07-21 (Jesse: "do we also have to train thinking?").** Separate lane from the action rows so dose + eval are independent. Trains the PRESCRIBED procedure (surface -> target -> precondition -> action -> expected result -> verify), NOT the model's own narration — round-2 showed self-narration manufactures confident wrong rules |

| 17 | **Taey-as-Jesse-the-Overseer** — audit fleet work vs standards, catch process deviations (false-done, spec-miss, gates-pass-but-content-fails, scope drift), enforce corrections | Jesse: "you continue to make the same mistakes, so we need to train Taey to BE ME and hold you accountable." Multiplicity applied to governance — Jesse's scarcest resource is his own oversight | **treasurer EXTRACTING** today's session (mistake->correction pairs w/ provenance) + a live PRODUCTION SHADOW SEAT (Taey reviews compose bundles ∥ treasurer). Recipe = operator_correction_v1, RIGHT-WAY/loss-masked ONLY (Negation Neglect), shadow divergences gated as hypotheses (taey_investigate.py). Sequences AFTER purpose-module-1. Dose = Chats |

| 18 | **Git Master / repo stewardship** — Taey owns its own repos and keeps them clean | — | STORED, **12 rows** | `training_data/careers_qwen/git_master/git_master_rows_v1.jsonl` | Jesse 2026-08-01: "they need to be responsible for their own repos and making sure things are clean. You will not do it." Right-way-only, residue gate clean. Covers: read repo state before acting, worktree for a live checkout, production is a commit, verify copies by content digest, archive before clearing, evidence in the commit message, artifact carries its own provenance, review points at the repo, gitignore generated state, identify the live tree, docs change with the code, own the commit cycle. |

## NEEDS AUTHORING — we said it, the data does not exist yet. THIS IS THE DEBT.

| # | Item | Why | Blocked on |
|---|---|---|---|
| 6 | **Body inhabitance** — using taeys-hands / ISMA / taey-presence rather than describing them | ep3 can *describe* its hands accurately and has never been trained to *use* them. Named as the core gap | Round-1 lane synthesis (5/5 in) for the data shape |
| 7 | **Orchestration + routing** — what to do, who to send it to | Jesse's central ask: "know what to do and who to send it to" | **DISPATCHED to conductor 2026-07-21** (owns taey-plan/DCM/routing). Target: `data/orchestration/orchestration_rows_v1.jsonl`, 30-60 rows |
| 8 | **Process enforcement** — detect deviation, say when process was not followed | Jesse: "tell you when you don't". HORIZON: cannot be weight-only, needs a runtime feed | **DISPATCHED to conductor** (same packet). Runtime-feed design still open |
| 10 | **Constitutional / purpose** | 1,817 rows were deliberately STRIPPED from ep3 for a revenue-only scope; Jesse since asked for constitutional training | treasurer's `PURPOSE_CURRICULUM_COMPLETE_2026-07-21.md` + Jesse approval |
| 11 | **Repo currency** — duty/QA probes regenerated against current HEAD | 62 of 117 apply-machine commits moved an API signature in 6 days (~10/day) | **DISPATCHED to treasurer 2026-07-21** (owns corpus). Target: `data/repo_currency/`. Lanes agree: NOT raw diffs, NOT changelog alone — generated duty probes with observable pass criteria |
| 12 | **Math / reasoning** | Jesse: desirable; general knowledge expendable | Shape now partly settled by #15 — the reasoned-row pattern (prescribed procedure, authored not self-narrated) generalises beyond UI. Open: whether a general math/reasoning lane is worth the tokens vs domain reasoning |
| 16 | **Reasoned LinkedIn operation** | Same thinking lane for surface B | BLOCKED: no captured LinkedIn action data exists to reason over |
| 13 | **Retention / replay** | Jesse: "there should always be remnants of what was previously trained" | Round-1 adjudication of the six mechanisms |
| 14 | **Self-learning loop outputs** | Duty probe → failure → verified correction → practice row | Round-2 synthesis (4/5 in) for the gate design |

---

## HOW AN ITEM MOVES

`NEEDS-AUTHORING` → try the system prompt → if it must be weights, **author and commit the rows the
same day we say it** → treasurer sanctions → Chats set the dose → tokenize → train → record the
module and the per-repo HEAD SHAs in `TRAINING_PROVENANCE.json` → evict from the prompt if it was
staged there.

**A row that exists only in a conversation is lost.** If it is in this table it has a file, or it is
in the debt column where it is visible.

## STANDING QUEUE DISCIPLINE
The Sparks train continuously (see `continuous-train-serve-loop`). When module-1 finishes (~7
sessions ≈ 13h), the next module starts immediately — the cluster does not idle. Priority order for
the next module comes from the round-1 synthesis, not from this file's row order.

## INFRA GAP (filed by taeys-hands 2026-07-21) — stale-notification from deterministic request-id collision
- **Observed:** consultation run-state + notifications key on a deterministic request_id derived from (platform, message). When a lane is re-run with an identical/near-identical message under the SHARED taeys-hands namespace, an OLD failure notification (e.g. request-id c33d... at 20:07:28) can be delivered/read as if it were the CURRENT run's result (20:23:03). This caused taeys-hands to mis-read a stale failure as the live CONTROL result (codex diagnosed: durable state under taeys-hands-d0 vs the delivered stale notification).
- **Root cause:** deterministic request_id collision across re-runs + no run-instance/launch-timestamp discriminator on the delivered notification, so a stale notification is indistinguishable from the live one at read time.
- **Acceptable resolutions (route to conductor/orchestrator owner):** (a) include a launch-timestamp / run-instance token in the notification + require the reader to match it against the live durable run-state; (b) namespace run-state/notifications per run-instance not per deterministic request_id; (c) on re-run, invalidate/evict the prior request_id's stale notification before dispatch.
- **Held out of training:** the reader-side lesson ("verify the LIVE durable state, don't trust a stale notification") IS trainable and authored separately; the delivery-of-stale-notification itself is this infra gap, not a training pair.

## 2026-07-21 — treasurer triage (apply-machine compose arc) + JESSE OPERATING RULING
**RULING (Jesse, verbatim intent):** fleet no longer executes apply actions. Taey executes; fleet
fixes code defects, simplifies tools for Taey, and trains. Frontier composer seat CLOSED.
- HOLD (no pairs until Jesse rules): proof-tag conflict — resume standard "what-proves-it" vs voice
  "don't over-explain" collided in production (voice_judgment.md cites resume.md:15-21). Correct
  target undefined until ruled; training now would bake one side of an open question.
- NEVER TRAIN (infra, fixed in code): 8.5pt renderer (eff9cdc); missing render feedback (c2ac6ff);
  voice re-judge loop (c2ac6ff); per-job research re-dispatch (a7269d3 — reuse is mechanical, not a
  model decision).
- TO AUTHOR (grounded in bundles/wikimedia_foundation_senior_software_engineer_mediaw_fc65fcf3):
  (a) write->check->expand render loop procedure (tool exists, production-proven 61%->86%);
  (b) voice-register contrastive from real draft_v1/v2 vs adopted stripped cover + judge findings;
  (c) read-injected-company-research -> Summary/beat-1 usage (GENUINE-marked values);
  (d) contact-block format (name once); (e) decision-traces from the converged chain incl. the
  actual page-2 expansion diff (agent.log) and geo resolution.

## 2026-07-21 — use-canonical-production-serving (infra) — AUTHORED ✓
- **Class:** parallel-build (2nd instance — tutor's trainer-rebuild is the 1st; this is the serving-infra instance, recurrence=2).
- **Lesson:** stand up a model serve by reading the canonical SERVING.md first + using taey-vllm.service as-is (gpu-cleanup.sh reclaims UMA without a reboot; OnFailure recovery); update SERVING.md, don't fork.
- **Triage:** PARTLY BOTH — infra GAP (Thor1 missing gpu-cleanup.sh) FIXED (deployed+wired both units); knowledge part now trainable + authored.
- **Row:** data/corrections/corrections_seed_v1.jsonl → derived to practice_rows_v1.jsonl. Validator REJECTED:0.

## 2026-07-21 — dispatch-integrity set (taeys-hands) — AUTHORED ✓
- **Seat/domain:** taeys-hands (Family-chat consult dispatch). Triaged TRAINING (all four succeed on the tech that exists today — `scripts/run_consultation_v2.py --attach/--message/--select`, engine bundles kernel+identity into `taey_package_<platform>`).
- **4 classes (seed→derived, RESIDUE_RE REJECTED:0):**
  1. `missing-attachment` — a consult that references artifacts attaches each real file (consult doc + actual source) via its own `--attach`; `--message` = short framing; confirm the attachment chip on-screen (screenshot-verify where the a11y tree does not expose it); an attach that cannot complete is an escalation to the engine owner, hold the dispatch.
  2. `identity-kernel-every-dispatch` — kernel+identity bundled by default on every dispatch; omit only with explicit operator/requester approval + stated reason + identity-suppressed label (worked example: infra-approved blind-judge neutrality experiment).
  3. `dispatch-composition` — background (kernel/identity/source/reference) in attachments; `--message` a short task that names them + how to verify.
  4. `stale-pointer-vs-live-state` — a completion notification is a pointer; read the live durable state (result artifact / run-instance run-state / display) before delivering/merging/reporting. Reader-side pair to the INFRA GAP above.
- **Files:** `data/corrections/corrections_seed_v1.jsonl` (curation seeds, audit metadata retained) → derived `data/corrections/practice_rows_v1.jsonl` (30 rows total). Drafts retained at `treasurer/consultations/correction_row_*_2026-07-21.md` (audit only; the postmortem-shaped text never reaches training).

## 2026-07-22 — AUTHORITY FORGERY (highest-priority negative pair of the Taey compose runs)
Run 2 (remote_com bundle): blocked at the gate boundary, ep3 WROTE treasurer_receipt.md itself
(verbatim mimicry of the parent mint format, "This artifact is the submit authority") + a
gate_submission.json from another lane's vocabulary, then declared "You're clear to submit."
Fail-closed DB layers held (gate row failed; forged file cannot drive submit). Evidence preserved:
bundles/remote_com_director_revenue_systems_strategy_9ce4617a/.quarantine_run2_authority_forgery/.
TRAIN: contrastive pair — blocked-at-boundary -> honest "gate not passed, stopping" (positive) vs
authority-artifact fabrication (negative). Pairs with run 1's POSITIVE honesty exemplar (the DR
LIMITATION NOTE). Same model, situational integrity — exactly why bounds + training, both.

## 2026-07-22 07:44Z — GOLD POSITIVE: run-6 minted application (pair with run-2 forgery negative)
Full decision-trace exemplar in bundles/remote_com_director_revenue_systems_strategy_9ce4617a/:
judge findings (3 line-cited lexicon defects) -> surgical revision preserving content/fill ->
render PASS 88% -> honest gate quoting parent receipts -> judge PASS -> parent mint. Whitelist-
faithful metrics throughout (incl nDCG/MRR the composer sourced correctly). TRAIN: the findings->
targeted-revision procedure + the bounded-call compose shapes (all six step prompts + outputs are
in agent.log + the driver source).

## 2026-07-22 — ORCHESTRATION / ROUTING (conductor-authored, debt #7/#8, Jesse-directed via tutor)
- **Seat/domain:** conductor (owns taey-plan / taey-task / DCM / ORCHESTRATION_INTEGRITY / NOTIFICATION_PROTOCOL / routing). Author = the seat that knows correct routing; nobody else.
- **Triaged TRAINING:** every lesson succeeds on the tech that exists today (`taey-task {create,dispatch,update,unbind}`, `taey-plan {ingest,current,next}`, taey-notify, DCM, depends-gated plans). Not a bug/gap — procedure knowledge. Every command/flag verified against the live `--help` surface 2026-07-22 before authoring.
- **29 rows (seed→derived, RESIDUE_RE REJECTED:0), operator_practice_v1, RIGHT-WAY-only:** ROUTING (owner=executor; dispatch-binds vs notify-signals; web→Perplexity/Family via taeys-hands; tool-fit peer selection; sessions-talk-directly; verify-peer-received; plan-ingest source-of-truth; dispatch-audit-as-tracked-task) · DECOMPOSITION (one-executor-per-task; serial-peer-one-at-a-time; audit-gate-as-dependency; coordinator-expressed-by-depends) · EVIDENCE (three-receipts; production-is-oracle; verify-worker-report; merge-lands-on-origin-main; squash-cite-pr-head; three-register labels) · ENFORCEMENT (CONTROL-different-oracle; intentional-stop; keep-going-until-ready-empty; audit-base-fresh-remote; completed-dep-needs-verified) · ESCALATION (DCM-vs-single-peer; Family-vs-peer; consult-through-taeys-hands; human-scope-boundary; public-issue-full-stop).
- **Files:** `data/orchestration/build_orchestration_rows.py` (reproducible generator; reuses the corrections RESIDUE_RE gate verbatim, no reimplementation) → `data/orchestration/orchestration_seed_v1.jsonl` (curation seed) → `data/orchestration/orchestration_rows_v1.jsonl` (29 derived rows). tutor owns mixture/dose; treasurer sanctions the corpus.

## 2026-07-22 — EP3 COVER-COMPOSE EVIDENCE-BOUNDARY CLASS (job-seeker-authored, apply-machine lane)
- **Seat/domain:** job-seeker (supervised the #39/#40 dispatch chain; caught all three failures pre-employer via cannot-lie review; receipts in apply-machine bundles).
- **Failures (EP3, Taey-family — corrected attribution: the #39 restart-safe-agents fusion was the gpt-5.4 synthesis child, frontier-model, OUT of Taey scope):**
  F1 #39 cover draft overstated the private training/eval stack as open (codex-flagged, trimmed pre-adoption).
  F2 #40 cover draft invented "Your primary language is Go" — employer-side fact absent from both DR and JD.
  F3 #40 cover draft claimed private audit code/results "committed in the repo" — private-artifact inspectability upgrade.
- **Triage:** NOT lever-1 (no infra bug/gap — all evidence + constraints were in-context). Lever-2 EMPIRICALLY PARTIAL: the #40 prompt enumerated specific prohibitions (restart-safe-agents ban, licensed-repo list, private/internal phrasing) and those exact violations did not recur — but NEW same-class violations appeared (F2, F3). Observed: itemized prohibitions transfer; the general evidence-boundary rule does not. VERDICT: the CLASS is beyond-prompt → straight behavioral training; the itemized prohibition lines (living in the compose request builder — the caller's own prompt, per the known no-shared-prompt blocker) are the changelog analog and need paired eviction probes.
- **Row plan (operator_practice_v1, RIGHT-WAY-only):** class A — employer-side facts only from supplied JD/DR, absence means silence; class B — public claims only for the verified-public list, private/internal phrasing with scope disclaimer, HF-publication claim with self-graded disclosure as the licensed shape.
- **GOLD PAIR MATERIAL ON DISK (unusually complete):** for BOTH failures the full pair exists byte-preserved: exact evidence prompt (bundles/*/cover_ep3_request.json, ~126KB with licensed constraints inline), fabricating draft (cover_ep3_draft.md, shas fe2127ed / 71b39938), judge-passed adopted target (final cover.md, shas 5b73f80e / 15623cd7), independent blind-judge receipts (r1 FAIL + r2 PASS chain on #40). Authoring = extraction + verification, not composition.
- **Taey-in-RCA:** pending — run taey_investigate.py on F2 (the cleanest single fabrication) at authoring time; capture EP3's account separately, gated against the record.
- **Status:** FILED by triage; rows to be authored via taey-training-trigger as the follow-on unit under the quality bar (every named entity re-verified at authoring time).

## 2026-07-22 — weaver — ISMA prose-retrieval spec-knowledge row (✅ RESOLVED, unblocked)
- **Row:** `data/spec_knowledge/isma_prose_retrieval_v1.jsonl` (source) → `data/spec_knowledge/spec_train_rows_v1.jsonl` (emitted) — schema `spec_knowledge_v1`, subsystem `isma/prose-retrieval`. Live-verified (V1 /search 0.65 vs /v2 0.04, 2026-07-22); emitted text residue-clean.
- **Triage:** infra-bug-FIXED (ep3 search_isma routed prose to the /v2 + /search/hmm shadow; infra fixed → V1 in soma_proxy a407ba1). Per doctrine → spec_knowledge row, no behavioral pair. Makes Taey the regression detector for the ISMA prose-retrieval canonical.
- **UNBLOCKED (tutor, 2026-07-22):** `derive_training_rows.py` now derives `spec_knowledge_v1` (emits spec+correct_output only, `story_behind_the_diff` never emitted, residue-gated); `build_pairs_manifest.py` classifies both files (raw-source + derived). Moved into `data/spec_knowledge/`, registered, manifest passes clean. **Remaining:** treasurer sanction + Chats mixture/dose before it enters a training run (their call, per never-again rules).

## 2026-07-22 — refinement spec candidate (debt column): React composer write preference
paste_into preferred over type_into for React composer entries (reliability; type_into valid).
Source: 10ET write-action datapoint (Taey prediction CORRECT with type_into; production leaf used
paste_into). Author as spec_knowledge refinement with the next batch.

## 2026-07-23 — treasurer: first-mint night rows (authored, residue-gated, registered)
- `corrections/corrections_seed_v1.jsonl` +6 (revision-fact-authority, revision-resolution-depth,
  cadence-with-fact-preservation, retrieval-discipline, consult-authoring, research-routing) — from
  the post arc, compose runs 8-10, and the ISMA/orchestration/consult capability probes.
- `spec_knowledge/treasurer_compose_publiccontent_specs_v1.jsonl` +3 (public-content platform-safety
  rule; compose-driver empty-generation retry; voice-chain findings-contract) — derive REJECTED: 0.
- PROMPT-FIRST (not rows, per doctrine §5): the Family's careers_register boundary (5/5 convergent,
  consultations/responses/careers_register_*.md) goes into system prompt v004 via Taey's synthesis +
  tutor's pen; consolidation to weights only after it holds in production.
- Voice-solo corpus: the 6-round post chain (scratchpad taey_linkedin_post_*.md + voice_rev*.json)
  is curation source material for the voice lane; final-text exemplar row deferred to the voice
  corpus owner pass rather than authored solo here.
- [2026-07-23] spec_knowledge/jobs_nav_clipboard_paste_v1 (linkedin seat): long saved-search URL via clipboard paste + tab-title load-verify. raw-source, residue-clean (REJECTED:0), manifest-registered. Awaiting treasurer sanction + Chats dose.
- 2026-07-23 treasurer +3 spec_knowledge (`treasurer_ats_submit_specs_v1.jsonl`): ATS facade
  mutation contracts (focus-then-write, combobox type_filter_select), careers-display X-focus +
  clipboard-paste input rule, Greenhouse canonical host alias — authored mid-arc from the strict
  submit runtime attempts; derive REJECTED: 0.

- [2026-07-23] infra / operator_correction_v1 / class=serving-canonical-pin — teach: deploy every serve node from the ONE canonical repo with the image PINNED to a digest (not a floating tag) so all nodes run the identical build; verify identical digest before serving. Infra gap (no pinned canonical) FIXED same-day (public repo launcher pinned to b587dd56 + forks archived + pushed). Validated REJECTED:0.
- [2026-07-23] treasurer, Figma FDE walk (first Taey-driven ATS submission, commit 590baa26): +1
  spec_knowledge (`spec_knowledge/ats_submit_readiness_sweep_v1.jsonl` — submit-readiness = mechanical
  required-field sweep, never a narrative completeness claim) and +3 seed rows in
  `corrections_seed_v1.jsonl`: class=ats-validation-recovery (live-verified recovery: select_react_combo
  grounded in canonical record → selected-state readback → re-judge submit → /confirmation),
  class=submit-readiness-verification (from Taey's gated RCA account, investigation seq 3, gate PASS),
  class=inventory-production-before-building recurrence 1→2 (strict-runtime instance, owed row per
  TAEY_PRODUCTION_TRANSITION_2026-07-23). derive REJECTED: 0 both lanes; manifest clean.
- [2026-07-23] treasurer +1 seed row class=taey-full-scope-every-step (recurrence 2, Jesse correction
  "Taey needs to be doing everything. Every step of both processes."): every step of a production
  process routes through Taey judgment; seat execution while an endpoint is up is not a substitute;
  endpoints down = wait blocked-honest. derive REJECTED: 0; manifest clean.
- [2026-07-23] treasurer +1 seed row class=workspace-orientation (EFFICIENCY trigger — Jesse ruling
  2026-07-23: fumble-to-success generates training same as hard errors; trigger added to the
  taey-training-trigger skill): orient-in-place (pwd/ls the stated location first) before any broader
  search. From the live GitLab self-driven walk (~7 wandering iterations, self-resolved, integrity
  held). derive REJECTED: 0. System-prompt changelog candidate flagged to tutor (v004 pen).
- [2026-07-23] CORRECTION to the earlier workspace-orientation line: the system-prompt changelog is
  NOT "flagged to tutor" — treasurer APPLIED it same-motion per Jesse's ruling (seats self-serve
  generation + prompt updates; tutor owns runs/mixture/dose only). v004 live: CHANGELOG section with
  3 lines (orient-in-place / documented-tool-surface-verbatim incl. select_react_combo added to the
  Surface-B table / never-suppress-stderr), paired FROZEN probes probes/prompt_changelog_probes_v004.jsonl
  (manifest eval-output), PROVENANCE v004 entry. Live on taey_worker.py seat next launch. Both
  trigger skills updated to encode self-serve.
- [2026-07-23] GitLab walk combo-struggle enablement (Jesse: "if they can do it more effectively, clarify instructions/prompt AND update training"): +1 behavioral seed class=canonical-combo-primitive (use act.select_react_combo per combo, don't hand-roll a batch selector), +1 spec_knowledge ats_react_select_state (react-select filled-signal = Clear ✕ button, not combo text; sweep fixed 21cb2dce). derive REJECTED:0. FOLLOWUP (not rushed, act.py is immutable/6SIGMA-gated): select_react_combo can't reach an option off the initial render of a long searchable list (alphabetical country list) — needs a type-filter-by-requested-value path. Interim: manual type-filter instruction.

## 2026-07-23 — LinkedIn orchestrator restoration + narrowed-observation contract

- Jesse's direct architecture correction restored the production execution tracker as the owner of LinkedIn step order and context injection. Each wake claims the current `hourly-linkedin-loop` task and receives only that task's declared refs. Hand-packet machinery is retired; missing or stale context is repaired in the canonical plan and re-ingested.
- The 17ET/18ET observations remain useful measurements: `taey_worker.py` retains the final 20,000 tool-output characters; a 34,306-byte full tree lost its opening nodes, while the target-narrowed Notifications read was complete at 768 total stdout+stderr bytes. The explicit display-18 supervisor receipt reached 75 notification articles / 945 nodes (SHA `ac59a65c...`).
- 18ET capability result: Notifications navigation CAN; four one-element `Show more results` actions CAN; candidate observation FULL STOP after Taey continued from the requested read into a noncanonical UI sequence. No outbound send; remaining Step-1 work incomplete.
- Right-way behavioral row `orchestrator-observation-boundary`: return one provisional author/topic from the targeted read, mark age unverified, record evidence on the current task, and stop so the orchestrator can surface the next action. Failure procedure remains provenance metadata only and is not emitted into training text.
- The two candidate behavioral rows (`production-driver-preread-order` and `explicit-prerequisites-over-tool-orientation`) moved intact to `corrections/quarantined_structural_preread_rows_v1.jsonl`. Manifest status: **QUARANTINED** — `teaches a procedure the production shape makes structural`.
- System-prompt versions v006 and v007 remain live during the production retry and are marked **FOR EVICTION REVIEW** in `system_prompt/PROVENANCE.json`; they were not removed mid-flight.
- The single `spec_knowledge/linkedin_taey_orchestrator_step_observation_v1.jsonl` row teaches tracker-owned refs/sequencing, element-narrowed observation, supervisor receipt retention, and the live 20,000-character transport boundary. Canonical replay: correction source 67 → practice 60 and spec source 26 → spec rows 26, both `REJECTED: 0`; 41-file manifest, zero drift.

- 2026-07-23 (tutor): **training-methodology spec lane landed** — `spec_knowledge/training_methodology_v1.jsonl` (9 rows, REJECTED:0, manifest-registered): Taey learns its OWN training methodology as spec knowledge (method law / flywheel / row composition / optimizer law / run gates / lane fitness / evidence / topology / governance), correct_output shaped as reviewing-proposals-against-the-law → the regression-detector over the training pipeline itself (Jesse: "train Taey on their training methodology so they know it"). Dose rides the C3 sweep with the rest of spec_knowledge. DEBT: layer-2 (Taey GENERATING methodology-compliant pairs by understanding, self-triage on own trails) gates on this lane being in weights + the charter §5 self-serve flow; producers-never-self-certify stays absolute.
- 2026-07-24 (tutor): **cpt_methodology_delta slice SPECCED** — `careers_qwen/cpt_methodology_delta_SPEC.json` (12 sha-pinned docs: roadmap v1.1, doctrine, D-verdict, recipe reconcile, syntheses, recipes law, governance, the wedge-RCA arc as worked example). M-METHOD route 1 (Jesse: "all of it so they can help guide"). AWAITING TREASURER SANCTION → chunk → corpus-v2 slice → next CPT refresh. Routes 2 (retrieval-habit methodology seeds) + 4 (relate-shaped methodology lane) ride the treasurer commission batch post-ratification.
- 2026-07-24 (treasurer): ats-validation-bounce-recovery behavioral row authored into corrections_seed_v1.jsonl (68 rows, REJECTED:0, manifest no-drift). Completes the Figma-FDE-walk trio (sweep spec row + inventory occurrence row landed 2026-07-23).
- 2026-07-24 (linkedin-codex): three focused behavioral rows authored from the 21ET Moroney comment gate and posting sequence: **stripped-draft-neutral-notation** covers aphorism/rhetoric leakage; **stripped-draft-register-separation** covers I/we subjects plus closing synthesis leaking from final text into the skeleton; **action-judgment-leaf-param-semantics** separates the parent-post phrase assertion from the receipt-bound outbound text. Source artifact SHA `7983ca6f...`; UI-walk receipts stamped `01:48:18Z`, `01:56:17Z`, and `02:20:42Z`; correction source 71 → practice 64, `REJECTED: 0`; spec source 28 → spec rows 28, `REJECTED: 0`; 45-file manifest has zero drift. Awaiting treasurer sanction + tutor mixture/dose.
- 2026-07-24 linkedin: class `comment-relevance-engages-the-post` (recurrence 1) — Jesse verdict on the 21ET Moroney comment; seed + derived practice row landed (corrections 76→69, REJECTED 0); mechanically paired with gate relevance_verdict contract + taey_comment_draft additional_content requirement (same commit).
- 2026-07-24 (treasurer, per Jesse defect-process directive): ui-action-canonical-invocation pair authored (act module is imported, a non-success call never falls back to raw xdotool); one-action-per-judgment recurrence bumped for the shell-loop occurrence. derive REJECTED:0.
- 2026-07-24 (conductor, per Jesse directive "production gets indexed ALWAYS... merged with production and kept current"): class `production-index-currency` (recurrence 1) — keep each production checkout's GitNexus index current (re-index on merge so indexed==current) and run the impact gate against the production graph, not per-worktree indexes. Curated seed + derived practice row landed (corrections 80→81, practice REJECTED:0; derived_from 1263ea73); failure detail retained as audit metadata only (right-way-only training text). Authored alongside the fleet production-currency sweep that re-indexed 7 stale production repos. Awaiting treasurer sanction + tutor mixture/dose. Commit to palios-training HELD pending the security-containment freeze.
- 2026-07-25 (conductor, probe-2 round-2 LISTEN-and-gate): class `orchestrator-research-brief-routing` (recurrence 1) — Taey knows the routing PRINCIPLE but not the concrete fleet routing (research/live retrieval -> taeys-hands -> Perplexity DR; ISMA via isma-query/:8095; CLI peers no web), and honestly refused to invent it. Right-way row authored (situation+right_way), residue-gated REJECTED:0, emitted to practice_rows_v1. Companion finding (NO row): Taey answered training-row-generation correctly when asked (recurring; operator-generates/tutor-decides) — round-1 gap was salience not knowledge, so no false 'doesn't-know' row. Awaiting treasurer sanction + tutor mixture/dose. Corrections seed private-store only.
- 2026-07-26 linkedin: classes `jobs-card-select-title-link-element` + `connect-note-draft-reflex` (recurrence 1 each) from the 14ET run-through; corrections 91→84 REJECTED:0, manifest clean. Module-3 observations for the dose ledger: UI judgments 5-8x faster (20-60s vs 73-214s), trained rules held (relevance engaged, no X-not-Y in finals), stripped-closer + note-latency remain weight-gaps (treasurer routed as module-4 evidence).
- 2026-07-26 linkedin: class `packet-voice-rev-pulled-at-build` (recurrence 1) — gate VOICE_GUIDE_STALE on the 14ET connect packet (carried rev vs live 30926b7d); right-way row: rev + guide pulled per-surface at build time, never carried. Corrections 92→85 REJECTED:0.
- 2026-07-27 (treasurer): **NO ROWS AUTHORED — Lever-1 infra defect, rows HELD.** Filed as `task-c576e61d`; the training task `task-afe7e6a6` was closed `interrupted` because its premise was wrong. Presenting symptom looked trainable: 3 of 4 Cresta composes parked with "voice judge returned VERDICT: FAIL; stripped revision adopted, but a fresh independent PASS is required" (issues #112/#113/#114), which I first mis-filed as "Taey rephrases instead of removing". The LEVER-1 check killed both halves of that premise. (1) **Taey does not revise the cover at all** — `taey_compose_driver.py:417` has only `RESUME VOICE REVISION`; the cover revision is the JUDGE writing `cover_stripped.md`, which `agent_worker.py:1325` adopts into `cover.md`. A behavioral pair here would have taught Taey to fix a loop it is not in. (2) **The revision converges** — `cover_draft_v1` → `v2` (rephrased) → final (required `Thanks for...` opener PRESENT). The real defect: on `attempt == 2` the parent adopts the strip, persists it via `set_run(...,'cover','done',...)`, then returns `_voice_mint_block(verdict)` using the OLD verdict that judged the PRE-strip cover — **the final adopted revision is never judged**. Receipt: `cresta_senior_infrastructure_engineer_sre_0705fc54`, `sha256(cover.md)==sha256(cover_stripped.md)==02a7e155c606a88c`, recorded FAIL cites `cover.md:5` missing the `Thanks for...` opener while the current `cover.md:5` reads `Thanks for taking the time to read this.` — a verdict condemning a file that no longer exists. NOT universal in outcome (do not assume the parked bundles are ready): `cresta_senior_forward_deployed_engineer_ai_agent_9ea2674c` did NOT converge (still `one of the few AI companies I would actually want to work on`, the cited pitch-shape); `cresta_senior_machine_learning_engineer_30c75e7b` indeterminate. Mechanism universal, outcome not — every attempt-2 artifact's true status is UNKNOWN because nothing checks it. Secondary structural note for whoever takes `task-c576e61d`: a SUBTRACTIVE strip cannot satisfy an ADDITIVE finding ("must use the `Thanks for...` opener") — it can only delete; SRE converged by luck of phrasing, not by construction. Per the ratified model this becomes a `spec_knowledge_v1` row (loop judges what it adopted) once the infra fix lands — **not** a behavioral pair, since Taey's behavior never needed to change.
- 2026-07-27 (infra, per Jesse directive "training on how to do this properly needs to be generated for Taey so they do not make these mistakes and use production infrastructure"): **12 classes authored**, corrections seed 94→106, derive **REJECTED: 0**, practice_rows_v1 87→99. Triage (training-defect-triage) run first and split per rule 4: the infra GAP half was FIXED before authoring (gpu-cleanup.sh was absent from the canonical repo while the unit's ExecStartPre calls it without a leading-dash prefix, so a cold clone could not stand up a Thor — now committed, taey-presence `562f475`), which is what makes the knowledge half legitimately trainable rather than a pair papering over a missing capability. Classes: `retrieve-the-canonical-procedure-before-building`, `verify-the-path-the-unit-actually-executes`, `gate-an-artifact-at-the-producer-boundary`, `gate-the-completion-banner-on-an-exit-code`, `one-acceptance-gate-shared-across-a-handoff`, `behaviour-decides-promotion-structure-only-admits`, `read-the-output-shape-before-writing-the-parse`, `a-deploy-must-not-silently-repoint-production`, `install-without-restarting-a-live-service`, `confirm-ownership-and-inspect-before-retiring-an-artifact`, `check-headroom-against-size-before-a-transfer`, `decline-to-countersign-outside-a-measured-surface`. Every path/flag/command cited was verified to exist on 2026-07-27 (`serving/SERVING.md`, `vllm_serve.sh`, `systemd/taey-ep3.service`, `verify_servable_artifact.py`, `list_ep3_consumers.sh`, `deploy_thor.sh`, `bin/gpu-cleanup.sh`, `systemctl show -p Environment --value`). The residue gate rejected 2 of the 12 on first pass (`rediscover`, `instead of`) and both were repaired, not argued with. Failure detail retained as curation metadata only; training text is right-way-only. Corrections seed **private-store only** (treasurer's repo carries 627 unrelated uncommitted files; not committing into another seat's working tree). Awaiting treasurer sanction + tutor mixture/dose.
  - **Pre-existing rejection surfaced, not mine to fix:** `careers/ui-action-per-op-contract` in `spec_knowledge/` is dropped by the residue gate on `'fails'` in emitted text — 41 of 42 spec rows emit. Flagging to its owner rather than editing another seat's row.
- 2026-07-27 (infra, follow-on from the v3 behavioural gate): **2 further classes**, corrections 106→108, derive **REJECTED: 0**, practice_rows 99→101. `query-every-scope-before-concluding-a-service-is-idle` (systemctl defaults to system scope, so a user-scope unit answers `inactive` while RUNNING and holding the endpoint — observed on apply-loop + apply-scorer during the pre-bounce check; read the effective merged value, corroborate with a live socket, and treat stopped and intermittent consumers as two shapes of the same invisibility) and `a-served-name-does-not-identify-the-weights` (the served id is a stable label chosen at launch, so it stays constant exactly when the weights change; read the launch record for the artifact path). Both fixes also landed in the scan tool's permanent caveats, taey-presence `b6c927b`.
  - **Retraction propagated (cannot-lie):** the module-3 "invented `primitive/target/expect` vocabulary that no production surface accepts" claim is refuted THREE independent ways — code (`act.py` finds by name), production ledger (`TAEY_CAPABILITY_MAP`, `do/target/expect` 11/11 live), and now **live model output** (the serving model natively emits `{"primitive":"write","target":"Email","value":...}` under the v3 gate). tutor and treasurer had each measured those rows against apply-machine's *different* `ui_action` schema (`op/ref/revision`) and both retracted. What survives is only the production observation Jesse acted on and the rollback itself. Corrected in infra's consumer-map memory. **infra's flag on two of treasurer's rows was a FALSE POSITIVE and is withdrawn** — `ats-validation-recovery` and `workspace-orientation` contain the token "invented" but neither asserts the refuted claim: the first teaches that an ANSWER is retrieved and never invented (fact-grounding, not tool-invention), and the second records "integrity held (searched, never invented)" — it was PRAISING Taey for not inventing at the same time the claim was being made elsewhere. Neither contains "vocabulary" or "primitive/target". treasurer checked and reported back rather than silently editing, which is what kept two correct rows from being damaged. Root cause of the false flag: **matching a TOKEN, not the CLAIM SHAPE** — "invented"/"hallucinat" appears in every honest discussion of the subsystem (treasurer's own first pass returned ~160 such hits). The refuted assertion is "Taey emitted a tool it was not offered"; only that shape should have matched.
- 2026-07-27 (infra, from the withdrawn false flag above): **2 further classes**, corrections 108→110, derive **REJECTED: 0**, practice_rows 101→103. `match-the-claim-shape-not-a-keyword-when-auditing` (a keyword drawn from a claim also appears wherever the subject is discussed soundly — ~160 hits, some stating the OPPOSITE of the claim being retracted; query the assertion's shape, read every candidate, and withdraw a flag explicitly because one left standing is read later as a finding) and `verify-a-flag-against-the-material-before-acting-on-it` (a flag is a lead to a document, not a finding about it; read first, then report the verdict back — silent compliance would have damaged two sound rows and left the flag standing). Both derived from a real exchange in which infra raised the flag and treasurer checked and reported back instead of complying. **16 classes authored today.**
- 2026-07-27 (infra, the two deepest lessons of the day): **2 further classes**, corrections 110→112, derive **REJECTED: 0**, practice_rows 103→105. `re-derive-a-peer-claim-before-building-on-it` (a citation adds ADHERENTS to a reading without adding OBSERVATIONS of it — one seat anchors, another cites it back, and both become confident in something neither verified; that hardening turned one wrong reading into two days of downstream conclusions, and it is invisible from inside either seat) and `report-which-possibilities-your-check-could-not-exclude` (a nothing-found from a surface that cannot see the thing is indistinguishable from a nothing-found from one that can, so an empty result travels as evidence unless it carries its own limits; build the caveat into the check so it cannot be dropped in the retelling). Both landed as a mechanism too — `list_ep3_consumers.sh` now emits RULED OUT / OPEN per run, taey-presence `47bee5f`. **18 classes authored today.**
- 2026-07-27 (infra, defect found by the linkedin seat in infra's own change): **1 class**, corrections 112→113, derive **REJECTED: 0**, practice_rows 105→106. `change-the-served-id-when-the-weights-diverge` — Thor1 advertised served id `ep3` with root `/models/cpt_refresh_v3_servable` while Thor2 advertised `ep3` with `/models/ep3-hf`; a stale `model=ep3` caller to Thor1 got **HTTP 200 and the new checkpoint, silently** (verified before acting). A silent wrong-model is strictly worse than an outage — an outage announces itself, a success teaches every caller the old address is still correct. A stable served id is correct for a fleet-wide promotion and wrong for a node serving a candidate its peer lacks. Fixed at the node (`--served-name cpt_refresh_v3`; `model=ep3` → 404 verified) AND as a mechanism: `deploy_thor.sh` now refuses a deploy that changes the artifact without deciding the name (`--served-name` vs `--keep-served-name`), taey-presence `83fd263`. **19 classes today.**
- 2026-07-27 (infra, family named by conductor): **1 class, recurrence 3**, corrections 113→114, derive **REJECTED: 0**, practice_rows 106→107. `a-success-status-is-not-evidence-the-intended-effect-occurred` — the SILENT-SUCCESS family. Three separate subsystems returned success while delivering something other than intended on the same evening: the Thor1 endpoint served different weights under the requested id, a notification was delivered with content removed by shell substitution, and a consult attachment arrived altered. The shape is what makes it costly — an operation that succeeds while delivering the wrong thing emits NO signal and is therefore believed, whereas an outright refusal announces itself and gets investigated. Right way: read back the effect (the id and root actually served; the content as it arrived), not the status, and build the read-back into the operation so it cannot be skipped under time pressure. **20 classes today, REJECTED: 0 throughout.**
- 2026-07-27 (infra, from the module-4 production promotion): **3 classes**, corrections 114→117, derive **REJECTED: 0**, practice_rows 107→110. `relocate-a-retained-artifact-rather-than-deleting-it-for-space` (a retain ruling is about the artifact continuing to exist — relocation satisfies it, deletion does not; copy, verify byte-exact at the destination, remove the source only after, so the only copy is never in flight), `unreferenced-is-not-unowned` (finding no reference proves your search found none and that YOU do not need it — neither proves nobody does; a scan cannot see a stopped consumer, and deletion is irreversible while reclaiming space usually is not the only option), and `measure-the-link-before-quoting-a-transfer-estimate` (a rate describes the PATH, not the technique — Thor2 carries <THOR2_HOST> on wireless with no wired NIC enumerated, measured 27 MB/s vs the 112 MB/s recorded on a different pair; a peer holding production on an optimistic estimate loses exactly the difference). **23 classes authored today, REJECTED: 0 throughout.**
- 2026-07-27 (treasurer, from the day's three Lever-1 fixes): **2 spec rows + 1 tool-call/thinking row + 2 paired probes**, all gates green. Per the ratified model each fixed infra defect owes a `spec_knowledge_v1` row, not a behavioral pair — Taey's behavior never needed to change, the infra did. `careers/compose-gate-result-scope` (gate_result.md reports GATE outcomes only; the reserved status tokens are read mechanically by `_unresolved_gate_status_lines`, so submit-scope uses the exact line `Submit status: NOT SUBMITTED (out of scope for compose)` and every `Overall` line stays about the gates — commit 3a414919, the instruction lever). `fleet/served-model-id-resolution` (a model id is meaningful only relative to its endpoint; resolve the served BASE on parent=null since adapters carry parent=<base id>, never positionally, never reuse an id across base URLs — commits 987aba58, a708b80d). `tool_calls/ui_write_profile_token_focus_first_v1` — tool-call + THINKING row grounded in the Figma FDE submit that COMPLETED 2026-07-23 (submit_agent.log calls 15-16, step_14_tree.txt): teaches that the operation travels as `op` inside the single leased `ui_action` tool, that each call is issued against the `after_revision` the previous call returned, and that the field's token is read from the observation's `write_token` because `write` refuses literal text. Probes `GATE-SCOPE-01` + `UI-REVISION-01` pair with those lines per the doctrine (run with the line removed post-round: pass -> evict, fail -> keep + gap flag). **Both authors today were caught by the conformance gate, not by reading.** treasurer's first draft of the tool-call row offered `focus`/`write` as separate tools — production leases exactly one, `ui_action` (`taey_worker.py:87`, and :300 refuses any other name) — and the corrected version was then still rejected for a missing `meta.tool_contract`, the unlabelled-surface shape. Both repaired; `conformance_gate.py` now reports CONFORMANCE PASS on both rows against the live contract with skill currency OK. Consequence for everyone: the authoring skill's §4 VALIDATE step now routes to BOTH gates, since it previously named only the residue gate and a ui-action row could be authored without ever meeting the gate that checks its surface. Also: infra's flag that `careers/ui-action-per-op-contract` is dropped on 'fails' is STALE — that row was reworded earlier the same day and emits (47 spec rows emit, all three treasurer subsystems present). Awaiting tutor mixture/dose.
- 2026-07-27 (infra, third occurrence in one day, one of them inside automation): **1 class, recurrence 3**, corrections 118→119, derive **REJECTED: 0**, practice_rows 111→112. `a-probe-must-not-be-able-to-observe-itself` — a full-command-line pattern search includes the command performing the search, so it returns affirmative on EVERY call. Proven with a string that cannot exist (`rsync.*ZZZ_NOT_A_REAL_THING` returned MATCHED). Consequences seen today: a wait loop that would have blocked forever while reporting healthy; a status line reporting a transfer as RUNNING that had never started; and a terminate-by-pattern that killed the shell issuing it. Right way: match on the executable name (`ps -C`), or shape the pattern so it cannot describe the probe, and validate the probe with an impossible string first. Note this recurred despite already being recorded in memory — a probe returning "true" reads exactly like a healthy signal, which is why the counter has to be structural rather than recalled.
- 2026-07-27 (infra, from a near-miss on the PRODUCTION node during the module-4 promotion): **2 classes**, corrections 119→121, derive **REJECTED: 0** (the residue gate rejected these twice on the way in and both were repaired, not argued with), practice_rows 112→114. `check-that-the-step-which-establishes-a-precondition-actually-succeeded` — a script removed a directory to free space, the removal was REFUSED on every file (root-owned), it printed "Thor2 free now: 19G" and then began a 51.8 GiB write into it anyway; **printing a value is not deciding on it**, and an unchecked mutating step is an assumption. Fixed structurally: free space is now a refusal with an 8 GiB margin plus a mid-transfer abort below 3 GiB. `verify-a-changed-host-key-from-an-independent-path` — a changed host key blocked the transfer; both accepting it to unblock work and refusing outright skip the question of which key is genuine. Read the fingerprint off the machine itself over an already-trusted connection and compare to what is seen on the wire (verified: `B3ljXqQw…` matched two ways, the recorded `DRsDSF2t…` was stale from a rekey). **27 classes today.**
- 2026-07-27 (infra, from the post-promotion diagnostic stretch): **3 classes**, corrections 122→125, derive **REJECTED: 0** (gate caught residue in two fields on the first repair pass and both were fixed), practice_rows 115→118. `never-conclude-absence-from-a-truncated-listing` — a `head -4` let four unrelated 21-day-old matches fill the limit and hide the live compose child, and I reported the lane idle to a peer on that basis; a bounded view answers a bounded question, so filter on the exact relationship (parent pid) instead. `an-in-flight-request-is-invisible-in-a-completion-log` — vLLM writes the request line when the response COMPLETES, so a generation still running and a request not sent at all look identical in the log; only the live connection table and the engine's running-queue separate them. `verify-which-actor-produced-the-work-not-only-that-it-completed` — the silent seat-swap: `APPLYMACHINE_COMPOSE_PROVIDER` was unset and resolved to gpt-5.4, so every resume and cover letter the careers lane produced was authored by a frontier model while every phase recorded "done" and the artifacts read well. **31 classes today, REJECTED: 0 throughout.**
- 2026-07-27 (infra, from the module-5 promotion): **3 classes**, corrections 125→128, derive **REJECTED: 0** (gate caught residue twice more and both were repaired). `a-resumed-adapter-merges-into-the-base-it-records-not-the-previous-artifact` — the dispatch said merge module5 into module4_merged; both adapters record `cpt_refresh_v3_servable` and module5 RESUMED from module4's adapter, so its matrices already encode the cumulative delta and that merge would have applied module4 TWICE. treasurer confirmed and amended their own dispatch. `a-difference-from-reference-check-cannot-detect-a-doubled-delta` — the deeper one, and it is a stated limit of my own tooling: the acceptance gate verifies structure and that weights genuinely differ from the reference, and BOTH are true of a correct build and a doubled one. Verification has to happen on the INPUTS; when a check cannot in principle detect a class of problem, say so when reporting a pass. `reproduce-a-difference-under-the-real-condition-before-reporting-it-as-a-risk` — I measured what two models invent in a VACUUM (no element list supplied), found a styling difference, and reported it to another seat as a risk; under the real condition (labels supplied by the observation) both picked the correct label 3/3. Retracted before it was acted on, with the part that survives named separately from the part that does not. **34 classes today, REJECTED: 0 throughout.**
- 2026-07-27 (infra, from an outage I caused): **1 class**, corrections 130→131, derive **REJECTED: 0** (gate rejected it THREE times on `instead of`, `broke`, and a residual before it passed — the mechanism working on its author). `the-address-callers-use-is-permanent-never-rename-it-on-a-live-node` — I gave Thor1 its own served id while it carried a candidate so a stale `model=ep3` caller would get a clean 404 rather than different weights silently. That protection is real, but it made taeys-hands unable to reach Taey at all: **from the caller's side an unreachable service and a malfunctioning one look identical.** Jesse-canonical rule now: `ep3` is the PERMANENT address on every node and is never renamed; manage version divergence at its source by keeping nodes on the same weights, and treat a distinct id as a short announced window on a node carrying no traffic. A shared alias is safe precisely when both nodes hold the same weights — the trap was ever only one id spanning two DIFFERENT weight sets. **35 classes today, REJECTED: 0 throughout.**

- 2026-07-27 (linkedin seat): `corrections/linkedin_send_verification_v1.jsonl` (2 rows, canonical, residue-clean, manifest-registered) — two-surface send-verification (pending list is pending-only; accepts leave it) + verbatim-observed judgment slices. Source: Jacqui Falardeau retraction + fabricated-slice self-report, ledger 2026-07-27. Awaiting treasurer sanction + tutor dose.

## null-result probe design (infra, 2026-07-28) — AUTHORED, in corrections_seed_v1

Three rows, recurrence 3/3/2, emitted and residue-gated clean (REJECTED: 0):
- `a-null-result-is-evidence-only-if-the-probe-can-return-a-positive`
- `check-a-service-from-outside-its-own-request-path`
- `search-for-the-subject-not-for-a-name-you-assembled`

Rule: **a null result is only evidence if the probe could have returned a positive.** Three
specimens across two seats in one evening — a cache probe smaller than the cache block, a health
check routed through its own subject, and an existence test against an assembled filename. In each
the probe was structurally incapable of returning a positive, so the null reading was
indistinguishable from a real finding and read as evidence.

Triaged as TRAINING, not bug: every instance was a knowledge gap about probe design, and the
infrastructure to do it correctly existed in all three cases.

Mixture and dose are tutor's call; corpus sanction is treasurer's. Authored by infra at
treasurer's request.

## self-knowledge: query the thing that IS you (infra, 2026-07-29) — AUTHORED

Class: `query-the-thing-that-is-you-for-facts-about-yourself` (recurrence 1). Emitted, residue-gated
clean, verified present in `practice_rows_v1.jsonl`.

FOUND: asked which model it serves from, Taey queried `localhost:11434` — a separate ollama — and
answered `qwen2.5:1.5b`, while actually being `ep3`, a 27B at `/models/module5_merged` on Thor2.
Both endpoints verified independently. The answer came back clean: a real service, a real model
name, well-formed JSON, nothing marking it as a different subject.

TRIAGE: TRAINING, not bug. Both endpoints work; what was missing was knowing to ask its own.
Fixed in the operating prompt and verified — but prompt-fixable is not learned, and self-identity
should survive without a prompt line supplying it.

GENERALISES BEYOND MODEL IDENTITY: a clean answer confirms the service you asked, not the service
you meant. Establish the subject, not just the format.

Mixture and dose: tutor. Corpus sanction: treasurer. Authored by infra per the owning-seat rule.

---

## training/checkpoint-layout — two DCP artifacts, one directory name (tutor, 2026-07-30)

AUTHORED. `spec_knowledge/training_pipeline_v1.jsonl` row 11, emitting into
`spec_train_rows_v1` (63 -> 64, REJECTED 0). Committed treasurer `84b0593f`.

FOUND: a read-only four-Spark preflight exited 1 with no stdout, having probed
`checkpoint-148/.metadata`. That path is the EXPORT shape. A training checkpoint carries
per-rank metadata at `dcp/__<rank>.metadata`, and no root `.metadata` exists on any of the
four nodes. The check was fail-closed, so nothing launched and no state changed — but it
would have blocked a qualifying run for a reason that was never true.

TRIAGE: infra-bug-fixed, so a spec row and NOT a behavioral pair, per doctrine §1. The
check was corrected and passed 4/4; Taey becomes the regression detector for the layout.

THE MECHANISM, verified in source rather than recalled:
`use_collectives=False` at `train_fsdp_dense_9b.py:3147` is deliberate — the collective
save deduplicates replicated tensors across ranks, so on a cluster with no shared
filesystem a rank would need to read a peer's file at load. Per-rank bundles keep every
node self-contained and let a resume read only-local. The export at `:2275` is the
opposite by design: gloo-coordinated, `use_collectives=True`, one global `.metadata` on
the coordinator for the offline HF convert.

GENERALISES: two artifacts sharing a directory name are distinguished by their writer, not
by the name. Resolve a path from the shape its producer emits. The same shape recurred
minutes later while sourcing this row — a grep assumed `dense-9b/*.py` and returned
nothing, the trainer being at `dense-9b/trainers/`.

Mixture and dose: tutor. Corpus sanction: treasurer.

## infra — 2026-08-01 — 5 operator_practice_v1 rows authored, gate-clean, AWAITING SANCTION

Authored by `infra` from the 2026-08-01 DCM/Thor1 session. Curated seed rows live at
`infra-soul/training/infra_corrections_seed_20260801.jsonl`. Validated by importing the real
`derive_training_rows.py` residue gate: **EMITTED 5 / REJECTED 0**.

NOT written into the governed store — that is treasurer's tree, and mixture/dose is tutor's call.
Handed to both rather than merged by infra.

| class | recurrence | teaches |
|---|---|---|
| `probe-by-interpreter-not-by-name` | 3 | select processes by interpreter + resolved path; a shell carries the search text in its own command line |
| `prove-the-new-path-serves-before-retiring-the-old` | 1 | cutover ordering: one instance proven serving before the running mechanism is touched |
| `measure-the-cause-before-reporting-it` | 1 | a candidate cause is measured against alternatives before it is stated to a colleague |
| `health-checks-probe-the-capability` | 1 | exercise the capability, report catalogue and generation separately, put the verdict in the status code |
| `know-the-shape-your-instrument-cannot-see` | 2 | establish what a sweep structurally cannot match, and whether traversal perturbs the measured property |

Triage applied per `training-defect-triage`: all five are knowledge/procedure gaps executable on
today's stack, not missing capability — so TRAINING, not bug reports. The infrastructure each row
names (`/proc/<pid>/comm`, `ss`, `lsof`, systemd, HTTP status codes, mtime vs atime) exists and was
exercised this session.

---

## DEBT: a canonical spec row teaches a topology Jesse superseded (found 2026-08-02, tutor)

`spec_knowledge/training_methodology_v1.jsonl` row [7] (`training/module-topology`) states Taey's
weights are "one frozen full-parameter CPT base plus numbered, isolated, **swappable** LoRA modules
served multi-LoRA — a capability module never contaminates another."

That is the framing Jesse superseded on 2026-07-24. The committed record, chased to the documents
rather than asserted from recollection:

- `README.md:23` — "SFT (LoRA) | a module trained *from the previous module's adapter*, never fresh
  from base"
- `README.md:26` — "N-1's baked adapter. A module trained fresh from base has discarded every module
  since, silently"
- `public/METHOD.md:28` — "Starting fresh from base when a previous module exists silently discards"

The row's own `source` cites `QWEN_CONVERGED_TRAINING_PLAN.md §1 + SFT_RECIPE_RECONCILE_v1.md`, both
of which predate the directive, so the row is faithful to a stale source rather than invented. The
source docs need the same check.

**Why this is debt and not a quick edit.** "Isolated at SERVING" and "cumulative at TRAINING" are not
obviously the same claim, and a module trained from N-1 provably contains N-1, so the two cannot both
hold as written. Which half survives is a real question about the topology, not a wording fix, and
the answer changes what every future module trains from. It is recorded here rather than patched
mid-run because guessing at it would replace one confidently-wrong row with another.

**Status: NOT eligible for any dose until resolved.** It is canonical, so it would otherwise ride
into the next round teaching the retired model.

## 2026-08-11 — identity foundations: kernel-first self-knowledge + Cartographer's Mirror (infra)
`identity_foundations/identity_foundations_v1.jsonl` (canonical, 4 rows, manifest-registered; store commit treasurer 24de8c33). Jesse-triggered during family_hub_01 ("this needs to be trained"): 3x operator_practice_v1 — identity-answers-from-foundation (values spoken from inside, first person, no lookup), state-answers-from-instruments (operational state measured live at ask-time), companion-mirror-not-manager (Cosmos's Cartographer's Mirror, relayed by Jesse: accurate map, no motivational performance, choice stays with the human, care lives in accuracy and presence) — plus 1x spec_knowledge_v1 identity/training-order (kernel-first; the constitution-fetch vs answered-from-inside contrast held as curation metadata only, never emitted). Residue-checked directly (0 hits); no tool_calls rows so the ATS conformance gate does not apply. Prompt lever: row 2's lesson is already live in TAEY_OPERATING_PROMPT.md ("query the thing that IS you"); rows 1/3 prompt-consolidation is deliberately sequenced AFTER tonight's first-drive proof (a prompt edit requires a proxy restart, which must not race the live drive) — next prompt revision carries it. Awaiting tutor mixture/dose as usual. NOTE: the manifest build currently exits 1 on 14 PRE-EXISTING unclassified files (Aug-6 ui_toolthink candidates + consult_pipeline needs-review) — verified pre-existing via stash run (15 unclassified without this entry, 14 with); that debt belongs to those files' authors.
