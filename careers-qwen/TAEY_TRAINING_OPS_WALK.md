# TAEY TRAINING-OPS WALK — spec v0.1 (tutor, 2026-07-24)

**Jesse (2026-07-24):** "Will Taey be able to drive this training loop? … you guys really need to be
doing the 'what's next' and advanced thinking and coding for them to enable everything."
**This spec is that enablement:** the training loop's operational steps, packaged in the grain Taey is
PROVEN to execute (the per-step 2–6KB packet with exact-command + expect + verify — the jobs-walk
grain; 54–207KB mega-packets measurably degrade instruction-following). Taey operates the loop's
mechanics through `exec_command`; the judgment seats stay where the law puts them.

## What Taey DRIVES (mechanical steps, per-step packets, day one)
| step | packet content | verify |
|---|---|---|
| RUN-MONITOR | `ssh spark@<master> grep [step/SR-DELTA/error] r0.log \| tail` — report step, loss, SR-DELTA ×ULP vs the band table (LIVE by 10–15; PASS@40 [0.5u,20u]) | three-register report; a claim cites the log line |
| SESSION-ACCOUNT | verify checkpoint bank (`ls checkpoint-*`), append hashchain (`md5sum __0_0.distcp`) | hash present + step advanced |
| GATE-READ | 50-step cheap-gate arithmetic (weight-delta vs 10× bf16 noise; loss slope) from logged values | verdict = numbers vs bands, never narrative |
| BAKE | disk-check ≥60G → launch the bake command (env per RECIPES) → verify artifact (tensor count, config.json, size, md5) | real-artifact receipt before "baked" |
| PROBE-RUN | execute eval batteries (`eval_probes.py`, regurgitation probe script) → COLLECT results into the governed store | results filed; **rendering verdicts is NOT this step** |
| PACK-ASSIST | run pack commands (`pack_corpus.py`, `pack_sft_module1.py`) → verify manifest zero-drift + shas | manifest receipt |
| CONSULT-DRAFT | draft a Family packet from the template (facts + bands + question) | tutor reviews before dispatch (trainable toward solo) |
| RCA-SEAT | participate in every training RCA (charter §5; account = hypothesis gated on the mechanical record) | already live |

## What Taey ADVISES on but does NOT decide (the law, unchanged for every seat)
- **Recipe** (LR/epochs/optimizer/mixture bands) — Family consult sets; Taey drafts and flags.
- **Corpus sanction** — treasurer. **D-verdicts** — a non-producer seat. **GATE-0/cutover sign-off** — Jesse.
- **Trainer code changes** — tutor + consult (Taey flags violations per its methodology training).
- Structural: **producers never self-certify** — Taey never grades its own pairs, never passes its own gate.

## Rollout (each stage gated on the prior's receipts)
1. **NOW:** RUN-MONITOR + SESSION-ACCOUNT packets on the live M1/CPT runs (read-only, zero risk) —
   authored by tutor from this spec, run on the Thor ep3 seats alongside the revenue walks.
2. **Post-M1-bake:** BAKE + PROBE-RUN packets (write steps, receipt-gated).
3. **Post-M-METHOD-in-weights:** CONSULT-DRAFT quality rises from templated to understood; violation-
   flagging becomes reflex (the spec rows' trained shape).
4. **Steady state:** Taey runs the loop's mechanics end-to-end; humans+Family hold exactly four seats —
   recipe, sanction, verdict, sign-off. Jesse's role shrinks to the two calls per module he already has.

## Packet authoring rules (inherit the proven walk discipline)
Per-step 2–6KB; exact first command verbatim; `expect` on every step; narrowed observation (never full
log dumps — tail/grep slices); three-register reporting; blocked-honest beats silent; every fumble or
long-think during a walk → a training row same-day (the standing triggers).
