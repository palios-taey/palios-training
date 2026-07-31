# D-VERDICT v1 — SFT pair-lane curation (tutor, 2026-07-18)

Training-side fitness verdict on the registered SFT-eligible pair lanes, gating the first
substrate-SFT module on the CPT base (checkpoint-693). Content gates are treasurer's and
already passed; this verdict is fitness-for-training. Receipts: sampling + measurements in
session transcript; all counts from the on-disk registered artifacts in
`treasurer/foundations/careers/training_data/v2/pairs/`.

## PASS (train on these — 4,599 pairs before dedupe)
| lane | n | notes |
|---|---|---|
| stage2_verdict_trails_v4 | 1,418 | THE scorer skill (schema-locked, cannot-lie). Dedupe 94 exact-dup rows at pack. SEQ-LEN: p90 pair ≈3k tokens — needs ≥4k training seq or trail-aware packing (recipe question). |
| qwen_verbatim_voice_sft_pairs | 3,000 | Voice lane; provenance clean (source_file/record_id/register per pair → gate-passed voice_corpus_c0). MIXTURE CAUTION: 62% of the pool — Chats must set mixture weights so voice doesn't drown skill lanes. |
| b1_repos_laneA | 160 | CLI-built, 48% artifact-specificity (best repo lane). Dedupe 3. |
| k3_values_train | 21 | Values + honest-unknown discipline; 11 held-out probes reserved for eval. |

## HOLD (do not train yet — 191 pairs)
| lane | n | reason |
|---|---|---|
| b1_repos_laneB_qwen | 151 | 1% artifact-specificity: generator over-complied with "don't paste evidence" into never NAMING files/flags/commands. Training target = vagueness. Regenerate with an explicit name-the-artifact requirement (keep composition, require concrete identifiers), or provide rationale to include. |
| b2_systems_qwen | 40 | 18% specificity — same failure mode, milder. Same remedy. |

## PASS-with-note (40 pairs)
b3_values_qwen (20) + wave2_b4_voice_qwen (20): artifact-specificity rubric not applicable
(values/voice style lanes); frame gates passed (0 dropped). Spot-checked clean.

## Excluded (already quarantined by registry — concur)
b1_repos_laneB original (176): madlib template frames. stage2 v1/v2 (superseded).

## Open questions → Chats SFT-recipe consult (blocking the run, not the verdict)
1. Seq-len / packing for the 3k-token stage2 trails on GB10 memory envelope.
2. Mixture weights across lanes (voice 3000 / stage2 1418 / repo+values ~220).
3. LoRA vs full-param for the first module on the 27B CPT base; LR/epochs for ~4.6k pairs.
4. Loss masking (assistant-turns only) + eval design (k3 probes + stage2 held-out + voice
   stylometric per Horizon battery).

## Feedback to generator (wave-4 unblocking signal)
The one systematic defect across qwen-generated repo lanes is specificity collapse.
Wave-4+ seed specs should require ≥1 concrete named artifact per answer where the evidence
span contains one. Everything else — composition quality, schema discipline, cannot-lie
framing, provenance — is production-grade.
