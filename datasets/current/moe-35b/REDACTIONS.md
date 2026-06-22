# Redactions applied to this training data

For security, a small set of strings was redacted from the published `.jsonl` before release. These were **incidental strings inside synthetic infra-Q&A content** (the model was taught about our cluster), not training-load-bearing signal — the data is otherwise byte-faithful to what trained the model.

## What was redacted

| Item | Replacement | Why | Logical count in `combined_v1_gated.jsonl` |
|---|---|---|---|
| Internal cluster network addressing — private management subnet and dual-rail RoCE subnets (and CIDR/`.x` forms) | `<REDACTED-NET>` | Publishing internal network topology has zero value to a reproducer (you use your own cluster addressing) and reconnaissance value to an attacker. | 693 |
| One dead local dev credential (former Neo4j password, no longer in use) | `<REDACTED-CRED>` | Credential hygiene, even though it is dead. | 1 |

Verification: after redaction, all shipped JSONL files remain valid JSON. The logical production SFT corpus is `combined_v1_gated.jsonl` (2,325 rows); the two shipped SFT input files (`constitutional_gated.jsonl` + `phase1_infra_v2_gated.jsonl`, 1,378 + 947 rows) regenerate that combined file with the seed-42 combiner. The combined public SHA is `6ecb0e82cff562d5ed851cb51bc8b445706592665e0c779d8f12271f05a780ad`; the pre-scrub original SHA was `6b54f163c0dfc35ed7cae4637146a0a959ff33f58601922a95f3cff7641dabfd`. A full secret scan (`gitleaks`) is clean.

## What was NOT redacted (transparency)

- **Already-published third-party emails** that ride inside public reference documents used as knowledge-SFT — e.g. arxiv paper author correspondence (Switch Transformer / abliteration authors), vendor support addresses (NVIDIA, Neo4j, vLLM), an open-source man-page copyright line. These are public addresses already in the source papers/docs/man-pages, not private contact data, so they are retained as part of the faithful source content.
- First-party PALIOS-TAEY identity/relationship content and the Chewy collection — included by choice.

To reproduce against your own cluster, substitute your node addressing where you see `<REDACTED-NET>` (the recipes already take node addresses via `NODE0_IP..NODE3_IP` / `NODE_RANK`).

## Operator de-identification (public release, 2026-06)
All **linkable** operator identifiers are removed from the corpus, configs, recipes, and audit artifacts:
surname/handle (`user`), email, `/Users/<operator>` + personal photo paths, and operator/fleet home
paths (`/home/{mira,spark,thor,jesse,facilitator}` → `/home/user`). Generic paths (`/home/ubuntu`) are kept.

The bare first name "Jesse" is **retained** in the training corpus by design: it is the narrative
subject, it is below the linkable-PII bar (common first name; the project-to-person link is already public via
the `@user` handle, so scrubbing it adds ~zero privacy), and the trainer's anonymity rule
(`audit/audit_pipeline.py` CORRECTION_INSTRUCTIONS) already governs that the **model** does not emit the name
even though "the corpus names them." Decision: remove-linkable-identifiers-only, first name retained.
