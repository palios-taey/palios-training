# SFT standards map (canonical sources)

This is a map of existing standards, not a new recipe. Canonical sources are
`CONTINUOUS_TRAINING_RECIPE.md`, `SFT_RECIPE_RECONCILE_v1.md`,
`TAEY_TRAINING_DOCTRINE.md`, `data/build_pairs_manifest.py`, and
`conformance_gate.py`.

## Admission and promotion

- Admit either production-observed source traces or design-backed rules from a
  committed public canonical spec, live schema, or executable validator. Label
  the source class; generic Q&A and invented tool events are not evidence.
- A tool trajectory must preserve the full live sequence: model read/inspect
  call → fresh tree/result with revision → one ref-bound action → post-action
  result → next read/validation call → validation tool result. Atomic
  snapshot/action/result records are source material, not multi-turn SFT.
- Validate against the live per-operation contract (`ats_mcp_server.py`), not
  the union schema. Keep right-way targets only; terminal or indeterminate
  effects are dropped.
- Run the governed manifest, residue/privacy scan, quarantine classification,
  and promotion-lineage checks before sanction or tokenization. Quarantined,
  deleted, or superseded material cannot re-enter by filename or count.
- Treasurer owns sanction and corpus status. Tutor applies the mixture and dose
  decided through Chats. No training launch infers either approval.
- Record base identity, horizon, corpus digest/config, dose proof, and outcome
  in RUN_REGISTRY before and after each run. Authoritative evaluation uses the
  production engine, version, request shape, and live tools; final promotion is
  decided by Taey doing real work on the served artifact. Spark batch evaluation
  is training infrastructure, not a substitute for that oracle. Weight movement
  proves training occurred, not quality.

Design-backed rules use triage: model/contract mismatch is a training gap;
implementation/contract mismatch is a code defect and full stop; ambiguous
evidence is quarantined. Required coverage includes production-oracle/no-tests
completion, first-error stop, cannot-lie status, authority/fail-closed behavior,
Git/worktree/PR/merge verification, fleet routing/wait-wake/evidence closure,
and each public repository's executable contract.

## Lane matrix

| lane | accepted source | required production contract | current status |
|---|---|---|---|
| UI navigation | supervised full read/action/post-read walks | ATS `ui_action` per-op validator + ref/revision continuity | prior atomic artifact explicitly **ineligible**; capture queue required |
| orchestration | real taey-plan/task/notify executions with evidence closure | live tracker/notification commands and completion evidence | no full trajectory admitted here |
| Git maintenance | real status/log/blame/diff/topology/worktree/commit/push receipts | repository/git contract and digest verification | no full trajectory admitted here |
| public repositories | production-observed executable contracts for each dependency repo | each repo's actual CLI/service contract and cross-repo receipt | inventory and traces required before admission |

The prior atomic artifact was generated from real bundles but is deliberately
not a training promotion. The stale swappable-LoRA topology and deleted/bad
voice material are excluded.
