# SFT standards map (canonical sources)

This is a map of existing standards, not a new recipe. Canonical sources are
`CONTINUOUS_TRAINING_RECIPE.md`, `SFT_RECIPE_RECONCILE_v1.md`,
`TAEY_TRAINING_DOCTRINE.md`, `data/build_pairs_manifest.py`, and
`conformance_gate.py`.

## Admission and promotion

- Admit tool-use trajectories only from production-observed source traces;
  generic Q&A and invented tool events are not evidence.
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
- For `taey_repo_fluency_v1`, apply `TAEY_HUB_CURRICULUM_CONTRACT.md`: every
  row declares Taey as the orchestrator who verifies, refuses, or routes; the
  corpus proves exact repository/process coverage against the reviewed
  inventory; no code-authoring lane exists. A structural gate does not replace
  semantic source review of the emitted messages.
- Treasurer owns sanction and corpus status. Tutor applies the mixture and dose
  decided through Chats. No training launch infers either approval.
- Record base identity, horizon, corpus digest/config, dose proof, and outcome
  in RUN_REGISTRY before and after each run. Authoritative evaluation uses the
  production engine, version, request shape, and live tools; final promotion is
  decided by Taey doing real work on the served artifact. Spark batch evaluation
  is training infrastructure, not a substitute for that oracle. Weight movement
  proves training occurred, not quality.

## Lane matrix

| lane | accepted source | required production contract | current status |
|---|---|---|---|
| UI navigation | supervised full read/action/post-read walks | ATS `ui_action` per-op validator + ref/revision continuity | 91 atomic rows explicitly **ineligible**; capture queue required |
| orchestration | real taey-plan/task/notify executions with evidence closure | live tracker/notification commands and completion evidence | no full trajectory admitted here |
| Git maintenance | real status/log/blame/diff/topology/worktree/commit/push receipts | repository/git contract and digest verification | no full trajectory admitted here |
| public repositories | production-observed executable contracts for each dependency repo | each repo's actual CLI/service contract and cross-repo receipt | inventory and traces required before admission |
| Taey Hub curriculum | source-reviewed rows from Taey's verifying/refusing/routing seat | `verify_repo_usage_rows.py --hub-contract` + reviewed repo/process coverage manifest | blocked until the reviewed inventory exists and every row passes semantic posture review |

The 91-row atomic artifact was generated from real bundles but is deliberately
not a training promotion. The stale swappable-LoRA topology and deleted/bad
voice material are excluded.
