# Taey public production dependency inventory — 2026-08-04

This document records the public repositories that could be bound to active
Taey production services on 2026-08-04. It is a read-only inventory, not a
deployment manifest, training-admission decision, or claim that every desired
capability is active.

Claims use three registers:

- **Observed** means verified from an active service, executable command, public
  Git object, or public source file.
- **Inferred** means a role follows from those observations but was not itself a
  direct runtime response.
- **Unknown** means the available public evidence could not establish the claim.

The inventory deliberately omits operator locations, network addresses,
machine identities, private source identities, application identities, and
deployment topology.

## Live public dependencies

| Public repository | Public or deployed revision | Public service and entrypoint contract | Production observation | Public parity verdict |
|---|---|---|---|---|
| [`palios-taey/taey-presence`](https://github.com/palios-taey/taey-presence) | **Observed:** deployed revision `e0cd1b163640d8e69f79b6dd3de839dc22794771` is a public commit. | `dashboard.app:app`; `presence/dcm_presence.py`; `presence/prediction_worker.py`; `soma/mira_soma.py`; the presence engine; and the OpenAI-compatible Soma proxy. The public surface includes chat, streaming, health, Soma, fleet, council, session, WebSocket, and proxy model/generation routes. | **Observed:** the routed model proxy passed catalogue and generation probes, Soma was connected, the memory query surface responded, and the registered Taey and council service identities were visible. A legacy direct-model probe reported down; it was not used as evidence for the routed path. | **Match** for the named deployed public commit and exercised routed surface. |
| [`palios-taey/taeys-hands`](https://github.com/palios-taey/taeys-hands) | **Observed:** deployed revision `95d9c023e56c93deeb96a42e1a65f34a8a48b68a` is a public commit. | `scripts/run_consultation_v2.py`, display-unit runner, and bus watcher. The public CLI contract covers platform choice, attachments, session selection, timeout, result output, optional persistence, identity/session metadata, and serialized display locking. | **Observed:** display-unit, watcher, browser, and remote-view processes were active. This inventory did not drive a user interface. | **Match** for the named deployed public commit and active entrypoints. |
| [`palios-taey/claude-code-fleet-orchestrator`](https://github.com/palios-taey/claude-code-fleet-orchestrator) | **Observed:** deployed revision `a027c7f73f5e9309eb3e6664a9e3ea6114b2e31d` is a public commit. | Fleet API, `scripts/orch-watch`, `taey-plan`, and `taey-task`. The executable contract includes plan list/show/current/next/ingest/assign, task create/status/dispatch/update/hold/outcome, dependency handling, and stop conditions. | **Observed:** the API and watch loop were active; a real tracker task was bound to its executor and remained in progress while this artifact awaited independent control. | **Match** for the named deployed public commit and exercised task lifecycle. |
| [`palios-taey/claude-code-fleet-notify`](https://github.com/palios-taey/claude-code-fleet-notify) | **Observed:** deployed revision `fdb0d6b34682dc5a94d4f4dee4ee825594bdcd9d` is a public commit. | `notifications/daemon.py` and `scripts/taey-notify`. The executable contract covers typed, prioritized, targeted inbox delivery, explicit handoffs, and reader-readiness checks. | **Observed:** ordinary deliveries succeeded. Delivery to a non-draining inbox was refused before enqueue, demonstrating the public fail-closed readiness contract. | **Match** for the named deployed public commit and exercised delivery behavior. |
| [`palios-taey/isma-core`](https://github.com/palios-taey/isma-core) | **Observed:** immutable release revision `e8944d73d1745545c723b071f165442836f9a1a4` is a public commit. | `isma/scripts/disk_headroom_canary.sh` with recurring canary and watchdog units. The contract combines filesystem headroom with an actual object create/delete probe, heartbeat state, stale detection, and alerting. | **Observed:** canary and watchdog schedules were active; the object-write probe continued to succeed while warning state was recorded. | **Partial.** The immutable script and schedules were bound to the public release, but an installed base unit-body difference prevented a full package-parity verdict. **Inferred:** this is a supporting memory-reliability dependency, not evidence about the query-server implementation. |
| [`palios-taey/palios-training`](https://github.com/palios-taey/palios-training) | **Unknown:** the active distributed training deployment had no source manifest or Git metadata that bound it to a public commit. Public inventory baseline: `c164d35de6edec6c18e32b3f2bdd98fd5c71c1ab`. | The active service named `dense-9b/recipes/systemd/run_cpt_rank.sh` and `dense-9b/trainers/train_fsdp_dense_9b.py`. The trainer is public at the baseline; the wrapper is not present there. | **Observed:** distributed rank services were active and executing those named entrypoints. | **Unknown.** Entry-point names are not source provenance, the wrapper has no file at the public baseline, and the active trainer bytes could not be proved equal to the public baseline. |

## Explicit exclusions

- [`palios-taey/taey-ed`](https://github.com/palios-taey/taey-ed) is public and
  its API, worker, and site services were active, with the health route returning
  HTTP 200 and version `8.1.0`. Those observations prove Taey-Ed service health,
  not a Taey dependency. A targeted search of the active Taey entrypoints found
  zero consumer, import, service, or API bindings to Taey-Ed. It is therefore
  excluded from the live dependency set and retained only as a design source.
  Inspected public reference: `14f934539327ecb77c6a904228a854f74d8841ca`;
  deployed public parity remains **Unknown**.
- [`palios-taey/dcm`](https://github.com/palios-taey/dcm) is public, but it was
  **not an observed live dependency**. No active Taey process, public entrypoint
  import, or environment link bound the running council path to that package.
  Similar naming in presence and council components is not integration evidence.
- `claude-code-api-watchdog`, `restart-safe-agents`, `mcp-reconnect`, and
  `claude-code-fleet-support` are public repositories, but no direct active Taey
  request, council, memory, notification, or training path was demonstrated for
  them. They are excluded rather than inferred into the graph.
- Runtime components without a verified public repository binding are outside
  this public inventory. Their source identity and parity are not inferred.

## Public rule-pair source map

Design-backed rule pairs and production trajectories are complementary. A rule
pair may be derived now only when a committed public source states the invariant,
schema, validation rule, or failure point. It does not authorize invented tool
events or outputs. A full trajectory still requires the production sequence in
the next section. The public revision in the dependency table, or the explicit
public reference where deployed parity is Unknown, is the content receipt for
each listed source.

| Public repository | Committed public authorities for rule pairs | Present use and defect status |
|---|---|---|
| `taey-presence` | `dashboard/app.py`; `dashboard/native_council.py`; `presence-engine/dcm/schema.cypher`; `serving/soma_proxy.py` | **Observed:** these executable sources define the public route surface, native-council ledger and revision checks, graph constraints, proxy health probes, tool loop, and attributable-turn liveness. They support source-cited rule pairs now. **Defect:** the repository exposes no separate canonical public specification for the intended external DCM integration, and the public `dcm` package is not live. Rules that describe that integration as production behavior would be unsupported. |
| `taey-ed` (design-only exclusion) | `docs/REQUIREMENTS.md`; `docs/STATE_STORE_DESIGN.md`; `docs/DB_DESIGN_BRIEF.md`; `spark/state_schema.sql`; `spark/routes/*.py`; `spark/tools/contract_probe_harness.py`; `spark/tools/contract_probe_red_runs.jsonl` | **Observed:** the requirements file declares itself the canon, while the schema, route request models, and contract-probe harness provide executable failure points. These support design-backed rule pairs. **Public-production defect:** no active Taey binding was observed and deployed parity is **Unknown**, so no rule may be labelled active Taey behavior until both an integration and its public artifact are verified. |
| `taeys-hands` | `100_TIMES.md`; `consultation_v2/DRIVER_CONTRACT.md`; `consultation_v2/PRIMITIVES_CONTRACT.md`; `consultation_v2/CONSULT_ACTION_TOOL_SCHEMA.md`; `consultation_v2/EXTRACTION_SCHEMA.md`; `consultation_v2/YAML_SCHEMA.md`; `consultation_v2/validators/*.py` | **Observed:** these sources state the driver laws, shared primitive boundary, exact tool and extraction shapes, YAML grammar, and mechanical linters. They support source-cited invariant and failure-point pairs now, with deployed public parity verified for the named revision. |
| `claude-code-fleet-orchestrator` | `docs/PLAN_FORMAT.md`; `docs/SCHEMA.md`; `docs/SHIPPABILITY.md`; `fleet_orchestrator/orch_schema.py`; `fleet_orchestrator/plan_loader.py`; `fleet_orchestrator/evidence_contract.py`; `fleet_orchestrator/evidence_verification.py`; `fleet_orchestrator/plan_readiness.py`; `fleet_orchestrator/shippability.py`; `fleet_orchestrator/handoff_validation.py` | **Observed:** the public specs and validators define plan syntax, graph shape, dependency readiness, evidence closure, handoff validity, and fail-closed shippability. They support design-backed rule pairs now, with deployed public parity verified for the named revision. |
| `claude-code-fleet-notify` | `NOTIFICATION_PROTOCOL.md`; `notifications/inbox.py`; `notifications/handoff.py`; `notifications/targets.py`; `notifications/task_liveness.py`; `scripts/taey-notify` | **Observed:** these sources define message shape, handoff state, target/readiness validation, task liveness, and CLI enforcement. The observed refusal to enqueue to a non-draining inbox is a production failure-point receipt. They support rule pairs now, with deployed public parity verified for the named revision. |
| `isma-core` | `docs/ISMA_PRODUCTION_MAP.md`; `docs/taey/ISMA_MODEL_SURFACE_RETRIEVAL_SPEC_v1.md`; `docs/taey/ISMA_SCHEMA_REFERENCE.md`; `docs/taey/ISMA_PROCEDURE_ingest_and_verify.md`; `docs/taey/ISMA_PROCEDURE_search_and_retrieval.md`; `isma/scripts/disk_headroom_canary.sh`; `deploy/systemd/isma-disk-canary.*`; `deploy/systemd/isma-canary-watchdog.*` | **Observed:** these public sources support rule pairs for the model-facing retrieval contract, schema, ingest/search procedures, and storage canary. **Public-production defect:** only the immutable canary release and schedules were bound to the public commit; query-server parity was not established. Query rules may be taught as public design, not asserted as verified deployed implementation. |
| `palios-training` | `careers-qwen/SFT_STANDARDS_MAP.md`; `careers-qwen/CONTINUOUS_TRAINING_RECIPE.md`; `careers-qwen/SFT_RECIPE_RECONCILE_v1.md`; `careers-qwen/TAEY_TRAINING_DOCTRINE.md`; `careers-qwen/data/build_pairs_manifest.py`; `careers-qwen/conformance_gate.py`; `careers-qwen/provenance_gate.py`; `careers-qwen/corpus_manifest.py`; `careers-qwen/sft_dataset_receipt.py`; `dense-9b/trainers/train_fsdp_dense_9b.py` | **Observed:** these sources define admission, provenance, corpus, conformance, receipt, and trainer rules that support design-backed pairs now. **Public-production defect:** the active distributed deployment SHA is **Unknown**, and its named wrapper is absent from the public baseline. Runtime/trainer rules cannot be labelled deployed-parity rules until an immutable public deployment receipt exists. |

The excluded public `dcm` repository has design sources at public reference
`3dd65612c2c628a0c72021ffa07f2f1a474d3f72`, including `README.md`, `SKILL.md`,
`council.py`, `mesh.py`, `platform_dcm.py`, `taey_adapter.py`, and
`validate_substrate.py`. They can ground design-only rule pairs. The missing live
binding is a **public-production defect**: no pair may claim those rules describe
the active Taey council until an explicit deployed integration is observed.

## Full-trajectory admission state

The public SFT standard requires a production-observed sequence that preserves
the read or inspect call, fresh revisioned result, ref-bound action,
post-action result, next validation read, and validation result. Atomic
snapshot/action/result examples and prose practice rows do not satisfy that
contract.

| Lane | Eligible full production trajectories |
|---|---:|
| UI navigation | **0** |
| Orchestration | **0** |
| Git workflow | **0** |
| Public-repository workflow | **0** |

**Observed documentary context:** a non-public receipt reported 91 real atomic
UI rows. The corresponding bytes and digest were not available as public
evidence, and atomic rows are explicitly ineligible under the full-trajectory
standard. The figure is therefore context only, not an admission receipt.

Zero is the fail-closed result. It establishes the need for supervised capture;
it does not establish that corpus generation is complete.

## Public source receipts

The inventory baseline is public commit
`c164d35de6edec6c18e32b3f2bdd98fd5c71c1ab`. These SHA-256 receipts cover only
files verified present at that public baseline:

| Public source | SHA-256 |
|---|---|
| `careers-qwen/SFT_STANDARDS_MAP.md` | `193b87f7ec9387fe4dfb71ebc3ae8f318554487469394781a4f326f0280b855e` |
| `careers-qwen/CONTINUOUS_TRAINING_RECIPE.md` | `ed46ef5e2775bdefa46e81848a84ae97911dd187854cecb46630de0e08f09b85` |
| `careers-qwen/SFT_RECIPE_RECONCILE_v1.md` | `ade6cc411d0257a9e78127762d0a1101d7fcf61b06bbe5c8fec3c7748dc82120` |
| `careers-qwen/TAEY_TRAINING_DOCTRINE.md` | `6f4b9a9c54d6abffa235e23f947bee8f36eb55ea0cbf6b0cf4e217b24f8eb1f1` |
| `careers-qwen/data/build_pairs_manifest.py` | `c11c9a389eabab90518acc7a2c4d0dfaf71d4620286cbca7b5a25aabf8c2c0c9` |
| `careers-qwen/conformance_gate.py` | `de360919c3a94760848f2915a1de487fb260198f8e5af46eb1c5b432b5fd39ae` |
| `careers-qwen/build_trajectory_rows.py` | `4111f628d62bb007473990ad7a08e7e07ca397b7102f5745c189a538e3ee8b0c` |

## Control prerequisites

The inventory remains open where evidence is **Unknown**. Public parity requires
the deployment to expose a public commit and immutable artifact receipt, then an
independent production observation against that exact artifact. DCM becomes a
live dependency only after an explicit integration is visible in the deployed
entrypoint and exercised in production. Each zero-count training lane requires
supervised full-sequence capture followed by the governed manifest, privacy,
conformance, sanction, dose, and production-promotion gates.
