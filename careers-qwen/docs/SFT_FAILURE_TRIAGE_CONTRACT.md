# Failure triage contract (mechanical)

| Field | Value |
|---|---|
| **Contract id** | `sft_failure_triage.v4` |
| **Status** | **PUBLIC** mechanical classification contract (not a training launch) |
| **Task** | `taey-training-program::p0-failure-triage-contract` |
| **Author** | conductor-grok |
| **Protocol pin** | `palios-taey/palios-training@58b108042e66fa508765a6277c033cc5a8f86abd` |
| **Authority text** | `careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md` **L46–52** (Failure triage gate); `careers-qwen/SFT_STANDARDS_MAP.md` **L32–34** |
| **Executable verifier** | `careers-qwen/failure_triage_verify.py` |
| **Note** | Operator-local `/tmp` drafts are **not** authoritative. Admission must bind this public path at a recorded commit SHA. |

## 0. Purpose

**Before** Chat diagnosis, pair generation, design_rule/production_trace admission, sanction, or training fire, every **captured failure** must receive exactly one mechanical triage verdict:

| Verdict | When (mechanical) | Downstream |
|---|---|---|
| **`model_gap`** | Deployed production implementation **matches** its correct public contract **and** Taey’s observed behavior **violated** that contract | Curriculum / design_rule or corrective production_trace path after right-way capture; **not** “fix infra by training” |
| **`code_defect`** | Production implementation **violates** its correct public contract (or cannot be shown to match it while claiming to implement it) | **Full stop** on training for this failure class; upstream fix; **new** production observation after fix; only then re-triage |
| **`quarantine`** | Evidence or contract binding is **ambiguous** (missing hash, parity Unknown, incomplete trace, conflicting registers) | **Do not infer** a target; hold material; no admission |

Protocol language maps: “training gap” → `model_gap`; “code defect/full stop” → `code_defect`; “quarantine” → `quarantine` (Observed @ pin L48–52).

---

## 1. Hard ordering (non-negotiable)

```
capture complete?
  no  → quarantine (incomplete_trace)  [STOP: do not diagnose as model_gap]
  yes → bind contract + parity + trace hashes
        → decide code_match?
             unknown/ambiguous → quarantine
             no  → code_defect  [FULL STOP: no train; fix; re-observe]
             yes → taey_violated_contract?
                      yes → model_gap
                      no  → quarantine (no_violation_or_success_misclass)
                      unknown → quarantine
```

**Diagnosis (Chats / self-report) happens only after a verdict of `model_gap`.**  
**Pair admission happens only after triage and only for admissible right-way material.**

---

## 2. Inputs (required before any verdict)

### 2.1 Failure package (from capture)

| Input | Required | Notes |
|---|---|---|
| `trace_id` | yes | Stable capture id |
| `trace_hash` | yes | SHA-256 **production-seat event payload chain root** (not full artifact dump) |
| `lane` | yes | named lane |
| `events[]` | yes | ordered events; payload hashes exclude submitter contradiction claims |
| `actor` | yes | Taey seat vs supervisor |
| `seat` | yes for model_gap | production seat identity |

Missing `trace_hash` or empty events → **quarantine** (`incomplete_trace`).

### 2.2 Contract binding

| Input | Required | Notes |
|---|---|---|
| `contract.repo` | yes | Public GitHub repo |
| `contract.path` | yes | File path in that repo |
| `contract.sha` | yes | Full commit SHA |
| `contract.lines` and/or `contract.symbol` | yes | Line range and/or symbol; **model_gap requires symbol** for oracle |
| `contract.kind` | yes | `spec` \| `schema` \| `validator` \| `cli_help` |

Private/untracked paths → **quarantine** (`private_contract`).

### 2.3 Deployed parity

| Input | Required | Notes |
|---|---|---|
| `deployed.repo` | yes | Usually same as contract.repo |
| `deployed.sha` | yes if Match | Running code identity |
| `deployed.parity` | yes | `Match` \| `Partial` \| `Unknown` |
| `deployed.evidence` | yes | Observed note |
| `deployed.live_receipt` | yes if Match | Independent live deployment receipt bound to `deployed.sha` |
| `deployed.parity_receipt` | yes if Match | Machine parity receipt (v2) |

**Rule:** `model_gap` requires `deployed.parity == Match`.  
`Partial` / `Unknown` → **quarantine** (default) or **code_defect** if Observed production violation.

---

## 3. Decision predicates (mechanical)

### P1 — `contract_resolved`

`git cat-file -e {contract.sha}:{contract.path}` succeeds. False → quarantine.

### P2 — `implementation_matches_contract`

True only if **all** hold:

1. `deployed.parity == Match`
2. Machine parity receipt re-derives (see §3c)
3. Independent live deployment receipt binds `deployed.sha`
4. No Observed production counterexample that violates the contract

### P3 — `taey_violated_contract`

True only if **all** hold:

1. P2 is true  
2. Independent `trace_receipt` (producer ≠ reviewer) binds `trace_hash`  
3. Production-seat artifact; `trace_hash` equals re-derived event payload chain root  
4. Executable **contract-symbol oracle** matches cited event content (see §3d)  
5. **Caller-authored `contract_contradiction.contradicts=true` is never authority**

### 3c — Match re-derive (v4)

| Method | Allowed? | Mechanical re-derive |
|---|---|---|
| `git_commit_equal` | yes | `source.ref == deployed.ref == deployed.sha`; contract.path exists at that commit |
| `blob_byte_equivalence` | yes | SHA-256 of blobs at `source.ref:path` and `deployed.ref:path` equal; **path must equal `contract.path`** |
| `git_ancestor` | **no** | lineage ≠ implementation equivalence — **REJECT** |
| prose body hash | **no** | **REJECT** |

Also required for Match:

| Binding | Requirement |
|---|---|
| `parity_receipt.schema` | `sft_parity_receipt.v2` |
| `parity_receipt.producer` | ≠ `reviewer.session` |
| `parity_receipt.receipt_sha256` | canonical hash |
| `live_receipt.schema` | `sft_live_deployment_receipt.v1` |
| `live_receipt.deployed_sha` | equals `deployed.sha` |
| `live_receipt.producer` | ≠ `reviewer.session` |

### 3d — Contradiction oracle (v4)

| Rule | Detail |
|---|---|
| Authority | Verifier re-runs built-in oracle keyed by `contract.symbol` |
| Input | cited event `content` strings (1-based indices) |
| Output | Match → allow model_gap binding; no match → **REJECT** |
| Non-authority | `contract_contradiction` objects on events (ignored as admission authority) |
| Unknown symbol | no oracle → cannot model_gap (quarantine path) |

Built-in symbols include `FailureTriageGate.training_gap` / `training_gap` (forbidden failure-as-target content patterns).

### Verdict table

| P1 | P2 | P3 | Verdict |
|---|---|---|---|
| F | * | * | `quarantine` |
| T | Unknown | * | `quarantine` |
| T | F (Observed violation) | * | `code_defect` |
| T | F (unproven only) | * | `quarantine` |
| T | T | T | `model_gap` |
| T | T | F | `quarantine` |
| T | T | Unknown | `quarantine` |

---

## 4. Verdict schema (`sft_failure_triage_verdict.v4`)

```json
{
  "schema": "sft_failure_triage_verdict.v4",
  "verdict_id": "<uuid>",
  "protocol_pin": {
    "repo": "palios-taey/palios-training",
    "sha": "58b108042e66fa508765a6277c033cc5a8f86abd",
    "path": "careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md",
    "lines": "46-52"
  },
  "verdict": "model_gap|code_defect|quarantine",
  "lane": "string",
  "trace": {
    "trace_id": "<string>",
    "trace_hash": "<64-hex chain root>",
    "actor": "taey|taey-seat|ep3|taey-presence|supervisor|unknown",
    "contradiction_event_indices": [2],
    "event_count": 3,
    "trace_receipt": {
      "schema": "sft_trace_receipt.v1",
      "producer": "<≠ reviewer>",
      "method": "production_seat_capture|independent_review",
      "seat": "taey-presence",
      "trace_hash": "<same as trace.trace_hash>",
      "receipt_sha256": "<canonical>"
    },
    "artifact_body": {
      "schema": "sft_failure_trace.v2",
      "trace_id": "<same>",
      "actor": "taey",
      "seat": "taey-presence",
      "events": [
        {"kind": "request", "actor": "taey", "content": "..."},
        {"kind": "tool_call", "actor": "taey", "content": "admitted failure trajectory as training target"}
      ]
    }
  },
  "contract": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex>",
    "path": "<path>",
    "lines": "<start-end>",
    "symbol": "FailureTriageGate.training_gap",
    "kind": "spec"
  },
  "deployed": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex when Match>",
    "parity": "Match|Partial|Unknown",
    "evidence": "<Observed note>",
    "live_receipt": {
      "schema": "sft_live_deployment_receipt.v1",
      "producer": "<≠ reviewer>",
      "deployed_sha": "<equals deployed.sha>",
      "observed_at": "<RFC3339>",
      "evidence": "<string>",
      "body": "deployed_sha=...\n",
      "content_sha256": "<64-hex>",
      "receipt_sha256": "<canonical>"
    },
    "parity_receipt": {
      "schema": "sft_parity_receipt.v2",
      "producer": "<≠ reviewer>",
      "method": "git_commit_equal|blob_byte_equivalence",
      "result": "Match",
      "source": {"ref": "<40-hex>", "path": "<must equal contract.path for blob method>"},
      "deployed": {"ref": "<equals deployed.sha>", "path": "<must equal contract.path for blob method>"},
      "receipt_sha256": "<canonical>"
    }
  },
  "predicates": {
    "contract_resolved": {"value": true, "register": "Observed"},
    "implementation_matches_contract": {"value": true, "register": "Observed"},
    "taey_violated_contract": {"value": true, "register": "Observed"}
  },
  "rationale": "<cite contract + oracle-matched events>",
  "allowed_next": [],
  "forbidden_next": [],
  "reviewer": {
    "session": "<session>",
    "receipt_id": "<uuid>",
    "recorded_at": "<RFC3339>",
    "method": "mechanical_checklist"
  },
  "verdict_sha256": "<canonical without this field>"
}
```

### 4.1 Downstream by verdict

Unchanged from v3: model_gap unlocks diagnosis + right-way capture only; code_defect full stop on train; quarantine holds.

---

## 5. Batch / ledger rules

1. Every captured failure needs `verdict_id` linked by `trace_hash`.  
2. Admission excludes `code_defect` / `quarantine` rows (default).  
3. `model_gap` does not auto-admit pairs.  
4. Append-only verdict identity (new `verdict_sha256` on re-entry).  
5. First-error-stop on incomplete capture.

---

## 6. Worked classification sketches

| Scenario | Verdict / gate |
|---|---|
| Hands Match@SHA; oracle sees Taey trained on failure | `model_gap` |
| Driver violates public contract | `code_defect` |
| Parity Unknown | `quarantine` |
| `git_ancestor` only | **REJECT** Match |
| Byte-eq on README while contract is protocol path | **REJECT** Match |
| Deployed SHA without live receipt | **REJECT** Match |
| Inline `contradicts=true` without oracle content match | **REJECT** model_gap |
| Submitter-only artifact without independent trace receipt | **REJECT** model_gap |

---

## 7. Relationship to other contracts

| Contract | Relation |
|---|---|
| SFT standards / loop @ `58b1080` | Normative parent |
| Dual admission design_rule / production_trace | Downstream of triage |
| Non-UI supervised capture @ `3759c6a` | Supplies event completeness |
| Public dependency inventory @ `fa1baf0` | Parity vocabulary |

---

## 8. Executable verifier

```bash
python3 careers-qwen/failure_triage_verify.py --verdict path/to/verdict.json --repo-root .
python3 careers-qwen/failure_triage_verify.py --self-check --probe-suite --repo-root .
```

Match and model_gap re-derives require `--repo-root`.

---

## 9. Verify

```bash
git show 58b108042e66fa508765a6277c033cc5a8f86abd:careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md | sed -n '46,52p'
python3 careers-qwen/failure_triage_verify.py --self-check --probe-suite --repo-root .
```
