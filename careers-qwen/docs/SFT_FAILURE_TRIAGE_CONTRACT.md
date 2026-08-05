# Failure triage contract (mechanical)

| Field | Value |
|---|---|
| **Contract id** | `sft_failure_triage.v2` |
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
**Pair admission happens only after triage and only for admissible right-way material** (protocol §6); failures are curriculum evidence, not targets, unless a governed format explicitly permits residue-gated failure material (default: **exclude**).

---

## 2. Inputs (required before any verdict)

### 2.1 Failure package (from capture)

Must include enough to hash:

| Input | Required | Notes |
|---|---|---|
| `trace_id` | yes | Stable capture id |
| `trace_hash` | yes | SHA-256 over canonical ordered event payloads (or seat-defined chain root) |
| `lane` | yes | `ui` \| `orchestration` \| `git` \| `public_repo` \| other named lane |
| `events[]` | yes | At least request + outcome/failure marker; prefer full supervised capture classes |
| `model_identity` | when model participated | e.g. `ep3` + engine `root` |
| `actor` | yes | who ran tools (Taey seat vs supervisor script) |

Missing `trace_hash` or empty events → **quarantine** (`incomplete_trace`).

### 2.2 Contract binding

| Input | Required | Notes |
|---|---|---|
| `contract.repo` | yes | Public GitHub repo owning the rule |
| `contract.path` | yes | File path in that repo |
| `contract.sha` | yes | **Full** commit SHA of the contract text/schema/validator |
| `contract.lines` or `contract.symbol` | yes | Line range and/or validator symbol |
| `contract.kind` | yes | `spec` \| `schema` \| `validator` \| `cli_help` |

Contract must be a **public** object at `contract.sha`. Private/untracked paths → **quarantine** (`private_contract`).

### 2.3 Deployed parity

| Input | Required | Notes |
|---|---|---|
| `deployed.repo` | yes | Usually same as contract.repo |
| `deployed.sha` | yes if known | Running code identity |
| `deployed.parity` | yes | `Match` \| `Partial` \| `Unknown` (inventory vocabulary) |
| `deployed.evidence` | yes | How parity was established (Observed command/service) |

**Rule:** `model_gap` requires `deployed.parity == Match` (or an explicit stronger proof that running bytes implement `contract.sha`).  
`Partial` or `Unknown` → **cannot** assert “implementation matches contract” → either **`code_defect`** (if production claims the contract while mismatched) or **`quarantine`** (if parity simply unproven). Default: **`quarantine`** (`parity_not_match`) unless a reviewer proves a concrete contract violation in production → **`code_defect`**.

### 2.4 Reviewer receipt

| Input | Required | Notes |
|---|---|---|
| `reviewer.session` | yes | e.g. `tutor-codex`, `conductor-grok` |
| `reviewer.receipt_id` | yes | UUID |
| `reviewer.recorded_at` | yes | RFC3339 |
| `reviewer.method` | yes | `mechanical_checklist` \| `mechanical_plus_human` |

---

## 3. Decision predicates (mechanical)

Define boolean checks. Each must be recorded as Observed/Inferred/Unknown on the verdict.

### P1 — `contract_resolved`

- `git cat-file -e {contract.sha}:{contract.path}` succeeds (or equivalent public API).  
- Line/symbol exists at that SHA.

False → **quarantine** (`contract_unresolved`).

### P2 — `implementation_matches_contract`

True only if **all** hold:

1. `deployed.parity == Match` for the dependency that implements the contract surface.  
2. Deployed SHA is recorded and is an ancestor/equal of the claimed implementation pin **or** byte-equivalent proof is attached.  
3. No Observed counterexample that production violates the contract (e.g. fuzzy match in driver while contract bans fuzzy; readiness gate bypassed).

False with Observed production violation → **`code_defect`**.  
Unknown → **quarantine**.

### P3 — `taey_violated_contract`

True only if **all** hold:

1. P2 is true.  
2. Trace shows Taey (not supervisor-scripted argv) selected or performed a step.  
3. That step **contradicts** the bound contract (wrong tool, missing validate, retry-on-fail, silent proceed, etc.).  
4. Contradiction is citeable to event sequence numbers + contract lines.

False when supervisor scripted the tools → **quarantine** or separate **process_defect** (treat as **quarantine** under this contract: not `model_gap`).  
Unknown → **quarantine**.


### P3b — Mechanical bindings (v2; not self-attested booleans)

**`taey_violated_contract=true` is insufficient alone.** The verdict must also bind:

| Binding | Requirement |
|---|---|
| `trace.trace_hash` | 64-hex SHA-256 of the capture chain |
| `trace.actor` | one of `taey`, `taey-seat`, `ep3`, `taey-presence` |
| `trace.contradiction_event_indices` | nonempty list of 1-based ints citing Taey-authored events that contradict the contract |
| `contract.lines` and/or `contract.symbol` | public contract locus |

Forged booleans, empty/missing indices, non-Taey actor, or missing line/symbol → **REJECT** (verifier), not `model_gap`.

**`deployed.parity=Match` is insufficient alone.** Match requires:

| Binding | Requirement |
|---|---|
| `deployed.sha` | full 40-hex deployed commit |
| `deployed.parity_receipt.content_sha256` | 64-hex hash of independent parity evidence body |
| `deployed.parity_receipt.producer` | **≠** `reviewer.session` (no self-review) |
| `parity_receipt.body` or validated `path` | body must cite `deployed.sha`; path hash must match |

Stale/wrong hash, Partial, Unknown, missing receipt, or self-review → **REJECT** for Match / `model_gap`.

### Verdict table

| P1 | P2 | P3 | Verdict |
|---|---|---|---|
| F | * | * | `quarantine` |
| T | Unknown | * | `quarantine` |
| T | F (Observed violation) | * | `code_defect` |
| T | F (unproven only) | * | `quarantine` |
| T | T | T | `model_gap` |
| T | T | F | `quarantine` (no model violation / mis-tagged success) |
| T | T | Unknown | `quarantine` |

---

## 4. Verdict schema (`sft_failure_triage_verdict.v2`)

```json
{
  "schema": "sft_failure_triage_verdict.v2",
  "verdict_id": "<uuid>",
  "protocol_pin": {
    "repo": "palios-taey/palios-training",
    "sha": "58b108042e66fa508765a6277c033cc5a8f86abd",
    "path": "careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md",
    "lines": "46-52"
  },
  "verdict": "model_gap|code_defect|quarantine",
  "lane": "ui|orchestration|git|public_repo|string",
  "trace": {
    "trace_id": "<string>",
    "trace_hash": "<64-hex>",
    "actor": "taey|taey-seat|ep3|taey-presence|supervisor|unknown",
    "contradiction_event_indices": [1, 3],
    "event_count": 0
  },
  "contract": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex>",
    "path": "<path>",
    "lines": "<start-end>",
    "symbol": "<optional symbol name>",
    "kind": "spec|schema|validator|cli_help"
  },
  "deployed": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex required when parity=Match>",
    "parity": "Match|Partial|Unknown",
    "evidence": "<Observed note>",
    "parity_receipt": {
      "producer": "<session distinct from reviewer>",
      "content_sha256": "<64-hex>",
      "body": "<must cite deployed.sha when Match>"
    }
  },
  "predicates": {
    "contract_resolved": {"value": true, "register": "Observed"},
    "implementation_matches_contract": {"value": true, "register": "Observed"},
    "taey_violated_contract": {"value": true, "register": "Observed"}
  },
  "rationale": "<one paragraph; cite contract lines and trace events>",
  "allowed_next": [],
  "forbidden_next": [],
  "reviewer": {
    "session": "<session>",
    "receipt_id": "<uuid>",
    "recorded_at": "<RFC3339>",
    "method": "mechanical_checklist"
  },
  "verdict_sha256": "<sha256 of canonical JSON without this field>"
}
```

### 4.1 `allowed_next` / `forbidden_next` by verdict

**`model_gap`**

- **Allowed:** targeted Taey self-report; optional neutral non-Claude Chat curriculum diagnosis; capture right-way via design_rule source map or fresh supervised success; later admission only of **right-way** material.  
- **Forbidden:** training on the failure trajectory as a target; inventing tool results; “fix by prompt forever without train/probe eviction” as resting place (doctrine) without re-triage.

**`code_defect`**

- **Allowed:** file defect; upstream fix PR; production observation after fix; re-run capture; re-triage from scratch.  
- **Forbidden:** **any** SFT admission/pair generation/sanction/train fire for this failure class until re-triage yields non-`code_defect`; synthetic trajectories that paper over the bug.

**`quarantine`**

- **Allowed:** request more evidence; improve capture; resolve parity; hold in quarantine ledger.  
- **Forbidden:** infer `model_gap` or `code_defect` from incomplete data; admit pairs; train.

---

## 5. Batch / ledger rules

1. **Every** captured failure in a diagnosis or admission pipeline must have a `verdict_id` linked by `trace_hash`.  
2. Admission ledgers **exclude** rows whose linked failure triage is `code_defect` or `quarantine` (unless a separate governed residue format explicitly allows curriculum-only storage — still **not** a training target).  
3. `model_gap` does **not** auto-admit a pair; it only unlocks diagnosis + right-way capture/design_rule mining.  
4. Re-entry from quarantine requires a **new** verdict_sha256 (new review), not editing the old verdict in place (append-only).  
5. First-error-stop: if triage cannot complete because capture is incomplete → quarantine and stop pair work for that trace (do not invent events).

---

## 6. Worked classification sketches (non-exhaustive)

| Scenario | Verdict |
|---|---|
| Hands deployed Match@SHA; contract bans fuzzy; Taey used fuzzy/guess | `model_gap` |
| Driver still has name_contains while contract forbids; Taey fails | `code_defect` |
| Notify readiness requires three checks; production skips check 2 | `code_defect` |
| Trace missing result bytes | `quarantine` |
| Parity Unknown for training deploy SHA | `quarantine` (cannot claim Match) |
| Supervisor scripted tool sequence; Taey did not choose | `quarantine` (not model_gap) |
| Contract path only in private `/tmp` | `quarantine` (`private_contract`) |

---

## 7. Relationship to other contracts

| Contract | Relation |
|---|---|
| SFT standards / loop @ `58b1080` | **Normative parent** for this triage |
| Dual admission design_rule / production_trace | Downstream of triage; failures not default targets |
| Non-UI supervised capture design @ `3759c6a` | Supplies trace_hash / event completeness |
| Public dependency inventory @ `fa1baf0` | Supplies parity vocabulary Match/Partial/Unknown |
| training-defect-triage skill (three levers) | Operator workflow complement; **does not** replace this mechanical gate before admission |

---

## 8. Executable verifier

```bash
python3 careers-qwen/failure_triage_verify.py --verdict path/to/verdict.json
python3 careers-qwen/failure_triage_verify.py --verdict path/to/verdict.json --repo-root .
# exit 0 = mechanically valid binding; exit 1 = reject
```

The verifier checks schema, required bindings, predicate/verdict table consistency, protocol pin, and (when `--repo-root` is set) that `contract.sha:contract.path` resolves as a Git object. It does **not** re-judge production parity from live deploys — parity must already be recorded as Observed on the verdict.

---

## 9. Verify

```bash
# protocol pin
git show 58b108042e66fa508765a6277c033cc5a8f86abd:careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md | sed -n '46,52p'

# this contract + verifier at the published commit
git show HEAD:careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md | head -20
python3 careers-qwen/failure_triage_verify.py --self-check --repo-root .
python3 careers-qwen/failure_triage_verify.py --probe-suite --repo-root .
```
