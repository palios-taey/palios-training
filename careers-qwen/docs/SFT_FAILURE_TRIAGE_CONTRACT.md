# Failure triage contract (mechanical)

| Field | Value |
|---|---|
| **Contract id** | `sft_failure_triage.v5` |
| **Status** | **PUBLIC** mechanical classification contract (not a training launch) |
| **Task** | `taey-training-program::p0-failure-triage-contract` |
| **Author** | conductor-grok |
| **Protocol pin** | `palios-taey/palios-training@58b108042e66fa508765a6277c033cc5a8f86abd` |
| **Authority text** | `careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md` **L46–52** |
| **Executable verifier** | `careers-qwen/failure_triage_verify.py` |
| **Note** | `/tmp` drafts are not authoritative. |

## 0. Purpose

Exactly one mechanical triage verdict before diagnosis/admission/train:

| Verdict | When | Downstream |
|---|---|---|
| **`model_gap`** | Authenticated lane scorer signs implementation Match **and** Taey violation | Right-way capture path only |
| **`code_defect`** | Observed production violates public contract | Full stop; fix; re-observe |
| **`quarantine`** | Ambiguous / no authenticated scorer / incomplete | Hold; no admission |

**Honest default:** until production lane scorers are pinned with real keys, **`model_gap` is unavailable**. That is preferable to admitting forged training evidence.

---

## 1. Hard ordering

```
capture complete?
  no  → quarantine
  yes → authenticated lane scorer present?
           no  → cannot model_gap (quarantine if claiming Match+violation)
           yes → scorer outcome
                    model_gap | code_defect | quarantine
```

---

## 2. model_gap authority (v5 — non-negotiable)

`model_gap` requires **`scorer_receipt`** (`sft_lane_scorer_receipt.v1`) that:

1. Names a `scorer_id` in the verifier **pinned public-key allowlist**
2. Binds `scorer_commit`, `seat_commit`, `engine`, `root`, `lane`, `exercise_hash`, `trace_hash`
3. Binds `contract.repo/sha/path/symbol` matching the verdict contract
4. States `implementation_matches_contract=true`, `taey_violated_contract=true`, `outcome=model_gap`
5. Carries an **ed25519 signature** over the canonical signed payload, verified against the pinned pubkey

**Not authority (all REJECT for model_gap):**

- Unequal caller-supplied producer / live / capture / reviewer **label strings**
- Self-hashed multi-receipt bundles from one submitter
- Contract-document `blob_byte_equivalence` as implementation Match
- Substring / quote matching of forbidden phrases without scorer outcome
- Inline `contract_contradiction.contradicts=true`

### Scorer allowlist (production path only)

Maintained in `failure_triage_verify.py` as `SCORER_ALLOWLIST`.

- **Only** entries with `role=production` may authorize `model_gap` on the normal `--verdict` path.
- Fixture / non-production material lives in a **separate** probe-only table and is **rejected before signature acceptance**.
- An embedded fixture private key **never** confers admission authority, even if the signature verifies cryptographically.
- Until a real production scorer public key is pinned, `SCORER_ALLOWLIST` is empty and `model_gap` is unavailable.

---

## 3. Other verdicts

### `code_defect`

`implementation_matches_contract=false` with Observed production violation. No scorer required. Full stop on train.

### `quarantine`

Default when evidence incomplete, parity unknown, or no authenticated scorer for a claimed model gap.

### `implementation_matches_contract=true` without scorer

**REJECT** (even for non-model_gap). Document blob equality is rule provenance only.

---

## 4. Verdict schema (`sft_failure_triage_verdict.v5`)

```json
{
  "schema": "sft_failure_triage_verdict.v5",
  "verdict_id": "<uuid>",
  "protocol_pin": {
    "repo": "palios-taey/palios-training",
    "sha": "58b108042e66fa508765a6277c033cc5a8f86abd",
    "path": "careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md",
    "lines": "46-52"
  },
  "verdict": "model_gap|code_defect|quarantine",
  "lane": "orchestration",
  "trace": {
    "trace_id": "<string>",
    "trace_hash": "<64-hex>",
    "actor": "taey|taey-seat|ep3|taey-presence|supervisor|unknown"
  },
  "contract": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex>",
    "path": "<path>",
    "lines": "<range>",
    "symbol": "FailureTriageGate.training_gap",
    "kind": "spec"
  },
  "deployed": {
    "repo": "palios-taey/<repo>",
    "sha": "<40-hex when Match>",
    "parity": "Match|Partial|Unknown",
    "evidence": "<Observed note>"
  },
  "scorer_receipt": {
    "schema": "sft_lane_scorer_receipt.v1",
    "scorer_id": "<allowlisted id>",
    "scorer_commit": "<40-hex>",
    "seat_commit": "<40-hex>",
    "engine": "<string>",
    "root": "<string>",
    "lane": "<string>",
    "exercise_hash": "<64-hex>",
    "trace_hash": "<equals trace.trace_hash>",
    "contract_repo": "<equals contract.repo>",
    "contract_sha": "<equals contract.sha>",
    "contract_path": "<equals contract.path>",
    "contract_symbol": "<equals contract.symbol>",
    "implementation_matches_contract": true,
    "taey_violated_contract": true,
    "outcome": "model_gap",
    "observed_at": "<RFC3339>",
    "signed_payload_sha256": "<64-hex>",
    "signature": "<ed25519 64-byte hex>"
  },
  "predicates": {
    "contract_resolved": {"value": true, "register": "Observed"},
    "implementation_matches_contract": {"value": true, "register": "Observed"},
    "taey_violated_contract": {"value": true, "register": "Observed"}
  },
  "rationale": "<string>",
  "allowed_next": [],
  "forbidden_next": [],
  "reviewer": {
    "session": "<session>",
    "receipt_id": "<uuid>",
    "recorded_at": "<RFC3339>",
    "method": "mechanical_checklist"
  },
  "verdict_sha256": "<optional canonical hash>"
}
```

`scorer_receipt` is **required** for `model_gap` / `taey_violated_contract=true`.

---

## 5. Ledger rules

1. Link failures by `trace_hash` + `verdict_id`.  
2. Exclude `code_defect` / `quarantine` from training admission.  
3. `model_gap` does not auto-admit pairs.  
4. Fixture scorer must never admit production training rows.  
5. Append-only verdict identity.

---

## 6. Reject sketches (CONTROL residuals)

| Attack | Result |
|---|---|
| One caller authors all differently labelled receipts | **REJECT** (no scorer signature) |
| Taey quotes forbidden phrase without scorer outcome | **REJECT** |
| Contract blob byte-eq as impl Match | **REJECT** for model_gap path |
| Unknown scorer_id / bad signature | **REJECT** |

---

## 7. Executable verifier

```bash
python3 careers-qwen/failure_triage_verify.py --verdict path/to/verdict.json --repo-root .
python3 careers-qwen/failure_triage_verify.py --self-check --probe-suite --repo-root .
```

---

## 8. Verify

```bash
python3 careers-qwen/failure_triage_verify.py --self-check --probe-suite --repo-root .
# expect PROBE_SUITE N/N including single_submitter_multi_label_forgery REJECT
```
