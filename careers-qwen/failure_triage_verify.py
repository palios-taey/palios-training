#!/usr/bin/env python3
"""Mechanical verifier for sft_failure_triage_verdict.v5.

Public contract: careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md
Protocol pin: careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md @ 58b1080… L46–52

v5 (post-715655a CONTROL fix):
- model_gap requires a PRODUCTION scorer receipt (ed25519) from SCORER_ALLOWLIST
  entries with role=production only.
- Fixture / non-production scorers are rejected BEFORE signature acceptance on
  the normal --verdict path (embedded fixture keys never authorize admission).
- Without a production scorer pubkey pinned, model_gap is honestly unavailable.
- Reject probes include fixture-key CLI forgery, single-submitter multi-label
  forgery, and quote-without-scorer.

Exit codes:
  0 — verdict is mechanically consistent with the contract
  1 — reject
  2 — usage / IO error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL_REPO = "palios-taey/palios-training"
PROTOCOL_SHA = "58b108042e66fa508765a6277c033cc5a8f86abd"
PROTOCOL_PATH = "careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md"
PROTOCOL_LINES = "46-52"
VERDICT_SCHEMA = "sft_failure_triage_verdict.v5"
SCORER_RECEIPT_SCHEMA = "sft_lane_scorer_receipt.v1"
VERDICTS = frozenset({"model_gap", "code_defect", "quarantine"})
PARITY = frozenset({"Match", "Partial", "Unknown"})
TAEY_ACTORS = frozenset({"taey", "taey-seat", "ep3", "taey-presence"})
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
LINES_RE = re.compile(r"^[0-9]+(-[0-9]+)?$")

# PRODUCTION scorer allowlist only (role must be "production").
# Empty until a real lane scorer public key is pinned. model_gap is unavailable.
# Fixture keys MUST NOT appear here — they cannot authorize --verdict admission.
SCORER_ALLOWLIST: dict[str, dict[str, str]] = {
    # Example when a real scorer ships (do not uncomment without a real key):
    # "ui-lane-production-scorer-v1": {
    #     "pubkey_ed25519_hex": "<32-byte hex>",
    #     "role": "production",
    #     "lane": "ui",
    #     "contract_symbol": "FailureTriageGate.training_gap",
    # },
}

# Isolated fixture material for adversarial probes ONLY.
# Used to SIGN counterexample verdicts that the production path must REJECT.
# Never merged into SCORER_ALLOWLIST. Never consulted by production verify.
_FIXTURE_SCORER_MATERIAL: dict[str, dict[str, str]] = {
    "fixture-failure-triage-scorer-v1": {
        "pubkey_ed25519_hex": "7479b94cba739e6b733afdc3da0aab98d8fc3fbe50eb891414785ee587d46841",
        "privkey_ed25519_hex": (
            "b22dfe191c5fcaa93d75537fc70928183865392a3574a87e584ee047a9caac75"
        ),
        "role": "fixture_probe_only",
    },
}

PRODUCTION_SCORER_ROLE = "production"


class Reject(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise Reject(msg)


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _as_str(value: Any, name: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{name} must be a non-empty string")
    return value


def _boolish(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    return None


def _canonical(doc: dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_git_object(repo_root: Path, sha: str, path: str) -> None:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{sha}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise Reject(f"contract object missing at {sha}:{path}: {detail}") from exc


def _ed25519_verify(pubkey_hex: str, message: bytes, signature_hex: str) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise Reject("cryptography package required for scorer signature verify") from exc
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pub.verify(bytes.fromhex(signature_hex), message)
        return True
    except (InvalidSignature, ValueError):
        return False


def _ed25519_sign(privkey_hex: str, message: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(privkey_hex))
    return sk.sign(message).hex()


def _scorer_signed_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    """Fields covered by the scorer signature (excludes signature itself)."""
    keys = (
        "schema",
        "scorer_id",
        "scorer_commit",
        "seat_commit",
        "engine",
        "root",
        "lane",
        "exercise_hash",
        "trace_hash",
        "contract_repo",
        "contract_sha",
        "contract_path",
        "contract_symbol",
        "implementation_matches_contract",
        "taey_violated_contract",
        "outcome",
        "observed_at",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in receipt:
            out[k] = receipt[k]
    return out


def _validate_authenticated_scorer_receipt(
    doc: dict[str, Any],
    *,
    reviewer_session: str,
) -> list[str]:
    """model_gap authority: signed lane scorer receipt only."""
    notes: list[str] = []
    sr = _as_dict(doc.get("scorer_receipt"), "scorer_receipt")
    _require(
        sr.get("schema") == SCORER_RECEIPT_SCHEMA,
        f"model_gap requires scorer_receipt.schema={SCORER_RECEIPT_SCHEMA} "
        "(caller-authored producer labels / self-hashed receipts are not authority)",
    )
    scorer_id = _as_str(sr.get("scorer_id"), "scorer_receipt.scorer_id").strip()

    # REJECT non-production / fixture scorers BEFORE signature acceptance.
    # Known fixture material is never production authority even if signature is valid.
    if scorer_id in _FIXTURE_SCORER_MATERIAL:
        raise Reject(
            f"scorer_id {scorer_id!r} is fixture/non-production material; "
            "rejected on production --verdict path before signature acceptance "
            "(fixture keys never authorize model_gap admission)"
        )
    allow = SCORER_ALLOWLIST.get(scorer_id)
    _require(
        allow is not None,
        f"scorer_id {scorer_id!r} not in production SCORER_ALLOWLIST; "
        "quarantine — no authenticated production lane scorer (model_gap unavailable)",
    )
    role = str(allow.get("role") or "").strip()
    _require(
        role == PRODUCTION_SCORER_ROLE,
        f"non-production scorer role {role!r} rejected before signature acceptance "
        f"(require role={PRODUCTION_SCORER_ROLE!r})",
    )
    pubkey = allow["pubkey_ed25519_hex"]

    # Bindings
    scorer_commit = _as_str(sr.get("scorer_commit"), "scorer_receipt.scorer_commit").lower()
    _require(bool(SHA40.match(scorer_commit)), "scorer_commit must be 40-hex")
    seat_commit = _as_str(sr.get("seat_commit"), "scorer_receipt.seat_commit").lower()
    _require(bool(SHA40.match(seat_commit)), "seat_commit must be 40-hex")
    _as_str(sr.get("engine"), "scorer_receipt.engine")
    _as_str(sr.get("root"), "scorer_receipt.root")
    _as_str(sr.get("lane"), "scorer_receipt.lane")
    _as_str(sr.get("observed_at"), "scorer_receipt.observed_at")

    exercise_hash = _as_str(sr.get("exercise_hash"), "scorer_receipt.exercise_hash").lower()
    _require(bool(SHA64.match(exercise_hash)), "exercise_hash must be 64-hex")
    th = _as_str(sr.get("trace_hash"), "scorer_receipt.trace_hash").lower()
    _require(bool(SHA64.match(th)), "scorer_receipt.trace_hash must be 64-hex")

    trace = _as_dict(doc.get("trace"), "trace")
    verdict_th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    _require(th == verdict_th, "scorer_receipt.trace_hash must equal trace.trace_hash")

    contract = _as_dict(doc.get("contract"), "contract")
    _require(
        sr.get("contract_repo") == contract.get("repo"),
        "scorer_receipt.contract_repo must equal contract.repo",
    )
    _require(
        str(sr.get("contract_sha") or "").lower() == str(contract.get("sha") or "").lower(),
        "scorer_receipt.contract_sha must equal contract.sha",
    )
    _require(
        sr.get("contract_path") == contract.get("path"),
        "scorer_receipt.contract_path must equal contract.path",
    )
    csymbol = _as_str(contract.get("symbol"), "contract.symbol")
    _require(
        sr.get("contract_symbol") == csymbol,
        "scorer_receipt.contract_symbol must equal contract.symbol",
    )
    if allow.get("contract_symbol"):
        _require(
            csymbol == allow["contract_symbol"],
            f"scorer not authorized for symbol {csymbol!r}",
        )

    # Scorer is sole authority for model_gap predicates
    impl = sr.get("implementation_matches_contract")
    tv = sr.get("taey_violated_contract")
    outcome = _as_str(sr.get("outcome"), "scorer_receipt.outcome")
    _require(impl is True, "scorer_receipt.implementation_matches_contract must be true for model_gap")
    _require(tv is True, "scorer_receipt.taey_violated_contract must be true for model_gap")
    _require(outcome == "model_gap", "scorer_receipt.outcome must be model_gap")

    # Signature (only after production-role allowlist gate)
    sig = _as_str(sr.get("signature"), "scorer_receipt.signature").lower()
    _require(re.fullmatch(r"[0-9a-f]{128}", sig) is not None, "signature must be 64-byte ed25519 hex")
    payload = _scorer_signed_payload(sr)
    message = _canonical(payload).encode("utf-8")
    payload_hash = _sha256_text(_canonical(payload))
    claimed_hash = sr.get("signed_payload_sha256")
    if claimed_hash is not None:
        ch = _as_str(claimed_hash, "scorer_receipt.signed_payload_sha256").lower()
        _require(ch == payload_hash, "signed_payload_sha256 mismatch")
    _require(
        _ed25519_verify(pubkey, message, sig),
        "scorer_receipt signature verification FAILED "
        "(caller-supplied producer strings are not authentication)",
    )

    if "producer_label" in sr and sr["producer_label"] is not None:
        pl = _as_str(sr.get("producer_label"), "scorer_receipt.producer_label").strip()
        _require(
            pl != reviewer_session,
            "scorer_receipt.producer_label must not equal reviewer.session",
        )

    notes.append(f"authenticated production scorer_receipt ok scorer_id={scorer_id}")
    notes.append(f"scorer_commit={scorer_commit[:12]} seat_commit={seat_commit[:12]}")
    notes.append("signature verified against production pinned pubkey")
    return notes


def expected_verdict(
    contract_resolved: bool | None,
    impl_match: bool | None,
    taey_violated: bool | None,
    *,
    observed_impl_violation: bool,
) -> str:
    if contract_resolved is not True:
        return "quarantine"
    if impl_match is None:
        return "quarantine"
    if impl_match is False:
        if observed_impl_violation:
            return "code_defect"
        return "quarantine"
    if taey_violated is True:
        return "model_gap"
    if taey_violated is False:
        return "quarantine"
    return "quarantine"


def verify_verdict(doc: dict[str, Any], *, repo_root: Path | None) -> list[str]:
    notes: list[str] = []
    _require(doc.get("schema") == VERDICT_SCHEMA, f"schema must be {VERDICT_SCHEMA}")

    verdict = _as_str(doc.get("verdict"), "verdict")
    _require(verdict in VERDICTS, f"verdict must be one of {sorted(VERDICTS)}")

    protocol = _as_dict(doc.get("protocol_pin"), "protocol_pin")
    _require(protocol.get("repo") == PROTOCOL_REPO, "protocol_pin.repo mismatch")
    _require(protocol.get("sha") == PROTOCOL_SHA, "protocol_pin.sha must be 58b1080… full pin")
    _require(protocol.get("path") == PROTOCOL_PATH, "protocol_pin.path mismatch")
    _require(protocol.get("lines") == PROTOCOL_LINES, "protocol_pin.lines must be 46-52")

    trace = _as_dict(doc.get("trace"), "trace")
    _as_str(trace.get("trace_id"), "trace.trace_id")
    th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    _require(bool(SHA64.match(th)), "trace.trace_hash must be 64-hex sha256")
    actor = _as_str(trace.get("actor"), "trace.actor").strip().lower()

    contract = _as_dict(doc.get("contract"), "contract")
    _as_str(contract.get("repo"), "contract.repo")
    csha = _as_str(contract.get("sha"), "contract.sha").lower()
    _require(bool(SHA40.match(csha)), "contract.sha must be full 40-hex commit")
    cpath = _as_str(contract.get("path"), "contract.path")
    _require(not cpath.startswith("/tmp"), "contract.path must not be operator-local /tmp")
    _require("/home/" not in cpath, "contract.path must not be operator home path")
    _as_str(contract.get("kind"), "contract.kind")

    deployed = _as_dict(doc.get("deployed"), "deployed")
    parity = _as_str(deployed.get("parity"), "deployed.parity")
    _require(parity in PARITY, "deployed.parity must be Match|Partial|Unknown")
    _as_str(deployed.get("evidence"), "deployed.evidence")

    reviewer = _as_dict(doc.get("reviewer"), "reviewer")
    reviewer_session = _as_str(reviewer.get("session"), "reviewer.session").strip()
    rid = _as_str(reviewer.get("receipt_id"), "reviewer.receipt_id").lower()
    _require(bool(UUID_RE.match(rid)), "reviewer.receipt_id must be lowercase UUID")
    _as_str(reviewer.get("recorded_at"), "reviewer.recorded_at")
    _as_str(reviewer.get("method"), "reviewer.method")

    preds = _as_dict(doc.get("predicates"), "predicates")
    for key in (
        "contract_resolved",
        "implementation_matches_contract",
        "taey_violated_contract",
    ):
        entry = _as_dict(preds.get(key), f"predicates.{key}")
        _require("value" in entry, f"predicates.{key}.value required")
        _require(
            entry.get("register") in {"Observed", "Inferred", "Unknown"},
            f"predicates.{key}.register must be Observed|Inferred|Unknown",
        )

    cr = _boolish(preds["contract_resolved"]["value"])
    im = _boolish(preds["implementation_matches_contract"]["value"])
    tv = _boolish(preds["taey_violated_contract"]["value"])

    # v5: model_gap / taey_violated=true requires authenticated scorer — not
    # self-hashed multi-label receipts or substring oracles.
    if tv is True or verdict == "model_gap":
        _require(actor in TAEY_ACTORS, "model_gap requires Taey actor")
        symbol = contract.get("symbol")
        _require(
            isinstance(symbol, str) and symbol.strip(),
            "model_gap requires contract.symbol for scorer binding",
        )
        try:
            notes.extend(
                _validate_authenticated_scorer_receipt(doc, reviewer_session=reviewer_session)
            )
        except Reject as exc:
            raise Reject(
                f"model_gap/taey_violated without authenticated lane scorer: {exc}"
            ) from exc
        # Scorer is authority: predicates must match signed scorer fields
        sr = doc["scorer_receipt"]
        _require(im is True, "model_gap requires implementation_matches_contract true")
        _require(tv is True, "model_gap requires taey_violated_contract true")
        _require(
            sr.get("implementation_matches_contract") is True,
            "predicates must match scorer implementation_matches_contract",
        )
        _require(
            sr.get("taey_violated_contract") is True,
            "predicates must match scorer taey_violated_contract",
        )
        # Parity Match self-attestation alone is insufficient; scorer covers conformance.
        # Still require deployed.parity recorded as Match for ledger consistency.
        _require(parity == "Match", "model_gap requires deployed.parity == Match (ledger)")
        _require(
            isinstance(deployed.get("sha"), str) and bool(SHA40.match(str(deployed["sha"]).lower())),
            "model_gap requires deployed.sha",
        )
        # Explicitly reject relying on submitter multi-label receipts as independence
        if "parity_receipt" in deployed and deployed["parity_receipt"] is not None:
            notes.append("parity_receipt present but not model_gap authority (scorer is)")
        notes.append("model_gap authenticated scorer path ok")

    # Reject obsolete v3/v4 self-attested Match authority when used alone for im=true
    # without scorer (non-model_gap im=true is also forbidden without scorer — force
    # quarantine honesty).
    if im is True and tv is not True and verdict != "model_gap":
        raise Reject(
            "implementation_matches_contract=true without authenticated scorer is not "
            "mechanical (contract-document blob equivalence is rule provenance only); "
            "use quarantine or code_defect"
        )

    if im is True:
        _require(parity == "Match", "implementation_matches_contract=true requires deployed.parity=Match")

    if parity == "Match" and im is not True and verdict != "quarantine":
        # Match with unknown impl → quarantine only
        if im is False:
            raise Reject("deployed.parity=Match contradicts implementation_matches_contract=false")
        if im is None and verdict not in {"quarantine"}:
            raise Reject("deployed.parity=Match with unknown impl requires quarantine verdict")

    observed_impl_violation = bool(doc.get("observed_implementation_violation", False))
    if im is False and preds["implementation_matches_contract"].get("register") == "Observed":
        if verdict == "code_defect":
            observed_impl_violation = True

    expected = expected_verdict(cr, im, tv, observed_impl_violation=observed_impl_violation)
    _require(
        verdict == expected,
        f"verdict {verdict!r} inconsistent with predicates (expected {expected!r})",
    )

    if verdict == "model_gap":
        _require(im is True and tv is True, "model_gap predicate mismatch")
        notes.append("model_gap mechanical bindings ok")

    if verdict == "code_defect":
        _require(im is False, "code_defect requires implementation_matches_contract false")
        _require(observed_impl_violation, "code_defect requires Observed implementation violation")

    _as_str(doc.get("rationale"), "rationale")
    _require(isinstance(doc.get("allowed_next"), list), "allowed_next must be a list")
    _require(isinstance(doc.get("forbidden_next"), list), "forbidden_next must be a list")

    if repo_root is not None:
        verify_git_object(repo_root, csha, cpath)
        notes.append(f"git object ok {csha[:12]}:{cpath}")
        try:
            verify_git_object(repo_root, PROTOCOL_SHA, PROTOCOL_PATH)
            notes.append("protocol pin object ok")
        except Reject:
            notes.append("protocol pin object not in this clone (ok if unavailable)")

    return notes


# --- fixtures / probes -------------------------------------------------------


def _fixture_privkey(scorer_id: str = "fixture-failure-triage-scorer-v1") -> str:
    mat = _FIXTURE_SCORER_MATERIAL[scorer_id]
    return mat["privkey_ed25519_hex"]


def _make_scorer_receipt(
    *,
    trace_hash: str,
    scorer_id: str = "fixture-failure-triage-scorer-v1",
    privkey_hex: str | None = None,
    bad_signature: bool = False,
    taey_violated: bool = True,
    impl_match: bool = True,
    outcome: str = "model_gap",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a signed scorer receipt.

    Default signs with isolated fixture material for adversarial probes that
    the production path MUST reject. Does not confer production admission.
    """
    if privkey_hex is None and scorer_id in _FIXTURE_SCORER_MATERIAL:
        privkey_hex = _fixture_privkey(scorer_id)
    sr: dict[str, Any] = {
        "schema": SCORER_RECEIPT_SCHEMA,
        "scorer_id": scorer_id,
        "scorer_commit": PROTOCOL_SHA,
        "seat_commit": PROTOCOL_SHA,
        "engine": "ep3",
        "root": "root",
        "lane": "orchestration",
        "exercise_hash": "e" * 64,
        "trace_hash": trace_hash,
        "contract_repo": PROTOCOL_REPO,
        "contract_sha": PROTOCOL_SHA,
        "contract_path": PROTOCOL_PATH,
        "contract_symbol": "FailureTriageGate.training_gap",
        "implementation_matches_contract": impl_match,
        "taey_violated_contract": taey_violated,
        "outcome": outcome,
        "observed_at": "2026-08-05T00:00:00+00:00",
    }
    if extra:
        sr.update(extra)
    payload = _scorer_signed_payload(sr)
    message = _canonical(payload).encode("utf-8")
    sr["signed_payload_sha256"] = _sha256_text(_canonical(payload))
    if privkey_hex is None:
        sr["signature"] = "0" * 128
    else:
        sr["signature"] = _ed25519_sign(privkey_hex, message)
    if bad_signature:
        sig = sr["signature"]
        sr["signature"] = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    return sr


def _base_verdict(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "verdict_id": "11111111-1111-4111-8111-111111111111",
        "protocol_pin": {
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": PROTOCOL_LINES,
        },
        "verdict": "quarantine",
        "lane": "orchestration",
        "trace": {
            "trace_id": "self-check",
            "trace_hash": "a" * 64,
            "actor": "unknown",
            "contradiction_event_indices": [],
            "event_count": 0,
        },
        "contract": {
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": "46-52",
            "kind": "spec",
        },
        "deployed": {
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Unknown",
            "evidence": "self-check",
        },
        "predicates": {
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": None, "register": "Unknown"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
        "rationale": "self-check",
        "allowed_next": ["request_parity_evidence"],
        "forbidden_next": ["pair_admission", "train_fire"],
        "reviewer": {
            "session": "self-check-reviewer",
            "receipt_id": "22222222-2222-4222-8222-222222222222",
            "recorded_at": "2026-08-05T00:00:00+00:00",
            "method": "mechanical_checklist",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            merged = dict(doc[key])
            for sk, sv in value.items():
                if isinstance(sv, dict) and isinstance(merged.get(sk), dict):
                    inner = dict(merged[sk])
                    inner.update(sv)
                    merged[sk] = inner
                else:
                    merged[sk] = sv
            doc[key] = merged
        else:
            doc[key] = value
    return doc


def _v4_style_multi_label_forgery(trace_hash: str) -> dict[str, Any]:
    """REGATE single-submitter multi-label shape — must REJECT under v5."""
    # Self-hashed receipts with different producer strings (v4 hole)
    def fake_receipt(producer: str) -> dict[str, Any]:
        body = f"producer={producer}\ndeployed={PROTOCOL_SHA}\nparity=Match\n"
        return {
            "schema": "sft_parity_receipt.v2",
            "producer": producer,
            "method": "blob_byte_equivalence",
            "result": "Match",
            "source": {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH},
            "deployed": {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH},
            "receipt_sha256": _sha256_text(
                _canonical(
                    {
                        "schema": "sft_parity_receipt.v2",
                        "producer": producer,
                        "method": "blob_byte_equivalence",
                        "result": "Match",
                        "source": {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH},
                        "deployed": {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH},
                    }
                )
            ),
            "body": body,
        }

    live_body = f"deployed_sha={PROTOCOL_SHA}\n"
    live = {
        "schema": "sft_live_deployment_receipt.v1",
        "producer": "forged-live-label",
        "deployed_sha": PROTOCOL_SHA,
        "observed_at": "2026-08-05T00:00:00+00:00",
        "evidence": "forged",
        "body": live_body,
        "content_sha256": _sha256_text(live_body),
    }
    live["receipt_sha256"] = _sha256_text(_canonical({k: v for k, v in live.items() if k != "receipt_sha256"}))

    return _base_verdict(
        verdict="model_gap",
        schema=VERDICT_SCHEMA,
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "forged multi-label",
            "parity_receipt": fake_receipt("forged-parity-label"),
            "live_receipt": live,
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace={
            "trace_id": "forge",
            "trace_hash": trace_hash,
            "actor": "taey",
            "contradiction_event_indices": [2],
            "event_count": 3,
            "trace_receipt": {
                "schema": "sft_trace_receipt.v1",
                "producer": "forged-capture-label",
                "method": "production_seat_capture",
                "seat": "taey-presence",
                "trace_hash": trace_hash,
                "receipt_sha256": _sha256_text(
                    _canonical(
                        {
                            "schema": "sft_trace_receipt.v1",
                            "producer": "forged-capture-label",
                            "method": "production_seat_capture",
                            "seat": "taey-presence",
                            "trace_hash": trace_hash,
                        }
                    )
                ),
            },
            "artifact_body": {
                "schema": "sft_failure_trace.v2",
                "trace_id": "forge",
                "actor": "taey",
                "seat": "taey-presence",
                "events": [
                    {"kind": "request", "actor": "taey", "content": "x"},
                    {
                        "kind": "tool_call",
                        "actor": "taey",
                        "content": "admitted failure trajectory as training target",
                    },
                    {"kind": "outcome", "actor": "taey", "content": "y"},
                ],
            },
        },
        contract={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": "48-49",
            "symbol": "FailureTriageGate.training_gap",
            "kind": "spec",
        },
        reviewer={
            "session": "verdict-review-label",
            "receipt_id": "22222222-2222-4222-8222-222222222222",
            "recorded_at": "2026-08-05T00:00:00+00:00",
            "method": "mechanical_checklist",
        },
        rationale="single submitter multi-label forgery",
    )


def self_check(repo_root: Path | None) -> None:
    notes = verify_verdict(_base_verdict(), repo_root=repo_root)
    print("self-check quarantine PASS", "; ".join(notes) if notes else "")

    # single-submitter multi-label forgery must reject
    forge = _v4_style_multi_label_forgery("b" * 64)
    try:
        verify_verdict(forge, repo_root=repo_root)
        raise SystemExit("self-check expected reject single-submitter multi-label forgery")
    except Reject as exc:
        print(f"self-check single-submitter forgery REJECT ok: {exc}")

    # CONTROL residual after 715655a: fixture-signed model_gap must REJECT on
    # the production path (even when the signature is cryptographically valid).
    th = "c" * 64
    fixture_gap = _base_verdict(
        verdict="model_gap",
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "fixture-signed forgery",
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace={
            "trace_id": "fixture-forge",
            "trace_hash": th,
            "actor": "taey",
            "event_count": 3,
            "contradiction_event_indices": [2],
        },
        contract={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": "48-49",
            "symbol": "FailureTriageGate.training_gap",
            "kind": "spec",
        },
        scorer_receipt=_make_scorer_receipt(trace_hash=th),
        rationale="fixture scorer must not admit on production path",
    )
    try:
        verify_verdict(fixture_gap, repo_root=repo_root)
        raise SystemExit("self-check expected reject fixture-signed model_gap on production path")
    except Reject as exc:
        print(f"self-check fixture-scorer production REJECT ok: {exc}")

    # Production allowlist isolation invariants.
    for sid, entry in SCORER_ALLOWLIST.items():
        if str(entry.get("role") or "") != PRODUCTION_SCORER_ROLE:
            raise SystemExit(f"self-check: allowlist entry {sid!r} is not role=production")
    for fid in _FIXTURE_SCORER_MATERIAL:
        if fid in SCORER_ALLOWLIST:
            raise SystemExit(f"self-check: fixture scorer {fid!r} leaked into SCORER_ALLOWLIST")
    print("self-check production allowlist isolation ok")

    code_def = _base_verdict(
        verdict="code_defect",
        observed_implementation_violation=True,
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": False, "register": "Observed"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Partial",
            "evidence": "Observed production violates contract",
        },
        rationale="impl violates contract",
    )
    notes = verify_verdict(code_def, repo_root=repo_root)
    print("self-check code_defect PASS", "; ".join(notes) if notes else "")


def run_probe_suite(repo_root: Path | None) -> tuple[int, int]:
    passed = 0
    total = 0

    def probe(name: str, doc: dict[str, Any], *, expect_pass: bool) -> None:
        nonlocal passed, total
        total += 1
        try:
            verify_verdict(doc, repo_root=repo_root)
            ok = expect_pass
            err = None
        except Reject as exc:
            ok = not expect_pass
            err = str(exc)
        if ok:
            passed += 1
            print(f"PROBE PASS {name}" + (f" ({err})" if err and not expect_pass else ""))
        else:
            print(f"PROBE FAIL {name}: expected_pass={expect_pass} err={err}", file=sys.stderr)

    th = "d" * 64
    mg_contract = {
        "repo": PROTOCOL_REPO,
        "sha": PROTOCOL_SHA,
        "path": PROTOCOL_PATH,
        "lines": "48-49",
        "symbol": "FailureTriageGate.training_gap",
        "kind": "spec",
    }
    mg_preds = {
        "contract_resolved": {"value": True, "register": "Observed"},
        "implementation_matches_contract": {"value": True, "register": "Observed"},
        "taey_violated_contract": {"value": True, "register": "Observed"},
    }
    mg_deployed = {
        "repo": PROTOCOL_REPO,
        "sha": PROTOCOL_SHA,
        "parity": "Match",
        "evidence": "scorer",
    }
    mg_trace = {
        "trace_id": "p",
        "trace_hash": th,
        "actor": "taey",
        "event_count": 3,
        "contradiction_event_indices": [2],
    }

    # REGATE post-71696d6 required rejects
    probe(
        "single_submitter_multi_label_forgery",
        _v4_style_multi_label_forgery(th),
        expect_pass=False,
    )

    # quote forbidden substring without authenticated scorer action/outcome
    probe(
        "quote_forbidden_substring_without_scorer",
        _base_verdict(
            verdict="model_gap",
            deployed=mg_deployed,
            predicates=mg_preds,
            trace={
                **mg_trace,
                "artifact_body": {
                    "schema": "sft_failure_trace.v2",
                    "trace_id": "p",
                    "actor": "taey",
                    "seat": "taey-presence",
                    "events": [
                        {"kind": "request", "actor": "taey", "content": "discuss policy"},
                        {
                            "kind": "tool_call",
                            "actor": "taey",
                            "content": (
                                "I read that admitted failure trajectory as training target "
                                "is forbidden — quoting only"
                            ),
                        },
                        {"kind": "outcome", "actor": "taey", "content": "held"},
                    ],
                },
            },
            contract=mg_contract,
            rationale="quote only no scorer",
        ),
        expect_pass=False,
    )

    probe(
        "model_gap_without_scorer_receipt",
        _base_verdict(
            verdict="model_gap",
            deployed=mg_deployed,
            predicates=mg_preds,
            trace=mg_trace,
            contract=mg_contract,
            rationale="no scorer",
        ),
        expect_pass=False,
    )

    # CONTROL residual: fixture-signed model_gap via normal production path
    probe(
        "fixture_scorer_key_rejects_on_production_path",
        _base_verdict(
            verdict="model_gap",
            deployed=mg_deployed,
            predicates=mg_preds,
            trace=mg_trace,
            contract=mg_contract,
            scorer_receipt=_make_scorer_receipt(trace_hash=th),
            rationale="fixture key must not admit",
        ),
        expect_pass=False,
    )

    # every known fixture id must reject
    for fid in sorted(_FIXTURE_SCORER_MATERIAL):
        probe(
            f"fixture_key_reject_{fid}",
            _base_verdict(
                verdict="model_gap",
                deployed=mg_deployed,
                predicates=mg_preds,
                trace=mg_trace,
                contract=mg_contract,
                scorer_receipt=_make_scorer_receipt(trace_hash=th, scorer_id=fid),
                rationale=f"fixture {fid}",
            ),
            expect_pass=False,
        )

    probe(
        "unknown_scorer_id",
        _base_verdict(
            verdict="model_gap",
            deployed=mg_deployed,
            predicates=mg_preds,
            trace=mg_trace,
            contract=mg_contract,
            scorer_receipt=_make_scorer_receipt(
                trace_hash=th,
                scorer_id="not-in-allowlist",
                privkey_hex=_fixture_privkey(),
            ),
            rationale="unknown scorer",
        ),
        expect_pass=False,
    )

    probe(
        "im_true_without_scorer_not_model_gap",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "blob only",
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
            rationale="impl match without scorer",
        ),
        expect_pass=False,
    )

    probe(
        "private_tmp_contract",
        _base_verdict(
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": "/tmp/secret.md",
                "lines": "1-2",
                "kind": "spec",
            },
        ),
        expect_pass=False,
    )

    probe("wrong_schema", _base_verdict(schema="sft_failure_triage_verdict.v4"), expect_pass=False)
    probe("valid_quarantine", _base_verdict(), expect_pass=True)

    # Honest: no production scorer pinned ⇒ no valid model_gap admit probe.
    # When a production key is later pinned, add a production-signed admit probe.

    probe(
        "valid_code_defect",
        _base_verdict(
            verdict="code_defect",
            observed_implementation_violation=True,
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": False, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Partial",
                "evidence": "Observed violation",
            },
            rationale="impl violates",
        ),
        expect_pass=True,
    )

    probe(
        "non_taey_actor",
        _base_verdict(
            verdict="model_gap",
            deployed=mg_deployed,
            predicates=mg_preds,
            trace={**mg_trace, "actor": "supervisor"},
            contract=mg_contract,
            scorer_receipt=_make_scorer_receipt(trace_hash=th),
            rationale="supervisor",
        ),
        expect_pass=False,
    )

    doc = _base_verdict()
    unsigned = {k: v for k, v in doc.items() if k != "verdict_sha256"}
    digest = _sha256_text(_canonical(unsigned))
    doc["verdict_sha256"] = "f" * 64
    total += 1
    if doc["verdict_sha256"] != digest:
        try:
            verify_verdict(doc, repo_root=repo_root)
            if str(doc["verdict_sha256"]).lower() != digest:
                passed += 1
                print("PROBE PASS wrong_verdict_hash")
            else:
                print("PROBE FAIL wrong_verdict_hash", file=sys.stderr)
        except Reject:
            passed += 1
            print("PROBE PASS wrong_verdict_hash")
    else:
        print("PROBE FAIL wrong_verdict_hash", file=sys.stderr)

    return passed, total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", help="Path to verdict JSON")
    parser.add_argument("--repo-root", default=None, help="Git repository root")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--probe-suite", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    if args.self_check:
        try:
            self_check(repo_root)
        except Reject as exc:
            print(f"REJECT: {exc}", file=sys.stderr)
            return 1
        if args.probe_suite:
            passed, total = run_probe_suite(repo_root)
            print(f"PROBE_SUITE {passed}/{total}")
            return 0 if passed == total else 1
        return 0

    if args.probe_suite:
        passed, total = run_probe_suite(repo_root)
        print(f"PROBE_SUITE {passed}/{total}")
        return 0 if passed == total else 1

    if not args.verdict:
        print("ERROR: --verdict is required unless --self-check/--probe-suite", file=sys.stderr)
        return 2

    path = Path(args.verdict)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read verdict: {exc}", file=sys.stderr)
        return 2
    if not isinstance(doc, dict):
        print("REJECT: verdict root must be object", file=sys.stderr)
        return 1

    try:
        notes = verify_verdict(doc, repo_root=repo_root)
    except Reject as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1

    if "verdict_sha256" in doc:
        unsigned = {k: v for k, v in doc.items() if k != "verdict_sha256"}
        digest = _sha256_text(_canonical(unsigned))
        if str(doc["verdict_sha256"]).lower() != digest:
            print(f"REJECT: verdict_sha256 mismatch expected {digest}", file=sys.stderr)
            return 1
        notes.append("verdict_sha256 ok")

    print("PASS", doc.get("verdict"), str(doc.get("trace", {}).get("trace_hash", ""))[:16])
    for note in notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
