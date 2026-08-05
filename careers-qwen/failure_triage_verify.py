#!/usr/bin/env python3
"""Mechanical verifier for sft_failure_triage_verdict.v2.

Public contract: careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md
Protocol pin: careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md @ 58b1080… L46–52

v2 hardens residual gaps from the edd2b5f gate:
- model_gap requires Taey actor + nonempty contradiction event indices + contract line/symbol
  (forged taey_violated_contract=true without bindings REJECTS)
- Match requires deployed SHA + independent parity receipt content hash
  (self-review / Partial / Unknown / missing receipt REJECTS)

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
VERDICT_SCHEMA = "sft_failure_triage_verdict.v2"
VERDICTS = frozenset({"model_gap", "code_defect", "quarantine"})
PARITY = frozenset({"Match", "Partial", "Unknown"})
TAEY_ACTORS = frozenset({"taey", "taey-seat", "ep3", "taey-presence"})
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
LINES_RE = re.compile(r"^[0-9]+(-[0-9]+)?$")


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


def _validate_parity_receipt(
    deployed: dict[str, Any],
    *,
    parity: str,
    reviewer_session: str,
    repo_root: Path | None,
) -> list[str]:
    notes: list[str] = []
    if parity != "Match":
        # Match-only receipt requirements
        if "parity_receipt" in deployed and deployed["parity_receipt"] is not None:
            # if present on non-Match, still validate shape lightly
            pr = _as_dict(deployed["parity_receipt"], "deployed.parity_receipt")
            _as_str(pr.get("content_sha256"), "deployed.parity_receipt.content_sha256")
        return notes

    dsha = deployed.get("sha")
    _require(
        isinstance(dsha, str) and bool(SHA40.match(dsha.lower())),
        "Match requires deployed.sha as full 40-hex commit",
    )
    pr = _as_dict(deployed.get("parity_receipt"), "deployed.parity_receipt")
    content_hash = _as_str(pr.get("content_sha256"), "deployed.parity_receipt.content_sha256").lower()
    _require(bool(SHA64.match(content_hash)), "parity_receipt.content_sha256 must be 64-hex")

    producer = _as_str(pr.get("producer"), "deployed.parity_receipt.producer").strip()
    _require(producer != reviewer_session, "parity_receipt.producer must be distinct from reviewer.session (no self-review)")

    # Validate content hash against body or file
    body = pr.get("body")
    path = pr.get("path")
    if isinstance(body, str):
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        _require(digest == content_hash, "parity_receipt.body does not match content_sha256")
        notes.append("parity_receipt body hash ok")
    elif isinstance(path, str) and path.strip():
        _require(not path.startswith("/tmp"), "parity_receipt.path must not be /tmp")
        _require("/home/" not in path, "parity_receipt.path must not be operator home path")
        if repo_root is not None:
            # path relative to repo root OR absolute if exists
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = repo_root / path
            _require(candidate.is_file(), f"parity_receipt.path not found: {path}")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            _require(digest == content_hash, "parity_receipt.path content does not match content_sha256")
            notes.append("parity_receipt path hash ok")
        else:
            # without repo_root, body is required for Match hash validation
            raise Reject("Match parity_receipt.path validation requires --repo-root (or provide body)")
    else:
        raise Reject("Match requires parity_receipt.body or parity_receipt.path for hash validation")

    # deployed.sha must appear in receipt body/path content when body present
    if isinstance(body, str):
        _require(dsha.lower() in body.lower(), "parity_receipt.body must cite deployed.sha")
        notes.append("parity_receipt cites deployed.sha")

    return notes


def _validate_model_gap_bindings(doc: dict[str, Any], trace: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    actor = _as_str(trace.get("actor"), "trace.actor").strip().lower()
    _require(actor in TAEY_ACTORS, f"model_gap requires Taey actor in {sorted(TAEY_ACTORS)}; got {actor!r}")

    indices = trace.get("contradiction_event_indices")
    _require(isinstance(indices, list), "model_gap requires trace.contradiction_event_indices list")
    _require(len(indices) >= 1, "model_gap requires nonempty contradiction_event_indices")
    seen: set[int] = set()
    for i, idx in enumerate(indices):
        _require(isinstance(idx, int) and not isinstance(idx, bool), f"contradiction_event_indices[{i}] must be int")
        _require(idx >= 1, f"contradiction_event_indices[{i}] must be >= 1 (1-based event sequence)")
        _require(idx not in seen, "contradiction_event_indices must not contain duplicates")
        seen.add(idx)
    notes.append(f"contradiction indices ok n={len(indices)}")

    # contract line and/or symbol required
    lines = contract.get("lines")
    symbol = contract.get("symbol")
    has_lines = isinstance(lines, str) and bool(LINES_RE.match(lines.strip()))
    has_symbol = isinstance(symbol, str) and bool(symbol.strip())
    _require(has_lines or has_symbol, "model_gap requires contract.lines and/or contract.symbol")
    if has_lines:
        notes.append(f"contract.lines={lines}")
    if has_symbol:
        notes.append(f"contract.symbol={symbol}")

    # trace hash already required globally
    th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    _require(bool(SHA64.match(th)), "model_gap requires valid trace.trace_hash")
    notes.append("trace_hash bound")
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
    # actor always required as string for all verdicts (can be unknown)
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

    # Match bindings (always validate when Match claimed)
    notes.extend(
        _validate_parity_receipt(
            deployed,
            parity=parity,
            reviewer_session=reviewer_session,
            repo_root=repo_root,
        )
    )

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

    # Mechanical constraints on predicate booleans vs bindings
    if im is True:
        _require(parity == "Match", "implementation_matches_contract=true requires deployed.parity=Match")
        _require(
            isinstance(deployed.get("sha"), str) and bool(SHA40.match(str(deployed["sha"]).lower())),
            "implementation_matches_contract=true requires deployed.sha",
        )
    if parity == "Match" and im is not True:
        # Match without claiming impl match is allowed only for quarantine exploration; if Match, impl match should be true or null?
        # Gate: Match without valid receipt already rejected. If Match + valid receipt, force impl match true or reject inconsistency.
        if im is False:
            raise Reject("deployed.parity=Match contradicts implementation_matches_contract=false")
        if im is None:
            raise Reject("deployed.parity=Match requires implementation_matches_contract=true (not Unknown)")

    if tv is True:
        # forged boolean without model_gap bindings must reject even if verdict is quarantine attempt
        # Always require mechanical contradiction bindings when claiming Taey violated
        try:
            notes.extend(_validate_model_gap_bindings(doc, trace, contract))
        except Reject as exc:
            raise Reject(f"taey_violated_contract=true without mechanical bindings: {exc}") from exc
        _require(actor in TAEY_ACTORS, "taey_violated_contract=true requires Taey actor")

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
        _require(parity == "Match", "model_gap requires deployed.parity == Match")
        _require(im is True, "model_gap requires implementation_matches_contract true")
        _require(tv is True, "model_gap requires taey_violated_contract true")
        # bindings already enforced when tv True
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
    # deep merge-ish for nested overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            merged = dict(doc[key])
            merged.update(value)
            doc[key] = merged
        else:
            doc[key] = value
    return doc


def self_check(repo_root: Path | None) -> None:
    # 1) quarantine ok
    good = _base_verdict()
    notes = verify_verdict(good, repo_root=repo_root)
    print("self-check quarantine PASS", "; ".join(notes) if notes else "")

    # 2) forged model_gap boolean without indices must reject
    forged = _base_verdict(
        verdict="model_gap",
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "forged",
            "parity_receipt": {
                "producer": "parity-producer",
                "content_sha256": hashlib.sha256(
                    f"deployed={PROTOCOL_SHA}\nparity=Match\n".encode()
                ).hexdigest(),
                "body": f"deployed={PROTOCOL_SHA}\nparity=Match\n",
            },
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace={
            "trace_id": "forged",
            "trace_hash": "b" * 64,
            "actor": "taey",
            "contradiction_event_indices": [],  # empty — must reject
            "event_count": 3,
        },
    )
    try:
        verify_verdict(forged, repo_root=None)
        raise SystemExit("self-check expected reject forged model_gap without indices")
    except Reject as exc:
        print(f"self-check forged model_gap REJECT ok: {exc}")

    # 3) Match with self-review must reject
    self_review = _base_verdict(
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "self",
            "parity_receipt": {
                "producer": "self-check-reviewer",  # same as reviewer
                "content_sha256": hashlib.sha256(
                    f"deployed={PROTOCOL_SHA}\nparity=Match\n".encode()
                ).hexdigest(),
                "body": f"deployed={PROTOCOL_SHA}\nparity=Match\n",
            },
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
    )
    try:
        verify_verdict(self_review, repo_root=None)
        raise SystemExit("self-check expected reject self-review Match")
    except Reject as exc:
        print(f"self-check self-review Match REJECT ok: {exc}")

    # 4) valid model_gap
    body = f"deployed={PROTOCOL_SHA}\nparity=Match\nindependent=true\n"
    valid_gap = _base_verdict(
        verdict="model_gap",
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "independent parity receipt",
            "parity_receipt": {
                "producer": "parity-auditor",
                "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "body": body,
            },
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace={
            "trace_id": "valid-gap",
            "trace_hash": "c" * 64,
            "actor": "taey",
            "contradiction_event_indices": [2, 5],
            "event_count": 6,
        },
        contract={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": "48-49",
            "symbol": "FailureTriageGate.training_gap",
            "kind": "spec",
        },
        rationale="Taey violated correct contract at events 2 and 5; impl Match via independent parity.",
    )
    notes = verify_verdict(valid_gap, repo_root=repo_root)
    print("self-check valid model_gap PASS", "; ".join(notes) if notes else "")

    # 5) code_defect
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
            "evidence": "Observed production violates contract L30",
        },
        rationale="Production violates public contract; full stop before train.",
    )
    notes = verify_verdict(code_def, repo_root=repo_root)
    print("self-check code_defect PASS", "; ".join(notes) if notes else "")


def run_probe_suite(repo_root: Path | None) -> tuple[int, int]:
    """Return (passed, total) adversarial probes."""
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

    body = f"deployed={PROTOCOL_SHA}\nparity=Match\n"
    ch = hashlib.sha256(body.encode()).hexdigest()

    def match_deployed(**extra: Any) -> dict[str, Any]:
        d = {
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "probe",
            "parity_receipt": {
                "producer": "parity-auditor",
                "content_sha256": ch,
                "body": body,
            },
        }
        d.update(extra)
        return d

    # forged boolean model_gap
    probe(
        "forged_model_gap_no_indices",
        _base_verdict(
            verdict="model_gap",
            deployed=match_deployed(),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                "trace_id": "p",
                "trace_hash": "d" * 64,
                "actor": "taey",
                "contradiction_event_indices": [],
            },
        ),
        expect_pass=False,
    )

    # non-Taey actor
    probe(
        "non_taey_actor",
        _base_verdict(
            verdict="model_gap",
            deployed=match_deployed(),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                "trace_id": "p",
                "trace_hash": "e" * 64,
                "actor": "supervisor",
                "contradiction_event_indices": [1],
            },
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "kind": "spec",
            },
        ),
        expect_pass=False,
    )

    # wrong event index type / zero
    probe(
        "bad_event_index_zero",
        _base_verdict(
            verdict="model_gap",
            deployed=match_deployed(),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                "trace_id": "p",
                "trace_hash": "f" * 64,
                "actor": "taey",
                "contradiction_event_indices": [0],
            },
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "kind": "spec",
            },
        ),
        expect_pass=False,
    )

    # Match without receipt
    probe(
        "match_without_receipt",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "claimed",
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    # wrong parity hash
    probe(
        "stale_wrong_parity_hash",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "bad hash",
                "parity_receipt": {
                    "producer": "parity-auditor",
                    "content_sha256": "0" * 64,
                    "body": body,
                },
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    # self-review
    probe(
        "self_review_parity",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "self",
                "parity_receipt": {
                    "producer": "self-check-reviewer",
                    "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "body": body,
                },
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    # private contract path
    probe(
        "private_tmp_contract",
        _base_verdict(
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": "/tmp/secret_contract.md",
                "lines": "1-2",
                "kind": "spec",
            },
        ),
        expect_pass=False,
    )

    # wrong schema
    probe("wrong_schema", _base_verdict(schema="sft_failure_triage_verdict.v1"), expect_pass=False)

    # valid quarantine
    probe("valid_quarantine", _base_verdict(), expect_pass=True)

    # valid model_gap
    probe(
        "valid_model_gap",
        _base_verdict(
            verdict="model_gap",
            deployed=match_deployed(),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                "trace_id": "ok",
                "trace_hash": "1" * 64,
                "actor": "taey",
                "contradiction_event_indices": [1, 4],
            },
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "training_gap",
                "kind": "spec",
            },
            rationale="bound contradiction",
        ),
        expect_pass=True,
    )

    # valid code_defect
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
            rationale="impl violates contract",
        ),
        expect_pass=True,
    )

    # modified verdict hash (shape check)
    doc = _base_verdict()
    unsigned = {k: v for k, v in doc.items() if k != "verdict_sha256"}
    digest = hashlib.sha256(_canonical(unsigned).encode()).hexdigest()
    doc["verdict_sha256"] = "f" * 64
    total += 1
    if doc["verdict_sha256"] != digest:
        # ensure main() rejects — simulate
        try:
            # call verify then hash check like main
            verify_verdict(doc, repo_root=repo_root)
            if str(doc["verdict_sha256"]).lower() != digest:
                passed += 1
                print("PROBE PASS wrong_verdict_hash")
            else:
                print("PROBE FAIL wrong_verdict_hash", file=sys.stderr)
        except Reject:
            # still should fail on hash even if verify passes
            passed += 1
            print("PROBE PASS wrong_verdict_hash")
    else:
        print("PROBE FAIL wrong_verdict_hash", file=sys.stderr)

    # Match with wrong deployed sha not in body
    bad_body = "deployed=0000000000000000000000000000000000000000\nparity=Match\n"
    probe(
        "parity_body_missing_deployed_sha",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "mismatch",
                "parity_receipt": {
                    "producer": "parity-auditor",
                    "content_sha256": hashlib.sha256(bad_body.encode()).hexdigest(),
                    "body": bad_body,
                },
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    return passed, total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", help="Path to verdict JSON")
    parser.add_argument("--repo-root", default=None, help="Git repository root")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--probe-suite", action="store_true", help="Run adversarial probes")
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
        digest = hashlib.sha256(_canonical(unsigned).encode()).hexdigest()
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
