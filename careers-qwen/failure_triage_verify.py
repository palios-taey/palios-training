#!/usr/bin/env python3
"""Mechanical verifier for sft_failure_triage_verdict.v1.

Public contract: careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md
Protocol pin: careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md @ 58b1080… L46–52

Exit codes:
  0 — verdict is mechanically consistent with the contract
  1 — reject (schema, binding, predicate table, or git object failure)
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
VERDICT_SCHEMA = "sft_failure_triage_verdict.v1"
VERDICTS = frozenset({"model_gap", "code_defect", "quarantine"})
PARITY = frozenset({"Match", "Partial", "Unknown"})
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


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


def expected_verdict(
    contract_resolved: bool | None,
    impl_match: bool | None,
    taey_violated: bool | None,
    *,
    observed_impl_violation: bool,
) -> str | None:
    """Return required verdict, or None if inputs cannot decide (must quarantine)."""
    if contract_resolved is not True:
        return "quarantine"
    if impl_match is None:
        return "quarantine"
    if impl_match is False:
        if observed_impl_violation:
            return "code_defect"
        return "quarantine"
    # impl_match True
    if taey_violated is True:
        return "model_gap"
    if taey_violated is False:
        return "quarantine"
    return "quarantine"


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

    contract = _as_dict(doc.get("contract"), "contract")
    _as_str(contract.get("repo"), "contract.repo")
    csha = _as_str(contract.get("sha"), "contract.sha").lower()
    _require(bool(SHA40.match(csha)), "contract.sha must be full 40-hex commit")
    cpath = _as_str(contract.get("path"), "contract.path")
    _require(not cpath.startswith("/tmp"), "contract.path must not be operator-local /tmp")
    _require("/home/" not in cpath, "contract.path must not be operator home path")
    _as_str(contract.get("lines"), "contract.lines")
    _as_str(contract.get("kind"), "contract.kind")

    deployed = _as_dict(doc.get("deployed"), "deployed")
    parity = _as_str(deployed.get("parity"), "deployed.parity")
    _require(parity in PARITY, "deployed.parity must be Match|Partial|Unknown")
    _as_str(deployed.get("evidence"), "deployed.evidence")
    dsha = deployed.get("sha")
    if dsha is not None and dsha != "":
        _require(isinstance(dsha, str) and bool(SHA40.match(dsha.lower())), "deployed.sha must be 40-hex when set")

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

    observed_impl_violation = bool(doc.get("observed_implementation_violation", False))
    if im is False and preds["implementation_matches_contract"].get("register") == "Observed":
        # Observed false match without explicit flag still treated as violation path when verdict is code_defect
        if verdict == "code_defect":
            observed_impl_violation = True

    expected = expected_verdict(cr, im, tv, observed_impl_violation=observed_impl_violation)
    _require(expected is not None, "internal: expected verdict missing")
    _require(
        verdict == expected,
        f"verdict {verdict!r} inconsistent with predicates (expected {expected!r})",
    )

    # model_gap hard requirements
    if verdict == "model_gap":
        _require(parity == "Match", "model_gap requires deployed.parity == Match")
        _require(im is True, "model_gap requires implementation_matches_contract true")
        _require(tv is True, "model_gap requires taey_violated_contract true")

    if verdict == "code_defect":
        _require(im is False, "code_defect requires implementation_matches_contract false")

    reviewer = _as_dict(doc.get("reviewer"), "reviewer")
    _as_str(reviewer.get("session"), "reviewer.session")
    rid = _as_str(reviewer.get("receipt_id"), "reviewer.receipt_id").lower()
    _require(bool(UUID_RE.match(rid)), "reviewer.receipt_id must be lowercase UUID")
    _as_str(reviewer.get("recorded_at"), "reviewer.recorded_at")
    _as_str(reviewer.get("method"), "reviewer.method")

    _as_str(doc.get("rationale"), "rationale")
    _require(isinstance(doc.get("allowed_next"), list), "allowed_next must be a list")
    _require(isinstance(doc.get("forbidden_next"), list), "forbidden_next must be a list")

    if repo_root is not None:
        verify_git_object(repo_root, csha, cpath)
        notes.append(f"git object ok {csha[:12]}:{cpath}")
        # protocol pin object when repo is this training tree
        try:
            verify_git_object(repo_root, PROTOCOL_SHA, PROTOCOL_PATH)
            notes.append("protocol pin object ok")
        except Reject:
            notes.append("protocol pin object not in this clone (ok if shallow)")

    return notes


def self_check(repo_root: Path | None) -> None:
    """Valid quarantine template must pass; invalid model_gap must fail."""
    good = {
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
            "evidence": "self-check: parity deliberately Unknown",
        },
        "predicates": {
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": None, "register": "Unknown"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
        "rationale": "self-check quarantine when parity Unknown",
        "allowed_next": ["request_parity_evidence"],
        "forbidden_next": ["pair_admission", "train_fire"],
        "reviewer": {
            "session": "self-check",
            "receipt_id": "22222222-2222-4222-8222-222222222222",
            "recorded_at": "2026-08-05T00:00:00+00:00",
            "method": "mechanical_checklist",
        },
    }
    notes = verify_verdict(good, repo_root=repo_root)
    print("self-check quarantine PASS", "; ".join(notes) if notes else "")

    bad = dict(good)
    bad["verdict"] = "model_gap"
    bad["predicates"] = {
        "contract_resolved": {"value": True, "register": "Observed"},
        "implementation_matches_contract": {"value": True, "register": "Observed"},
        "taey_violated_contract": {"value": True, "register": "Observed"},
    }
    bad["deployed"] = dict(good["deployed"])
    bad["deployed"]["parity"] = "Unknown"
    try:
        verify_verdict(bad, repo_root=None)
        raise SystemExit("self-check expected reject for model_gap with Unknown parity")
    except Reject as exc:
        print(f"self-check invalid model_gap REJECT ok: {exc}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", help="Path to verdict JSON")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Git repository root for resolving contract.sha:path (optional)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run built-in positive/negative mechanical checks",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    if args.self_check:
        try:
            self_check(repo_root)
        except Reject as exc:
            print(f"REJECT: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args.verdict:
        print("ERROR: --verdict is required unless --self-check", file=sys.stderr)
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

    # Optional integrity: if verdict_sha256 present, check canonical form without that field
    if "verdict_sha256" in doc:
        unsigned = {k: v for k, v in doc.items() if k != "verdict_sha256"}
        canonical = json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        got = str(doc["verdict_sha256"]).lower()
        if got != digest:
            print(f"REJECT: verdict_sha256 mismatch expected {digest}", file=sys.stderr)
            return 1
        notes.append("verdict_sha256 ok")

    print("PASS", doc.get("verdict"), doc.get("trace", {}).get("trace_hash", "")[:16])
    for note in notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
