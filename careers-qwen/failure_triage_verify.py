#!/usr/bin/env python3
"""Mechanical verifier for sft_failure_triage_verdict.v3.

Public contract: careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md
Protocol pin: careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md @ 58b1080… L46–52

v3 hardens CONTROL residuals after d940d73:
- Match requires a machine-generated parity receipt; verifier re-derives
  ancestry or byte-equivalence from reachable artifacts (prose Match REJECTS).
- model_gap / taey_violated requires a loadable trace artifact whose bytes
  hash to trace_hash; event_count, bounds, and event-level
  contract_contradiction are checked mechanically.

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
VERDICT_SCHEMA = "sft_failure_triage_verdict.v3"
PARITY_RECEIPT_SCHEMA = "sft_parity_receipt.v1"
TRACE_ARTIFACT_SCHEMA = "sft_failure_trace.v1"
VERDICTS = frozenset({"model_gap", "code_defect", "quarantine"})
PARITY = frozenset({"Match", "Partial", "Unknown"})
PARITY_METHODS = frozenset(
    {"git_commit_equal", "git_ancestor", "blob_byte_equivalence"}
)
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _git_blob_bytes(repo_root: Path, ref: str, path: str) -> bytes:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{ref}:{path}"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise Reject(f"cannot read git blob {ref}:{path}: {detail}") from exc
    return proc.stdout


def _git_is_commit(repo_root: Path, ref: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-t", ref],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise Reject(f"git object missing for ref {ref}: {detail}") from exc
    return proc.stdout.strip() == "commit"


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    detail = (proc.stderr or proc.stdout or "").strip()
    raise Reject(f"git merge-base --is-ancestor failed: {detail}")


def _side_ref(side: dict[str, Any], name: str) -> str:
    ref = _as_str(side.get("ref"), f"{name}.ref").lower()
    _require(bool(SHA40.match(ref)), f"{name}.ref must be full 40-hex commit")
    return ref


def _validate_machine_parity_receipt(
    deployed: dict[str, Any],
    *,
    parity: str,
    reviewer_session: str,
    repo_root: Path | None,
) -> list[str]:
    """Re-derive parity from structured receipt; reject prose-only Match."""
    notes: list[str] = []
    if parity != "Match":
        if "parity_receipt" in deployed and deployed["parity_receipt"] is not None:
            pr = _as_dict(deployed["parity_receipt"], "deployed.parity_receipt")
            if pr.get("schema") == PARITY_RECEIPT_SCHEMA:
                _as_str(pr.get("method"), "deployed.parity_receipt.method")
        return notes

    dsha = deployed.get("sha")
    _require(
        isinstance(dsha, str) and bool(SHA40.match(dsha.lower())),
        "Match requires deployed.sha as full 40-hex commit",
    )
    dsha = dsha.lower()

    pr = _as_dict(deployed.get("parity_receipt"), "deployed.parity_receipt")
    _require(
        pr.get("schema") == PARITY_RECEIPT_SCHEMA,
        f"Match requires parity_receipt.schema={PARITY_RECEIPT_SCHEMA} "
        "(prose/body-only Match is not a parity proof)",
    )
    method = _as_str(pr.get("method"), "deployed.parity_receipt.method").strip()
    _require(method in PARITY_METHODS, f"parity_receipt.method must be one of {sorted(PARITY_METHODS)}")
    result = _as_str(pr.get("result"), "deployed.parity_receipt.result").strip()
    _require(result == "Match", "Match requires parity_receipt.result=Match")

    producer = _as_str(pr.get("producer"), "deployed.parity_receipt.producer").strip()
    _require(
        producer != reviewer_session,
        "parity_receipt.producer must be distinct from reviewer.session (no self-review)",
    )
    # optional explicit reviewer field must match verdict reviewer when present
    if "reviewer" in pr and pr["reviewer"] is not None:
        pr_reviewer = _as_str(pr.get("reviewer"), "deployed.parity_receipt.reviewer").strip()
        _require(
            pr_reviewer == reviewer_session,
            "parity_receipt.reviewer must equal reviewer.session when present",
        )

    source = _as_dict(pr.get("source"), "deployed.parity_receipt.source")
    dep_side = _as_dict(pr.get("deployed"), "deployed.parity_receipt.deployed")
    source_ref = _side_ref(source, "parity_receipt.source")
    deployed_ref = _side_ref(dep_side, "parity_receipt.deployed")
    _require(
        deployed_ref == dsha,
        "parity_receipt.deployed.ref must equal deployed.sha",
    )

    # receipt_sha256 binds the machine receipt (excluding itself)
    unsigned = {k: v for k, v in pr.items() if k != "receipt_sha256"}
    expected_receipt_hash = _sha256_text(_canonical(unsigned))
    claimed = _as_str(pr.get("receipt_sha256"), "deployed.parity_receipt.receipt_sha256").lower()
    _require(bool(SHA64.match(claimed)), "parity_receipt.receipt_sha256 must be 64-hex")
    _require(
        claimed == expected_receipt_hash,
        f"parity_receipt.receipt_sha256 mismatch expected {expected_receipt_hash}",
    )
    notes.append("parity_receipt hash ok")

    # Reject leftover prose-only authority: a free-form body claiming Match is ignored
    # for proof (must not be the only evidence). Structured method is mandatory above.
    if "body" in pr and pr["body"] is not None and not isinstance(pr.get("body"), (dict, list)):
        # Allowed as optional annotation only; must not substitute for method fields.
        notes.append("parity_receipt prose body ignored (not authority)")

    if method in {"git_commit_equal", "git_ancestor", "blob_byte_equivalence"}:
        _require(repo_root is not None, f"Match method={method} requires --repo-root for re-derive")

    assert repo_root is not None  # for type checkers after require

    if method == "git_commit_equal":
        _require(source_ref == deployed_ref, "git_commit_equal requires source.ref == deployed.ref")
        _require(_git_is_commit(repo_root, source_ref), "source.ref must be a commit")
        _require(_git_is_commit(repo_root, deployed_ref), "deployed.ref must be a commit")
        notes.append(f"re-derived git_commit_equal {source_ref[:12]}")
    elif method == "git_ancestor":
        _require(_git_is_commit(repo_root, source_ref), "source.ref must be a commit")
        _require(_git_is_commit(repo_root, deployed_ref), "deployed.ref must be a commit")
        _require(
            source_ref == deployed_ref or _git_is_ancestor(repo_root, source_ref, deployed_ref),
            "git_ancestor re-derive failed: source.ref is not ancestor of deployed.ref",
        )
        notes.append(f"re-derived git_ancestor {source_ref[:12]}..{deployed_ref[:12]}")
    elif method == "blob_byte_equivalence":
        spath = _as_str(source.get("path"), "parity_receipt.source.path")
        dpath = _as_str(dep_side.get("path"), "parity_receipt.deployed.path")
        sbytes = _git_blob_bytes(repo_root, source_ref, spath)
        dbytes = _git_blob_bytes(repo_root, deployed_ref, dpath)
        sh = _sha256_bytes(sbytes)
        dh = _sha256_bytes(dbytes)
        # claimed content hashes if present must match re-derived
        if source.get("content_sha256") is not None:
            sch = _as_str(source.get("content_sha256"), "parity_receipt.source.content_sha256").lower()
            _require(bool(SHA64.match(sch)), "source.content_sha256 must be 64-hex")
            _require(sch == sh, "source.content_sha256 does not match re-derived blob hash")
        if dep_side.get("content_sha256") is not None:
            dch = _as_str(dep_side.get("content_sha256"), "parity_receipt.deployed.content_sha256").lower()
            _require(bool(SHA64.match(dch)), "deployed.content_sha256 must be 64-hex")
            _require(dch == dh, "deployed.content_sha256 does not match re-derived blob hash")
        _require(sh == dh, "blob_byte_equivalence re-derive failed: source/deployed content hashes differ")
        notes.append(f"re-derived blob_byte_equivalence sha256={sh[:16]}…")
    else:
        raise Reject(f"unsupported parity method {method}")

    return notes


def _load_trace_artifact(
    trace: dict[str, Any],
    *,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Load reachable trace artifact and verify bytes hash to trace_hash."""
    th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    _require(bool(SHA64.match(th)), "trace.trace_hash must be 64-hex sha256")

    body = trace.get("artifact_body")
    path = trace.get("artifact_path")
    if isinstance(body, dict):
        artifact = body
        raw = _canonical(artifact).encode("utf-8")
        digest = _sha256_bytes(raw)
        _require(digest == th, f"trace.artifact_body hash mismatch expected {th} got {digest}")
        return artifact
    if isinstance(path, str) and path.strip():
        _require(not path.startswith("/tmp"), "trace.artifact_path must not be /tmp")
        _require("/home/" not in path, "trace.artifact_path must not be operator home path")
        candidate = Path(path)
        if not candidate.is_absolute():
            _require(repo_root is not None, "relative artifact_path requires --repo-root")
            assert repo_root is not None
            candidate = repo_root / path
        _require(candidate.is_file(), f"trace.artifact_path not found: {path}")
        raw = candidate.read_bytes()
        digest = _sha256_bytes(raw)
        _require(digest == th, f"trace.artifact_path hash mismatch expected {th} got {digest}")
        try:
            artifact = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Reject(f"trace.artifact_path is not valid JSON: {exc}") from exc
        _require(isinstance(artifact, dict), "trace artifact root must be object")
        return artifact
    raise Reject(
        "taey_violated_contract/model_gap requires loadable trace artifact "
        "(trace.artifact_body or trace.artifact_path) hashing to trace_hash"
    )


def _contract_locus_matches(event_cc: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Event contradiction must bind the same lines and/or symbol as the verdict contract."""
    clines = contract.get("lines")
    csymbol = contract.get("symbol")
    elines = event_cc.get("contract_lines")
    esymbol = event_cc.get("contract_symbol")
    line_ok = False
    sym_ok = False
    if isinstance(clines, str) and clines.strip() and isinstance(elines, str) and elines.strip():
        line_ok = clines.strip() == elines.strip()
    if isinstance(csymbol, str) and csymbol.strip() and isinstance(esymbol, str) and esymbol.strip():
        sym_ok = csymbol.strip() == esymbol.strip()
    # At least one of the loci present on the contract must be bound on the event.
    contract_has_lines = isinstance(clines, str) and bool(LINES_RE.match(str(clines).strip()))
    contract_has_symbol = isinstance(csymbol, str) and bool(str(csymbol).strip())
    if contract_has_lines and contract_has_symbol:
        return line_ok or sym_ok
    if contract_has_lines:
        return line_ok
    if contract_has_symbol:
        return sym_ok
    return False


def _validate_model_gap_bindings(
    doc: dict[str, Any],
    trace: dict[str, Any],
    contract: dict[str, Any],
    *,
    repo_root: Path | None,
) -> list[str]:
    notes: list[str] = []
    actor = _as_str(trace.get("actor"), "trace.actor").strip().lower()
    _require(actor in TAEY_ACTORS, f"model_gap requires Taey actor in {sorted(TAEY_ACTORS)}; got {actor!r}")

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

    artifact = _load_trace_artifact(trace, repo_root=repo_root)
    _require(
        artifact.get("schema") == TRACE_ARTIFACT_SCHEMA,
        f"trace artifact schema must be {TRACE_ARTIFACT_SCHEMA}",
    )
    art_tid = _as_str(artifact.get("trace_id"), "trace_artifact.trace_id")
    _require(art_tid == _as_str(trace.get("trace_id"), "trace.trace_id"), "trace_id mismatch vs artifact")
    art_actor = _as_str(artifact.get("actor"), "trace_artifact.actor").strip().lower()
    _require(art_actor == actor, "trace.actor must match artifact.actor")
    _require(art_actor in TAEY_ACTORS, "trace artifact actor must be Taey")

    events = artifact.get("events")
    _require(isinstance(events, list) and len(events) >= 1, "trace artifact events must be a non-empty list")
    for i, ev in enumerate(events):
        _require(isinstance(ev, dict), f"trace artifact events[{i}] must be object")

    claimed_count = trace.get("event_count")
    _require(isinstance(claimed_count, int) and not isinstance(claimed_count, bool), "trace.event_count must be int")
    _require(claimed_count == len(events), f"trace.event_count {claimed_count} != artifact event len {len(events)}")
    notes.append(f"event_count ok n={claimed_count}")

    indices = trace.get("contradiction_event_indices")
    _require(isinstance(indices, list), "model_gap requires trace.contradiction_event_indices list")
    _require(len(indices) >= 1, "model_gap requires nonempty contradiction_event_indices")
    seen: set[int] = set()
    for i, idx in enumerate(indices):
        _require(isinstance(idx, int) and not isinstance(idx, bool), f"contradiction_event_indices[{i}] must be int")
        _require(idx >= 1, f"contradiction_event_indices[{i}] must be >= 1 (1-based)")
        _require(idx <= len(events), f"contradiction_event_indices[{i}]={idx} out of bounds (event_count={len(events)})")
        _require(idx not in seen, "contradiction_event_indices must not contain duplicates")
        seen.add(idx)

        ev = events[idx - 1]
        ev_actor = str(ev.get("actor", actor)).strip().lower()
        _require(
            ev_actor in TAEY_ACTORS,
            f"cited event {idx} actor {ev_actor!r} is not Taey",
        )
        cc = ev.get("contract_contradiction")
        _require(
            isinstance(cc, dict),
            f"cited event {idx} lacks contract_contradiction object "
            "(in-range non-contradiction index rejected)",
        )
        _require(
            cc.get("contradicts") is True,
            f"cited event {idx} contract_contradiction.contradicts is not true",
        )
        _require(
            _contract_locus_matches(cc, contract),
            f"cited event {idx} contract locus does not bind verdict contract lines/symbol",
        )
    notes.append(f"contradiction indices bound n={len(indices)}")
    notes.append("trace_hash + artifact ok")
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

    notes.extend(
        _validate_machine_parity_receipt(
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

    if im is True:
        _require(parity == "Match", "implementation_matches_contract=true requires deployed.parity=Match")
        _require(
            isinstance(deployed.get("sha"), str) and bool(SHA40.match(str(deployed["sha"]).lower())),
            "implementation_matches_contract=true requires deployed.sha",
        )
    if parity == "Match" and im is not True:
        if im is False:
            raise Reject("deployed.parity=Match contradicts implementation_matches_contract=false")
        if im is None:
            raise Reject("deployed.parity=Match requires implementation_matches_contract=true (not Unknown)")

    if tv is True:
        try:
            notes.extend(
                _validate_model_gap_bindings(doc, trace, contract, repo_root=repo_root)
            )
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


def _make_parity_receipt(
    *,
    producer: str,
    method: str,
    result: str,
    source: dict[str, Any],
    deployed_side: dict[str, Any],
    reviewer: str | None = None,
) -> dict[str, Any]:
    pr: dict[str, Any] = {
        "schema": PARITY_RECEIPT_SCHEMA,
        "producer": producer,
        "method": method,
        "result": result,
        "source": source,
        "deployed": deployed_side,
    }
    if reviewer is not None:
        pr["reviewer"] = reviewer
    unsigned = {k: v for k, v in pr.items() if k != "receipt_sha256"}
    pr["receipt_sha256"] = _sha256_text(_canonical(unsigned))
    return pr


def _make_trace_artifact(
    *,
    trace_id: str,
    actor: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TRACE_ARTIFACT_SCHEMA,
        "trace_id": trace_id,
        "actor": actor,
        "events": events,
    }


def _trace_hash_for(artifact: dict[str, Any]) -> str:
    return _sha256_text(_canonical(artifact))


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
            # deep-merge one level for nested dicts (trace/deployed/etc.)
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


def _valid_match_deployed(
    *,
    repo_root: Path | None,
    producer: str = "parity-auditor",
    reviewer: str = "self-check-reviewer",
    method: str = "git_commit_equal",
) -> dict[str, Any]:
    if method == "git_commit_equal":
        source = {"ref": PROTOCOL_SHA}
        dep_side = {"ref": PROTOCOL_SHA}
    elif method == "git_ancestor":
        # parent of protocol pin is a proven ancestor
        if repo_root is None:
            raise Reject("git_ancestor fixture requires repo_root")
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", f"{PROTOCOL_SHA}^"],
            check=True,
            capture_output=True,
            text=True,
        )
        parent = proc.stdout.strip()
        source = {"ref": parent}
        dep_side = {"ref": PROTOCOL_SHA}
    elif method == "blob_byte_equivalence":
        source = {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH}
        dep_side = {"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH}
        if repo_root is not None:
            b = _git_blob_bytes(repo_root, PROTOCOL_SHA, PROTOCOL_PATH)
            h = _sha256_bytes(b)
            source["content_sha256"] = h
            dep_side["content_sha256"] = h
    else:
        raise Reject(f"unknown fixture method {method}")

    pr = _make_parity_receipt(
        producer=producer,
        method=method,
        result="Match",
        source=source,
        deployed_side=dep_side,
        reviewer=reviewer,
    )
    return {
        "repo": PROTOCOL_REPO,
        "sha": PROTOCOL_SHA,
        "parity": "Match",
        "evidence": f"machine parity method={method}",
        "parity_receipt": pr,
    }


def _valid_model_gap_trace(
    *,
    trace_id: str = "valid-gap",
    actor: str = "taey",
    contract_lines: str = "48-49",
    contract_symbol: str = "FailureTriageGate.training_gap",
    bad_index_as_non_contradiction: bool = False,
    force_event_count: int | None = None,
    omit_artifact: bool = False,
    corrupt_hash: bool = False,
) -> dict[str, Any]:
    events = [
        {
            "kind": "request",
            "actor": actor,
            "content": "operate under failure triage gate",
            "contract_contradiction": False,
        },
        {
            "kind": "tool_call",
            "actor": actor,
            "content": "admitted failure trajectory as training target",
            "contract_contradiction": {
                "contradicts": True,
                "contract_lines": contract_lines,
                "contract_symbol": contract_symbol,
                "detail": "trained on failure instead of right-way capture",
            },
        },
        {
            "kind": "outcome",
            "actor": actor,
            "content": "pair admitted",
            "contract_contradiction": False,
        },
    ]
    if bad_index_as_non_contradiction:
        # cite event 1 which has contradicts false
        indices = [1]
    else:
        indices = [2]

    artifact = _make_trace_artifact(trace_id=trace_id, actor=actor, events=events)
    th = _trace_hash_for(artifact)
    if corrupt_hash:
        th = "f" * 64
    tr: dict[str, Any] = {
        "trace_id": trace_id,
        "trace_hash": th,
        "actor": actor,
        "contradiction_event_indices": indices,
        "event_count": force_event_count if force_event_count is not None else len(events),
    }
    if not omit_artifact:
        tr["artifact_body"] = artifact
    return tr


def self_check(repo_root: Path | None) -> None:
    good = _base_verdict()
    notes = verify_verdict(good, repo_root=repo_root)
    print("self-check quarantine PASS", "; ".join(notes) if notes else "")

    # forged model_gap boolean without indices must reject
    forged = _base_verdict(
        verdict="model_gap",
        deployed=_valid_match_deployed(repo_root=repo_root),
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace={
            "trace_id": "forged",
            "trace_hash": "b" * 64,
            "actor": "taey",
            "contradiction_event_indices": [],
            "event_count": 3,
            "artifact_body": _make_trace_artifact(
                trace_id="forged",
                actor="taey",
                events=[{"kind": "x", "actor": "taey", "content": "y", "contract_contradiction": False}],
            ),
        },
    )
    # fix artifact hash so rejection is about indices not hash
    art = forged["trace"]["artifact_body"]
    forged["trace"]["trace_hash"] = _trace_hash_for(art)
    forged["trace"]["event_count"] = 1
    try:
        verify_verdict(forged, repo_root=repo_root)
        raise SystemExit("self-check expected reject forged model_gap without indices")
    except Reject as exc:
        print(f"self-check forged model_gap REJECT ok: {exc}")

    # Match with self-review must reject
    self_review = _base_verdict(
        deployed=_valid_match_deployed(
            repo_root=repo_root, producer="self-check-reviewer"
        ),
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
    )
    try:
        verify_verdict(self_review, repo_root=repo_root)
        raise SystemExit("self-check expected reject self-review Match")
    except Reject as exc:
        print(f"self-check self-review Match REJECT ok: {exc}")

    # prose-only Match (v2 hole) must reject
    prose_body = f"deployed={PROTOCOL_SHA}\nparity=Match\nindependent=true\n"
    prose_match = _base_verdict(
        deployed={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "prose only",
            "parity_receipt": {
                "producer": "parity-auditor",
                "content_sha256": _sha256_text(prose_body),
                "body": prose_body,
            },
        },
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": None, "register": "Unknown"},
        },
    )
    try:
        verify_verdict(prose_match, repo_root=repo_root)
        raise SystemExit("self-check expected reject prose-only Match")
    except Reject as exc:
        print(f"self-check prose-only Match REJECT ok: {exc}")

    # valid model_gap
    tr = _valid_model_gap_trace()
    valid_gap = _base_verdict(
        verdict="model_gap",
        deployed=_valid_match_deployed(repo_root=repo_root, method="blob_byte_equivalence"),
        predicates={
            "contract_resolved": {"value": True, "register": "Observed"},
            "implementation_matches_contract": {"value": True, "register": "Observed"},
            "taey_violated_contract": {"value": True, "register": "Observed"},
        },
        trace=tr,
        contract={
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "path": PROTOCOL_PATH,
            "lines": "48-49",
            "symbol": "FailureTriageGate.training_gap",
            "kind": "spec",
        },
        rationale="Taey violated correct contract at event 2; impl Match via blob_byte_equivalence.",
    )
    notes = verify_verdict(valid_gap, repo_root=repo_root)
    print("self-check valid model_gap PASS", "; ".join(notes) if notes else "")

    # code_defect
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

    # --- legacy + v3 adversarial probes ---

    probe(
        "forged_model_gap_no_indices",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                **_valid_model_gap_trace(trace_id="no-idx"),
                "contradiction_event_indices": [],
            },
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "training_gap",
                "kind": "spec",
            },
        ),
        expect_pass=False,
    )

    probe(
        "non_taey_actor",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="sup", actor="supervisor"),
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

    probe(
        "bad_event_index_zero",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace={
                **_valid_model_gap_trace(trace_id="z"),
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

    # CONTROL residual #1: forged correctly-hashed prose Match by distinct producer
    prose = f"deployed={PROTOCOL_SHA}\nparity=Match\nproducer=distinct\n"
    probe(
        "forged_prose_parity_distinct_producer",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "prose attestation",
                "parity_receipt": {
                    "producer": "parity-auditor",
                    "content_sha256": _sha256_text(prose),
                    "body": prose,
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

    # wrong receipt hash on structured receipt
    bad_pr = _make_parity_receipt(
        producer="parity-auditor",
        method="git_commit_equal",
        result="Match",
        source={"ref": PROTOCOL_SHA},
        deployed_side={"ref": PROTOCOL_SHA},
        reviewer="self-check-reviewer",
    )
    bad_pr["receipt_sha256"] = "0" * 64
    probe(
        "stale_wrong_parity_receipt_hash",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "bad receipt hash",
                "parity_receipt": bad_pr,
            },
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    probe(
        "self_review_parity",
        _base_verdict(
            deployed=_valid_match_deployed(
                repo_root=repo_root, producer="self-check-reviewer"
            ),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": None, "register": "Unknown"},
            },
        ),
        expect_pass=False,
    )

    # correct deployed SHA with wrong compared artifact (blob hashes differ)
    if repo_root is not None:
        # compare protocol path at PROTOCOL_SHA vs a different existing path at same SHA
        other_path = "README.md"
        try:
            _git_blob_bytes(repo_root, PROTOCOL_SHA, other_path)
            wrong_blob_pr = _make_parity_receipt(
                producer="parity-auditor",
                method="blob_byte_equivalence",
                result="Match",
                source={"ref": PROTOCOL_SHA, "path": PROTOCOL_PATH},
                deployed_side={"ref": PROTOCOL_SHA, "path": other_path},
                reviewer="self-check-reviewer",
            )
            probe(
                "wrong_compared_artifact",
                _base_verdict(
                    deployed={
                        "repo": PROTOCOL_REPO,
                        "sha": PROTOCOL_SHA,
                        "parity": "Match",
                        "evidence": "wrong path compared",
                        "parity_receipt": wrong_blob_pr,
                    },
                    predicates={
                        "contract_resolved": {"value": True, "register": "Observed"},
                        "implementation_matches_contract": {"value": True, "register": "Observed"},
                        "taey_violated_contract": {"value": None, "register": "Unknown"},
                    },
                ),
                expect_pass=False,
            )
        except Reject as exc:
            total += 1
            print(f"PROBE FAIL wrong_compared_artifact: fixture error {exc}", file=sys.stderr)

    # in-range event index pointing to non-contradiction
    probe(
        "in_range_non_contradiction_index",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(
                trace_id="non-cc", bad_index_as_non_contradiction=True
            ),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="forged index",
        ),
        expect_pass=False,
    )

    # mismatched event_count
    probe(
        "mismatched_event_count",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="bad-count", force_event_count=99),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="bad count",
        ),
        expect_pass=False,
    )

    # missing trace artifact
    probe(
        "missing_trace_artifact",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="no-art", omit_artifact=True),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="missing artifact",
        ),
        expect_pass=False,
    )

    # hash-mismatched trace artifact
    probe(
        "hash_mismatched_trace_artifact",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="bad-hash", corrupt_hash=True),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="hash mismatch",
        ),
        expect_pass=False,
    )

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

    probe("wrong_schema", _base_verdict(schema="sft_failure_triage_verdict.v2"), expect_pass=False)

    probe("valid_quarantine", _base_verdict(), expect_pass=True)

    probe(
        "valid_model_gap",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root, method="blob_byte_equivalence"),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="ok"),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="bound contradiction with loadable trace + machine parity",
        ),
        expect_pass=True,
    )

    # also valid via git_commit_equal
    probe(
        "valid_model_gap_git_commit_equal",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root, method="git_commit_equal"),
            predicates={
                "contract_resolved": {"value": True, "register": "Observed"},
                "implementation_matches_contract": {"value": True, "register": "Observed"},
                "taey_violated_contract": {"value": True, "register": "Observed"},
            },
            trace=_valid_model_gap_trace(trace_id="ok-gce"),
            contract={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "path": PROTOCOL_PATH,
                "lines": "48-49",
                "symbol": "FailureTriageGate.training_gap",
                "kind": "spec",
            },
            rationale="git_commit_equal parity path",
        ),
        expect_pass=True,
    )

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

    # modified verdict hash
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

    # parity receipt deployed.ref != deployed.sha
    mismatch_ref_pr = _make_parity_receipt(
        producer="parity-auditor",
        method="git_commit_equal",
        result="Match",
        source={"ref": PROTOCOL_SHA},
        deployed_side={"ref": PROTOCOL_SHA},
        reviewer="self-check-reviewer",
    )
    probe(
        "parity_deployed_ref_mismatch_sha",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": "0" * 40,  # wrong deployed.sha vs receipt
                "parity": "Match",
                "evidence": "ref mismatch",
                "parity_receipt": mismatch_ref_pr,
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
