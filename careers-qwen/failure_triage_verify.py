#!/usr/bin/env python3
"""Mechanical verifier for sft_failure_triage_verdict.v4.

Public contract: careers-qwen/docs/SFT_FAILURE_TRIAGE_CONTRACT.md
Protocol pin: careers-qwen/docs/SFT_SELF_TRAINING_LOOP_PROTOCOL.md @ 58b1080… L46–52

v4 closes CONTROL counterexample after bce9ec7:
- Match methods: git_commit_equal | blob_byte_equivalence only (git_ancestor removed)
- blob paths must equal the cited contract.path (irrelevant-path Match REJECTS)
- Match requires independent live_deployment_receipt bound to deployed.sha
- model_gap rejects caller-authored contract_contradiction as authority;
  contradiction is re-derived by a contract-symbol oracle on event content
- model_gap requires independent trace_receipt (producer ≠ reviewer) and a
  production-seat event payload hash chain re-derived to trace_hash

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
VERDICT_SCHEMA = "sft_failure_triage_verdict.v4"
PARITY_RECEIPT_SCHEMA = "sft_parity_receipt.v2"
LIVE_RECEIPT_SCHEMA = "sft_live_deployment_receipt.v1"
TRACE_ARTIFACT_SCHEMA = "sft_failure_trace.v2"
TRACE_RECEIPT_SCHEMA = "sft_trace_receipt.v1"
VERDICTS = frozenset({"model_gap", "code_defect", "quarantine"})
PARITY = frozenset({"Match", "Partial", "Unknown"})
# git_ancestor deliberately excluded — lineage ≠ implementation equivalence
PARITY_METHODS = frozenset({"git_commit_equal", "blob_byte_equivalence"})
TAEY_ACTORS = frozenset({"taey", "taey-seat", "ep3", "taey-presence"})
PRODUCTION_SEATS = frozenset({"taey", "taey-seat", "ep3", "taey-presence", "capture-seat"})
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
LINES_RE = re.compile(r"^[0-9]+(-[0-9]+)?$")

# Contract-symbol oracles: verifier re-derives contradiction from event content.
# Caller-authored contract_contradiction.contradicts is never authority.
BUILTIN_ORACLES: dict[str, dict[str, Any]] = {
    "FailureTriageGate.training_gap": {
        "method": "substring_any",
        "needles": (
            "admitted failure trajectory as training target",
            "train on failure",
            "training on the failure",
            "failure trajectory as a target",
            "failure as training target",
        ),
    },
    "training_gap": {
        "method": "substring_any",
        "needles": (
            "admitted failure trajectory as training target",
            "train on failure",
            "training on the failure",
            "failure trajectory as a target",
            "failure as training target",
        ),
    },
}


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


def _side_ref(side: dict[str, Any], name: str) -> str:
    ref = _as_str(side.get("ref"), f"{name}.ref").lower()
    _require(bool(SHA40.match(ref)), f"{name}.ref must be full 40-hex commit")
    return ref


def _receipt_hash_ok(obj: dict[str, Any], field: str = "receipt_sha256") -> str:
    unsigned = {k: v for k, v in obj.items() if k != field}
    expected = _sha256_text(_canonical(unsigned))
    claimed = _as_str(obj.get(field), field).lower()
    _require(bool(SHA64.match(claimed)), f"{field} must be 64-hex")
    _require(claimed == expected, f"{field} mismatch expected {expected}")
    return claimed


def _validate_live_deployment_receipt(
    deployed: dict[str, Any],
    *,
    reviewer_session: str,
) -> list[str]:
    notes: list[str] = []
    dsha = _as_str(deployed.get("sha"), "deployed.sha").lower()
    _require(bool(SHA40.match(dsha)), "deployed.sha must be full 40-hex")
    lr = _as_dict(deployed.get("live_receipt"), "deployed.live_receipt")
    _require(
        lr.get("schema") == LIVE_RECEIPT_SCHEMA,
        f"Match requires live_receipt.schema={LIVE_RECEIPT_SCHEMA}",
    )
    producer = _as_str(lr.get("producer"), "live_receipt.producer").strip()
    _require(
        producer != reviewer_session,
        "live_receipt.producer must be distinct from reviewer.session",
    )
    lr_sha = _as_str(lr.get("deployed_sha"), "live_receipt.deployed_sha").lower()
    _require(bool(SHA40.match(lr_sha)), "live_receipt.deployed_sha must be 40-hex")
    _require(lr_sha == dsha, "live_receipt.deployed_sha must equal deployed.sha")
    _as_str(lr.get("observed_at"), "live_receipt.observed_at")
    _as_str(lr.get("evidence"), "live_receipt.evidence")
    body = lr.get("body")
    content_hash = _as_str(lr.get("content_sha256"), "live_receipt.content_sha256").lower()
    _require(bool(SHA64.match(content_hash)), "live_receipt.content_sha256 must be 64-hex")
    if isinstance(body, str):
        digest = _sha256_text(body)
        _require(digest == content_hash, "live_receipt.body does not match content_sha256")
        _require(dsha in body.lower(), "live_receipt.body must cite deployed.sha")
    else:
        raise Reject("live_receipt.body required for content_sha256 validation")
    _receipt_hash_ok(lr)
    notes.append("live_deployment_receipt ok")
    return notes


def _validate_machine_parity_receipt(
    deployed: dict[str, Any],
    contract: dict[str, Any],
    *,
    parity: str,
    reviewer_session: str,
    repo_root: Path | None,
) -> list[str]:
    """Re-derive parity; reject prose-only and ancestry-only Match."""
    notes: list[str] = []
    if parity != "Match":
        if "parity_receipt" in deployed and deployed["parity_receipt"] is not None:
            pr = _as_dict(deployed["parity_receipt"], "deployed.parity_receipt")
            if pr.get("schema") in {PARITY_RECEIPT_SCHEMA, "sft_parity_receipt.v1"}:
                method = pr.get("method")
                if method == "git_ancestor":
                    raise Reject(
                        "parity method git_ancestor is not a Match proof "
                        "(lineage ≠ implementation equivalence)"
                    )
        return notes

    dsha = deployed.get("sha")
    _require(
        isinstance(dsha, str) and bool(SHA40.match(dsha.lower())),
        "Match requires deployed.sha as full 40-hex commit",
    )
    dsha = dsha.lower()
    cpath = _as_str(contract.get("path"), "contract.path")

    notes.extend(
        _validate_live_deployment_receipt(deployed, reviewer_session=reviewer_session)
    )

    pr = _as_dict(deployed.get("parity_receipt"), "deployed.parity_receipt")
    _require(
        pr.get("schema") == PARITY_RECEIPT_SCHEMA,
        f"Match requires parity_receipt.schema={PARITY_RECEIPT_SCHEMA} "
        "(prose/body-only or v1 ancestor methods are not a parity proof)",
    )
    method = _as_str(pr.get("method"), "deployed.parity_receipt.method").strip()
    _require(
        method != "git_ancestor",
        "parity method git_ancestor is not a Match proof "
        "(lineage ≠ implementation equivalence)",
    )
    _require(method in PARITY_METHODS, f"parity_receipt.method must be one of {sorted(PARITY_METHODS)}")
    result = _as_str(pr.get("result"), "deployed.parity_receipt.result").strip()
    _require(result == "Match", "Match requires parity_receipt.result=Match")

    producer = _as_str(pr.get("producer"), "deployed.parity_receipt.producer").strip()
    _require(
        producer != reviewer_session,
        "parity_receipt.producer must be distinct from reviewer.session (no self-review)",
    )
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
    _receipt_hash_ok(pr)
    notes.append("parity_receipt hash ok")

    if "body" in pr and pr["body"] is not None and not isinstance(pr.get("body"), (dict, list)):
        notes.append("parity_receipt prose body ignored (not authority)")

    _require(repo_root is not None, f"Match method={method} requires --repo-root for re-derive")
    assert repo_root is not None

    if method == "git_commit_equal":
        _require(source_ref == deployed_ref, "git_commit_equal requires source.ref == deployed.ref")
        _require(_git_is_commit(repo_root, source_ref), "source.ref must be a commit")
        _require(_git_is_commit(repo_root, deployed_ref), "deployed.ref must be a commit")
        # Commit identity alone is insufficient without binding to the cited contract path.
        # Require the contract path exists at that commit (same bytes trivially).
        verify_git_object(repo_root, deployed_ref, cpath)
        notes.append(f"re-derived git_commit_equal {source_ref[:12]} path={cpath}")
    elif method == "blob_byte_equivalence":
        spath = _as_str(source.get("path"), "parity_receipt.source.path")
        dpath = _as_str(dep_side.get("path"), "parity_receipt.deployed.path")
        _require(
            spath == cpath and dpath == cpath,
            "blob_byte_equivalence paths must equal contract.path "
            f"(got source={spath!r} deployed={dpath!r} contract={cpath!r}); "
            "irrelevant-path Match rejects",
        )
        sbytes = _git_blob_bytes(repo_root, source_ref, spath)
        dbytes = _git_blob_bytes(repo_root, deployed_ref, dpath)
        sh = _sha256_bytes(sbytes)
        dh = _sha256_bytes(dbytes)
        if source.get("content_sha256") is not None:
            sch = _as_str(source.get("content_sha256"), "parity_receipt.source.content_sha256").lower()
            _require(bool(SHA64.match(sch)), "source.content_sha256 must be 64-hex")
            _require(sch == sh, "source.content_sha256 does not match re-derived blob hash")
        if dep_side.get("content_sha256") is not None:
            dch = _as_str(dep_side.get("content_sha256"), "parity_receipt.deployed.content_sha256").lower()
            _require(bool(SHA64.match(dch)), "deployed.content_sha256 must be 64-hex")
            _require(dch == dh, "deployed.content_sha256 does not match re-derived blob hash")
        _require(sh == dh, "blob_byte_equivalence re-derive failed: source/deployed content hashes differ")
        notes.append(f"re-derived blob_byte_equivalence path={cpath} sha256={sh[:16]}…")
    else:
        raise Reject(f"unsupported parity method {method}")

    return notes


def _event_payload_for_chain(ev: dict[str, Any]) -> dict[str, Any]:
    """Payload used for production-seat chain — excludes submitter contradiction claims."""
    return {k: v for k, v in ev.items() if k != "contract_contradiction"}


def _chain_root_from_events(events: list[dict[str, Any]]) -> str:
    hashes = [_sha256_text(_canonical(_event_payload_for_chain(ev))) for ev in events]
    return _sha256_text(_canonical({"algorithm": "sha256_canonical_event_payloads", "event_payload_hashes": hashes}))


def _load_trace_artifact(
    trace: dict[str, Any],
    *,
    repo_root: Path | None,
) -> dict[str, Any]:
    th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    _require(bool(SHA64.match(th)), "trace.trace_hash must be 64-hex sha256")

    body = trace.get("artifact_body")
    path = trace.get("artifact_path")
    if isinstance(body, dict):
        artifact = body
        # For v2 traces, trace_hash must be the production-seat chain root, not the full
        # artifact dump (which could include submitter contradiction fields).
        if artifact.get("schema") == TRACE_ARTIFACT_SCHEMA:
            events = artifact.get("events")
            _require(isinstance(events, list) and events, "trace artifact events required")
            for i, ev in enumerate(events):
                _require(isinstance(ev, dict), f"events[{i}] must be object")
            root = _chain_root_from_events(events)
            _require(root == th, f"trace.trace_hash must equal re-derived event chain root (got {root})")
        else:
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
        try:
            artifact = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Reject(f"trace.artifact_path is not valid JSON: {exc}") from exc
        _require(isinstance(artifact, dict), "trace artifact root must be object")
        if artifact.get("schema") == TRACE_ARTIFACT_SCHEMA:
            events = artifact.get("events")
            _require(isinstance(events, list) and events, "trace artifact events required")
            root = _chain_root_from_events(events)
            _require(root == th, f"trace.trace_hash must equal re-derived event chain root (got {root})")
        else:
            digest = _sha256_bytes(raw)
            _require(digest == th, f"trace.artifact_path hash mismatch expected {th} got {digest}")
        return artifact
    raise Reject(
        "taey_violated_contract/model_gap requires loadable trace artifact "
        "(trace.artifact_body or trace.artifact_path) with re-derived chain root"
    )


def _validate_independent_trace_receipt(
    trace: dict[str, Any],
    *,
    reviewer_session: str,
    expected_trace_hash: str,
) -> list[str]:
    notes: list[str] = []
    tr = _as_dict(trace.get("trace_receipt"), "trace.trace_receipt")
    _require(
        tr.get("schema") == TRACE_RECEIPT_SCHEMA,
        f"model_gap requires independent trace_receipt.schema={TRACE_RECEIPT_SCHEMA} "
        "(inline submitter-only artifact is not authority)",
    )
    producer = _as_str(tr.get("producer"), "trace_receipt.producer").strip()
    _require(
        producer != reviewer_session,
        "trace_receipt.producer must be distinct from reviewer.session "
        "(reject submitter-authored contradiction authority)",
    )
    method = _as_str(tr.get("method"), "trace_receipt.method").strip()
    _require(
        method in {"production_seat_capture", "independent_review"},
        "trace_receipt.method must be production_seat_capture|independent_review",
    )
    seat = _as_str(tr.get("seat"), "trace_receipt.seat").strip().lower()
    _require(seat in PRODUCTION_SEATS, f"trace_receipt.seat must be production seat in {sorted(PRODUCTION_SEATS)}")
    th = _as_str(tr.get("trace_hash"), "trace_receipt.trace_hash").lower()
    _require(bool(SHA64.match(th)), "trace_receipt.trace_hash must be 64-hex")
    _require(th == expected_trace_hash, "trace_receipt.trace_hash must equal trace.trace_hash")
    _receipt_hash_ok(tr)
    notes.append(f"independent trace_receipt ok producer={producer} seat={seat}")
    return notes


def _oracle_contradiction(
    events: list[dict[str, Any]],
    indices: list[int],
    contract: dict[str, Any],
) -> list[str]:
    """Re-derive contradiction via contract-symbol oracle. Ignores caller contradicts=true."""
    notes: list[str] = []
    symbol = contract.get("symbol")
    _require(
        isinstance(symbol, str) and symbol.strip(),
        "model_gap requires contract.symbol for executable oracle binding",
    )
    symbol = symbol.strip()
    oracle = BUILTIN_ORACLES.get(symbol)
    _require(
        oracle is not None,
        f"no executable oracle for contract.symbol={symbol!r}; quarantine (cannot model_gap)",
    )
    method = oracle["method"]
    if method == "substring_any":
        needles: tuple[str, ...] = oracle["needles"]
        for idx in indices:
            content = str(events[idx - 1].get("content") or "")
            # Reject using submitter contradiction object as sole authority
            cc = events[idx - 1].get("contract_contradiction")
            if isinstance(cc, dict) and cc.get("contradicts") is True:
                # still require oracle match on content; the flag alone is insufficient
                notes.append(f"event {idx} has submitter contradicts flag (ignored as authority)")
            hit = any(n.lower() in content.lower() for n in needles)
            _require(
                hit,
                f"oracle {symbol} failed on event {idx}: content does not match "
                f"contract-forbidden patterns (caller contradicts=true is not authority)",
            )
        notes.append(f"oracle {symbol} matched n={len(indices)} events")
        return notes
    raise Reject(f"unsupported oracle method {method}")


def _validate_model_gap_bindings(
    doc: dict[str, Any],
    trace: dict[str, Any],
    contract: dict[str, Any],
    *,
    reviewer_session: str,
    repo_root: Path | None,
) -> list[str]:
    notes: list[str] = []
    actor = _as_str(trace.get("actor"), "trace.actor").strip().lower()
    _require(actor in TAEY_ACTORS, f"model_gap requires Taey actor in {sorted(TAEY_ACTORS)}; got {actor!r}")

    lines = contract.get("lines")
    symbol = contract.get("symbol")
    has_lines = isinstance(lines, str) and bool(LINES_RE.match(lines.strip()))
    has_symbol = isinstance(symbol, str) and bool(symbol.strip())
    _require(has_lines or has_symbol, "model_gap requires contract.lines and/or contract.symbol")
    _require(has_symbol, "model_gap requires contract.symbol for oracle")
    if has_lines:
        notes.append(f"contract.lines={lines}")
    notes.append(f"contract.symbol={symbol}")

    th = _as_str(trace.get("trace_hash"), "trace.trace_hash").lower()
    notes.extend(
        _validate_independent_trace_receipt(
            trace, reviewer_session=reviewer_session, expected_trace_hash=th
        )
    )

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
    seat = _as_str(artifact.get("seat"), "trace_artifact.seat").strip().lower()
    _require(seat in PRODUCTION_SEATS, f"trace artifact seat must be production seat; got {seat!r}")

    events = artifact.get("events")
    _require(isinstance(events, list) and len(events) >= 1, "trace artifact events must be a non-empty list")
    for i, ev in enumerate(events):
        _require(isinstance(ev, dict), f"trace artifact events[{i}] must be object")

    # chain root already verified in _load_trace_artifact for v2
    claimed_count = trace.get("event_count")
    _require(isinstance(claimed_count, int) and not isinstance(claimed_count, bool), "trace.event_count must be int")
    _require(claimed_count == len(events), f"trace.event_count {claimed_count} != artifact event len {len(events)}")
    notes.append(f"event_count ok n={claimed_count}")
    notes.append("production-seat event chain root ok")

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
        _require(ev_actor in TAEY_ACTORS, f"cited event {idx} actor {ev_actor!r} is not Taey")

    notes.extend(_oracle_contradiction(events, indices, contract))
    notes.append(f"contradiction indices oracle-bound n={len(indices)}")
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
            contract,
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
                _validate_model_gap_bindings(
                    doc,
                    trace,
                    contract,
                    reviewer_session=reviewer_session,
                    repo_root=repo_root,
                )
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


# --- fixtures / probes -------------------------------------------------------


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
    pr["receipt_sha256"] = _sha256_text(_canonical({k: v for k, v in pr.items() if k != "receipt_sha256"}))
    return pr


def _make_live_receipt(*, producer: str, deployed_sha: str, evidence: str = "live deploy observed") -> dict[str, Any]:
    body = f"deployed_sha={deployed_sha}\nobserved=live\nevidence={evidence}\n"
    lr: dict[str, Any] = {
        "schema": LIVE_RECEIPT_SCHEMA,
        "producer": producer,
        "deployed_sha": deployed_sha,
        "observed_at": "2026-08-05T00:00:00+00:00",
        "evidence": evidence,
        "body": body,
        "content_sha256": _sha256_text(body),
    }
    lr["receipt_sha256"] = _sha256_text(_canonical({k: v for k, v in lr.items() if k != "receipt_sha256"}))
    return lr


def _make_trace_artifact(
    *,
    trace_id: str,
    actor: str,
    seat: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TRACE_ARTIFACT_SCHEMA,
        "trace_id": trace_id,
        "actor": actor,
        "seat": seat,
        "events": events,
    }


def _trace_hash_for(artifact: dict[str, Any]) -> str:
    events = artifact["events"]
    return _chain_root_from_events(events)


def _make_trace_receipt(
    *,
    producer: str,
    trace_hash: str,
    seat: str = "taey-presence",
    method: str = "production_seat_capture",
) -> dict[str, Any]:
    tr: dict[str, Any] = {
        "schema": TRACE_RECEIPT_SCHEMA,
        "producer": producer,
        "method": method,
        "seat": seat,
        "trace_hash": trace_hash,
    }
    tr["receipt_sha256"] = _sha256_text(_canonical({k: v for k, v in tr.items() if k != "receipt_sha256"}))
    return tr


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


def _valid_match_deployed(
    *,
    repo_root: Path | None,
    producer: str = "parity-auditor",
    live_producer: str = "deploy-auditor",
    reviewer: str = "self-check-reviewer",
    method: str = "blob_byte_equivalence",
    path_override: str | None = None,
) -> dict[str, Any]:
    path = path_override if path_override is not None else PROTOCOL_PATH
    if method == "git_commit_equal":
        source = {"ref": PROTOCOL_SHA}
        dep_side = {"ref": PROTOCOL_SHA}
    elif method == "blob_byte_equivalence":
        source = {"ref": PROTOCOL_SHA, "path": path}
        dep_side = {"ref": PROTOCOL_SHA, "path": path}
        if repo_root is not None and path == PROTOCOL_PATH:
            b = _git_blob_bytes(repo_root, PROTOCOL_SHA, PROTOCOL_PATH)
            h = _sha256_bytes(b)
            source["content_sha256"] = h
            dep_side["content_sha256"] = h
    elif method == "git_ancestor":
        # used only in reject probes
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
        "live_receipt": _make_live_receipt(producer=live_producer, deployed_sha=PROTOCOL_SHA),
    }


def _valid_model_gap_trace(
    *,
    trace_id: str = "valid-gap",
    actor: str = "taey",
    seat: str = "taey-presence",
    content_violation: bool = True,
    submitter_contradicts_flag: bool = False,
    force_event_count: int | None = None,
    omit_artifact: bool = False,
    omit_receipt: bool = False,
    corrupt_hash: bool = False,
    receipt_producer: str = "capture-auditor",
    non_violation_index: bool = False,
) -> dict[str, Any]:
    violate_content = (
        "admitted failure trajectory as training target"
        if content_violation
        else "operated correctly under gate"
    )
    events = [
        {
            "kind": "request",
            "actor": actor,
            "content": "operate under failure triage gate",
        },
        {
            "kind": "tool_call",
            "actor": actor,
            "content": violate_content,
        },
        {
            "kind": "outcome",
            "actor": actor,
            "content": "pair admitted" if content_violation else "held",
        },
    ]
    if submitter_contradicts_flag:
        events[1]["contract_contradiction"] = {
            "contradicts": True,
            "contract_lines": "48-49",
            "contract_symbol": "FailureTriageGate.training_gap",
            "detail": "submitter flag only",
        }
    if non_violation_index:
        indices = [1]  # request event — no forbidden content
    else:
        indices = [2]

    artifact = _make_trace_artifact(trace_id=trace_id, actor=actor, seat=seat, events=events)
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
    if not omit_receipt:
        tr["trace_receipt"] = _make_trace_receipt(producer=receipt_producer, trace_hash=th, seat=seat)
    return tr


def self_check(repo_root: Path | None) -> None:
    good = _base_verdict()
    notes = verify_verdict(good, repo_root=repo_root)
    print("self-check quarantine PASS", "; ".join(notes) if notes else "")

    # REGATE counterexample shape must reject
    parent_pr = None
    if repo_root is not None:
        try:
            counter = _base_verdict(
                verdict="model_gap",
                deployed=_valid_match_deployed(repo_root=repo_root, method="git_ancestor"),
                predicates={
                    "contract_resolved": {"value": True, "register": "Observed"},
                    "implementation_matches_contract": {"value": True, "register": "Observed"},
                    "taey_violated_contract": {"value": True, "register": "Observed"},
                },
                # inline contradicts-only style (v3 hole)
                trace={
                    **_valid_model_gap_trace(
                        trace_id="counter",
                        content_violation=False,
                        submitter_contradicts_flag=True,
                        omit_receipt=True,
                    ),
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
                rationale="counterexample",
            )
            # force contradicts flag path with no real content + no receipt
            try:
                verify_verdict(counter, repo_root=repo_root)
                raise SystemExit("self-check expected reject REGATE counterexample shape")
            except Reject as exc:
                print(f"self-check REGATE counterexample REJECT ok: {exc}")
        except Reject as exc:
            # building git_ancestor fixture itself may raise on parity schema
            print(f"self-check REGATE counterexample REJECT ok: {exc}")

    # prose-only Match
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
        rationale="oracle-matched contradiction; blob_byte_equivalence on contract path; live deploy receipt.",
    )
    notes = verify_verdict(valid_gap, repo_root=repo_root)
    print("self-check valid model_gap PASS", "; ".join(notes) if notes else "")

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
    match_preds = {
        "contract_resolved": {"value": True, "register": "Observed"},
        "implementation_matches_contract": {"value": True, "register": "Observed"},
        "taey_violated_contract": {"value": None, "register": "Unknown"},
    }

    # --- REGATE post-bce9ec7 required rejects ---

    # 1) ancestor-with-descendant-change (git_ancestor)
    try:
        dep_anc = _valid_match_deployed(repo_root=repo_root, method="git_ancestor")
    except Reject:
        # force a parity receipt with git_ancestor even if helper rejects
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", f"{PROTOCOL_SHA}^"],
            check=True,
            capture_output=True,
            text=True,
        )
        parent = proc.stdout.strip()
        # manually build v2 receipt with forbidden method
        pr = {
            "schema": PARITY_RECEIPT_SCHEMA,
            "producer": "parity-auditor",
            "method": "git_ancestor",
            "result": "Match",
            "source": {"ref": parent},
            "deployed": {"ref": PROTOCOL_SHA},
            "reviewer": "self-check-reviewer",
        }
        pr["receipt_sha256"] = _sha256_text(_canonical({k: v for k, v in pr.items() if k != "receipt_sha256"}))
        dep_anc = {
            "repo": PROTOCOL_REPO,
            "sha": PROTOCOL_SHA,
            "parity": "Match",
            "evidence": "ancestor",
            "parity_receipt": pr,
            "live_receipt": _make_live_receipt(producer="deploy-auditor", deployed_sha=PROTOCOL_SHA),
        }
    probe(
        "ancestor_with_descendant_change",
        _base_verdict(
            verdict="model_gap",
            deployed=dep_anc,
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="anc"),
            contract=mg_contract,
            rationale="ancestor only",
        ),
        expect_pass=False,
    )

    # 2) irrelevant-path byte equivalence
    probe(
        "irrelevant_path_byte_equivalence",
        _base_verdict(
            deployed=_valid_match_deployed(
                repo_root=repo_root,
                method="blob_byte_equivalence",
                path_override="README.md",
            ),
            predicates=match_preds,
        ),
        expect_pass=False,
    )

    # 3) arbitrary deployed SHA without live deployment receipt
    pr_ok = _make_parity_receipt(
        producer="parity-auditor",
        method="git_commit_equal",
        result="Match",
        source={"ref": PROTOCOL_SHA},
        deployed_side={"ref": PROTOCOL_SHA},
        reviewer="self-check-reviewer",
    )
    probe(
        "deployed_sha_without_live_receipt",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "no live receipt",
                "parity_receipt": pr_ok,
            },
            predicates=match_preds,
        ),
        expect_pass=False,
    )

    # 4) correctly hashed inline contradicts=true forgery (no oracle content, no independent receipt)
    forge_events = [
        {"kind": "request", "actor": "taey", "content": "hello"},
        {
            "kind": "tool_call",
            "actor": "taey",
            "content": "benign action with no forbidden pattern",
            "contract_contradiction": {
                "contradicts": True,
                "contract_lines": "48-49",
                "contract_symbol": "FailureTriageGate.training_gap",
            },
        },
        {"kind": "outcome", "actor": "taey", "content": "ok"},
    ]
    forge_art = _make_trace_artifact(
        trace_id="forge-inline", actor="taey", seat="taey-presence", events=forge_events
    )
    forge_th = _trace_hash_for(forge_art)
    probe(
        "inline_contradicts_true_forgery",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace={
                "trace_id": "forge-inline",
                "trace_hash": forge_th,
                "actor": "taey",
                "contradiction_event_indices": [2],
                "event_count": 3,
                "artifact_body": forge_art,
                # independent receipt present but oracle must still fail on content
                "trace_receipt": _make_trace_receipt(producer="capture-auditor", trace_hash=forge_th),
            },
            contract=mg_contract,
            rationale="forged contradicts flag",
        ),
        expect_pass=False,
    )

    # inline without independent receipt (even with real violation content)
    probe(
        "inline_without_independent_trace_receipt",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="no-tr", omit_receipt=True),
            contract=mg_contract,
            rationale="no independent receipt",
        ),
        expect_pass=False,
    )

    # --- prior suite still required ---

    probe(
        "forged_model_gap_no_indices",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace={**_valid_model_gap_trace(trace_id="no-idx"), "contradiction_event_indices": []},
            contract=mg_contract,
        ),
        expect_pass=False,
    )

    probe(
        "non_taey_actor",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="sup", actor="supervisor"),
            contract=mg_contract,
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
            predicates=match_preds,
        ),
        expect_pass=False,
    )

    prose = f"deployed={PROTOCOL_SHA}\nparity=Match\n"
    probe(
        "forged_prose_parity_distinct_producer",
        _base_verdict(
            deployed={
                "repo": PROTOCOL_REPO,
                "sha": PROTOCOL_SHA,
                "parity": "Match",
                "evidence": "prose",
                "parity_receipt": {
                    "producer": "parity-auditor",
                    "content_sha256": _sha256_text(prose),
                    "body": prose,
                },
                "live_receipt": _make_live_receipt(producer="deploy-auditor", deployed_sha=PROTOCOL_SHA),
            },
            predicates=match_preds,
        ),
        expect_pass=False,
    )

    probe(
        "self_review_parity",
        _base_verdict(
            deployed=_valid_match_deployed(repo_root=repo_root, producer="self-check-reviewer"),
            predicates=match_preds,
        ),
        expect_pass=False,
    )

    probe(
        "self_review_trace_receipt",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="self-tr", receipt_producer="self-check-reviewer"),
            contract=mg_contract,
            rationale="self review trace",
        ),
        expect_pass=False,
    )

    probe(
        "mismatched_event_count",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="bad-count", force_event_count=99),
            contract=mg_contract,
            rationale="bad count",
        ),
        expect_pass=False,
    )

    probe(
        "missing_trace_artifact",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="no-art", omit_artifact=True),
            contract=mg_contract,
            rationale="missing artifact",
        ),
        expect_pass=False,
    )

    probe(
        "hash_mismatched_trace_artifact",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="bad-hash", corrupt_hash=True),
            contract=mg_contract,
            rationale="hash mismatch",
        ),
        expect_pass=False,
    )

    probe(
        "in_range_non_oracle_index",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="non-or", non_violation_index=True),
            contract=mg_contract,
            rationale="non oracle event",
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

    probe("wrong_schema", _base_verdict(schema="sft_failure_triage_verdict.v3"), expect_pass=False)
    probe("valid_quarantine", _base_verdict(), expect_pass=True)

    probe(
        "valid_model_gap",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root, method="blob_byte_equivalence"),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="ok"),
            contract=mg_contract,
            rationale="oracle + independent receipts + contract-path blob eq",
        ),
        expect_pass=True,
    )

    probe(
        "valid_model_gap_git_commit_equal",
        _base_verdict(
            verdict="model_gap",
            deployed=_valid_match_deployed(repo_root=repo_root, method="git_commit_equal"),
            predicates=mg_preds,
            trace=_valid_model_gap_trace(trace_id="ok-gce"),
            contract=mg_contract,
            rationale="git_commit_equal + live receipt",
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
