#!/usr/bin/env python3
"""Fail-closed cross-object validation for an autonomous terminal release receipt.

JSON Schema proves each envelope's shape.  This helper proves the cross-object facts schemas cannot:
all named digests resolve to their canonical bytes, promotion was approved by The Hub, campaign/spec
bindings agree, the collector binds the normalized node map, and the lifecycle predecessor chain
ends in the promoted artifact's completed observation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_sha256(value) -> str:
    """Digest canonical JSON bytes for a receipt, manifest, or other JSON record."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def record_sha256(value) -> str:
    """Digest a binary artifact as bytes and every other record as canonical JSON."""
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return canonical_sha256(value)


def _record(records, digest, label, errors):
    value = records.get(digest)
    if value is None:
        errors.append(f"{label}: digest {digest!r} does not resolve")
        return None
    actual = record_sha256(value)
    if actual != digest:
        errors.append(f"{label}: digest {digest!r} does not match resolved bytes {actual!r}")
        return None
    return value


def _mapping(value, label, errors):
    if not isinstance(value, Mapping):
        errors.append(f"{label}: resolved record is not an object")
        return None
    return value


def _campaign(record, campaign_id, label, errors):
    body = _mapping(record, label, errors)
    if body is not None and body.get("campaign_id") != campaign_id:
        errors.append(f"{label}: campaign_id does not match terminal receipt")
    return body


def validate_release_chain(terminal, records) -> list[str]:
    """Return all terminal-chain findings; any finding is a fail-closed refusal.

    `records` maps each SHA-256 reference to the raw artifact bytes or to the decoded JSON value
    whose canonical JSON bytes earned that digest. Callers validate individual values against their
    schemas before invoking this cross-object validator.
    """
    errors = []
    if not isinstance(records, Mapping):
        return ["records: digest store is not a mapping"]
    terminal = _mapping(terminal, "terminal", errors)
    if terminal is None:
        return errors
    campaign_id = terminal.get("campaign_id")
    released_artifact = terminal.get("released_artifact_sha256")
    aliases = terminal.get("consumer_aliases")
    if aliases != ["taey", "ep3"]:
        errors.append("terminal: consumer_aliases must be exactly ['taey', 'ep3']")

    campaign = _campaign(
        _record(records, terminal.get("campaign_spec_sha256"), "campaign spec", errors),
        campaign_id,
        "campaign spec",
        errors,
    )
    if campaign is not None:
        content_sha = campaign.get("content_sha")
        if not isinstance(content_sha, list) or not content_sha:
            errors.append("campaign spec: content_sha is not a non-empty entry array")
        else:
            paths = [
                entry.get("relative_path")
                for entry in content_sha
                if isinstance(entry, Mapping)
            ]
            if len(paths) != len(content_sha) or len(set(paths)) != len(paths):
                errors.append("campaign spec: content_sha has malformed or duplicate relative paths")
    hub = _campaign(
        _record(records, terminal.get("hub_decision_receipt_sha256"), "Hub decision", errors),
        campaign_id,
        "Hub decision",
        errors,
    )
    if hub is not None:
        if hub.get("decision") != "approved":
            errors.append("Hub decision: promotion is not approved")
        if hub.get("transition") != "promote":
            errors.append("Hub decision: terminal receipt requires a promote transition")
        if hub.get("campaign_spec_sha256") != terminal.get("campaign_spec_sha256"):
            errors.append("Hub decision: campaign_spec_sha256 does not match terminal receipt")
        subject = hub.get("subject")
        if not isinstance(subject, Mapping):
            errors.append("Hub decision: subject is missing")
        else:
            if subject.get("artifact_sha256") != released_artifact:
                errors.append("Hub decision: subject artifact does not match released artifact")
            if not subject.get("rollback_artifact_sha256"):
                errors.append("Hub decision: promote receipt has no rollback artifact")
            if subject.get("consumer_aliases") != ["taey", "ep3"]:
                errors.append("Hub decision: promote aliases are not exactly ['taey', 'ep3']")

    evaluation = _campaign(
        _record(records, terminal.get("evaluation_contract_sha256"), "evaluation contract", errors),
        campaign_id,
        "evaluation contract",
        errors,
    )
    if evaluation is not None:
        if evaluation.get("transition") != "promote":
            errors.append("evaluation contract: terminal receipt requires a promote transition")
        if evaluation.get("subject_sha256") != released_artifact:
            errors.append("evaluation contract: subject does not match released artifact")

    checkpoint = _campaign(
        _record(records, terminal.get("checkpoint_manifest_sha256"), "checkpoint manifest", errors),
        campaign_id,
        "checkpoint manifest",
        errors,
    )
    if checkpoint is not None:
        node_map = checkpoint.get("node_local_fragments")
        if not isinstance(node_map, Mapping):
            errors.append("checkpoint manifest: node_local_fragments is not a normalized map")
        collector = _campaign(
            _record(
                records,
                checkpoint.get("collector_receipt_sha256"),
                "collector receipt",
                errors,
            ),
            campaign_id,
            "collector receipt",
            errors,
        )
        if collector is not None and isinstance(node_map, Mapping):
            if collector.get("node_local_fragments_sha256") != canonical_sha256(node_map):
                errors.append("collector receipt: does not bind the checkpoint node-local fragments")

    lifecycle = _campaign(
        _record(records, terminal.get("lifecycle_chain_sha256"), "lifecycle chain", errors),
        campaign_id,
        "lifecycle chain",
        errors,
    )
    if lifecycle is not None:
        observations = lifecycle.get("observations")
        if not isinstance(observations, list) or not observations:
            errors.append("lifecycle chain: has no observations")
        else:
            previous_digest = None
            last = None
            for expected_sequence, entry in enumerate(observations, 1):
                if not isinstance(entry, Mapping) or entry.get("sequence") != expected_sequence:
                    errors.append(
                        f"lifecycle chain: observation index {expected_sequence} has the wrong sequence"
                    )
                    continue
                digest = entry.get("observation_sha256")
                observation = _campaign(
                    _record(records, digest, f"lifecycle observation {expected_sequence}", errors),
                    campaign_id,
                    f"lifecycle observation {expected_sequence}",
                    errors,
                )
                if observation is None:
                    continue
                if observation.get("sequence") != expected_sequence:
                    errors.append(
                        f"lifecycle observation {expected_sequence}: sequence does not match chain"
                    )
                if expected_sequence == 1:
                    if "previous_observation_sha256" in observation:
                        errors.append("lifecycle genesis observation has a predecessor")
                elif observation.get("previous_observation_sha256") != previous_digest:
                    errors.append(
                        f"lifecycle observation {expected_sequence}: predecessor digest does not match"
                    )
                previous_digest = digest
                last = observation
            if last is not None:
                if last.get("transition") != "promote" or last.get("event") != "completed":
                    errors.append("lifecycle chain: final observation is not a completed promotion")
                if last.get("hub_decision_receipt_sha256") != terminal.get(
                    "hub_decision_receipt_sha256"
                ):
                    errors.append("lifecycle chain: final observation does not bind the promote receipt")

    _record(records, released_artifact, "released artifact", errors)
    if campaign is not None and campaign.get("campaign_id") != campaign_id:
        errors.append("campaign spec: campaign_id does not match terminal receipt")
    return errors
