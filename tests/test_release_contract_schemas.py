#!/usr/bin/env python3
"""Stdlib-only conformance tests for the additive autonomous-release schemas.

The repository intentionally has no JSON Schema runtime dependency.  This small validator covers
the keywords used by these schemas and exercises both accepted instances and fail-closed rejects.
It is not a general JSON Schema implementation.
"""

import copy
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.validate_release_chain import canonical_sha256, validate_release_chain


SCHEMA_DIR = ROOT / "schemas" / "release"
FIXTURE_DIR = SCHEMA_DIR / "fixtures"
SCHEMA_PATHS = {
    "campaign": "campaign-spec.schema.json",
    "hub": "hub-decision-receipt.schema.json",
    "evaluation": "evaluation-contract.schema.json",
    "checkpoint": "checkpoint-manifest.schema.json",
    "lifecycle": "lifecycle-observation.schema.json",
    "lifecycle_chain": "lifecycle-chain.schema.json",
    "terminal": "terminal-release-receipt.schema.json",
}
SHA256 = "a" * 64
COMMIT = "b" * 40
TIMESTAMP = "2026-08-19T13:54:00Z"
TRANSITIONS = ["train", "bake", "promote"]
FORBIDDEN_APPROVAL_FIELDS = {
    "approved_by",
    "authorized_by",
    "human_approval",
    "user_approval",
}


def load_schemas():
    return {
        name: json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        for name, filename in SCHEMA_PATHS.items()
    }


def validate(instance, schema, path="$"):
    """Return errors for the small Draft 2020-12 subset used in this directory."""
    errors = []

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")
    if "not" in schema and not validate(instance, schema["not"], path):
        errors.append(f"{path}: must not match the prohibited schema")
    for child in schema.get("allOf", []):
        errors.extend(validate(instance, child, path))
    if "if" in schema:
        condition_errors = validate(instance, schema["if"], path)
        if not condition_errors and "then" in schema:
            errors.extend(validate(instance, schema["then"], path))
        if condition_errors and "else" in schema:
            errors.extend(validate(instance, schema["else"], path))

    expected_type = schema.get("type")
    type_map = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
    }
    if expected_type:
        if expected_type not in type_map:
            raise AssertionError(f"test validator does not implement type {expected_type!r}")
        if not type_map[expected_type](instance):
            return errors + [f"{path}: expected {expected_type}"]

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                errors.extend(validate(value, additional, f"{path}.{key}"))
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: requires at least {schema['minProperties']} properties")
        if "propertyNames" in schema:
            for key in instance:
                errors.extend(validate(key, schema["propertyNames"], f"{path}.<property-name>"))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: requires at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        for index, item_schema in enumerate(schema.get("prefixItems", [])):
            if index < len(instance):
                errors.extend(validate(instance[index], item_schema, f"{path}[{index}]"))
        if "items" in schema:
            if schema["items"] is False and len(instance) > len(schema.get("prefixItems", [])):
                errors.append(f"{path}: does not allow items after prefixItems")
            elif isinstance(schema["items"], dict):
                start = len(schema.get("prefixItems", []))
                for index, value in enumerate(instance[start:], start):
                    errors.extend(validate(value, schema["items"], f"{path}[{index}]"))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: requires at least {schema['minLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: does not match {schema['pattern']!r}")

    if isinstance(instance, int) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: is below minimum {schema['minimum']}")

    return errors


def property_names(schema):
    """Collect property names recursively without treating descriptive text as a field."""
    names = set(schema.get("properties", {}))
    for child in schema.get("properties", {}).values():
        names.update(property_names(child))
    for child in schema.get("allOf", []):
        names.update(property_names(child))
    for key in ("if", "then", "else"):
        if isinstance(schema.get(key), dict):
            names.update(property_names(schema[key]))
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        names.update(property_names(additional))
    items = schema.get("items")
    if isinstance(items, dict):
        names.update(property_names(items))
    return names


def valid_instances():
    return {
        "campaign": {
            "schema_version": 1,
            "campaign_id": "cpt-prod-v4",
            "transition": "train",
            "content_sha": [
                {
                    "relative_path": "scripts/taey-train",
                    "sha256": SHA256,
                }
            ],
            "consumer_plane": {
                "taey": {"sha256": SHA256},
                "ep3": {"sha256": SHA256},
            },
        },
        "hub": {
            "schema_version": 1,
            "receipt_id": "hub-train-cpt-prod-v4",
            "campaign_id": "cpt-prod-v4",
            "campaign_spec_sha256": SHA256,
            "transition": "train",
            "decision": "approved",
            "authority": {
                "surface": "the-hub",
                "actor_type": "taey",
                "actor_id": "taey-release-router",
                "signer_identity": "taey-release-router",
                "signature_namespace": "taey-release",
                "trust_policy_sha256": SHA256,
            },
            "authorization_plane": "taey-family-chats",
            "issued_at": TIMESTAMP,
            "evidence": [
                {
                    "repository_commit": COMMIT,
                    "receipt_sha256": SHA256,
                }
            ],
            "subject": {
                "artifact_sha256": SHA256,
            },
        },
        "evaluation": {
            "schema_version": 1,
            "evaluation_id": "bake-gate-cpt-prod-v4",
            "campaign_id": "cpt-prod-v4",
            "transition": "bake",
            "subject_sha256": SHA256,
            "assertions": [
                {
                    "assertion_id": "artifact-evidence",
                    "evidence_required": ["artifact receipt"],
                }
            ],
        },
        "checkpoint": {
            "schema_version": 1,
            "campaign_id": "cpt-prod-v4",
            "checkpoint_step": 85,
            "node_local_fragments": {
                "rank-0": {
                    "files": {
                        "dcp/__0.metadata": SHA256,
                    }
                }
            },
            "collector_receipt_sha256": SHA256,
        },
        "lifecycle": {
            "schema_version": 1,
            "observation_id": "cpt-prod-v4-001",
            "campaign_id": "cpt-prod-v4",
            "sequence": 1,
            "observed_at": TIMESTAMP,
            "transition": "train",
            "event": "completed",
            "hub_decision_receipt_sha256": SHA256,
            "evidence_sha256": SHA256,
        },
        "lifecycle_chain": {
            "schema_version": 1,
            "campaign_id": "cpt-prod-v4",
            "observations": [
                {
                    "sequence": 1,
                    "observation_sha256": SHA256,
                }
            ],
        },
        "terminal": {
            "schema_version": 1,
            "release_id": "release-cpt-prod-v4",
            "campaign_id": "cpt-prod-v4",
            "state": "released",
            "campaign_spec_sha256": SHA256,
            "hub_decision_receipt_sha256": SHA256,
            "evaluation_contract_sha256": SHA256,
            "checkpoint_manifest_sha256": SHA256,
            "lifecycle_chain_sha256": SHA256,
            "released_artifact_sha256": SHA256,
            "consumer_aliases": ["taey", "ep3"],
        },
    }


def release_chain_fixture(
    *,
    decision="approved",
    evaluation_campaign_id="cpt-prod-v4",
    collector_node_fragments_sha256=None,
    hub_campaign_spec_sha256=None,
    hub_subject_artifact_sha256=None,
    lifecycle_event="completed",
):
    """Build a digest-consistent JSON/bytes store for cross-object chain validation."""
    artifact = b"served artifact bytes"
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    campaign = {
        "schema_version": 1,
        "campaign_id": "cpt-prod-v4",
        "transition": "promote",
        "content_sha": [{"relative_path": "scripts/taey-train", "sha256": SHA256}],
        "consumer_plane": {"taey": {"sha256": SHA256}, "ep3": {"sha256": SHA256}},
    }
    campaign_sha256 = canonical_sha256(campaign)
    hub = {
        "schema_version": 1,
        "receipt_id": "hub-promote-cpt-prod-v4",
        "campaign_id": "cpt-prod-v4",
        "campaign_spec_sha256": (
            hub_campaign_spec_sha256
            if hub_campaign_spec_sha256 is not None
            else campaign_sha256
        ),
        "transition": "promote",
        "decision": decision,
        "authority": {
            "surface": "the-hub",
            "actor_type": "taey",
            "actor_id": "taey-release-router",
            "signer_identity": "taey-release-router",
            "signature_namespace": "taey-release",
            "trust_policy_sha256": SHA256,
        },
        "authorization_plane": "taey-family-chats",
        "issued_at": TIMESTAMP,
        "evidence": [{"repository_commit": COMMIT, "receipt_sha256": SHA256}],
        "subject": {
            "artifact_sha256": (
                hub_subject_artifact_sha256
                if hub_subject_artifact_sha256 is not None
                else artifact_sha256
            ),
            "rollback_artifact_sha256": SHA256,
            "consumer_aliases": ["taey", "ep3"],
        },
    }
    hub_sha256 = canonical_sha256(hub)
    evaluation = {
        "schema_version": 1,
        "evaluation_id": "promote-gate-cpt-prod-v4",
        "campaign_id": evaluation_campaign_id,
        "transition": "promote",
        "subject_sha256": artifact_sha256,
        "assertions": [{"assertion_id": "artifact-evidence", "evidence_required": ["receipt"]}],
    }
    evaluation_sha256 = canonical_sha256(evaluation)
    node_local_fragments = {"rank-0": {"files": {"dcp/__0.metadata": SHA256}}}
    collector = {
        "campaign_id": "cpt-prod-v4",
        "node_local_fragments_sha256": (
            collector_node_fragments_sha256
            if collector_node_fragments_sha256 is not None
            else canonical_sha256(node_local_fragments)
        ),
    }
    collector_sha256 = canonical_sha256(collector)
    checkpoint = {
        "schema_version": 1,
        "campaign_id": "cpt-prod-v4",
        "checkpoint_step": 85,
        "node_local_fragments": node_local_fragments,
        "collector_receipt_sha256": collector_sha256,
    }
    checkpoint_sha256 = canonical_sha256(checkpoint)
    observation = {
        "schema_version": 1,
        "observation_id": "cpt-prod-v4-001",
        "campaign_id": "cpt-prod-v4",
        "sequence": 1,
        "observed_at": TIMESTAMP,
        "transition": "promote",
        "event": lifecycle_event,
        "hub_decision_receipt_sha256": hub_sha256,
        "evidence_sha256": SHA256,
    }
    observation_sha256 = canonical_sha256(observation)
    lifecycle_chain = {
        "schema_version": 1,
        "campaign_id": "cpt-prod-v4",
        "observations": [{"sequence": 1, "observation_sha256": observation_sha256}],
    }
    lifecycle_chain_sha256 = canonical_sha256(lifecycle_chain)
    terminal = {
        "schema_version": 1,
        "release_id": "release-cpt-prod-v4",
        "campaign_id": "cpt-prod-v4",
        "state": "released",
        "campaign_spec_sha256": campaign_sha256,
        "hub_decision_receipt_sha256": hub_sha256,
        "evaluation_contract_sha256": evaluation_sha256,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "lifecycle_chain_sha256": lifecycle_chain_sha256,
        "released_artifact_sha256": artifact_sha256,
        "consumer_aliases": ["taey", "ep3"],
    }
    records = {
        campaign_sha256: campaign,
        hub_sha256: hub,
        evaluation_sha256: evaluation,
        collector_sha256: collector,
        checkpoint_sha256: checkpoint,
        observation_sha256: observation,
        lifecycle_chain_sha256: lifecycle_chain,
        artifact_sha256: artifact,
    }
    return terminal, records


class ReleaseContractSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = load_schemas()

    def test_every_schema_is_json_schema_and_fails_closed_at_its_root(self):
        self.assertEqual(set(self.schemas), set(SCHEMA_PATHS))
        for name, schema in self.schemas.items():
            with self.subTest(name=name):
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertEqual(schema["type"], "object")
                self.assertIs(schema["additionalProperties"], False)

    def test_valid_contract_objects_validate(self):
        for name, instance in valid_instances().items():
            with self.subTest(name=name):
                self.assertEqual(validate(instance, self.schemas[name]), [])

    def test_unknown_and_approval_fields_are_rejected(self):
        instances = valid_instances()
        for name, instance in instances.items():
            with self.subTest(name=name):
                invalid = copy.deepcopy(instance)
                invalid["approved_by"] = "not-a-valid-field"
                self.assertTrue(validate(invalid, self.schemas[name]))
        for name, schema in self.schemas.items():
            with self.subTest(name=name):
                self.assertFalse(FORBIDDEN_APPROVAL_FIELDS & property_names(schema))

    def test_hub_receipt_requires_hub_repo_evidence_and_attributable_receipt(self):
        receipt = valid_instances()["hub"]
        self.assertEqual(validate(receipt, self.schemas["hub"]), [])
        for field, value in (("authorization_plane", "manual"), ("evidence", [])):
            with self.subTest(field=field):
                invalid = copy.deepcopy(receipt)
                invalid[field] = value
                self.assertTrue(validate(invalid, self.schemas["hub"]))
        invalid = copy.deepcopy(receipt)
        invalid["authority"]["actor_type"] = "user"
        self.assertTrue(validate(invalid, self.schemas["hub"]))
        invalid = copy.deepcopy(receipt)
        del invalid["evidence"][0]["receipt_sha256"]
        self.assertTrue(validate(invalid, self.schemas["hub"]))

    def test_hub_receipt_has_the_canonical_root_envelope(self):
        schema = self.schemas["hub"]
        expected = {
            "schema_version",
            "receipt_id",
            "campaign_id",
            "campaign_spec_sha256",
            "transition",
            "decision",
            "authority",
            "authorization_plane",
            "issued_at",
            "evidence",
            "subject",
        }
        self.assertEqual(set(schema["required"]), expected)
        self.assertEqual(set(schema["properties"]), expected)
        self.assertEqual(
            set(schema["properties"]["authority"]["properties"]),
            {
                "surface",
                "actor_type",
                "actor_id",
                "signer_identity",
                "signature_namespace",
                "trust_policy_sha256",
            },
        )

    def test_canonical_promote_fixture_validates_and_requires_rollback_aliases(self):
        fixture = json.loads(
            (FIXTURE_DIR / "hub-decision-promote.valid.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate(fixture, self.schemas["hub"]), [])
        self.assertEqual(fixture["subject"]["consumer_aliases"], ["taey", "ep3"])
        for field in ("rollback_artifact_sha256", "consumer_aliases"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(fixture)
                del invalid["subject"][field]
                self.assertTrue(validate(invalid, self.schemas["hub"]))
        invalid = copy.deepcopy(fixture)
        invalid["subject"]["consumer_aliases"] = ["ep3", "taey"]
        self.assertTrue(validate(invalid, self.schemas["hub"]))
        invalid = copy.deepcopy(fixture)
        invalid["authority"]["actor_type"] = "user"
        self.assertTrue(validate(invalid, self.schemas["hub"]))
        invalid = copy.deepcopy(fixture)
        invalid["authority"]["signature_namespace"] = "wrong-namespace"
        self.assertTrue(validate(invalid, self.schemas["hub"]))

    def test_campaign_content_entries_are_exact_and_normalized(self):
        campaign = valid_instances()["campaign"]
        self.assertEqual(validate(campaign, self.schemas["campaign"]), [])
        invalid = copy.deepcopy(campaign)
        invalid["content_sha"][0]["approved_by"] = "unsafe"
        self.assertTrue(validate(invalid, self.schemas["campaign"]))
        invalid = copy.deepcopy(campaign)
        invalid["content_sha"][0]["relative_path"] = "../PRODUCTION_MANIFEST.yml"
        self.assertTrue(validate(invalid, self.schemas["campaign"]))
        self.assertEqual(
            set(self.schemas["campaign"]["properties"]["content_sha"]["items"]["properties"]),
            {"relative_path", "sha256"},
        )

    def test_all_transition_schemas_allow_only_train_bake_and_promote(self):
        for name in ("campaign", "hub", "evaluation", "lifecycle"):
            with self.subTest(name=name):
                transition = self.schemas[name]["properties"]["transition"]
                self.assertEqual(transition["enum"], TRANSITIONS)
                invalid = copy.deepcopy(valid_instances()[name])
                invalid["transition"] = "deploy"
                self.assertTrue(validate(invalid, self.schemas[name]))

    def test_checkpoint_requires_node_local_fragments_and_collector_receipt(self):
        manifest = valid_instances()["checkpoint"]
        self.assertEqual(validate(manifest, self.schemas["checkpoint"]), [])
        for field in ("node_local_fragments", "collector_receipt_sha256"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(manifest)
                del invalid[field]
                self.assertTrue(validate(invalid, self.schemas["checkpoint"]))
        invalid = copy.deepcopy(manifest)
        invalid["node_local_fragments"]["rank-0"]["files"] = {"/shared/dcp": SHA256}
        self.assertTrue(validate(invalid, self.schemas["checkpoint"]))
        invalid = copy.deepcopy(manifest)
        invalid["node_local_fragments"]["rank-0"]["files"]["dcp/__0.metadata"] = {
            "sha256": SHA256
        }
        self.assertTrue(validate(invalid, self.schemas["checkpoint"]))

    def test_terminal_receipt_is_digest_bound_and_names_both_consumer_aliases(self):
        receipt = valid_instances()["terminal"]
        self.assertEqual(validate(receipt, self.schemas["terminal"]), [])
        invalid = copy.deepcopy(receipt)
        invalid["state"] = "promoted"
        self.assertTrue(validate(invalid, self.schemas["terminal"]))
        invalid = copy.deepcopy(receipt)
        invalid["consumer_aliases"] = ["ep3", "taey"]
        self.assertTrue(validate(invalid, self.schemas["terminal"]))
        self.assertEqual(
            self.schemas["terminal"]["properties"]["consumer_aliases"]["prefixItems"],
            [{"const": "taey"}, {"const": "ep3"}],
        )

    def test_lifecycle_predecessor_is_absent_at_genesis_and_required_later(self):
        genesis = valid_instances()["lifecycle"]
        self.assertEqual(validate(genesis, self.schemas["lifecycle"]), [])
        invalid = copy.deepcopy(genesis)
        invalid["previous_observation_sha256"] = SHA256
        self.assertTrue(validate(invalid, self.schemas["lifecycle"]))
        later = copy.deepcopy(genesis)
        later["sequence"] = 2
        later["observation_id"] = "cpt-prod-v4-002"
        self.assertTrue(validate(later, self.schemas["lifecycle"]))
        later["previous_observation_sha256"] = SHA256
        self.assertEqual(validate(later, self.schemas["lifecycle"]), [])

    def test_release_chain_validator_rejects_mixed_or_undereferenced_chain(self):
        terminal, records = release_chain_fixture()
        self.assertEqual(validate_release_chain(terminal, records), [])

        missing_artifact = dict(records)
        del missing_artifact[terminal["released_artifact_sha256"]]
        self.assertTrue(validate_release_chain(terminal, missing_artifact))

        rejected_terminal, rejected_records = release_chain_fixture(decision="rejected")
        self.assertTrue(validate_release_chain(rejected_terminal, rejected_records))

        mixed_terminal, mixed_records = release_chain_fixture(evaluation_campaign_id="other-campaign")
        self.assertTrue(validate_release_chain(mixed_terminal, mixed_records))

        broken_terminal, broken_records = release_chain_fixture(
            collector_node_fragments_sha256=SHA256
        )
        self.assertTrue(validate_release_chain(broken_terminal, broken_records))

        spec_mismatch_terminal, spec_mismatch_records = release_chain_fixture(
            hub_campaign_spec_sha256=SHA256
        )
        self.assertTrue(validate_release_chain(spec_mismatch_terminal, spec_mismatch_records))

        lifecycle_mismatch_terminal, lifecycle_mismatch_records = release_chain_fixture(
            lifecycle_event="failed"
        )
        self.assertTrue(validate_release_chain(lifecycle_mismatch_terminal, lifecycle_mismatch_records))

        artifact_mismatch_terminal, artifact_mismatch_records = release_chain_fixture(
            hub_subject_artifact_sha256=SHA256
        )
        self.assertTrue(validate_release_chain(artifact_mismatch_terminal, artifact_mismatch_records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
