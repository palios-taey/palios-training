#!/usr/bin/env python3
"""Repair a corpus-lineage field from an artifact-bound corpus manifest."""

import argparse
import hashlib
import json
import os
import sys

from corpus_manifest import verify_manifest


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_exclusive(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if open(path, "rb").read() != data:
            raise ValueError(f"audit copy already exists with different bytes: {path}")


def write_atomic(path, record):
    stage = path + ".lineage-repair.tmp"
    with open(stage, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--expected-current-provenance-sha256", required=True)
    parser.add_argument("--audit-copy", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    provenance_path = os.path.join(args.artifact, "training_provenance.json")
    try:
        corpus_receipt = verify_manifest(args.corpus, args.corpus_manifest)
        current_bytes = open(provenance_path, "rb").read()
        record = json.loads(current_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"REFUSE: {error}") from error

    target_inputs = corpus_receipt["corpus_inputs"]
    target_manifest = corpus_receipt["corpus_manifest_sha256"]
    current_sha = sha256_bytes(current_bytes)
    if (record.get("corpus_inputs") == target_inputs
            and record.get("corpus_manifest_sha256") == target_manifest):
        print(f"SKIP: provenance already binds corpus manifest {target_manifest}")
        print(f"PROVENANCE_SHA256={current_sha}")
        return 0
    if current_sha != args.expected_current_provenance_sha256:
        raise SystemExit(
            "REFUSE: current provenance SHA differs from the incident-bound source"
        )
    if (record.get("corpus_sha256") != corpus_receipt["corpus_sha256"]
            or record.get("corpus_rows") != corpus_receipt["corpus_rows"]):
        raise SystemExit("REFUSE: provenance and corpus manifest identify different packed corpora")
    if record.get("stage") != "graft" or record.get("artifact_tensors") != 1199:
        raise SystemExit("REFUSE: provenance is not a 1199-tensor graft record")

    try:
        write_exclusive(args.audit_copy, current_bytes)
    except (OSError, ValueError) as error:
        raise SystemExit(f"REFUSE: {error}") from error

    record["corpus_inputs"] = target_inputs
    record["corpus_manifest_sha256"] = target_manifest
    record["lineage_correction"] = {
        "schema": "palios.provenance_lineage_correction.v1",
        "date": args.date,
        "previous_provenance_sha256": current_sha,
        "corpus_manifest_sha256": target_manifest,
        "reason": args.reason,
    }
    write_atomic(provenance_path, record)
    repaired_sha = sha256_bytes(open(provenance_path, "rb").read())
    print(f"REPAIRED: {provenance_path}")
    print(f"PROVENANCE_SHA256={repaired_sha}")
    print(f"CORPUS_MANIFEST_SHA256={target_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
