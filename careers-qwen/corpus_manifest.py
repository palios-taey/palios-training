#!/usr/bin/env python3
"""Create and verify the immutable input receipt for a packed CPT corpus."""

import argparse
import hashlib
import json
import os
import sys


SCHEMA = "palios.cpt_packed_corpus.v1"
GENERATION_SCHEMA = "palios.cpt_packed_generation.v1"
HEX = frozenset("0123456789abcdef")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_rows(path):
    rows = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            rows += chunk.count(b"\n")
    return rows


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def verify_manifest(corpus_path, manifest_path):
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported corpus manifest schema: {manifest.get('schema')!r}")
    if manifest.get("corpus_filename") != os.path.basename(corpus_path):
        raise ValueError("corpus filename does not match its manifest")

    actual_sha = sha256_file(corpus_path)
    actual_bytes = os.path.getsize(corpus_path)
    actual_rows = corpus_rows(corpus_path)
    if manifest.get("corpus_sha256") != actual_sha:
        raise ValueError("corpus SHA-256 does not match its manifest")
    if manifest.get("corpus_bytes") != actual_bytes:
        raise ValueError("corpus byte count does not match its manifest")
    if manifest.get("corpus_rows") != actual_rows:
        raise ValueError("corpus row count does not match its manifest")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("corpus manifest has no input receipts")
    names = set()
    labels = []
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("corpus input receipt is not an object")
        name = item.get("name")
        rows = item.get("rows")
        sha256 = item.get("sha256")
        registered = item.get("registered_sha256_prefix")
        if (not isinstance(name, str) or not name or os.path.basename(name) != name
                or "," in name or "@" in name):
            raise ValueError(f"invalid corpus input name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate corpus input name: {name}")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"invalid row count for corpus input {name}")
        if not _valid_sha256(sha256):
            raise ValueError(f"invalid SHA-256 for corpus input {name}")
        if (not isinstance(registered, str) or len(registered) != 16
                or set(registered) - HEX or not sha256.startswith(registered)):
            raise ValueError(f"registered SHA prefix does not bind corpus input {name}")
        names.add(name)
        labels.append(f"{name.removesuffix('.jsonl')}@{registered}")

    return {
        "corpus_sha256": actual_sha,
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "corpus_inputs": labels,
        "corpus_rows": actual_rows,
        "corpus_bytes": actual_bytes,
        "manifest": manifest,
    }


def write_manifest(path, manifest):
    stage = path + ".tmp"
    with open(stage, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)


def generation_pointer_path(logical_corpus):
    return logical_corpus + ".generation"


def write_generation_pointer(pointer_path, corpus_path, manifest_path):
    """Publish one atomic generation pointer after both artifacts exist and verify."""
    verify_manifest(corpus_path, manifest_path)
    payload = {
        "schema": GENERATION_SCHEMA,
        "corpus": os.path.basename(corpus_path),
        "manifest": os.path.basename(manifest_path),
        "corpus_sha256": sha256_file(corpus_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    stage = pointer_path + ".tmp"
    with open(stage, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, pointer_path)


def pointer_member(directory, name, label):
    """Resolve a pointer payload name to a file in directory. Basenames only."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"generation pointer {label} is missing")
    if name != os.path.basename(name) or name in (".", ".."):
        raise ValueError(f"generation pointer {label} must be a basename, not {name!r}")
    if os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"generation pointer {label} must be a basename, not {name!r}")
    directory = os.path.abspath(directory)
    path = os.path.abspath(os.path.join(directory, name))
    if os.path.dirname(path) != directory:
        raise ValueError(f"generation pointer {label} escapes the pointer directory")
    return path


def resolve_generation(logical_corpus):
    """Return (corpus, manifest). Pointer mismatch is mixed-generation and is refused.

    No pointer: historical layout (logical corpus + logical.manifest.json).
    """
    pointer = generation_pointer_path(logical_corpus)
    if not os.path.exists(pointer):
        return os.path.abspath(logical_corpus), os.path.abspath(logical_corpus + ".manifest.json")
    with open(pointer) as handle:
        payload = json.load(handle)
    if payload.get("schema") != GENERATION_SCHEMA:
        raise ValueError(f"unsupported generation pointer schema: {payload.get('schema')!r}")
    directory = os.path.dirname(os.path.abspath(pointer))
    corpus = pointer_member(directory, payload.get("corpus"), "corpus")
    manifest = pointer_member(directory, payload.get("manifest"), "manifest")
    if sha256_file(corpus) != payload.get("corpus_sha256"):
        raise ValueError("generation pointer corpus sha256 does not match the corpus file")
    if sha256_file(manifest) != payload.get("manifest_sha256"):
        raise ValueError("generation pointer manifest sha256 does not match the manifest file")
    verify_manifest(corpus, manifest)
    return corpus, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "resolve"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--receipt-lines", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "resolve":
            corpus, _manifest = resolve_generation(args.corpus)
            print(corpus)
            return 0
        if not args.manifest:
            raise SystemExit("REFUSE: --manifest is required for verify")
        receipt = verify_manifest(args.corpus, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"REFUSE: {error}") from error
    if args.receipt_lines:
        print(receipt["corpus_sha256"])
        print(receipt["corpus_manifest_sha256"])
        print(",".join(receipt["corpus_inputs"]))
    else:
        print(json.dumps({
            key: receipt[key]
            for key in (
                "corpus_sha256",
                "corpus_manifest_sha256",
                "corpus_inputs",
                "corpus_rows",
                "corpus_bytes",
            )
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
