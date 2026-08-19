#!/usr/bin/env python3
"""Measured content gates for the post-CPT bake and delivery path."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile


VISION_SAMPLES = (
    "model.visual.blocks.0.attn.proj.bias",
    "model.visual.blocks.15.attn.proj.bias",
    "model.visual.patch_embed.proj.bias",
)
EXPECTED_TRAINING = 851
EXPECTED_SERVABLE = 1199
DIFF_MIN = 5e-5
DIFF_MAX = 8e-4
MANIFEST = "ARTIFACT_SHA256SUMS"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index(root):
    path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(path):
        raise ContractError(f"no model.safetensors.index.json under {root}")
    try:
        weight_map = json.load(open(path, encoding="utf-8"))["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError(f"invalid safetensors index at {path}: {exc}") from exc
    if not isinstance(weight_map, dict):
        raise ContractError(f"weight_map is not an object at {path}")
    missing = sorted(
        shard for shard in set(weight_map.values()) if not os.path.isfile(os.path.join(root, shard))
    )
    if missing:
        raise ContractError(f"{root} is missing indexed shards: {missing[:5]}")
    return weight_map


def tensor_hash(root, weight_map, name):
    if name not in weight_map:
        raise ContractError(f"donor is missing measured vision tensor {name}")
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise ContractError(f"torch and safetensors are required for donor measurement: {exc}") from exc
    try:
        with safe_open(os.path.join(root, weight_map[name]), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous().view(torch.uint8)
    except (OSError, RuntimeError, KeyError) as exc:
        raise ContractError(f"cannot measure donor tensor {name}: {exc}") from exc
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def resolve_donor(args):
    loaded = os.path.abspath(args.model_path_loaded)
    loaded_map = index(loaded)
    count = len(loaded_map)
    if count == EXPECTED_SERVABLE:
        donor = loaded
        source = loaded
        resolution = "loaded-1199"
    elif count == EXPECTED_TRAINING:
        sidecar_path = os.path.join(loaded, "DERIVED_FROM.json")
        if not os.path.isfile(sidecar_path):
            raise ContractError(
                f"MODEL_PATH_LOADED has 851 tensors but {sidecar_path} is absent; "
                "refusing fixed-pin fallback"
            )
        try:
            sidecar = json.load(open(sidecar_path, encoding="utf-8"))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ContractError(f"invalid {sidecar_path}: {exc}") from exc
        if not isinstance(sidecar, dict):
            raise ContractError(f"{sidecar_path} is not a JSON object")
        source = sidecar.get("source")
        if not isinstance(source, str) or not os.path.isabs(source):
            raise ContractError(f"{sidecar_path} has no absolute source")
        donor = os.path.abspath(source)
        resolution = "derived-851-source"
    else:
        raise ContractError(
            f"MODEL_PATH_LOADED tensor count is {count}; expected 851 or 1199"
        )
    donor_map = index(donor)
    if len(donor_map) != EXPECTED_SERVABLE:
        raise ContractError(
            f"resolved donor {donor} has {len(donor_map)} tensors; expected {EXPECTED_SERVABLE}"
        )
    hashes = {name: tensor_hash(donor, donor_map, name) for name in VISION_SAMPLES}
    receipt = {
        "model_path_loaded": loaded,
        "loaded_tensors": count,
        "resolution": resolution,
        "source": os.path.abspath(source),
        "donor": donor,
        "donor_tensors": len(donor_map),
        "vision_tensor_sha256": hashes,
    }
    print(json.dumps(receipt, sort_keys=True))


def artifact_files(root):
    root = os.path.abspath(root)
    rows = []
    for current, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        symlink_dirs = [name for name in dirs if os.path.islink(os.path.join(current, name))]
        if symlink_dirs:
            raise ContractError(f"artifact contains symlink directory: {symlink_dirs[0]}")
        for name in files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise ContractError(f"artifact contains symlink: {path}")
            rel = os.path.relpath(path, root)
            if rel == MANIFEST:
                continue
            if "\n" in rel:
                raise ContractError(f"artifact filename contains newline: {rel!r}")
            rows.append((rel, path))
    if not rows:
        raise ContractError(f"artifact has no files: {root}")
    return rows


def seal_artifact(args):
    root = os.path.abspath(args.root)
    rows = artifact_files(root)
    manifest_path = os.path.join(root, MANIFEST)
    descriptor, stage = tempfile.mkstemp(
        prefix=f".{os.path.basename(root)}.{MANIFEST}.", dir=os.path.dirname(root), text=True
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for rel, path in rows:
            handle.write(f"{sha256_file(path)}  {rel}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, manifest_path)
    digest = sha256_file(manifest_path)
    print(json.dumps({"artifact_sha256": digest, "files": len(rows), "root": root}, sort_keys=True))


def parse_manifest(root):
    manifest_path = os.path.join(root, MANIFEST)
    if not os.path.isfile(manifest_path):
        raise ContractError(f"artifact has no {MANIFEST}: {root}")
    expected = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.rstrip("\n")
            if "  " not in raw:
                raise ContractError(f"{MANIFEST} line {line_no} is malformed")
            digest, rel = raw.split("  ", 1)
            if not SHA256_RE.fullmatch(digest) or not rel or os.path.isabs(rel):
                raise ContractError(f"{MANIFEST} line {line_no} is invalid")
            if rel in expected:
                raise ContractError(f"{MANIFEST} repeats {rel}")
            expected[rel] = digest
    if not expected:
        raise ContractError(f"{MANIFEST} has no file records")
    return manifest_path, expected


def verify_artifact(args):
    root = os.path.abspath(args.root)
    manifest_path, expected = parse_manifest(root)
    actual_files = {rel: path for rel, path in artifact_files(root)}
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    mismatched = sorted(
        rel for rel in set(expected) & set(actual_files)
        if sha256_file(actual_files[rel]) != expected[rel]
    )
    digest = sha256_file(manifest_path)
    if missing or extra or mismatched:
        raise ContractError(
            f"artifact verification root={root} missing={missing[:3]} extra={extra[:3]} "
            f"sha_mismatch={mismatched[:3]}"
        )
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ContractError(
            f"artifact manifest digest actual={digest} expected={args.expected_sha256} root={root}"
        )
    print(json.dumps({"artifact_sha256": digest, "files": len(expected), "root": root}, sort_keys=True))


def weight_value(path):
    try:
        record = json.load(open(path, encoding="utf-8"))
        return float(record["abs_mean_dW"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid weight-diff receipt {path}: {exc}") from exc


def weight_gate(args):
    value = weight_value(args.weight_diff)
    if not DIFF_MIN <= value <= DIFF_MAX:
        verdict = "BELOW" if value < DIFF_MIN else "ABOVE"
        raise ContractError(
            f"weight-diff actual={value:.9e} band={DIFF_MIN:.0e}..{DIFF_MAX:.0e} verdict={verdict}"
        )
    print(f"WEIGHT_DIFF {value:.9e} IN_BAND")


def final_gate(args):
    base = len(index(args.training_base))
    converted = len(index(args.converted))
    servable = len(index(args.servable))
    value = weight_value(args.weight_diff)
    failures = []
    if base != EXPECTED_TRAINING:
        failures.append(f"training_base={base} expected={EXPECTED_TRAINING}")
    if converted != EXPECTED_TRAINING:
        failures.append(f"converted={converted} expected={EXPECTED_TRAINING}")
    if servable != EXPECTED_SERVABLE:
        failures.append(f"servable={servable} expected={EXPECTED_SERVABLE}")
    if not DIFF_MIN <= value <= DIFF_MAX:
        failures.append(
            f"mean|dW|={value:.9e} band={DIFF_MIN:.0e}..{DIFF_MAX:.0e}"
        )
    if failures:
        raise ContractError("; ".join(failures))
    print(
        f"POST_CPT GATE PASS training_base={base} converted={converted} "
        f"servable={servable} mean|dW|={value:.9e}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    donor = sub.add_parser("resolve-donor")
    donor.add_argument("--model-path-loaded", required=True)
    donor.set_defaults(func=resolve_donor)
    seal = sub.add_parser("seal-artifact")
    seal.add_argument("--root", required=True)
    seal.set_defaults(func=seal_artifact)
    verify = sub.add_parser("verify-artifact")
    verify.add_argument("--root", required=True)
    verify.add_argument("--expected-sha256")
    verify.set_defaults(func=verify_artifact)
    weight = sub.add_parser("weight-gate")
    weight.add_argument("--weight-diff", required=True)
    weight.set_defaults(func=weight_gate)
    gate = sub.add_parser("final-gate")
    gate.add_argument("--training-base", required=True)
    gate.add_argument("--converted", required=True)
    gate.add_argument("--servable", required=True)
    gate.add_argument("--weight-diff", required=True)
    gate.set_defaults(func=final_gate)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ContractError, OSError, ValueError) as exc:
        print(f"POST_CPT CONTRACT FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
