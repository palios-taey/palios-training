#!/usr/bin/env python3
"""Validate and bind a merged DDP-LoRA artifact before atomic promotion."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-completion", required=True)
    parser.add_argument("--adapter-sha", required=True)
    parser.add_argument("--adapter-config-sha", required=True)
    parser.add_argument("--base-manifest-sha", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--tooling-commit", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def base_key(adapter_key):
    value = adapter_key.replace(
        "base_model.model.model.",
        "model.language_model.",
        1,
    )
    return (
        value.replace(".lora_A.default.weight", ".weight")
        .replace(".lora_B.default.weight", ".weight")
        .replace(".lora_A.weight", ".weight")
        .replace(".lora_B.weight", ".weight")
    )


def adapter_targets(adapter_path):
    from safetensors import safe_open

    pairs = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    for key in keys:
        target = base_key(key)
        if ".lora_A." in key:
            pairs.setdefault(target, set()).add("A")
        elif ".lora_B." in key:
            pairs.setdefault(target, set()).add("B")
        else:
            raise RuntimeError(f"unexpected adapter tensor: {key}")
    incomplete = sorted(key for key, value in pairs.items() if value != {"A", "B"})
    if incomplete:
        raise RuntimeError(f"incomplete adapter pairs: {incomplete[:5]}")
    return keys, set(pairs)


def preserve_base_receipts(output):
    moves = (
        ("training_provenance.json", "base_training_provenance.json"),
        ("weight_diff.json", "base_cpt_weight_diff.json"),
        ("GRAFT_COMPLETE", "BASE_GRAFT_COMPLETE"),
    )
    for source_name, target_name in moves:
        source = output / source_name
        target = output / target_name
        if source.exists():
            if target.exists():
                raise RuntimeError(f"refusing to overwrite {target}")
            os.replace(source, target)
    manifest = output / "SOURCE_SHA256SUMS"
    if manifest.exists():
        manifest.unlink()


def main():
    args = parse()
    base = Path(args.base)
    adapter = Path(args.adapter)
    output = Path(args.output)
    completion_path = Path(args.training_completion)
    for path in (base, adapter, output):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if len(args.adapter_sha) != 64 or len(args.adapter_config_sha) != 64:
        raise ValueError("adapter digests must be full SHA-256 values")
    if (
        len(args.base_manifest_sha) != 64
        or len(args.plan_sha) != 64
        or len(args.tooling_commit) != 40
    ):
        raise ValueError("plan and tooling digests are malformed")

    adapter_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if sha256_file(adapter_path) != args.adapter_sha:
        raise RuntimeError("adapter digest changed before bake validation")
    if sha256_file(config_path) != args.adapter_config_sha:
        raise RuntimeError("adapter config digest changed before bake validation")

    config = read_json(config_path)
    expected_targets = {
        "down_proj",
        "gate_proj",
        "in_proj_qkv",
        "k_proj",
        "o_proj",
        "out_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    }
    if (
        config.get("r") != 16
        or config.get("lora_alpha") != 32
        or not math.isclose(config.get("lora_dropout", -1), 0.05)
        or set(config.get("target_modules", [])) != expected_targets
        or Path(config.get("base_model_name_or_path", "")).name
        != f"{args.run_tag}_servable"
    ):
        raise RuntimeError("adapter configuration differs from production training")

    completion = read_json(completion_path)
    if (
        completion.get("format") != "stage2-ddp-lora-complete-v1"
        or completion.get("run_tag") != args.run_tag
        or completion.get("steps") != args.steps
        or completion.get("plan_sha256") != args.plan_sha
        or completion.get("adapter_sha256") != args.adapter_sha
    ):
        raise RuntimeError("training completion receipt does not bind this adapter")

    adapter_keys, targets = adapter_targets(adapter_path)
    if len(adapter_keys) != 704 or len(targets) != 352:
        raise RuntimeError(
            f"adapter coverage differs from qualification: {len(adapter_keys)}/"
            f"{len(targets)}"
        )
    target_digest = hashlib.sha256()
    for target in sorted(targets):
        target_digest.update(target.encode("utf-8"))
        target_digest.update(b"\n")

    merge_receipt_path = output / "lora_merge_receipt.json"
    merge_receipt = read_json(merge_receipt_path)
    base_manifest_sha = args.base_manifest_sha
    if (
        merge_receipt.get("format") != "palios-lora-merge-v1"
        or merge_receipt.get("adapter_sha256") != args.adapter_sha
        or merge_receipt.get("adapter_config_sha256") != args.adapter_config_sha
        or merge_receipt.get("adapter_tensors") != 704
        or merge_receipt.get("target_matrices") != 352
        or merge_receipt.get("applied_target_matrices") != 352
        or merge_receipt.get("base_manifest_sha256") != base_manifest_sha
        or merge_receipt.get("target_key_sha256") != target_digest.hexdigest()
        or not math.isfinite(merge_receipt.get("delta_abs_mean", math.nan))
        or merge_receipt.get("delta_abs_mean", 0) <= 0
    ):
        raise RuntimeError("merge receipt does not bind the qualified adapter and base")

    base_index = read_json(base / "model.safetensors.index.json")["weight_map"]
    output_index = read_json(output / "model.safetensors.index.json")["weight_map"]
    if base_index != output_index or len(base_index) != 1199:
        raise RuntimeError("merged model index differs from the 1199-tensor base")
    if not targets.issubset(base_index):
        missing = sorted(targets - set(base_index))
        raise RuntimeError(f"adapter targets are absent from base index: {missing[:5]}")

    changed = set()
    indexed_seen = set()
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    delta_elements = 0
    for shard_name in sorted(set(base_index.values())):
        base_shard = base / shard_name
        output_shard = output / shard_name
        base_tensors = load_file(base_shard, device="cpu")
        output_tensors = load_file(output_shard, device="cpu")
        with (
            safe_open(base_shard, framework="pt", device="cpu") as base_handle,
            safe_open(output_shard, framework="pt", device="cpu") as output_handle,
        ):
            if base_handle.metadata() != output_handle.metadata():
                raise RuntimeError(f"safetensors metadata changed in {shard_name}")
        if set(base_tensors) != set(output_tensors):
            raise RuntimeError(f"tensor-key mismatch in {shard_name}")
        for key, base_tensor in base_tensors.items():
            indexed_seen.add(key)
            output_tensor = output_tensors[key]
            if base_tensor.shape != output_tensor.shape or base_tensor.dtype != output_tensor.dtype:
                raise RuntimeError(f"tensor structure changed for {key}")
            equal = torch.equal(base_tensor, output_tensor)
            if key in targets:
                if equal:
                    raise RuntimeError(f"trained target did not change after BF16 merge: {key}")
                changed.add(key)
                delta = (output_tensor.float() - base_tensor.float()).abs()
                delta_abs_sum += delta.sum().item()
                delta_abs_max = max(delta_abs_max, delta.max().item())
                delta_elements += delta.numel()
            elif not equal:
                raise RuntimeError(f"non-LoRA tensor changed during merge: {key}")
        del base_tensors, output_tensors

    if indexed_seen != set(base_index) or changed != targets or delta_elements <= 0:
        raise RuntimeError(
            "merged tensor coverage mismatch: "
            f"indexed={len(indexed_seen)} changed={len(changed)} "
            f"targets={len(targets)}"
        )
    merged_delta = {
        "abs_max_dW": delta_abs_max,
        "abs_mean_dW": delta_abs_sum / delta_elements,
        "adapter_sha256": args.adapter_sha,
        "changed_target_tensors": len(changed),
        "delta_elements": delta_elements,
        "format": "palios-sft-merged-weight-diff-v1",
        "unchanged_non_target_tensors": len(indexed_seen - changed),
    }
    if not math.isfinite(merged_delta["abs_mean_dW"]) or merged_delta["abs_mean_dW"] <= 0:
        raise RuntimeError("merged output has no finite non-zero weight delta")

    preserve_base_receipts(output)
    write_json(output / "merged_weight_diff.json", merged_delta)
    provenance = {
        "adapter": {
            "config_sha256": args.adapter_config_sha,
            "sha256": args.adapter_sha,
            "tensors": len(adapter_keys),
        },
        "artifact_tensors": len(base_index),
        "base": {
            "manifest_sha256": base_manifest_sha,
            "path": str(base),
        },
        "merge_receipt_sha256": sha256_file(merge_receipt_path),
        "run_tag": args.run_tag,
        "stage": "sft_lora_merge",
        "target_matrices": len(targets),
        "tooling_commit": args.tooling_commit,
        "training": completion,
        "weight_diff": merged_delta,
    }
    write_json(output / "training_provenance.json", provenance)
    complete = {
        "adapter_sha256": args.adapter_sha,
        "artifact_tensors": len(base_index),
        "format": "stage2-ddp-sft-merge-complete-v1",
        "plan_sha256": args.plan_sha,
        "run_tag": args.run_tag,
        "steps": args.steps,
        "target_matrices": len(targets),
        "tooling_commit": args.tooling_commit,
    }
    write_json(output / "SFT_MERGE_COMPLETE.json", complete)
    print(
        "SFT_MERGE_VALIDATED "
        + json.dumps(
            {
                **complete,
                **merged_delta,
                "base_manifest_sha256": base_manifest_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
