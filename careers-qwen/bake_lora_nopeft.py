#!/usr/bin/env python3
"""
bake_lora.py -- Manually apply LoRA weights into safetensor shards.

Produces a checkpoint structurally identical to the base model but with
LoRA deltas baked into the target weight matrices. No config changes,
no architecture changes, no visual weight loss. vLLM loads the output
exactly as it loads the unmodified base model.

Formula: W_new = W_base + (alpha / r) * B @ A
"""
import glob
import hashlib
import json
import math
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = os.environ.get("BASE_MODEL", os.path.expanduser("~/models/Huihui-Qwen3.5-35B-A3B-abliterated"))
LORA = os.environ.get("LORA_PATH", os.path.expanduser("~/models/taey-lora-v1"))
OUT = os.environ.get("OUTPUT_PATH", os.path.expanduser("~/models/taey-baked"))
BASE_MANIFEST_SHA256 = os.environ.get("BASE_MANIFEST_SHA256")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    if (
        BASE_MANIFEST_SHA256 is None
        or len(BASE_MANIFEST_SHA256) != 64
        or any(value not in "0123456789abcdef" for value in BASE_MANIFEST_SHA256)
    ):
        raise RuntimeError("BASE_MANIFEST_SHA256 must bind the verified source bytes")

    # Load adapter config and weights
    config_path = os.path.join(LORA, "adapter_config.json")
    adapter_path = os.path.join(LORA, "adapter_model.safetensors")
    with open(config_path, encoding="utf-8") as handle:
        _cfg = json.load(handle)
    lora_weights = load_file(adapter_path, device="cpu")
    r = _cfg["r"]
    alpha = _cfg["lora_alpha"]
    if not isinstance(r, int) or not isinstance(alpha, int) or r <= 0 or alpha <= 0:
        raise RuntimeError(f"invalid LoRA scale contract: r={r!r} alpha={alpha!r}")
    scale = alpha / r
    print(f"LoRA: r={r}, alpha={alpha}, scale={scale}")

    # Build lookup: map base model weight key -> {A, B} tensors
    # Adapter keys: "base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight"
    # Base keys:    "model.language_model.layers.N.self_attn.q_proj.weight"
    def get_base_key(lora_key):
        language_model_targets = (
            ".self_attn.q_proj.",
            ".linear_attn.in_proj_qkv.",
            ".linear_attn.in_proj_z.",
            ".linear_attn.out_proj.",
        )

        if any(target in lora_key for target in language_model_targets):
            k = lora_key.replace("base_model.model.model.", "model.language_model.", 1)
        else:
            k = lora_key.replace("base_model.model.model.", "model.language_model.")

        k = k.replace(".lora_A.default.weight", ".weight").replace(".lora_B.default.weight", ".weight").replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
        return k

    modules = {}
    for k, v in lora_weights.items():
        base_k = get_base_key(k)
        if ".lora_A." in k:
            modules.setdefault(base_k, {})["A"] = v
        elif ".lora_B." in k:
            modules.setdefault(base_k, {})["B"] = v

    incomplete = sorted(
        key for key, value in modules.items() if set(value) != {"A", "B"}
    )
    if incomplete:
        raise RuntimeError(f"incomplete LoRA matrix pairs: {incomplete[:5]}")
    print(f"LoRA targets {len(modules)} weight matrices")

    # Load base model index
    with open(
        os.path.join(BASE, "model.safetensors.index.json"),
        encoding="utf-8",
    ) as f:
        index = json.load(f)["weight_map"]
    missing_targets = sorted(set(modules) - set(index))
    if missing_targets:
        raise RuntimeError(
            f"{len(missing_targets)} LoRA targets are absent from the base index: "
            f"{missing_targets[:5]}"
        )

    # Process each shard
    applied = 0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    delta_elements = 0
    for shard_name in sorted(set(index.values())):
        print(f"Processing {shard_name}...")
        base_shard = os.path.join(BASE, shard_name)
        weights = load_file(base_shard, device="cpu")
        with safe_open(base_shard, framework="pt", device="cpu") as handle:
            shard_metadata = handle.metadata()
        modified = False

        for wkey in list(weights.keys()):
            if wkey in modules:
                tensor = weights[wkey]
                A = modules[wkey]["A"].to(torch.float32)
                B = modules[wkey]["B"].to(torch.float32)
                delta = scale * (B @ A)
                if delta.shape != tensor.shape:
                    raise RuntimeError(
                        f"shape mismatch: {wkey}: {tensor.shape} vs {delta.shape}"
                    )
                if not torch.isfinite(delta).all():
                    raise RuntimeError(f"non-finite LoRA delta for {wkey}")
                delta_abs = delta.abs()
                delta_abs_sum += delta_abs.sum().item()
                delta_abs_max = max(delta_abs_max, delta_abs.max().item())
                delta_elements += delta.numel()
                weights[wkey] = tensor + delta.to(tensor.dtype)
                modified = True
                applied += 1
                print(f"  Applied LoRA to {wkey}")

        save_file(weights, os.path.join(OUT, shard_name), metadata=shard_metadata)
        if not modified:
            print("  (no LoRA targets in this shard)")
        del weights

    print(f"\nTotal LoRA applications: {applied}")
    if applied != len(modules) or delta_elements <= 0 or delta_abs_sum <= 0:
        raise RuntimeError(
            "LoRA merge coverage failed: "
            f"applied={applied} targets={len(modules)} "
            f"elements={delta_elements} abs_sum={delta_abs_sum}"
        )

    # Copy all non-safetensor files unchanged
    for f in glob.glob(os.path.join(BASE, "*")):
        fname = os.path.basename(f)
        if not fname.endswith(".safetensors"):
            shutil.copy2(f, os.path.join(OUT, fname))
            print(f"Copied {fname}")

    target_digest = hashlib.sha256()
    for target in sorted(modules):
        target_digest.update(target.encode("utf-8"))
        target_digest.update(b"\n")
    receipt = {
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_tensors": len(lora_weights),
        "alpha": alpha,
        "applied_target_matrices": applied,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_model": BASE,
        "delta_abs_max": delta_abs_max,
        "delta_abs_mean": delta_abs_sum / delta_elements,
        "delta_elements": delta_elements,
        "format": "palios-lora-merge-v1",
        "lora_path": LORA,
        "output_path": OUT,
        "r": r,
        "scale": scale,
        "target_key_sha256": target_digest.hexdigest(),
        "target_matrices": len(modules),
    }
    if not math.isfinite(receipt["delta_abs_mean"]):
        raise RuntimeError("non-finite aggregate LoRA delta")
    with open(
        os.path.join(OUT, "lora_merge_receipt.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print("MERGE_RECEIPT " + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    print(f"\nBaked model at {OUT}")
    print("BAKE COMPLETE")


if __name__ == "__main__":
    main()
