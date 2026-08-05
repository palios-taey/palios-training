#!/usr/bin/env python3
"""Fail-closed inspection of one exact per-rank DCP checkpoint bundle."""

import argparse
import glob
import json
import os
from pathlib import Path


def require_regular_file(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required regular file is missing: {path.name}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"required file is empty: {path.name}")
    return size


def parse_complete(path):
    marker = path.read_text(encoding="utf-8").strip()
    fields = {}
    for token in marker.split():
        if token.count("=") != 1:
            raise RuntimeError("COMPLETE contains a malformed token")
        key, value = token.split("=", 1)
        if key in fields:
            raise RuntimeError(f"COMPLETE contains duplicate key {key}")
        fields[key] = value
    expected_keys = {"step", "epoch", "data_pos", "rank"}
    if set(fields) != expected_keys:
        raise RuntimeError(
            f"COMPLETE keys differ: actual={sorted(fields)} expected={sorted(expected_keys)}"
        )
    try:
        return {key: int(value) for key, value in fields.items()}
    except ValueError as error:
        raise RuntimeError("COMPLETE contains a non-integer value") from error


def inspect_checkpoint(checkpoint, step, rank, world_size):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        raise RuntimeError("checkpoint path must be absolute")
    if checkpoint_path.is_symlink():
        raise RuntimeError("checkpoint path must not be a symlink")
    if checkpoint_path.name != f"checkpoint-{step}":
        raise RuntimeError(
            f"checkpoint basename {checkpoint_path.name!r} does not select step {step} exactly"
        )
    if not checkpoint_path.is_dir():
        raise RuntimeError("checkpoint directory does not exist")

    complete_path = checkpoint_path / "COMPLETE"
    complete_bytes = require_regular_file(complete_path)
    complete = parse_complete(complete_path)
    if complete["step"] != step or complete["rank"] != rank:
        raise RuntimeError(
            f"COMPLETE identity mismatch: step={complete['step']} rank={complete['rank']}"
        )

    trainer_meta_path = checkpoint_path / "trainer_meta.pt"
    trainer_meta_bytes = require_regular_file(trainer_meta_path)
    import torch

    trainer_meta = torch.load(trainer_meta_path, map_location="cpu", weights_only=True)
    if not isinstance(trainer_meta, dict):
        raise RuntimeError("trainer_meta.pt is not a dictionary")
    if trainer_meta.get("format") != "dcp_v2":
        raise RuntimeError(f"trainer_meta format is {trainer_meta.get('format')!r}, not 'dcp_v2'")
    if trainer_meta.get("step") != step:
        raise RuntimeError(
            f"trainer_meta step is {trainer_meta.get('step')!r}, expected {step}"
        )
    if trainer_meta.get("num_ranks") != world_size:
        raise RuntimeError(
            f"trainer_meta num_ranks is {trainer_meta.get('num_ranks')!r}, expected {world_size}"
        )

    dcp_path = checkpoint_path / "dcp"
    if dcp_path.is_symlink() or not dcp_path.is_dir():
        raise RuntimeError("dcp must be a real directory")
    rank_metadata_path = dcp_path / f"__{rank}.metadata"
    rank_metadata_bytes = require_regular_file(rank_metadata_path)
    shard_paths = [Path(path) for path in sorted(glob.glob(str(dcp_path / f"__{rank}_*.distcp")))]
    if not shard_paths:
        raise RuntimeError(f"no rank-{rank} DCP shards found")
    shard_bytes = sum(require_regular_file(path) for path in shard_paths)

    return {
        "checkpoint": str(checkpoint_path),
        "complete_bytes": complete_bytes,
        "dcp_metadata_bytes": rank_metadata_bytes,
        "dcp_shard_bytes": shard_bytes,
        "dcp_shards": len(shard_paths),
        "epoch": complete["epoch"],
        "rank": rank,
        "step": step,
        "trainer_meta_bytes": trainer_meta_bytes,
        "world_size": world_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.step < 0:
        parser.error("--step must be non-negative")
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be within the world")
    receipt = inspect_checkpoint(args.checkpoint, args.step, args.rank, args.world_size)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
