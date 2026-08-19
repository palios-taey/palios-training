#!/usr/bin/env python3
"""Verify distributed or assembled Artifact-B shards against rank manifests."""

import argparse
import glob
import hashlib
import json
import os
import sys


class ArtifactError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    try:
        record = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    for field in ("rank", "world", "files"):
        if field not in record:
            raise ArtifactError(f"{path} has no {field}")
    if not isinstance(record["files"], list) or not record["files"]:
        raise ArtifactError(f"{path} has no shard records")
    if not isinstance(record["rank"], int) or not isinstance(record["world"], int):
        raise ArtifactError(f"{path} rank/world are not integers")
    for item in record["files"]:
        if not isinstance(item, dict):
            raise ArtifactError(f"{path} contains a non-object shard record")
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ArtifactError(f"{path} contains a shard record with no name")
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
            raise ArtifactError(f"{path} shard {item['name']} has invalid bytes")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactError(f"{path} shard {item['name']} has invalid sha256")
    return record


def verify_rank(root, rank, world, require_metadata=False):
    manifest_path = os.path.join(root, f"manifest.rank{rank}.json")
    manifest = read_manifest(manifest_path)
    if manifest["rank"] != rank or manifest["world"] != world:
        raise ArtifactError(
            f"rank manifest identity actual={manifest['rank']}/{manifest['world']} "
            f"expected={rank}/{world} path={manifest_path}"
        )
    expected_names = {f"__{rank}_0.distcp"}
    manifest_names = {item.get("name") for item in manifest["files"]}
    if manifest_names != expected_names:
        raise ArtifactError(
            f"rank {rank} shard set actual={sorted(str(v) for v in manifest_names)} "
            f"expected={sorted(expected_names)}"
        )
    for item in manifest["files"]:
        path = os.path.join(root, item["name"])
        if not os.path.isfile(path):
            raise ArtifactError(f"rank {rank} missing shard {item['name']}")
        size = os.path.getsize(path)
        digest = sha256_file(path)
        if size != item.get("bytes") or digest != item.get("sha256"):
            raise ArtifactError(
                f"rank {rank} shard {item['name']} bytes={size}/{item.get('bytes')} "
                f"sha256={digest}/{item.get('sha256')}"
            )
    ready = os.path.join(root, f"READY.rank{rank}")
    if not os.path.isfile(ready):
        raise ArtifactError(f"rank {rank} missing READY.rank{rank}")
    if require_metadata and not os.path.isfile(os.path.join(root, ".metadata")):
        raise ArtifactError("rank 0 missing global .metadata")
    return len(manifest["files"])


def rank_command(args):
    shards = verify_rank(args.root, args.rank, args.world, args.rank == 0)
    print(
        f"ARTIFACT_B RANK PASS rank={args.rank} world={args.world} "
        f"shards={shards} root={args.root}"
    )


def assembled_command(args):
    manifests = sorted(glob.glob(os.path.join(args.root, "manifest.rank*.json")))
    if not manifests:
        raise ArtifactError(f"no manifest.rank*.json under {args.root}")
    records = [read_manifest(path) for path in manifests]
    worlds = {record["world"] for record in records}
    ranks = {record["rank"] for record in records}
    if len(worlds) != 1:
        raise ArtifactError(f"world set is {sorted(worlds)}")
    world = next(iter(worlds))
    expected_ranks = set(range(world))
    if ranks != expected_ranks:
        raise ArtifactError(f"rank set actual={sorted(ranks)} expected={sorted(expected_ranks)}")
    if len(manifests) != world:
        raise ArtifactError(f"manifest count actual={len(manifests)} expected={world}")
    total_shards = 0
    for rank in sorted(ranks):
        total_shards += verify_rank(args.root, rank, world, rank == 0)
    print(
        f"ARTIFACT_B ASSEMBLED PASS ranks={len(ranks)} shards={total_shards} root={args.root}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank")
    rank.add_argument("--root", required=True)
    rank.add_argument("--rank", type=int, required=True)
    rank.add_argument("--world", type=int, required=True)
    rank.set_defaults(func=rank_command)
    assembled = sub.add_parser("assembled")
    assembled.add_argument("--root", required=True)
    assembled.set_defaults(func=assembled_command)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"ARTIFACT_B INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
