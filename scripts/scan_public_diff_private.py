#!/usr/bin/env python3
"""Mechanical fail-closed scanner for a proposed public diff."""

import argparse
import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path


PRIVATE_PATH_PATTERNS = (
    re.compile(r"(^|/)HANDOFF_[^/]*$"),
    re.compile(r"(^|/)TRAINING_PROVENANCE\.json$"),
    re.compile(r"(^|/)instrumentation/results/"),
)
CONTENT_INDICATORS = {
    "operator_home": re.compile(rb"/(?:home|Users)/(?:mira|spark)(?:/|\b)"),
    "private_ipv4": re.compile(
        rb"(?<![0-9])(?:10\.[0-9]{1,3}(?:\.[0-9]{1,3}){2}|"
        rb"192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"
    ),
    "private_authority": re.compile(bytes.fromhex("676f7665726e65645f736674")),
    "private_corpus_identity": re.compile(bytes.fromhex("6370745f636c65616e5f6964656e746974795f7631")),
    "credential_shape": re.compile(
        b"(?:" + bytes.fromhex("6768705f") + b"|" + bytes.fromhex("6769746875625f7061745f")
        + b"|" + bytes.fromhex("736b2d") + b"[A-Za-z0-9])"
    ),
}


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def source_hashes(inventory_path):
    restricted = set()
    classifications = set()
    with open(inventory_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            classification = row["classification"]
            classifications.add(classification)
            if classification in {"private", "mixed"}:
                restricted.add(row["sha256"])
    allowed = {"public", "private", "mixed"}
    unexpected = classifications - allowed
    if unexpected:
        raise RuntimeError(f"inventory contains unresolved classifications: {sorted(unexpected)}")
    if not restricted:
        raise RuntimeError("inventory supplies no private or mixed source hashes")
    return restricted


def changed_paths(repo, base):
    tracked = git(repo, "diff", "--name-only", "--diff-filter=ACMR", base, "--").decode().splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    return sorted(set(tracked + untracked))


def added_bytes(repo, base, path, untracked):
    if untracked:
        return Path(repo, path).read_bytes()
    patch = git(repo, "diff", "--no-color", "--unified=0", base, "--", path)
    lines = []
    for line in patch.splitlines():
        if line.startswith(b"+") and not line.startswith(b"+++"):
            lines.append(line[1:])
    return b"\n".join(lines)


def scan(repo, base, inventory):
    restricted_hashes = source_hashes(inventory)
    paths = changed_paths(repo, base)
    if not paths:
        raise RuntimeError("proposed diff is empty")
    untracked_paths = set(
        git(repo, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    )
    failures = []
    scanned_bytes = 0
    for path in paths:
        if any(pattern.search(path) for pattern in PRIVATE_PATH_PATTERNS):
            failures.append((path, "private_path"))
        file_path = Path(repo, path)
        if file_path.is_symlink():
            failures.append((path, "symlink"))
            continue
        if not file_path.is_file():
            failures.append((path, "missing_or_nonregular"))
            continue
        content = file_path.read_bytes()
        scanned_bytes += len(content)
        if hashlib.sha256(content).hexdigest() in restricted_hashes:
            failures.append((path, "exact_private_or_mixed_source_blob"))
        additions = added_bytes(repo, base, path, path in untracked_paths)
        for name, pattern in CONTENT_INDICATORS.items():
            if pattern.search(additions):
                failures.append((path, name))
    if failures:
        for path, indicator in sorted(set(failures)):
            print(f"PRIVATE DIFF FAIL path={path} indicator={indicator}")
        raise RuntimeError(f"private-material scanner found {len(set(failures))} failure(s)")
    print(
        f"PRIVATE DIFF PASS files={len(paths)} bytes={scanned_bytes} "
        f"restricted_source_hashes={len(restricted_hashes)} base={base}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--inventory", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    git(repo, "rev-parse", "--verify", f"{args.base}^{{commit}}")
    scan(repo, args.base, args.inventory)


if __name__ == "__main__":
    main()
