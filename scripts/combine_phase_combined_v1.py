#!/usr/bin/env python3
"""Build phase_combined_v1 by concatenating identity + infra and shuffling."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _load_jsonl_lines(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(line.rstrip("\n"))
    return rows


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate phase 1 constitutional + infra gated corpora and shuffle with seed 42.",
    )
    parser.add_argument("--identity", required=True, type=Path, help="phase1_constitutional_gated.jsonl")
    parser.add_argument("--infra", required=True, type=Path, help="phase1_infra_v2_gated.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="combined_v1_gated.jsonl destination")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed; canonical value is 42")
    args = parser.parse_args()

    rows = _load_jsonl_lines(args.identity) + _load_jsonl_lines(args.infra)
    random.Random(args.seed).shuffle(rows)
    _write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
