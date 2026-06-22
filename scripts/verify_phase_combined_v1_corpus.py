#!/usr/bin/env python3
"""Verify canonical phase_combined_v1 corpus counts, bytes, and token limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


CANONICAL_COMBINED_SHA = "83a62ae1666897d306ecf6cd9892ccc7ac06b1242571abb912e69d99effd8ebd"  # PII+topology-scrubbed public corpus
PRE_SCRUB_ORIGINAL_COMBINED_SHA = "6b54f163c0dfc35ed7cae4637146a0a959ff33f58601922a95f3cff7641dabfd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_rows(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON") from exc
            rows.append(stripped)
    return rows


def _rendered_token_count(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _assert_count(label: str, rows: list[str], expected: int) -> None:
    actual = len(rows)
    if actual != expected:
        raise AssertionError(f"{label} rows: expected {expected}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--infra", required=True, type=Path)
    parser.add_argument("--combined", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, help="Operator-supplied tokenizer/model path or HF id")
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--expected-identity-rows", type=int, default=1378)
    parser.add_argument("--expected-infra-rows", type=int, default=947)
    parser.add_argument("--expected-combined-rows", type=int, default=2325)
    parser.add_argument("--expected-combined-sha", default=CANONICAL_COMBINED_SHA)
    parser.add_argument("--seed", type=int, default=42, help="Expected combiner shuffle seed")
    args = parser.parse_args()

    identity_rows = _load_raw_rows(args.identity)
    infra_rows = _load_raw_rows(args.infra)
    combined_rows = _load_raw_rows(args.combined)

    _assert_count("identity", identity_rows, args.expected_identity_rows)
    _assert_count("infra", infra_rows, args.expected_infra_rows)
    _assert_count("combined", combined_rows, args.expected_combined_rows)

    combined_sha = _sha256(args.combined)
    if combined_sha != args.expected_combined_sha:
        raise AssertionError(
            f"combined sha: expected {args.expected_combined_sha}, got {combined_sha}"
        )

    expected_combined_rows = identity_rows + infra_rows
    random.Random(args.seed).shuffle(expected_combined_rows)
    if expected_combined_rows != combined_rows:
        raise AssertionError(
            f"combined rows are not the seed-{args.seed} shuffle of identity+infra"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    violations = 0
    max_seen = 0
    for row_num, raw in enumerate(combined_rows, 1):
        data = json.loads(raw)
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise AssertionError(f"combined row {row_num}: missing messages list")
        count = _rendered_token_count(tokenizer, messages)
        max_seen = max(max_seen, count)
        if count > args.max_seq:
            violations += 1

    if violations:
        raise AssertionError(f"token violations: {violations} rows exceeded {args.max_seq}")

    print(
        "verified "
        f"identity={len(identity_rows)} infra={len(infra_rows)} combined={len(combined_rows)} "
        f"sha={combined_sha} max_tokens={max_seen} violations=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
