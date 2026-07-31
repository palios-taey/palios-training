#!/usr/bin/env python3
"""Quality gate for canonical 35B SFT corpora.

The canonical reproduction path uses `--pack-mode exact`: long conversations are
first chunked with the historical head/body/tail strategy, then adjacent rows are
merged only when the fully rendered chat-template token count stays within
`--max-seq`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


@dataclass(frozen=True)
class GateConfig:
    max_seq: int
    pack_target: int
    anchor: int
    overlap: int
    min_tail_tokens: int
    pack_mode: str


def rendered_token_count(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _messages_to_text(messages: list[dict]) -> str:
    out: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        out.append(f"<|{role}|>\n{content}")
    return "\n".join(out)


def _text_to_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Continue the constitutional training document faithfully."},
        {"role": "assistant", "content": text},
    ]


def chunk_document(
    text: str,
    tokenizer,
    *,
    source_file: str = "chunk",
    tier: str = "chunked",
    max_tokens: int = 8192,
    anchor_size: int = 256,
    overlap_tokens: int = 256,
    min_tail_tokens: int = 256,
) -> list[dict]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [{"text": text, "source": source_file, "tier": tier, "chunk": 0}]

    if max_tokens <= (2 * anchor_size + 1):
        raise ValueError("max_tokens must leave room for head/body/tail chunking")

    head_tokens = tokens[:anchor_size]
    tail_tokens = tokens[-anchor_size:]
    body_tokens = tokens[anchor_size:-anchor_size]
    body_capacity = max_tokens - (2 * anchor_size) - 1
    overlap = min(overlap_tokens, body_capacity // 2)
    stride = body_capacity - overlap
    if stride <= 0:
        raise ValueError("chunk stride must be positive")

    chunks: list[dict] = []
    for start in range(0, len(body_tokens), stride):
        body = body_tokens[start:start + body_capacity]
        if len(body) < min_tail_tokens and chunks:
            break

        head_text = tokenizer.decode(head_tokens, skip_special_tokens=False)
        body_text = tokenizer.decode(body, skip_special_tokens=False)
        tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=False)
        chunk_text = (
            f"[DOCUMENT: {os.path.basename(source_file)} | CHUNK {len(chunks)} | TIER: {tier}]\n"
            f"{head_text}\n"
            "[...continued...]\n"
            f"{body_text}\n"
            "[...end section...]\n"
            f"{tail_text}"
        )
        chunks.append(
            {
                "text": chunk_text,
                "source": source_file,
                "tier": tier,
                "chunk": len(chunks),
            }
        )

    return chunks


def normalize_item(item: dict) -> list[dict]:
    if isinstance(item.get("messages"), list):
        return item["messages"]
    if isinstance(item.get("text"), str):
        return _text_to_messages(item["text"])
    if {"prompt", "chosen"}.issubset(item):
        return [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
    raise ValueError("row must contain messages, text, or prompt+chosen")


def pack_items_exact(items: Iterable[list[dict]], tokenizer, pack_target: int) -> list[list[dict]]:
    packed: list[list[dict]] = []
    current: list[dict] = []

    for messages in items:
        if not current:
            current = list(messages)
            continue

        candidate = current + list(messages)
        if rendered_token_count(tokenizer, candidate) <= pack_target:
            current = candidate
        else:
            packed.append(current)
            current = list(messages)

    if current:
        packed.append(current)
    return packed


def pack_items_fast(items: Iterable[list[dict]], tokenizer, pack_target: int) -> list[list[dict]]:
    packed: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    budget = int(pack_target * 0.90)

    for messages in items:
        count = rendered_token_count(tokenizer, messages)
        if current and current_tokens + count > budget:
            packed.append(current)
            current = []
            current_tokens = 0
        current.extend(messages)
        current_tokens += count

    if current:
        packed.append(current)
    return packed


def _iter_input_rows(input_glob: str) -> Iterable[tuple[Path, int, dict]]:
    for path_str in sorted(glob.glob(input_glob)):
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield path, line_num, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON") from exc


def process(input_glob: str, output_path: Path, tokenizer, cfg: GateConfig) -> dict:
    messages: list[list[dict]] = []
    chunked = 0

    for path, line_num, row in _iter_input_rows(input_glob):
        row_messages = normalize_item(row)
        count = rendered_token_count(tokenizer, row_messages)
        if count <= cfg.max_seq:
            messages.append(row_messages)
            continue

        text = _messages_to_text(row_messages)
        for chunk in chunk_document(
            text,
            tokenizer,
            source_file=f"{path.name}:{line_num}",
            tier="chunked",
            max_tokens=cfg.max_seq,
            anchor_size=cfg.anchor,
            overlap_tokens=cfg.overlap,
            min_tail_tokens=cfg.min_tail_tokens,
        ):
            messages.append(_text_to_messages(chunk["text"]))
            chunked += 1

    if cfg.pack_mode == "exact":
        packed = pack_items_exact(messages, tokenizer, cfg.pack_target)
    elif cfg.pack_mode == "fast":
        packed = pack_items_fast(messages, tokenizer, cfg.pack_target)
    else:
        raise ValueError(f"unknown pack mode: {cfg.pack_mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_seen = 0
    violations = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in packed:
            count = rendered_token_count(tokenizer, row)
            max_seen = max(max_seen, count)
            if count > cfg.max_seq:
                violations += 1
            handle.write(json.dumps({"messages": row}, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "input_items": len(messages),
        "chunked_items": chunked,
        "packed_rows": len(packed),
        "max_seen": max_seen,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Input JSONL glob")
    parser.add_argument("--output", required=True, type=Path, help="Output gated JSONL")
    parser.add_argument("--tokenizer", required=True, help="Operator-supplied tokenizer/model path or HF id")
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--pack-target", type=int, default=8192)
    parser.add_argument("--anchor", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-tail-tokens", type=int, default=256)
    parser.add_argument("--pack-mode", choices=("exact", "fast"), default="exact")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cfg = GateConfig(
        max_seq=args.max_seq,
        pack_target=args.pack_target,
        anchor=args.anchor,
        overlap=args.overlap,
        min_tail_tokens=args.min_tail_tokens,
        pack_mode=args.pack_mode,
    )
    stats = process(args.input_glob, args.output, tokenizer, cfg)
    print(json.dumps(stats, sort_keys=True))
    if stats["violations"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
