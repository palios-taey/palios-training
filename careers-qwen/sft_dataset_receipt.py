#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
import sys

from transformers import AutoTokenizer


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("production_sft_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import production trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq", required=True, type=int)
    args = parser.parse_args()

    if args.max_seq <= 256:
        raise SystemExit("--max-seq must exceed the 256-token overlap")
    for path in (args.trainer, args.corpus):
        if not os.path.isfile(path):
            raise SystemExit(f"required file is absent: {path}")
    if not os.path.isdir(args.model):
        raise SystemExit(f"model directory is absent: {args.model}")

    trainer = load_trainer(args.trainer)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    corpus_digest = hashlib.sha256()
    sample_shape_digest = hashlib.sha256()
    rows = 0
    samples = 0
    over_max_rows = 0
    max_tokens = 0
    assistant_tokens = 0
    min_assistant_tokens = None
    zero_assistant_samples = 0
    zero_assistant_details = []

    with open(args.corpus, "rb") as raw:
        for chunk in iter(lambda: raw.read(8 * 1024 * 1024), b""):
            corpus_digest.update(chunk)

    with open(args.corpus, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank SFT row at line {line_number}")
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"invalid SFT JSON at line {line_number}: {exc}"
                ) from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RuntimeError(
                    f"SFT row {line_number} has no non-empty messages list"
                )
            tools = row.get("tools")
            if tools and any(
                message.get("role") == "system"
                and "<tools>" in (message.get("content") or "")
                for message in messages
            ):
                tools = None
            try:
                input_ids, labels = trainer._tokenize_sft_pair(
                    messages,
                    tokenizer,
                    tools=tools,
                    require_assistant_labels=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"SFT tokenization failed at line {line_number}: {exc}"
                ) from exc

            row_assistant_tokens = sum(label != -100 for label in labels)
            if row_assistant_tokens == 0:
                raise RuntimeError(
                    f"SFT row {line_number} has no assistant loss tokens"
                )
            assistant_tokens += row_assistant_tokens
            rows += 1
            max_tokens = max(max_tokens, len(input_ids))
            if len(input_ids) > args.max_seq:
                over_max_rows += 1
            chunks = trainer._supervised_sft_windows(
                input_ids,
                labels,
                args.max_seq,
            )
            emitted_row_assistant_tokens = 0
            for chunk_index, (chunk_ids, chunk_labels) in enumerate(chunks):
                chunk_assistant_tokens = sum(
                    label != -100 for label in chunk_labels
                )
                emitted_row_assistant_tokens += chunk_assistant_tokens
                if chunk_assistant_tokens == 0:
                    zero_assistant_samples += 1
                    zero_assistant_details.append(
                        (
                            line_number,
                            chunk_index,
                            len(chunk_ids),
                            len(input_ids),
                        )
                    )
                min_assistant_tokens = (
                    chunk_assistant_tokens
                    if min_assistant_tokens is None
                    else min(min_assistant_tokens, chunk_assistant_tokens)
                )
                sample_shape_digest.update(
                    (
                        f"{line_number}:{len(chunk_ids)}:"
                        f"{chunk_assistant_tokens}\n"
                    ).encode("ascii")
                )
            if emitted_row_assistant_tokens != row_assistant_tokens:
                raise RuntimeError(
                    "SFT supervised-window label coverage failed at "
                    f"line {line_number}: source={row_assistant_tokens} "
                    f"emitted={emitted_row_assistant_tokens}"
                )
            samples += len(chunks)

    if rows <= 0 or samples < rows or min_assistant_tokens is None:
        raise RuntimeError(
            f"invalid SFT receipt rows={rows} samples={samples}"
        )
    for line_number, chunk_index, chunk_tokens, row_tokens in zero_assistant_details:
        print(
            "ZERO_ASSISTANT_WINDOW "
            f"line={line_number} chunk={chunk_index} "
            f"chunk_tokens={chunk_tokens} row_tokens={row_tokens}",
            file=sys.stderr,
        )
    print(
        "SFT_DATASET_RECEIPT",
        corpus_digest.hexdigest(),
        rows,
        samples,
        samples - rows,
        over_max_rows,
        max_tokens,
        assistant_tokens,
        zero_assistant_samples,
        min_assistant_tokens,
        sample_shape_digest.hexdigest(),
    )


if __name__ == "__main__":
    main()
