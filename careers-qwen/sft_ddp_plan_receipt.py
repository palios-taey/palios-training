#!/usr/bin/env python3
"""Bind a production SFT corpus to the exact adaptive DDP batch plan."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_receipt(path):
    digest = hashlib.sha256()
    rows = 0
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            rows += chunk.count(b"\n")
            size += len(chunk)
    return {
        "bytes": size,
        "rows": rows,
        "sha256": digest.hexdigest(),
    }


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--ddp-trainer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq", required=True, type=int)
    parser.add_argument("--short-max", required=True, type=int)
    parser.add_argument("--mid-max", required=True, type=int)
    parser.add_argument("--short-batch", required=True, type=int)
    parser.add_argument("--mid-batch", required=True, type=int)
    parser.add_argument("--long-batch", required=True, type=int)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse()
    if args.world <= 0:
        raise ValueError("world size must be positive")
    for path in (args.trainer, args.ddp_trainer, args.corpus, args.model):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    canonical = load_module("production_sft_trainer", args.trainer)
    ddp = load_module("production_ddp_trainer", args.ddp_trainer)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = canonical.BucketSFTDataset(
        args.corpus,
        tokenizer,
        args.max_seq,
        strict_one_sample_per_row=False,
    )
    dataset_value, lengths = ddp.dataset_receipt(dataset)
    plan, full_plan, plan_sha = ddp.build_batch_plan(
        lengths,
        args.world,
        args.short_max,
        args.mid_max,
        args.short_batch,
        args.mid_batch,
        args.long_batch,
        args.seed,
        0,
    )
    if len(plan) != len(full_plan):
        raise RuntimeError("production plan is not the full adaptive schedule")

    buckets = {}
    for group in full_plan:
        value = buckets.setdefault(
            group["bucket"],
            {
                "partial_steps": 0,
                "samples": 0,
                "steps": 0,
                "tokens": 0,
            },
        )
        value["steps"] += 1
        value["samples"] += group["real_samples"]
        value["tokens"] += group["useful_tokens"]
        if group["real_samples"] < group["local_batch"] * args.world:
            value["partial_steps"] += 1

    receipt = {
        "batches": [
            args.short_batch,
            args.mid_batch,
            args.long_batch,
        ],
        "buckets": buckets,
        "corpus": corpus_receipt(args.corpus),
        "dataset": dataset_value,
        "format": "sft-ddp-plan-v1",
        "full_schedule_steps": len(full_plan),
        "max_seq": args.max_seq,
        "plan_sha256": plan_sha,
        "seed": args.seed,
        "thresholds": [args.short_max, args.mid_max],
        "world_size": args.world,
    }
    print(
        "SFT_DDP_PLAN_RECEIPT "
        + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
