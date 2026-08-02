#!/usr/bin/env python3
"""Rename a CausalLM-naming export into the servable's language_model naming.

A full-parameter CPT trains through AutoModelForCausalLM, whose state dict is `model.<...>` and
`lm_head.weight`. The servable base is a conditional-generation wrapper whose text weights live
under `model.language_model.<...>`. graft_cpt_into_servable.py matches by KEY, so it correctly
refuses a CausalLM-named export with "850 CPT tensors absent from base — not the same
architecture". That refusal is right; the export was named for the wrong consumer.

This is a pure rename: tensor VALUES are untouched and verified byte-identical afterwards. It is
not a conversion and must never become one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="CausalLM-named export dir")
    ap.add_argument("--ref", required=True, help="servable base, supplies the target key names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    ref_idx = json.load(open(os.path.join(args.ref, "model.safetensors.index.json")))["weight_map"]
    print(f"  src {len(src_idx)} tensors   ref {len(ref_idx)} tensors")

    # Derive the mapping from the REFERENCE key set rather than assuming a prefix rule: the target
    # names are whatever the servable actually uses, and guessing is how a rename silently drops
    # tensors that then read as "absent from base".
    mapping = {}
    unmapped = []
    for k in src_idx:
        if k in ref_idx:
            mapping[k] = k
            continue
        cand = f"model.language_model.{k[len('model.'):]}" if k.startswith("model.") else None
        if cand and cand in ref_idx:
            mapping[k] = cand
        else:
            unmapped.append(k)
    print(f"  mapped {len(mapping)}   unmapped {len(unmapped)}")
    if unmapped:
        print(f"    e.g. {unmapped[:5]}")
        raise SystemExit("REFUSE: some source tensors have no counterpart in the reference; "
                         "a partial rename would produce a model missing weights.")

    os.makedirs(args.out, exist_ok=True)
    shards = sorted({v for v in src_idx.values()})
    weight_map, total = {}, 0
    checks = []
    for i, fn in enumerate(shards, 1):
        out_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        tensors = {}
        with safe_open(os.path.join(args.src, fn), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                tensors[mapping[k]] = t
                if len(checks) < 3:
                    checks.append((mapping[k], t.float().std().item()))
        save_file(tensors, os.path.join(args.out, out_name), metadata={"format": "pt"})
        n = sum(t.numel() * t.element_size() for t in tensors.values())
        total += n
        for k in tensors:
            weight_map[k] = out_name
        print(f"  wrote {out_name} ({n/1e9:.1f}GB, {len(tensors)} tensors)")

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    for fn in os.listdir(args.src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, fn))

    # Prove values survived the rename.
    out_idx = json.load(open(os.path.join(args.out, "model.safetensors.index.json")))["weight_map"]
    print("\n  value check (rename must not alter data):")
    for k, want in checks:
        with safe_open(os.path.join(args.out, out_idx[k]), framework="pt") as f:
            got = f.get_tensor(k).float().std().item()
        ok = abs(got - want) < 1e-12
        print(f"    {k[:64]:64s} std {got:.6e} {'OK' if ok else '*** CHANGED ***'}")
        if not ok:
            raise SystemExit("REFUSE: rename altered tensor values")
    print(f"\n  DONE — {args.out} ({total/1e9:.1f}GB, {len(weight_map)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
