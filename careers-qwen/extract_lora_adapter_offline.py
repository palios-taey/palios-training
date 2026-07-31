#!/usr/bin/env python3
"""extract_lora_adapter_offline.py — DCP checkpoint -> PEFT adapter dir, SINGLE PROCESS.

Produces adapter_model.safetensors + adapter_config.json directly from a LoRA training
checkpoint. No cluster, no torchrun, no collective, no merged model.

WHY THIS EXISTS
---------------
The in-trainer path (save_lora_only_fsdp) calls
    get_model_state_dict(model, full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
which is a FULL-MODEL collective across all ranks. On 2026-07-27 that deadlocked: all four ranks
sat inside it for 45+ minutes at ~10W with no compute, on the first LoRA bake attempted against
the grafted multimodal base (1199 tensors, 348 of them frozen vision/mtp that the FSDP policy —
Qwen3_5DecoderLayer — does not wrap). Module 3 baked fine with identical code against an
851-tensor text-only base.

infra's reframe is what makes this the root-cause fix rather than a route around it: THAT GATHER
EXISTS TO MATERIALISE A MERGED MODEL. The serving path does not want a merged model — vllm_serve.sh
mounts a bare adapter (--enable-lora --lora-modules NAME=/models/NAME --max-lora-rank 64) on top of
the base as its own served id. If the deliverable is an adapter, gathering 1199 mostly-frozen
tensors is work nobody needs.

WHY SINGLE-PROCESS IS SAFE HERE, MEASURED
-----------------------------------------
The FSDP wrap policy is Qwen3_5DecoderLayer with NO LoRA WRAP, so the adapter tensors are not
sharded. Verified on module4_lora/final: every LoRA entry reports chunks=1 with its full logical
size ([16, 6144], [5120, 16], ...). 2816 LoRA keys of 3680 total. A single reader gets whole
tensors — this is NOT the sharded-full-model case where offline readers scramble the layout.
The script REFUSES if that assumption does not hold (see the chunks check below).

USAGE
    python3 extract_lora_adapter_offline.py --ckpt <dir with dcp/> --out <adapter dir>
        --r 16 --alpha 32 --dropout 0.05 [--dry-run]
"""

import argparse
import json
import os
import pickle
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing dcp/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--base", default="", help="base_model_name_or_path recorded in adapter_config")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    import torch.distributed.checkpoint as dcp  # noqa: F401 (registers metadata classes for pickle)
    from torch.distributed.checkpoint import FileSystemReader
    from safetensors.torch import save_file

    dcp_dir = os.path.join(a.ckpt, "dcp")
    meta_path = os.path.join(dcp_dir, "__0.metadata")
    if not os.path.exists(meta_path):
        raise SystemExit(f"ABORT: no __0.metadata in {dcp_dir}")

    with open(meta_path, "rb") as fh:
        md = pickle.load(fh)
    keys = list(md.state_dict_metadata.keys())
    # A LoRA checkpoint carries BOTH the adapter weights and the optimizer state for them.
    # On module4_lora/final: 2816 keys match "lora_", of which 2112 are `optim.*` (including
    # BytesStorageMetadata entries like `.step` that have no .size) and only 704 are the
    # `model.*` weights — which matches the trainer's own "AdamW on 704 params" line.
    # Filter on BOTH the model prefix and the presence of a tensor size; taking everything
    # matching "lora_" pulls in optimizer state and crashes on the non-tensor entries.
    sm = md.state_dict_metadata
    lora_keys = [k for k in keys
                 if "lora_" in k and k.startswith("model.") and hasattr(sm[k], "size")]
    print(f"checkpoint keys : {len(keys)}   adapter weight tensors: {len(lora_keys)} "
          f"(excluded {sum(1 for k in keys if 'lora_' in k) - len(lora_keys)} optimizer entries)")
    if not lora_keys:
        raise SystemExit("ABORT: no model.* lora_ tensors — this is not a LoRA checkpoint")

    # PRECONDITION: adapter tensors must be WHOLE, not sharded. If a future wrap policy shards
    # them, a single reader would silently return a fragment — refuse rather than emit a partial
    # adapter that loads without error and carries a quarter of the weights.
    sharded = []
    for k in lora_keys:
        ch = getattr(md.state_dict_metadata[k], "chunks", None)
        if ch is not None and len(ch) != 1:
            sharded.append((k, len(ch)))
    if sharded:
        raise SystemExit(
            f"ABORT: {len(sharded)} adapter tensors are SHARDED (e.g. {sharded[:2]}). "
            f"A single-process read would return fragments. Use a collective path for this "
            f"checkpoint, or confirm the FSDP wrap policy still excludes LoRA."
        )
    print(f"shard check     : all {len(lora_keys)} adapter tensors are whole (chunks=1)")

    if a.dry_run:
        print("--dry-run: nothing read or written.")
        print("sample keys:", lora_keys[:2])
        return 0

    # Read only the adapter tensors. Everything frozen stays on disk and is never materialised —
    # which is the entire point.
    sd = {k: torch.empty(md.state_dict_metadata[k].size, dtype=torch.bfloat16) for k in lora_keys}
    dcp.load(sd, storage_reader=FileSystemReader(dcp_dir))

    # Checkpoint keys carry the trainer's module prefix; PEFT expects to start at base_model.
    #
    # AND THE ADAPTER NAME MUST BE STRIPPED (fixed 2026-07-27). A LIVE PEFT module is named
    # `...lora_A.default.weight` — the `.default` is the ADAPTER NAME, injected because a model
    # can hold several adapters at once. A SAVED adapter file does not carry it: on load,
    # `set_peft_model_state_dict(model, sd, adapter_name="default")` inserts the name itself.
    # Passing the live name through means it gets inserted TWICE:
    #     file: ...lora_A.default.weight   ->   lookup: ...lora_A.default.default.weight
    # and the load fails with every tensor simultaneously `missing` and `unexpected` — which is
    # the signature of a NAMING mismatch, not an incomplete file.
    #
    # This shipped undetected because the only consumer so far was bake_lora_nopeft.py, which
    # does raw tensor arithmetic over the shards and never asks PEFT to resolve a module path.
    # Merging worked; RESUMING did not. The defect surfaced the first time an adapter was loaded
    # back as a starting point — i.e. the first genuinely cumulative module.
    def peft_key(k):
        i = k.find("base_model.")
        k = k[i:] if i >= 0 else k
        return k.replace(".default.weight", ".weight").replace(".default.bias", ".bias")

    out_sd, targets = {}, set()
    for k, v in sd.items():
        pk = peft_key(k)
        out_sd[pk] = v.contiguous()
        parts = pk.split(".")
        for j, p in enumerate(parts):
            if p.startswith("lora_") and j >= 1:
                targets.add(parts[j - 1])
    if not out_sd:
        raise SystemExit("ABORT: nothing extracted")

    os.makedirs(a.out, exist_ok=True)
    save_file(out_sd, os.path.join(a.out, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": a.r, "lora_alpha": a.alpha, "lora_dropout": a.dropout,
        "target_modules": sorted(targets), "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "base_model_name_or_path": a.base or None,
    }
    with open(os.path.join(a.out, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    total = sum(v.numel() * v.element_size() for v in out_sd.values())
    print(f"extracted       : {len(out_sd)} tensors, {total/1e6:.1f} MB")
    print(f"target_modules  : {sorted(targets)}")
    print(f"wrote           : {a.out}")
    print("VERIFY BEFORE HANDOFF: load the adapter and confirm it carries non-zero lora_B "
          "(lora_B initialises to zero, so an untrained or mis-read adapter is all-zero there).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
