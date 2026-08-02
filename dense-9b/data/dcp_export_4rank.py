#!/usr/bin/env python3
"""Export a per-rank DCP checkpoint to a servable HF directory, using a REAL process group.

WHY THIS EXISTS. `dcp_to_hf.py` loads the same checkpoint in a single process and produces a
RANDOMLY-INITIALISED model while printing DONE. Measured 2026-08-02: its output sits at 0.908 of
fresh-random-init std, where every genuinely trained model — including the Qwen3.6-27B foundation
used as a control — sits at 0.543.

The mechanism, so nobody re-derives it: our checkpoints are written per-rank with
`use_collectives=False`, giving `__0..__3.metadata` + `__N_0.distcp` and NO global `.metadata`.
A single-process `dcp.load` finds no global metadata, reads nothing, and returns without error.
`load_state_dict(strict=True)` then PASSES, because strict validates key names and shapes — never
whether a value was written. Silent, successful, and wrong.

THE FIX is to use the path that provably works: resume reads these bundles every session, with a
real 4-rank process group where each rank loads ITS OWN shard. This runs under torchrun with the
same world size the checkpoint was written at, loads into an FSDP-sharded model, then gathers full
tensors on rank 0 and calls save_pretrained.

Run with:
  torchrun --nnodes=4 --node_rank=N --nproc_per_node=1 --master_addr=<rail> --master_port=29501 \
      dcp_export_4rank.py --dcp <dir> --base <servable_dir> --out <hf_out_dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FSDPModule, fully_shard
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"  {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dcp", required=True, help="checkpoint dir, per-rank bundles, LOCAL to each node")
    ap.add_argument("--base", required=True, help="servable dir supplying config + tokenizer")
    ap.add_argument("--out", required=True, help="output HF dir (written by rank 0 only)")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)
    log(f"world={world} rank={rank}")

    cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True).get_text_config()
    log(f"text_config: vocab={cfg.vocab_size} hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")

    # Build the model in the SAME sharded shape the checkpoint was written from. dcp.load matches
    # by key AND by shard layout; a differently-shaped model is exactly how a load silently
    # becomes a no-op.
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    log("model built and fully_shard'd to match the checkpoint's layout")

    sd = {"model": model.state_dict()}
    log(f"dcp.load into {len(sd['model'])} tensors from {args.dcp} ...")
    dcp.load(sd, checkpoint_id=args.dcp)
    model.load_state_dict(sd["model"], strict=True)
    dist.barrier()
    log("load complete on all ranks")

    # PROVE the load populated something before writing 54GB. strict=True cannot do this: it
    # validates names and shapes, which is precisely why the single-process path shipped noise.
    probe = None
    for name, p in model.named_parameters():
        if name.endswith("layers.32.mlp.down_proj.weight"):
            probe = p
            break
    if probe is not None:
        full = probe.full_tensor() if hasattr(probe, "full_tensor") else probe
        std = full.float().std().item()
        ratio = std / 2.0e-2  # fresh-init std for this architecture, measured
        log(f"LOAD PROBE layers.32.mlp.down_proj.weight: std={std:.4e} ratio_to_init={ratio:.3f}")
        if ratio > 0.85:
            raise RuntimeError(
                f"REFUSE: loaded weights sit at {ratio:.3f} of random init — the load did not "
                f"populate. This is the dcp_to_hf.py failure mode; do not write the artifact."
            )
        log("LOAD PROBE PASS — weights are far from init, the load populated real values")

    # full_tensor() is an ALL-GATHER: every rank must call it or the ones that do hang forever
    # waiting for peers that skipped it. Gathering inside `if rank == 0` deadlocked at SeqNum=9
    # and the NCCL watchdog killed the run. All ranks participate; only rank 0 keeps the result.
    log(f"gathering full tensors (collective — all ranks participate) -> {args.out}")
    full_sd = {}
    for name, p in model.state_dict().items():
        t = p.full_tensor() if hasattr(p, "full_tensor") else p
        if rank == 0:
            full_sd[name] = t.cpu()
        del t
    dist.barrier()
    log("gather complete on all ranks")

    if rank == 0:
        # Free the sharded model first: it is no longer needed and holds 13.5GB.
        del model
        gc.collect()

        # Write shards DIRECTLY from the gathered state dict. The previous version instantiated a
        # second full Qwen3_5ForCausalLM purely to call save_pretrained, which needs another 54GB:
        #   13.5 sharded + 54 gathered + 54 second model = 121.5GB against 119GB available.
        # rank0 was OOM-killed mid-save, leaving an empty output directory and no traceback.
        # The gathered state dict already IS the model; a second copy buys nothing but the API.
        from safetensors.torch import save_file

        os.makedirs(args.out, exist_ok=True)
        limit = 5 * 1000**3
        shards: list[dict] = [{}]
        sizes = [0]
        for name in sorted(full_sd):
            t = full_sd[name].contiguous()
            n = t.numel() * t.element_size()
            if sizes[-1] + n > limit and shards[-1]:
                shards.append({})
                sizes.append(0)
            shards[-1][name] = t
            sizes[-1] += n

        total_bytes = sum(sizes)
        weight_map = {}
        for i, shard in enumerate(shards, 1):
            fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, os.path.join(args.out, fn), metadata={"format": "pt"})
            for k in shard:
                weight_map[k] = fn
            log(f"  wrote {fn} ({sizes[i-1]/1e9:.1f}GB, {len(shard)} tensors)")

        with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)
        cfg.save_pretrained(args.out)

        for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "special_tokens_map.json", "vocab.json", "merges.txt", "generation_config.json"):
            src = os.path.join(args.base, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, fn))
        total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
                    if os.path.isfile(os.path.join(args.out, f)))
        log(f"DONE — {args.out} ({total/1e9:.1f}GB, {len(os.listdir(args.out))} files, "
            f"{len(weight_map)} tensors)")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
