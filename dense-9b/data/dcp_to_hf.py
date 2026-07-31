#!/usr/bin/env python3
"""dcp_to_hf.py — turn a sharded DCP base checkpoint into a from_pretrained-able HF MODEL DIRECTORY
(config + sharded safetensors + tokenizer) so LoRA modules can branch from it AND it can be served.

WHY a separate HF dir (not just base_v1.pt): the LoRA-module trainer loads its base via
from_pretrained(MODEL_PATH) (rank-split mmap), and serving (vLLM/HF) needs a model dir. A single
torch state_dict .pt is not from_pretrained-able. This converter produces the servable/branchable form.

Runs on a Spark NODE (needs the qwen3_5 modeling in transformers + the base model dir for text_config
+ tokenizer + ~54GB CPU RAM to materialize the bf16 state dict — GB10 128GB OK). Single-process,
no process group: offline dcp.load into a real CPU model, then save_pretrained.

The DCP checkpoint is our per-rank use_collectives=False layout (4 self-contained bundles
__0..__3.metadata + __N_0.distcp); all 4 must be gathered into <dcp_dir> first. Offline dcp.load reads
every shard from the per-rank metadata (same mechanism as dcp_to_torch_save, already proven path A).

Usage: dcp_to_hf.py <dcp_dir> <base_model_dir_for_config_and_tokenizer> <out_hf_dir>
"""
import sys, os, shutil, torch
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def main():
    dcp_dir, base_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    for p in (dcp_dir, base_dir):
        if not os.path.isdir(p):
            raise SystemExit(f"not a dir: {p}")
    print(f"dcp_dir contents: {sorted(os.listdir(dcp_dir))}", flush=True)

    # text_config drives the CausalLM (the base is a Qwen3_5ForConditionalGeneration wrapper whose
    # text params live under text_config; the trained/saved state dict is the text CausalLM).
    cfg = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    tcfg = cfg.get_text_config()
    print(f"text_config: vocab={tcfg.vocab_size} hidden={tcfg.hidden_size} "
          f"layers={tcfg.num_hidden_layers} interm={tcfg.intermediate_size}", flush=True)

    print("instantiating Qwen3_5ForCausalLM on CPU (materializes ~54GB) ...", flush=True)
    model = Qwen3_5ForCausalLM(tcfg)
    model = model.to(torch.bfloat16)

    # offline single-process DCP load: fills the model's existing tensors in place from all rank shards
    sd = model.state_dict()
    print(f"offline dcp.load into {len(sd)} tensors ...", flush=True)
    dcp.load({"model": sd}, checkpoint_id=dcp_dir)
    model.load_state_dict(sd, strict=True)
    print("state_dict loaded strict=True", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    print(f"save_pretrained → {out_dir} (safetensors shards) ...", flush=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")

    # tokenizer + any aux files for a complete servable dir
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "special_tokens_map.json", "vocab.json", "merges.txt", "generation_config.json"):
        src = os.path.join(base_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
                if os.path.isfile(os.path.join(out_dir, f)))
    print(f"DONE — HF model dir at {out_dir} ({total/1e9:.1f}GB, {len(os.listdir(out_dir))} files)",
          flush=True)


if __name__ == "__main__":
    main()
