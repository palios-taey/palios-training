#!/usr/bin/env python3
"""Bake q0-3 Condition-D correctly: load the FULL multimodal Qwen3_5ForConditionalGeneration
(NOT AutoModelForCausalLM — that drops the 348 vision/MTP tensors), apply the D LoRA to the text
decoder, merge, save all 1199 keys. Runs inside the vLLM Jetson image (has the Qwen3_5 classes)."""
import torch, os, json, sys
from transformers import AutoConfig, AutoTokenizer
import transformers
from peft import PeftModel
BASE=os.environ.get("BASE","/base"); ADAPTER=os.environ.get("ADAPTER","/adapter"); OUT=os.environ.get("OUT","/out")
cfg=AutoConfig.from_pretrained(BASE)
arch=cfg.architectures[0]  # Qwen3_5ForConditionalGeneration
print("loading FULL model class:", arch, flush=True)
cls=getattr(transformers, arch)
model=cls.from_pretrained(BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print("applying D LoRA + merge...", flush=True)
model=PeftModel.from_pretrained(model, ADAPTER)
model=model.merge_and_unload()
print("saving merged (all keys)...", flush=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="10GB")
for f in os.listdir(BASE):
    if f.endswith((".json",".txt",".jinja")) and f!="model.safetensors.index.json":
        import shutil; shutil.copy(os.path.join(BASE,f), os.path.join(OUT,f))
idx=json.load(open(os.path.join(OUT,"model.safetensors.index.json")))["weight_map"]
c=json.load(open(os.path.join(OUT,"config.json")))
print(f"BAKED: {len(idx)} keys | model_type={c.get('model_type')} arch={c.get('architectures')}", flush=True)
