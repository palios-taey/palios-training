#!/usr/bin/env python3
"""RAM-only validation of an Artifact-B export — no HF write (dodges the 51GB output-disk need).

Loads the coordinated DCP export single-process into a CPU bf16 model, then compares the probe
tensors against the base model's safetensors directly. Two verdicts from the numbers:
  (A) PATH-CORRECT: model.norm ≈ base (~0.96), NOT scrambled (~0.23), no NaN → the gloo-coordinated
      export + offline no_dist load assembles weights correctly.
  (B) GATE-3 bake-diff: decoder mlp meanabs / changed-fraction vs the Chats' band.

Usage: python3 validate_export_diff.py --assembled <dir> --base <base-model-dir>
"""
import argparse, json, os, sys

PROBES = ["model.norm.weight",
          "model.layers.32.input_layernorm.weight",
          "model.layers.30.mlp.down_proj.weight",
          "model.layers.50.mlp.gate_proj.weight"]


def base_tensor(base, needle):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    ks = [k for k in idx if k.endswith(needle)]
    if not ks:
        return None, None
    k = ks[0]
    with safe_open(os.path.join(base, idx[k]), "pt") as sf:
        return k, sf.get_tensor(k).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.assembled, ".metadata")):
        sys.exit(f"ABORT: no global .metadata in {args.assembled}")

    import torch
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM

    print(f"[validate] torch {torch.__version__} | loading base CPU bf16 target", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    sd = model.state_dict()
    print(f"[validate] dcp.load(no_dist) ← {args.assembled} into {len(sd)} tensors", flush=True)
    dcp.load({"model": sd}, checkpoint_id=args.assembled, no_dist=True)
    model.load_state_dict(sd, strict=True)

    # full-model NaN/Inf scan (integrity)
    nan = sum(1 for t in model.state_dict().values() if torch.isnan(t).any() or torch.isinf(t).any())
    print(f"[validate] NaN/Inf tensors: {nan}", flush=True)

    md = model.state_dict()
    # EXACT key match — runtime state_dict keys == base safetensors index keys (same arch loaded via
    # from_pretrained). (An earlier greedy endswith fallback wrongly matched a 128-dim per-head
    # linear_attn.norm instead of the 5120-dim model.norm — never fuzzy-match FQNs.)
    print("[validate] === probe diffs (V = dcp-loaded export, B = base safetensors) ===", flush=True)
    for needle in PROBES:
        kb, tb = base_tensor(args.base, needle)
        tvraw = md.get(needle) if needle in md else md.get(needle.split("model.", 1)[-1])
        if tvraw is None or tb is None:
            print(f"{needle}: MISSING (v_in_md={needle in md} b={kb})"); continue
        tv = tvraw.float()
        if tv.shape != tb.shape:
            print(f"{needle}: SHAPE MISMATCH v={tuple(tv.shape)} b={tuple(tb.shape)}"); continue
        d = tv - tb
        changed = (d != 0).float().mean().item()
        print(f"{needle}: shape={tuple(tv.shape)} IDENTICAL={bool((d==0).all())} "
              f"maxabs={d.abs().max().item():.6e} meanabs={d.abs().mean().item():.6e} "
              f"changed_frac={changed:.6f} V_meanabs={tv.abs().mean().item():.5f} "
              f"B_meanabs={tb.abs().mean().item():.5f}", flush=True)


if __name__ == "__main__":
    main()
