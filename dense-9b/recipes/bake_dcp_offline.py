#!/usr/bin/env python3
"""Offline Artifact-B → HF converter (UNANIMOUS Chats ruling, BAKE_ARCHITECTURE_27b.md 2026-07-14).

Runs OFF the 4-rank training cluster (single process, no multi-node collective → the gather-wedge
class cannot occur). Consumes the portable model-only COORDINATED DCP written by the trainer's
EXPORT_DCP mode (each rank's *.distcp shard + the single global .metadata, collected into ONE dir),
loads it single-process into a CPU-allocated bf16 model, and save_pretrained → servable HF.

HARD requirement (Horizon): run where torch == the TRAINING torch (2.10.0) and the exact
transformers/model-code revision are installed — DCP gives NO cross-version compatibility guarantee.
Intended host: Thor1 (infra-offered; aarch64 + serving stack) or any node with the pinned stack.
NEVER create fp32 masters — bf16 params stay bf16.

Collection (do BEFORE running this): rsync every rank's *.distcp + the rank0 .metadata + all
manifest.rankN.json into <assembled_dir> on the convert host; verify each file's sha256 against its
manifest; confirm READY.rankN present for every rank and .metadata present. Only then convert.

Usage:
  python3 bake_dcp_offline.py --assembled <dir-with-shards+.metadata> \
      --base /path/to/Qwen3.6-27B --out /path/to/hf_out [--verify-manifests]
"""
import argparse, glob, hashlib, json, os, sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifests(assembled):
    ok = True
    metas = glob.glob(os.path.join(assembled, "manifest.rank*.json"))
    if not metas:
        print("[convert] MISSING manifest.rank*.json", flush=True)
        return False
    worlds = set()
    ranks = set()
    for m in sorted(metas):
        man = json.load(open(m))
        worlds.add(man["world"])
        ranks.add(man["rank"])
        for f in man["files"]:
            p = os.path.join(assembled, f["name"])
            if not os.path.exists(p):
                print(f"[convert] MISSING shard {f['name']} (rank {man['rank']})"); ok = False; continue
            h = sha256_file(p)
            if h != f["sha256"]:
                print(f"[convert] SHA MISMATCH {f['name']}: {h[:16]} != {f['sha256'][:16]}"); ok = False
            elif os.path.getsize(p) != f["bytes"]:
                print(f"[convert] SIZE MISMATCH {f['name']}"); ok = False
    if len(worlds) != 1:
        print(f"[convert] WORLD MISMATCH in manifests: {sorted(worlds)}")
        ok = False
    elif ranks != set(range(next(iter(worlds)))):
        print(f"[convert] RANK SET MISMATCH: ranks={sorted(ranks)} world={next(iter(worlds))}")
        ok = False
    ready = {
        int(os.path.basename(path).removeprefix("READY.rank"))
        for path in glob.glob(os.path.join(assembled, "READY.rank*"))
    }
    if ready != ranks:
        print(f"[convert] READY SET MISMATCH: ready={sorted(ready)} ranks={sorted(ranks)}")
        ok = False
    if ok:
        print(f"[convert] manifests VERIFIED ({len(metas)} ranks, all sha256 match)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", required=True, help="dir with collected *.distcp + global .metadata")
    ap.add_argument("--base", help="base model dir (for config/architecture/tokenizer)")
    ap.add_argument("--out")
    ap.add_argument("--verify-manifests", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.assembled, ".metadata")):
        sys.exit(f"ABORT: no global .metadata in {args.assembled} — this is NOT a coordinated "
                 f"(use_collectives=True) export; offline reassembly of per-rank bundles is proven broken.")
    if args.verify_manifests and not verify_manifests(args.assembled):
        sys.exit("ABORT: manifest verification failed — do not convert a corrupt/incomplete collection.")
    if args.verify_only:
        print(f"[convert] VERIFY ONLY PASS → {args.assembled}", flush=True)
        return 0
    if not args.base or not args.out:
        ap.error("--base and --out are required unless --verify-only is set")

    import torch
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"[convert] torch {torch.__version__} | allocating CPU bf16 model from {args.base}", flush=True)
    cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    # allocate the ordinary unsharded model on CPU in bf16 (NO fp32 masters)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    sd = model.state_dict()  # destination tensors must be pre-allocated for dcp.load

    print(f"[convert] dcp.load(no_dist=True) ← {args.assembled} into {len(sd)} tensors", flush=True)
    dcp.load({"model": sd}, checkpoint_id=args.assembled, no_dist=True)
    model.load_state_dict(sd, strict=True)

    # integrity: no NaN/Inf, dtype preserved
    bad = 0
    for n, t in model.state_dict().items():
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[convert] NaN/Inf in {n}"); bad += 1
    if bad:
        sys.exit(f"ABORT: {bad} tensors have NaN/Inf — do not ship.")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    try:
        AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"[convert] WARN tokenizer copy: {e}")
    # carry config/generation_config/chat_template (Horizon: pin exact serving config)
    cfg.save_pretrained(args.out)
    with open(os.path.join(args.out, "CONVERT_COMPLETE"), "w") as marker:
        marker.write("bake_dcp_offline.py completed successfully\n")
    print(f"[convert] DONE → {args.out} (bf16 HF, safetensors) | "
          f"run the golden parity / gate_diff check before trusting for production.", flush=True)


if __name__ == "__main__":
    main()
