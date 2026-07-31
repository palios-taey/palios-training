#!/usr/bin/env python3
"""consolidate_base.py — turn a sharded DCP base checkpoint into a SINGLE loadable file so LoRA
MODULES can branch from the frozen base (the base+adapters architecture).

WRINKLE (our checkpoints are use_collectives=False / per-rank bundles): the base DCP checkpoint is
4 self-contained per-rank bundles (__R.metadata + __R_0.distcp), scattered across the 4 nodes and
collected to one dir via checkpoint_sync.sh. Offline consolidation runs single-process (no process
group) and must merge all rank shards into the full unsharded state_dict, then save one file.

Two paths are attempted in order:
  (A) torch.distributed.checkpoint.format_utils.dcp_to_torch_save(dcp_dir, out.pt) — the built-in
      offline consolidator. Works if it accepts the per-rank __R.metadata layout.
  (B) fallback: _load_state_dict_from_keys / a manual offline load into a full state dict, then
      torch.save. Used if (A) rejects the no-global-.metadata layout.

The full model state dict is saved as out.pt ({fqn: tensor}); load it into a fresh HF model with
model.load_state_dict(...) to get the branchable base for LoRA.

Usage: consolidate_base.py <dcp_dir> <out.pt>   (dcp_dir = the collected checkpoint's dcp/ subdir)
Run on a node (needs torch + enough RAM to hold the full 54GB bf16 state dict on CPU — GB10 128GB OK).
"""
import sys, os, torch

def main():
    dcp_dir, out_path = sys.argv[1], sys.argv[2]
    if not os.path.isdir(dcp_dir):
        raise SystemExit(f"dcp_dir not found: {dcp_dir}")
    files = sorted(os.listdir(dcp_dir))
    print(f"dcp_dir contents: {files}", flush=True)

    # (A) built-in offline consolidator
    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        print("[A] trying dcp_to_torch_save ...", flush=True)
        dcp_to_torch_save(dcp_dir, out_path)
        sd = torch.load(out_path, map_location="cpu", weights_only=False)
        inner = sd.get("model", sd) if isinstance(sd, dict) else sd
        print(f"[A] SUCCESS — consolidated {len(inner)} tensors → {out_path} "
              f"({os.path.getsize(out_path)/1e9:.1f}GB)", flush=True)
        return
    except Exception as e:
        print(f"[A] dcp_to_torch_save failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    # (B) fallback: offline DCP load into an empty state dict via the checkpoint's own metadata
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
        print("[B] trying offline _load_state_dict_from_keys ...", flush=True)
        # single-process, no_dist load: reads all shards from the metadata and materializes full tensors
        state = _load_state_dict_from_keys(checkpoint_id=dcp_dir)  # returns the full state dict
        inner = state.get("model", state) if isinstance(state, dict) else state
        torch.save(inner, out_path)
        print(f"[B] SUCCESS — consolidated {len(inner)} tensors → {out_path} "
              f"({os.path.getsize(out_path)/1e9:.1f}GB)", flush=True)
        return
    except Exception as e:
        print(f"[B] fallback failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    raise SystemExit("consolidation FAILED — both paths errored; likely the per-rank (use_collectives"
                     "=False) layout needs a global-.metadata rewrite first. Next step: consult / test "
                     "dcp.load in a 1-proc group then dcp.save with use_collectives=True to one dir.")

if __name__ == "__main__":
    main()
