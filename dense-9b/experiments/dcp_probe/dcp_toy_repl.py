"""DCP node-local probe WITH A REPLICATED TENSOR (the failure mode the first toy missed).
The real 27B has replicated params/buffers; default collective save DEDUPLICATES them across ranks
(saved in only one rank's file) → cross-rank read on load → FileNotFoundError on a no-shared-FS cluster.
This tests: save (use_collectives from env) to NODE-LOCAL dir, then load reading ONLY the local dir.
USE_COLLECTIVES=0 (self-contained per-rank __R.metadata bundle) should load only-local; =1 should FAIL.
Run multi-node via torchrun, 1 rank/node, DIR node-local. Args: MODE(save|load) DIR"""
import os, sys, glob
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
import torch.distributed.checkpoint as dcp

mode = sys.argv[1]; dcp_dir = sys.argv[2]
use_coll = os.environ.get("USE_COLLECTIVES", "0") == "1"
dist.init_process_group("gloo")
rank = dist.get_rank(); world = dist.get_world_size()
mesh = init_device_mesh("cpu", (world,))
# a SHARDED tensor (each rank owns a slice) + a REPLICATED tensor (every rank has the full copy)
sharded_full = torch.arange(world * 8 * 16, dtype=torch.float32).reshape(world * 8, 16)
repl_full = torch.arange(64, dtype=torch.float32).reshape(8, 8) * 3.0
def build():
    return {"sharded": distribute_tensor(sharded_full, mesh, [Shard(0)]),
            "replicated": distribute_tensor(repl_full, mesh, [Replicate()])}

if mode == "save":
    os.makedirs(dcp_dir, exist_ok=True)
    dcp.save(build(), checkpoint_id=dcp_dir, use_collectives=use_coll)
    dist.barrier()
    here = sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/*") + glob.glob(dcp_dir + "/.*"))
    print(f"[SAVE rank{rank} use_collectives={use_coll}] node-local dir: {here}", flush=True)
else:
    here = sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/*") + glob.glob(dcp_dir + "/.*"))
    print(f"[LOAD rank{rank}] node-local dir holds ONLY: {here}", flush=True)
    sd = {"sharded": distribute_tensor(torch.zeros(world * 8, 16), mesh, [Shard(0)]),
          "replicated": distribute_tensor(torch.zeros(8, 8), mesh, [Replicate()])}
    try:
        dcp.load(sd, checkpoint_id=dcp_dir)
        ok_s = torch.allclose(sd["sharded"].to_local(), sharded_full[rank*8:(rank+1)*8])
        ok_r = torch.allclose(sd["replicated"].to_local(), repl_full)
        print(f"[LOAD rank{rank}] SUCCESS sharded_ok={ok_s} replicated_ok={ok_r}", flush=True)
    except Exception as e:
        print(f"[LOAD rank{rank}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
dist.barrier(); dist.destroy_process_group()
