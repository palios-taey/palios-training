"""Multi-node DCP node-local probe. MODE=save|load, DIR=node-local dcp dir.
save: each rank writes ONLY its own shard to its node-local DIR (+ rank0 writes .metadata).
load: each rank reads from its node-local DIR (which — after Mira scatters .metadata — holds
      ONLY that rank's own __R_0.distcp + a copy of .metadata). If load succeeds, DCP same-world
      resume reads ONLY-LOCAL → node-local sharded checkpoints are viable. Ground truth for Q2.
Backend gloo/CPU (no GPU, no reboot needed). Run via torchrun multi-node."""
import os, sys, glob
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
import torch.distributed.checkpoint as dcp

mode = sys.argv[1]           # save | load
dcp_dir = sys.argv[2]
dist.init_process_group("gloo")
rank = dist.get_rank(); world = dist.get_world_size()
mesh = init_device_mesh("cpu", (world,))
full = torch.arange(world * 8 * 16, dtype=torch.float32).reshape(world * 8, 16)  # deterministic

if mode == "save":
    dt = distribute_tensor(full, mesh, [Shard(0)])
    os.makedirs(dcp_dir, exist_ok=True)
    dcp.save({"w": dt}, checkpoint_id=dcp_dir)
    dist.barrier()
    local_files = sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/*"))
    hidden = sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/.*"))
    print(f"[SAVE rank{rank}] node-local dir now holds: {local_files} + hidden {hidden}", flush=True)
else:  # load — each node holds ONLY its own shard + a scattered copy of .metadata
    have = sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/*")) + \
           sorted(os.path.basename(f) for f in glob.glob(dcp_dir + "/.*"))
    print(f"[LOAD rank{rank}] reading from node-local dir holding: {have}", flush=True)
    dt2 = distribute_tensor(torch.zeros(world * 8, 16), mesh, [Shard(0)])
    try:
        dcp.load({"w": dt2}, checkpoint_id=dcp_dir)
        expected = full[rank * 8:(rank + 1) * 8]
        ok = torch.allclose(dt2.to_local(), expected)
        print(f"[LOAD rank{rank}] SUCCESS — local shard correct: {ok}", flush=True)
    except Exception as e:
        print(f"[LOAD rank{rank}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
dist.barrier()
dist.destroy_process_group()
