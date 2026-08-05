# Supervised CPT ranks

`palios-cpt-rank@.service` makes each node-side launcher a system service instead of a child of
SSH or tmux. The existing launcher remains the only source for NCCL, allocator, rank, model, and
training behavior. The service adds process ownership, persistent journald, and a stop/status
surface.

The unit is a template. `start_cpt_rank_service.sh` substitutes the target checkout, run user,
home, and per-run environment-file path at install time. No node address or operator path is
committed. `run_4node_27b_cpt.sh` sends its existing `RUN_ENV` allowlist on stdin, so moving the
process under systemd does not create a second set of training defaults.

`NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`, and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` are required
caller-supplied manifest values with no service defaults. The live-process gate therefore detects
any inner-launcher overwrite of the ruled HCA/port, GDR, or heartbeat selection instead of trusting
either layer in isolation.

The unit is deliberately started, not enabled, and has `Restart=no`. A node reboot or any rank
failure requires the four-node orchestrator to stop the whole job, re-establish the pre-launch
gates and rank order, and relaunch all ranks together. One rank must never independently join the
old process group or rendezvous.

`LimitMEMLOCK=infinity` preserves the RDMA registration capacity available to the normal Spark
shell. The starter refuses to install a rendered unit without that setting; systemd's default
8 MB limit is insufficient for NCCL completion queues, queue pairs, and registered memory.

## What remains unchanged

- rank 0 starts first and receives the existing 12-second rendezvous lead;
- `launch_cpt_qwen36_27b_fsdp.sh` still exports the validated NCCL block;
- current output is still written to `${SPARK_HOME}/cpt27b_logs/rN.log`;
- an older rank log is still timestamp-rotated before a new attempt;
- the same output is also present under `journalctl -u palios-cpt-rank@N.service`.

The flat log is retained because the production monitors consume it. Journald is additive and is
made persistent by the node-side installer before the unit starts.

## Mechanical gate before deployment

Render the template with non-operator fixture paths, then verify it with the target host's
`systemd-analyze verify`. Run `bash -n` on all three shell files. The orchestrator must remain a
syntax-clean Bash script.

## Production gates after launch

The launch receipt must report an active unit and nonzero `MainPID`. Once trainer liveness reaches
4/4, the orchestrator automatically runs the verifier on every rank before it can accept an
optimizer step. Any mismatch stops and confirms every attempted rank unit. The commands below are
the manual diagnostic form of the same receipt:

```bash
sudo systemctl show palios-cpt-rank@N.service \
  --property=ActiveState,SubState,MainPID,ControlGroup
bash dense-9b/recipes/systemd/verify_cpt_rank_process_env.sh N
```

The environment gate reads the trainer under the unit's live cgroup and compares every `KEY=VALUE`
in the installed `cpt-rank.env` with the actual process environment. It also requires the
launcher-owned allocator setting `expandable_segments:False`. A value in the unit or environment
file is not this receipt; the live process must match the complete installed run manifest.

To prove session independence, record `MainPID`, end the SSH session that started the unit, open a
new session, and require the same `MainPID` to remain active. This checks the property that failed
under tmux instead of treating `systemctl start` as evidence.

## Stop and inspect

```bash
sudo systemctl stop palios-cpt-rank@N.service
sudo systemctl status --no-pager palios-cpt-rank@N.service
sudo journalctl -u palios-cpt-rank@N.service --no-pager
```

`KillMode=control-group` stops the wrapper, torchrun agent, trainer, and log tee together. Units
never restart independently. After any rank failure, stop every rank and repeat the whole-job gates
before relaunch.
