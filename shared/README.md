# shared — cross-cutting infra (to be extracted as the lines converge)

This directory is a placeholder for infrastructure patterns that are **common to both training lines** ([`../dense-9b/`](../dense-9b/) and [`../moe-35b/`](../moe-35b/)) but currently live **embedded per-trainer** rather than as standalone shared modules.

> **Status: patterns identified, extraction pending.** Nothing is extracted here yet. The patterns below recur across the trainers; they will be pulled into standalone modules once the two lines have converged enough to share a stable interface. Until then, each trainer carries its own copy — read the trainer to see the authoritative version.

## Patterns currently duplicated across trainers

- **CPU-load + FSDP `sync_module_states`** — load weights on CPU (UMA-friendly) then let FSDP broadcast/shard with `sync_module_states=True`, rather than loading sharded onto each rank. Avoids the page-cache doubling / UMA thrash on CUDA load.
- **`summon_full_params` checkpoint save** — gather the full (unsharded) parameters under `FSDP.summon_full_params` before writing the unified safetensors, instead of DCP sharded save (which hit a PEFT `KeyError` with `ignore_frozen_params`).
- **The NCCL env block** — the dual-rail ConnectX-7 fabric config (`NCCL_IB_HCA`, `NCCL_IB_TC`, `NCCL_IB_TIMEOUT`, `NCCL_NET_GDR_LEVEL`, `NCCL_SOCKET_IFNAME`, …) is copied verbatim into the head of nearly every recipe. This is the single most-duplicated block and the first candidate for a sourced `shared/nccl_env.sh`.
- **The tool-call format-gate concept** — the prelaunch check that the inference template and the training format agree (the dense line's [`../dense-9b/inference/toolcall_format_gate.py`](../dense-9b/inference/toolcall_format_gate.py) is the concrete instance; the concept generalizes to any format-sensitive line).

## Why not extracted yet

Premature extraction would couple two lines that are still moving at different speeds (the MoE line has shipped; the dense line has not — see each line's README). Extracting now risks a shared module that fits neither cleanly. The honest state: the duplication is known and acknowledged, and the convergence point is the trigger to refactor.
