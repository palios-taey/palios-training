# SPEC (ref, NOT code) — 9B CPT launcher on the validated config

Implementer: conductor (to this spec). Audit: conductor-grok + infra (vs nccl_rescue_bank). Author: tutor (spec only; no code touched). Goal: the 9B CPT FSDP/accelerate launcher must carry the VALIDATED fleet NCCL/env config, with ZERO invented settings. All values are specified BY REFERENCE to existing validated files — copy verbatim, do not retype from this doc.

## 1. NCCL / env block — REPLACE the current recipe's block ENTIRELY
- **Source of truth:** `careers-qwen/launch_4node.sh` **lines 6–26** (the "validated NCCL block, NET_PLUGIN=none proven 21.7GB/s"). Infra confirms this == `nccl_rescue_bank`.
- **Action:** copy those lines VERBATIM into the CPT launcher's env section. This REPLACES the current recipe's entire NCCL/FLA/allocator block (`dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh` lines ~28–64). **Nothing from the old recipe's NCCL block carries over** — specifically the old `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, `PYTORCH_CUDA_ALLOC_CONF=...True...`, the missing SOCKET_IFNAME, and any `NCCL_NET_GDR_C2C/READ/PXN_C2C` are ALL dropped (those last three were tutor's firmware-chase inventions — NOT in the validated block, do not add them).
- **Confirm (answer to Conductor Q1):** yes, `launch_4node.sh:6–26` is the COMPLETE set. Do not merge/keep any env from the old recipe's NCCL block.

## 2. Rank pattern (Conductor Q2) — EXPLICIT POSITIONAL, like the validated launcher
- Take `RANK` as positional arg `$1` (exactly `careers-qwen/launch_4node.sh:5` `RANK=$1`). NO IP-detection (the current recipe's `MY_IP`/`ip addr | head -1` returns the management IP on these nodes → wrong network; that is the bug being removed).
- Invoked once per node with its explicit rank. Mapping (per `the-conductor/deploy/nodes.json` + the validated master): **Spark 1 = rank 0, Spark 2 = rank 1, Spark 3 = rank 2, Spark 4 = rank 3.**

## 3. Master (rail) — accelerate form of the validated `--master_addr`
- Validated source uses `--master_addr=PRIVATE_RAIL_IP_0` (rail, Spark 1 = rank 0). Accelerate equivalent: **`--main_process_ip=PRIVATE_RAIL_IP_0 --main_process_port=29500`**.
- `--machine_rank=$RANK` (from §2), `--num_machines=4 --num_processes=4`.

## 4. What STAYS VERBATIM from the current CPT recipe (`launch_cpt_phase2_qwen35_9b_fsdp.sh`)
Keep unchanged (these are the CPT-specific, non-NCCL parts — do NOT reinvent):
- `MODEL_PATH` default + the raw-multimodal-base pre-flight guard (lines ~69, 83–91).
- `SFT_DIR=/nonexistent/cpt_mode_sentinel` (CPT-mode routing, line ~76).
- `CPT_DATA` default + the stale-corpus guard (lines ~78, 92–99).
- `OUTPUT_DIR`, and the trainer knobs: `BATCH_SIZE_PER_RANK`, `GRAD_ACCUM`, `TOTAL_STEPS` (required), `SAVE_EVERY`, `SESSION_LIMIT`, `WARMUP_STEPS`, `LR`, `ADAFACTOR_CLIP_THRESHOLD` (lines ~111–130).
- The `accelerate launch ... train_fsdp_dense_9b.py` invocation shape (lines ~174–182), with the master/rank flags from §3 substituted for the IP-detection.
- **ACCEL_CONFIG:** keep the recipe's DEFAULT (`configs/fsdp_dense_9b.yaml`) — do NOT override to DDP. (FSDP is the recipe's proven path; tutor's earlier `ddp_dense_9b.yaml` override was an experiment, not the validated default.) [AUDIT FLAG for infra/grok: confirm FSDP is the intended CPT config, not DDP.]

## 5. MAX_SEQ (Conductor Q3) — default 16384
- The deployed corpus `cpt_v3_dense_9b.jsonl` is pre-chunked at ~15800 tok; the trainer's no-truncate guard REQUIRES `MAX_SEQ >= max row` (4096 fails: "CPT row exceeds max_seq: 4563>4096"). **Default `MAX_SEQ=16384`** (overridable). The wedge was NOT seq-related (it was the NCCL/heartbeat config), so 16384 is safe once §1 is applied; and with the validated `HEARTBEAT=120`, any memory issue at 16384 aborts CLEANLY (traceback) instead of wedging.

## Audit contract
grok + infra verify: (a) every §1 line == `careers-qwen/launch_4node.sh:6–26` (diff, zero substitutions); (b) master = rail `PRIVATE_RAIL_IP_0`, socket ifname = `enp1s0f0np0`, heartbeat = 120, expandable_segments:False present; (c) no `NCCL_NET_GDR_C2C/READ/PXN_C2C` added; (d) §4 CPT env/guards intact. run-cpt (tutor) unblocks only on PASS.
