---
type: consultation
to: family
subject: 4-node GB10 FSDP CPT — silent hard-hang RCA
available_context_inventory:
  - hardware/versions (GB10 ×4, ConnectX-7 RoCE, kernel/driver/NCCL/torch)
  - failure matrix across seq-length × node-count
  - raw UDP telemetry capture of the wedging node into the freeze (mem/PSI/temp/power/RoCE-pause)
  - all-nodes telemetry snapshot at the freeze instant
  - netconsole result (kernel printk off-box)
  - verbatim NCCL env config
  - verbatim FSDP/accelerate config
  - verbatim RoCE NIC config (ethtool -a, mlnx_qos PFC/DSCP/buffers)
  - what we've tried + hard constraints
  - our own current hypothesis (labeled Inferred — to be challenged)
---

# Consultation — silent hard-hang of a node during 4-node FSDP CPT (raw capture + full configs)

Reach your own conclusion from the raw data below. Our hypothesis is in "Prior proposal" and may be wrong — challenge it.

## Problem
During 4-node FSDP full-parameter CPT of Qwen3.5-9B on 4× DGX Spark (GB10), one node (rank1 = .80) goes
**hard-down within the first optimizer step** — unreachable on BOTH NICs ("no route to host"), recoverable
ONLY by a physical power cycle. No kernel panic/oops; the SBSA watchdog is armed but never fires. We need the
mechanism and the fix.

## Ground truth
**[Observed] Hardware/versions:** 4× DGX Spark (GB10, Grace-Blackwell aarch64, sm_121, 128GB unified mem). Inter-node:
2× ConnectX-7 RoCE (~200Gb), no NVLink. Mgmt 10.0.0.{68,80,12,19}; fabric 192.168.100.{10,11,12,13}+192.168.101.x.
Kernel 6.11.0-1016-nvidia, driver 580.95.05, NCCL 2.28.9, torch 2.10.0+cu130, CUDA 13, transformers 5.3.0.
Workload: FSDP FULL_SHARD bf16, 4 nodes ×1 GPU, accelerate launch, Adafactor lr=2e-5, batch1/rank ga4.

**[Observed] Failure matrix (seq × nodes → outcome):**
- 2048, 3-node: trained to step 1700, no wedge.
- 4096, 4-node: rendezvoused, reached "Starting steps", .80 hard-wedged before step 1.
- 16384, 4-node: .80 AND .19 hard-wedged before step 1.
- (Historical 16384 4-node *SFT* ran 4367 steps/9h — SFT seqs mostly << MAX; CPT corpus is uniformly near-MAX, ~3.75× per-step memory/traffic.)

**[Observed] Telemetry of .80 INTO the freeze** (UDP off-box every 0.5s; survives the hang). freeMB=MemAvailable, PSI avg10, RoCE-pause=sum NIC pause counters:
```
pre-train:        freeMB=102633  psiMem=0 psiIo=0     temp=45C pwr=18W pause=22
"Starting steps": freeMB=60643   psiMem=0 psiIo=0.69  temp=59C pwr=43W pause=22
+30s:             freeMB=42542   psiMem=0 psiIo=0.03  temp=78C pwr=63W pause=128
+60s:             freeMB=35642   psiMem=0 psiIo=0      temp=73C pwr=64W pause=224
final ~40s flat:  freeMB≈23950   psiMem=0 psiIo=0     temp=74-81C pwr=68-73W pause=529  -> then vanishes, no more beats.
```
**[Observed] All-nodes snapshot at the freeze instant:**
```
.68 (rank0/MASTER): freeMB=23725 temp=64C pwr=20W pause=9972   <- 18x the others
.12 (rank2):        freeMB=23972 temp=62C pwr=20W pause=953
.19 (rank3):        freeMB=24166 temp=62C pwr=21W pause=303
```
**[Observed] netconsole (kernel printk streamed off-box):** NOTHING from .80 at the hang — no RCU stall/SMMU/AER/soft-lockup. Silent.

**[Observed] NCCL config (verbatim):**
```
NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1  NCCL_IB_TC=104  NCCL_IB_GID_INDEX=3  NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_TIMEOUT=23  NCCL_IB_RETRY_CNT=7  NCCL_NET_GDR_LEVEL=0  NCCL_TIMEOUT=1800  TORCH_NCCL_DUMP_ON_TIMEOUT=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
```
**[Observed] FSDP/accelerate config (verbatim):**
```
FULL_SHARD, TRANSFORMER_BASED_WRAP(Qwen3_5DecoderLayer), backward_prefetch=BACKWARD_PRE, forward_prefetch=TRUE,
mixed_precision='no', use_orig_params=true, sync_module_states=true, rdzv_backend=static, same_network=true.
CPT_BUCKETING=1 (buckets short16/mid4/long1, TOKEN_BUDGET_PER_STEP=262144).
```
**[Observed] RoCE NIC config (ethtool -a / mlnx_qos):**
```
ethtool -a: Autoneg off, RX pause OFF, TX pause OFF (link-level global pause disabled).
Priority trust=dscp. NCCL_IB_TC=104 -> DSCP 26 -> priority 3. PFC enabled on priority 3 ONLY (buffer on prio3).
Receive buffer: 19872,523296,0,0,0,0,0,0 (max 2039616). Idle counters: rx_ecn_mark=0, rx_out_of_buffer=0,
tx_pause_ctrl_phy=22, *_discards_phy=0, rx_prioN_discards=0.
```

## Constraints
- **[Constraint]** GDR (GPUDirect RDMA) UNSUPPORTED on these Sparks by design — NCCL_NET_GDR_LEVEL=0, host-staged.
- **[Constraint]** iommu.passthrough=1 is TOXIC on Grace SMMUv3 (tested; wedges at init) — kept 0.
- **[Constraint]** Recovery is physical-power-cycle ONLY. SBSA watchdog armed (panic=15, softlockup/hung_task panic on) but never fires on this hang.
- **[Observed]** Lowering MAX_SEQ 16384→4096 reduced breadth (2 nodes→1) and got further (to "Starting steps") but did NOT eliminate the wedge.

## Objective
Tell us the MECHANISM and the FIX. Specifically:
1. From the raw data, what is the most likely mechanism of the silent, watchdog-evading, power-cycle-only hang? Challenge our Prior proposal.
2. Ranked config changes to test (NCCL QPS/BUFFSIZE/IB_SPLIT_DATA_ON_QPS/TC/timeout; mlnx_qos PFC/ECN/buffer; allocator; FSDP prefetch) — each tied to a specific datum above.
3. What additional instrumentation isolates causality direction (PFC-storm→hang vs hang→pause-pileup)?
4. Any known ConnectX-7 / GB10 / RoCE-on-aarch64 failure mode matching a silent hard hang under sustained heavy lossless-RoCE collective traffic?

**[Prior proposal — Inferred, challenge it]** We currently read 2b as RULING OUT memory exhaustion (free flat ~24GB,
not→0) and memory/IO stall (PSI=0); thermal as secondary (~80C, stabilized, no runaway); and the standout as 2c —
master .68 RoCE pause=9972 (18× others) with .80 hanging silently at the NIC/fabric level → we infer a RoCE/PFC
fabric-congestion → NIC/PCIe hard-hang. **[Unknown]** causality direction (does the PFC backpressure cause .80's hang,
or does .80's NIC hanging first pile up the master's pause counter?). We could not disentangle this from the data.
