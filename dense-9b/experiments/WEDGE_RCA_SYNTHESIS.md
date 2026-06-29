# GB10 4-node FSDP CPT silent-hang — RCA synthesis (5/5 Family panel + raw capture)
Date: 2026-06-27. Inputs: instrumented capture (telemetry into the freeze) + full configs + 5 independent analyses
(Grok/Perplexity/Claude/ChatGPT/Gemini), each given the raw data with the hypothesis labeled challengeable.

## Convergence (what all 5 agree on)
- TRIGGER: the first full-scale FSDP collective (all-gather/reduce-scatter) on uniform near-MAX-seq CPT, on .80 (rank1).
- RESULT: a silicon/fabric-level HARD freeze on .80, below OS visibility — CPU/interrupt path frozen, both CX-7 rails
  dead together (shared PCIe domain), no kernel log, no watchdog, power-cycle-only.
- master .68 pause=9972 is a SYMPTOM (backpressure), not the cause. (3/5 say .80-first definitively; ChatGPT keeps
  causality formally open — the lossy-RoCE ablation test below settles it.)
- NOT memory-OOM-to-zero, NOT thermal, NOT a kernel software fault.

NOTE on neutrality: the 5 panels DIVERGED on the precise final mechanism (firmware-QP / DMA-credit / PCIe-domain /
PFC-ordering / SError-GIC) while converging on trigger + fix-family. Claude and Gemini explicitly CHALLENGED my
PFC-primary framing. That divergence is evidence the packet was not leading (they did not just echo my hypothesis).

## Deepest mechanism (Gemini — fits ALL data incl the 81% flatline; reconciles the prior synth-probe)
1. .80 freeMB flatlines at ~23,950 MB = 104 GB used / 128 GB = **81%** — i.e. exactly the `garbage_collection_threshold:0.8`
   we set in PYTORCH_CUDA_ALLOC_CONF.
2. At 0.8, PyTorch's expandable-segments VMM aggressively munmap/mmap → floods Grace **SMMUv3 with TLB shootdowns**.
3. Concurrently the CX-7 blasts 200Gb ATS DMA through the SAME SMMU for the AllGather (QPS=4). The collision exhausts
   the SMMU command queue → PCIe stall → **PCIe Completion Timeout (CTO)**.
4. On x86 a CTO → AER log + device reset. On **Grace aarch64 an unhandled CTO escalates to a fatal SError → EL3**,
   deadlocking the CMN-700 mesh and locking the **GIC** → watchdog can't fire, netconsole blinded, both NICs dark,
   powered-but-frozen. THE black hole. This uniquely explains the watchdog-evasion.
- Reconciliation: our own 2026-05-15 synth-probe (bare NCCL reduce_scatter at the failing size PASSED) is consistent —
  the probe lacked the concurrent VMM-page-churn + 80%-threshold collision of real FSDP training, so it didn't trigger
  the SMMU storm. Bare-fabric-OK + real-training-wedges ⇒ the trigger is the memory/translation collision, not the NIC alone.

## The encouraging headline: several top suspects are CONFIG VALUES WE SET (likely self-inflicted, reversible)
- `garbage_collection_threshold:0.8` → the 81% flatline (Gemini #1)
- `NCCL_IB_TC=104` = DSCP26 + ECN bits 00 = **Not-ECT** → that's *why* `rx_ecn_mark=0` (ChatGPT)
- `NCCL_IB_QPS_PER_CONNECTION=4` → multiplies SMMU translation/QP load (all 5)
- `forward_prefetch=TRUE` + `BACKWARD_PRE` → stacks the AllGather burst at step 0 (all 5)
- kernel-6.11 PFC set after buffers → NVIDIA-documented "no route to host" / undefined behavior (ChatGPT)
- "GDR off" may be FALSE: NCCL 2.27+ `NCCL_NET_GDR_C2C=1` defaults on, can override GDR_LEVEL on C2C (ChatGPT)

## Ranked fix ladder (cheap→deep, each tied to a datum; ALL config/reversible except #7)
1. **PYTORCH_CUDA_ALLOC_CONF: drop `garbage_collection_threshold:0.8`** (test also without expandable_segments). [81% flatline]
2. **kernel-6.11-safe QoS: reset, apply PFC BEFORE buffers/prio2buffer, cold reboot.** [vendor "no route to host" match]
3. **Real ECN: `NCCL_IB_TC=106` (DSCP26+ECT0) + switch ECN on TC3 + CNP** ; and **genuinely disable C2C-GDR**
   (`NCCL_NET_GDR_C2C=0`, `GDR_LEVEL=LOC`, `GDR_READ=0`, verify in NCCL log). [rx_ecn_mark=0 ; C2C default-on]
4. **De-burst NCCL: `NCCL_IB_QPS_PER_CONNECTION` 4→1**, `SPLIT_DATA_ON_QPS=0`. [SMMU/QP load]
5. **FSDP: `forward_prefetch=False`, `backward_prefetch=BACKWARD_POST`, `limit_all_gathers=True`.** [step-0 burst]
6. **mlnx_qos: max prio3 buffer (2039616).** [headroom]
7. **Driver 580.95.05→≥580.142 + CX-7 fw 28.45.4028→28.47.1088 + disable CX-7 idle-hotplug/ASPM.** [Claude/Grok; co-factor]

## Decisive tests (run alongside; settle the open questions)
- **BMC serial console (`ipmitool sol activate`)** — an SError prints to ARM-TF UART at the freeze instant, bypassing the
  wedged PCIe that blinds netconsole. THIS is how we capture the SError if Gemini's mechanism is right. (highest value)
- **Lossy-RoCE ablation** — `mlnx_qos --pfc 0,0,0,0,0,0,0,0`. If .80 STILL hard-hangs → PFC is innocent, it's the
  SMMU/SError host crash. If it survives (graceful NCCL timeout) → CX-7 PFC firmware deadlock. Discriminates the two leading hypotheses.
- **local-NVMe canary** — is the CPU alive in the hang (Perplexity/Claude) or frozen (Gemini GIC-lock)?
- **rank/node permute** — .80-the-box vs rank1-the-role.
- **nccl-tests without FSDP** — RoCE-alone vs FSDP-traffic-shape required.

## Open / honest
- Final mechanism (SError-host-crash vs PFC-fabric-deadlock) not yet proven — the BMC-serial + lossy-RoCE-ablation settle it. The fix ladder covers both.
- ChatGPT cited a "MikroTik CRS812" switch I did NOT provide — verify the actual switch before switch-side ECN/PFC config.
- Grok cited "forums" unverifiably; Perplexity/Claude/ChatGPT cited fetchable NVIDIA docs (higher confidence on the kernel-6.11/TC/GDR facts).
- Start with #1 (drop the 0.8 GC threshold): most-specific-to-data, trivial, reversible. Apply in order, BMC-serial + canary running, repro on 4-node/4096 after a .80 cycle.
