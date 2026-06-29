---
type: consultation
to: family
subject: GB10 silent-hang RCA — ROUND 2 (fix#1 refuted + confirmed firmware/driver versions + hard no-crash constraint)
available_context_inventory:
  - your own Round-1 analyses (attached)
  - NEW experimental result: fix #1 (drop garbage_collection_threshold:0.8) applied + verified, STILL wedged
  - NEW confirmed non-destructive facts: driver 580.95.05, CX-7 fw 28.45.4028, NCCL 2.28.9
  - hard new operating constraint: every wedge = catastrophic physical-power-cycle; we can no longer crash-test freely
---

# Consultation ROUND 2 — refine to a high-confidence fix + a NON-DESTRUCTIVE validation path

Round 1 (your attached analyses) converged: a NIC/PCIe-domain stall on .80 (rank1) during the first full-scale FSDP
collective; CPU-state vs silicon-freeze debated; master pause=symptom. We then ran the #1 ranked fix and it failed.
We need you to (a) re-rank given the new evidence, and (b) tell us how to validate a fix WITHOUT crashing the cluster.

## Problem
The wedge persists, and crash-testing is now off the table: each wedge hard-bricks a node (physical power cycle,
catastrophic). We need the highest-confidence fix AND a way to confirm it that does not risk another crash.

## Ground truth (new since Round 1)
- **[Observed] Fix #1 REFUTED.** We removed `garbage_collection_threshold:0.8` (set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`),
  VERIFIED in the live `/proc/<pid>/environ` of the training process, and re-ran 4-node/4096. It STILL wedged — and TWO
  nodes (.80 AND .19) went down (vs only .80 previously). So Gemini's "0.8 GC-threshold → 81% flatline → SMMU-storm"
  trigger is **not** the (sole) cause; the memory/VMM-churn path is not it.
- **[Observed] Confirmed versions (read-only, non-destructive):**
  - GPU driver **580.95.05** (older than 580.126 which carried a confirmed DGX-Spark CX-7 PCIe-power-throttle bug, and older than 580.142 which fixed it).
  - ConnectX-7 firmware **28.45.4028 (PSID NVD0000000087)** on both rails — this is the exact firmware family in public DGX-Spark "ConnectX-7 NICs disappear / cx7-pcie-hotplug" reports. **28.47.1088** is available.
  - NCCL **2.28.9** (≥2.27 → `NCCL_NET_GDR_C2C` defaults to 1, which can override `NCCL_NET_GDR_LEVEL=0` on C2C-attached NICs; we have NOT verified whether a C2C-GDR path is actually being selected).
- **[Observed] Failure matrix (unchanged):** 2048/3-node ok (1700 steps); 4096/4-node wedges; 16384/4-node wedges 2 nodes. CPT corpus uniformly near-MAX.
- **[Observed] At-freeze telemetry (Round-1 capture):** .80 freeMB flat ~24GB, PSI=0, temp ~80C, master pause=9972 (symptom). No kernel log; SBSA + softlockup + hung_task watchdogs all silent; both rails dead; power-cycle-only.
- **[Constraint]** No in-band BMC (`/dev/ipmi0` absent) — cannot get an ARM-TF serial SError dump in-band. Switch model unconfirmed (no LLDP); fabric MACs are 4c:bb:47:* (Mellanox).

## Constraints
- **[Constraint] HARD: do not propose anything that requires crashing a node to learn from it.** Every wedge is a catastrophic power-cycle. Validation must be non-destructive or abortable-before-freeze.
- **[Constraint]** Firmware/driver updates are low-level changes requiring careful, reversible, vendor-correct procedure (brick risk on a GB10/aarch64 custom platform) — treat the *procedure* as part of the answer.
- **[Constraint]** GDR legacy-peermem unsupported on aarch64; iommu.passthrough=0 required for CUDA; reboots OK.

## Objective
1. **Re-rank the fix given fix#1's failure + the confirmed buggy firmware/driver.** Is the firmware (28.45.4028→28.47.1088) + driver (580.95.05→≥580.142) update now the PRIMARY fix, or a co-factor? Where do the config levers (C2C-GDR-off, forward_prefetch=False, QPS 4→1, kernel-6.11 PFC-ordering, ECN/TC=106, disable CX-7 idle-hotplug/ASPM) rank relative to it?
2. **A NON-DESTRUCTIVE validation path.** How do we confirm a candidate fix WITHOUT a wedge-prone full CPT run? E.g. nccl-tests all_gather/reduce_scatter ramped to the failing message size (does it crash standalone?); an instrumented run with an automatic kill the instant a leading indicator crosses threshold (what indicator/threshold fires BEFORE the irreversible freeze?); or pure read-only confirmation (is C2C-GDR active in NCCL logs? PCIe AER? mlx5 devlink health?).
3. **Config-only holding pattern.** Can config changes ALONE (no firmware) make 4-node training survive at some seq length, so we can train while a firmware update is planned/approved? If so, the exact minimal set.
4. **Safe firmware-update procedure** on GB10/DGX-Spark CX-7 (mlxfwmanager vs DGX-OS capsule; order; rollback; what confirms success) — since this likely needs doing and we must not brick the fleet.
