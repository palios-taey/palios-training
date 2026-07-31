#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL GB10 COMMS ENV — FIXED. THE SAME EVERY TIME. DO NOT VARY OR EXPERIMENT.
# ══════════════════════════════════════════════════════════════════════════════
# Verbatim from the PROVEN recipe that ran 4367 steps clean over 9 hours
# (dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh, Phase-1 SFT commit dd9e12e).
#
# THE RULE (Jesse, 2026-07-07): the comms must be right EVERY time and the SAME every
# time. Every multi-node run sources THIS file. No inline NCCL vars, no single-vs-dual
# rail experiments, no re-derived subsets. If a change is ever needed it happens HERE,
# once, deliberately — never ad-hoc in a launcher.
#
# Fabric (verified 2026-07-07): each GB10 node has TWO live 200Gb/s RoCE rails —
#   rail A  rocep1s0f0  / enp1s0f0np0  / PRIVATE_RAIL_A_SUBNET   (UP)
#   rail B  roceP2p1s0f0 / enP2p1s0f0np0 / PRIVATE_RAIL_B_SUBNET  (UP)
# The f1 ports of each NIC are DOWN (no cable). Dual-rail = the two f0 ports below.
# ──────────────────────────────────────────────────────────────────────────────

# ── NCCL RoCE — dual-rail, host-staged CPU proxy (GB10 has NO GPUDirect-RDMA) ──
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1   # BOTH 200Gb rails — never single-rail
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=1   # WEDGE_RCA fix #4: 4→1 de-burst the SMMUv3 (4 QPs × peers × rails retry-storm the cmd queue at the first big collective). Was 4 (SFT-proven, but SFT's small collectives never hit the wedge).
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_C2C=0     # NCCL 2.27+ defaults =1 on C2C NICs and silently re-enables GDR — force off
export NCCL_NET_GDR_READ=0
export NCCL_PXN_C2C=0         # WEDGE_RCA_R3 (Claude): NCCL 2.28 added NCCL_PXN_C2C defaulting to 1 (was 0 in 2.27) — a SECOND C2C→NIC DMA path we never closed; matches the DMA-triggered CX-7 freeze. Force off.
export NCCL_NET_PLUGIN=none
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# ── GB10 sm_121 hardening + allocator (tied to the SMMUv3/SError wedge RCA) ──
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # WEDGE_RCA fix #1 (PRIMARY): DROPPED garbage_collection_threshold:0.8 — the node flatlines at 81% UMA (=0.8) → expandable-segments VMM floods Grace SMMUv3 with TLB shootdowns, collides with CX-7 collective DMA → PCIe Completion Timeout → fatal SError → GIC lock → node black-holes (both rails dark, power-cycle only). THIS is the wedge. SFT never hit it (small collectives); CPT near-16384-seq does.
export TOKENIZERS_PARALLELISM=false
