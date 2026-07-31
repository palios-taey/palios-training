# fabric_env.sh — proven GB10 (4× DGX-Spark) NCCL fabric + CUDA-13 toolchain block.
# EXTRACTED VERBATIM from dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh (the validated 27B CPT
# launcher). SOURCE this (do NOT execute); do not substitute NCCL, allocator, socket, or toolchain
# values. No RANK / master / FSDP env here — the launcher owns those.
NCCL_LIB=$HOME/.local/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB:${LD_LIBRARY_PATH:-}
# --- validated NCCL block for 4× GB10 dual-rail RoCE CPU-proxy ---
export NCCL_NET_PLUGIN=none          # CRITICAL: AWS-OFI plugin fails on GB10 (proven)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_MERGE_NICS=0          # disable dual-port merging (dual-rail isolation w/o 10GbE fallback)
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=8388608
export NCCL_TIMEOUT=1800   # NOTE: not a documented NCCL var (silent no-op); real PG timeout is InitProcessGroupKwargs in the trainer. Kept for parity.
# RDMA retry-exhaustion mitigation (IBV_WC_RETRY_EXC_ERR under heavy reduce-scatter load).
export NCCL_IB_TIMEOUT=22            # ~4.3s->~17s per retry; ~60-120s total window
export NCCL_IB_RETRY_CNT=7           # hardware-saturating (3-bit QP field)
# heartbeat watchdog: clean traceback+abort instead of a silent hung-collective → OOM → panic.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120   # 120=clean OOM abort, not 1800 (30min hang→freeze risk)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # override to INFO to capture topology + tuned bus-bandwidth
[ -n "${NCCL_DEBUG_FILE:-}" ] && export NCCL_DEBUG_FILE
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"   # INIT shows topology detection + tuned bandwidths
# GAIA fragrdma reframe: force GPUDirect-RDMA off + disable every buffer-registration path so NCCL
# never registers torch's remappable VMM pages with the NIC (the whole-node-death mechanism).
export NCCL_NET_GDR_C2C=0          # <<< THE one: 2.28 default 1 overrides GDR_LEVEL on C2C parts
export NCCL_NET_GDR_LEVEL=LOC      # (overrides the =0 above; LOC = keep GDR genuinely off)
export NCCL_DMABUF_ENABLE=0        # no dma-buf export of remappable VMM pages
export NCCL_LOCAL_REGISTER=0       # stop ncclCommRegister of allocator segments
export NCCL_GRAPH_REGISTER=0       # stop graph-capture buffer registration
export NCCL_WIN_ENABLE=0           # window registration (cuMem-dependent)
export NCCL_NVLS_ENABLE=0          # no NVSwitch on Spark
export NCCL_CUMEM_HOST_ENABLE=0    # host-staged path: force cudaHostAlloc so host-buf accounting is legible
# NCCL_CUMEM_ENABLE left UNSET (auto).
export HF_HUB_DISABLE_XET=1 HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false
# TRITON on sm_121: point Triton at the SYSTEM CUDA-13.0 ptxas (bundled ptxas lacks sm_121a).
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton_cache}"
# PLAIN allocator (expandable_segments:False) — expandable_segments:True breaks multi-node RDMA on GB10.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False   # new-name var, matches CUDA_ALLOC_CONF
# CUDA-13 toolchain (sm_121 / GB10).
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
