#!/usr/bin/env bash
# Run on every Spark: launch_4node.sh <NODE_RANK 0-3> [train_ddp_lora.py args].
set -euo pipefail

: "${SPARK_HOME:?SPARK_HOME is required}"
: "${SPARK_RAIL_MASTER:?SPARK_RAIL_MASTER is required}"
[ "$#" -ge 1 ] || {
  echo "usage: launch_4node.sh <rank 0-3> [trainer args]" >&2
  exit 1
}
RANK=$1
shift
case "$RANK" in 0|1|2|3) ;; *) echo "invalid rank: $RANK" >&2; exit 1;; esac

NCCL_LIB=${SPARK_HOME}/.local/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH="$NCCL_LIB:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_MERGE_NICS=0
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=8388608
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
export NCCL_NET_GDR_C2C=0
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_DMABUF_ENABLE=0
export NCCL_LOCAL_REGISTER=0
export NCCL_GRAPH_REGISTER=0
export NCCL_WIN_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export HF_HUB_DISABLE_XET=1
export HF_HOME=${SPARK_HOME}/hf_cache
export TOKENIZERS_PARALLELISM=false
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SPARK_HOME}/.triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8
export PYTORCH_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8

cd "${SPARK_HOME}/palios-training"
exec python3 -m torch.distributed.run \
  --nnodes=4 \
  --node_rank="$RANK" \
  --nproc_per_node=1 \
  --master_addr="$SPARK_RAIL_MASTER" \
  --master_port=29500 \
  careers-qwen/train_ddp_lora.py "$@"
