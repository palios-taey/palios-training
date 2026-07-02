#!/bin/bash
# Run on EACH of the 4 nodes: launch_4node.sh <NODE_RANK 0-3> [extra train args]
# Clean 4-node DDP-LoRA. Validated NCCL block (NET_PLUGIN=none proven 21.7GB/s).
set -uo pipefail
RANK=$1; shift
NCCL_LIB=/home/spark/.local/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB:${LD_LIBRARY_PATH:-}
# --- validated NCCL block for 4× GB10 dual-rail RoCE CPU-proxy ---
export NCCL_NET_PLUGIN=none          # CRITICAL: AWS-OFI plugin fails on GB10 (proven)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_MERGE_NICS=1
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=8388608
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export HF_HUB_DISABLE_XET=1 HF_HOME=/home/spark/hf_cache TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/spark/careers-qwen
python3 -m torch.distributed.run \
  --nnodes=4 --node_rank=$RANK --nproc_per_node=1 \
  --master_addr=192.168.100.10 --master_port=29500 \
  train_ddp_lora.py "$@"
