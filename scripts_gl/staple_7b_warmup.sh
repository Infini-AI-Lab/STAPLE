#!/usr/bin/env bash
set -ex

# STEM DAG OLMo2-7B training on P11 cluster.
# 16 stem layers with per-layer embedding tables (vocab_size x hidden_dim).
#
# Usage (single node, 8 GPUs):
#   cd /dev/shm/STAPLE/repos/torchtitan
#   bash scripts_gl/staple_7b_warmup.sh

# WANDB_API_KEY is read from the environment -- never hardcode it here.
export WANDB_TEAM="randomresearch"
export WANDB_PROJECT="stem"
export WANDB_RUN_NAME="stemdag-7b-warmup"
export WANDB_MODE=online

NNODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-47778}"

SHARED_ROOT="${SHARED_ROOT:-/dev/shm/STAPLE}"
SHARED_CODE="${SHARED_ROOT}/repos/torchtitan"
OUTPUT_DIR="${SHARED_ROOT}/outputs/torchtitan/stem-dag-7b-warmup-${NNODES}n"

BASE_CKPT="${SHARED_ROOT}/checkpoints/OLMo2-7B-base-token5T"
INIT_CKPT="${SHARED_ROOT}/checkpoints/OLMo2-7B-base-token5T-torchtitan"

export STEM_DATA_ROOT="${SHARED_ROOT}/data/pretraining"

# NCCL settings for P11 cluster (InfiniBand interconnect)
if [ "${NNODES}" -gt 1 ]; then
    export NCCL_NET_PLUGIN=""
    export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9"
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=1
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_ADAPTIVE_ROUTING=1
    export NCCL_MIN_NCHANNELS=8
    export NCCL_MAX_NCHANNELS=16
else
    export NCCL_NET_PLUGIN=""
fi
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Ban the LL protocol: the tuner was selecting RING_LL for full-size FSDP
# all-gather/reduce-scatter (~14-18 GB/s effective vs ~400+ GB/s NVLink on H200).
# Simple lets Ring/NVLS run at full bandwidth. Leave NCCL_ALGO unset so the tuner
# can still use Ring/Tree for Broadcast (NVLS has no Broadcast kernel).
export NCCL_PROTO=Simple
unset NCCL_ALGO
echo "NCCL ALGO: ${NCCL_ALGO:-<tuner-chosen>}  NCCL PROTO: ${NCCL_PROTO}"

mkdir -p "${OUTPUT_DIR}"
cd "${SHARED_CODE}"

DP_REPLICATE=${NNODES}

torchrun \
    --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --node-rank=${NODE_RANK} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    -m torchtitan.train \
    --module stem_dag \
    --config stem_dag_7b_prefine \
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE} \
    --parallelism.data-parallel-shard-degree 8 \
    --parallelism.expert-parallel-degree 8 \
    --activation-checkpoint.mode selective \
    --training.local-batch-size 4 \
    --training.global-batch-size 1024 \
    --checkpoint.enable \
    --checkpoint.interval 500 \
    --checkpoint.initial_load_path ${INIT_CKPT} \
    --checkpoint.no_initial_load_model_only \
    --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state \
    --checkpoint.no_last_save_model_only \
    --dump-folder "${OUTPUT_DIR}"



    # --training.local-batch-size 8 \
    # --activation-checkpoint.mode selective \
    # --checkpoint.enable \
    # --checkpoint.initial_load_path /dev/shm/STAPLE/checkpoints/OLMo2-7B-base-token5T-torchtitan \
    # --checkpoint.no_initial_load_model_only \
    # --checkpoint.exclude_from_loading dataloader,lr_scheduler \
