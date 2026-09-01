#!/usr/bin/env bash
set -ex

# WANDB_API_KEY is read from the environment -- never hardcode it here.
export WANDB_TEAM="randomresearch"
export WANDB_PROJECT="stem"
export WANDB_RUN_NAME="olmo2-7b-warmup"
export WANDB_MODE=offline

NNODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

RUN_NAME="olmo2-7b-midfine-w"
RDZV_ID="${JOB_NAME:-${RUN_NAME}}"
MASTER_ADDR="${MASTER_ADDR:-${RUN_NAME}-worker-0.${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WANDB_RUN_NAME="${RUN_NAME}"
export MLFLOW_RUN_NAME="${RUN_NAME}"

SHARED_ROOT="${SHARED_ROOT:-/dev/shm/STAPLE}"
SHARED_CODE="${SHARED_ROOT}/repos/torchtitan"
OUTPUT_DIR="${SHARED_ROOT}/outputs/torchtitan/${RUN_NAME}-${NNODES}n"

BASE_CKPT="${SHARED_ROOT}/checkpoints/OLMo2-7B-base-token5T/"
INIT_CKPT="${SHARED_ROOT}/outputs/torchtitan/olmo2-7b-warmup-8n/checkpoint/step-11500/"

export STEM_DATA_ROOT="${SHARED_ROOT}/data/midtraining/dolmino-mix_shuffled/"

mkdir -p "${OUTPUT_DIR}"
cd "${SHARED_CODE}"

DP_REPLICATE=${NNODES}

# MLflow destination. Point these at your own tracking server / experiment.
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-staple/stem}"

torchrun \
    --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id=${JOB_NAME} \
    -m torchtitan.train \
    --module olmo2 \
    --config olmo2_7b_midfine \
    --hf-assets-path ${BASE_CKPT} \
    --training.local-batch-size 8 \
    --training.global-batch-size 1024 \
    --checkpoint.enable \
    --checkpoint.interval 500 \
    --checkpoint.initial_load_path ${INIT_CKPT} \
    --checkpoint.no_initial_load_model_only \
    --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state \
    --checkpoint.no_last_save_model_only \
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE} \
    --activation_checkpoint.mode "selective" \
    --metrics.enable_mlflow \
    --metrics.mlflow_tracking_uri "${MLFLOW_TRACKING_URI}" \
    --metrics.mlflow_experiment_name "${MLFLOW_EXPERIMENT_NAME}" \
    --dump-folder "${OUTPUT_DIR}"
