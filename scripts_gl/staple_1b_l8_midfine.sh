#!/usr/bin/env bash
set -ex

# WANDB_API_KEY is read from the environment -- never hardcode it here.
export WANDB_TEAM="randomresearch"
export WANDB_PROJECT="stem"
export WANDB_MODE=offline

NNODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

RUN_NAME="staple-1b-L8-wm-midfine"
RDZV_ID="${JOB_NAME:-${RUN_NAME}}"
MASTER_ADDR="${MASTER_ADDR:-${RUN_NAME}-worker-0.${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WANDB_RUN_NAME="${RUN_NAME}"
export MLFLOW_RUN_NAME="${RUN_NAME}"

SHARED_ROOT="${SHARED_ROOT:-/dev/shm/STAPLE}"
SHARED_CODE="${SHARED_ROOT}/repos/torchtitan"
OUTPUT_DIR="${SHARED_ROOT}/outputs/torchtitan/${RUN_NAME}-${NNODES}n"

BASE_CKPT="${SHARED_ROOT}/checkpoints/olmo2-1b-base-token4T-torchtitan/"
INIT_CKPT="${SHARED_ROOT}/outputs/torchtitan/staple-1b-L8-warmup-mix-4n/checkpoint/step-50000/"

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
    --rdzv_id=${RDZV_ID} \
    --rdzv-conf join_timeout=1800 \
    -m torchtitan.train \
    --comm.train_timeout_seconds 600 \
    --module stem_dag \
    --config stem_dag_1b_l8_midfine \
    --hf-assets-path ${BASE_CKPT} \
    --training.local-batch-size 8 \
    --training.global-batch-size 512 \
    --checkpoint.enable \
    --checkpoint.interval 500 \
    --checkpoint.initial_load_path ${INIT_CKPT} \
    --checkpoint.no_initial_load_model_only \
    --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state \
    --checkpoint.no_last_save_model_only \
    --checkpoint.keep_latest_k 3 \
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE} \
    --activation-checkpoint.mode selective \
    --metrics.enable_mlflow \
    --metrics.mlflow_tracking_uri "${MLFLOW_TRACKING_URI}" \
    --metrics.mlflow_experiment_name "${MLFLOW_EXPERIMENT_NAME}" \
    --dump-folder "${OUTPUT_DIR}"
