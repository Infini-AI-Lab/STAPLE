#!/usr/bin/env bash
set -ex

export WANDB_TEAM="${WANDB_TEAM:-randomresearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-stem}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NCCL_NET_PLUGIN_PATH="${NCCL_NET_PLUGIN_PATH:-/opt/aws-ofi-nccl/lib/libnccl-net.so}"
if [[ ! -f "${NCCL_NET_PLUGIN_PATH}" ]]; then
    echo "Required EFA plugin not found: ${NCCL_NET_PLUGIN_PATH}" >&2
    exit 1
fi
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_PATH}"
export OFI_NCCL_PROTOCOL=SENDRECV
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

NNODES="${NUM_NODES:-1}"
RUN_NAME="${RUN_NAME:-${JOB_NAME:-e8g8t8-8l-stem12-midfine}}"
RDZV_ID="${JOB_NAME:-${RUN_NAME}}"
MASTER_ADDR="${MASTER_ADDR:-${RUN_NAME}-worker-0.${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WANDB_RUN_NAME="${RUN_NAME}"
export MLFLOW_RUN_NAME="${RUN_NAME}"

SHARED_ROOT="${SHARED_ROOT:-/dev/shm/STAPLE}"
SHARED_CODE="${SHARED_CODE:-${SHARED_ROOT}/repos/torchtitan}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_ROOT}/outputs/torchtitan/${RUN_NAME}-${NNODES}n}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/${RUN_NAME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/${RUN_NAME}/inductor}"

UPCYCLED_CKPT="${UPCYCLED_CKPT:-${SHARED_ROOT}/checkpoints/staple-1b-e8g8t8-8l-stem12-upcycled-torchtitan}"
INIT_CKPT="${INIT_CKPT:-${UPCYCLED_CKPT}}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
TRAINING_STEPS="${TRAINING_STEPS:-50000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
EXPERT_LR="${EXPERT_LR:-2e-4}"
STEM_LR="${STEM_LR:-4e-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

export STEM_DATA_ROOT="${STEM_DATA_ROOT:-${SHARED_ROOT}/data/midtraining/dolmino-mix_shuffled/}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${SHARED_CODE}"

DP_REPLICATE="${NNODES}"
echo "NCCL EFA: plugin=${NCCL_NET_PLUGIN} protocol=${OFI_NCCL_PROTOCOL}"
echo "STAPLE MoE: stem_layers=1-12 moe_layers=1,3,...,15"
echo "MoE topology: E8G8T8 dp_replicate=${DP_REPLICATE} dp_shard=8 ep=4 efsdp=2"
echo "Fresh optimizer: warmup=${WARMUP_STEPS} expert_lr=${EXPERT_LR} stem_lr=${STEM_LR}"

# MLflow destination. Point these at your own tracking server / experiment.
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-staple/stem}"

torchrun \
    --nproc_per_node=8 \
    --nnodes="${NNODES}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv-conf join_timeout=1800 \
    -m torchtitan.train \
    --comm.train_timeout_seconds 600 \
    --module olmo2_moe_stem_dag \
    --config olmo2_moe_stem_dag_static_1b_e8g8t8_8l_stem12_midfine \
    --hf-assets-path "${UPCYCLED_CKPT}" \
    --training.local-batch-size "${LOCAL_BATCH_SIZE}" \
    --training.global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --training.steps "${TRAINING_STEPS}" \
    --expert-hyperparams.lr "${EXPERT_LR}" \
    --stem-hyperparams.lr "${STEM_LR}" \
    --lr_scheduler.warmup_steps "${WARMUP_STEPS}" \
    --checkpoint.enable \
    --checkpoint.interval "${CHECKPOINT_INTERVAL}" \
    --checkpoint.initial_load_path "${INIT_CKPT}" \
    --checkpoint.initial_load_model_only \
    --checkpoint.no_last_save_model_only \
    --checkpoint.keep_latest_k 3 \
    --parallelism.data-parallel-replicate-degree "${DP_REPLICATE}" \
    --parallelism.data-parallel-shard-degree 8 \
    --parallelism.expert-parallel-degree 8 \
    --activation-checkpoint.mode selective \
    --compile.components model,loss \
    --compile.no-fullgraph \
    --metrics.enable_mlflow \
    --metrics.mlflow_tracking_uri "${MLFLOW_TRACKING_URI}" \
    --metrics.mlflow_experiment_name "${MLFLOW_EXPERIMENT_NAME}" \
    --dump-folder "${OUTPUT_DIR}" \
    "$@"
