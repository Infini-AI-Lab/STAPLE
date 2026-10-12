#!/usr/bin/env bash
set -ex

# ===========================================================================
# staple-7b prefine90B -> midfine100B  |  OBX (EKS volcano) launch, 8x p5e
#
# Launch with:  bash submit/launch_titan_gl.sh staple_7b_midfine 8 p5e
#
# Data + checkpoints are read/written DIRECTLY on the FSx/Lustre mount -- NO
# node-local /dev/shm copy. Verified on 8x p5e (2026-07-03): reading the full
# dolmino mix straight off FSx sustains ~6,500 tps / ~2.5 s per step (~90% of a
# /dev/shm-local copy) at ~100% MFU with zero NCCL watchdog timeouts, once the
# EFA fix below is in place. Skipping the copy removes ~3 h of uneven, S3-backed
# FSx->/dev/shm staging (and the straggler/rendezvous risk that came with it).
# Only logs/TB/profiler stay on fast node-local /dev/shm (small-file writes to
# FSx are ~250 ms/create and add per-step jitter; keep them off FSx).
# ===========================================================================

# WANDB_API_KEY is read from the environment -- never hardcode it here.
export WANDB_TEAM="randomresearch"
export WANDB_PROJECT="stem"
RUN_NAME="staple-7b-prefine90B-midfine100B-fsx"
export WANDB_RUN_NAME="${RUN_NAME}"
export WANDB_MODE=offline
export MLFLOW_RUN_NAME="${RUN_NAME}"

NNODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
JOB_NAME="${RUN_NAME}"
MASTER_ADDR="${MASTER_ADDR:-${JOB_NAME}-worker-0.${JOB_NAME}}"
MASTER_PORT="${MASTER_PORT:-29500}"

SHARED_ROOT="${SHARED_ROOT:-/dev/shm/STAPLE}"
SHARED_CODE="${SHARED_ROOT}/repos/torchtitan"

# Logs/TB/profiler on fast node-local /dev/shm (tmpfs, ephemeral -- fine, MLflow
# ships metrics remotely and checkpoints are durable on FSx below).
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/dev/shm/STAPLE}"
OUTPUT_DIR="${LOCAL_SCRATCH}/outputs/torchtitan/${RUN_NAME}-${NNODES}n"

BASE_CKPT="${SHARED_ROOT}/checkpoints/OLMo2-7B-base-token5T/"

# ---- Data + checkpoints: read/write DIRECTLY on FSx (no node-local copy) ----
# Data: the JSONL dataloader streams chunks off FSx per step. This is fast enough
# on p5e once EFA is used (see header); num_workers=0 (in-process loader) avoids
# the Lustre fork/mmap deadlock.
STEM_DATA_ROOT="${SHARED_ROOT}/data/midtraining/dolmino-mix_shuffled"
export STEM_DATA_ROOT="${STEM_DATA_ROOT}/"

# Cold-start init checkpoint (consulted only on a COLD start; once FSX_CKPT_DIR
# has a checkpoint the run resumes from there and this is ignored). Loaded
# straight off FSx -- no copy.
INIT_CKPT="${SHARED_ROOT}/outputs/torchtitan/stem-dag-7b-warmup-8n/checkpoint/step-22800/"

# Checkpoints written to (and resumed from) FSx: complete across shard/ep ranks,
# durable across node recycles, visible to the FSx eval watchers. On relaunch
# with the same RUN_NAME the checkpointer finds FSX_CKPT_DIR/step-N and resumes
# (step/optimizer/LR/dataloader all restored).
#
# async_mode MUST be `disabled` on this p5e EKS fabric. With `async`, the
# CheckpointManager calls dist.new_group(backend="gloo") at init
# (components/checkpoint.py:504) to build the async-save PG -- and that gloo
# new_group DEADLOCKS here (all ranks hang in _new_process_group_helper before
# "Training starts"; verified via py-spy on the 8-node SDB). This was the cause
# of the repeated OBX failures (crash/hang at the same point every relaunch),
# NOT hardware. `disabled` skips that new_group entirely and trains cleanly
# (grad_accum=4, MFU ~137%). Trade-off: saves are synchronous (block the step
# loop ~minutes every checkpoint.interval) -- acceptable at interval 500.
FSX_CKPT_DIR="${SHARED_ROOT}/outputs/torchtitan/${RUN_NAME}-${NNODES}n/checkpoint"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${FSX_CKPT_DIR}"
cd "${SHARED_CODE}"

DP_REPLICATE=${NNODES}

# c10d rendezvous timeout: generous so a slow node (e.g. cold FSx checkpoint
# load) doesn't trip the TCPStore connect timeout during startup.
export TORCH_DIST_STORE_TIMEOUT="${TORCH_DIST_STORE_TIMEOUT:-1800}"

# ---------------------------------------------------------------------------
# EFA fix -- WITHOUT THIS, INTER-NODE COMM SILENTLY FALLS BACK TO TCP (~16x slower).
#
# 1) The pod manifest injects NCCL_NET_PLUGIN=/opt/amazon/aws-ofi-nccl/lib/
#    libnccl-net.so, a path that DOES NOT EXIST in the stem:v3-ofi120 image (the
#    plugin is at /opt/aws-ofi-nccl/lib, discovered via LD_LIBRARY_PATH only when
#    NCCL_NET_PLUGIN is UNSET). With the bogus value, NCCL logs "NET/Plugin: Could
#    not find ..." + "NET/IB : No device found" and uses NET/Socket (TCP over the
#    host NIC), leaving all 32 EFA nics idle -> HSDP grad all-reduce ~879ms/call,
#    ~35s/step, ~7% MFU. Point it at the plugin that actually exists.
# 2) libfabric 2.4 only exposes the `efa-direct` fabric, which aws-ofi-nccl 1.20.0
#    fails to init (FI_OPT_MAX_MSG_SIZE -22) -> TCP fallback again. Forcing the
#    SENDRECV protocol selects the classic `efa` fabric that inits cleanly.
# Verified on p5e SDB: step 35s->2.16s, busbw 3.3->119 GB/s, MFU 7->117%.
# At startup confirm the NCCL log shows "Using network Libfabric" / "fabric is efa
# (found 32 nics)" -- NOT "Using network Socket".
if [ ! -e "${NCCL_NET_PLUGIN:-/nonexistent}" ]; then
    for cand in /opt/aws-ofi-nccl/lib/libnccl-net.so \
                /opt/amazon/ofi-nccl/lib/libnccl-net.so; do
        [ -e "${cand}" ] && { export NCCL_NET_PLUGIN="${cand}"; break; }
    done
fi
export OFI_NCCL_PROTOCOL="${OFI_NCCL_PROTOCOL:-SENDRECV}"
echo "NCCL EFA: NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN} OFI_NCCL_PROTOCOL=${OFI_NCCL_PROTOCOL}"

# MLflow destination. Point these at your own tracking server / experiment.
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-staple/stem}"

torchrun \
    --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id=${JOB_NAME} \
    --rdzv-conf join_timeout=1800,read_timeout=1800 \
    -m torchtitan.train \
    --comm.train_timeout_seconds 600 \
    --module stem_dag \
    --config stem_dag_7b_midfine \
    --hf-assets-path ${BASE_CKPT} \
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE} \
    --parallelism.data-parallel-shard-degree 8 \
    --parallelism.expert-parallel-degree 8 \
    --activation-checkpoint.mode selective \
    --training.local-batch-size 4 \
    --training.global-batch-size 1024 \
    --checkpoint.enable \
    --checkpoint.folder "${FSX_CKPT_DIR}" \
    --checkpoint.async_mode disabled \
    --checkpoint.interval 500 \
    --checkpoint.initial_load_path ${INIT_CKPT} \
    --checkpoint.no_initial_load_model_only \
    --checkpoint.exclude_from_loading dataloader,lr_scheduler,train_state \
    --checkpoint.no_last_save_model_only \
    --checkpoint.keep_latest_k 3 \
    --metrics.enable_mlflow \
    --metrics.mlflow_tracking_uri "${MLFLOW_TRACKING_URI}" \
    --metrics.mlflow_experiment_name "${MLFLOW_EXPERIMENT_NAME}" \
    --dump-folder "${OUTPUT_DIR}"
