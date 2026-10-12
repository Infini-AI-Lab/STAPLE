#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
export EXPERT_LR="${EXPERT_LR:-2e-4}"
export STEM_STRATEGY=static
export TRAINING_STAGE=midfine
exec "${SCRIPT_DIR}/_moe_staple_1b_e8g1t1_8l.sh" \
    --activation-checkpoint.mode selective \
    "$@"
