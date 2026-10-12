#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export STEM_STRATEGY=static
export TRAINING_STAGE=warmup
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
exec "${SCRIPT_DIR}/_moe_staple_1b_e8g1t1_8l.sh" "$@"
