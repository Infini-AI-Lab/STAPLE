#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_STAGE=midfine
exec "${SCRIPT_DIR}/_moe_staple_static_1b_e8g8t8_8l_stem12.sh" "$@"
