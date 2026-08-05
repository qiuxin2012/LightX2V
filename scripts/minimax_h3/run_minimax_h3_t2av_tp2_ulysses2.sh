#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_JSON=${CONFIG_JSON:-${REPO_ROOT}/configs/minimax_h3/minimax_h3_t2av_tp2_ulysses2_block_offload.json}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
# Diagnostic mapping: exchange physical devices 0 and 1 while preserving the
# distributed rank topology. Override with 0,1,2,3 for the normal mapping.
export LIGHTX2V_XPU_DEVICE_MAP=${LIGHTX2V_XPU_DEVICE_MAP:-1,0,2,3}

exec "${SCRIPT_DIR}/run_minimax_h3_t2av_ulysses.sh"
