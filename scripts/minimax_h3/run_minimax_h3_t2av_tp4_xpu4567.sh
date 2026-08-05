#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_JSON=${CONFIG_JSON:-${REPO_ROOT}/configs/minimax_h3/minimax_h3_t2av_tp4_block_offload.json}
export OUTPUT_PATH=${OUTPUT_PATH:-${REPO_ROOT}/outputs/minimax_h3_t2av_tp4_xpu4567.mp4}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-4,5,6,7}
export LIGHTX2V_XPU_DEVICE_MAP=${LIGHTX2V_XPU_DEVICE_MAP:-0,1,2,3}

exec "${SCRIPT_DIR}/run_minimax_h3_t2av_ulysses.sh"
