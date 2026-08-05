#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_JSON=${CONFIG_JSON:-${REPO_ROOT}/configs/minimax_h3/minimax_h3_t2av_tp8_no_offload.json}
export OUTPUT_PATH=${OUTPUT_PATH:-${REPO_ROOT}/outputs/minimax_h3_t2av_tp8_no_offload.mp4}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export LIGHTX2V_XPU_DEVICE_MAP=${LIGHTX2V_XPU_DEVICE_MAP:-0,1,2,3,4,5,6,7}

exec "${SCRIPT_DIR}/run_minimax_h3_t2av_ulysses.sh"
