#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_JSON=${CONFIG_JSON:-${REPO_ROOT}/configs/minimax_h3/minimax_h3_t2av_tp2_ulysses4_block_offload.json}
export NUM_PROCESSES=${NUM_PROCESSES:-8}

exec "${SCRIPT_DIR}/run_minimax_h3_t2av_ulysses.sh"
