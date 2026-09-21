#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export CONFIG_JSON=${CONFIG_JSON:-${REPO_ROOT}/configs/platforms/intel_xpu/qwen_image_21_t2i_int8.json}
export OUTPUT_PATH=${OUTPUT_PATH:-${REPO_ROOT}/save_results/qwen_image_21_xpu_t2i_int8.png}
exec "${SCRIPT_DIR}/run_qwen_image_21_t2i.sh" "$@"
