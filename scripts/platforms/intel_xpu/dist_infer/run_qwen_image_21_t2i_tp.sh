#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/Qwen-Image-2.1}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/dist_infer/qwen_image_21_t2i_tp.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_qwen_image_21_t2i_tp.png}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH=${PYTHONPATH:-}

source "${lightx2v_path}/scripts/base/base.sh"
mkdir -p "$(dirname -- "${output_path}")"

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls qwen_image_21 \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-A capybara wearing a wizard hat, oil painting}" \
  --size "${HEIGHT:-1024}" "${WIDTH:-1024}" \
  --seed "${SEED:-42}" \
  --save_result_path "${output_path}"
