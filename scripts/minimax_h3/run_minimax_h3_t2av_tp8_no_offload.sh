#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3/FL2VA}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_tp8_no_offload.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/outputs/minimax_h3_t2av_tp8_no_offload.mp4}
num_processes=${NUM_PROCESSES:-8}
dtype=${DTYPE:-BF16}
sensitive_layer_dtype=${SENSITIVE_LAYER_DTYPE:-FP32}

export PLATFORM=${PLATFORM:-intel_xpu}
export LIGHTX2V_XPU_DEVICE_MAP=${LIGHTX2V_XPU_DEVICE_MAP:-0,1,2,3,4,5,6,7}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH=${PYTHONPATH:-}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=${dtype}
export SENSITIVE_LAYER_DTYPE=${sensitive_layer_dtype}
mkdir -p "$(dirname -- "${output_path}")"

torchrun --standalone --nproc_per_node="${num_processes}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-A cinematic fox walking through a snowy forest, with natural synchronized ambience.}" \
  --seed "${SEED:-42}" \
  --save_result_path "${output_path}"
