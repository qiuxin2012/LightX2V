#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/Qwen-Image-2.1}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/qwen_image_21_t2i.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/qwen_image_21_xpu_t2i.png}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export PLATFORM=${PLATFORM:-intel_xpu}
export DTYPE=${DTYPE:-BF16}
export SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE:-None}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:-}

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}" >&2; exit 1; }
mkdir -p "$(dirname -- "${output_path}")"

source "${lightx2v_path}/scripts/base/base.sh"

prompt=${PROMPT:-A small red panda reading a book beside a warm lamp, watercolor illustration}
height=${HEIGHT:-256}
width=${WIDTH:-256}
seed=${SEED:-42}

echo "Running Qwen-Image-2.1 XPU smoke test on ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
echo "size=${height}x${width}, config=${config_json}, output=${output_path}"

python -m lightx2v.infer \
  --model_cls qwen_image_21 \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${prompt}" \
  --size "${height}" "${width}" \
  --seed "${seed}" \
  --save_result_path "${output_path}"
