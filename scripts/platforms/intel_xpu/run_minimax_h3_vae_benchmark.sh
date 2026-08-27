#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${REPO_ROOT}/configs/platforms/intel_xpu/minimax_h3_t2av.json}
output_json=${OUTPUT_JSON:-${REPO_ROOT}/save_results/minimax_h3_vae_benchmark.json}

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}" >&2; exit 1; }

mkdir -p "$(dirname -- "${output_json}")"

export PLATFORM=${PLATFORM:-intel_xpu}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}

echo "MiniMax-H3 model: ${model_path}"
echo "Benchmark component: ${COMPONENT:-both}"
echo "Benchmark result: ${output_json}"

python "${SCRIPT_DIR}/benchmark_minimax_h3_vae.py" \
  --model-path "${model_path}" \
  --config-json "${config_json}" \
  --component "${COMPONENT:-both}" \
  --warmup "${WARMUP:-0}" \
  --iterations "${ITERATIONS:-1}" \
  --output-json "${output_json}" \
  "$@"
