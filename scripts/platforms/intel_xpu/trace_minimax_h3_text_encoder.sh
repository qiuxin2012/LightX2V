#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${REPO_ROOT}/configs/platforms/intel_xpu/minimax_h3_t2av.json}
trace_dir=${ZE_TRACE_DIR:-${REPO_ROOT}/save_results/ze_trace_minimax_h3_text_encoder}
trace_tool=${ZE_TRACE_TOOL:-unitrace}

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}" >&2; exit 1; }
command -v "${trace_tool}" >/dev/null 2>&1 || {
  echo "ZE trace tool '${trace_tool}' was not found in PATH." >&2
  echo "Source the Intel oneAPI/pti environment, or set ZE_TRACE_TOOL and ZE_TRACE_ARGS." >&2
  exit 1
}

mkdir -p "${trace_dir}"
export PLATFORM=intel_xpu
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export DTYPE=${DTYPE:-BF16}
export SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE:-BF16}
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}
export TOKENIZERS_PARALLELISM=false

if [[ -n "${ZE_TRACE_ARGS:-}" ]]; then
  read -r -a trace_args <<< "${ZE_TRACE_ARGS}"
else
  trace_args=(--chrome-kernel-logging --chrome-device-logging --output-dir-path "${trace_dir}")
fi

echo "Trace output: ${trace_dir}"
echo "Trace command: ${trace_tool} ${trace_args[*]}"
"${trace_tool}" "${trace_args[@]}" \
  python "${SCRIPT_DIR}/benchmark_minimax_h3_text_encoder.py" \
    --model-path "${model_path}" \
    --config-json "${config_json}" \
    --warmup "${WARMUP:-1}" \
    --iterations "${ITERATIONS:-1}" \
    --output-json "${trace_dir}/benchmark.json" \
    "$@"
