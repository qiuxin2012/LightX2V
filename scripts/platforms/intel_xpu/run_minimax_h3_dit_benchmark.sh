#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${REPO_ROOT}/configs/platforms/intel_xpu/minimax_h3_t2av.json}
trace_dir=${VTUNE_OUTPUT_DIR:-${REPO_ROOT}/save_results/vtune_minimax_h3_dit}
vtune_cli=${VTUNE_CLI:-vtune}
result_dir=${VTUNE_RESULT_DIR:-${trace_dir}/result_$(date +%Y%m%d_%H%M%S)_$$}
output_json=${OUTPUT_JSON:-${trace_dir}/benchmark.json}
trace_last_step=${TRACE_LAST_STEP:-1}

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}" >&2; exit 1; }
command -v "${vtune_cli}" >/dev/null 2>&1 || {
  echo "VTune CLI '${vtune_cli}' was not found in PATH. Load the VTune environment or set VTUNE_CLI." >&2
  exit 1
}

mkdir -p "${trace_dir}" "$(dirname -- "${output_json}")"

export PLATFORM=${PLATFORM:-intel_xpu}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export DTYPE=${DTYPE:-BF16}
export SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE:-BF16}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}

if [[ -n "${VTUNE_ARGS:-}" ]]; then
  read -r -a trace_args <<< "${VTUNE_ARGS}"
else
  trace_args=(-collect gpu-hotspots)
fi

if [[ "${trace_last_step}" == "1" ]]; then
  trace_args+=(-start-paused)
  benchmark_trace_args=(--trace-last-step)
  echo "Trace scope: final DiT step only"
elif [[ "${trace_last_step}" == "0" ]]; then
  benchmark_trace_args=()
  echo "Trace scope: all DiT steps"
else
  echo "TRACE_LAST_STEP must be 0 or 1, got: ${trace_last_step}" >&2
  exit 1
fi

echo "MiniMax-H3 DiT model: ${model_path}"
echo "Configuration: ${config_json}"
echo "Benchmark result: ${output_json}"
echo "VTune result: ${result_dir}"

if [[ -n "${DENOISING_STEPS:-}" ]]; then
  step_args=(--denoising-steps "${DENOISING_STEPS}")
  echo "Denoising steps: ${DENOISING_STEPS}"
else
  step_args=()
fi

if [[ -n "${DIT_LAYERS:-}" ]]; then
  layer_args=(--num-layers "${DIT_LAYERS}")
  echo "DiT layers per step: ${DIT_LAYERS}"
else
  layer_args=()
fi

export VTUNE_CLI="${vtune_cli}"
export VTUNE_RESULT_DIR="${result_dir}"

"${vtune_cli}" "${trace_args[@]}" -result-dir "${result_dir}" -- \
  python "${SCRIPT_DIR}/benchmark_minimax_h3_dit.py" \
  --model-path "${model_path}" \
  --config-json "${config_json}" \
  --warmup "${WARMUP:-0}" \
  --iterations "${ITERATIONS:-1}" \
  --output-json "${output_json}" \
  "${step_args[@]}" \
  "${layer_args[@]}" \
  "${benchmark_trace_args[@]}" \
  "$@"
