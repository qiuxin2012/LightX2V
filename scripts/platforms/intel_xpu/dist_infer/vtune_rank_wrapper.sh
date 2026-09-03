#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
result_root=${VTUNE_OUTPUT_DIR:-${repo_root}/logs/vtune}
global_rank=${RANK:-unknown}
local_rank=${LOCAL_RANK:-unknown}
profile_ranks=,${VTUNE_RANKS:-0},
vtune_bin=${VTUNE_BIN:-/root/oneapi/vtune/latest/bin64/vtune}

if [[ "${profile_ranks}" != *,"${global_rank}",* ]]; then
  exec python -m lightx2v.infer "$@"
fi

run_id=${VTUNE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
result_dir=${result_root}/rank_${global_rank}_local_${local_rank}_${run_id}
mkdir -p "${result_root}"

exec "${vtune_bin}" \
  -collect xpu-offload \
  -knob collect-programming-api=true \
  -knob enable-characterization-insights=false \
  -knob enable-stack-collection=false \
  -no-follow-child \
  -result-dir "${result_dir}" \
  -- python -m lightx2v.infer "$@"
