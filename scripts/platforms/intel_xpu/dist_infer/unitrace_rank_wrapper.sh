#!/usr/bin/env bash
set -euo pipefail

trace_dir=${UNITRACE_OUTPUT_DIR:-unitrace_results}
global_rank=${RANK:-unknown}
local_rank=${LOCAL_RANK:-unknown}
trace_ranks=,${UNITRACE_RANKS:-0},

# Tracing every oneCCL process concurrently can crash in Level Zero tracing
# during collective initialization. Keep all ranks running, but inject
# Unitrace only into the explicitly selected rank(s).
if [[ "${trace_ranks}" != *,"${global_rank}",* ]]; then
  exec python -m lightx2v.infer "$@"
fi

rank_trace_dir=${trace_dir}/rank_${global_rank}_local_${local_rank}
mkdir -p "${rank_trace_dir}"

exec unitrace \
  --chrome-kernel-logging \
  --chrome-device-logging \
  --output-dir-path "${rank_trace_dir}" \
  --follow-child-process 0 \
  python -m lightx2v.infer "$@"
