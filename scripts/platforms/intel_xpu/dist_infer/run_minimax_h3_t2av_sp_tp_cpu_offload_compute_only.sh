#!/usr/bin/env bash
set -euo pipefail

# Benchmark-only launcher: TP/SP model collectives are replaced by local
# tensors with the same shapes.  Output values are intentionally invalid.
export LIGHTX2V_SKIP_DISTRIBUTED_COMM=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_minimax_h3_t2av_sp_tp_cpu_offload.sh" "$@"
