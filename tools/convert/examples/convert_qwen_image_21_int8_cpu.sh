#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

source_dir=${SOURCE_DIR:-/llm/models/Qwen-Image-2.1/transformer}
output_dir=${OUTPUT_DIR:-/llm/models/Qwen-Image-2.1/quantized/int8}

python "${REPO_ROOT}/tools/convert/converter.py" \
  --source "${source_dir}" \
  --output "${output_dir}" \
  --output_name qwen_image_21_int8 \
  --model_type qwen_image_21_dit \
  --quantized \
  --linear_type int8 \
  --device cpu \
  --single_file
