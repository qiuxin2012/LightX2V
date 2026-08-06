#!/bin/bash

lightx2v_path=/llm/xin/LightX2V-main
model_path=/llm/models/MiniMax-H3

export PLATFORM=intel_xpu

source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_tp.json \
--prompt "A cinematic fox walking through a snowy forest" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp4.mp4 \
--seed 42
