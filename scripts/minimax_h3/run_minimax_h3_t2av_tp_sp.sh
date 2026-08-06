#!/bin/bash

lightx2v_path=/llm/xin/LightX2V-main
model_path=/llm/models/MiniMax-H3

export PLATFORM=intel_xpu

source ${lightx2v_path}/scripts/base/base.sh
export PYTHONPATH=${lightx2v_path}/lightx2v_kernel_xpu/python:${PYTHONPATH}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

# Stable oneCCL paths for the validated MiniMax-H3 TP4+SP2 setup.  Keep these
# before torchrun so they are visible when XCCL process groups are created.
export CCL_SYCL_ALLTOALL_ARC_LL=${CCL_SYCL_ALLTOALL_ARC_LL:-1}
export CCL_SYCL_ALLTOALL_TMP_BUF=${CCL_SYCL_ALLTOALL_TMP_BUF:-1}
export CCL_SYCL_CCL_BARRIER=${CCL_SYCL_CCL_BARRIER:-1}
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-4294967296}
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-4294967296}
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-4294967296}

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_tp_sp.json \
--prompt "A vintage steam train crosses a stone bridge in the misty mountains at sunrise, with billowing smoke, rhythmic wheel clatter, and a distant whistle" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp4_sp2.mp4 \
--seed 42
