#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3-official}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_tp2_b65.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp2_b65.mp4}
prompt_cache=${PROMPT_CACHE:-${lightx2v_path}/save_results/minimax_h3_t2av_tp2_b65_prompt_cache.pt}
prompt=${PROMPT:-'integrated_multimodal_description: [Shot 1] Cinematic low-angle tracking shot following a stylish woman from behind as she strolls down a bustling post-rain Tokyo street. The asphalt is completely wet, acting as a black mirror that perfectly reflects the dense canopy of overhead neon signs—warm pink lanterns, icy cyan katakana signage, and giant animated billboards playing silently. She walks slightly to the left of frame, revealing the back of her sleek black leather jacket, which glistens with specular highlights, and the hem of a flowing crimson dress that swirls around her calves. Black heeled boots splash subtly in shallow puddles, and a black leather purse hangs from her shoulder. The camera slowly pushes forward and gently rises, while out-of-focus pedestrians in modern clothing cross the frame, adding life. [Shot 2] At 00:03.500, a sharp cut to a medium profile shot from her right side, camera dollying sideways in perfect sync. She comes into clear view: oversized black sunglasses perched on her nose reflect a giant LED screen across the street, with purple and blue animations gliding across the lenses. Her bold matte red lipstick stands out against fair skin, and a hint of a confident smile plays on her lips. The sharp tailoring of her jacket catches rim light, and her stride is poised and rhythmic. The background is a bokeh of neon blur, while the wet ground distorts the red dress’s reflection into abstract color streaks. A subtle handheld camera shake increases immediacy. [Shot 3] At 00:06.800, a stylized slow-motion frontal medium close-up as she walks directly toward the lens, which pulls back. Time stretches—she casually removes her sunglasses in one smooth motion, revealing sharp winged eyeliner and a piercing gaze that locks directly with the viewer. A shaft of hot pink neon light sweeps across her cheekbones, then she slides the glasses back on with a soft click. The camera then racks focus from her face to the endless corridor of neon-lit street behind her as she walks past, dissolving into a blur of vibrant city lights.
overall_soundscape: Rich city atmosphere on wet streets: a constant damp hiss of car tires rolling through water in the distance, the resonant electrical hum and faint crackle of neon transformers overhead, a muffled J-pop bassline leaking from a nearby record store, layers of pedestrian chatter and soft laughter in Japanese, and in the foreground, the crisp, wet footsteps of her heeled boots striking the mirrored asphalt, with occasional tiny splashes. When she removes her sunglasses, a delicate, intimate "click" of the frame folding is audible, momentarily cutting through the noise.
non_diegetic_music: A lo-fi electronic city-pop track with a relaxed breakbeat and dreamy analog synth pads, setting a confident, seductive mood. As she takes off her sunglasses in slow motion, a warm, soulful saxophone phrase sweeps in with reverb, then gently settles back into the groove as she walks on, gradually fading out with the ambient hum.'}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:-}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}

# torchrun does not provide oneCCL's MPI launcher variables. On this host,
# selecting loopback explicitly avoids oneCCL picking a Docker bridge for KVS.
export CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER:-none}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi}
export CCL_KVS_IFACE=${CCL_KVS_IFACE:-lo}
export FI_TCP_IFACE=${FI_TCP_IFACE:-lo}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
mkdir -p "$(dirname -- "${output_path}")" "$(dirname -- "${prompt_cache}")"

if [[ ${SKIP_ENCODE:-0} != 1 ]]; then
  MINIMAX_H3_PHASE=encode MINIMAX_H3_PROMPT_CACHE="${prompt_cache}" \
  torchrun --standalone --nproc_per_node=2 \
    "${lightx2v_path}/scripts/minimax_h3/minimax_h3_t2av_sequential_load.py" \
    --model_cls minimax_h3 \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --prompt "${prompt}" \
    --save_result_path "${output_path}" \
    --seed "${SEED:-0}"
elif [[ ! -f ${prompt_cache} ]]; then
  echo "Prompt cache does not exist: ${prompt_cache}" >&2
  exit 1
fi

MINIMAX_H3_PHASE=infer MINIMAX_H3_PROMPT_CACHE="${prompt_cache}" \
torchrun --standalone --nproc_per_node=2 \
  "${lightx2v_path}/scripts/minimax_h3/minimax_h3_t2av_sequential_load.py" \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${prompt}" \
  --save_result_path "${output_path}" \
  --seed "${SEED:-0}"
