#!/usr/bin/env python3
"""Standalone MiniMax-H3 DiT denoising benchmark for Intel XPU."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--text-tokens", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-json")
    parser.add_argument("--trace-last-step", action="store_true")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--offload-granularity", choices=("block", "model"))
    return parser.parse_args()


def synchronize():
    torch.xpu.synchronize()


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")
    if args.text_tokens < 1:
        raise ValueError("--text-tokens must be >= 1")
    if args.denoising_steps is not None and args.denoising_steps < 1:
        raise ValueError("--denoising-steps must be >= 1")
    if args.num_layers is not None and args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable; run with PLATFORM=intel_xpu in an XPU-enabled environment")

    from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
    from lightx2v.models.networks.minimax_h3.packing import TEXT_TAG
    from lightx2v.models.schedulers.minimax_h3 import MiniMaxH3Scheduler
    from lightx2v.utils.envs import GET_DTYPE

    with open(args.config_json, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "model_path": args.model_path,
            "model_cls": "minimax_h3",
            "task": "t2av",
            "seq_parallel": False,
            "tensor_parallel": False,
            "cfg_parallel": False,
            "device_mesh": None,
        }
    )
    if args.cpu_offload is not None:
        config["cpu_offload"] = args.cpu_offload
    if args.offload_granularity is not None:
        config["offload_granularity"] = args.offload_granularity
    if args.denoising_steps is not None:
        # The scheduler stores sigma grid points, so N model evaluations need
        # N + 1 grid points.
        config["infer_steps"] = args.denoising_steps + 1
    if args.num_layers is not None:
        config["num_layers"] = args.num_layers

    height = args.height or int(config["target_height"])
    width = args.width or int(config["target_width"])
    num_frames = args.num_frames or int(config["target_video_length"])
    context_dim = int(config.get("caption_channels", 5120))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt_embeds = torch.randn((args.text_tokens, context_dim), generator=generator, dtype=torch.float32).to("xpu")
    text_token_tags = torch.full((args.text_tokens,), TEXT_TAG, dtype=torch.int32)
    inputs = {"text_encoder_output": {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}}

    torch.set_grad_enabled(False)
    init_device = torch.device("cpu" if config.get("cpu_offload", False) else "xpu")
    synchronize()
    load_start = time.perf_counter()
    scheduler = MiniMaxH3Scheduler(config)
    model = MiniMaxH3Model(model_path=args.model_path, config=config, device=init_device)
    model.set_scheduler(scheduler)
    if config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "model":
        model.to_cuda()
    synchronize()
    load_seconds = time.perf_counter() - load_start

    def prepare():
        scheduler.prepare(
            seed=args.seed,
            num_frames=num_frames,
            height=height,
            width=width,
            text_token_tags=text_token_tags,
        )

    def set_trace_state(action):
        result_dir = os.environ.get("VTUNE_RESULT_DIR")
        vtune_cli = os.environ.get("VTUNE_CLI", "vtune")
        if not result_dir:
            raise RuntimeError("--trace-last-step requires VTUNE_RESULT_DIR")
        result = subprocess.run(
            (vtune_cli, "-command", action, "-result-dir", result_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if output:
            print(output.rstrip(), flush=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"VTune failed to {action} collection in {result_dir!r} "
                f"(exit code {result.returncode})"
            )

    def denoise(run_name, trace_last_step=False):
        step_durations = []
        for step_index in range(scheduler.infer_steps):
            print(f"{run_name} DiT step {step_index + 1}/{scheduler.infer_steps} started", flush=True)
            is_last_step = step_index == scheduler.infer_steps - 1
            if trace_last_step and is_last_step:
                set_trace_state("resume")
                print("VTune resumed for the final DiT step", flush=True)
            synchronize()
            step_start = time.perf_counter()
            scheduler.step_pre(step_index)
            model.infer(inputs)
            scheduler.step_post()
            synchronize()
            step_elapsed = time.perf_counter() - step_start
            step_durations.append(step_elapsed)
            if trace_last_step and is_last_step:
                set_trace_state("pause")
                print("VTune paused after the final DiT step", flush=True)
            print(
                f"{run_name} DiT step {step_index + 1}/{scheduler.infer_steps}: "
                f"{step_elapsed:.6f} s",
                flush=True,
            )
        return step_durations

    for index in range(args.warmup):
        prepare()
        denoise(f"warmup {index + 1}/{args.warmup}")
        synchronize()
        print(f"warmup {index + 1}/{args.warmup} complete", flush=True)

    durations = []
    iteration_step_durations = []
    for index in range(args.iterations):
        prepare()
        synchronize()
        start = time.perf_counter()
        step_durations = denoise(
            f"iteration {index + 1}/{args.iterations}", trace_last_step=args.trace_last_step
        )
        synchronize()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        iteration_step_durations.append(step_durations)
        print(f"iteration {index + 1}/{args.iterations}: {elapsed:.6f} s", flush=True)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "config_json": str(Path(args.config_json).resolve()),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "text_tokens": args.text_tokens,
        "prompt_embeds_shape": list(prompt_embeds.shape),
        "dtype": str(GET_DTYPE()),
        "denoising_steps": scheduler.infer_steps,
        "num_layers": int(config.get("num_layers", 50)),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trace_last_step": args.trace_last_step,
        "load_seconds": load_seconds,
        "duration_seconds": durations,
        "dit_step_duration_seconds": iteration_step_durations,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "packed_sequence_length": scheduler.layout.sequence_length,
        "video_rows": scheduler.video_latents.shape[0],
        "audio_rows": scheduler.audio_latents.shape[0],
        "effective_dit_config": {
            key: config.get(key)
            for key in (
                "cpu_offload",
                "offload_granularity",
                "dit_quantized",
                "dit_quant_scheme",
                "attn_type",
                "rms_type",
                "rope_type",
                "use_compile",
            )
        },
    }
    print("BENCHMARK_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
