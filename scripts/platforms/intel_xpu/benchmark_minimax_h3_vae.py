#!/usr/bin/env python3
"""Standalone MiniMax-H3 video/audio VAE decode benchmark for Intel XPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--component", choices=("video", "audio", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def synchronize():
    torch.xpu.synchronize()


def summarize(durations):
    return {
        "duration_seconds": durations,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
    }


def benchmark(name, decode, latents, warmup, iterations):
    for index in range(warmup):
        output = decode(latents)
        synchronize()
        del output
        print(f"{name} warmup {index + 1}/{warmup} complete", flush=True)

    durations = []
    output = None
    for index in range(iterations):
        synchronize()
        start = time.perf_counter()
        output = decode(latents)
        synchronize()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"{name} iteration {index + 1}/{iterations}: {elapsed:.6f} s", flush=True)

    result = summarize(durations)
    result["input_shape"] = list(latents.shape)
    result["output_shape"] = list(output.shape)
    del output
    return result


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable; run with PLATFORM=intel_xpu in an XPU-enabled environment")

    # Delay project imports so --help works without an XPU runtime.
    from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
    from lightx2v.models.networks.minimax_h3.packing import audio_latent_num_frames, video_latent_num_frames
    from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
    from lightx2v.utils.envs import DTYPE_MAP

    with open(args.config_json, encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_path"] = args.model_path
    if args.cpu_offload is not None:
        config["vae_cpu_offload"] = args.cpu_offload

    height = args.height or int(config["target_height"])
    width = args.width or int(config["target_width"])
    num_frames = args.num_frames or int(config["target_video_length"])
    spatial_scale = int(config.get("vae_spatial_scale_factor", 16))
    if height % spatial_scale or width % spatial_scale:
        raise ValueError(f"height and width must be divisible by VAE scale {spatial_scale}")

    latent_frames = video_latent_num_frames(num_frames)
    audio_frames = audio_latent_num_frames(num_frames)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video_latents = torch.randn(
        (1, int(config.get("in_channels", 24)), latent_frames, height // spatial_scale, width // spatial_scale),
        generator=generator,
        dtype=torch.float32,
    )
    audio_latents = torch.randn(
        (int(config.get("audio_channels", 2)), int(config.get("audio_in_channels", 32)), audio_frames),
        generator=generator,
        dtype=torch.float32,
    )

    torch.set_grad_enabled(False)
    cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
    results = {}
    load_seconds = {}

    if args.component in ("video", "both"):
        synchronize()
        start = time.perf_counter()
        quantized = config.get("video_vae_quantized", False)
        video_vae = MiniMaxH3VideoVAE.from_pretrained(
            args.model_path,
            device="xpu",
            cpu_offload=cpu_offload,
            checkpoint_path=config.get("video_vae_quantized_ckpt") if quantized else None,
            quant_scheme=config.get("video_vae_quant_scheme") if quantized else None,
            sensitive_layer_dtype=DTYPE_MAP[config.get("vae_sensitive_layer_dtype", "fp32")],
            use_compile=config.get("vae_use_compile", False),
            attn_type=config.get("vae_attn_type", "torch_sdpa"),
        )
        tile_shapes = config.get("vae_decode_tile_shape", {})
        tile_shape = tile_shapes.get(f"{height}x{width}") if isinstance(tile_shapes, dict) else None
        if tile_shape:
            video_vae.set_decode_tile_shape(*tile_shape)
        synchronize()
        load_seconds["video"] = time.perf_counter() - start
        results["video"] = benchmark("video", video_vae.decode, video_latents, args.warmup, args.iterations)
        del video_vae

    if args.component in ("audio", "both"):
        synchronize()
        start = time.perf_counter()
        audio_vae = MiniMaxH3AudioVAE.from_pretrained(args.model_path, device="xpu", cpu_offload=cpu_offload)
        synchronize()
        load_seconds["audio"] = time.perf_counter() - start
        results["audio"] = benchmark("audio", audio_vae.decode, audio_latents, args.warmup, args.iterations)
        del audio_vae

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "config_json": str(Path(args.config_json).resolve()),
        "component": args.component,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "vae_cpu_offload": cpu_offload,
        "load_seconds": load_seconds,
        "benchmarks": results,
    }
    print("BENCHMARK_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
