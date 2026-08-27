#!/usr/bin/env python3
"""Standalone MiniMax-H3 text-encoder benchmark for Intel XPU."""

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
    parser.add_argument("--prompt", default="A cinematic fox walking through a snowy forest")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output-json")
    parser.add_argument("--host-pinned", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--release-block-offload-buffers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--offload-granularity", choices=("block", "model"))
    return parser.parse_args()


def synchronize():
    torch.xpu.synchronize()


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable; run with PLATFORM=intel_xpu in an XPU-enabled environment")

    # Import only after argument parsing and the device check so --help also
    # works on development hosts without an XPU runtime.
    from lightx2v.models.input_encoders.hf.minimax_h3 import MiniMaxH3Qwen3VLTextEncoder

    with open(args.config_json, encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_path"] = args.model_path
    overrides = {
        "text_encoder_host_pinned": args.host_pinned,
        "text_encoder_release_block_offload_buffers": args.release_block_offload_buffers,
        "text_encoder_cpu_offload": args.cpu_offload,
        "text_encoder_offload_granularity": args.offload_granularity,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    torch.set_grad_enabled(False)
    synchronize()
    load_start = time.perf_counter()
    encoder = MiniMaxH3Qwen3VLTextEncoder(config)
    synchronize()
    load_seconds = time.perf_counter() - load_start

    token_count = len(encoder.tokenizer(args.prompt, add_special_tokens=False)["input_ids"])
    for index in range(args.warmup):
        encoder.infer(args.prompt)
        synchronize()
        print(f"warmup {index + 1}/{args.warmup} complete", flush=True)

    durations = []
    output = None
    for index in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        output = encoder.infer(args.prompt)
        synchronize()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"iteration {index + 1}/{args.iterations}: {elapsed:.6f} s", flush=True)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "config_json": str(Path(args.config_json).resolve()),
        "token_count": token_count,
        "load_seconds": load_seconds,
        "iterations": args.iterations,
        "duration_seconds": durations,
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "prompt_embeds_shape": list(output["prompt_embeds"].shape),
        "effective_text_encoder_config": {
            key: config.get(key)
            for key in (
                "text_encoder_cpu_offload",
                "text_encoder_offload_granularity",
                "text_encoder_host_pinned",
                "text_encoder_release_block_offload_buffers",
                "text_encoder_quantized",
                "text_encoder_quant_scheme",
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
