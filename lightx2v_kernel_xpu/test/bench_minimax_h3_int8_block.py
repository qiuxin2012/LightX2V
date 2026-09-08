#!/usr/bin/env python3
"""Replay the MiniMax-H3 TP=2 linear workload for complete transformer blocks.

This deliberately excludes attention and collectives. It answers whether the
six main Linear calls and SwiGLU alone reproduce the accumulated INT8 GEMM
regression seen in a full-model VTune capture.
"""

import argparse
import statistics
import time

import sycl_kernels
import torch
import torch.nn.functional as F


SHAPES = {
    "q": (5376, 3584),
    "k": (5376, 3584),
    "v": (5376, 3584),
    "attn_out": (3584, 5376),
    "ffn_in": (5376, 14336),
    "ffn_out": (7168, 5376),
}


def make_weights(dtype):
    weights = {}
    scales = {}
    for name, (k, n) in SHAPES.items():
        if dtype == torch.int8:
            weights[name] = torch.randint(-127, 128, (n, k), device="xpu", dtype=dtype)
            scales[name] = torch.ones((n,), device="xpu", dtype=torch.float32)
        else:
            weights[name] = torch.randn((n, k), device="xpu", dtype=dtype)
    return weights, scales


def linear(x, name, weights, scales, int8):
    if int8:
        return sycl_kernels.onednn_w8a8_int8(x, weights[name], scales[name], None)
    return F.linear(x, weights[name])


def replay_block(hidden, weights, scales, int8):
    q = linear(hidden, "q", weights, scales, int8)
    k = linear(hidden, "k", weights, scales, int8)
    v = linear(hidden, "v", weights, scales, int8)

    # Attention is intentionally omitted. Preserve its TP-local input shape so
    # attn_out exercises exactly [M, 3584] x [5376, 3584]^T.
    attn_result = (q + k + v) * (1.0 / 3.0)
    hidden = hidden + linear(attn_result, "attn_out", weights, scales, int8)

    value, gate = linear(hidden, "ffn_in", weights, scales, int8).chunk(2, dim=-1)
    hidden = hidden + linear(value * F.silu(gate), "ffn_out", weights, scales, int8)
    return hidden


def measure(mode, m, blocks, warmup, iterations):
    int8 = mode == "int8"
    weights, scales = make_weights(torch.int8 if int8 else torch.bfloat16)
    hidden = torch.randn((m, 5376), device="xpu", dtype=torch.bfloat16)

    def workload():
        output = hidden
        for _ in range(blocks):
            output = replay_block(output, weights, scales, int8)
        return output

    for _ in range(warmup):
        workload()
    torch.xpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = workload()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    del output, hidden, weights, scales
    torch.xpu.empty_cache()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=9650)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mode", choices=("both", "bf16", "int8"), default="both")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable")
    if min(args.m, args.blocks, args.iterations) <= 0 or args.warmup < 0:
        parser.error("m, blocks and iterations must be positive; warmup must be non-negative")

    modes = ("bf16", "int8") if args.mode == "both" else (args.mode,)
    results = {}
    print(
        f"device={torch.xpu.get_device_name()} M={args.m} blocks={args.blocks} "
        f"warmup={args.warmup} iterations={args.iterations}"
    )
    for mode in modes:
        samples = measure(mode, args.m, args.blocks, args.warmup, args.iterations)
        median = statistics.median(samples)
        results[mode] = median
        print(f"{mode:5s}: median={median:.3f} ms  per_block={median / args.blocks:.3f} ms")
    if len(results) == 2:
        print(f"speedup: {results['bf16'] / results['int8']:.3f}x")


if __name__ == "__main__":
    main()
