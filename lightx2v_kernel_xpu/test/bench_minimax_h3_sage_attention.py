"""Benchmark dense Sage-style attention for MiniMax-H3 DiT on Intel XPU."""

import argparse
import json
import time

import torch

from sycl_kernels import cute_sdp, minimax_h3_sage_attention


def timed(callable_, warmup, iterations):
    for _ in range(warmup):
        callable_()
    torch.xpu.synchronize()
    begin = time.perf_counter()
    for _ in range(iterations):
        callable_()
    torch.xpu.synchronize()
    return (time.perf_counter() - begin) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(21349, 41773), nargs="+", default=(21349, 41773))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is required")

    results = []
    for tokens in args.tokens:
        q = torch.randn((1, tokens, 56, 128), device="xpu", dtype=torch.bfloat16) * 0.1
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        dense_ms = timed(lambda: cute_sdp(q, k, v), args.warmup, args.iterations)
        latency = timed(lambda: minimax_h3_sage_attention(q, k, v), args.warmup, args.iterations)
        results.append({"tokens": tokens, "heads": 56, "head_dim": 128,
                        "dtype": "bfloat16", "dense_ms": dense_ms,
                        "sage_ms": latency, "speedup": dense_ms / latency})
        print(f"tokens={tokens} heads=56 dense={dense_ms:.2f} ms "
              f"sage={latency:.2f} ms speedup={dense_ms / latency:.3f}x")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
