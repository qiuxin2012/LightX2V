#!/usr/bin/env python3
"""Compare BF16 and W8A8 GEMMs for the MiniMax-H3 TP=2 shapes.

The defaults reproduce the two FFN GEMMs identified in the rank-0 VTune
capture.  INT8 timing includes dynamic row-wise activation quantization,
oneDNN GEMM, scaling, and BF16 output conversion, matching inference.
"""

import argparse
from pathlib import Path
import statistics
import sys
import time

# Prefer the extension built in this checkout.  The installed package may not
# expose the prequantized benchmark API yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import sycl_kernels
import torch
import torch.nn.functional as F


CASES = {
    "qkv": (5376, 3584),
    "attn_out": (3584, 5376),
    "ffn_in": (5376, 14336),
    "ffn_out": (7168, 5376),
}


def measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    torch.xpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def stats(samples, m, k, n):
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "min_ms": min(samples),
        "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "tflops": 2.0 * m * k * n / (median_ms * 1e9),
    }


def benchmark_case(name, m, k, n, warmup, iterations):
    x = torch.randn((m, k), device="xpu", dtype=torch.bfloat16)

    # F.linear consumes the checkpoint-native contiguous [N, K] layout.
    bf16_weight = torch.randn((n, k), device="xpu", dtype=torch.bfloat16)
    bf16_samples = measure(lambda: F.linear(x, bf16_weight), warmup, iterations)
    del bf16_weight
    torch.xpu.empty_cache()

    int8_weight = torch.randint(-127, 128, (n, k), device="xpu", dtype=torch.int8)
    weight_scales = torch.ones((n,), device="xpu", dtype=torch.float32)
    int8_samples = measure(
        lambda: sycl_kernels.onednn_w8a8_int8(x, int8_weight, weight_scales, None),
        warmup,
        iterations,
    )

    # Quantize once so this path measures the same oneDNN W8A8 GEMM and output
    # conversion without charging each invocation for row-wise quantization.
    quantized_x, x_scales = sycl_kernels.quantize_int8_rowwise(x)
    torch.xpu.synchronize()
    int8_prequantized_samples = measure(
        lambda: sycl_kernels.onednn_w8a8_int8_prequantized(
            x, quantized_x, x_scales, int8_weight, weight_scales, None
        ),
        warmup,
        iterations,
    )
    quantize_samples = measure(
        lambda: sycl_kernels.quantize_int8_rowwise(x), warmup, iterations
    )

    bf16 = stats(bf16_samples, m, k, n)
    int8 = stats(int8_samples, m, k, n)
    int8_prequantized = stats(int8_prequantized_samples, m, k, n)
    quantize_ms = statistics.median(quantize_samples)
    print(f"\n{name}: [{m}, {k}] x [{n}, {k}]^T -> [{m}, {n}]")
    print(
        f"  BF16       median={bf16['median_ms']:8.3f} ms  "
        f"min={bf16['min_ms']:8.3f} ms  p95={bf16['p95_ms']:8.3f} ms  "
        f"{bf16['tflops']:7.2f} TFLOPS"
    )
    print(
        f"  INT8 W8A8  median={int8['median_ms']:8.3f} ms  "
        f"min={int8['min_ms']:8.3f} ms  p95={int8['p95_ms']:8.3f} ms  "
        f"{int8['tflops']:7.2f} TFLOPS"
    )
    print(
        f"  INT8 no-Q  median={int8_prequantized['median_ms']:8.3f} ms  "
        f"min={int8_prequantized['min_ms']:8.3f} ms  "
        f"p95={int8_prequantized['p95_ms']:8.3f} ms  "
        f"{int8_prequantized['tflops']:7.2f} TFLOPS"
    )
    print(f"  rowwise-Q  median={quantize_ms:8.3f} ms")
    print(f"  full INT8 speedup: {bf16['median_ms'] / int8['median_ms']:.3f}x")
    print(
        "  no-Q INT8 speedup: "
        f"{bf16['median_ms'] / int8_prequantized['median_ms']:.3f}x"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("ffn", "all", *CASES),
        default="ffn",
        help="FFN cases reproduce the slow 2560x4 VTune kernel (default: ffn)",
    )
    parser.add_argument("--m", type=int, default=9650, help="TP/SP-local token count")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable")
    if args.m <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("--m and --iterations must be positive; --warmup must be non-negative")

    if args.case == "ffn":
        selected = ("ffn_in", "ffn_out")
    elif args.case == "all":
        selected = tuple(CASES)
    else:
        selected = (args.case,)

    print(f"device={torch.xpu.get_device_name()}  M={args.m}  warmup={args.warmup}  iterations={args.iterations}")
    for name in selected:
        k, n = CASES[name]
        benchmark_case(name, args.m, k, n, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
