#!/usr/bin/env python3
"""Benchmark eager and torch.compile row-wise dynamic INT8 quantization."""

import argparse
import statistics
import time

import torch

import sycl_kernels


def quantize_int8_rowwise(x):
    x_float = x.float()
    scales = (x_float.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-30)
    quantized = torch.round(x_float / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile", "sycl"), default="compile")
    parser.add_argument("--m", type=int, default=9650)
    parser.add_argument("--k", type=int, default=5376)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    operation = quantize_int8_rowwise
    if args.mode == "compile":
        operation = torch.compile(operation, fullgraph=True)
    elif args.mode == "sycl":
        operation = sycl_kernels.quantize_int8_rowwise

    x = torch.randn((args.m, args.k), device="xpu", dtype=torch.bfloat16)
    for _ in range(args.warmup):
        operation(x)
    torch.xpu.synchronize()

    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        quantized, scales = operation(x)
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    print(
        f"mode={args.mode} shape=[{args.m}, {args.k}] "
        f"median={statistics.median(samples):.3f} ms min={min(samples):.3f} ms"
    )
    print(f"output={tuple(quantized.shape)} {quantized.dtype}, scales={tuple(scales.shape)} {scales.dtype}")


if __name__ == "__main__":
    main()
