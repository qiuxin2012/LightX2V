#!/usr/bin/env python3
"""Benchmark the MiniMax-H3 BF16 GEMM and CUTE FMHA shapes seen in VTune.

The defaults reproduce one rank of the 960x544, 124-frame, SP=2/TP=2 run:
  * GEMM input: [9650, 5376] (8 replicated text + 19284 / 2 main tokens)
  * CUTE FMHA Q/K/V: [1, 19292, {7, 14, 28, 56}, 128]
  * 29 denoising steps and 50 transformer blocks

This is a kernel microbenchmark.  It intentionally excludes tensor-parallel
collectives, sequence-parallel all-to-all, CPU offload, and framework code.
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


GEMM_CASES = (
    # name, M, K, N, logical calls in one 29-step inference
    ("attn_q", 9650, 5376, 3584, 1450),
    ("attn_k", 9650, 5376, 3584, 1450),
    ("attn_v", 9650, 5376, 3584, 1450),
    ("attn_out", 9650, 3584, 5376, 1450),
    ("ffn_in_value_gate", 9650, 5376, 14336, 1450),
    ("ffn_out", 9650, 7168, 5376, 1450),
)


def synchronize():
    torch.xpu.synchronize()


def percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def measure(op, warmup, iterations):
    for _ in range(warmup):
        output = op()
    synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = op()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return output, samples


def sample_stats(samples):
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p95_ms": percentile(samples, 0.95),
    }


def tensor_mib(shape, element_size=2):
    elements = 1
    for extent in shape:
        elements *= extent
    return elements * element_size / (1024**2)


def benchmark_gemm(name, m, k, n, model_calls, warmup, iterations, implementation):
    # Match MiniMax-H3's default MMWeight path exactly:
    #   * checkpoint weights arrive as contiguous Linear [N, K]
    #   * loading transposes them once to runtime [K, N] (transpose strides)
    #   * apply() allocates an output and calls torch.mm(..., out=output)
    x = torch.randn((m, k), device="xpu", dtype=torch.bfloat16)
    checkpoint_weight = torch.randn((n, k), device="xpu", dtype=torch.bfloat16)
    weight = checkpoint_weight.t()
    effective_weight = weight

    if implementation == "mm_out":
        def gemm():
            output = torch.empty((m, n), dtype=x.dtype, device=x.device, requires_grad=False)
            return torch.mm(x, weight, out=output)
    elif implementation == "mm_out_reuse":
        reusable_output = torch.empty((m, n), dtype=x.dtype, device=x.device, requires_grad=False)

        def gemm():
            return torch.mm(x, weight, out=reusable_output)
    elif implementation == "mm_out_contiguous":
        contiguous_weight = weight.contiguous()
        effective_weight = contiguous_weight

        def gemm():
            output = torch.empty((m, n), dtype=x.dtype, device=x.device, requires_grad=False)
            return torch.mm(x, contiguous_weight, out=output)
    elif implementation == "mm":
        def gemm():
            return torch.mm(x, weight)
    elif implementation == "linear":
        def gemm():
            return F.linear(x, checkpoint_weight)
    else:
        raise ValueError(f"unknown GEMM implementation: {implementation}")

    output, samples = measure(gemm, warmup, iterations)
    stats = sample_stats(samples)
    stats.update(
        {
            "name": f"{name}:{implementation}",
            "kind": "gemm",
            "implementation": implementation,
            "input_shape": list(x.shape),
            "checkpoint_weight_shape": list(checkpoint_weight.shape),
            "runtime_weight_shape": list(weight.shape),
            "runtime_weight_stride": list(effective_weight.stride()),
            "output_shape": list(output.shape),
            "input_mib": tensor_mib(x.shape),
            "weight_mib": tensor_mib(weight.shape),
            "output_mib": tensor_mib(output.shape),
            "model_calls": model_calls,
            "tflops": 2.0 * m * n * k / (stats["median_ms"] * 1e9),
            "estimated_model_seconds": stats["median_ms"] * model_calls / 1000.0,
        }
    )
    del output, weight, checkpoint_weight, x
    return stats


def benchmark_attention(sequence_length, heads, head_dim, model_calls, warmup, iterations):
    import sycl_kernels

    shape = (1, sequence_length, heads, head_dim)
    q = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    v = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    output, samples = measure(lambda: sycl_kernels.cute_sdp(q, k, v), warmup, iterations)
    stats = sample_stats(samples)
    # Approximate dense-attention FLOPs: QK^T and softmax(V), each 2*B*H*L^2*D.
    flops = 4.0 * heads * sequence_length * sequence_length * head_dim
    stats.update(
        {
            "name": "cute_fmha",
            "kind": "attention",
            "qkv_shape": list(shape),
            "output_shape": list(output.shape),
            "qkv_mib": 3.0 * tensor_mib(shape),
            "output_mib": tensor_mib(output.shape),
            "model_calls": model_calls,
            "tflops": flops / (stats["median_ms"] * 1e9),
            "estimated_model_seconds": stats["median_ms"] * model_calls / 1000.0,
        }
    )
    del output, v, k, q
    return stats


def print_result(result):
    if result["kind"] == "gemm":
        shape = f'{result["input_shape"]} x {result["runtime_weight_shape"]} -> {result["output_shape"]}'
    else:
        shape = f'Q/K/V {result["qkv_shape"]} -> {result["output_shape"]}'
    print(
        f'{result["name"]:20s} {shape}\n'
        f'  median={result["median_ms"]:.3f} ms  min={result["min_ms"]:.3f} ms  '
        f'p95={result["p95_ms"]:.3f} ms  {result["tflops"]:.2f} TFLOPS  '
        f'calls={result["model_calls"]}  estimated_total={result["estimated_model_seconds"]:.3f} s'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("all", "gemm", "attention"), default="all")
    parser.add_argument(
        "--gemm-implementation",
        choices=("mm_out", "mm_out_reuse", "mm_out_contiguous", "mm", "linear", "compare", "compare_all"),
        default="mm_out",
        help="GEMM call path; compare benchmarks mm_out/linear, compare_all benchmarks every path (default: mm_out)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=19292)
    parser.add_argument("--local-sequence-length", type=int, default=9650)
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=(7, 14, 28, 56),
        help="one or more attention head counts (default: 7 14 28 56)",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=29)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is unavailable")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")

    model_calls = args.steps * args.layers
    results = []
    if args.kernel in ("all", "gemm"):
        if args.gemm_implementation == "compare":
            implementations = ("mm_out", "linear")
        elif args.gemm_implementation == "compare_all":
            implementations = ("mm_out", "mm_out_reuse", "mm_out_contiguous", "mm", "linear")
        else:
            implementations = (args.gemm_implementation,)
        for name, _, k, n, _ in GEMM_CASES:
            for implementation in implementations:
                results.append(
                    benchmark_gemm(
                        name,
                        args.local_sequence_length,
                        k,
                        n,
                        model_calls,
                        args.warmup,
                        args.iterations,
                        implementation,
                    )
                )
                torch.xpu.empty_cache()
        if args.gemm_implementation == "compare_all":
            # Candidate optimization: replace the three equal Q/K/V projections
            # with one projection whose output dimension is three times larger.
            results.append(
                benchmark_gemm(
                    "attn_qkv_fused",
                    args.local_sequence_length,
                    GEMM_CASES[0][2],
                    3 * GEMM_CASES[0][3],
                    model_calls,
                    args.warmup,
                    args.iterations,
                    "mm_out",
                )
            )
            torch.xpu.empty_cache()
    if args.kernel in ("all", "attention"):
        for heads in args.heads:
            results.append(
                benchmark_attention(
                    args.sequence_length,
                    heads,
                    args.head_dim,
                    model_calls,
                    args.warmup,
                    args.iterations,
                )
            )
            torch.xpu.empty_cache()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print_result(result)


if __name__ == "__main__":
    main()
