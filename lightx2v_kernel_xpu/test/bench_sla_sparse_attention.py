#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def summarize(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure_peak_memory(fn):
    """Return peak XPU memory added by one invocation of ``fn``.

    Inputs captured by fn are already resident, so peak_delta_bytes measures the
    operator's output and temporary allocations rather than Q/K/V storage.
    """
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    baseline = torch.xpu.memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    result = fn()
    torch.xpu.synchronize()
    peak = torch.xpu.max_memory_allocated()
    measurement = {
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "peak_delta_bytes": max(0, peak - baseline),
        "peak_delta_mib": max(0, peak - baseline) / (1024**2),
    }
    del result
    torch.xpu.empty_cache()
    return measurement


def output_quality(actual, expected):
    """Compute SLA-vs-dense error metrics with FP32 reductions."""
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    error = actual_fp32 - expected_fp32
    actual_norm = torch.linalg.vector_norm(actual_fp32)
    expected_norm = torch.linalg.vector_norm(expected_fp32)
    eps = torch.finfo(torch.float32).eps
    metrics = {
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (torch.linalg.vector_norm(error) / expected_norm.clamp_min(eps)).item(),
        "cosine_similarity": (
            (actual_fp32 * expected_fp32).sum()
            / (actual_norm * expected_norm).clamp_min(eps)
        ).item(),
    }
    del actual_fp32, expected_fp32, error
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="_cmake_build")
    parser.add_argument("--text-tokens", type=int, default=2047)
    parser.add_argument("--audio-tokens", type=int, default=414)
    parser.add_argument("--video-tokens", type=int, default=39312)
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="override the sum of text/audio/video tokens for a synthetic custom layout",
    )
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--keep-ratio", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="also write the JSON result to this file")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("No Intel XPU is available")
    if not 0 < args.keep_ratio <= 1:
        parser.error("--keep-ratio must be in (0, 1]")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    modality_lengths = {
        "text": args.text_tokens,
        "audio": args.audio_tokens,
        "video": args.video_tokens,
    }
    if any(length < 0 for length in modality_lengths.values()):
        parser.error("modality token counts must be non-negative")
    packed_length = sum(modality_lengths.values())
    sequence_length = args.sequence_length if args.sequence_length is not None else packed_length
    if sequence_length < 1:
        parser.error("packed sequence length must be positive")
    torch.xpu.set_device(args.device)
    torch.manual_seed(args.seed)

    build_dir = Path(args.build_dir).resolve()
    torch.ops.load_library(str(build_dir / "cute_fmha_minimax_h3_sparse_torch.so"))
    torch.ops.load_library(str(build_dir / "cute_fmha_minimax_h3_torch.so"))
    from sycl_kernels.sla import sla_block_map

    shape = (1, sequence_length, args.heads, 128)
    q = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lut = sla_block_map(q, k, args.keep_ratio, 128, 128)

    sparse_fn = lambda: torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp(q, k, v, lut)
    router_fn = lambda: sla_block_map(q, k, args.keep_ratio, 128, 128)
    dense_fn = lambda: torch.ops.sycl_kernels_cute_minimax_h3.sdp(q, k, v)

    def sla_end_to_end():
        routed_lut = router_fn()
        return torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp(q, k, v, routed_lut)

    sparse_ms = measure(sparse_fn, args.warmup, args.iterations)
    router_ms = measure(router_fn, args.warmup, args.iterations)
    dense_ms = measure(dense_fn, args.warmup, args.iterations)
    end_to_end_ms = measure(sla_end_to_end, args.warmup, args.iterations)
    sparse_median = statistics.median(sparse_ms)
    dense_median = statistics.median(dense_ms)
    end_to_end_median = statistics.median(end_to_end_ms)

    memory = {
        "dense": measure_peak_memory(dense_fn),
        "sla_kernel_only_prebuilt_lut": measure_peak_memory(sparse_fn),
        "sla_end_to_end": measure_peak_memory(sla_end_to_end),
    }
    dense_output = dense_fn()
    sparse_output = sparse_fn()
    torch.xpu.synchronize()
    quality = output_quality(sparse_output, dense_output)

    result = {
        "device": torch.xpu.get_device_name(args.device),
        "torch_version": torch.__version__,
        "shape_blhd": list(shape),
        "packed_layout": {
            **modality_lengths,
            "total": sequence_length,
            "matches_modality_sum": sequence_length == packed_length,
        },
        "dtype": str(q.dtype),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "keep_ratio_requested": args.keep_ratio,
        "keep_ratio_effective": lut.shape[-1] / ((sequence_length + 127) // 128),
        "lut_shape": list(lut.shape),
        "sla_router_device_kernels": 2,
        "sla_router_kernel_breakdown": [
            "fused_qk_block_pool",
            "xmx_dpas_score_topk_lut",
        ],
        "latency": {
            "dense": summarize(dense_ms),
            "sla_kernel_only_prebuilt_lut": summarize(sparse_ms),
            "sla_router": summarize(router_ms),
            "sla_end_to_end": summarize(end_to_end_ms),
        },
        "speedup": {
            "sla_kernel_only_vs_dense": dense_median / sparse_median,
            "sla_end_to_end_vs_dense": dense_median / end_to_end_median,
        },
        "memory": memory,
        "output_quality_vs_dense": quality,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
