"""Benchmark BF16 Flash Attention with FP16 versus FP32 SxV accumulation."""

import argparse
import time

import sycl_kernels
import torch


HEAD_DIM = 128


def benchmark(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=19921)
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--torch-sdpa", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    kv_len = args.q_len if args.kv_len is None else args.kv_len

    if not torch.xpu.is_available():
        raise RuntimeError("This benchmark requires an Intel XPU device")
    if args.device < 0 or args.device >= torch.xpu.device_count():
        raise ValueError(f"invalid XPU device index {args.device}")

    torch.xpu.set_device(args.device)
    device = torch.device(f"xpu:{args.device}")
    torch.manual_seed(args.seed)
    shape_q = (1, args.q_len, args.heads, HEAD_DIM)
    shape_kv = (1, kv_len, args.heads, HEAD_DIM)
    q = torch.randn(shape_q, dtype=torch.bfloat16, device=device)
    k = torch.randn(shape_kv, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape_kv, dtype=torch.bfloat16, device=device)

    def run_fp16_accum():
        return sycl_kernels.sdp(q, k, v, use_fp32_accum=False)

    def run_fp32_accum():
        return sycl_kernels.sdp(q, k, v, use_fp32_accum=True)

    out_fp16 = run_fp16_accum()
    out_fp32 = run_fp32_accum()
    torch.xpu.synchronize()
    diff = (out_fp16.float() - out_fp32.float()).abs()
    max_diff = diff.max().item()
    relative_rms = (
        diff.square().sum() / out_fp32.float().square().sum().clamp_min(1e-20)
    ).sqrt().item()

    fp16_ms = benchmark(run_fp16_accum, args.warmup, args.iterations)
    fp32_ms = benchmark(run_fp32_accum, args.warmup, args.iterations)
    flops = 4.0 * args.q_len * kv_len * args.heads * HEAD_DIM
    fp16_tflops = flops / (fp16_ms / 1000.0) / 1e12
    fp32_tflops = flops / (fp32_ms / 1000.0) / 1e12

    props = torch.xpu.get_device_properties(args.device)
    print(f"device:       {props.name} (index={args.device}, pci_id=0x{props.device_id:04x})")
    print(f"hardware:     EUs={props.gpu_eu_count}, subslices={props.gpu_subslice_count}, driver={props.driver_version}")
    print(f"shape:        Q={shape_q}, K/V={shape_kv}")
    print(f"FP16 accum:   {fp16_ms:.3f} ms  {fp16_tflops:.2f} TFLOPS")
    print(f"FP32 accum:   {fp32_ms:.3f} ms  {fp32_tflops:.2f} TFLOPS")
    print(f"FP32 / FP16:  {fp32_ms / fp16_ms:.3f}x latency")
    print(f"output diff:  max={max_diff:.6f}, relative_rms={relative_rms:.3e}")

    if args.torch_sdpa:
        q_sdpa = q.permute(0, 2, 1, 3).contiguous()
        k_sdpa = k.permute(0, 2, 1, 3).contiguous()
        v_sdpa = v.permute(0, 2, 1, 3).contiguous()

        def run_torch_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)

        sdpa_ms = benchmark(run_torch_sdpa, args.warmup, args.iterations)
        sdpa_tflops = flops / (sdpa_ms / 1000.0) / 1e12
        print(f"Torch SDPA:   {sdpa_ms:.3f} ms  {sdpa_tflops:.2f} TFLOPS")
        print(f"FP32 / SDPA:  {fp32_ms / sdpa_ms:.3f}x latency")


if __name__ == "__main__":
    main()
