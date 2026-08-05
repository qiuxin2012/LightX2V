#!/usr/bin/env python3
"""Stress Intel XPU BF16 GEMM kernels in isolated child processes."""

import argparse
import math
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0", help="Comma-separated physical XPU indices")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--m", type=int, default=422)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=16128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-device", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_child(args):
    import torch

    device_index = args.child_device
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is unavailable; reset the XPU or restart the host/container")
    if device_index >= torch.xpu.device_count():
        raise ValueError(f"xpu:{device_index} does not exist; count={torch.xpu.device_count()}")

    torch.xpu.set_device(device_index)
    device = torch.device("xpu", device_index)
    torch.manual_seed(args.seed)
    x = torch.randn((args.m, args.k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((args.k, args.n), dtype=torch.bfloat16, device=device)
    bias = torch.randn((args.n,), dtype=torch.bfloat16, device=device)
    output = torch.empty((args.m, args.n), dtype=torch.bfloat16, device=device)
    torch.xpu.synchronize(device)

    total = args.warmup + args.iterations
    started = time.perf_counter()
    for iteration in range(total):
        # This matches MMWeight.apply's failing bias path.
        torch.addmm(bias, x, weight, out=output)
        torch.xpu.synchronize(device)
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"non-finite addmm output at iteration {iteration}")

        # Also exercise the no-bias out= path used by other model projections.
        torch.mm(x, weight, out=output)
        torch.xpu.synchronize(device)
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"non-finite mm output at iteration {iteration}")

        # Change inputs so repeated execution cannot return a cached result.
        x.add_(math.ldexp(1.0, -10))
        if iteration + 1 == args.warmup:
            started = time.perf_counter()
        if iteration < args.warmup or (iteration + 1) % 10 == 0:
            checksum = float(output.float().mean().item())
            print(f"xpu:{device_index} iteration={iteration + 1}/{total} checksum={checksum:.6g}", flush=True)

    elapsed = time.perf_counter() - started
    operations = 4.0 * args.iterations * args.m * args.k * args.n
    print(f"PASS xpu:{device_index}: {args.iterations} addmm+mm pairs in {elapsed:.3f}s, {operations / elapsed / 1e12:.3f} TFLOP/s", flush=True)


def run_parent(args):
    devices = [int(value.strip()) for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device index")
    failures = []
    for device in devices:
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--child-device",
            str(device),
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
            "--m",
            str(args.m),
            "--k",
            str(args.k),
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
        ]
        print(f"\n=== Testing physical xpu:{device} ===", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            if result.returncode < 0:
                detail = f"signal {-result.returncode}"
            else:
                detail = f"exit code {result.returncode}"
            print(f"FAIL xpu:{device}: child terminated with {detail}", flush=True)
            failures.append(device)
    if failures:
        raise SystemExit(f"GEMM stress failed on devices: {failures}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.child_device is None:
        run_parent(arguments)
    else:
        run_child(arguments)
