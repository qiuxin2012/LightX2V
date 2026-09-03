#!/usr/bin/env python3
"""Communication-only benchmark for the MiniMax-H3 TP2/SP2 workload."""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


LOCAL_SEQUENCE = 9_650
MAIN_SHARD_SEQUENCE = 9_642
HIDDEN_SIZE = 5_376
TP_LOCAL_HEADS = 28
HEAD_DIM = 128
LAYERS = 50


def synchronize():
    torch.xpu.synchronize()


def timed(name, calls, peer_bytes_per_call, operation, rank, first_round_calls=None):
    synchronize()
    start = time.perf_counter()
    for call_index in range(1, calls + 1):
        operation()
        if rank == 0 and first_round_calls is not None and call_index == first_round_calls:
            synchronize()
            print(
                f"First communication round completed: {name} "
                f"{call_index}/{calls} calls, elapsed={time.perf_counter() - start:.6f}s",
                flush=True,
            )
    synchronize()
    elapsed = time.perf_counter() - start
    total_peer_bytes = calls * peer_bytes_per_call
    if rank == 0:
        print(
            f"{name}: calls={calls}, time={elapsed:.6f}s, "
            f"latency={elapsed / calls * 1e3:.3f}ms/call, "
            f"peer_traffic={total_peer_bytes / 1e9:.3f}GB/rank, "
            f"effective_bandwidth={total_peer_bytes / elapsed / 1e9:.3f}GB/s",
            flush=True,
        )
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=int, default=29, help="Transformer evaluations to replay")
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--warmup-calls", type=int, default=5)
    args = parser.parse_args()

    if args.evals < 1 or args.layers < 1 or args.warmup_calls < 0:
        parser.error("--evals/--layers must be positive and --warmup-calls non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(local_rank)
    device = torch.device("xpu", local_rank)
    dist.init_process_group(backend="xccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise RuntimeError("This benchmark requires exactly 4 ranks (TP=2, SP=2).")

    # Mesh layout and process-group construction exactly match LightX2V.
    mesh = init_device_mesh("xpu", (2, 2), mesh_dim_names=("seq_p", "tensor_p"))
    tp_group = mesh.get_group("tensor_p")
    sp_group = mesh.get_group("seq_p")

    dtype = torch.bfloat16
    # Ulysses Q/K/V/output all use this number of BF16 elements per rank.
    sp_input = torch.zeros((2, MAIN_SHARD_SEQUENCE, TP_LOCAL_HEADS // 2, HEAD_DIM), dtype=dtype, device=device)
    sp_output = torch.empty_like(sp_input)
    tp_tensor = torch.zeros((LOCAL_SEQUENCE, HIDDEN_SIZE), dtype=dtype, device=device)
    gather_input = torch.zeros((MAIN_SHARD_SEQUENCE, HIDDEN_SIZE), dtype=dtype, device=device)
    gather_output = [torch.empty_like(gather_input) for _ in range(2)]

    def sp_a2a():
        dist.all_to_all_single(sp_output, sp_input, group=sp_group)

    def tp_all_reduce():
        dist.all_reduce(tp_tensor, op=dist.ReduceOp.SUM, group=tp_group)

    def final_sp_gather():
        dist.all_gather(gather_output, gather_input, group=sp_group)

    for _ in range(args.warmup_calls):
        sp_a2a()
        tp_all_reduce()
        final_sp_gather()
    synchronize()

    sp_calls = args.evals * args.layers * 4
    tp_calls = args.evals * args.layers * 2
    gather_calls = args.evals
    # For a two-rank A2A each rank sends half its input to its peer.  For the
    # two-rank all-reduce/all-gather, peer traffic equals one local tensor.
    sp_peer_bytes = sp_input.numel() * sp_input.element_size() // 2
    tp_peer_bytes = tp_tensor.numel() * tp_tensor.element_size()
    gather_peer_bytes = gather_input.numel() * gather_input.element_size()

    if rank == 0:
        print(
            f"MiniMax-H3 communication-only: evals={args.evals}, layers={args.layers}, "
            f"world=4, TP=2, SP=2, dtype=BF16",
            flush=True,
        )

    sp_time = timed(
        "SP all-to-all",
        sp_calls,
        sp_peer_bytes,
        sp_a2a,
        rank,
        first_round_calls=args.layers * 4,
    )
    tp_time = timed("TP all-reduce", tp_calls, tp_peer_bytes, tp_all_reduce, rank)
    gather_time = timed("Final SP all-gather", gather_calls, gather_peer_bytes, final_sp_gather, rank)

    if rank == 0:
        total_time = sp_time + tp_time + gather_time
        total_bytes = sp_calls * sp_peer_bytes + tp_calls * tp_peer_bytes + gather_calls * gather_peer_bytes
        print(
            f"TOTAL: time={total_time:.6f}s, time/eval={total_time / args.evals:.6f}s, "
            f"peer_traffic={total_bytes / 1e9:.3f}GB/rank, "
            f"effective_bandwidth={total_bytes / total_time / 1e9:.3f}GB/s",
            flush=True,
        )

    # oneCCL on this platform can block indefinitely while tearing down
    # multiple overlapping TP/SP process groups.  All timed collectives have
    # completed here, so flush the report and bypass interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
