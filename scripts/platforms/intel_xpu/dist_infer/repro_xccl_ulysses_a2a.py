#!/usr/bin/env python3
"""Minimal XCCL reproducer for the Wan Ulysses all-to-all failure.

Run one case per torchrun invocation.  A Level Zero DEVICE_LOST error can leave
the process/device unusable, so cases intentionally are not chained here.
"""

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist


def log(message, rank):
    print(f"[rank{rank}] {message}", flush=True)


def synchronize(stage, rank, device, tensor=None):
    detail = "" if tensor is None else f" shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
    log(f"synchronize before {stage}{detail}", rank)
    torch.xpu.synchronize(device)
    log(f"synchronized {stage}", rank)


def log_memory(stage, rank, device):
    free_bytes, total_bytes = torch.xpu.mem_get_info(device)
    mib = 1024 * 1024
    allocated = torch.xpu.memory_allocated(device) / mib
    reserved = torch.xpu.memory_reserved(device) / mib
    log(
        f"memory {stage}: free={free_bytes / mib:.0f} MiB, "
        f"total={total_bytes / mib:.0f} MiB, allocated={allocated:.0f} MiB, "
        f"reserved={reserved:.0f} MiB",
        rank,
    )


def reserve_device_memory(reserve_mib, rank, device):
    """Commit XPU allocations and retain them for the duration of the case."""
    allocations = []
    remaining = reserve_mib
    chunk_mib = 512
    while remaining > 0:
        current_mib = min(remaining, chunk_mib)
        tensor = torch.empty(current_mib * 1024 * 1024, dtype=torch.uint8, device=device)
        tensor.fill_(rank & 0xFF)
        allocations.append(tensor)
        remaining -= current_mib
    synchronize(f"after reserving {reserve_mib} MiB", rank, device)
    log_memory("after reservation", rank, device)
    return allocations


def make_groups(world_size):
    """Create the same orthogonal SP2 x TP2 groups as mesh [seq_p, tensor_p]."""
    if world_size != 4:
        raise ValueError(f"subgroup cases require WORLD_SIZE=4, got {world_size}")

    # Every rank must call new_group in exactly the same global order.
    tp_rank_lists = ([0, 1], [2, 3])
    sp_rank_lists = ([0, 2], [1, 3])
    tp_groups = [dist.new_group(ranks=list(ranks), backend="xccl") for ranks in tp_rank_lists]
    sp_groups = [dist.new_group(ranks=list(ranks), backend="xccl") for ranks in sp_rank_lists]
    rank = dist.get_rank()
    tp_group = tp_groups[0] if rank < 2 else tp_groups[1]
    sp_group = sp_groups[rank % 2]
    return sp_group, tp_group


def rank_filled(shape, group, device, dtype=torch.bfloat16):
    """Each destination chunk is filled with this rank's group-local rank."""
    group_rank = dist.get_rank(group)
    return torch.full(shape, float(group_rank), dtype=dtype, device=device)


def verify_rank_filled_output(rank, output, group):
    group_size = dist.get_world_size(group)
    if output.shape[0] != group_size:
        raise AssertionError(f"A2A output dim 0 is {output.shape[0]}, expected {group_size}")
    # Works for both the two-dimensional small tensor and model's 5-D tensor.
    for source_rank in range(group_size):
        actual = output[source_rank].reshape(-1)[0].float().item()
        if actual != float(source_rank):
            raise AssertionError(
                f"source chunk {source_rank} contains {actual}, expected {source_rank}"
            )
    log(f"verification passed; output shape={tuple(output.shape)}", rank)


def direct_a2a(input_tensor, group):
    output = torch.empty_like(input_tensor)
    dist.all_to_all_single(output, input_tensor, group=group)
    return output


def run_small_case(rank, device, group, label):
    group_size = dist.get_world_size(group)
    # 256 KiB total at WORLD=4 and 128 KiB total at SP=2.
    input_tensor = rank_filled((group_size, 32768), group, device)
    synchronize(f"{label} A2A", rank, device, input_tensor)
    output = direct_a2a(input_tensor, group)
    synchronize(f"after {label} A2A", rank, device, output)
    verify_rank_filled_output(rank, output, group)


def pack_model_qkv(q, k, v, sp_size, qkv_fusion):
    local_len, heads, head_dim = q.shape
    shard_heads = heads // sp_size
    if qkv_fusion:
        q = q.reshape(local_len, sp_size, shard_heads, head_dim)
        k = k.reshape(local_len, sp_size, shard_heads, head_dim)
        v = v.reshape(local_len, sp_size, shard_heads, head_dim)
        return [torch.stack((q, k, v), dim=2).permute(1, 0, 2, 3, 4).contiguous()]
    return [
        tensor.reshape(local_len, sp_size, shard_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
        for tensor in (q, k, v)
    ]


def run_model_case(args, rank, device, sp_group, tp_group):
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size != 2:
        raise ValueError(f"model case represents SP=2, got {sp_size}")

    log_memory("before reservation", rank, device)
    memory_reservations = reserve_device_memory(args.reserve_mib, rank, device)

    # Wan2.2 SP2 x TP2 block-0 self-attention input seen in the failing run.
    shape = (13200, 12, 128)
    q = torch.full(shape, float(sp_rank), dtype=torch.bfloat16, device=device)
    k = torch.full(shape, float(sp_rank), dtype=torch.bfloat16, device=device)
    v = torch.full(shape, float(sp_rank), dtype=torch.bfloat16, device=device)

    if not args.no_tp_precollective:
        log(f"running TP all-reduce precollectives on q and k; shape={shape}", rank)
        dist.all_reduce(q, group=tp_group)
        dist.all_reduce(k, group=tp_group)
        synchronize("after TP precollectives", rank, device, q)
        # Restore simple source-rank contents so the A2A result is verifiable.
        q.fill_(float(sp_rank))
        k.fill_(float(sp_rank))

    payloads = pack_model_qkv(q, k, v, sp_size, not args.no_qkv_fusion)
    del q, k, v
    for index, payload in enumerate(payloads):
        mib = payload.numel() * payload.element_size() / (1024 * 1024)
        log(f"model payload {index}: shape={tuple(payload.shape)}, size={mib:.1f} MiB", rank)
        synchronize(f"model payload {index} A2A", rank, device, payload)
        output = direct_a2a(payload, sp_group)
        synchronize(f"after model payload {index} A2A", rank, device, output)
        verify_rank_filled_output(rank, output, sp_group)
        del output, payload
    # Keep allocations alive through all collectives.
    if len(memory_reservations) != (args.reserve_mib + 511) // 512:
        raise AssertionError("memory reservation was released unexpectedly")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("world-small", "subgroup-small", "subgroup-model"),
    )
    parser.add_argument("--no-device-id", action="store_true")
    parser.add_argument("--no-tp-precollective", action="store_true")
    parser.add_argument("--no-qkv-fusion", action="store_true")
    parser.add_argument(
        "--reserve-mib",
        type=int,
        default=0,
        help="MiB of committed XPU memory to retain per rank during the model case",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("xpu", local_rank)
    torch.xpu.set_device(device)
    init_kwargs = {"backend": "xccl"}
    if not args.no_device_id:
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        log(
            f"case={args.case}, torch={torch.__version__}, world={world_size}, "
            f"device={device}, device_id={not args.no_device_id}",
            rank,
        )
        if args.case == "world-small":
            run_small_case(rank, device, dist.group.WORLD, "WORLD small")
        else:
            sp_group, tp_group = make_groups(world_size)
            if args.case == "subgroup-small":
                run_small_case(rank, device, sp_group, "SP subgroup small")
            else:
                run_model_case(args, rank, device, sp_group, tp_group)
        dist.barrier()
        log(f"PASS: {args.case}", rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
