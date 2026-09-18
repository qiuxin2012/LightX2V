"""SageAttention-style INT8-QK attention for Intel XPU.

Q and K use symmetric per-token INT8 quantization.  K is centred along the
sequence dimension before quantization (the shift cancels in softmax).  The
attention kernel keeps QK on INT8 XMX/DPAS and computes PV in the input dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_bhld_int8(
    x_ptr,
    mean_ptr,
    q_ptr,
    scale_ptr,
    length: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    subtract_mean: tl.constexpr,
    blhd_layout: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    batch_head = row // length
    token = row - batch_head * length
    head = batch_head % heads
    batch = batch_head // heads
    mask = d < head_dim
    if blhd_layout:
        offset = ((batch * length + token) * heads + head) * head_dim
    else:
        offset = row * head_dim
    x = tl.load(x_ptr + offset + d, mask=mask, other=0.0)
    if subtract_mean:
        x -= tl.load(mean_ptr + batch_head * head_dim + d, mask=mask, other=0.0)
    amax = tl.max(tl.abs(x), axis=0)
    safe_amax = tl.maximum(amax, 1.27e-6)
    scale = safe_amax * (1.0 / 127.0)
    rounded = tl.where(x >= 0.0, x * (127.0 / safe_amax) + 0.5,
                       x * (127.0 / safe_amax) - 0.5)
    quant = tl.maximum(-127.0, tl.minimum(127.0, rounded))
    tl.store(q_ptr + offset + d, quant.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _quantize_blhd_rows_int8(
    x_ptr, q_ptr, scale_ptr,
    length: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Quantize several adjacent BLHD tokens in one workgroup."""
    block = tl.program_id(0)
    blocks_per_head = tl.cdiv(length, BLOCK_M)
    batch_head = block // blocks_per_head
    block_m = block - batch_head * blocks_per_head
    head = batch_head % heads
    batch = batch_head // heads
    token = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    mask = (token[:, None] < length) & (d[None, :] < head_dim)
    offset = ((batch * length + token[:, None]) * heads + head) * head_dim + d[None, :]
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    safe_amax = tl.maximum(amax, 1.27e-6)
    inv_scale = 127.0 / safe_amax
    rounded = tl.where(x >= 0.0, x * inv_scale[:, None] + 0.5,
                       x * inv_scale[:, None] - 0.5)
    quant = tl.maximum(-127.0, tl.minimum(127.0, rounded))
    tl.store(q_ptr + offset, quant.to(tl.int8), mask=mask)
    scale_row = batch_head * length + token
    tl.store(scale_ptr + scale_row, safe_amax * (1.0 / 127.0),
             mask=token < length)


@triton.jit
def _quantize_k_block_int8(
    x_ptr, mean_ptr, q_ptr, scale_ptr,
    length: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr,
    subtract_mean: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    block = tl.program_id(0)
    blocks_per_head = tl.cdiv(length, BLOCK_N)
    batch_head = block // blocks_per_head
    block_n = block % blocks_per_head
    head = batch_head % heads
    batch = batch_head // heads
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)
    mask = (n[:, None] < length) & (d[None, :] < head_dim)
    offsets = ((batch * length + n[:, None]) * heads + head) * head_dim + d[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if subtract_mean:
        x -= tl.load(mean_ptr + batch_head * head_dim + d[None, :],
                     mask=d[None, :] < head_dim, other=0.0)
    row_max = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(tl.max(row_max, axis=0) / 127.0, 1.0e-8)
    rounded = tl.where(x >= 0.0, x / scale + 0.5, x / scale - 0.5)
    quant = tl.maximum(-127.0, tl.minimum(127.0, rounded))
    tl.store(q_ptr + offsets, quant.to(tl.int8), mask=mask)
    tl.store(scale_ptr + block, scale)


@triton.jit
def _quantize_qk_pair_blhd_int8(
    q_ptr, k_ptr, qi_ptr, ki_ptr, qs_ptr, ks_ptr,
    length: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % length
    batch_head = row // length
    head = batch_head % heads
    batch = batch_head // heads
    d = tl.arange(0, BLOCK_D)
    mask = d < head_dim
    offset = ((batch * length + token) * heads + head) * head_dim + d
    q = tl.load(q_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    q_scale = tl.maximum(tl.max(tl.abs(q), axis=0) / 127.0, 1.0e-8)
    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / 127.0, 1.0e-8)
    qr = tl.where(q >= 0.0, q / q_scale + 0.5, q / q_scale - 0.5)
    kr = tl.where(k >= 0.0, k / k_scale + 0.5, k / k_scale - 0.5)
    tl.store(qi_ptr + offset, tl.maximum(-127.0, tl.minimum(127.0, qr)).to(tl.int8), mask=mask)
    tl.store(ki_ptr + offset, tl.maximum(-127.0, tl.minimum(127.0, kr)).to(tl.int8), mask=mask)
    tl.store(qs_ptr + row, q_scale)
    tl.store(ks_ptr + row, k_scale)


@triton.jit
def _sage_int8_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    out_ptr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    sm_scale,
    causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // q_heads
    q_head = bh % q_heads
    kv_head = q_head // (q_heads // kv_heads)
    q_base = (batch * q_heads + q_head) * q_len
    kv_base = (batch * kv_heads + kv_head) * kv_len

    oq = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    ok = tl.arange(0, BLOCK_K)
    od = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + (q_base + oq[:, None]) * HEAD_DIM + od[None, :],
                mask=oq[:, None] < q_len, other=0)
    qs = tl.load(q_scale_ptr + q_base + oq, mask=oq < q_len, other=0.0)

    row_max = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for start in tl.range(0, kv_len, BLOCK_K, num_stages=2):
        kp = start + ok
        valid = kp < kv_len
        k = tl.load(k_ptr + (kv_base + kp[:, None]) * HEAD_DIM + od[None, :],
                    mask=valid[:, None], other=0)
        ks = tl.load(k_scale_ptr + kv_base + kp, mask=valid, other=0.0)
        v = tl.load(v_ptr + (kv_base + kp[:, None]) * HEAD_DIM + od[None, :],
                    mask=valid[:, None], other=0.0)

        score_i32 = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        scores = score_i32.to(tl.float32) * qs[:, None] * ks[None, :] * (sm_scale * log2e)
        score_mask = valid[None, :] & (oq[:, None] < q_len)
        if causal:
            score_mask &= kp[None, :] <= oq[:, None]
        scores = tl.where(score_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp2(row_max - new_max)
        p = tl.exp2(scores - new_max[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        row_max = new_max

    acc /= row_sum[:, None]
    tl.store(out_ptr + (q_base + oq[:, None]) * HEAD_DIM + od[None, :], acc,
             mask=oq[:, None] < q_len)


def sage_attention(q, k, v, *, is_causal=False, scale=None, smooth_k=True):
    """Run per-token INT8-QK, FP16/BF16-PV attention on Intel XPU.

    Inputs and output use contiguous ``[B, L, H, D]`` layout.  GQA is allowed
    when the number of query heads is divisible by the number of KV heads.
    """
    if q.device.type != "xpu" or k.device != q.device or v.device != q.device:
        raise ValueError("sage_attention requires Q, K and V on one Intel XPU")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K and V must be [B, L, H, D]")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K and V must have the same FP16 or BF16 dtype")
    b, q_len, q_heads, d = q.shape
    bk, kv_len, kv_heads, dk = k.shape
    if bk != b or v.shape != k.shape or dk != d or q_heads % kv_heads:
        raise ValueError("incompatible Q/K/V shapes or GQA head counts")
    if d not in (64, 128):
        raise ValueError("sage_attention currently supports head_dim 64 or 128")
    if is_causal and q_len != kv_len:
        raise ValueError("causal attention currently requires equal Q and KV lengths")

    qh, kh, vh, qi, ki, qs, ks = quantize_qk(q, k, v, smooth_k=smooth_k)

    out = torch.empty_like(qh)
    block_q = 128
    block_k = 64
    _sage_int8_attention[(triton.cdiv(q_len, block_q), b * q_heads)](
        qi, ki, vh, qs, ks, out, q_len, kv_len, q_heads, kv_heads,
        d**-0.5 if scale is None else scale, is_causal,
        BLOCK_Q=block_q, BLOCK_K=block_k, HEAD_DIM=d, num_warps=8, num_stages=3)
    return out.permute(0, 2, 1, 3)


def quantize_qk(q, k, v, *, smooth_k=True):
    """Quantize BLHD Q/K and return contiguous BHLD operands and scales."""
    b, q_len, q_heads, d = q.shape
    kv_len, kv_heads = k.shape[1], k.shape[2]
    # The generic Triton attention consumes BHLD. MiniMax's native CUTE path
    # uses the returned BLHD quantized tensors directly and avoids these copies.
    qh = q.permute(0, 2, 1, 3).contiguous()
    kh = k.permute(0, 2, 1, 3).contiguous()
    vh = v.permute(0, 2, 1, 3).contiguous()
    k_mean = kh.float().mean(dim=2) if smooth_k else torch.empty(1, device=q.device)
    qi = torch.empty_like(qh, dtype=torch.int8)
    ki = torch.empty_like(kh, dtype=torch.int8)
    qs = torch.empty((b, q_heads, q_len), device=q.device, dtype=torch.float32)
    ks = torch.empty((b, kv_heads, kv_len), device=q.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    _quantize_bhld_int8[(b * q_heads * q_len,)](
        qh, k_mean, qi, qs, q_len, q_heads, d, False, False,
        BLOCK_D=block_d, num_warps=1)
    _quantize_bhld_int8[(b * kv_heads * kv_len,)](
        kh, k_mean, ki, ks, kv_len, kv_heads, d, smooth_k, False,
        BLOCK_D=block_d, num_warps=1)

    return qh, kh, vh, qi, ki, qs, ks


def quantize_qk_blhd(q, k, *, smooth_k=True, scale_dtype=torch.float32):
    """Quantize contiguous BLHD tensors without materializing BHLD copies."""
    b, q_len, q_heads, d = q.shape
    kv_len, kv_heads = k.shape[1], k.shape[2]
    k_mean = k.float().mean(dim=1) if smooth_k else k
    if q.shape == k.shape:
        qk_int8 = torch.empty((2, *q.shape), device=q.device, dtype=torch.int8)
        qi, ki = qk_int8[0], qk_int8[1]
        qk_scale = torch.empty((2, b, q_heads, q_len), device=q.device, dtype=scale_dtype)
        qs, ks = qk_scale[0], qk_scale[1]
    else:
        qi = torch.empty_like(q, dtype=torch.int8)
        ki = torch.empty_like(k, dtype=torch.int8)
        qs = torch.empty((b, q_heads, q_len), device=q.device, dtype=scale_dtype)
        ks = torch.empty((b, kv_heads, kv_len), device=q.device, dtype=scale_dtype)
    block_d = triton.next_power_of_2(d)
    if not smooth_k and q_len == kv_len and q_heads == kv_heads:
        block_m = 2
        _quantize_blhd_rows_int8[(b * q_heads * triton.cdiv(q_len, block_m),)](
            q, qi, qs, q_len, q_heads, d,
            BLOCK_M=block_m, BLOCK_D=block_d, num_warps=1)
        _quantize_blhd_rows_int8[(b * kv_heads * triton.cdiv(kv_len, block_m),)](
            k, ki, ks, kv_len, kv_heads, d,
            BLOCK_M=block_m, BLOCK_D=block_d, num_warps=1)
    else:
        _quantize_bhld_int8[(b * q_heads * q_len,)](
            q, k_mean, qi, qs, q_len, q_heads, d, False, True,
            BLOCK_D=block_d, num_warps=1)
        _quantize_bhld_int8[(b * kv_heads * kv_len,)](
            k, k_mean, ki, ks, kv_len, kv_heads, d, smooth_k, True,
            BLOCK_D=block_d, num_warps=1)
    return qi, ki, qs, ks


def quantize_qk_blhd_fused(q, k):
    """One-launch Q/K quantization for MiniMax-H3's equal-shape inputs."""
    b, length, heads, d = q.shape
    qi = torch.empty_like(q, dtype=torch.int8)
    ki = torch.empty_like(k, dtype=torch.int8)
    qs = torch.empty((b, heads, length), device=q.device, dtype=torch.float32)
    ks = torch.empty_like(qs)
    _quantize_qk_pair_blhd_int8[(b * heads * length,)](
        q, k, qi, ki, qs, ks, length, heads, d,
        BLOCK_D=triton.next_power_of_2(d), num_warps=1)
    return qi, ki, qs, ks


def quantize_qk_blocks_blhd(q, k):
    """Quantize Q/K with scales matching the native CUTE Q256/K32 tiles."""
    b, length, heads, d = q.shape
    q_block, k_block = 256, 32
    q_blocks, k_blocks = triton.cdiv(length, q_block), triton.cdiv(length, k_block)
    qi = torch.empty_like(q, dtype=torch.int8)
    ki = torch.empty_like(k, dtype=torch.int8)
    qs = torch.empty((b, heads, q_blocks), device=q.device, dtype=torch.float32)
    ks = torch.empty((b, heads, k_blocks), device=q.device, dtype=torch.float32)
    dummy = torch.empty(1, device=q.device)
    block_d = triton.next_power_of_2(d)
    _quantize_k_block_int8[(b * heads * q_blocks,)](
        q, dummy, qi, qs, length, heads, d, False,
        BLOCK_N=q_block, BLOCK_D=block_d, num_warps=8)
    _quantize_k_block_int8[(b * heads * k_blocks,)](
        k, dummy, ki, ks, length, heads, d, False,
        BLOCK_N=k_block, BLOCK_D=block_d, num_warps=8)
    return qi, ki, qs, ks
