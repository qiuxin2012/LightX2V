import pytest
import torch

import sycl_kernels


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="MiniMax-H3 fused kernels require an Intel XPU",
)

HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96
HIDDEN_SIZE = 5376
EPS = 1e-5


def _rms_norm_reference(x, weight):
    scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS)
    return (x.float() * scale * weight.float()).to(torch.bfloat16)


def _qk_inputs(rows, heads, packed):
    torch.manual_seed(3000 + rows + heads + int(packed))
    if packed:
        qkv = torch.randn(
            rows,
            3 * heads * HEAD_DIM,
            device="xpu",
            dtype=torch.bfloat16,
        )
        q, k, _ = qkv.split(heads * HEAD_DIM, dim=-1)
        q = q.view(rows, heads, HEAD_DIM)
        k = k.view(rows, heads, HEAD_DIM)
    else:
        q = torch.randn(
            rows, heads, HEAD_DIM, device="xpu", dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
    q_weight = torch.randn(HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    k_weight = torch.randn_like(q_weight)
    freqs = torch.randn(
        rows, ROTARY_DIM, device="xpu", dtype=torch.float32
    )
    return q_weight, k_weight, q, k, freqs


def _qk_reference(weight, x, freqs):
    normalized = _rms_norm_reference(x, weight)
    x_rot = normalized[..., :ROTARY_DIM]
    x_pass = normalized[..., ROTARY_DIM:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated_half = torch.cat((-x2, x1), dim=-1)
    cos = freqs.cos().to(torch.bfloat16).unsqueeze(1)
    sin = freqs.sin().to(torch.bfloat16).unsqueeze(1)
    rotated = x_rot * cos + rotated_half * sin
    return torch.cat((rotated, x_pass), dim=-1)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("heads", [56, 28, 14, 7])
def test_qk_rmsnorm_rope_matches_reference(heads, packed):
    values = _qk_inputs(17, heads, packed)
    q_weight, k_weight, q, k, freqs = values
    q_expected = _qk_reference(q_weight, q, freqs)
    k_expected = _qk_reference(k_weight, k, freqs)

    q_actual, k_actual = sycl_kernels.fused_minimax_h3_qk_rmsnorm_rope(
        *values, eps=EPS
    )

    assert q_actual.is_contiguous()
    assert k_actual.is_contiguous()
    torch.testing.assert_close(q_actual, q_expected, rtol=0.02, atol=0.03125)
    torch.testing.assert_close(k_actual, k_expected, rtol=0.02, atol=0.03125)


def test_qk_rmsnorm_rope_uses_current_stream():
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        values = _qk_inputs(17, 14, True)
        q_expected = _qk_reference(values[0], values[2], values[4])
        k_expected = _qk_reference(values[1], values[3], values[4])
        q_actual, k_actual = sycl_kernels.fused_minimax_h3_qk_rmsnorm_rope(
            *values, eps=EPS
        )
    stream.synchronize()
    torch.testing.assert_close(q_actual, q_expected, rtol=0.02, atol=0.03125)
    torch.testing.assert_close(k_actual, k_expected, rtol=0.02, atol=0.03125)


def _adaln_inputs(rows, compact_rows, packed):
    torch.manual_seed(5376 + rows + compact_rows + int(packed))
    value = torch.randn(
        rows, HIDDEN_SIZE, device="xpu", dtype=torch.bfloat16
    )
    weight = torch.randn(HIDDEN_SIZE, device="xpu", dtype=torch.bfloat16)
    if packed:
        modulation = torch.randn(
            compact_rows,
            6 * HIDDEN_SIZE,
            device="xpu",
            dtype=torch.bfloat16,
        )
        shift = modulation[:, :HIDDEN_SIZE]
        scale = modulation[:, HIDDEN_SIZE : 2 * HIDDEN_SIZE]
    else:
        scale = torch.randn(
            compact_rows,
            HIDDEN_SIZE,
            device="xpu",
            dtype=torch.bfloat16,
        )
        shift = torch.randn_like(scale)
    indices = torch.randint(
        compact_rows, (rows,), device="xpu", dtype=torch.int64
    )
    return weight, value, scale, shift, indices


def _adaln_reference(weight, value, scale, shift, indices):
    normalized = _rms_norm_reference(value, weight)
    selected_scale = scale.index_select(0, indices)
    selected_shift = shift.index_select(0, indices)
    return (normalized * (1.0 + selected_scale) + selected_shift).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rows", [1, 17, 129])
def test_indexed_rms_adaln_matches_reference(rows, packed):
    values = _adaln_inputs(rows, 7, packed)
    expected = _adaln_reference(*values)
    actual = sycl_kernels.fused_minimax_h3_indexed_rms_adaln(
        *values, eps=EPS
    )
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03125)


def test_indexed_rms_adaln_bounds_invalid_indices():
    values = list(_adaln_inputs(2, 3, False))
    values[4][0] = -1
    actual = sycl_kernels.fused_minimax_h3_indexed_rms_adaln(*values)
    assert not torch.isfinite(actual[0]).any()
