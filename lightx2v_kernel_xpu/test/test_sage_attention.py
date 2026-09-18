import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")


def _reference(q, k, v, causal=False, scale=None):
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    return torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=causal,
        scale=scale,
    ).permute(0, 2, 1, 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sage_attention_matches_dense(dtype, head_dim, causal):
    from sycl_kernels import sage_attention

    torch.manual_seed(123)
    q = torch.randn((1, 129, 2, head_dim), device="xpu", dtype=dtype) * 0.25
    k = torch.randn_like(q) * 0.25
    v = torch.randn_like(q) * 0.25
    actual = sage_attention(q, k, v, is_causal=causal)
    expected = _reference(q, k, v, causal)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def test_sage_attention_gqa_batch_and_custom_scale():
    from sycl_kernels import sage_attention

    torch.manual_seed(7)
    q = torch.randn((2, 73, 4, 64), device="xpu", dtype=torch.float16)
    k = torch.randn((2, 91, 2, 64), device="xpu", dtype=torch.float16)
    v = torch.randn_like(k)
    actual = sage_attention(q, k, v, scale=0.2)
    expected = _reference(q, k, v, scale=0.2)
    torch.xpu.synchronize()
    error = (actual.float() - expected.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0)
    assert error.mean() < 3e-3
    assert cosine > 0.9999


def test_k_smoothing_preserves_attention_semantics():
    from sycl_kernels import sage_attention

    torch.manual_seed(9)
    q = torch.randn((1, 128, 2, 128), device="xpu", dtype=torch.float16)
    k = torch.randn_like(q) + 8.0
    v = torch.randn_like(q)
    smooth = sage_attention(q, k, v, smooth_k=True)
    unsmoothed = sage_attention(q, k, v, smooth_k=False)
    expected = _reference(q, k, v)
    smooth_error = (smooth.float() - expected.float()).square().mean()
    raw_error = (unsmoothed.float() - expected.float()).square().mean()
    assert smooth_error < raw_error
