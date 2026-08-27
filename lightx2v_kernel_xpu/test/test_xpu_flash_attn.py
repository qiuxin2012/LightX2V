import pytest
import torch

if not torch.xpu.is_available():
    pytest.skip("XPU is unavailable", allow_module_level=True)

from lightx2v_platform.ops.attn.intel_xpu import xpu_flash_attn


def test_hd64_single_sequence_uses_sycl_backend(monkeypatch):
    calls = []

    def fake_sdp(q, k, v):
        calls.append((q.shape, k.shape, v.shape))
        return q

    monkeypatch.setattr(xpu_flash_attn, "_sdp_fn", fake_sdp)
    q = torch.randn(1, 256, 32, 64)

    output = xpu_flash_attn.IntelXpuFlashAttnWeight().apply(q, q, q)

    assert calls == [((1, 256, 32, 64),) * 3]
    assert output.shape == (256, 32 * 64)
    torch.testing.assert_close(output, q.reshape(256, 32 * 64))


def test_hd64_unaligned_sequence_falls_back_for_accuracy(monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("unaligned HD64 input must not use the ESIMD kernel")

    monkeypatch.setattr(xpu_flash_attn, "_sdp_fn", fail_if_called)
    q = torch.randn(1, 17, 2, 64)

    output = xpu_flash_attn.IntelXpuFlashAttnWeight().apply(q, q, q)

    assert output.shape == (17, 2 * 64)
