from types import SimpleNamespace

import torch

import lightx2v.common.ops.mm.mm_weight as mm_weight
from lightx2v.models.networks.minimax_h3.model import H3_CHANNEL_QUANT_SCHEMES


def test_minimax_h3_accepts_intel_xpu_fp8():
    assert "fp8-intel-xpu" in H3_CHANNEL_QUANT_SCHEMES


def test_intel_xpu_fp8_forwards_bias_to_kernel(monkeypatch):
    layer = object.__new__(mm_weight.MMWeightFp8IntelXpu)
    layer.weight = torch.empty((3, 4), dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.ones((3, 1), dtype=torch.bfloat16)
    layer.bias = torch.randn(3, dtype=torch.bfloat16)
    layer.has_diff = False

    expected = torch.randn(2, 3, dtype=torch.bfloat16)
    captured = {}

    def fake_kernel(input_tensor, weight, weight_scale, bias):
        captured["args"] = (input_tensor, weight, weight_scale, bias)
        return expected

    monkeypatch.setattr(
        mm_weight,
        "sycl_kernels",
        SimpleNamespace(onednn_w8a16_fp8=fake_kernel),
    )
    input_tensor = torch.randn(2, 4, dtype=torch.bfloat16)

    actual = layer.apply(input_tensor)

    assert actual is expected
    assert captured["args"][0] is input_tensor
    assert captured["args"][1] is layer.weight
    assert captured["args"][2].dtype == torch.float32
    assert captured["args"][3] is layer.bias
