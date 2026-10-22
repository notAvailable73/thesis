"""TMLR plan Task 3 — single-site parallel bottleneck (`btl_ss`).

btl_ss wraps the SAME conv LoRA's single-site arm modifies, with a parallel
1x1 down -> ReLU -> 1x1 up branch (with biases). With the ReLU removed and both
biases zero it computes exactly what LoRAConv2d computes, so btl_ss vs lora_ss
isolates the non-linearity + biases at an identical site.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.adapters import (
    ConvParallelBottleneck, LoRAConv2d, SingleSiteBottleneckAdapter,
    build_adapter, default_lora_targets,
)
from src.backbones import build_backbone
from src.utils import count_trainable_params


def _bb(name):
    return build_backbone(name, pretrained=False)


@pytest.mark.parametrize("bb,r,expected", [
    ("resnet18", 15, 12047), ("resnet18", 40, 31272), ("resnet34", 15, 12047),
    ("mobilenetv3_small", 10, 6826), ("mobilenetv3_small", 14, 9518),
    ("mobilenetv2", 9, 11849), ("mobilenetv2", 13, 16973)])
def test_param_counts(bb, r, expected):
    backbone = _bb(bb)
    build_adapter({"type": "bottleneck", "placement": "conv_parallel", "rank": r},
                  dim=0, backbone=backbone)
    assert count_trainable_params(backbone) == expected


def test_identity_at_init():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    ref = _bb("resnet18")
    with torch.no_grad():
        y_ref = ref(x)
    backbone = _bb("resnet18")
    backbone.load_state_dict(ref.state_dict())
    SingleSiteBottleneckAdapter(backbone, rank=15)
    with torch.no_grad():
        assert torch.equal(backbone(x), y_ref)


@pytest.mark.parametrize("bb", ["resnet18", "mobilenetv3_small", "mobilenetv2"])
def test_same_site_as_lora(bb):
    backbone = _bb(bb)
    expected = default_lora_targets(backbone)[0]
    adapter = SingleSiteBottleneckAdapter(backbone, rank=4)
    assert adapter.target == expected
    assert isinstance(backbone.get_submodule(expected), ConvParallelBottleneck)
    assert adapter.backbone_trainable is True


def test_stride_preserved():
    base = _bb("resnet18").get_submodule("layer4.0.downsample.0")   # 256->512, stride 2
    wrapped = ConvParallelBottleneck(base, rank=8)
    x = torch.randn(1, 256, 14, 14)
    with torch.no_grad():
        assert wrapped(x).shape == base(x).shape == (1, 512, 7, 7)


def test_rejects_kxk_base():
    with pytest.raises(ValueError):
        ConvParallelBottleneck(nn.Conv2d(8, 8, 3, padding=1), rank=2)


def test_linear_equivalence_with_lora():
    torch.manual_seed(0)
    base = nn.Conv2d(16, 32, 1, stride=2, bias=False)
    btl = ConvParallelBottleneck(base, rank=4)
    btl.act = nn.Identity()
    with torch.no_grad():
        btl.up.weight.normal_()
        btl.down.bias.zero_()
        btl.up.bias.zero_()
    lora = LoRAConv2d(base, rank=4, alpha=4)
    with torch.no_grad():
        lora.lora_A.weight.copy_(btl.down.weight)
        lora.lora_B.weight.copy_(btl.up.weight)
    x = torch.randn(2, 16, 10, 10)
    with torch.no_grad():
        assert torch.allclose(btl(x), lora(x), atol=1e-6)
