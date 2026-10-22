"""TMLR plan Task 2 — multi-site LoRA (`lora_ms`) with per-site ranks.

lora_ms puts one LoRAConv2d in every stage, at the same stage-final blocks the
parallel bottleneck uses: ResNet -> `<stage>.conv2` (3x3); inverted residual ->
the block's 1x1 projection conv. Per-site ranks let it match the bottleneck's
parameter budget (plan A3; counts verified on the real torchvision modules).
"""
from __future__ import annotations

import pytest
import torch

from src.adapters import build_adapter, multisite_lora_targets
from src.backbones import build_backbone
from src.utils import count_trainable_params


def _bb(name):
    return build_backbone(name, pretrained=False)


def test_multisite_targets():
    assert multisite_lora_targets(_bb("resnet18")) == [
        "layer1.1.conv2", "layer2.1.conv2", "layer3.1.conv2", "layer4.1.conv2"]
    assert multisite_lora_targets(_bb("resnet34")) == [
        "layer1.2.conv2", "layer2.3.conv2", "layer3.5.conv2", "layer4.2.conv2"]
    assert multisite_lora_targets(_bb("mobilenetv3_small")) == [
        "features.3.block.2.0", "features.6.block.3.0",
        "features.8.block.3.0", "features.11.block.3.0"]
    assert multisite_lora_targets(_bb("mobilenetv2")) == [
        f"features.{i}.conv.2" for i in (3, 6, 10, 13, 16)]


@pytest.mark.parametrize("bb,ranks,expected", [
    ("resnet18", [1, 1, 2, 1], 12160), ("resnet18", [3, 3, 4, 3], 31360),
    ("resnet34", [1, 1, 2, 1], 12160),
    ("mobilenetv3_small", [6, 6, 6, 5], 6864), ("mobilenetv3_small", [7, 7, 7, 8], 9464),
    ("mobilenetv2", [4, 4, 4, 4, 5], 11648), ("mobilenetv2", [7, 6, 6, 6, 7], 17080)])
def test_per_target_ranks_param_count(bb, ranks, expected):
    backbone = _bb(bb)
    adapter = build_adapter({"type": "lora", "rank": max(ranks),
                             "lora_targets": "multisite", "lora_ranks": ranks},
                            dim=0, backbone=backbone)
    assert adapter.ranks == ranks
    assert count_trainable_params(backbone) == expected


def test_identity_at_init():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    ref = _bb("resnet18")
    with torch.no_grad():
        y_ref = ref(x)
    backbone = _bb("resnet18")
    backbone.load_state_dict(ref.state_dict())
    build_adapter({"type": "lora", "rank": 3, "lora_targets": "multisite",
                   "lora_ranks": [3, 3, 4, 3]}, dim=0, backbone=backbone)
    with torch.no_grad():
        assert torch.equal(backbone(x), y_ref)


def test_rank_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_adapter({"type": "lora", "rank": 2, "lora_targets": "multisite",
                       "lora_ranks": [1, 2]}, dim=0, backbone=_bb("resnet18"))


def test_nonpositive_rank_raises():
    with pytest.raises(ValueError):
        build_adapter({"type": "lora", "rank": 2, "lora_targets": "multisite",
                       "lora_ranks": [1, 0, 1, 1]}, dim=0, backbone=_bb("resnet18"))


def test_single_target_unchanged():
    backbone = _bb("resnet18")
    adapter = build_adapter({"type": "lora", "rank": 16,
                             "lora_targets": ["layer4.0.downsample.0"]},
                            dim=0, backbone=backbone)
    assert count_trainable_params(backbone) == 12288
    assert adapter.ranks == [16]
