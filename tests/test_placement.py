"""Step 6 tests for serial / parallel adapter placement (RQ1).

Covers: shape preservation + identity-at-init (zero-init up), that only the
inserted bottlenecks train (backbone stays frozen), the analytical param
count, gradient flow to the adapter, and build_adapter routing.

Tests build ResNet-18 with weights=None (offline, no ImageNet download) —
mirrors tests/test_lora.py:_bare_frozen_resnet18.
"""
import torch
import torch.nn as nn
import pytest

from torchvision.models import resnet18

from src.adapters import PlacementAdapter, Conv1x1Bottleneck, build_adapter


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def _pooled(backbone, x):
    return backbone(x)  # fc=Identity -> pooled (B, 512)


# analytical trainable params: per stage body = down(C*r + r) + up(r*C + C)
def _expected_params(channels, rank):
    return sum((C * rank + rank) + (rank * C + C) for C in channels)


# --------------------------------------------------------------------------
# Conv1x1Bottleneck
# --------------------------------------------------------------------------
def test_conv1x1_bottleneck_is_zero_at_init():
    body = Conv1x1Bottleneck(channels=64, rank=16)
    x = torch.randn(2, 64, 8, 8)
    out = body(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


# --------------------------------------------------------------------------
# PlacementAdapter — identity at init + shape preservation (spec RISK smoke)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_identity_at_init(placement):
    torch.manual_seed(0)
    bb = _bare_frozen_resnet18()
    x = torch.randn(2, 3, 224, 224)
    ref = _pooled(bb, x).clone()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[0, 1, 2, 3])
    out = _pooled(bb, x)
    assert out.shape == ref.shape                 # output shape unchanged
    assert torch.allclose(out, ref, atol=1e-6)    # zero-init up -> identity
    assert ad.num_placed == 4
    assert ad.backbone_trainable is True


# --------------------------------------------------------------------------
# Only the inserted bottlenecks train; backbone stays frozen
# --------------------------------------------------------------------------
@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_only_adapter_trains(placement):
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[0, 1, 2, 3])
    assert all(not p.requires_grad for p in bb.parameters())
    trainable = [p for p in ad.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == _expected_params(
        (64, 128, 256, 512), 16)


def test_param_count_single_stage_matches_post_pool():
    # stage 3 alone (512ch) equals the post-pool Bottleneck-16 count (16,912).
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement="serial", block_ids=[3])
    assert sum(p.numel() for p in ad.parameters()) == 16_912


# --------------------------------------------------------------------------
# Gradients reach the adapter after a backward
# --------------------------------------------------------------------------
@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_gradients_reach_adapter(placement):
    torch.manual_seed(0)
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[3])
    for body in ad.bodies:                      # break the zero-init no-op
        nn.init.normal_(body.up.weight, std=0.01)
    x = torch.randn(2, 3, 224, 224)
    _pooled(bb, x).pow(2).sum().backward()
    for body in ad.bodies:
        assert body.down.weight.grad is not None
        assert torch.isfinite(body.down.weight.grad).all()


# --------------------------------------------------------------------------
# build_adapter routing
# --------------------------------------------------------------------------
def test_build_adapter_routes_placement():
    bb = _bare_frozen_resnet18()
    ad = build_adapter(
        {"type": "bottleneck", "rank": 16, "placement": "parallel",
         "block_ids": [0, 1, 2, 3]}, dim=512, backbone=bb)
    assert isinstance(ad, PlacementAdapter)
    assert ad.backbone_trainable is True


def test_build_adapter_post_pool_default_unchanged():
    # No placement key -> the existing post-pool BottleneckAdapter.
    from src.adapters import BottleneckAdapter
    ad = build_adapter({"type": "bottleneck", "rank": 16}, dim=512, backbone=None)
    assert isinstance(ad, BottleneckAdapter)


def test_placement_requires_backbone():
    with pytest.raises(ValueError, match="requires a backbone"):
        build_adapter({"type": "bottleneck", "rank": 16, "placement": "serial"},
                      dim=512, backbone=None)


def test_invalid_block_id_rejected():
    bb = _bare_frozen_resnet18()
    with pytest.raises(ValueError, match="out of range"):
        PlacementAdapter(bb, rank=16, placement="serial", block_ids=[4])
