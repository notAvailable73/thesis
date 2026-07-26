"""Step 8 tests — MobileNetV3-Small backbone (Phase 5, RQ4 prep).

Covers: the frozen-backbone feature contract at 576-d, the backbone registry +
feature-dim guard, programmatic stage discovery for in-block placement, and that
every adapter type builds and trains on this backbone (post-pool Bottleneck at
dim=576, serial/parallel placement, LoRA's family default target, BitFit,
Linear-Probe).

Every test builds the architecture with `pretrained=False` / `weights=None` so
the suite stays offline — same convention as tests/test_lora.py and
tests/test_placement.py. The last section is a REGRESSION guard: ResNet-18's
placement sites and param count must be exactly what Step 6 reported, since
Step 8 generalised the stage-resolution code they share.
"""
import pytest
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, resnet18

from src.backbones import (
    build_backbone, build_frozen_mobilenetv3_small, mobilenetv3_stage_paths,
    backbone_feature_dim, canonical_backbone_name,
    MOBILENETV3_SMALL_FEATURE_DIM,
)
from src.adapters import (
    build_adapter, BottleneckAdapter, PlacementAdapter, LoRAAdapter,
    BitFitAdapter, resolve_stage_paths, infer_block_channels,
    default_lora_targets,
)

RANK = 16
#: MobileNetV3-Small stage-final residual blocks and their widths.
MBNET_STAGES = ["features.3", "features.6", "features.8", "features.11"]
MBNET_CHANNELS = (24, 40, 48, 96)


def _bare_frozen_mbnet():
    """MobileNetV3-Small, classifier removed, frozen — no weight download."""
    m = mobilenet_v3_small(weights=None)
    m.classifier = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def _placement_params(channels, rank):
    """down(C*r + r) + up(r*C + C) per placed 1x1 bottleneck."""
    return sum((C * rank + rank) + (rank * C + C) for C in channels)


# --------------------------------------------------------------------------
# 8.1 — backbone feature contract
# --------------------------------------------------------------------------
def test_pooled_feature_is_576d():
    bb = _bare_frozen_mbnet()
    with torch.no_grad():
        out = bb(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, MOBILENETV3_SMALL_FEATURE_DIM) == (2, 576)


def test_backbone_is_frozen_and_in_eval_mode():
    bb = build_frozen_mobilenetv3_small(pretrained=False)
    assert not any(p.requires_grad for p in bb.parameters())
    assert not bb.training
    assert isinstance(bb.classifier, nn.Identity)


def test_build_backbone_routes_mobilenetv3():
    bb = build_backbone("mobilenetv3_small", pretrained=False)
    with torch.no_grad():
        assert bb(torch.randn(1, 3, 224, 224)).shape == (1, 576)


@pytest.mark.parametrize("alias", ["mobilenetv3_small", "mobilenet_v3_small",
                                   "mobilenetv3", "MobileNetV3", "mbnet"])
def test_name_aliases_resolve(alias):
    assert canonical_backbone_name(alias) == "mobilenetv3_small"
    assert backbone_feature_dim(alias) == 576


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="Unknown backbone"):
        canonical_backbone_name("effnet_b0")


def test_resnet18_feature_dim_unchanged():
    assert backbone_feature_dim("resnet18") == 512


# --------------------------------------------------------------------------
# Stage discovery for in-block placement
# --------------------------------------------------------------------------
def test_stage_paths_are_the_stage_final_residual_blocks():
    bb = _bare_frozen_mbnet()
    assert mobilenetv3_stage_paths(bb) == MBNET_STAGES
    assert resolve_stage_paths(bb) == MBNET_STAGES


def test_stage_blocks_are_shape_preserving():
    """Parallel placement adds body(block_input) to the block OUTPUT, so every
    eligible site must have input shape == output shape."""
    bb = _bare_frozen_mbnet()
    for path in mobilenetv3_stage_paths(bb):
        block = bb.get_submodule(path)
        assert block.use_res_connect, f"{path} is not a residual block"
        C = infer_block_channels(block)
        x = torch.randn(2, C, 7, 7)
        with torch.no_grad():
            assert block(x).shape == x.shape


def test_stage_channel_widths():
    bb = _bare_frozen_mbnet()
    got = tuple(infer_block_channels(bb.get_submodule(p))
                for p in mobilenetv3_stage_paths(bb))
    assert got == MBNET_CHANNELS


# --------------------------------------------------------------------------
# 8.2 — every adapter type works on this backbone
# --------------------------------------------------------------------------
def test_post_pool_bottleneck_at_576():
    """Bottleneck-16 on a 576-d pooled feature: 576*16+16 + 16*576+576 = 19,024."""
    ad = build_adapter({"type": "bottleneck", "rank": RANK}, dim=576,
                       backbone=None)
    assert isinstance(ad, BottleneckAdapter)
    assert sum(p.numel() for p in ad.parameters()) == 19_024
    x = torch.randn(4, 576)
    out = ad(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-7)      # zero-init up -> identity


@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_placement_identity_at_init(placement):
    torch.manual_seed(0)
    bb = _bare_frozen_mbnet()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = bb(x).clone()
    ad = PlacementAdapter(bb, rank=RANK, placement=placement,
                          block_ids=[0, 1, 2, 3])
    with torch.no_grad():
        out = bb(x)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5)    # zero-init up -> identity
    assert ad.num_placed == 4
    assert ad.stage_paths == MBNET_STAGES
    assert tuple(ad.stage_channels) == MBNET_CHANNELS
    assert ad.backbone_trainable is True


@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_placement_param_count_and_backbone_stays_frozen(placement):
    bb = _bare_frozen_mbnet()
    ad = PlacementAdapter(bb, rank=RANK, placement=placement,
                          block_ids=[0, 1, 2, 3])
    assert all(not p.requires_grad for p in bb.parameters())
    n = sum(p.numel() for p in ad.parameters() if p.requires_grad)
    assert n == _placement_params(MBNET_CHANNELS, RANK) == 6_928


@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_placement_gradients_reach_adapter(placement):
    torch.manual_seed(0)
    bb = _bare_frozen_mbnet()
    ad = PlacementAdapter(bb, rank=RANK, placement=placement, block_ids=[3])
    for body in ad.bodies:                        # break the zero-init no-op
        nn.init.normal_(body.up.weight, std=0.01)
    bb(torch.randn(2, 3, 224, 224)).pow(2).sum().backward()
    for body in ad.bodies:
        assert body.down.weight.grad is not None
        assert torch.isfinite(body.down.weight.grad).all()


def test_build_adapter_routes_mbnet_placement():
    bb = _bare_frozen_mbnet()
    ad = build_adapter(
        {"type": "bottleneck", "rank": RANK, "placement": "parallel",
         "block_ids": [0, 1, 2, 3]}, dim=576, backbone=bb)
    assert isinstance(ad, PlacementAdapter)
    assert ad.stage_paths == MBNET_STAGES


def test_explicit_stage_paths_override():
    bb = _bare_frozen_mbnet()
    ad = build_adapter(
        {"type": "bottleneck", "rank": RANK, "placement": "serial",
         "stage_paths": ["features.11"]}, dim=576, backbone=bb)
    assert ad.stage_paths == ["features.11"]
    assert ad.stage_channels == [96]


def test_lora_default_target_on_mbnet():
    """features.11.block.3.0 is the 1x1 project conv (576 -> 96):
    rank * (in + out) = 16 * 672 = 10,752 trainable A/B params."""
    bb = _bare_frozen_mbnet()
    assert default_lora_targets(bb) == ["features.11.block.3.0"]
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = bb(x).clone()
    ad = LoRAAdapter(bb, rank=RANK)
    assert ad.num_injected == 1
    n = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    assert n == RANK * (576 + 96) == 10_752
    with torch.no_grad():
        assert torch.allclose(bb(x), ref, atol=1e-5)   # B=0 -> identity at init


def test_bitfit_on_mbnet_unfreezes_only_biases():
    bb = _bare_frozen_mbnet()
    ad = BitFitAdapter(backbone=bb)
    assert ad.num_bias_params > 0
    trainable = {n for n, p in bb.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(".bias") for n in trainable)
    assert sum(p.numel() for p in bb.parameters() if p.requires_grad) \
        == ad.num_bias_params


def test_linear_probe_on_mbnet_has_no_params():
    from src.adapters import LinearProbeAdapter
    ad = build_adapter({"type": "linear_probe"}, dim=576, backbone=None)
    assert isinstance(ad, LinearProbeAdapter)
    assert sum(p.numel() for p in ad.parameters()) == 0
    x = torch.randn(3, 576)
    assert torch.allclose(ad(x), x)


# --------------------------------------------------------------------------
# REGRESSION — Step 6 (ResNet-18) behaviour must be untouched by the
# backbone-agnostic refactor of the stage resolution.
# --------------------------------------------------------------------------
def test_resnet18_stage_paths_are_the_last_block_of_each_layer():
    bb = _bare_frozen_resnet18()
    assert resolve_stage_paths(bb) == ["layer1.1", "layer2.1", "layer3.1",
                                       "layer4.1"]
    # the resolved paths must be the SAME module objects Step 6 hooked via [-1]
    for path, attr in zip(resolve_stage_paths(bb),
                          ["layer1", "layer2", "layer3", "layer4"]):
        assert bb.get_submodule(path) is getattr(bb, attr)[-1]


def test_resnet18_placement_param_count_unchanged():
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=RANK, placement="parallel",
                          block_ids=[0, 1, 2, 3])
    assert tuple(ad.stage_channels) == (64, 128, 256, 512)
    # Step 6 reported 31,744 trainable adapter params for serial/parallel.
    assert sum(p.numel() for p in ad.parameters()) == 31_744


def test_resnet18_lora_default_target_unchanged():
    bb = _bare_frozen_resnet18()
    assert default_lora_targets(bb) == ["layer4.0.downsample.0"]
