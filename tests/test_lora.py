"""Step 5 tests for the rewritten LoRA adapter.

Covers: identity-at-init (LoRALayer, LoRAConv2d, and the injected backbone),
the analytical trainable-param count, and that injection leaves the rest of
the frozen backbone frozen.

NOTE ON THE ANALYTICAL FORMULA: implementation.txt 5.1 quotes the LoRA
param count as `2 * rank * (in + out)`. The honest count for a standard
A/B decomposition (A: in*rank, B: rank*out) is `rank * (in + out)` — exactly
half. These tests assert the correct value; the discrepancy is flagged in
step_writeups/step5.txt.
"""
import torch
import torch.nn as nn
import pytest

from torchvision.models import resnet18

from src.adapters import LoRALayer, LoRAConv2d, LoRAAdapter, build_adapter


def _bare_frozen_resnet18():
    """A ResNet-18 with random weights (no ImageNet download), fc removed and
    every parameter frozen — mirrors build_frozen_resnet18 without the network
    round-trip, so the LoRA-injection unit tests stay offline."""
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


# --------------------------------------------------------------------------
# LoRALayer (linear reference form)
# --------------------------------------------------------------------------
def test_lora_layer_delta_is_zero_at_init():
    layer = LoRALayer(in_features=64, out_features=32, rank=4)
    x = torch.randn(8, 64)
    out = layer(x)                       # base is None -> returns delta only
    assert out.shape == (8, 32)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_lora_layer_identity_over_base_at_init():
    base = nn.Linear(64, 32)
    layer = LoRALayer(64, 32, rank=4, base=base)
    x = torch.randn(8, 64)
    assert torch.allclose(layer(x), base(x), atol=1e-6)


def test_lora_layer_param_count():
    layer = LoRALayer(in_features=64, out_features=32, rank=4)
    n = sum(p.numel() for p in layer.parameters())
    # rank*(in+out) = 4*(64+32) = 384
    assert n == 4 * (64 + 32)


# --------------------------------------------------------------------------
# LoRAConv2d
# --------------------------------------------------------------------------
def test_lora_conv2d_identity_at_init():
    base = nn.Conv2d(8, 16, kernel_size=1)
    lora = LoRAConv2d(base, rank=4)
    x = torch.randn(2, 8, 7, 7)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_conv2d_1x1_param_count():
    base = nn.Conv2d(8, 16, kernel_size=1, bias=False)
    lora = LoRAConv2d(base, rank=4)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    # 1x1 conv: A = rank*in, B = out*rank -> rank*(in+out) = 4*(8+16) = 96
    assert trainable == 4 * (8 + 16)


# --------------------------------------------------------------------------
# LoRAAdapter (injection into the backbone)
# --------------------------------------------------------------------------
def test_lora_adapter_injects_target():
    bb = _bare_frozen_resnet18()
    LoRAAdapter(bb, rank=16, targets=["layer4.0.downsample.0"])
    assert isinstance(bb.get_submodule("layer4.0.downsample.0"), LoRAConv2d)


def test_lora_adapter_trainable_param_count():
    bb = _bare_frozen_resnet18()
    LoRAAdapter(bb, rank=16, targets=["layer4.0.downsample.0"])
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    # layer4.0.downsample.0 is a 1x1 conv 256->512: 16*(256+512) = 12288
    assert trainable == 16 * (256 + 512)


def test_lora_adapter_identity_at_init():
    torch.manual_seed(0)
    bb_ref = _bare_frozen_resnet18()
    torch.manual_seed(0)
    bb_lora = _bare_frozen_resnet18()
    LoRAAdapter(bb_lora, rank=16, targets=["layer4.0.downsample.0"])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_ref = bb_ref(x)
        out_lora = bb_lora(x)
    # B=0 at init -> injecting LoRA must not change the backbone's output.
    assert torch.allclose(out_ref, out_lora, atol=1e-5)


def test_lora_adapter_backbone_trainable_flag():
    assert LoRAAdapter.backbone_trainable is True


def test_build_adapter_lora_via_factory():
    bb = _bare_frozen_resnet18()
    a = build_adapter({"type": "lora", "rank": 16}, dim=512, backbone=bb)
    assert isinstance(a, LoRAAdapter)
    assert isinstance(bb.get_submodule("layer4.0.downsample.0"), LoRAConv2d)


def test_lora_adapter_rejects_non_conv_target():
    bb = _bare_frozen_resnet18()
    with pytest.raises(TypeError):
        LoRAAdapter(bb, rank=4, targets=["layer4.0.bn2"])
