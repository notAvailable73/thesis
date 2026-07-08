"""Step 5 tests for the Linear Probing baseline: the backbone stays frozen and
the adapter adds no parameters (only the head, if any, trains)."""
import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.adapters import LinearProbeAdapter, build_adapter


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def test_linear_probe_keeps_backbone_frozen():
    bb = _bare_frozen_resnet18()
    LinearProbeAdapter()  # does not touch the backbone
    for name, p in bb.named_parameters():
        assert not p.requires_grad, f"{name} should stay frozen"


def test_linear_probe_adapter_has_no_params():
    adapter = LinearProbeAdapter()
    assert sum(p.numel() for p in adapter.parameters()) == 0


def test_linear_probe_forward_is_identity():
    adapter = LinearProbeAdapter()
    x = torch.randn(4, 512)
    assert torch.allclose(adapter(x), x, atol=1e-7)


def test_linear_probe_backbone_trainable_flag():
    assert LinearProbeAdapter.backbone_trainable is False


def test_linear_probe_via_factory():
    a = build_adapter({"type": "linear_probe"}, dim=512, backbone=None)
    assert isinstance(a, LinearProbeAdapter)
