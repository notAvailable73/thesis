"""Step 5 tests for the Full Fine-Tuning baseline: every backbone parameter
becomes trainable."""
import torch.nn as nn
from torchvision.models import resnet18

from src.adapters import FullFTAdapter, build_adapter


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def test_full_ft_unfreezes_everything():
    bb = _bare_frozen_resnet18()
    FullFTAdapter(bb)
    for name, p in bb.named_parameters():
        assert p.requires_grad, f"{name} should be trainable under full-FT"


def test_full_ft_param_count():
    bb = _bare_frozen_resnet18()
    total = sum(p.numel() for p in bb.parameters())
    adapter = FullFTAdapter(bb)
    assert adapter.num_backbone_params == total
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    assert trainable == total
    # ResNet-18 backbone (fc removed) is ~11.2M params.
    assert trainable > 11_000_000


def test_full_ft_backbone_trainable_flag():
    assert FullFTAdapter.backbone_trainable is True


def test_full_ft_via_factory_requires_backbone():
    import pytest
    with pytest.raises(ValueError):
        build_adapter({"type": "full_ft"}, dim=512, backbone=None)
