"""Step 5 tests for BitFit: only bias parameters become trainable, and the
count matches the analytical (sum of `.bias` numel)."""
import torch.nn as nn
from torchvision.models import resnet18

from src.adapters import BitFitAdapter, build_adapter


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def test_bitfit_unfreezes_only_biases():
    bb = _bare_frozen_resnet18()
    BitFitAdapter(bb)
    for name, p in bb.named_parameters():
        if name.endswith(".bias"):
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should stay frozen"


def test_bitfit_param_count_matches_analytical():
    bb = _bare_frozen_resnet18()
    expected = sum(p.numel() for n, p in bb.named_parameters()
                   if n.endswith(".bias"))
    adapter = BitFitAdapter(bb)
    assert expected > 0
    assert adapter.num_bias_params == expected
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    assert trainable == expected


def test_bitfit_backbone_trainable_flag():
    assert BitFitAdapter.backbone_trainable is True


def test_bitfit_via_factory_requires_backbone():
    import pytest
    with pytest.raises(ValueError):
        build_adapter({"type": "bitfit"}, dim=512, backbone=None)
