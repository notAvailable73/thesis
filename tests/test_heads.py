import pytest
import torch
from src.heads import LinearHead, EvidentialHead, build_head


def test_linear_head_shape():
    head = LinearHead(in_dim=64, num_classes=5)
    out = head(torch.randn(3, 64))
    assert out.shape == (3, 5)


@pytest.mark.parametrize("act", ["softplus", "exp_clamp", "relu"])
def test_evidential_head_is_nonnegative(act):
    head = EvidentialHead(in_dim=64, num_classes=5, activation=act)
    out = head(torch.randn(8, 64) * 5.0)
    assert out.shape == (8, 5)
    assert (out >= 0).all(), f"evidence must be non-negative for {act}"


def test_evidential_head_rejects_bad_activation():
    with pytest.raises(ValueError):
        EvidentialHead(in_dim=4, num_classes=2, activation="sigmoid")


def test_build_head_factory():
    h1 = build_head({"type": "softmax"}, in_dim=8, num_classes=3)
    h2 = build_head({"type": "evidential", "activation": "exp_clamp"},
                    in_dim=8, num_classes=3)
    assert isinstance(h1, LinearHead)
    assert isinstance(h2, EvidentialHead)
    assert h2.activation == "exp_clamp"
