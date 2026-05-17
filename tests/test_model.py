"""End-to-end model wiring test. Skips if torchvision weights aren't available."""
import pytest
import torch

from src.utils import load_config, count_trainable_params
from src.models import build_model


@pytest.fixture(scope="module")
def base_cfg(tmp_path_factory):
    from pathlib import Path
    return load_config(Path(__file__).resolve().parents[1] / "configs/exp_step1.yaml")


def test_model_forward_evidential(base_cfg):
    model = build_model(base_cfg)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)
    assert (out >= 0).all(), "Evidential head must produce non-negative evidence"


def test_only_adapter_and_head_are_trainable(base_cfg):
    model = build_model(base_cfg)
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    adapter_trainable = all(p.requires_grad for p in model.adapter.parameters())
    head_trainable = all(p.requires_grad for p in model.head.parameters())
    assert not backbone_trainable, "Backbone must be frozen"
    assert adapter_trainable, "Adapter must be trainable"
    assert head_trainable, "Head must be trainable"


def test_trainable_param_count_matches_step1(base_cfg):
    """Step 1 reported 19,477 trainable params for bottleneck + 5-way head."""
    model = build_model(base_cfg)
    n = count_trainable_params(model)
    assert n == 19477, f"Expected 19,477 trainable params, got {n}"
