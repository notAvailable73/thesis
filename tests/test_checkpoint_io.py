"""TMLR plan Task 6 — adapter-only checkpoints.

Only trainable tensors (+ the head's evidence scalars) are saved: a few KB per
run instead of ~45 MB, and the frozen backbone comes back from the public
torchvision weights. Losing the raw archives once is why this exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import src.models.bpeft_model as bm
from src.backbones import build_backbone as _real_build_backbone
from src.models import build_model
from src.utils import count_trainable_params, load_config
from src.utils.checkpoint_io import load_trainable_state, trainable_state_dict

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(bm, "build_backbone",
                        lambda name, pretrained=True: _real_build_backbone(name, pretrained=False))
    c = load_config(REPO / "configs/grid/mini_5shot_r18_lora_evidential_seed42.yaml")
    c["adapter"] = {"type": "lora", "rank": 4, "lora_targets": "multisite",
                    "lora_ranks": [3, 3, 4, 3]}
    return c


def _identical_backbones(a, b):
    """Random-weight backbones differ per build; copy the frozen part only."""
    frozen = {k: v for k, v in a.state_dict().items()
              if k not in trainable_state_dict(a)}
    b.load_state_dict(frozen, strict=False)


def test_roundtrip_reproduces_logits(cfg, tmp_path):
    torch.manual_seed(0)
    m1 = build_model(cfg)
    with torch.no_grad():
        for p in m1.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    torch.save(trainable_state_dict(m1), tmp_path / "adapter.pt")

    m2 = build_model(cfg)
    _identical_backbones(m1, m2)
    load_trainable_state(m2, torch.load(tmp_path / "adapter.pt"))
    m1.eval(); m2.eval()
    sx, sy = torch.randn(5, 3, 32, 32), torch.arange(5)
    qx = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m1.forward_proto(sx, sy, qx), m2.forward_proto(sx, sy, qx))


def test_trainable_state_size(cfg):
    m = build_model(cfg)
    sd = trainable_state_dict(m)
    n = sum(t.numel() for t in sd.values())
    assert n == count_trainable_params(m)            # evidential: affine params are trainable
    assert n == 31360 + 2


def test_unknown_key_raises(cfg):
    m = build_model(cfg)
    with pytest.raises(KeyError):
        load_trainable_state(m, {"not.a.key": torch.zeros(1)})


def test_missing_trainable_key_raises(cfg):
    m = build_model(cfg)
    sd = trainable_state_dict(m)
    sd.pop(next(iter(sd)))
    with pytest.raises(KeyError):
        load_trainable_state(build_model(cfg), sd)
