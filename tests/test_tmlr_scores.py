"""TMLR plan Task 6 — MaxLogit + Mahalanobis scores, signed calibration gap,
VAL-only evaluation. All on CPU with random-weight backbones and synthetic data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import rq_core as R  # noqa: E402
import rq_drivers as D  # noqa: E402
import src.models.bpeft_model as bm  # noqa: E402
from src.backbones import build_backbone as _real_build_backbone  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402

GRID_CFG = REPO / "configs/grid/cifar_5shot_r18_parallel_evidential_seed42.yaml"


class _Synthetic(torch.utils.data.Dataset):
    """n_classes x per_class random 3x32x32 images with a .targets list."""

    def __init__(self, n_classes=5, per_class=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_classes * per_class, 3, 32, 32, generator=g)
        self.targets = [c for c in range(n_classes) for _ in range(per_class)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


@pytest.fixture
def offline(monkeypatch):
    """Random-weight backbones + synthetic splits; no download, no data dir."""
    monkeypatch.setattr(bm, "build_backbone",
                        lambda name, pretrained=True: _real_build_backbone(name, pretrained=False))
    splits = {"train": _Synthetic(8, 6, 1), "val": _Synthetic(5, 4, 2),
              "test": _Synthetic(5, 4, 3)}
    monkeypatch.setattr(R, "cached_id_split", lambda dcfg, split: splits[split])
    cfg = load_config(GRID_CFG)
    cfg["dataset"]["k_shot"] = 1
    cfg["dataset"]["q_query"] = 2
    return cfg


def test_maxlogit_is_max_logit():
    logits = torch.tensor([[1.0, 3.0, -2.0], [0.5, 0.2, 0.9]])
    from src.heads.prototype_head import PrototypeHead
    s = R.all_id_scores(logits, PrototypeHead(metric="cosine"), num_classes=3,
                        temperature=None, prior_per_class=1.0, affine_valfit=(1.0, 0.0))
    assert torch.equal(s["maxlogit"], torch.tensor([3.0, 0.9]))
    assert "maxlogit" in R.FACTORIAL_SCORES and "mahalanobis" in R.FACTORIAL_SCORES


def test_mahalanobis_identity_precision():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    protos = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    s = R.mahalanobis_score(q, protos, torch.eye(2))
    expected = -torch.tensor([0.0, min(2.0, 0.36 + 0.04)])
    assert torch.allclose(s, expected, atol=1e-6)


def test_mahalanobis_prefers_near_prototype():
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    near = torch.tensor([[0.99, 0.14]])
    far = torch.tensor([[-0.7, -0.7]])
    P = torch.tensor([[2.0, 0.3], [0.3, 1.5]])
    assert R.mahalanobis_score(near, protos, P) > R.mahalanobis_score(far, protos, P)


def test_fit_base_precision_leaves_global_rng(offline):
    model = build_model(offline)
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    P = R.fit_base_precision(model, offline, torch.device("cpu"), n_per_class=3)
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert P.shape == (512, 512) and torch.isfinite(P).all()
    assert torch.allclose(P, P.T, atol=1e-5)


def test_existing_keys_unchanged(offline):
    torch.manual_seed(0)
    model = build_model(offline)
    pools = {"svhn_far": torch.randn(6, 512), "tin_near": torch.randn(6, 512)}
    kw = dict(test_seeds=[0, 1, 2], ood_pools=pools, device=torch.device("cpu"),
              temperature=1.3, affine_valfit=(2.0, -6.0), prior_per_class=1.0,
              log_every=0)
    # Production order: the precision is fitted BEFORE the evaluation it feeds.
    P = R.fit_base_precision(model, offline, torch.device("cpu"), n_per_class=3)
    full = R.factorial_evaluate(model, offline, precision=P, **kw)
    base = R.factorial_evaluate(model, offline, **kw)
    for k, v in base.items():
        assert full[k] == v, k
    assert "ood_auroc__svhn_far__mahalanobis" in full
    assert "ood_auroc__svhn_far__maxlogit" in base
    assert "ood_auroc__svhn_far__mahalanobis" not in base


def test_signed_gap_sign(offline):
    torch.manual_seed(0)
    model = build_model(offline)
    s = R.factorial_evaluate(model, offline, test_seeds=[0, 1], ood_pools={},
                             device=torch.device("cpu"), temperature=None,
                             affine_valfit=(2.0, -6.0), prior_per_class=1.0, log_every=0)
    for name in ("softmax", "evidential_native"):
        conf_minus_acc = s[f"signed_calibration_gap__{name}"]
        assert -1.0 <= conf_minus_acc <= 1.0
    probs = torch.tensor([[0.95, 0.05], [0.9, 0.1]])
    assert R.signed_calibration_gap(probs, torch.tensor([1, 0])) == pytest.approx(0.925 - 0.5)


def test_val_only_never_touches_test(offline, tmp_path, monkeypatch):
    cfg_path = tmp_path / "cell.yaml"
    cfg_path.write_text(yaml.safe_dump({"extends": str(GRID_CFG),
                                        "dataset": {"k_shot": 1, "q_query": 2}}))
    model = build_model(load_config(cfg_path))
    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "best_val_epoch": 1}, ckpt)

    def _boom(*a, **k):
        raise AssertionError("TEST seeds must not be loaded in val_only mode")
    monkeypatch.setattr(D, "_load_test_seeds", _boom)
    real_split = R.cached_id_split

    def _no_test(dcfg, split):
        if split == "test":
            raise AssertionError("val_only must never build the TEST split")
        return real_split(dcfg, split)
    monkeypatch.setattr(R, "cached_id_split", _no_test)   # real VAL path, TEST poisoned
    rec = D.val_only_run_one(REPO, cfg_path, ckpt, tmp_path / "out.json",
                             device=torch.device("cpu"), log=lambda *a: None)
    for k in ("val_ece_pooled__evidential_native", "val_accuracy__evidential_native",
              "val_nll__evidential_native", "env", "n_params"):
        assert k in rec
    assert (tmp_path / "out.json").exists()
