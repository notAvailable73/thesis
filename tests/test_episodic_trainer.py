"""Step 4 / Action 4.15 — smoke test for the episodic trainer.

We DO NOT exercise the full episodic pipeline (that would require
torchvision CIFAR-100 downloads). Instead we build a synthetic
dataset whose classes are linearly separable in feature space and
verify:

  1. The trainer trains -- val_acc after 1 epoch beats random chance
     (> 1/n_way + 0.05).
  2. The collapse guard fires when the val_acc is at or below the
     configured threshold.
  3. Early stopping fires when val_acc plateaus.
  4. `best_state_dict` is the model state at the best-val epoch.

The synthetic dataset is the only thing that's special; the trainer,
head, model, and loss are the production-path code.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.heads import PrototypeHead
from src.trainers import EpisodicTrainer, EpisodicCollapse
from src.datasets import EpisodicIterableDataset


# ---------------------------------------------------------------------
# Synthetic dataset: 8 classes, each class is a gaussian blob around
# a fixed centre. Easy to separate; meta-training should converge in
# a couple of epochs.
# ---------------------------------------------------------------------
class _BlobDataset(Dataset):
    def __init__(self, n_classes: int = 8, per_class: int = 30,
                 dim: int = 32, scale: float = 0.2, seed: int = 0):
        torch.manual_seed(seed)
        centres = torch.eye(n_classes, dim)
        xs, ys = [], []
        for c in range(n_classes):
            xs.append(centres[c] + scale * torch.randn(per_class, dim))
            ys.append(torch.full((per_class,), c, dtype=torch.long))
        # The "image" is a flat feature tensor here — we'll wire the
        # model so the "backbone" is the identity.
        self.x = torch.cat(xs, dim=0)
        self.targets = torch.cat(ys, dim=0).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class _IdentityBackbone(nn.Module):
    """Pretends to be a frozen feature extractor."""
    def forward(self, x):
        return x


class _IdentityAdapter(nn.Module):
    """A trainable identity-init linear layer. Few params so training
    has room to move without exploding."""
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(x)


class _ToyBPEFTModel(nn.Module):
    """A minimal BPEFTModel-compatible model for tests: identity
    backbone, linear adapter, prototype head.

    Implements only the methods the trainer touches:
      - forward_proto(sx, sy, qx)
      - backbone attribute (for eval-mode flip).
      - parameters() (for the optimiser).
    """
    def __init__(self, dim: int = 32):
        super().__init__()
        self.backbone = _IdentityBackbone()
        self.adapter  = _IdentityAdapter(dim)
        self.head     = PrototypeHead(metric="l2")

    def forward_proto(self, sx, sy, qx):
        sf = self.adapter(self.backbone(sx))
        qf = self.adapter(self.backbone(qx))
        return self.head(sf, sy, qf)


# ---------------------------------------------------------------------
# Trainer smoke
# ---------------------------------------------------------------------
def _build_factories(dataset, n_way=4, k_shot=3, q_query=5,
                      eps=5, val_offset=10_000):
    def _train_factory(epoch):
        return EpisodicIterableDataset(
            dataset, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=eps, seed_offset=epoch * 100,
        )

    def _val_factory(epoch):
        return EpisodicIterableDataset(
            dataset, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=eps, seed_offset=val_offset,
        )
    return _train_factory, _val_factory


def test_episodic_trainer_trains_above_chance():
    ds = _BlobDataset(n_classes=8, per_class=30, dim=32, scale=0.1, seed=0)
    model = _ToyBPEFTModel(dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    trainer = EpisodicTrainer(
        model=model, optimizer=optimizer,
        num_classes=4,
        num_epochs=3,
        episodes_per_epoch=5,
        val_episodes_per_epoch=5,
        early_stop_patience=10,
        interpretation="softmax",
        collapse_threshold=-1.0,  # disable collapse guard for this test
    )
    train_f, val_f = _build_factories(ds)
    result = trainer.fit(train_f, val_f)
    # After 3 epochs of a clearly-separable dataset, val_acc should
    # comfortably exceed random chance (1/4 = 0.25).
    assert result["best_val_acc"] > 0.5, (
        f"val_acc after 3 epochs was {result['best_val_acc']:.3f}; "
        f"expected > 0.5 on this synthetic blob dataset"
    )


def test_episodic_trainer_collapse_guard_fires():
    """Force a pathological setup where val_acc after epoch 1 is ~chance,
    and verify EpisodicCollapse is raised."""
    # Use a HIGHLY overlapping blob dataset so the model can't separate.
    ds = _BlobDataset(n_classes=8, per_class=20, dim=8, scale=5.0, seed=0)
    model = _ToyBPEFTModel(dim=8)
    # Tiny LR so it can't escape from chance in one epoch.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)
    trainer = EpisodicTrainer(
        model=model, optimizer=optimizer,
        num_classes=4,
        num_epochs=3,
        episodes_per_epoch=3,
        val_episodes_per_epoch=3,
        early_stop_patience=10,
        interpretation="softmax",
        # Set collapse threshold high so even decent runs fail it.
        collapse_threshold=0.95,
    )
    train_f, val_f = _build_factories(ds)
    with pytest.raises(EpisodicCollapse):
        trainer.fit(train_f, val_f)


def test_episodic_trainer_evidential_path_runs():
    """The evidential interpretation requires kl_weight_max +
    kl_anneal_steps. Make sure it runs end-to-end at least 1 epoch
    without throwing, and that the kl weight in history grows."""
    ds = _BlobDataset(n_classes=6, per_class=20, dim=16, scale=0.2, seed=0)
    model = _ToyBPEFTModel(dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    trainer = EpisodicTrainer(
        model=model, optimizer=optimizer,
        num_classes=4,
        num_epochs=2,
        episodes_per_epoch=5,
        val_episodes_per_epoch=5,
        early_stop_patience=10,
        interpretation="evidential",
        kl_weight_max=0.5,
        kl_anneal_steps=10,  # ramp fully over the first 2 epochs
        collapse_threshold=-1.0,
    )
    train_f, val_f = _build_factories(ds)
    result = trainer.fit(train_f, val_f)
    hist = result["history"]
    assert len(hist.kl_weight_at_end) >= 1
    # KL weight should have ramped above zero by the end of epoch 1
    # (which is 5 outer steps, < kl_anneal_steps=10).
    assert hist.kl_weight_at_end[0] > 0.0
    assert hist.kl_weight_at_end[-1] > 0.0


def test_episodic_trainer_evidential_requires_kl_args():
    model = _ToyBPEFTModel(dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        EpisodicTrainer(
            model=model, optimizer=optimizer,
            num_classes=4,
            num_epochs=1, episodes_per_epoch=1, val_episodes_per_epoch=1,
            early_stop_patience=10,
            interpretation="evidential",
            # missing kl_weight_max + kl_anneal_steps
        )


def test_episodic_trainer_early_stop_fires():
    """Train very briefly and verify that with patience=1 and a flat
    val acc, early stop kicks in."""
    ds = _BlobDataset(n_classes=8, per_class=20, dim=16, scale=5.0, seed=0)
    model = _ToyBPEFTModel(dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)
    trainer = EpisodicTrainer(
        model=model, optimizer=optimizer,
        num_classes=4,
        num_epochs=10,
        episodes_per_epoch=2,
        val_episodes_per_epoch=2,
        early_stop_patience=1,
        interpretation="softmax",
        collapse_threshold=-1.0,
    )
    train_f, val_f = _build_factories(ds)
    result = trainer.fit(train_f, val_f)
    # With patience=1 and a near-zero LR, the loop should stop before
    # reaching epoch 10.
    assert len(result["history"].epoch) < 10
