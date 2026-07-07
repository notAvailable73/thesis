"""Step 4.5 / W3 — evaluate_episodic score x OOD-pool matrix."""
import torch
import torch.nn as nn

from src.heads import PrototypeHead
from src.evaluators.episodic import evaluate_episodic


class _StubModel(nn.Module):
    """Minimal model matching what evaluate_episodic calls: an Identity
    backbone (inputs are already features) + a PrototypeHead, with the
    adapter folded into an identity forward_proto_from_features."""

    def __init__(self, head):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = head

    def forward_proto_from_features(self, sf, sy, qf):
        return self.head(sf, sy, qf)


def _make_episodes(n_episodes=3, n_way=3, k_shot=2, q_per=2, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_way, D) * 5.0  # separated class centers
    episodes = []
    for _ in range(n_episodes):
        sx, sy, qx, qy = [], [], [], []
        for c in range(n_way):
            sx.append(centers[c] + 0.1 * torch.randn(k_shot, D, generator=g))
            sy += [c] * k_shot
            qx.append(centers[c] + 0.1 * torch.randn(q_per, D, generator=g))
            qy += [c] * q_per
        episodes.append((torch.cat(sx), torch.tensor(sy),
                         torch.cat(qx), torch.tensor(qy)))
    return episodes


def _pools(D=8, n=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "svhn_far": torch.randn(n, D, generator=g) * 3.0,
        "cifar100_near": torch.randn(n, D, generator=g) * 0.5 + 2.0,
    }


def test_softmax_matrix_keys_and_ts():
    head = PrototypeHead(metric="cosine", cosine_scale=10.0)
    model = _StubModel(head)
    out = evaluate_episodic(
        model, _make_episodes(), _pools(), num_classes=3,
        interpretation="softmax", temperature=1.5,
    )
    s = out["summary"]
    # matrix cells: {msp, energy, ts_msp} x {svhn_far, cifar100_near}
    for pool in ("svhn_far", "cifar100_near"):
        for score in ("msp", "energy", "ts_msp"):
            assert f"ood_auroc__{pool}__{score}" in s
            assert f"fpr_at_95_tpr__{pool}__{score}" in s
    # legacy keys preserved (primary pool + native msp score)
    assert "ood_auroc_mean" in s and "fpr_at_95_tpr_mean" in s
    assert s["primary_ood_pool"] == "svhn_far"
    # temperature-scaled calibration present
    assert "ece_ts" in s and "brier_ts" in s


def test_evidential_matrix_uses_vacuity_and_no_ts():
    head = PrototypeHead(metric="cosine", cosine_scale=10.0)
    model = _StubModel(head)
    out = evaluate_episodic(
        model, _make_episodes(), _pools(), num_classes=3,
        interpretation="evidential",
    )
    s = out["summary"]
    assert "ood_auroc__svhn_far__vacuity" in s
    assert "ood_auroc__cifar100_near__vacuity" in s
    assert "ood_auroc_mean" in s          # legacy preserved
    assert "ece_ts" not in s              # no temperature for evidential
    # softmax-only scores must not appear for the evidential head
    assert "ood_auroc__svhn_far__msp" not in s
