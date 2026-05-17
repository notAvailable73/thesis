import torch
from src.losses import (
    evidential_mse_loss, kl_divergence_dirichlet, softmax_ce_loss, build_loss,
)


def test_evidential_loss_finite_and_decreases_with_evidence():
    # As we pile evidence on the true class, the MSE term should drop.
    target = torch.tensor([0])
    target_oh = torch.eye(3)[target]
    weak = torch.tensor([[0.1, 0.0, 0.0]])
    strong = torch.tensor([[10.0, 0.0, 0.0]])
    L_weak = evidential_mse_loss(weak, target_oh, num_classes=3, kl_weight=0.0)
    L_strong = evidential_mse_loss(strong, target_oh, num_classes=3, kl_weight=0.0)
    assert torch.isfinite(L_weak) and torch.isfinite(L_strong)
    assert L_strong.item() < L_weak.item()


def test_kl_uniform_is_zero():
    # KL(Dir(1,1,1) || Dir(1,1,1)) = 0
    alpha = torch.ones(2, 3)
    kl = kl_divergence_dirichlet(alpha, num_classes=3)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-5)


def test_ce_loss_decreases_with_correct_logits():
    correct = torch.tensor([[5.0, 0.0, 0.0]])
    wrong = torch.tensor([[0.0, 0.0, 5.0]])
    y = torch.tensor([0])
    assert softmax_ce_loss(correct, y).item() < softmax_ce_loss(wrong, y).item()


def test_kl_anneal_is_monotone_in_step():
    fn = build_loss({"type": "evidential", "kl_weight_max": 0.5, "kl_anneal_steps": 100})
    target = torch.tensor([0])
    # Same large evidence -> MSE term is constant. KL contribution grows with step.
    evidence = torch.tensor([[5.0, 0.5, 0.5]])
    l_early = float(fn(evidence, target, step=1, num_classes=3))
    l_mid   = float(fn(evidence, target, step=50, num_classes=3))
    l_full  = float(fn(evidence, target, step=200, num_classes=3))
    # Wrong-class evidence > 0 => KL term is positive; loss should grow with step.
    assert l_early <= l_mid <= l_full + 1e-6
