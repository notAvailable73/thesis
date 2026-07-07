import torch
from src.losses import (
    evidential_mse_loss, kl_divergence_dirichlet, softmax_ce_loss, build_loss,
)
from src.evaluators import evidence_to_probs_and_vacuity


def test_redl_defaults_match_legacy_loss():
    """Back-compat: prior_per_class=1.0 + use_variance=True reproduces the
    original Sensoy MSE+KL loss bit-for-bit."""
    torch.manual_seed(0)
    evidence = torch.rand(16, 5) * 3.0
    target = torch.eye(5)[torch.randint(0, 5, (16,))]
    legacy = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.3)
    explicit = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.3,
                                   prior_per_class=1.0, use_variance=True)
    assert torch.allclose(legacy, explicit)


def test_redl_drop_variance_changes_loss():
    """R-EDL relaxation: dropping the (non-negative) variance term lowers loss."""
    torch.manual_seed(0)
    evidence = torch.rand(16, 5) * 3.0
    target = torch.eye(5)[torch.randint(0, 5, (16,))]
    with_var = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.0,
                                   use_variance=True)
    no_var = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.0,
                                 use_variance=False)
    assert not torch.allclose(with_var, no_var)
    assert no_var < with_var  # variance term is non-negative


def test_vacuity_uses_prior_per_class():
    """Smaller prior_per_class => smaller S => larger vacuity K/S."""
    evidence = torch.ones(4, 5) * 2.0
    _, vac1 = evidence_to_probs_and_vacuity(evidence, 5, prior_per_class=1.0)
    _, vac_small = evidence_to_probs_and_vacuity(evidence, 5, prior_per_class=0.1)
    assert (vac_small > vac1).all()


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
