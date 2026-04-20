# Evidential loss based on Sensoy et al. 2018 "Evidential Deep Learning to
# Quantify Classification Uncertainty" (NeurIPS 2018).
# Reference implementation: github.com/dougbrion/pytorch-classification-uncertainty

import torch
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    KL divergence between Dir(alpha) and uniform Dir(1,...,1).
    Sensoy 2018, Eq 13.
    Returns shape: (batch,)
    """
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)

    kl = (
        torch.lgamma(sum_alpha).squeeze(-1)
        - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
        - torch.lgamma(alpha).sum(dim=-1)
        + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1)
    )
    return kl


def evidential_mse_loss(evidence: torch.Tensor, target_onehot: torch.Tensor,
                        num_classes: int, kl_weight: float) -> torch.Tensor:
    """
    Evidential MSE loss (Sensoy 2018, Eq 5) + KL regularization.
    evidence: (B, K) non-negative, output of model in evidential mode
    target_onehot: (B, K) one-hot float tensor
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    err_term = (target_onehot - p) ** 2
    var_term = p * (1.0 - p) / (S + 1.0)
    loss_per_sample = (err_term + var_term).sum(dim=-1)

    # Zero out true-class alpha before KL so we only penalise wrong-class evidence
    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)

    return (loss_per_sample + kl_weight * kl).mean()


def softmax_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss for the softmax baseline."""
    return F.cross_entropy(logits, targets)


if __name__ == "__main__":
    evidence = torch.rand(4, 5) * 2
    target_oh = torch.eye(5)[[0, 1, 2, 3]]
    ev_loss = evidential_mse_loss(evidence, target_oh, num_classes=5, kl_weight=1.0)
    print(f"Evidential loss: {ev_loss.item():.6f}  finite={ev_loss.isfinite().item()}")
    assert ev_loss.isfinite()

    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 2, 3])
    ce_loss = softmax_ce_loss(logits, targets)
    print(f"CE loss:         {ce_loss.item():.6f}  finite={ce_loss.isfinite().item()}")
    assert ce_loss.isfinite()

    print("OK")
