import torch


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1,...,1)]. Sensoy 2018, Eq. 13. Returns shape (B,)."""
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    K = torch.tensor(float(num_classes), device=alpha.device)
    return (
        torch.lgamma(sum_alpha).squeeze(-1)
        - torch.lgamma(K)
        - torch.lgamma(alpha).sum(dim=-1)
        + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1)
    )


def evidential_mse_loss(evidence: torch.Tensor, target_onehot: torch.Tensor,
                        num_classes: int, kl_weight: float) -> torch.Tensor:
    """Sensoy 2018 Eq. 5 + Eq. 13 KL prior. KL only penalises wrong-class evidence."""
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    err = (target_onehot - p) ** 2
    var = p * (1.0 - p) / (S + 1.0)
    mse_term = (err + var).sum(dim=-1)

    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)

    return (mse_term + kl_weight * kl).mean()
