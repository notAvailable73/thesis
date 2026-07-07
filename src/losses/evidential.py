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
                        num_classes: int, kl_weight: float,
                        *, prior_per_class: float = 1.0,
                        use_variance: bool = True) -> torch.Tensor:
    """Sensoy 2018 Eq. 5 + Eq. 13 KL prior, with R-EDL knobs (Survey EDL).

    prior_per_class: added Dirichlet mass per class (alpha = evidence +
      prior_per_class). 1.0 recovers Sensoy's rigid "+1" prior; smaller
      values give less prior mass (sharper mean, higher vacuity K/S).
    use_variance: include the Bayes-risk variance term (True = Sensoy;
      False = the R-EDL relaxation that drops the variance-minimising
      regulariser, which the Survey-EDL summary flags as a driver of
      miscalibration).

    Defaults (prior_per_class=1.0, use_variance=True) reproduce the
    original loss bit-for-bit. KL still only penalises wrong-class evidence.
    """
    alpha = evidence + float(prior_per_class)
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    mse_term = ((target_onehot - p) ** 2).sum(dim=-1)
    if use_variance:
        var = p * (1.0 - p) / (S + 1.0)
        mse_term = mse_term + var.sum(dim=-1)

    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)

    return (mse_term + kl_weight * kl).mean()
