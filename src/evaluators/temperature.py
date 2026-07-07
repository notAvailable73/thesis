"""Post-hoc temperature scaling (Guo et al. 2017).

Fits a single scalar T>0 that minimizes NLL of softmax(logits/T). Used as
the fair calibration baseline the evidential head must beat: T is fit ONCE
on the frozen validation episodes (seeds 10000-10099), then frozen and
applied to every test episode. T>0 does not move the argmax, so accuracy
is unchanged.

Reasoning (thesis instructions): source = Calibration summary (Guo et al.).
Pros: trivial, strong, the mandatory calibration baseline. Cons: needs a
val logit dump pass. Fit: directly answers "did you beat temperature
scaling?" — the biggest threat to the evidential-calibration claim.
Deviation from Guo: the "held-out validation set" is the frozen VAL
EPISODES (pooled query logits), the episodic analog of a fixed val set.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                    *, max_iter: int = 200, lr: float = 0.01) -> float:
    """Return scalar T>0 minimizing cross-entropy(softmax(logits/T), targets).

    Optimizes log_T (so T = exp(log_T) stays strictly positive) with Adam.
    """
    logits = logits.detach().float()
    targets = targets.detach().long()
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    opt = torch.optim.Adam([log_T], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), targets)
        loss.backward()
        opt.step()
    return float(log_T.exp().item())


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """softmax(logits / T). T>0 preserves argmax (accuracy unchanged)."""
    return torch.softmax(logits / float(T), dim=-1)
