import numpy as np
import torch


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor,
                                num_bins: int = 15) -> float:
    """Standard ECE on (probs, targets). 15 bins (Guo et al. 2017)."""
    confidences, preds = probs.max(dim=-1)
    correct = (preds == targets).float()

    c = confidences.detach().cpu().numpy()
    a = correct.detach().cpu().numpy()
    n = len(c)
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c > lo) & (c <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(a[mask].mean() - c[mask].mean())
    return float(ece)


def brier_score(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return ((probs - onehot) ** 2).sum(dim=-1).mean().item()
