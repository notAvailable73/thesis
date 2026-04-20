import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = probs.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor,
                                num_bins: int = 10) -> float:
    confidences, preds = probs.max(dim=-1)
    correct = (preds == targets).float()

    confidences = confidences.cpu().numpy()
    correct = correct.cpu().numpy()

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def brier_score(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return ((probs - onehot) ** 2).sum(dim=-1).mean().item()


def ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    id_scores: higher = more in-distribution (e.g. 1 - vacuity)
    ood_scores: same scale but from OOD samples
    Labels: in-distribution=1, OOD=0
    """
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


if __name__ == "__main__":
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(50, 5), dim=-1)
    targets = torch.randint(0, 5, (50,))

    print(f"Accuracy : {accuracy(probs, targets):.3f}")
    print(f"ECE      : {expected_calibration_error(probs, targets):.3f}")
    print(f"Brier    : {brier_score(probs, targets, 5):.3f}")

    id_scores  = np.random.uniform(0.7, 1.0, 75)
    ood_scores = np.random.uniform(0.0, 0.4, 500)
    print(f"OOD AUROC: {ood_auroc(id_scores, ood_scores):.3f}  (expect > 0.9)")
