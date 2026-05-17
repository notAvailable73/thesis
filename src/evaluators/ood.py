import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """In-distribution=1, OOD=0. Higher score should mean more in-distribution."""
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def evidence_to_probs_and_vacuity(evidence: torch.Tensor, num_classes: int):
    """Convert non-negative evidence (B, K) to Dirichlet mean probs and vacuity = K/S."""
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / S
    vacuity = (num_classes / S).squeeze(-1)
    return probs, vacuity


def logits_to_probs_and_uncertainty(logits: torch.Tensor):
    """Softmax baseline: uncertainty = 1 - max_p."""
    probs = torch.softmax(logits, dim=-1)
    return probs, 1.0 - probs.max(dim=-1).values
