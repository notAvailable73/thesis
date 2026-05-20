"""Out-of-distribution detection scoring.

Step 1-3 used ood_auroc only. Step 4 (Phase 2) adds FPR@95 — the false-
positive rate at the 95% true-positive-rate operating point — because
proposal §7 lists it under Reliability metrics.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """In-distribution=1, OOD=0. Higher score should mean more in-distribution."""
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """False positive rate (on OOD samples) at the threshold that gives
    95% true positive rate on in-distribution samples.

    In-distribution is the POSITIVE class (label = 1). A high "score"
    means the sample looks in-distribution. The threshold is set so
    that 95% of ID samples have score >= threshold; the FPR is the
    fraction of OOD samples that also pass that threshold.

    Lower is better. Reported in proposal §7 alongside AUROC.

    Edge cases:
      - if either input is empty, returns 1.0 (worst possible).
      - if no threshold reaches 95% TPR exactly, returns the FPR at the
        nearest threshold that achieves >= 0.95 TPR.
    """
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return 1.0
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    # roc_curve returns FPR/TPR ordered by decreasing threshold.
    # Find the smallest TPR >= 0.95; report its FPR.
    mask = tpr >= 0.95
    if not mask.any():
        return 1.0
    return float(fpr[mask][0])


def evidence_to_probs_and_vacuity(evidence: torch.Tensor, num_classes: int):
    """Convert non-negative evidence (B, K) to Dirichlet mean probs and
    vacuity = K/S. `evidence` is the head's OWN output (Linear+softplus
    EvidentialHead) OR softplus(prototype_logits) — the math is the same.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / S
    vacuity = (num_classes / S).squeeze(-1)
    return probs, vacuity


def logits_to_probs_and_uncertainty(logits: torch.Tensor):
    """Softmax baseline: uncertainty = 1 - max_p."""
    probs = torch.softmax(logits, dim=-1)
    return probs, 1.0 - probs.max(dim=-1).values
