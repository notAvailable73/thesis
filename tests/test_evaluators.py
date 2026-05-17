import numpy as np
import torch
from src.evaluators import (
    accuracy, expected_calibration_error, brier_score, ood_auroc,
    evidence_to_probs_and_vacuity, logits_to_probs_and_uncertainty,
)


def test_accuracy_perfect():
    probs = torch.eye(4)
    targets = torch.arange(4)
    assert accuracy(probs, targets) == 1.0


def test_ece_perfect_calibration_is_zero():
    # All-confident, all-correct -> ECE = 0
    probs = torch.zeros(10, 3); probs[:, 0] = 1.0
    targets = torch.zeros(10, dtype=torch.long)
    assert expected_calibration_error(probs, targets, num_bins=15) == 0.0


def test_brier_zero_for_perfect_predictions():
    probs = torch.eye(4)
    targets = torch.arange(4)
    assert brier_score(probs, targets, num_classes=4) == 0.0


def test_ood_auroc_perfect_separation():
    id_scores = np.array([0.9, 0.95, 0.99])
    ood_scores = np.array([0.1, 0.2, 0.05])
    assert ood_auroc(id_scores, ood_scores) == 1.0


def test_evidence_probs_sum_to_one_and_vacuity_in_zero_one():
    evidence = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs, vac = evidence_to_probs_and_vacuity(evidence, num_classes=3)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    assert (vac >= 0).all() and (vac <= 1).all()
    # All-zero evidence -> alpha = 1 -> S = K -> vacuity = 1
    assert torch.isclose(vac[1], torch.tensor(1.0))


def test_logits_uncertainty():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    probs, unc = logits_to_probs_and_uncertainty(logits)
    assert torch.allclose(probs.sum(-1), torch.ones(1))
    assert unc.item() < 0.01
