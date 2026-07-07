"""Step 4.5 / W1 — energy OOD score (Liu et al. 2020) unit tests."""
import torch

from src.evaluators import energy_score


def test_energy_higher_for_confident_logits():
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    flat = torch.zeros(1, 5)
    assert energy_score(confident).item() > energy_score(flat).item()


def test_energy_shape_and_finiteness():
    logits = torch.randn(32, 5)
    s = energy_score(logits)
    assert s.shape == (32,)
    assert torch.isfinite(s).all()
