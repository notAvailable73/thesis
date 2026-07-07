"""Step 4.5 / W1 — temperature scaling (Guo et al. 2017) unit tests."""
import torch
import torch.nn.functional as F

from src.evaluators import fit_temperature, apply_temperature


def test_fit_temperature_reduces_nll():
    """The defining guarantee of temperature scaling: the fitted T achieves
    NLL no worse than T=1 on the fit set (it optimizes exactly that)."""
    torch.manual_seed(0)
    logits = torch.randn(500, 5) * 4.0
    targets = torch.randint(0, 5, (500,))  # noisy labels -> non-trivial T
    T = fit_temperature(logits, targets)
    nll_1 = F.cross_entropy(logits / 1.0, targets)
    nll_T = F.cross_entropy(logits / T, targets)
    assert nll_T <= nll_1 + 1e-4


def test_fit_temperature_softens_overconfident_logits():
    # Hugely overconfident logits that are often WRONG -> optimal T > 1.
    torch.manual_seed(0)
    logits = torch.randn(500, 5) * 20.0     # extreme confidence
    targets = torch.randint(0, 5, (500,))   # random labels -> should soften
    T = fit_temperature(logits, targets)
    assert T > 1.5


def test_apply_temperature_preserves_argmax():
    torch.manual_seed(0)
    logits = torch.randn(64, 5)
    p1 = apply_temperature(logits, 1.0)
    p7 = apply_temperature(logits, 7.0)
    assert torch.equal(p1.argmax(-1), p7.argmax(-1))
    assert torch.allclose(p7.sum(-1), torch.ones(64), atol=1e-5)
