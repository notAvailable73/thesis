"""Step 7 tests for the Gaussian-noise far-OOD pool.

Covers: shape/dtype, reproducibility (seeded), clamp range, seed sensitivity.
Offline (no network, no GPU).
"""
import torch

from src.datasets import get_gaussian_ood


def test_shape_and_dtype():
    x = get_gaussian_ood(image_size=224, num_samples=500, seed=42)
    assert x.shape == (500, 3, 224, 224)
    assert x.dtype == torch.float32


def test_reproducible_same_seed():
    a = get_gaussian_ood(num_samples=16, seed=7)
    b = get_gaussian_ood(num_samples=16, seed=7)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = get_gaussian_ood(num_samples=16, seed=1)
    b = get_gaussian_ood(num_samples=16, seed=2)
    assert not torch.equal(a, b)


def test_clamped_range():
    x = get_gaussian_ood(num_samples=32, seed=3, clamp=3.0)
    assert float(x.min()) >= -3.0
    assert float(x.max()) <= 3.0
