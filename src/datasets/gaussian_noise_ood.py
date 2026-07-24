"""Far-OOD Gaussian-noise pool (Step 7, RQ3).

Pure far-OOD sanity/ablation: seeded N(0,1) noise at 3x224x224, clamped to
[-clamp, clamp]. Returned in the same backbone-ready form as get_svhn_ood /
get_tinyimagenet_ood (a float tensor fed straight to the frozen backbone).
ImageNet-normalized real images sit at ~unit per-channel scale, so raw N(0,1)
is the right magnitude for a pure-noise far-OOD. Deterministic via a
torch.Generator so the pool is byte-identical across runs (respects the repo's
reproducibility invariant).

Role in RQ3: Gaussian is the EASY far-OOD end (if the model can't separate white
noise from CIFAR features something is broken) — a sanity check / ablation, not
the discriminating test. The discriminating near-OOD evidence stays the
cifar100_near (clean, disjoint) and tin_near pools.
"""
import torch


def get_gaussian_ood(image_size: int = 224, num_samples: int = 500,
                     seed: int = 42, clamp: float = 3.0) -> torch.Tensor:
    """Return (num_samples, 3, image_size, image_size) seeded N(0,1) noise,
    clamped to [-clamp, clamp]. Deterministic in `seed`."""
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(int(num_samples), 3, int(image_size), int(image_size),
                    generator=g)
    if clamp is not None:
        x = x.clamp(-float(clamp), float(clamp))
    return x
