import glob
import os
import random
from typing import Optional

import torch
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_MAT_NAME = "test_32x32.mat"


def _find_staged_svhn_root(data_root: str) -> Optional[str]:
    """A directory that already contains test_32x32.mat -- searched under
    data_root and anywhere in /kaggle/input at ANY depth (same convention as
    tinyimagenet_ood.py's `_find_extracted_tin_root`: Kaggle's dataset
    mount can nest the file arbitrarily deep under an owner/slug wrapper)."""
    search_roots = [data_root]
    if os.path.isdir("/kaggle/input"):
        search_roots.append("/kaggle/input")
    for root in search_roots:
        for hit in glob.glob(os.path.join(root, "**", _MAT_NAME), recursive=True):
            return os.path.dirname(hit)
    return None


def get_svhn_ood(data_root: str = "data", image_size: int = 224,
                 num_samples: int = 500, seed: int = 42) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    svhn_root = _find_staged_svhn_root(data_root)
    # download=False on the staged path: the discovered directory may be a
    # read-only /kaggle/input mount, so we trust what we already confirmed
    # is there rather than letting torchvision attempt a write if its own
    # md5 check ever disagreed.
    download = svhn_root is None
    if svhn_root is None:
        svhn_root = os.path.join(data_root, "svhn")
        # Browser-UA pre-fetch (same rationale as CIFAR): robust on fresh clones.
        from ._robust_download import ensure_archive
        ensure_archive(
            svhn_root, _MAT_NAME,
            ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
             "https://ufldl.stanford.edu/housenumbers/test_32x32.mat"],
        )
        download = True
    dataset = datasets.SVHN(root=svhn_root, split="test",
                             download=download, transform=transform)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return torch.stack([dataset[i][0] for i in indices])
