import os
import random
import torch
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_svhn_ood(data_root: str = "data", image_size: int = 224,
                 num_samples: int = 500, seed: int = 42) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    svhn_root = os.path.join(data_root, "svhn")
    dataset = datasets.SVHN(root=svhn_root, split="test",
                             download=True, transform=transform)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return torch.stack([dataset[i][0] for i in indices])
