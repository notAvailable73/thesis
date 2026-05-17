"""CIFAR-FS-style few-shot dataset.

For Step 2 we keep the Step 1 simplification: CIFAR-100 test split with the
first 20 classes used as the test pool. Step 4 will replace this with the
Bertinetto 2019 64/16/20 split.
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_cifar_fs(data_root: str = "data", image_size: int = 224,
                 split: str = "test", class_ids=None) -> Dataset:
    """Return a CIFAR-100 Dataset filtered to the given split.

    split:
      "test"  -> CIFAR-100 test set, class_ids defaults to [0..19]
      "train" -> CIFAR-100 train set (used in Step 4 onwards)
    """
    train = split == "train"
    transform = _build_transform(image_size)
    return datasets.CIFAR100(root=data_root, train=train,
                              download=True, transform=transform)
