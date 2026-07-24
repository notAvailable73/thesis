from .cifar_fs import (
    get_cifar_fs,
    get_cifar_fs_heldout_ood,
    load_cifar_fs_split,
    CIFAR100_CLASS_NAMES,
)
from .svhn_ood import get_svhn_ood
from .tinyimagenet_ood import get_tinyimagenet_ood
from .gaussian_noise_ood import get_gaussian_ood
from .episode_sampler import sample_episode, EpisodicIterableDataset


def build_dataset(spec: dict):
    """spec: {name: cifar_fs, split: train|val|test, ...} | {name: svhn_ood, ...}.

    Step 4 (Phase 2) extends `cifar_fs` to read the Bertinetto 64/16/20
    split from data/cifar_fs_split.json. If `class_ids` is in the spec
    (legacy Step 1-3 path), it overrides the split file and filters to
    those literal CIFAR-100 class IDs.
    """
    name = spec["name"]
    if name == "cifar_fs":
        return get_cifar_fs(
            data_root=spec.get("data_root", "data"),
            image_size=spec.get("image_size", 224),
            split=spec.get("split", "test"),
            class_ids=spec.get("class_ids"),
        )
    raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "build_dataset",
    "get_cifar_fs",
    "get_cifar_fs_heldout_ood",
    "load_cifar_fs_split",
    "CIFAR100_CLASS_NAMES",
    "get_svhn_ood",
    "get_tinyimagenet_ood",
    "get_gaussian_ood",
    "sample_episode",
    "EpisodicIterableDataset",
]
