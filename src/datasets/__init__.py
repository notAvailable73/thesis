from .cifar_fs import get_cifar_fs
from .svhn_ood import get_svhn_ood
from .episode_sampler import sample_episode


def build_dataset(spec: dict):
    """spec: {name: cifar_fs | svhn_ood, ...}."""
    name = spec["name"]
    if name == "cifar_fs":
        return get_cifar_fs(
            data_root=spec.get("data_root", "data"),
            image_size=spec.get("image_size", 224),
            split=spec.get("split", "test"),
            class_ids=spec.get("class_ids"),
        )
    raise ValueError(f"Unknown dataset: {name}")


__all__ = ["build_dataset", "get_cifar_fs", "get_svhn_ood", "sample_episode"]
