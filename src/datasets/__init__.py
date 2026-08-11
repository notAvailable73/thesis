from .cifar_fs import (
    get_cifar_fs,
    get_cifar_fs_heldout_ood,
    load_cifar_fs_split,
    CIFAR100_CLASS_NAMES,
)
from .mini_imagenet import (
    get_mini_imagenet,
    get_mini_imagenet_heldout_ood,
    load_mini_imagenet_split,
    MINI_IMAGENET_SPLIT,
    MINI_IMAGENET_ALL_WNIDS,
)
from .svhn_ood import get_svhn_ood
from .tinyimagenet_ood import get_tinyimagenet_ood
from .gaussian_noise_ood import get_gaussian_ood
from .episode_sampler import sample_episode, EpisodicIterableDataset


def build_dataset(spec: dict):
    """spec: {name: cifar_fs, split: train|val|test, ...} |
    {name: mini_imagenet, split: train|val|test, ...} | {name: svhn_ood, ...}.

    Step 4 (Phase 2) extends `cifar_fs` to read the Bertinetto 64/16/20
    split from data/cifar_fs_split.json. If `class_ids` is in the spec
    (legacy Step 1-3 path), it overrides the split file and filters to
    those literal CIFAR-100 class IDs.

    Step 9 adds `mini_imagenet`, reading the Ravi & Larochelle 64/16/20
    split from data/mini_imagenet_split.json. There is no `class_ids`
    override for mini_imagenet -- every source layout is already split by
    design (see src/datasets/mini_imagenet.py's module docstring).
    """
    name = spec["name"]
    if name == "cifar_fs":
        return get_cifar_fs(
            data_root=spec.get("data_root", "data"),
            image_size=spec.get("image_size", 224),
            split=spec.get("split", "test"),
            class_ids=spec.get("class_ids"),
        )
    if name == "mini_imagenet":
        return get_mini_imagenet(
            data_root=spec.get("data_root", "data"),
            image_size=spec.get("image_size", 224),
            split=spec.get("split", "test"),
        )
    raise ValueError(f"Unknown dataset: {name}")


def get_id_split(dataset_cfg, split: str):
    """Route to the per-dataset in-distribution split loader based on
    `dataset_cfg.name` (defaults to "cifar_fs" for pre-Step-9 configs, which
    never set this key). This is the single dispatch point the episodic
    train/eval paths use instead of calling get_cifar_fs directly, so adding
    a dataset does not require touching scripts/train.py or
    scripts/evaluate.py beyond this call.

    dataset_cfg: a ConfigDict/dict with at least `name`; `data_root` and
    `image_size` fall back to the same defaults get_cifar_fs/get_mini_imagenet
    use if absent.
    """
    name = dataset_cfg.get("name", "cifar_fs")
    data_root = dataset_cfg.get("data_root", "data")
    image_size = int(dataset_cfg.get("image_size", 224))
    if name == "cifar_fs":
        return get_cifar_fs(
            data_root=data_root, image_size=image_size, split=split,
            class_ids=dataset_cfg.get("class_ids"),
        )
    if name == "mini_imagenet":
        return get_mini_imagenet(data_root=data_root, image_size=image_size,
                                 split=split)
    raise ValueError(f"Unknown dataset: {name!r}")


def get_heldout_near_ood(dataset_cfg, num_samples: int, seed: int,
                        heldout_split: str = "val"):
    """Free near-OOD pool for whichever in-distribution dataset is
    configured. Returns (pool_name, tensor): the pool NAME differs per
    dataset (cifar100_near / mini_near) so evaluate.py's OOD-pool dict stays
    free of dataset conditionals -- summary keys are already generated
    dynamically per pool name by src/evaluators/episodic.py, so a new name
    needs no evaluator change.
    """
    name = dataset_cfg.get("name", "cifar_fs")
    data_root = dataset_cfg.get("data_root", "data")
    image_size = int(dataset_cfg.get("image_size", 224))
    if name == "cifar_fs":
        x = get_cifar_fs_heldout_ood(
            data_root=data_root, image_size=image_size,
            num_samples=num_samples, seed=seed, heldout_split=heldout_split)
        return "cifar100_near", x
    if name == "mini_imagenet":
        x = get_mini_imagenet_heldout_ood(
            data_root=data_root, image_size=image_size,
            num_samples=num_samples, seed=seed, heldout_split=heldout_split)
        return "mini_near", x
    raise ValueError(f"Unknown dataset: {name!r}")


__all__ = [
    "build_dataset",
    "get_id_split",
    "get_heldout_near_ood",
    "get_cifar_fs",
    "get_cifar_fs_heldout_ood",
    "load_cifar_fs_split",
    "CIFAR100_CLASS_NAMES",
    "get_mini_imagenet",
    "get_mini_imagenet_heldout_ood",
    "load_mini_imagenet_split",
    "MINI_IMAGENET_SPLIT",
    "MINI_IMAGENET_ALL_WNIDS",
    "get_svhn_ood",
    "get_tinyimagenet_ood",
    "get_gaussian_ood",
    "sample_episode",
    "EpisodicIterableDataset",
]
