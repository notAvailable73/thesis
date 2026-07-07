"""CIFAR-FS few-shot dataset (Bertinetto 2019 64/16/20 split).

Step 4 (Phase 2) replaces the Step 1/2 stand-in (CIFAR-100 test, classes
0..19) with the proper Bertinetto split. The split is loaded from
data/cifar_fs_split.json as class NAMES; the loader converts names to
CIFAR-100 integer IDs at runtime using torchvision's `.classes` attribute.

The split file is canonical (do not regenerate). The committed file ships
with a synthetic alphabetical fallback and a `_status: synthetic_fallback`
flag — notebooks/step4_episodic.ipynb fetches the canonical Bertinetto
split and overwrites the file before any training run. `load_cifar_fs_split`
WARNS if `_status == 'synthetic_fallback'` so accidental training on the
fallback is loud, not silent.

Three asserts at load time:
  1. train ∩ val ∩ test == ∅  (no class appears in two splits)
  2. train ∪ val ∪ test == set(CIFAR-100)   (every class is in one split)
  3. |train| == 64, |val| == 16, |test| == 20  (matches the spec)

The function returned by `get_cifar_fs(split=...)` is a torchvision
CIFAR-100 dataset filtered to the split's class IDs, with labels
RE-INDEXED into a contiguous range:
  train -> 0..63, val -> 0..15, test -> 0..19.

This re-indexing is what lets the episode sampler treat the in-split
class IDs as a simple 0..N range without surprises.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_PATH = _REPO_ROOT / "data" / "cifar_fs_split.json"


def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def load_cifar_fs_split(split_path: Path | str | None = None,
                        cifar100_classes: List[str] | None = None,
                        ) -> Dict[str, List[int]]:
    """Load and validate the CIFAR-FS class split.

    Returns a dict with keys "train", "val", "test", values are sorted
    lists of CIFAR-100 integer class IDs (0..99).

    Args:
        split_path: optional explicit path to the split JSON. Defaults to
                    data/cifar_fs_split.json.
        cifar100_classes: optional list of 100 CIFAR-100 class names in
                    integer-ID order (i.e. cifar100_classes[i] is the name
                    of class i). If None, the canonical CIFAR-100 list is
                    used (built into this module).

    Raises:
        ValueError: if the split file is missing, malformed, or fails any
                    of the three structural assertions.
    """
    if cifar100_classes is None:
        cifar100_classes = CIFAR100_CLASS_NAMES
    name_to_id = {name: i for i, name in enumerate(cifar100_classes)}

    path = Path(split_path) if split_path else _SPLIT_PATH
    if not path.exists():
        raise ValueError(
            f"CIFAR-FS split file not found: {path}. "
            f"Run notebooks/step4_episodic.ipynb to fetch the canonical "
            f"Bertinetto split, or restore the synthetic fallback."
        )
    with open(path) as f:
        raw = json.load(f)

    if raw.get("_status") == "synthetic_fallback":
        warnings.warn(
            "CIFAR-FS split is the SYNTHETIC FALLBACK (not Bertinetto 2019). "
            "Run notebooks/step4_episodic.ipynb's 'fetch canonical split' "
            "cell before reporting Phase 2 results.",
            UserWarning,
            stacklevel=2,
        )

    out: Dict[str, List[int]] = {}
    for split_name, expected_count in [("train", 64), ("val", 16), ("test", 20)]:
        if split_name not in raw:
            raise ValueError(f"split file is missing required key {split_name!r}")
        names = list(raw[split_name])
        if len(names) != expected_count:
            raise ValueError(
                f"split {split_name!r} has {len(names)} entries, "
                f"expected {expected_count}"
            )
        try:
            ids = sorted(name_to_id[n] for n in names)
        except KeyError as e:
            raise ValueError(
                f"split {split_name!r} contains unknown CIFAR-100 class "
                f"name {e}. Allowed names: {sorted(name_to_id)[:5]}..."
            ) from e
        out[split_name] = ids

    train_set, val_set, test_set = (set(out[k]) for k in ("train", "val", "test"))
    overlap = (train_set & val_set) | (val_set & test_set) | (train_set & test_set)
    if overlap:
        raise ValueError(
            f"CIFAR-FS splits overlap on class IDs: {sorted(overlap)}"
        )
    union = train_set | val_set | test_set
    if union != set(range(100)):
        missing = set(range(100)) - union
        raise ValueError(
            f"CIFAR-FS splits do not cover all 100 CIFAR-100 classes; "
            f"missing IDs: {sorted(missing)}"
        )
    return out


class _RelabelledCIFAR100(Dataset):
    """Wraps CIFAR-100 (full) to only yield items whose original label is in
    `keep_ids`, AND maps the kept original labels into a contiguous local
    range [0..len(keep_ids)).

    The episode sampler then sees a tidy 0..N range on the local labels.
    """

    def __init__(self, base: Dataset, keep_ids: List[int]):
        self.base = base
        # CIFAR-100 exposes either `targets` (CIFAR100) or `labels`.
        if hasattr(base, "targets"):
            self._orig_targets = list(base.targets)
        elif hasattr(base, "labels"):
            self._orig_targets = list(base.labels)
        else:
            self._orig_targets = [int(base[i][1]) for i in range(len(base))]
        keep = set(int(c) for c in keep_ids)
        self._indices = [i for i, t in enumerate(self._orig_targets) if int(t) in keep]
        self._global_to_local = {c: i for i, c in enumerate(sorted(keep))}
        # Expose `targets` so the episode sampler's _label_index path works.
        self.targets = [
            self._global_to_local[int(self._orig_targets[i])] for i in self._indices
        ]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        x, _ = self.base[self._indices[i]]
        return x, self.targets[i]


def get_cifar_fs(data_root: str = "data", image_size: int = 224,
                 split: str = "test",
                 split_path: Path | str | None = None,
                 cifar100_classes: List[str] | None = None,
                 class_ids: List[int] | None = None) -> Dataset:
    """Return a CIFAR-FS Dataset filtered to one of the Bertinetto splits.

    split: "train" | "val" | "test"
       - "train" -> 64-class split, drawn from CIFAR-100's TRAIN partition
                    (more images per class)
       - "val"   -> 16-class split, drawn from CIFAR-100's TRAIN partition
       - "test"  -> 20-class split, drawn from CIFAR-100's TEST partition
                    (matches the Step 1-3 in-distribution evaluation pool)

    class_ids: optional override. If provided, the Bertinetto split file
       is ignored and the dataset is filtered to these literal CIFAR-100
       IDs. This is the backward-compat path used by Step 1-3's exp_step1
       configs (class_ids = [0..19]).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")

    transform = _build_transform(image_size)
    use_train_partition = split in ("train", "val")
    # Pre-fetch with a browser User-Agent: the Toronto host 403s torchvision's
    # default downloader, which breaks fresh (gitignored) clones on Colab.
    from ._robust_download import ensure_archive
    ensure_archive(
        data_root, "cifar-100-python.tar.gz",
        ["https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
         "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"],
        extracted_dirname="cifar-100-python",
    )
    base = datasets.CIFAR100(
        root=data_root, train=use_train_partition,
        download=True, transform=transform,
    )

    if class_ids is not None:
        # Backwards-compat: legacy configs pass class_ids directly.
        return _RelabelledCIFAR100(base, list(class_ids))

    split_map = load_cifar_fs_split(
        split_path=split_path, cifar100_classes=cifar100_classes,
    )
    return _RelabelledCIFAR100(base, split_map[split])


def get_cifar_fs_heldout_ood(data_root: str = "data", image_size: int = 224,
                             num_samples: int = 500, seed: int = 42,
                             heldout_split: str = "val") -> torch.Tensor:
    """Zero-download near-OOD pool for the episodic evaluator.

    Samples ``num_samples`` images from the CIFAR-FS ``heldout_split`` classes
    (default "val" — the 16 val classes, disjoint from the 20 test-episode
    classes). Same visual domain as the ID data but novel classes → a clean,
    free near-OOD (OpenOOD near-OOD is "semantic shift only"). Returns
    (N, 3, image_size, image_size) ImageNet-normalized tensors, matching
    ``get_svhn_ood`` so the evaluator treats every OOD pool identically.
    """
    import random
    ds = get_cifar_fs(data_root=data_root, image_size=image_size,
                      split=heldout_split)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
    return torch.stack([ds[i][0] for i in idx])


# Canonical CIFAR-100 class names in integer-ID order.
# Source: torchvision.datasets.CIFAR100.classes (alphabetical).
# Frozen here so the loader does not require an instantiated CIFAR100
# object to do name->ID conversion.
CIFAR100_CLASS_NAMES: List[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]
assert len(CIFAR100_CLASS_NAMES) == 100
assert len(set(CIFAR100_CLASS_NAMES)) == 100
