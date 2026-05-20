"""Step 4 / Action 4.14 — structural tests for data/cifar_fs_split.json.

These do NOT need CIFAR-100 to be downloaded; they only inspect the
class-name list and check structural invariants against the canonical
CIFAR-100 class list embedded in src/datasets/cifar_fs.py.

The split itself is canonical Bertinetto when the notebook has fetched
it, and the synthetic fallback otherwise. Both versions pass the
structural tests below — that's the point of having a fallback at all.
"""
import json
from pathlib import Path

import pytest

from src.datasets import CIFAR100_CLASS_NAMES, load_cifar_fs_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data" / "cifar_fs_split.json"


def _read_split_raw() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def test_split_file_exists():
    assert SPLIT_PATH.exists(), (
        f"data/cifar_fs_split.json is missing. Run the fetch cell in "
        f"notebooks/step4_episodic.ipynb to create it."
    )


def test_split_has_required_top_level_keys():
    raw = _read_split_raw()
    for key in ("train", "val", "test"):
        assert key in raw, f"split file missing top-level key {key!r}"


def test_split_sizes_64_16_20():
    raw = _read_split_raw()
    assert len(raw["train"]) == 64, f"train has {len(raw['train'])} classes, expected 64"
    assert len(raw["val"])   == 16, f"val has {len(raw['val'])} classes, expected 16"
    assert len(raw["test"])  == 20, f"test has {len(raw['test'])} classes, expected 20"


def test_split_names_are_known_cifar100_classes():
    raw = _read_split_raw()
    known = set(CIFAR100_CLASS_NAMES)
    for k in ("train", "val", "test"):
        for name in raw[k]:
            assert name in known, (
                f"split {k!r} contains unknown CIFAR-100 class name {name!r}. "
                f"Allowed names start with: {sorted(known)[:5]}..."
            )


def test_split_loader_passes_structural_assertions():
    """load_cifar_fs_split should return three disjoint lists whose union
    is exactly {0..99}, with sizes 64/16/20."""
    split = load_cifar_fs_split()
    assert set(split.keys()) == {"train", "val", "test"}
    assert len(split["train"]) == 64
    assert len(split["val"])   == 16
    assert len(split["test"])  == 20
    # Disjoint.
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Cover-all.
    assert train | val | test == set(range(100))


def test_canonical_cifar100_class_names_constant():
    """CIFAR100_CLASS_NAMES is the alphabetical list torchvision uses;
    sanity-check its length and a couple of well-known entries."""
    assert len(CIFAR100_CLASS_NAMES) == 100
    assert len(set(CIFAR100_CLASS_NAMES)) == 100
    assert CIFAR100_CLASS_NAMES[0] == "apple"
    assert CIFAR100_CLASS_NAMES[99] == "worm"
