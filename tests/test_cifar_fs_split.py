"""Step 4 / Action 4.14 — structural tests for data/cifar_fs_split.json.

These do NOT need CIFAR-100 to be downloaded; they only inspect the
class-name list and check structural invariants against the canonical
CIFAR-100 class list embedded in src/datasets/cifar_fs.py.

The split file is fetched by notebooks/step4_episodic.ipynb (Cell 10).
All tests that need the file are decorated with
    @pytest.mark.skipif(not SPLIT_PATH.exists(), reason=...)
so the pre-flight pytest (Cell 8, which runs before Cell 10) skips them
gracefully instead of failing. They run in full when the file is present.
"""
import json
from pathlib import Path

import pytest

from src.datasets import CIFAR100_CLASS_NAMES, load_cifar_fs_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data" / "cifar_fs_split.json"

_SPLIT_MISSING = not SPLIT_PATH.exists()
_SKIP_REASON   = "data/cifar_fs_split.json not yet fetched; run Cell 10 in step4_episodic.ipynb"


def _read_split_raw() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


@pytest.mark.skipif(_SPLIT_MISSING, reason=_SKIP_REASON)
def test_split_file_exists():
    assert SPLIT_PATH.exists()


@pytest.mark.skipif(_SPLIT_MISSING, reason=_SKIP_REASON)
def test_split_has_required_top_level_keys():
    raw = _read_split_raw()
    for key in ("train", "val", "test"):
        assert key in raw, f"split file missing top-level key {key!r}"


@pytest.mark.skipif(_SPLIT_MISSING, reason=_SKIP_REASON)
def test_split_sizes_64_16_20():
    raw = _read_split_raw()
    assert len(raw["train"]) == 64, f"train has {len(raw['train'])} classes, expected 64"
    assert len(raw["val"])   == 16, f"val has {len(raw['val'])} classes, expected 16"
    assert len(raw["test"])  == 20, f"test has {len(raw['test'])} classes, expected 20"


@pytest.mark.skipif(_SPLIT_MISSING, reason=_SKIP_REASON)
def test_split_names_are_known_cifar100_classes():
    raw = _read_split_raw()
    known = set(CIFAR100_CLASS_NAMES)
    for k in ("train", "val", "test"):
        for name in raw[k]:
            assert name in known, (
                f"split {k!r} contains unknown CIFAR-100 class name {name!r}. "
                f"Allowed names start with: {sorted(known)[:5]}..."
            )


@pytest.mark.skipif(_SPLIT_MISSING, reason=_SKIP_REASON)
def test_split_loader_passes_structural_assertions():
    """load_cifar_fs_split should return three disjoint lists whose union
    is exactly {0..99}, with sizes 64/16/20."""
    split = load_cifar_fs_split()
    assert set(split.keys()) == {"train", "val", "test"}
    assert len(split["train"]) == 64
    assert len(split["val"])   == 16
    assert len(split["test"])  == 20
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(100))


def test_canonical_cifar100_class_names_constant():
    """CIFAR100_CLASS_NAMES is the alphabetical list torchvision uses;
    sanity-check its length and a couple of well-known entries."""
    assert len(CIFAR100_CLASS_NAMES) == 100
    assert len(set(CIFAR100_CLASS_NAMES)) == 100
    assert CIFAR100_CLASS_NAMES[0] == "apple"
    assert CIFAR100_CLASS_NAMES[99] == "worm"
