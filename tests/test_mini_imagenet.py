"""Step 9 — structural tests for the MiniImageNet split + dataset contract.

Mirrors tests/test_cifar_fs_split.py's philosophy: everything here inspects
the frozen wnid constant or a synthetic in-memory array, so it needs neither
the ~1.8 GB of real images nor a network call. The internal decoders
(_decode_pkl / _decode_csv_layout / _decode_imagefolder / the _find_* layout
detectors) take an explicit `wnids` list rather than going through
load_mini_imagenet_split, so they can be unit-tested with a handful of fake
classes without satisfying the real 64/16/20 split-size assertions.
"""
import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.mini_imagenet import (
    MINI_IMAGENET_SPLIT,
    MINI_IMAGENET_ALL_WNIDS,
    load_mini_imagenet_split,
    _MiniImageNetSplit,
    _WNID_RE,
    _load_pkl_cache,
    _decode_pkl,
    _decode_csv_layout,
    _decode_imagefolder,
    _find_zenodo_pkls,
    _find_csv_layout,
    _find_imagefolder_layout,
    _ZENODO_FILES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data" / "mini_imagenet_split.json"


# =====================================================================
# Frozen split constant
# =====================================================================
def test_frozen_split_sizes_64_16_20():
    assert len(MINI_IMAGENET_SPLIT["train"]) == 64
    assert len(MINI_IMAGENET_SPLIT["val"]) == 16
    assert len(MINI_IMAGENET_SPLIT["test"]) == 20


def test_frozen_split_wnids_are_well_formed():
    for split in ("train", "val", "test"):
        for w in MINI_IMAGENET_SPLIT[split]:
            assert _WNID_RE.match(w), f"{w!r} is not a WordNet ID (nXXXXXXXX)"


def test_frozen_split_is_disjoint_and_totals_100():
    tr, va, te = (set(MINI_IMAGENET_SPLIT[k]) for k in ("train", "val", "test"))
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert len(tr | va | te) == 100
    assert MINI_IMAGENET_ALL_WNIDS == (tr | va | te)


def test_load_mini_imagenet_split_passes_structural_assertions():
    """Falls back to the frozen constant (with a UserWarning) when
    data/mini_imagenet_split.json hasn't been materialized -- same
    synthetic-fallback convention as load_cifar_fs_split."""
    split = load_mini_imagenet_split()
    assert set(split.keys()) == {"train", "val", "test"}
    assert len(split["train"]) == 64
    assert len(split["val"]) == 16
    assert len(split["test"]) == 20
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == MINI_IMAGENET_ALL_WNIDS


def test_split_file_matches_frozen_constant_if_materialized():
    """If scripts/build_mini_imagenet_split.py has already written
    data/mini_imagenet_split.json, it must agree with the frozen constant --
    the script itself asserts this at fetch time; this test guards against
    the JSON being hand-edited afterwards."""
    if not SPLIT_PATH.exists():
        pytest.skip("data/mini_imagenet_split.json not materialized")
    with open(SPLIT_PATH) as f:
        raw = json.load(f)
    for split in ("train", "val", "test"):
        assert set(raw[split]) == set(MINI_IMAGENET_SPLIT[split])


# =====================================================================
# TinyImageNet overlap — documents the Step 9 OOD-filtering decision.
#
# Verified during Step 9 planning by diffing MiniImageNet's 100 wnids against
# TinyImageNet-200's wnid list (a public mirror of the official
# tiny-imagenet-200/wnids.txt). Frozen here as literals so a future change to
# either split's wnids fails this test loudly instead of silently changing
# what "wnid-filtered TinyImageNet" actually filters (see
# src/datasets/tinyimagenet_ood.py's exclude_wnids param and
# scripts/evaluate.py's MINI_IMAGENET_ALL_WNIDS wiring).
# =====================================================================
_TIN_OVERLAP_TEST = {"n02099601", "n02129165", "n03544143", "n04146614", "n04149813"}
_TIN_OVERLAP_VAL = {"n01855672", "n02950826", "n03584254", "n03770439",
                    "n03980874", "n09256479"}
_TIN_OVERLAP_TRAIN = {"n01910747", "n02074367", "n02165456", "n02795169",
                      "n02823428", "n03400231", "n03838899", "n03854065",
                      "n04067472", "n04251144", "n04275548", "n04596742",
                      "n07747607", "n09246464"}


def test_tinyimagenet_overlap_literals_match_the_frozen_split():
    assert _TIN_OVERLAP_TEST <= set(MINI_IMAGENET_SPLIT["test"])
    assert _TIN_OVERLAP_VAL <= set(MINI_IMAGENET_SPLIT["val"])
    assert _TIN_OVERLAP_TRAIN <= set(MINI_IMAGENET_SPLIT["train"])
    total = _TIN_OVERLAP_TEST | _TIN_OVERLAP_VAL | _TIN_OVERLAP_TRAIN
    assert len(total) == 25, "expected 25 wnids shared with TinyImageNet-200"
    assert len(_TIN_OVERLAP_TEST) == 5, (
        "expected 5 of the 20 MiniImageNet TEST classes to leak into an "
        "unfiltered TinyImageNet near-OOD pool")


# =====================================================================
# _MiniImageNetSplit contract (mirrors _RelabelledCIFAR100's in cifar_fs.py)
# =====================================================================
def _identity_transform(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def test_mini_imagenet_split_contract_contiguous_targets():
    images = np.zeros((9, 84, 84, 3), dtype=np.uint8)
    for i in range(9):
        images[i] = i * 25
    # Deliberately unsorted input labels -- output local ids must still be
    # contiguous [0, num_classes) regardless of input ordering.
    labels = ["n00000001"] * 3 + ["n00000003"] * 3 + ["n00000002"] * 3
    ds = _MiniImageNetSplit(images, labels, _identity_transform)
    assert len(ds) == 9
    assert set(ds.targets) == {0, 1, 2}
    assert ds.wnids == ["n00000001", "n00000002", "n00000003"]
    x, y = ds[0]
    assert x.shape == (3, 84, 84)
    assert isinstance(y, int)


def test_mini_imagenet_split_works_with_sample_episode():
    from src.datasets import sample_episode
    n_per_class = 25
    wnids = [f"n0000000{i}" for i in range(5)]
    labels = [w for w in wnids for _ in range(n_per_class)]
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(len(labels), 84, 84, 3), dtype=np.uint8)
    ds = _MiniImageNetSplit(images, labels, _identity_transform)
    sx, sy, qx, qy = sample_episode(
        dataset=ds, class_ids=None, n_way=5, k_shot=5, q_query=15, seed=0)
    assert sx.shape == (25, 3, 84, 84)
    assert qx.shape == (75, 3, 84, 84)
    assert sy.min().item() == 0 and sy.max().item() == 4


# =====================================================================
# Source-layout decoders (each takes explicit wnids -- no 64/16/20
# constraint, so a handful of fake classes is enough).
# =====================================================================
_FAKE_WNIDS = ["n00000001", "n00000002", "n00000003"]


def _write_fake_image(path: Path, fill: int):
    img = Image.new("RGB", (10, 10), (fill, fill, fill))
    img.save(path)


def test_decode_imagefolder(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for i, w in enumerate(_FAKE_WNIDS):
        wdir = root / w
        wdir.mkdir()
        for j in range(2):
            _write_fake_image(wdir / f"{j}.jpg", fill=(i * 50 + j))
    imgs, labels = _decode_imagefolder(root, _FAKE_WNIDS)
    assert imgs.shape == (6, 84, 84, 3)
    assert imgs.dtype == np.uint8
    assert labels == [w for w in sorted(_FAKE_WNIDS) for _ in range(2)]


def test_decode_imagefolder_missing_class_dir_raises(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / _FAKE_WNIDS[0]).mkdir()
    _write_fake_image(root / _FAKE_WNIDS[0] / "0.jpg", fill=10)
    with pytest.raises(RuntimeError, match="not found"):
        _decode_imagefolder(root, _FAKE_WNIDS)


def test_decode_csv_layout(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    rows = []
    for i, w in enumerate(_FAKE_WNIDS):
        for j in range(2):
            fname = f"{w}{j:08d}.jpg"
            _write_fake_image(root / "images" / fname, fill=(i * 50 + j))
            rows.append((fname, w))
    with open(root / "train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)
    imgs, labels = _decode_csv_layout(root, "train", _FAKE_WNIDS)
    assert imgs.shape == (6, 84, 84, 3)
    assert set(labels) == set(_FAKE_WNIDS)


def test_decode_csv_layout_no_matching_rows_raises(tmp_path):
    root = tmp_path
    (root / "images").mkdir()
    with open(root / "train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerow(["x.jpg", "nUNRELATED"])
    with pytest.raises(RuntimeError, match="no rows matched"):
        _decode_csv_layout(root, "train", _FAKE_WNIDS)


def _make_fake_pkl(path: Path, wnids, n_per_class=2):
    n = len(wnids) * n_per_class
    image_data = np.zeros((n, 84, 84, 3), dtype=np.uint8)
    class_dict = {}
    idx = 0
    for w in wnids:
        idxs = []
        for _ in range(n_per_class):
            image_data[idx] = idx
            idxs.append(idx)
            idx += 1
        class_dict[w] = idxs
    with open(path, "wb") as f:
        pickle.dump({"image_data": image_data, "class_dict": class_dict}, f)


def test_decode_pkl_valid_schema(tmp_path):
    p = tmp_path / "fake.pkl"
    _make_fake_pkl(p, _FAKE_WNIDS, n_per_class=2)
    imgs, labels = _decode_pkl(p, _FAKE_WNIDS)
    assert imgs.shape == (6, 84, 84, 3)
    assert labels == [w for w in sorted(_FAKE_WNIDS) for _ in range(2)]


def test_load_pkl_cache_rejects_unknown_schema(tmp_path):
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as f:
        pickle.dump(["not", "the", "right", "schema"], f)
    with pytest.raises(RuntimeError, match="unrecognized"):
        _load_pkl_cache(p)


def test_decode_pkl_rejects_wnid_mismatch(tmp_path):
    p = tmp_path / "fake.pkl"
    _make_fake_pkl(p, _FAKE_WNIDS, n_per_class=2)
    with pytest.raises(RuntimeError, match="do not match"):
        _decode_pkl(p, ["n00000001", "n00000002", "nDIFFERENT"])


# =====================================================================
# Source-layout discovery
# =====================================================================
def test_find_zenodo_pkls_staged(tmp_path):
    for fname, _size, _md5 in _ZENODO_FILES.values():
        (tmp_path / fname).write_bytes(b"x")
    found = _find_zenodo_pkls(str(tmp_path))
    assert found is not None
    assert set(found.keys()) == {"train", "val", "test"}


def test_find_zenodo_pkls_absent_returns_none(tmp_path):
    assert _find_zenodo_pkls(str(tmp_path)) is None


def test_find_csv_layout_staged(tmp_path):
    (tmp_path / "images").mkdir()
    for s in ("train", "val", "test"):
        (tmp_path / f"{s}.csv").write_text("filename,label\n")
    assert _find_csv_layout(str(tmp_path)) == tmp_path


def test_find_csv_layout_absent_returns_none(tmp_path):
    assert _find_csv_layout(str(tmp_path)) is None


def test_find_imagefolder_layout_staged(tmp_path):
    for i in range(95):
        (tmp_path / f"n{i:08d}").mkdir()
    assert _find_imagefolder_layout(str(tmp_path)) == tmp_path


def test_find_imagefolder_layout_too_few_class_dirs_returns_none(tmp_path):
    for i in range(3):
        (tmp_path / f"n{i:08d}").mkdir()
    assert _find_imagefolder_layout(str(tmp_path)) is None
