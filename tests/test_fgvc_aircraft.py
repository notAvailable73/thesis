"""TMLR plan Task 5 — FGVC-Aircraft loader, cache build and dispatch.
Synthetic fixtures only (a fake fgvc-aircraft-2013b layout); never real data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.datasets.fgvc_aircraft as A
from src.datasets import (EpisodicIterableDataset, get_heldout_near_ood, get_id_split,
                          tin_exclusions, MINI_IMAGENET_ALL_WNIDS)


def test_split_frozen():
    s = A.AIRCRAFT_SPLIT
    assert [len(s[k]) for k in ("train", "val", "test")] == [64, 16, 20]
    allv = s["train"] + s["val"] + s["test"]
    assert len(set(allv)) == 100
    assert all(s[k] == sorted(s[k]) for k in s)


def test_split_hash():
    assert hashlib.sha256(json.dumps(A.AIRCRAFT_SPLIT, sort_keys=True).encode()).hexdigest() \
        == A.AIRCRAFT_SPLIT_SHA256


def test_make_split_reproduces_frozen():
    allv = A.AIRCRAFT_SPLIT["train"] + A.AIRCRAFT_SPLIT["val"] + A.AIRCRAFT_SPLIT["test"]
    assert A.make_split(list(reversed(allv)), 2026) == A.AIRCRAFT_SPLIT
    assert A.make_split(allv, 2027) != A.AIRCRAFT_SPLIT


def _variants():
    return sorted(A.AIRCRAFT_SPLIT["train"] + A.AIRCRAFT_SPLIT["val"] + A.AIRCRAFT_SPLIT["test"])


def _fake_raw(root: Path, per_class: int = 3) -> Path:
    """fgvc-aircraft-2013b/data with per_class images per variant, 60x50 px with
    a 10-px white 'banner' at the bottom that the box excludes."""
    raw = root / "aircraft_raw" / "fgvc-aircraft-2013b" / "data"
    (raw / "images").mkdir(parents=True)
    lines = {"train": [], "val": [], "test": []}
    boxes = []
    n = 0
    for v in _variants():
        for j in range(per_class):
            img_id = f"{1000000 + n:07d}"
            arr = np.full((50, 60, 3), (n * 7) % 200, dtype=np.uint8)
            arr[40:] = 255
            Image.fromarray(arr).save(raw / "images" / f"{img_id}.jpg", quality=95)
            lines[("train", "val", "test")[j % 3]].append(f"{img_id} {v}")
            boxes.append(f"{img_id} 1 1 60 40")
            n += 1
    for sub, ls in lines.items():
        (raw / f"images_variant_{sub}.txt").write_text("\n".join(ls) + "\n")
    (raw / "images_box.txt").write_text("\n".join(boxes) + "\n")
    return raw


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "AIRCRAFT_CACHE_SHA256", "")
    monkeypatch.setattr(A, "_LOADED", {})
    monkeypatch.setattr(A, "_kaggle_candidate_dirs", lambda max_depth=6: [])
    _fake_raw(tmp_path)
    return tmp_path


def test_crop_removes_banner():
    arr = np.zeros((50, 60, 3), dtype=np.uint8)
    arr[40:] = 255
    out = A.crop_resize(Image.fromarray(arr), (1, 1, 60, 40))
    assert out.shape == (84, 84, 3) and out.dtype == np.uint8 and out.max() == 0


def test_find_raw_and_build(raw_root):
    raw = A.find_raw_root(str(raw_root))
    assert raw is not None and raw.name == "data"
    images, labels, variants = A.build_arrays(raw, log=None)
    assert images.shape == (300, 84, 84, 3) and len(variants) == 100
    assert np.bincount(labels).tolist() == [3] * 100


def test_cache_build_once_and_hash_guard(raw_root, monkeypatch):
    p = A.ensure_aircraft_cache(str(raw_root), log=lambda *a: None)
    assert p == raw_root / A.CACHE_NAME and p.exists()
    images, labels, variants = A.load_aircraft(str(raw_root))
    good = A.content_sha256(images, labels, variants)
    monkeypatch.setattr(A, "AIRCRAFT_CACHE_SHA256", good)
    monkeypatch.setattr(A, "_LOADED", {})
    A.load_aircraft(str(raw_root))                       # verified load passes
    p.unlink()
    monkeypatch.setattr(A, "AIRCRAFT_CACHE_SHA256", "0" * 64)
    with pytest.raises(A.CacheMismatch):                 # a differently-decoded build refuses
        A.ensure_aircraft_cache(str(raw_root), log=lambda *a: None)
    assert not p.exists()


def test_loader_on_fixture(raw_root):
    ds = get_id_split({"name": "fgvc_aircraft", "data_root": str(raw_root)}, "val")
    assert len(ds) == 48 and set(ds.targets) == set(range(16))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and 0 <= y < 16
    assert sorted(ds.wnids) == A.AIRCRAFT_SPLIT["val"]


def test_episodes_on_fixture(raw_root):
    ds = get_id_split({"name": "fgvc_aircraft", "data_root": str(raw_root), "image_size": 32},
                      "test")
    it = iter(EpisodicIterableDataset(ds, n_way=5, k_shot=1, q_query=2,
                                      num_episodes=2))
    ep = next(it)
    assert ep[0].shape[0] == 5 and ep[2].shape[0] == 10


def test_dispatch_near_pool(raw_root):
    name, x = get_heldout_near_ood({"name": "fgvc_aircraft", "data_root": str(raw_root),
                                    "image_size": 32}, num_samples=20, seed=1)
    assert name == "aircraft_near" and x.shape == (20, 3, 32, 32)


def test_tin_exclusions():
    assert tin_exclusions("mini_imagenet") is MINI_IMAGENET_ALL_WNIDS
    assert tin_exclusions("cifar_fs") is None
    # No TinyImageNet class is an aircraft: empty, so the draw equals CIFAR-FS's.
    assert tin_exclusions("fgvc_aircraft") == frozenset() and not tin_exclusions("fgvc_aircraft")
