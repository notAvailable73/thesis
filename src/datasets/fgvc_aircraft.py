"""In-distribution loader: FGVC-Aircraft (Maji et al. 2013), TMLR plan Task 5.

Same Dataset contract as mini_imagenet.py (it reuses `_MiniImageNetSplit` and
`_build_transform`), so episodes, training and evaluation need no changes.

Preprocessing (PAPER SUMMARIES/FEW SHOTpaper_summaries.txt, FGVC-Aircraft and
Meta-Dataset entries):
  - every image is cropped to its bounding box (Meta-Dataset convention; it
    also removes the 20-px copyright banner Maji et al. require removing). The
    Maji et al. note "boxes should not be used for testing" refers to the
    original 100-way benchmark; this deviation is flagged in the summary.
  - the crop is resized to 84x84 with PIL bilinear, like the MiniImageNet
    cache, and the shared transform then resizes to the model's input size.
  - the official image-level train/val/test lists are merged: few-shot splits
    partition the 100 variant CLASSES (AIRCRAFT_SPLIT, 64/16/20).

The decoded arrays live in one `aircraft_84.npz`. It is built from the raw
`fgvc-aircraft-2013b` directory (a local copy or the dataset attached to a
Kaggle notebook) and its CONTENT hash must equal AIRCRAFT_CACHE_SHA256, so
every machine trains on byte-identical images or refuses to run.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .mini_imagenet import _MiniImageNetSplit, _build_transform, _kaggle_candidate_dirs

CACHE_NAME = "aircraft_84.npz"
CACHE_SIZE = 84
_RAW_MARKER = "images_box.txt"

# Seeded 64/16/20 class split over the 100 variants: sorted variant names,
# numpy RandomState(2026).permutation, first 64 train / next 16 val / last 20
# test, each list sorted. Printed by
#   scripts/tmlr_build_aircraft_cache.py --make-split --seed 2026
# DO NOT REGENERATE — frozen (2026-10-23, before any Aircraft run).
AIRCRAFT_SPLIT: Dict[str, List[str]] = {
    "train": [
        "707-320", "737-400", "737-500", "737-800", "737-900", "747-100", "747-200",
        "747-300", "747-400", "757-200", "757-300", "767-200", "767-400", "777-200",
        "A300B4", "A310", "A318", "A319", "A340-300", "A340-500", "A340-600", "ATR-42",
        "ATR-72", "BAE-125", "Boeing 717", "C-130", "C-47", "CRJ-200", "CRJ-700",
        "CRJ-900", "Cessna 525", "Cessna 560", "DC-10", "DC-3", "DC-8", "DH-82",
        "DHC-1", "DHC-6", "DHC-8-300", "DR-400", "Dornier 328", "E-190", "E-195",
        "EMB-120", "Embraer Legacy 600", "Eurofighter Typhoon", "F-16A/B",
        "Falcon 2000", "Falcon 900", "Fokker 70", "Global Express", "Gulfstream IV",
        "Hawk T1", "L-1011", "MD-11", "MD-87", "MD-90", "Metroliner", "Model B200",
        "PA-28", "Tornado", "Tu-134", "Tu-154", "Yak-42",
    ],
    "val": [
        "737-300", "767-300", "A320", "An-12", "BAE 146-200", "BAE 146-300",
        "Beechcraft 1900", "Cessna 172", "Cessna 208", "DHC-8-100", "E-170", "ERJ 135",
        "F/A-18", "Gulfstream V", "MD-80", "Saab 2000",
    ],
    "test": [
        "727-200", "737-200", "737-600", "737-700", "777-300", "A321", "A330-200",
        "A330-300", "A340-200", "A380", "Challenger 600", "DC-6", "DC-9-30", "ERJ 145",
        "Fokker 100", "Fokker 50", "Il-76", "SR-20", "Saab 340", "Spitfire",
    ],
}
#: sha256(json.dumps(AIRCRAFT_SPLIT, sort_keys=True)).
AIRCRAFT_SPLIT_SHA256 = "6da5d5eae68e00223c89dc6beec14299581b8823ed1fb93b6f4e2bab0a067472"
#: content_sha256() of the decoded cache built from fgvc-aircraft-2013b
#: (macOS, Pillow recorded in data/aircraft_84.meta.json, 2026-10-23).
AIRCRAFT_CACHE_SHA256 = "df52796fb833c3765a5beb4da309852122ec635c33f07b1cd8bc390a1f7a27c8"

# TinyImageNet-200 classes to drop from the near-OOD pool when Aircraft is
# in-distribution. Fixed 2026-10-23 by searching TinyImageNet's 200 class names
# (wnids.txt sha256 a36eb4b6...527b, words.txt) for airliner / warplane /
# airship / space shuttle / aircraft / plane / jet / helicopter / glider /
# balloon / blimp: NO TinyImageNet class is an aircraft. The nearest class,
# n04008634 "projectile, missile", is not an aircraft and is kept as a genuine
# near-OOD class. Empty, so the TinyImageNet draw equals the CIFAR-FS one.
AIRCRAFT_TIN_EXCLUDE_WNIDS: frozenset = frozenset()


def split_sha256(split: Dict[str, List[str]]) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()


def make_split(variants: List[str], seed: int = 2026) -> Dict[str, List[str]]:
    names = sorted(variants)
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError(f"expected 100 unique variants, got {len(set(names))}")
    perm = np.random.RandomState(seed).permutation(len(names))
    pick = lambda idx: sorted(names[i] for i in idx)  # noqa: E731
    return {"train": pick(perm[:64]), "val": pick(perm[64:80]), "test": pick(perm[80:])}


# ---- raw archive -> arrays ---------------------------------------------------

def find_raw_root(data_root: str = "data") -> Optional[Path]:
    """The archive's `data/` directory (holds images_box.txt and images/),
    under data_root or anywhere in /kaggle/input."""
    # Bounded walk that neither follows symlinks (data/ links to a 120k-file
    # TinyImageNet on Kaggle) nor enters image folders.
    frontier = [Path(data_root)] if Path(data_root).is_dir() else []
    for _ in range(5):
        nxt = []
        for d in frontier:
            if (d / _RAW_MARKER).is_file() and (d / "images").is_dir():
                return d
            try:
                nxt += [Path(e.path) for e in os.scandir(d)
                        if e.is_dir(follow_symlinks=False) and e.name != "images"]
            except OSError:
                pass
        frontier = sorted(nxt)
    for d in _kaggle_candidate_dirs(max_depth=8):
        if (d / _RAW_MARKER).is_file() and (d / "images").is_dir():
            return d
    return None


def _read_pairs(path: Path) -> Dict[str, str]:
    """`<image id> <rest of line>` -> {id: rest}."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            img_id, rest = line.strip().split(" ", 1)
            out[img_id] = rest
    return out


def read_raw_index(raw: Path) -> Tuple[List[str], List[str], Dict[str, Tuple[int, int, int, int]]]:
    """(image ids sorted, variant name per id, box per id), from the official
    variant lists of all three image subsets merged."""
    variant_of: Dict[str, str] = {}
    for sub in ("train", "val", "test"):
        for img_id, v in _read_pairs(raw / f"images_variant_{sub}.txt").items():
            if variant_of.setdefault(img_id, v) != v:
                raise ValueError(f"{img_id}: two variant labels")
    boxes = {k: tuple(int(x) for x in v.split()) for k, v in _read_pairs(raw / _RAW_MARKER).items()}
    ids = sorted(variant_of)
    missing = [i for i in ids if i not in boxes]
    if missing:
        raise ValueError(f"{len(missing)} images have no bounding box, e.g. {missing[:3]}")
    return ids, [variant_of[i] for i in ids], {i: boxes[i] for i in ids}


def crop_resize(img: Image.Image, box: Tuple[int, int, int, int], size: int = CACHE_SIZE) -> np.ndarray:
    """Box is the archive's (xmin, ymin, xmax, ymax), 1-based and inclusive, so
    the PIL crop (0-based, exclusive right/bottom) is (xmin-1, ymin-1, xmax, ymax)."""
    x0, y0, x1, y1 = box
    crop = img.convert("RGB").crop((x0 - 1, y0 - 1, x1, y1))
    return np.asarray(crop.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_arrays(raw: Path, size: int = CACHE_SIZE, log=print):
    ids, variant_per_img, boxes = read_raw_index(raw)
    variants = sorted(set(variant_per_img))
    index = {v: i for i, v in enumerate(variants)}
    images = np.empty((len(ids), size, size, 3), dtype=np.uint8)
    for n, img_id in enumerate(ids):
        with Image.open(raw / "images" / f"{img_id}.jpg") as im:
            images[n] = crop_resize(im, boxes[img_id], size)
        if log and (n + 1) % 2000 == 0:
            log(f"  aircraft: decoded {n + 1}/{len(ids)}")
    labels = np.array([index[v] for v in variant_per_img], dtype=np.int16)
    return images, labels, variants


def content_sha256(images: np.ndarray, labels: np.ndarray, variants: List[str]) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(images, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.int16).tobytes())
    h.update(json.dumps(list(variants)).encode())
    return h.hexdigest()


def write_cache(path: Path, images: np.ndarray, labels: np.ndarray, variants: List[str]) -> None:
    """Atomic: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, images=images, labels=labels, variants=np.array(variants))
    os.replace(tmp, path)


# ---- cache discovery, build, load --------------------------------------------

def _find_aircraft_cache(data_root: str = "data") -> Optional[Path]:
    p = Path(data_root) / CACHE_NAME
    if p.is_file():
        return p
    for d in _kaggle_candidate_dirs():
        if (d / CACHE_NAME).is_file():
            return d / CACHE_NAME
    return None


class CacheMismatch(RuntimeError):
    pass


def ensure_aircraft_cache(data_root: str = "data", log=print) -> Path:
    """Path of a verified aircraft_84.npz, building it from the raw archive
    once if absent. Run before parallel lanes start (tmlr_run_shard.prepare_data)."""
    found = _find_aircraft_cache(data_root)
    if found is not None:
        return found
    raw = find_raw_root(data_root)
    if raw is None:
        raise FileNotFoundError(
            "FGVC-Aircraft not found: attach the Kaggle dataset "
            "seryouxblaster764/fgvc-aircraft (or put fgvc-aircraft-2013b under "
            f"{data_root}/), or provide {CACHE_NAME}.")
    log(f"aircraft: building {CACHE_NAME} from {raw}")
    images, labels, variants = build_arrays(raw, log=log)
    got = content_sha256(images, labels, variants)
    if AIRCRAFT_CACHE_SHA256 and got != AIRCRAFT_CACHE_SHA256:
        raise CacheMismatch(
            f"aircraft images built here hash to {got}, expected "
            f"{AIRCRAFT_CACHE_SHA256}: this machine decodes or resizes JPEGs "
            "differently. Refusing to train on different images.")
    out = Path(data_root) / CACHE_NAME
    write_cache(out, images, labels, variants)
    log(f"aircraft: wrote {out} ({len(labels)} images, sha {got[:12]})")
    return out


_LOADED: Dict[str, tuple] = {}


def load_aircraft(data_root: str = "data") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = ensure_aircraft_cache(data_root)
    key = str(path.resolve())
    if key not in _LOADED:
        with np.load(path) as z:
            images, labels = z["images"], z["labels"]
            variants = [str(v) for v in z["variants"]]
        got = content_sha256(images, labels, variants)
        if AIRCRAFT_CACHE_SHA256 and got != AIRCRAFT_CACHE_SHA256:
            raise CacheMismatch(f"{path} hashes to {got}, expected {AIRCRAFT_CACHE_SHA256}")
        _LOADED[key] = (images, labels, variants)
    return _LOADED[key]


def get_fgvc_aircraft(data_root: str = "data", image_size: int = 224,
                      split: str = "test") -> Dataset:
    """One AIRCRAFT_SPLIT partition: train 64 / val 16 / test 20 variants."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    if split_sha256(AIRCRAFT_SPLIT) != AIRCRAFT_SPLIT_SHA256:
        raise RuntimeError("AIRCRAFT_SPLIT does not match AIRCRAFT_SPLIT_SHA256")
    images, labels, variants = load_aircraft(data_root)
    wanted = set(AIRCRAFT_SPLIT[split])
    unknown = wanted - set(variants)
    if unknown:
        raise RuntimeError(f"split names absent from the cache: {sorted(unknown)[:3]}")
    keep = np.array([variants[int(l)] in wanted for l in labels])
    return _MiniImageNetSplit(images[keep], [variants[int(l)] for l in labels[keep]],
                              _build_transform(image_size))


def get_aircraft_heldout_ood(data_root: str = "data", image_size: int = 224,
                             num_samples: int = 500, seed: int = 42,
                             heldout_split: str = "val") -> torch.Tensor:
    """Near-OOD pool from the 16 val variants (disjoint from the 20 test
    variants); mirrors get_mini_imagenet_heldout_ood."""
    ds = get_fgvc_aircraft(data_root=data_root, image_size=image_size, split=heldout_split)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
    return torch.stack([ds[i][0] for i in idx])
