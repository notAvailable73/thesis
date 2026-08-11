"""In-distribution loader: MiniImageNet (Ravi & Larochelle 2017 64/16/20 split).

Mirrors src/datasets/cifar_fs.py's structure and the same Dataset contract
(`.targets` = contiguous local ints, `__getitem__ -> (float tensor (3,H,W),
int)`, `__len__`) so the dataset-agnostic episode samplers
(src/datasets/episode_sampler.py) need zero changes.

The key structural difference from CIFAR-FS: CIFAR-100 is one shared
torchvision Dataset filtered by class id, so CIFAR-FS's train/val/test all
draw from the same underlying array. MiniImageNet has no such single upstream
object — every distribution source (the learn2learn Zenodo pkl caches, a
Kaggle-hosted images/+csv mirror, a Kaggle-hosted <wnid>/*.jpg mirror) is
already split into train/val/test parts, so there is no "class_ids override"
path equivalent to CIFAR-FS's `class_ids=` (Step 1-3 backward-compat) — Step 9
only ever needs Bertinetto-style whole-split routing.

Images are decoded once into a per-split uint8 (N, 84, 84, 3) memmap-able
cache (`data/mini_imagenet_84_<split>.npy` + a JSON sidecar), matching the
"never repeatedly decode a many-file archive at __getitem__ time" rule that
motivates tinyimagenet_ood.py's zip-read pattern.

DATA SOURCE — verified live during Step 9 planning, not assumed:
    Zenodo record 7978538 ("learn2learn: Few-Shot Learning Datasets") serves
    mini-imagenet-cache-{train,validation,test}.pkl over plain HTTPS with
    published MD5s (no auth). The pkl schema this module assumes
    (`{"image_data": uint8 (N,84,84,3), "class_dict": {wnid: [row indices]}}`)
    is the de-facto community convention (used by the MAML++ / learn2learn /
    many Kaggle re-hosts of this exact release) but is NOT stated in any of
    this project's paper_summaries.txt files -- it is a data-engineering fact,
    not a research claim, so thesis_implementation_instructions.txt's
    paper-grounding process does not apply to it. The loader validates the
    schema at load time and raises with the observed keys/shape rather than
    silently guessing if a staged file doesn't match.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_PATH = _REPO_ROOT / "data" / "mini_imagenet_split.json"

_WNID_RE = re.compile(r"^n\d{8}$")

# Zenodo record 7978538 ("learn2learn: Few-Shot Learning Datasets"). Sizes and
# MD5s below were read from the record's own file listing (its `files[].size`
# / `files[].checksum` fields), not assumed.
_ZENODO_BASE = "https://zenodo.org/records/7978538/files"
_ZENODO_FILES: Dict[str, Tuple[str, int, str]] = {
    # split -> (filename, size_bytes, md5)
    "train": ("mini-imagenet-cache-train.pkl", 1145461190,
              "ee4e80dc7b716f3ceec891d199b5277a"),
    "val": ("mini-imagenet-cache-validation.pkl", 292661258,
            "c1145a80890b88d54a67d6e0022d53a5"),
    "test": ("mini-imagenet-cache-test.pkl", 353647539,
              "389b6c59d348b2e1ae9610f00a5e6724"),
}
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# Ravi & Larochelle (2017) canonical 64/16/20 split, as WordNet IDs.
# Source: github.com/twitter/meta-learning-lstm data/miniImagenet/{train,val,
# test}.csv (the canonical release location for this exact split; also the
# source scripts/build_mini_imagenet_split.py fetches). Frozen here, sorted
# alphabetically per split, so structural tests need no network access and
# scripts/build_mini_imagenet_split.py has something to validate its fetch
# against (mirrors CIFAR100_CLASS_NAMES in cifar_fs.py serving the same role).
MINI_IMAGENET_SPLIT: Dict[str, List[str]] = {
    "train": [
        "n01532829", "n01558993", "n01704323", "n01749939", "n01770081",
        "n01843383", "n01910747", "n02074367", "n02089867", "n02091831",
        "n02101006", "n02105505", "n02108089", "n02108551", "n02108915",
        "n02111277", "n02113712", "n02120079", "n02165456", "n02457408",
        "n02606052", "n02687172", "n02747177", "n02795169", "n02823428",
        "n02966193", "n03017168", "n03047690", "n03062245", "n03207743",
        "n03220513", "n03337140", "n03347037", "n03400231", "n03476684",
        "n03527444", "n03676483", "n03838899", "n03854065", "n03888605",
        "n03908618", "n03924679", "n03998194", "n04067472", "n04243546",
        "n04251144", "n04258138", "n04275548", "n04296562", "n04389033",
        "n04435653", "n04443257", "n04509417", "n04515003", "n04596742",
        "n04604644", "n04612504", "n06794110", "n07584110", "n07697537",
        "n07747607", "n09246464", "n13054560", "n13133613",
    ],
    "val": [
        "n01855672", "n02091244", "n02114548", "n02138441", "n02174001",
        "n02950826", "n02971356", "n02981792", "n03075370", "n03417042",
        "n03535780", "n03584254", "n03770439", "n03773504", "n03980874",
        "n09256479",
    ],
    "test": [
        "n01930112", "n01981276", "n02099601", "n02110063", "n02110341",
        "n02116738", "n02129165", "n02219486", "n02443484", "n02871525",
        "n03127925", "n03146219", "n03272010", "n03544143", "n03775546",
        "n04146614", "n04149813", "n04418357", "n04522168", "n07613480",
    ],
}
assert len(MINI_IMAGENET_SPLIT["train"]) == 64
assert len(MINI_IMAGENET_SPLIT["val"]) == 16
assert len(MINI_IMAGENET_SPLIT["test"]) == 20

# All 100 wnids across every partition. Used by evaluate.py to filter the
# TinyImageNet near-OOD pool: TinyImageNet-200 shares 25 wnids with
# MiniImageNet (5 of which are MiniImageNet TEST classes), which would
# otherwise let real in-distribution classes leak into an "OOD" pool.
MINI_IMAGENET_ALL_WNIDS = frozenset(
    MINI_IMAGENET_SPLIT["train"]) | frozenset(
    MINI_IMAGENET_SPLIT["val"]) | frozenset(MINI_IMAGENET_SPLIT["test"])
assert len(MINI_IMAGENET_ALL_WNIDS) == 100, \
    "MINI_IMAGENET_SPLIT wnids are not 100 disjoint IDs"


def load_mini_imagenet_split(split_path: Path | str | None = None,
                             ) -> Dict[str, List[str]]:
    """Return {"train": [...64 wnids], "val": [...16], "test": [...20]},
    validated disjoint with union size 100.

    Reads data/mini_imagenet_split.json if present (written by
    scripts/build_mini_imagenet_split.py); falls back to the frozen
    MINI_IMAGENET_SPLIT constant above (with a UserWarning) if the file is
    absent, mirroring load_cifar_fs_split's synthetic-fallback behaviour.
    """
    path = Path(split_path) if split_path else _SPLIT_PATH
    if not path.exists():
        warnings.warn(
            "data/mini_imagenet_split.json not found; falling back to the "
            "frozen MINI_IMAGENET_SPLIT constant embedded in "
            "src/datasets/mini_imagenet.py. Run "
            "scripts/build_mini_imagenet_split.py to materialize the file "
            "(fetches the canonical CSVs and validates them against this "
            "same constant).",
            UserWarning, stacklevel=2,
        )
        raw: dict = {
            "train": MINI_IMAGENET_SPLIT["train"],
            "val": MINI_IMAGENET_SPLIT["val"],
            "test": MINI_IMAGENET_SPLIT["test"],
        }
    else:
        with open(path) as f:
            raw = json.load(f)
        if raw.get("_status") == "synthetic_fallback":
            warnings.warn(
                "MiniImageNet split is the SYNTHETIC FALLBACK (not Ravi & "
                "Larochelle 2017) -- results are not comparable to any "
                "canonical MiniImageNet benchmark.",
                UserWarning, stacklevel=2,
            )

    out: Dict[str, List[int]] = {}
    for split_name, expected_count in [("train", 64), ("val", 16), ("test", 20)]:
        if split_name not in raw:
            raise ValueError(f"split file is missing required key {split_name!r}")
        wnids = list(raw[split_name])
        if len(wnids) != expected_count:
            raise ValueError(
                f"split {split_name!r} has {len(wnids)} entries, "
                f"expected {expected_count}")
        bad = [w for w in wnids if not _WNID_RE.match(w)]
        if bad:
            raise ValueError(
                f"split {split_name!r} contains non-WordNet-ID entries: {bad[:5]}")
        out[split_name] = sorted(wnids)

    train_set, val_set, test_set = (set(out[k]) for k in ("train", "val", "test"))
    overlap = (train_set & val_set) | (val_set & test_set) | (train_set & test_set)
    if overlap:
        raise ValueError(f"MiniImageNet splits overlap on wnids: {sorted(overlap)}")
    if len(train_set | val_set | test_set) != 100:
        raise ValueError(
            "MiniImageNet splits do not total 100 disjoint wnids: got "
            f"{len(train_set | val_set | test_set)}")
    return out


# =====================================================================
# Source-layout discovery. Nothing is assumed staged; each finder returns
# None if its layout isn't present, in the priority order _build_split_cache
# tries them. Kaggle attaches datasets under /kaggle/input -- searched in
# addition to data_root (same convention as tinyimagenet_ood.py's Colab/Drive
# note). /kaggle/input is much faster per-file than a Drive FUSE mount, but
# not free: it is still a fused remote mount, so a scan of it is a per-entry
# cost, paid again in every one of the ~20 train.py/evaluate.py subprocesses
# a Step 9 run spawns. Each finder therefore checks data_root FIRST and only
# falls through to /kaggle/input when something is genuinely missing, and the
# /kaggle/input walk itself is depth-bounded and prunes image directories
# (see _kaggle_candidate_dirs). Per-image DECODING is still the expensive
# part, and that happens exactly once per split (cached to .npy afterwards).
# =====================================================================
_KAGGLE_MAX_DEPTH = 6


def _kaggle_candidate_dirs(max_depth: int = _KAGGLE_MAX_DEPTH) -> List[Path]:
    """Every plausible dataset-root directory under /kaggle/input.

    Was one level deep only (root + its immediate children), which matched
    the classic mount layout (/kaggle/input/<slug>/...). Step 9 found a
    newer Kaggle notebook environment nests one level deeper still,
    grouping by source type first (/kaggle/input/datasets/<owner>/<slug>/,
    verified live) -- a fixed depth silently misses every manually-attached
    dataset there, so this walks rather than assuming an exact depth.

    Two bounds keep that walk from degenerating, both learned the hard way
    on Kaggle (the previous implementation was a bare
    `glob('/kaggle/input/**', recursive=True)`, which enumerates every FILE
    under the mount and then stats each one):
      - `max_depth` -- a dataset root nests at most ~5 levels below
        /kaggle/input even on the deepest mount layout observed.
      - pruning -- never descend into a `<wnid>/` or `images/` directory.
        Those hold the images themselves, never a nested dataset root, and
        they are exactly where the file counts explode (an attached
        TinyImageNet is ~120k files under `train/<wnid>/images/` and
        `val/images/`). Their PARENTS are still returned, so the
        `<wnid>/*.jpg` and `images/`+csv layouts below stay discoverable.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    out = [root]
    frontier = [root]
    for _ in range(max_depth):
        nxt: List[Path] = []
        for d in frontier:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if e.name == "images" or _WNID_RE.match(e.name):
                    continue
                try:
                    if e.is_dir():
                        nxt.append(Path(e.path))
                except OSError:
                    continue
        if not nxt:
            break
        out.extend(nxt)
        frontier = nxt
    return out


def _find_zenodo_pkls(data_root: str) -> Optional[Dict[str, Path]]:
    """Whichever of the 3 pkl caches are staged, either dropped directly in
    data_root or inside an attached Kaggle dataset (searched at any depth --
    see `_kaggle_candidate_dirs`). May return a PARTIAL dict: a caller only
    ever needs the one split it's about to load (`_build_split_cache` checks
    `split in pkls`), so a dataset that only ships e.g. the test cache (all
    an eval-only run needs) must not force a full ~1.8GB re-download of the
    other two splits. Returns None only if nothing at all was found.

    data_root is checked FIRST and, when it already holds all three, the
    /kaggle/input walk is skipped entirely. Same precedence as before
    (data_root wins per split), but a notebook that stages the pkls into
    data/ no longer pays a mount-wide directory scan in every one of the 20
    train/evaluate subprocesses a Step 9 run spawns."""
    found: Dict[str, Path] = {}
    root = Path(data_root)
    for split, (fname, _, _) in _ZENODO_FILES.items():
        if (root / fname).is_file():
            found[split] = root / fname
    if len(found) == len(_ZENODO_FILES):
        return found
    for d in _kaggle_candidate_dirs():
        for split, (fname, _, _) in _ZENODO_FILES.items():
            if split in found:
                continue
            cand = d / fname
            if cand.is_file():
                found[split] = cand
        if len(found) == len(_ZENODO_FILES):
            break
    return found if found else None


def _find_csv_layout(data_root: str) -> Optional[Path]:
    """Ravi & Larochelle's own release shape: one `images/` dir of flat
    JPEGs + train.csv/val.csv/test.csv naming which wnid each file belongs
    to (the exact shape scripts/build_mini_imagenet_split.py's source CSVs
    describe)."""
    def _is_csv_layout(d: Path) -> bool:
        return ((d / "images").is_dir()
                and all((d / f"{s}.csv").exists()
                        for s in ("train", "val", "test")))

    root = Path(data_root)
    if root.is_dir() and _is_csv_layout(root):
        return root
    for d in _kaggle_candidate_dirs():
        if _is_csv_layout(d):
            return d
    return None


def _find_imagefolder_layout(data_root: str,
                             min_classes: int = 90) -> Optional[Path]:
    """A directory whose immediate children are (mostly) MiniImageNet
    wnid-named dirs of images -- the common shape for third-party Kaggle
    re-hosts.

    The match is against MINI_IMAGENET_ALL_WNIDS specifically, not against
    "looks like a wnid" (`_WNID_RE`). A wnid-shaped test was wrong in one
    concrete, live case: an attached TinyImageNet-200 has 200 wnid-named
    dirs under `train/`, of which only 25 are MiniImageNet classes, so it
    matched here and was handed to `_decode_imagefolder` as a MiniImageNet
    mirror. That raised a confusing "expected class dir ... not found"
    instead of the real problem (a MiniImageNet pkl was missing), which is
    what this finder falling through to None reports.
    """
    def _covers_split(d: Path) -> bool:
        try:
            children = {c.name for c in d.iterdir() if c.is_dir()}
        except OSError:
            return False
        return len(children & MINI_IMAGENET_ALL_WNIDS) >= min_classes

    root = Path(data_root)
    if root.is_dir() and _covers_split(root):
        return root
    for d in _kaggle_candidate_dirs():
        if _covers_split(d):
            return d
    return None


# =====================================================================
# Per-layout decoders. Each returns (images uint8 (N,84,84,3), labels: List[
# str] wnid per row), in a DETERMINISTIC order (wnid ascending, then
# within-class ascending) so the cache is reproducible regardless of
# filesystem iteration order.
# =====================================================================
def _load_pkl_cache(path: Path) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, dict) and "image_data" in obj and "class_dict" in obj):
        raise RuntimeError(
            f"{path}: unrecognized MiniImageNet cache format. Expected a dict "
            f"with keys 'image_data' (uint8 array) and 'class_dict' "
            f"(wnid -> row indices) -- the standard learn2learn/MAML++ cache "
            f"layout. Got type={type(obj).__name__}"
            + (f", keys={list(obj.keys())[:10]}" if isinstance(obj, dict) else "")
            + ". Inspect the file manually; do not assume the schema."
        )
    image_data = np.asarray(obj["image_data"])
    if image_data.ndim != 4 or image_data.shape[1:3] != (84, 84) or \
            image_data.dtype != np.uint8:
        raise RuntimeError(
            f"{path}: image_data has unexpected shape/dtype "
            f"{image_data.shape}/{image_data.dtype}; expected (N,84,84,3) uint8."
        )
    class_dict = {str(k): [int(i) for i in v] for k, v in obj["class_dict"].items()}
    return image_data, class_dict


def _decode_pkl(path: Path, wnids: List[str]) -> Tuple[np.ndarray, List[str]]:
    image_data, class_dict = _load_pkl_cache(path)
    found = set(class_dict.keys())
    expected = set(wnids)
    if found != expected:
        raise RuntimeError(
            f"{path}: class_dict wnids do not match the frozen Ravi & "
            f"Larochelle split for this partition. "
            f"missing={sorted(expected - found)[:5]} "
            f"unexpected={sorted(found - expected)[:5]}"
        )
    imgs, labels = [], []
    for w in sorted(wnids):
        for i in sorted(class_dict[w]):
            imgs.append(image_data[i])
            labels.append(w)
    return np.stack(imgs).astype(np.uint8), labels


def _decode_csv_layout(root: Path, split_name: str,
                       wnids: List[str]) -> Tuple[np.ndarray, List[str]]:
    rows: List[Tuple[str, str]] = []
    with open(root / f"{split_name}.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], r["label"]))
    keep = set(wnids)
    rows = [(fn, w) for fn, w in rows if w in keep]
    if not rows:
        raise RuntimeError(
            f"{root / (split_name + '.csv')}: no rows matched the frozen "
            f"{split_name} wnid list -- wrong CSV, or a different split "
            f"variant than Ravi & Larochelle's canonical release."
        )
    rows.sort()  # deterministic: (filename, wnid) ascending
    imgs, labels = [], []
    for fname, wnid in rows:
        img = Image.open(root / "images" / fname).convert("RGB")
        img = img.resize((84, 84), Image.BILINEAR)
        imgs.append(np.asarray(img, dtype=np.uint8))
        labels.append(wnid)
    return np.stack(imgs), labels


def _decode_imagefolder(root: Path, wnids: List[str]) -> Tuple[np.ndarray, List[str]]:
    imgs, labels = [], []
    for w in sorted(wnids):
        wdir = root / w
        if not wdir.is_dir():
            raise RuntimeError(
                f"expected class dir {wdir} not found under {root} -- the "
                f"attached dataset may not cover the full Ravi & Larochelle "
                f"split.")
        files = sorted(p for p in wdir.iterdir() if p.suffix.lower() in _IMG_EXTS)
        if not files:
            raise RuntimeError(f"class dir {wdir} has no images")
        for p in files:
            img = Image.open(p).convert("RGB").resize((84, 84), Image.BILINEAR)
            imgs.append(np.asarray(img, dtype=np.uint8))
            labels.append(w)
    return np.stack(imgs), labels


def _build_split_cache(data_root: str, split: str,
                       wnids: List[str]) -> Tuple[np.ndarray, List[str], str]:
    """Resolve whichever source layout is staged (checked in the order docs
    above list), decode it, and return (images, labels, source_layout) for
    ONE split. Falls back to downloading the Zenodo pkls if nothing is
    staged."""
    pkls = _find_zenodo_pkls(data_root)
    if pkls is not None and split in pkls:
        imgs, labels = _decode_pkl(pkls[split], wnids)
        return imgs, labels, "staged_zenodo_pkl"

    csv_root = _find_csv_layout(data_root)
    if csv_root is not None:
        imgs, labels = _decode_csv_layout(csv_root, split, wnids)
        return imgs, labels, "staged_images_plus_csv"

    folder_root = _find_imagefolder_layout(data_root)
    if folder_root is not None:
        imgs, labels = _decode_imagefolder(folder_root, wnids)
        return imgs, labels, "staged_wnid_imagefolder"

    from ._robust_download import ensure_archive
    for s, (fname, _, md5) in _ZENODO_FILES.items():
        # user_agent=None: Zenodo's WAF 403s ensure_archive's default browser
        # UA (verified live) while allowing urllib's own default UA through --
        # the opposite of CIFAR's cs.toronto.edu, which is what that default
        # exists for. See _robust_download.py's module docstring.
        ensure_archive(data_root, fname, [f"{_ZENODO_BASE}/{fname}"], md5=md5,
                       user_agent=None)
    pkl_path = Path(data_root) / _ZENODO_FILES[split][0]
    imgs, labels = _decode_pkl(pkl_path, wnids)
    return imgs, labels, "downloaded_zenodo_pkl"


def _cache_paths(data_root: str, split: str) -> Tuple[Path, Path]:
    base = os.path.join(data_root, f"mini_imagenet_84_{split}")
    return Path(base + ".npy"), Path(base + ".json")


def _ensure_cache(data_root: str, split: str,
                  split_map: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Return (images uint8 memmap (N,84,84,3), labels: List[str] wnid per
    row) for `split`, building + writing the cache once if absent or stale."""
    npy_path, json_path = _cache_paths(data_root, split)
    wnids = split_map[split]

    if npy_path.exists() and json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        if sorted(meta.get("wnids", [])) == sorted(wnids):
            return np.load(npy_path, mmap_mode="r"), list(meta["labels"])
        warnings.warn(
            f"{json_path} does not match the requested split's wnids; "
            f"rebuilding the {split} cache.",
            UserWarning, stacklevel=2,
        )

    images, labels, source_layout = _build_split_cache(data_root, split, wnids)
    os.makedirs(data_root, exist_ok=True)
    np.save(npy_path, images)
    sha256 = hashlib.sha256(images.tobytes()).hexdigest()
    with open(json_path, "w") as f:
        json.dump({
            "split": split,
            "wnids": sorted(wnids),
            "labels": labels,
            "num_images": len(labels),
            "source_layout": source_layout,
            "sha256": sha256,
        }, f, indent=2)
    return np.load(npy_path, mmap_mode="r"), labels


class _MiniImageNetSplit(Dataset):
    """Wraps a decoded (images, labels) pair with contiguous local integer
    targets, mirroring _RelabelledCIFAR100's contract in cifar_fs.py."""

    def __init__(self, images: np.ndarray, labels: List[str], transform):
        assert images.shape[0] == len(labels)
        self._images = images
        unique = sorted(set(labels))
        self._wnid_to_local = {w: i for i, w in enumerate(unique)}
        self.wnids = unique
        self.targets = [self._wnid_to_local[w] for w in labels]
        self._transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        arr = np.asarray(self._images[i])
        img = Image.fromarray(arr, mode="RGB")
        return self._transform(img), self.targets[i]


def get_mini_imagenet(data_root: str = "data", image_size: int = 224,
                     split: str = "test",
                     split_path: Path | str | None = None) -> Dataset:
    """Return a MiniImageNet Dataset for one Ravi & Larochelle split.

    split: "train" (64 classes) | "val" (16 classes) | "test" (20 classes).
    Unlike get_cifar_fs there is no `class_ids=` override -- every source
    layout is already split by design, so Step 9 only needs whole-split
    routing (see module docstring).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    split_map = load_mini_imagenet_split(split_path=split_path)
    images, labels = _ensure_cache(data_root, split, split_map)
    transform = _build_transform(image_size)
    return _MiniImageNetSplit(images, labels, transform)


def get_mini_imagenet_heldout_ood(data_root: str = "data", image_size: int = 224,
                                  num_samples: int = 500, seed: int = 42,
                                  heldout_split: str = "val") -> torch.Tensor:
    """Zero-extra-download near-OOD pool, mirroring
    get_cifar_fs_heldout_ood: samples from the 16 MiniImageNet val classes
    (disjoint from the 20 test-episode classes), same visual domain, novel
    classes only."""
    import random
    ds = get_mini_imagenet(data_root=data_root, image_size=image_size,
                           split=heldout_split)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
    return torch.stack([ds[i][0] for i in idx])
