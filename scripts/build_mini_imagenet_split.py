"""Materialize the canonical Ravi & Larochelle (2017) MiniImageNet 64/16/20
split.

Source (authoritative; the canonical release location for this exact split):
the twitter/meta-learning-lstm repo's data/miniImagenet/{train,val,test}.csv,
each a `filename,label` table where `label` is the WordNet ID (wnid) of the
image's class. We keep only the distinct wnids per file -- this script does
NOT need the image filenames, only which classes belong to which partition.

Deterministic parse (NOT a summariser) so the 100 wnids cannot be silently
dropped/altered. Validates: counts 64/16/20, pairwise-disjoint, union == 100
wnids, AND cross-checks the fetched result against the MINI_IMAGENET_SPLIT
constant frozen in src/datasets/mini_imagenet.py -- if a future re-run of
this script against the live URLs ever disagrees with the frozen constant,
that is a loud error, not a silent overwrite, because every offline
structural test in tests/test_mini_imagenet.py is written against the frozen
constant.

Writes data/mini_imagenet_split.json with _status="canonical_ravi_larochelle"
(mirrors cifar_fs_split.json's "canonical_bertinetto_via_torchmeta" role).

Run:  python scripts/build_mini_imagenet_split.py
If network is unavailable, src/datasets/mini_imagenet.py's
load_mini_imagenet_split() falls back to the same frozen constant offline
(with a UserWarning) -- so this script is a convenience materializer, not a
hard dependency.
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from src.datasets.mini_imagenet import MINI_IMAGENET_SPLIT  # noqa: E402

_BASE = ("https://raw.githubusercontent.com/twitter/meta-learning-lstm/"
         "master/data/miniImagenet")
_SPLIT_FILES = {"train": f"{_BASE}/train.csv",
                "val":   f"{_BASE}/val.csv",
                "test":  f"{_BASE}/test.csv"}
_OUT = _REPO_ROOT / "data" / "mini_imagenet_split.json"


def _fetch_wnids(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    wnids = sorted({row["label"] for row in reader})
    return wnids


def main() -> None:
    split_wnids = {k: _fetch_wnids(u) for k, u in _SPLIT_FILES.items()}

    counts = {"train": 64, "val": 16, "test": 20}
    for split, wnids in split_wnids.items():
        if len(wnids) != counts[split]:
            raise SystemExit(
                f"{split}: got {len(wnids)} wnids, expected {counts[split]}")

    tr, va, te = (set(split_wnids[k]) for k in ("train", "val", "test"))
    if (tr & va) or (va & te) or (tr & te):
        raise SystemExit("splits overlap on wnids")
    if len(tr | va | te) != 100:
        raise SystemExit(f"splits do not total 100 disjoint wnids: got {len(tr | va | te)}")

    for split in ("train", "val", "test"):
        fetched = set(split_wnids[split])
        frozen = set(MINI_IMAGENET_SPLIT[split])
        if fetched != frozen:
            raise SystemExit(
                f"{split}: fetched wnids disagree with the frozen "
                f"MINI_IMAGENET_SPLIT constant in "
                f"src/datasets/mini_imagenet.py -- "
                f"missing={sorted(frozen - fetched)[:5]} "
                f"extra={sorted(fetched - frozen)[:5]}. "
                f"Update the constant (and re-check every offline test) "
                f"before trusting this fetch."
            )

    status = "canonical_ravi_larochelle"
    out = {
        "_comment": "MiniImageNet class split (Ravi & Larochelle 2017), "
                    "WordNet IDs (wnids); every source layout in "
                    "src/datasets/mini_imagenet.py is pre-split by design, "
                    "so no name->id mapping step is needed (unlike CIFAR-FS).",
        "_canonical_source": _BASE,
        "_status": status,
        "_freeze_after_fetch": "FROZEN — do not regenerate.",
        "train": sorted(split_wnids["train"]),
        "val":   sorted(split_wnids["val"]),
        "test":  sorted(split_wnids["test"]),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {_OUT}  (64/16/20, disjoint, union=100, status={status})")


if __name__ == "__main__":
    main()
