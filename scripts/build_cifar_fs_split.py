"""Materialize the canonical Bertinetto 2019 CIFAR-FS 64/16/20 split.

Source (authoritative, already used by notebooks/step4_episodic.ipynb): the
torchmeta mirror of the CIFAR-FS asset files. The original bertinetto/r2d2 repo
no longer hosts the lists. Each file is a JSON list of [superclass, class_name]
pairs; we keep only class_name and map it to a CIFAR-100 integer id via
torchvision's alphabetical class order (src/datasets/cifar_fs.CIFAR100_CLASS_NAMES).

Deterministic parse (NOT a summariser) so the 100 class names cannot be
silently dropped/altered. Validates: counts 64/16/20, pairwise-disjoint,
union == 0..99. Writes data/cifar_fs_split.json with
_status="canonical_bertinetto_via_torchmeta" — the SAME status label already
used by notebooks/step4_episodic.ipynb's fetch cell (same torchmeta source),
so a split fetched by either mechanism is recognized as canonical.

Run:  python scripts/build_cifar_fs_split.py
If network is unavailable, run notebooks/step4_episodic.ipynb's fetch cell
instead (it embeds the same canonical class names offline as a fallback).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from src.datasets.cifar_fs import CIFAR100_CLASS_NAMES  # noqa: E402

_BASE = ("https://raw.githubusercontent.com/tristandeleu/pytorch-meta/"
         "master/torchmeta/datasets/assets/cifar100/cifar-fs")
_SPLIT_FILES = {"train": f"{_BASE}/train.json",
                "val":   f"{_BASE}/val.json",
                "test":  f"{_BASE}/test.json"}
_OUT = _REPO_ROOT / "data" / "cifar_fs_split.json"


def _fetch_names(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=30) as r:
        pairs = json.loads(r.read().decode("utf-8"))
    # Each entry is [superclass, class_name]; keep class_name only.
    return [p[1] for p in pairs]


def main() -> None:
    name_to_id = {n: i for i, n in enumerate(CIFAR100_CLASS_NAMES)}
    split_names = {k: _fetch_names(u) for k, u in _SPLIT_FILES.items()}

    counts = {"train": 64, "val": 16, "test": 20}
    for split, names in split_names.items():
        if len(names) != counts[split]:
            raise SystemExit(f"{split}: got {len(names)} names, expected {counts[split]}")
        unknown = [n for n in names if n not in name_to_id]
        if unknown:
            raise SystemExit(f"{split}: unknown CIFAR-100 class names {unknown}")

    ids = {k: sorted(name_to_id[n] for n in v) for k, v in split_names.items()}
    tr, va, te = (set(ids[k]) for k in ("train", "val", "test"))
    if (tr & va) or (va & te) or (tr & te):
        raise SystemExit("splits overlap on class ids")
    if (tr | va | te) != set(range(100)):
        raise SystemExit("splits do not cover all 100 CIFAR-100 ids")

    # Status string matches the convention already established by
    # notebooks/step4_episodic.ipynb's fetch cell (same torchmeta source) —
    # NOT a new invented label — so a split fetched by either mechanism is
    # recognized as canonical by anything checking `_status`.
    status = "canonical_bertinetto_via_torchmeta"
    out = {
        "_comment": "CIFAR-FS class split (Bertinetto 2019), class NAMES; "
                    "loader maps names -> ids via torchvision CIFAR-100 order.",
        "_canonical_source": _BASE,
        "_status": status,
        "_freeze_after_fetch": "FROZEN — do not regenerate.",
        "train": split_names["train"],
        "val":   split_names["val"],
        "test":  split_names["test"],
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {_OUT}  (64/16/20, disjoint, union=100, status={status})")


if __name__ == "__main__":
    main()
