"""TMLR plan Task 5 — build data/aircraft_84.npz from the raw FGVC-Aircraft
archive, print its content hash and (once) the frozen class split.

    python scripts/tmlr_build_aircraft_cache.py --raw data/aircraft_raw --out data
    python scripts/tmlr_build_aircraft_cache.py --raw data/aircraft_raw --out data --make-split --seed 2026

--raw may be the archive's `data/` directory or any folder above it. Kaggle
sessions do not need this script: they build the same file from the attached
dataset (src/datasets/fgvc_aircraft.ensure_aircraft_cache) and refuse to run
if its hash differs from AIRCRAFT_CACHE_SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import PIL  # noqa: E402

from src.datasets import fgvc_aircraft as A  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data")
    ap.add_argument("--make-split", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)

    raw = A.find_raw_root(str(a.raw))
    if raw is None:
        print(f"no fgvc-aircraft-2013b data/ directory under {a.raw}")
        return 1
    t0 = time.time()
    images, labels, variants = A.build_arrays(raw)
    sha = A.content_sha256(images, labels, variants)
    counts = np.bincount(labels, minlength=len(variants))
    print(f"N={len(labels)} variants={len(variants)} per-class min={counts.min()} "
          f"max={counts.max()} ({time.time() - t0:.0f} s)")
    print(f"AIRCRAFT_CACHE_SHA256 = {sha!r}")
    if A.AIRCRAFT_CACHE_SHA256 and sha != A.AIRCRAFT_CACHE_SHA256:
        print(f"MISMATCH: the committed hash is {A.AIRCRAFT_CACHE_SHA256}")
        return 1
    out = a.out / A.CACHE_NAME
    A.write_cache(out, images, labels, variants)
    meta = {
        "content_sha256": sha,
        "n_images": int(len(labels)),
        "n_variants": len(variants),
        "crop": "images_box.txt, 1-based inclusive -> PIL (x0-1, y0-1, x1, y1)",
        "resize": f"PIL.Image.BILINEAR to {A.CACHE_SIZE}x{A.CACHE_SIZE}, RGB",
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "source": "fgvc-aircraft-2013b (Kaggle mirror seryouxblaster764/fgvc-aircraft)",
        "images_box_sha256": hashlib.sha256((raw / "images_box.txt").read_bytes()).hexdigest(),
    }
    (a.out / "aircraft_84.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(f"wrote {out}")
    if a.make_split:
        split = A.make_split(variants, a.seed)
        print("\nAIRCRAFT_SPLIT = " + json.dumps(split, indent=4))
        print(f"AIRCRAFT_SPLIT_SHA256 = {A.split_sha256(split)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
