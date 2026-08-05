"""Step 10 (MVT grid execution) — the "16 plots" deliverable (implementation.
txt Section 10 exit criteria / plan.md Section 10.6).

Reading note (flag this explicitly in step_writeups/step10.txt so it isn't
read as a silently changed exit criterion): the exit-criterion phrase
"reliability + OOD histogram per dataset x head" is only 2 x 2 = 4
combinations by itself. plan.md's reading — which this script implements —
is 16 = dataset (2) x shots (2) x head (2) x plot type (2): shots is folded
in as a third selection axis alongside dataset/head, and backbone/adapter are
collapsed down to one "best" cell per (dataset, shot, head) rather than being
separate plot axes.

For each of the 8 (dataset, k_shot, head) combinations, the "representative
cell" is the (backbone, adapter) pair with the highest aggregated
accuracy_mean in results/mvt_results.json among the 4 PEFT (backbone,
adapter) choices — a data-driven pick (never assume the Step 6 CIFAR-FS
"parallel wins" finding transfers to every dataset/shot/head cell; Step 10
is partly what tests whether it does) — at seed 42 specifically (the grid's
designated reproducibility seed).

scripts/evaluate.py already writes a reliability + OOD-histogram PNG per run
at dpi=200 (src/utils/plots.py); it does NOT persist the raw pooled
probabilities / ID-vs-OOD scores anywhere a downstream script could re-plot
from, so "re-renders at 300 dpi" here means a faithful mechanical raster
re-render of the EXISTING PNG at a higher DPI (upscaled, not vector-
regenerated from new data) — done this way specifically so this script never
needs to touch scripts/evaluate.py (a previous step's frozen entry point).

Usage:
    python scripts/grid_plots.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_MVT = REPO_ROOT / "results" / "mvt_results.json"
DEFAULT_INDEX = REPO_ROOT / "configs" / "grid" / "_index.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "grid_plots"
SOURCE_DPI = 200   # src/utils/plots.py's fixed save dpi
TARGET_DPI = 300

DATASETS = ["cifar_fs", "mini_imagenet"]
SHOTS = [5, 1]
HEADS = ["evidential", "softmax"]
BACKBONES = ["resnet18", "mobilenetv3_small"]
PEFT_ADAPTERS = ["bottleneck_parallel", "lora"]
PLOT_KINDS = ["reliability", "ood_histogram"]


def _pick_best_cell(mvt: dict, dataset: str, k_shot: int, head: str):
    """Highest accuracy_mean among the 4 (backbone, adapter) PEFT choices.
    Returns (backbone, adapter) or None if nothing is present yet."""
    best = None
    best_acc = float("-inf")
    for backbone in BACKBONES:
        for adapter in PEFT_ADAPTERS:
            try:
                node = mvt["results"][dataset][f"{k_shot}shot"][backbone][adapter][head]
            except KeyError:
                continue
            acc = node.get("accuracy_mean", {}).get("mean")
            if acc is not None and acc > best_acc:
                best_acc, best = acc, (backbone, adapter)
    return best


def _find_index_cell(index: list[dict], dataset: str, k_shot: int, backbone: str,
                     adapter: str, head: str, seed: int) -> dict | None:
    for c in index:
        if (c["dataset"] == dataset and c["k_shot"] == k_shot
                and c["backbone"] == backbone and c["adapter"] == adapter
                and c["head"] == head and c["seed"] == seed):
            return c
    return None


def _source_png_path(results_json_rel: str, kind: str) -> Path:
    stem = results_json_rel[: -len("_metrics.json")]
    return REPO_ROOT / f"{stem}_{kind}.png"


def _rel(p: Path) -> str:
    """Path relative to REPO_ROOT for display, falling back to the absolute
    path when `p` lies outside it (e.g. a test pointed --out-dir elsewhere)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _rerender_at_300dpi(src: Path, dst: Path, title: str) -> None:
    img = Image.open(src)
    w, h = img.size
    fig = plt.figure(figsize=(w / SOURCE_DPI, h / SOURCE_DPI))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title, y=0.02, fontsize=6, color="gray")
    fig.savefig(dst, dpi=TARGET_DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvt", default=str(DEFAULT_MVT))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mvt_path, index_path = Path(args.mvt), Path(args.index)
    if not mvt_path.exists():
        raise SystemExit(f"{mvt_path} not found — run scripts/aggregate_grid.py first.")
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found — run scripts/build_grid_configs.py first.")
    mvt = json.load(open(mvt_path))
    index = json.load(open(index_path))["cells"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_rendered = 0
    for dataset in DATASETS:
        for k_shot in SHOTS:
            for head in HEADS:
                combo_id = f"{dataset}_{k_shot}shot_{head}"
                best = _pick_best_cell(mvt, dataset, k_shot, head)
                if best is None:
                    print(f"[skip] {combo_id}: no aggregated accuracy yet "
                         f"for any (backbone, adapter) — run the grid + "
                         f"aggregate_grid.py first.")
                    manifest.append({"combo": combo_id, "status": "no_data"})
                    continue
                backbone, adapter = best
                cell = _find_index_cell(index, dataset, k_shot, backbone,
                                        adapter, head, args.seed)
                if cell is None:
                    print(f"[skip] {combo_id}: best cell ({backbone}, "
                         f"{adapter}) has no matching seed-{args.seed} grid "
                         f"index entry.")
                    manifest.append({"combo": combo_id, "status": "no_index_cell"})
                    continue

                for kind in PLOT_KINDS:
                    src = _source_png_path(cell["results_json"], kind)
                    dst = out_dir / f"{combo_id}_{kind}.png"
                    entry = {"combo": combo_id, "kind": kind,
                            "backbone": backbone, "adapter": adapter,
                            "source": _rel(src), "dest": _rel(dst)}
                    if not src.exists():
                        print(f"[skip] {combo_id}/{kind}: source PNG not "
                             f"found ({_rel(src)}) — run the grid first.")
                        entry["status"] = "missing_source"
                    else:
                        _rerender_at_300dpi(
                            src, dst,
                            title=f"{combo_id} — {backbone}/{adapter} (seed {args.seed})")
                        entry["status"] = "rendered"
                        n_rendered += 1
                        print(f"[ok] {_rel(dst)}  <- {_rel(src)}")
                    manifest.append(entry)

    manifest_path = out_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"n_rendered": n_rendered, "n_expected": 16,
                  "plots": manifest}, f, indent=2, sort_keys=True)
    print(f"\n{n_rendered}/16 plots rendered; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
