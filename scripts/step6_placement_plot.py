"""Step 6 (RQ1) — placement comparison bar chart.

Reads the 3 placement variants x 2 heads (6 metrics JSONs) and draws grouped
bars for accuracy / OOD-AUROC(SVHN far, native score) / ECE(pooled), annotated
with trainable-param counts. Missing JSONs are skipped with a warning so the
plot can be drawn from a partial run.

  post_pool : results/step45_bottleneck_prototype-<head>_metrics.json   (reused)
  serial    : results/phase3_placement_serial_bottleneck_prototype-<head>_metrics.json
  parallel  : results/phase3_placement_parallel_bottleneck_prototype-<head>_metrics.json

Usage:
    python scripts/step6_placement_plot.py
    -> results/step6_placement_comparison.png (300 dpi)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

PLACEMENTS = ["post_pool", "serial", "parallel"]
HEADS = ["evidential", "softmax"]

# (metric key in JSON, axis title, whether higher is better)
METRICS = [
    ("accuracy_mean", "Accuracy", True),
    ("ood_auroc_mean", "OOD AUROC (SVHN far, native)", True),
    ("ece_pooled", "ECE (pooled)", False),
]


def _path(placement: str, head: str) -> Path:
    if placement == "post_pool":
        return RESULTS / f"step45_bottleneck_prototype-{head}_metrics.json"
    return RESULTS / (
        f"phase3_placement_{placement}_bottleneck_prototype-{head}_metrics.json")


def _load() -> dict:
    """Return {(placement, head): metrics_dict} for every JSON that exists."""
    out = {}
    for placement in PLACEMENTS:
        for head in HEADS:
            p = _path(placement, head)
            if p.exists():
                out[(placement, head)] = json.load(open(p))
            else:
                print(f"[warn] missing {p.name} — skipped", file=sys.stderr)
    return out


def main() -> None:
    data = _load()
    if not data:
        raise SystemExit(
            "No result JSONs found. Run the Step 6 configs (and keep the "
            "step45 post_pool baselines) before plotting.")

    x = np.arange(len(PLACEMENTS))
    width = 0.38
    colors = {"evidential": "#4C72B0", "softmax": "#DD8452"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title, higher_better) in zip(axes, METRICS):
        for j, head in enumerate(HEADS):
            vals = [data.get((pl, head), {}).get(key, np.nan) for pl in PLACEMENTS]
            bars = ax.bar(x + (j - 0.5) * width, vals, width,
                          label=head, color=colors[head])
            for rect, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height(), f"{v:.3f}", ha="center",
                            va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PLACEMENTS)
        ax.set_title(title + ("  (↑ better)" if higher_better else "  (↓ better)"),
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("value")
    axes[0].legend(title="head", fontsize=9)

    # Param-count annotation (trainable) per placement — read from any head JSON.
    param_bits = []
    for pl in PLACEMENTS:
        for head in HEADS:
            d = data.get((pl, head))
            if d and d.get("n_params") is not None:
                param_bits.append(f"{pl}={d['n_params']:,}")
                break
    fig.suptitle(
        "Step 6 (RQ1) — adapter placement on CIFAR-FS (5-way 5-shot, N=600)\n"
        "trainable params: " + "   ".join(param_bits),
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out = RESULTS / "step6_placement_comparison.png"
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
