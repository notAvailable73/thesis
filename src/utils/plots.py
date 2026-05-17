"""Plot helpers for Step 3 final artifacts: reliability diagram, OOD score
histogram, confusion matrix. Each function writes a PNG and returns the path
so callers can hand it to WandbRun.log_image().

These were inline in notebooks/step1_calibration_fix.ipynb. Lifting them
into src/ means scripts/evaluate.py can emit the same plots without
re-running the Step 1 notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless; works under subprocess on Colab + CI
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reliability_diagram(probs: torch.Tensor, targets: torch.Tensor,
                        save_path: str | Path, *, num_bins: int = 15,
                        title: str = "Reliability diagram") -> str:
    """Confidence-vs-accuracy plot. Diagonal is perfect calibration."""
    p = _to_numpy(probs)
    y = _to_numpy(targets)
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_count = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if i == 0:
            mask |= (confidences == 0.0)
        if mask.any():
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confidences[mask].mean()
            bin_count[i] = mask.sum()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    width = 1.0 / num_bins
    ax.bar(centers, bin_acc, width=width * 0.95, edgecolor="black",
           color="#4C9F70", alpha=0.85, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="perfect")
    ax.scatter(bin_conf[bin_count > 0], bin_acc[bin_count > 0],
               s=40, color="#1F3A93", zorder=3, label="bin (conf, acc)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return str(save_path)


def ood_histogram(id_scores, ood_scores, save_path: str | Path,
                  *, title: str = "ID vs OOD uncertainty score") -> str:
    """Overlapping histograms of in-distribution and OOD detection scores.
    Higher = more confidently in-distribution."""
    ids = _to_numpy(id_scores).ravel()
    ood = _to_numpy(ood_scores).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(ids.min(), ood.min()), max(ids.max(), ood.max()), 40)
    ax.hist(ids, bins=bins, alpha=0.65, label=f"ID  (n={len(ids)})", color="#4C9F70")
    ax.hist(ood, bins=bins, alpha=0.65, label=f"OOD (n={len(ood)})", color="#C0392B")
    ax.set_xlabel("score (1 - vacuity)"); ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return str(save_path)


def confusion_matrix(probs: torch.Tensor, targets: torch.Tensor,
                     save_path: str | Path, *, num_classes: int,
                     title: str = "Confusion matrix") -> str:
    """Row-normalised confusion matrix (rows = true class)."""
    p = _to_numpy(probs); y = _to_numpy(targets)
    pred = p.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, q in zip(y, pred):
        cm[int(t), int(q)] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return str(save_path)


__all__ = ["reliability_diagram", "ood_histogram", "confusion_matrix"]
