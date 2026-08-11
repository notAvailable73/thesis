"""Step 8 (Phase 5) — cross-backbone consolidation: ResNet-18 vs MobileNetV3-Small.

Reads the existing ResNet-18 result JSONs (Step 4.5 post_pool + Step 6 parallel)
and the four new MobileNetV3-Small ones, and writes:

  results/phase5_backbone_table.json    -- accuracy / F1 / ECE / Brier / native
                                           OOD AUROC per pool / n_params, per
                                           (backbone, placement, head) cell
  results/step8_backbone_comparison.png -- accuracy + far/near OOD AUROC + ECE
                                           bars, ResNet-18 vs MobileNetV3-Small
  results/step8_params_vs_accuracy.png  -- RQ4-prep scatter: trainable params
                                           (log x) vs accuracy, with the Step-5
                                           Full-FT baseline for scale

Backbone / placement are NOT read from the metrics JSONs: their schema is frozen
by the byte-identical-rerun invariant, so each cell is identified by its result
FILENAME here (same convention as scripts/step6_placement_plot.py and
scripts/step7_ood_consolidate.py). Missing JSONs are skipped with a warning so a
partial run still produces a table.

Usage:
    python scripts/step8_backbone_compare.py
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

HEADS = ["evidential", "softmax"]
#: native ID-ness score per head (same convention as Step 7)
NATIVE = {"evidential": "vacuity", "softmax": "msp"}

#: OOD pools, in far -> near order. svhn + cifar100 are in every run;
#: tin needs --use-tinyimagenet and gaussian needs --use-gaussian, so a cell
#: that was evaluated without them simply omits those columns.
POOLS = [("svhn_far", "far"), ("gaussian_far", "far"),
         ("cifar100_near", "near"), ("tin_near", "near")]

#: (label, backbone, placement, filename template). The ResNet-18 rows are
#: REUSED — no re-run — so those numbers stay exactly as reported in
#: step_writeups/step4_5.txt (post_pool) and step7.txt (parallel; Step 7's
#: re-eval of the Step-6 winner reproduced Step 6's svhn/cifar100/tin numbers
#: exactly and adds the gaussian pool, so it is the richer of the two files).
CELLS = [
    ("r18/post_pool", "resnet18", "post_pool",
     "step45_bottleneck_prototype-{head}_metrics.json"),
    ("r18/parallel", "resnet18", "parallel",
     "phase4_parallel_bottleneck_prototype-{head}_metrics.json"),
    ("mbnet/post_pool", "mobilenetv3_small", "post_pool",
     "phase5_mbnet_bottleneck_prototype-{head}_metrics.json"),
    ("mbnet/parallel", "mobilenetv3_small", "parallel",
     "phase5_mbnet_parallel_bottleneck_prototype-{head}_metrics.json"),
]

#: Step-5 Full-FT reference point for the params-vs-accuracy plot (ResNet-18).
FULL_FT = "phase3_full_ft_prototype-{head}_metrics.json"


def _load() -> dict:
    """{(label, head): metrics_dict} for every result JSON that exists."""
    out = {}
    for label, _bb, _pl, template in CELLS:
        for head in HEADS:
            p = RESULTS / template.format(head=head)
            if p.exists():
                out[(label, head)] = json.load(open(p))
            else:
                print(f"[warn] missing {p.name} — skipped", file=sys.stderr)
    return out


def _cell_row(d: dict, head: str) -> dict:
    """Pull the reported metrics for one (cell, head) into a flat dict."""
    score = NATIVE[head]
    ood = {}
    for pool, regime in POOLS:
        auroc = d.get(f"ood_auroc__{pool}__{score}")
        fpr = d.get(f"fpr_at_95_tpr__{pool}__{score}")
        if auroc is not None:
            ood[pool] = {
                "regime": regime,
                "auroc": round(float(auroc), 4),
                "fpr_at_95": round(float(fpr), 4) if fpr is not None else None,
            }
    row = {
        "accuracy_mean": round(float(d["accuracy_mean"]), 4),
        "accuracy_ci95": round(float(d.get("accuracy_ci95", float("nan"))), 4),
        "f1_macro_mean": (round(float(d["f1_macro_mean"]), 4)
                          if d.get("f1_macro_mean") is not None else None),
        "ece_pooled": round(float(d["ece_pooled"]), 4),
        "brier_mean": round(float(d["brier_mean"]), 4),
        "n_params": int(d.get("n_params", 0)),
        "num_episodes": int(d.get("num_episodes", 0)),
        "best_val_epoch": int(d.get("best_val_epoch", -1)),
        "ood_score": score,
        "ood": ood,
    }
    ece_ts = d.get("ece_ts")
    if ece_ts is not None:
        row["ece_ts"] = round(float(ece_ts), 4)
    # Softmax runs also carry the two fair non-native baselines (temperature-
    # scaled MSP + energy) that Steps 4.5/5/6/7 always report alongside vacuity —
    # keep them in the table so the Step 8 comparison is on the same footing.
    for alt in ("ts_msp", "energy"):
        alt_ood = {}
        for pool, _regime in POOLS:
            auroc = d.get(f"ood_auroc__{pool}__{alt}")
            if auroc is not None:
                alt_ood[pool] = round(float(auroc), 4)
        if alt_ood:
            row[f"ood_auroc_{alt}"] = alt_ood
    return row


def _fmt(v, spec="{:.3f}"):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def main() -> None:
    data = _load()
    if not data:
        raise SystemExit(
            "No result JSONs found. Run the Step 8 configs (and keep the "
            "Step 4.5 / Step 6 ResNet-18 baselines) before consolidating.")

    table = {}
    for label, backbone, placement, _t in CELLS:
        for head in HEADS:
            d = data.get((label, head))
            if d is None:
                continue
            table.setdefault(label, {
                "backbone": backbone, "placement": placement, "heads": {},
            })["heads"][head] = _cell_row(d, head)

    out_json = RESULTS / "phase5_backbone_table.json"
    with open(out_json, "w") as f:
        json.dump({
            "protocol": "CIFAR-FS Bertinetto, 5-way 5-shot, frozen test seeds",
            "score_per_head": NATIVE,
            "cells": table,
        }, f, indent=2, sort_keys=True)
    print(f"saved {out_json}")

    # --- console table (what gets transcribed into step8.txt) -------------
    print()
    print(f"{'cell':18s} {'head':11s} {'acc':>7s} {'F1':>7s} {'ECE':>7s} "
          f"{'Brier':>7s} {'svhn':>7s} {'gauss':>7s} {'c100':>7s} {'tin':>7s} "
          f"{'params':>10s} {'bestEp':>7s}")
    print("-" * 112)
    for label in table:
        for head in HEADS:
            r = table[label]["heads"].get(head)
            if r is None:
                continue
            g = lambda p: r["ood"].get(p, {}).get("auroc")
            print(f"{label:18s} {head:11s} "
                  f"{_fmt(r['accuracy_mean']):>7s} {_fmt(r['f1_macro_mean']):>7s} "
                  f"{_fmt(r['ece_pooled']):>7s} {_fmt(r['brier_mean']):>7s} "
                  f"{_fmt(g('svhn_far')):>7s} {_fmt(g('gaussian_far')):>7s} "
                  f"{_fmt(g('cifar100_near')):>7s} {_fmt(g('tin_near')):>7s} "
                  f"{r['n_params']:>10,d} {r['best_val_epoch']:>7d}")
    print()

    labels = [lb for lb, *_ in CELLS if lb in table]
    x = np.arange(len(labels))
    width = 0.38
    colors = {"evidential": "#4C72B0", "softmax": "#DD8452"}

    # --- grouped bars: accuracy / far-OOD / near-OOD / ECE ----------------
    panels = [
        ("accuracy_mean", None, "Accuracy", True),
        (None, "svhn_far", "far-OOD AUROC (SVHN, native)", True),
        (None, "cifar100_near", "near-OOD AUROC (CIFAR-100, native)", True),
        ("ece_pooled", None, "ECE (pooled)", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(19, 4.6))
    for ax, (key, pool, title, higher) in zip(axes, panels):
        for j, head in enumerate(HEADS):
            vals = []
            for lb in labels:
                r = table[lb]["heads"].get(head)
                if r is None:
                    vals.append(np.nan)
                elif pool is not None:
                    vals.append(r["ood"].get(pool, {}).get("auroc", np.nan))
                else:
                    vals.append(r[key])
            bars = ax.bar(x + (j - 0.5) * width, vals, width, label=head,
                          color=colors[head])
            for rect, v in zip(bars, vals):
                if not (v is None or np.isnan(v)):
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height(), f"{v:.3f}", ha="center",
                            va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title + ("  (↑ better)" if higher else "  (↓ better)"),
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("value")
    axes[0].legend(title="head", fontsize=9)
    param_bits = [f"{lb}={next(iter(table[lb]['heads'].values()))['n_params']:,}"
                  for lb in labels]
    fig.suptitle(
        "Step 8 (Phase 5) — ResNet-18 vs MobileNetV3-Small on CIFAR-FS "
        "(5-way 5-shot, N=600)\ntrainable params: " + "   ".join(param_bits),
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_png = RESULTS / "step8_backbone_comparison.png"
    fig.savefig(out_png, dpi=300)
    print(f"saved {out_png}")

    # --- RQ4 prep: trainable params vs accuracy --------------------------
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.2))
    markers = {"resnet18": "o", "mobilenetv3_small": "s"}
    for lb in labels:
        bb = table[lb]["backbone"]
        for head in HEADS:
            r = table[lb]["heads"].get(head)
            if r is None or not r["n_params"]:
                continue
            ax2.scatter(r["n_params"], r["accuracy_mean"], s=90,
                        marker=markers.get(bb, "^"), color=colors[head],
                        edgecolor="black", linewidth=0.6, zorder=3)
            ax2.annotate(f"{lb}\n{head}", (r["n_params"], r["accuracy_mean"]),
                         textcoords="offset points", xytext=(8, -3), fontsize=7)
    for head in HEADS:
        p = RESULTS / FULL_FT.format(head=head)
        if p.exists():
            d = json.load(open(p))
            ax2.scatter(int(d.get("n_params", 0)), float(d["accuracy_mean"]),
                        s=90, marker="x", color="grey", zorder=3)
            ax2.annotate(f"r18/full-FT\n{head}",
                         (int(d.get("n_params", 0)), float(d["accuracy_mean"])),
                         textcoords="offset points", xytext=(-58, -14),
                         fontsize=7, color="grey")
    ax2.set_xscale("log")
    ax2.set_xlabel("trainable parameters (log scale)")
    ax2.set_ylabel("5-way 5-shot accuracy (N=600)")
    ax2.set_title("Step 8 (RQ4 prep) — trainable params vs accuracy\n"
                  "circles = ResNet-18, squares = MobileNetV3-Small, "
                  "x = Step-5 Full-FT", fontsize=10)
    ax2.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[h], label=h)
               for h in HEADS]
    ax2.legend(handles=handles, title="head", fontsize=9, loc="lower right")
    fig2.tight_layout()
    out_png2 = RESULTS / "step8_params_vs_accuracy.png"
    fig2.savefig(out_png2, dpi=300)
    print(f"saved {out_png2}")


if __name__ == "__main__":
    main()
