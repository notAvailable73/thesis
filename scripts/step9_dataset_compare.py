"""Step 9 (Phase 5) — cross-dataset consolidation: CIFAR-FS vs MiniImageNet.

Reads the existing CIFAR-FS result JSONs (Step 4.5 post_pool, Step 6 parallel,
Step 8 mbnet post_pool/parallel, Step 5 linear-probe baseline) and the ten new
MiniImageNet ones (Step 9), and writes:

  results/phase5_dataset_table.json    -- accuracy / F1 / ECE / Brier / native
                                          OOD AUROC per pool / n_params, per
                                          (dataset, backbone, placement, head)
                                          cell
  results/step9_dataset_comparison.png -- accuracy + far/near OOD AUROC + ECE
                                          bars, CIFAR-FS vs MiniImageNet
  results/step9_adapter_uplift.png     -- (parallel accuracy - linear-probe
                                          accuracy) per dataset per head: the
                                          adapter's uplift over raw frozen
                                          ImageNet features, the number that
                                          turns the "MiniImageNet classes were
                                          seen during ImageNet pretraining"
                                          confound into something measured
                                          rather than only caveated.

Dataset / backbone / placement are NOT read from the metrics JSONs: their
schema is frozen by the byte-identical-rerun invariant, so each cell is
identified by its result FILENAME here (same convention as
scripts/step8_backbone_compare.py). Missing JSONs are skipped with a warning
so a partial run still produces a table.

The near-OOD pool key differs by dataset (cifar100_near for CIFAR-FS,
mini_near for MiniImageNet -- see src/datasets/__init__.py:
get_heldout_near_ood); both are remapped to one logical "heldout_near" column
here so the two datasets' near-OOD numbers sit in the same table column.

Usage:
    python scripts/step9_dataset_compare.py
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
#: native ID-ness score per head (same convention as Steps 7/8)
NATIVE = {"evidential": "vacuity", "softmax": "msp"}

#: raw pool key -> (regime, canonical output column). "near" is resolved per
#: dataset via NEAR_POOL_BY_DATASET below and always written to the
#: "heldout_near" column regardless of which raw key produced it.
NEAR_POOL_BY_DATASET = {"cifar_fs": "cifar100_near", "mini_imagenet": "mini_near"}

#: (label, backbone, placement, dataset, filename template). CIFAR-FS rows are
#: REUSED — no re-run — so those numbers stay exactly as reported in
#: step_writeups/step4_5.txt (post_pool), step6.txt/step7.txt (parallel),
#: step5.txt (linear_probe) and step8.txt (mbnet). MiniImageNet rows are the
#: new Step 9 runs.
CELLS = [
    ("cifar_fs/r18/post_pool", "resnet18", "post_pool", "cifar_fs",
     "step45_bottleneck_prototype-{head}_metrics.json"),
    ("cifar_fs/r18/parallel", "resnet18", "parallel", "cifar_fs",
     "phase4_parallel_bottleneck_prototype-{head}_metrics.json"),
    ("cifar_fs/mbnet/post_pool", "mobilenetv3_small", "post_pool", "cifar_fs",
     "phase5_mbnet_bottleneck_prototype-{head}_metrics.json"),
    ("cifar_fs/mbnet/parallel", "mobilenetv3_small", "parallel", "cifar_fs",
     "phase5_mbnet_parallel_bottleneck_prototype-{head}_metrics.json"),
    ("cifar_fs/r18/linear_probe", "resnet18", "linear_probe", "cifar_fs",
     "phase3_linear_probe_prototype-{head}_metrics.json"),
    ("mini/r18/post_pool", "resnet18", "post_pool", "mini_imagenet",
     "phase5_mini_postpool_bottleneck_prototype-{head}_metrics.json"),
    ("mini/r18/parallel", "resnet18", "parallel", "mini_imagenet",
     "phase5_mini_parallel_bottleneck_prototype-{head}_metrics.json"),
    ("mini/mbnet/post_pool", "mobilenetv3_small", "post_pool", "mini_imagenet",
     "phase5_mini_mbnet_postpool_bottleneck_prototype-{head}_metrics.json"),
    ("mini/mbnet/parallel", "mobilenetv3_small", "parallel", "mini_imagenet",
     "phase5_mini_mbnet_parallel_bottleneck_prototype-{head}_metrics.json"),
    ("mini/r18/linear_probe", "resnet18", "linear_probe", "mini_imagenet",
     "phase5_mini_linear_probe_prototype-{head}_metrics.json"),
]

#: pairs used for the adapter-uplift plot: (parallel label, linear_probe label,
#: dataset). Both sides are ResNet-18, so the uplift isolates the adapter,
#: not a backbone swap.
UPLIFT_PAIRS = [
    ("cifar_fs/r18/parallel", "cifar_fs/r18/linear_probe", "cifar_fs"),
    ("mini/r18/parallel", "mini/r18/linear_probe", "mini_imagenet"),
]


def _load() -> dict:
    """{(label, head): metrics_dict} for every result JSON that exists."""
    out = {}
    for label, _bb, _pl, _ds, template in CELLS:
        for head in HEADS:
            p = RESULTS / template.format(head=head)
            if p.exists():
                out[(label, head)] = json.load(open(p))
            else:
                print(f"[warn] missing {p.name} — skipped", file=sys.stderr)
    return out


def _cell_row(d: dict, head: str, dataset: str) -> dict:
    """Pull the reported metrics for one (cell, head) into a flat dict."""
    score = NATIVE[head]
    near_key = NEAR_POOL_BY_DATASET[dataset]
    pool_map = [("svhn_far", "far", "svhn_far"),
                ("gaussian_far", "far", "gaussian_far"),
                (near_key, "near", "heldout_near"),
                ("tin_near", "near", "tin_near")]
    ood = {}
    for raw_pool, regime, out_key in pool_map:
        auroc = d.get(f"ood_auroc__{raw_pool}__{score}")
        fpr = d.get(f"fpr_at_95_tpr__{raw_pool}__{score}")
        if auroc is not None:
            ood[out_key] = {
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
    # scaled MSP + energy) that every prior step reports alongside vacuity —
    # keep them so the Step 9 comparison is on the same footing.
    for alt in ("ts_msp", "energy"):
        alt_ood = {}
        for raw_pool, _regime, out_key in pool_map:
            auroc = d.get(f"ood_auroc__{raw_pool}__{alt}")
            if auroc is not None:
                alt_ood[out_key] = round(float(auroc), 4)
        if alt_ood:
            row[f"ood_auroc_{alt}"] = alt_ood
    return row


def _fmt(v, spec="{:.3f}"):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def main() -> None:
    data = _load()
    if not data:
        raise SystemExit(
            "No result JSONs found. Run the Step 9 MiniImageNet configs "
            "(and keep the committed CIFAR-FS result JSONs) before "
            "consolidating.")

    table = {}
    for label, backbone, placement, dataset, _t in CELLS:
        for head in HEADS:
            d = data.get((label, head))
            if d is None:
                continue
            table.setdefault(label, {
                "backbone": backbone, "placement": placement,
                "dataset": dataset, "heads": {},
            })["heads"][head] = _cell_row(d, head, dataset)

    out_json = RESULTS / "phase5_dataset_table.json"
    with open(out_json, "w") as f:
        json.dump({
            "protocol": "5-way 5-shot, frozen test seeds (600 CIFAR-FS / "
                        "600 MiniImageNet episodes)",
            "score_per_head": NATIVE,
            "near_pool_by_dataset": NEAR_POOL_BY_DATASET,
            "cells": table,
        }, f, indent=2, sort_keys=True)
    print(f"saved {out_json}")

    # --- console table (what gets transcribed into step9.txt) -------------
    print()
    print(f"{'cell':26s} {'head':11s} {'acc':>7s} {'F1':>7s} {'ECE':>7s} "
          f"{'Brier':>7s} {'svhn':>7s} {'near':>7s} {'tin':>7s} "
          f"{'params':>10s} {'bestEp':>7s}")
    print("-" * 120)
    for label in table:
        for head in HEADS:
            r = table[label]["heads"].get(head)
            if r is None:
                continue
            g = lambda p: r["ood"].get(p, {}).get("auroc")
            print(f"{label:26s} {head:11s} "
                  f"{_fmt(r['accuracy_mean']):>7s} {_fmt(r['f1_macro_mean']):>7s} "
                  f"{_fmt(r['ece_pooled']):>7s} {_fmt(r['brier_mean']):>7s} "
                  f"{_fmt(g('svhn_far')):>7s} {_fmt(g('heldout_near')):>7s} "
                  f"{_fmt(g('tin_near')):>7s} "
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
        (None, "heldout_near", "near-OOD AUROC (held-out classes, native)", True),
        ("ece_pooled", None, "ECE (pooled)", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 5.2))
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
                            va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title + ("  (↑ better)" if higher else "  (↓ better)"),
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("value")
    axes[0].legend(title="head", fontsize=9)
    fig.suptitle(
        "Step 9 (Phase 5) — CIFAR-FS vs MiniImageNet, both backbones "
        "(5-way 5-shot, N=600)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = RESULTS / "step9_dataset_comparison.png"
    fig.savefig(out_png, dpi=300)
    print(f"saved {out_png}")

    # --- adapter uplift: parallel accuracy - linear-probe accuracy --------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ds_x = np.arange(len(UPLIFT_PAIRS))
    for j, head in enumerate(HEADS):
        vals = []
        for par_lb, lp_lb, _ds in UPLIFT_PAIRS:
            par = table.get(par_lb, {}).get("heads", {}).get(head)
            lp = table.get(lp_lb, {}).get("heads", {}).get(head)
            if par is None or lp is None:
                vals.append(np.nan)
            else:
                vals.append(par["accuracy_mean"] - lp["accuracy_mean"])
        bars = ax2.bar(ds_x + (j - 0.5) * width, vals, width, label=head,
                       color=colors[head])
        for rect, v in zip(bars, vals):
            if not (v is None or np.isnan(v)):
                ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                         f"{v:+.3f}", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(ds_x)
    ax2.set_xticklabels([ds for _, _, ds in UPLIFT_PAIRS], fontsize=10)
    ax2.set_ylabel("parallel accuracy − linear-probe accuracy")
    ax2.set_title(
        "Step 9 — adapter uplift over raw frozen ImageNet features\n"
        "(ResNet-18, parallel Bottleneck vs Linear-Probe, per dataset)",
        fontsize=10)
    ax2.legend(title="head", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_png2 = RESULTS / "step9_adapter_uplift.png"
    fig2.savefig(out_png2, dpi=300)
    print(f"saved {out_png2}")


if __name__ == "__main__":
    main()
