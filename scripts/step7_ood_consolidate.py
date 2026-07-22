"""Step 7 (RQ3) — consolidate the extended OOD matrix into phase4_ood_table.json.

Reads the parallel-winner metrics JSONs (evidential + softmax) produced with
--use-tinyimagenet --use-gaussian, pulls each head's NATIVE OOD score
(vacuity for evidential, msp for softmax) per pool, and writes:

  results/phase4_ood_table.json   -- {config, n_id, n_ood, heads:{head:{ds:{auroc,fpr_at_95}}}}
  results/step7_ood_comparison.png -- far (svhn/gaussian) vs near (cifar100/tin) AUROC bars

Usage:
    python scripts/step7_ood_consolidate.py
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

# pool key in the metrics JSON -> (table name, regime)
POOLS = [
    ("svhn_far", "svhn", "far"),
    ("gaussian_far", "gaussian", "far"),
    ("cifar100_near", "cifar100", "near"),
    ("tin_near", "tinyimagenet", "near"),
]
HEADS = ["evidential", "softmax"]
NATIVE = {"evidential": "vacuity", "softmax": "msp"}


def _json_path(head: str) -> Path:
    return RESULTS / f"phase4_parallel_bottleneck_prototype-{head}_metrics.json"


def main() -> None:
    heads_out = {}
    for head in HEADS:
        p = _json_path(head)
        if not p.exists():
            print(f"[warn] missing {p.name} — skipped", file=sys.stderr)
            continue
        d = json.load(open(p))
        score = NATIVE[head]
        ds_out = {}
        for pool_key, name, _regime in POOLS:
            auroc = d.get(f"ood_auroc__{pool_key}__{score}")
            fpr = d.get(f"fpr_at_95_tpr__{pool_key}__{score}")
            if auroc is not None:
                ds_out[name] = {"auroc": round(float(auroc), 4),
                                "fpr_at_95": (round(float(fpr), 4)
                                              if fpr is not None else None)}
        heads_out[head] = ds_out

    if not heads_out:
        raise SystemExit(
            "No phase4 result JSONs found. Run the parallel configs with "
            "--use-tinyimagenet --use-gaussian --results-suffix phase4_parallel first.")

    table = {
        "config": "exp_phase3_placement_parallel",
        "n_id": "600 episodes x 75 query",
        "n_ood": 500,
        "score_per_head": NATIVE,
        "heads": heads_out,
    }
    out_json = RESULTS / "phase4_ood_table.json"
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    print(f"saved {out_json}")

    # --- far-vs-near AUROC bar chart -------------------------------------
    names = [name for _k, name, _r in POOLS]
    regimes = [r for _k, _n, r in POOLS]
    x = np.arange(len(names))
    width = 0.38
    colors = {"evidential": "#4C72B0", "softmax": "#DD8452"}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for j, head in enumerate(HEADS):
        vals = [heads_out.get(head, {}).get(n, {}).get("auroc", np.nan) for n in names]
        bars = ax.bar(x + (j - 0.5) * width, vals, width,
                      label=f"{head} ({NATIVE[head]})", color=colors[head])
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({r}-OOD)" for n, r in zip(names, regimes)])
    ax.axvline(1.5, color="grey", ls="--", alpha=0.5)  # far | near divider
    ax.set_ylabel("OOD AUROC (↑ better)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Step 7 (RQ3) — far vs near OOD, parallel placement (N=600)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = RESULTS / "step7_ood_comparison.png"
    fig.savefig(out_png, dpi=300)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
