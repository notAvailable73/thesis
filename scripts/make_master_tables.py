"""Step 10 (MVT grid execution) — three master tables from
results/mvt_results.json (implementation.txt Section 10.4 / plan.md Section
10.5), each as a LaTeX booktabs .tex + a 300 dpi PNG:

  1. Accuracy      — evidential vs softmax, PEFT rows + a separate
                      "Baselines" (Full-FT, Linear-Probe) block.
  2. Calibration    — ECE (pooled), ECE after temperature scaling
                      (softmax only), Brier.
  3. OOD AUROC      — SVHN (far) + TinyImageNet (near), evidential-vacuity
                      vs softmax-MSP / TS-MSP / energy.

Every row is read from mvt_results.json's {dataset}.{kshot}.{backbone}.
{adapter}.{head} nodes (scripts/aggregate_grid.py's schema) — nothing here
recomputes a metric, it only formats what aggregate_grid.py already
aggregated across seeds. A missing cell renders as "-" rather than raising,
since a partial grid (a session that hasn't finished yet) still produces a
readable partial table.

Usage:
    python scripts/make_master_tables.py
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

DEFAULT_IN = REPO_ROOT / "results" / "mvt_results.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results"

DATASET_LABEL = {"cifar_fs": "CIFAR-FS", "mini_imagenet": "MiniImageNet"}
BACKBONE_LABEL = {"resnet18": "ResNet-18", "mobilenetv3_small": "MobileNetV3-Small"}
ADAPTER_LABEL = {
    "bottleneck_parallel": "Bottleneck (parallel)",
    "lora": "LoRA",
    "full_ft": "Full-FT",
    "linear_probe": "Linear-Probe",
}
PEFT_ADAPTERS = ["bottleneck_parallel", "lora"]
BASELINE_ADAPTERS = ["full_ft", "linear_probe"]
DATASETS = ["cifar_fs", "mini_imagenet"]
BACKBONES = ["resnet18", "mobilenetv3_small"]
SHOTS = [5, 1]


def _get(mvt: dict, dataset: str, k_shot: int, backbone: str, adapter: str,
        head: str) -> dict | None:
    try:
        return mvt["results"][dataset][f"{k_shot}shot"][backbone][adapter][head]
    except KeyError:
        return None


def _stat(node: dict | None, metric: str) -> tuple[float, float] | None:
    if node is None or metric not in node:
        return None
    m = node[metric]
    return (m["mean"], m["ci95"])


def _fmt(stat: tuple[float, float] | None, digits: int = 3) -> str:
    if stat is None:
        return "-"
    mean, ci95 = stat
    return f"{mean:.{digits}f}±{ci95:.{digits}f}"


def _peft_rows() -> list[tuple]:
    rows = []
    for dataset in DATASETS:
        for k_shot in SHOTS:
            for backbone in BACKBONES:
                for adapter in PEFT_ADAPTERS:
                    rows.append((dataset, k_shot, backbone, adapter))
    return rows


def _baseline_rows() -> list[tuple]:
    return [("cifar_fs", k_shot, "resnet18", adapter)
           for k_shot in SHOTS for adapter in BASELINE_ADAPTERS]


def _row_label(dataset: str, k_shot: int, backbone: str, adapter: str) -> list[str]:
    return [DATASET_LABEL[dataset], f"{k_shot}-shot", BACKBONE_LABEL[backbone],
           ADAPTER_LABEL[adapter]]


# --------------------------------------------------------------------------
# Table builders: each returns (header, [row_cells...], [section_break_idx...])
# --------------------------------------------------------------------------
def build_accuracy_table(mvt: dict) -> tuple[list[str], list[list[str]], set[int]]:
    header = ["Dataset", "Shots", "Backbone", "Adapter",
             "Acc (evidential)", "Acc (softmax)", "Params (evid/softmax)"]
    rows = []
    for cell in _peft_rows():
        dataset, k_shot, backbone, adapter = cell
        evid = _get(mvt, dataset, k_shot, backbone, adapter, "evidential")
        sm = _get(mvt, dataset, k_shot, backbone, adapter, "softmax")
        n_e = _stat(evid, "n_params")
        n_s = _stat(sm, "n_params")
        params = (f"{int(n_e[0]):,}/{int(n_s[0]):,}" if n_e and n_s else "-")
        rows.append(_row_label(*cell) + [
            _fmt(_stat(evid, "accuracy_mean")),
            _fmt(_stat(sm, "accuracy_mean")),
            params,
        ])
    section_breaks = {len(rows)}
    for cell in _baseline_rows():
        dataset, k_shot, backbone, adapter = cell
        evid = _get(mvt, dataset, k_shot, backbone, adapter, "evidential")
        sm = _get(mvt, dataset, k_shot, backbone, adapter, "softmax")
        n_e = _stat(evid, "n_params")
        n_s = _stat(sm, "n_params")
        params = (f"{int(n_e[0]):,}/{int(n_s[0]):,}" if n_e and n_s else "-")
        rows.append(_row_label(*cell) + [
            _fmt(_stat(evid, "accuracy_mean")),
            _fmt(_stat(sm, "accuracy_mean")),
            params,
        ])
    return header, rows, section_breaks


def build_calibration_table(mvt: dict) -> tuple[list[str], list[list[str]], set[int]]:
    header = ["Dataset", "Shots", "Backbone", "Adapter",
             "ECE (evidential)", "ECE (softmax)", "ECE-TS (softmax)",
             "Brier (evidential)", "Brier (softmax)"]
    all_cells = _peft_rows() + _baseline_rows()
    rows = []
    for cell in all_cells:
        dataset, k_shot, backbone, adapter = cell
        evid = _get(mvt, dataset, k_shot, backbone, adapter, "evidential")
        sm = _get(mvt, dataset, k_shot, backbone, adapter, "softmax")
        rows.append(_row_label(*cell) + [
            _fmt(_stat(evid, "ece_pooled")),
            _fmt(_stat(sm, "ece_pooled")),
            _fmt(_stat(sm, "ece_ts")),
            _fmt(_stat(evid, "brier_mean")),
            _fmt(_stat(sm, "brier_mean")),
        ])
    return header, rows, {len(_peft_rows())}


def build_ood_auroc_table(mvt: dict) -> tuple[list[str], list[list[str]], set[int]]:
    header = ["Dataset", "Shots", "Backbone", "Adapter",
             "SVHN far (vacuity)", "SVHN far (MSP)", "SVHN far (TS-MSP)",
             "SVHN far (energy)", "TIN near (vacuity)", "TIN near (MSP)",
             "TIN near (TS-MSP)", "TIN near (energy)"]
    all_cells = _peft_rows() + _baseline_rows()
    rows = []
    for cell in all_cells:
        dataset, k_shot, backbone, adapter = cell
        evid = _get(mvt, dataset, k_shot, backbone, adapter, "evidential")
        sm = _get(mvt, dataset, k_shot, backbone, adapter, "softmax")
        rows.append(_row_label(*cell) + [
            _fmt(_stat(evid, "ood_auroc__svhn_far__vacuity")),
            _fmt(_stat(sm, "ood_auroc__svhn_far__msp")),
            _fmt(_stat(sm, "ood_auroc__svhn_far__ts_msp")),
            _fmt(_stat(sm, "ood_auroc__svhn_far__energy")),
            _fmt(_stat(evid, "ood_auroc__tin_near__vacuity")),
            _fmt(_stat(sm, "ood_auroc__tin_near__msp")),
            _fmt(_stat(sm, "ood_auroc__tin_near__ts_msp")),
            _fmt(_stat(sm, "ood_auroc__tin_near__energy")),
        ])
    return header, rows, {len(_peft_rows())}


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def _escape_tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")


def write_latex(path: Path, title: str, header: list[str], rows: list[list[str]],
                section_breaks: set[int]) -> None:
    n_cols = len(header)
    lines = [
        r"\begin{table}[t]", r"\centering",
        f"\\caption{{{title}}}",
        f"\\begin{{tabular}}{{{'l' * n_cols}}}", r"\toprule",
        " & ".join(_escape_tex(h) for h in header) + r" \\", r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i in section_breaks and i > 0:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(n_cols) +
                        r"}{l}{\textit{Baselines}} \\")
            lines.append(r"\midrule")
        lines.append(" & ".join(_escape_tex(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_png(path: Path, title: str, header: list[str], rows: list[list[str]],
             section_breaks: set[int]) -> None:
    display_rows = []
    for i, row in enumerate(rows):
        if i in section_breaks and i > 0:
            display_rows.append(["Baselines"] + [""] * (len(header) - 1))
        display_rows.append(row)

    fig_h = 0.42 * (len(display_rows) + 2)
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(header)), fig_h))
    ax.axis("off")
    table = ax.table(cellText=display_rows, colLabels=header, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(title, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found — run scripts/aggregate_grid.py "
                        f"first.")
    mvt = json.load(open(in_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        ("mvt_table_accuracy", "MVT Grid — Accuracy (mean ± 95% CI, N=600)",
         build_accuracy_table(mvt)),
        ("mvt_table_calibration", "MVT Grid — Calibration (ECE / Brier)",
         build_calibration_table(mvt)),
        ("mvt_table_ood_auroc", "MVT Grid — OOD AUROC (far: SVHN, near: TinyImageNet)",
         build_ood_auroc_table(mvt)),
    ]
    for stem, title, (header, rows, breaks) in tables:
        tex_path = out_dir / f"{stem}.tex"
        png_path = out_dir / f"{stem}.png"
        write_latex(tex_path, title, header, rows, breaks)
        write_png(png_path, title, header, rows, breaks)
        n_filled = sum(1 for row in rows for c in row[4:] if c != "-")
        n_total = sum(len(row) - 4 for row in rows)
        print(f"saved {tex_path} and {png_path} "
             f"({n_filled}/{n_total} data cells filled)")


if __name__ == "__main__":
    main()
