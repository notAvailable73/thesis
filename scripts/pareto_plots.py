"""Step 11 (RQ4) -- Pareto-frontier plots + the audit record.

Reads results/mvt_results.json (required) and results/efficiency_table.json
(optional -- the latency-vs-AUROC half degrades to "skipped, not yet
measured" until a session has produced it; the params-vs-accuracy half needs
NO new measurement and is fully computable today).

Two families of output:

  EXIT-CRITERIA FIGURES, flat in results/ (the phase4_ood_table.json /
  step8_*.png precedent -- Chapter 5 cites these paths literally):
    results/pareto_params_vs_accuracy.png              (2x2: rows=dataset, cols=shots)
    results/pareto_params_vs_accuracy__cifar_fs.png    (1x2: 5-shot | 1-shot)
    results/pareto_params_vs_accuracy__mini_imagenet.png
    results/pareto_latency_vs_auroc.png                (2x2, same layout)
    results/pareto_latency_vs_auroc__cifar_fs.png
    results/pareto_latency_vs_auroc__mini_imagenet.png
    results/pareto_frontier.json   -- frontier membership under EVERY axis
                                       variant considered, so the primary-axis
                                       choice (step_writeups/step11.txt
                                       Section 0) is auditable, not asserted.

  AUDIT FIGURES, subdir results/pareto_audit/ (the grid_plots/ precedent):
    results/pareto_audit/quality_axis_variants__{dataset}.png
    results/pareto_audit/_manifest.json

PRE-REGISTERED AXES (must not change after looking at numbers -- see
step_writeups/step11.txt Section 0):
  cost (11.3)    : CPU 1-thread per-image median latency (ms) -- RQ4 is an
                   edge question; batch-1 is the edge regime; single-thread
                   matches Howard 2019's phone-latency methodology.
  cost (11.4)    : trainable parameters (log axis).
  quality        : ood_auroc__tin_near__<native>  (vacuity for evidential,
                   msp for softmax) -- the only near-OOD pool common to both
                   datasets; near-OOD is what RQ3 found the low-data trend
                   on; far-OOD pools are near-saturated and stop
                   discriminating.

Baselines (Full-FT, Linear-Probe) are ELIGIBLE frontier points -- excluding
the cheapest (0-param) and most-expensive (11.18M-param) reference points
would flatter the frontier by construction. They exist only for
cifar_fs/resnet18, so MiniImageNet panels carry zero baseline points --
`n_baseline_points` is recorded per panel rather than left implicit.

Usage:
    python scripts/pareto_plots.py
    python scripts/pareto_plots.py --env local_cpu
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

from src.utils.pareto import pareto_front, recommended_point

DEFAULT_MVT = REPO_ROOT / "results" / "mvt_results.json"
DEFAULT_EFFICIENCY = REPO_ROOT / "results" / "efficiency_table.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_AUDIT_DIR = REPO_ROOT / "results" / "pareto_audit"

DATASETS = ["cifar_fs", "mini_imagenet"]
SHOTS = [5, 1]
BACKBONES = ["resnet18", "mobilenetv3_small"]
PEFT_ADAPTERS = ["bottleneck_parallel", "lora"]
BASELINE_ADAPTERS = ["full_ft", "linear_probe"]
HEADS = ["evidential", "softmax"]
NATIVE = {"evidential": "vacuity", "softmax": "msp"}
NEAR_POOL_BY_DATASET = {"cifar_fs": "cifar100_near", "mini_imagenet": "mini_near"}

COLORS = {"evidential": "#4C72B0", "softmax": "#DD8452"}
MARKERS = {"resnet18": "o", "mobilenetv3_small": "s"}

#: pre-registered recommendation tolerance (Section 0): cheapest point within
#: this margin of the panel's best quality.
TOL_ACCURACY = 0.01     # 1.0 pp
TOL_AUROC = 0.010


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
    return (float(m["mean"]), float(m.get("ci95", 0.0)))


def _cells(dataset: str) -> list[tuple[str, str]]:
    """(backbone, adapter) pairs available for `dataset`."""
    pairs = [(bb, ad) for bb in BACKBONES for ad in PEFT_ADAPTERS]
    if dataset == "cifar_fs":
        pairs += [("resnet18", ad) for ad in BASELINE_ADAPTERS]
    return pairs


def _quality_variant(mvt, dataset: str, k_shot: int, backbone: str,
                     adapter: str, head: str, variant: str
                     ) -> tuple[float, float] | None:
    node = _get(mvt, dataset, k_shot, backbone, adapter, head)
    if node is None:
        return None
    native = NATIVE[head]
    if variant == "accuracy":
        return _stat(node, "accuracy_mean")
    if variant == "near_tin_native":
        return _stat(node, f"ood_auroc__tin_near__{native}")
    if variant == "far_svhn_native":
        return _stat(node, f"ood_auroc__svhn_far__{native}")
    if variant == "far_gaussian_native":
        return _stat(node, f"ood_auroc__gaussian_far__{native}")
    if variant == "near_dataset_native":
        return _stat(node, f"ood_auroc__{NEAR_POOL_BY_DATASET[dataset]}__{native}")
    if variant == "near_tin_best_score":
        if head == "evidential":
            return _stat(node, "ood_auroc__tin_near__vacuity")
        candidates = [_stat(node, f"ood_auroc__tin_near__{s}")
                     for s in ("msp", "ts_msp", "energy")]
        candidates = [c for c in candidates if c is not None]
        return max(candidates, key=lambda c: c[0]) if candidates else None
    raise ValueError(f"Unknown quality variant: {variant!r}")


def _load_efficiency(path: Path | None = None) -> dict | None:
    """`path` defaults to DEFAULT_EFFICIENCY. Found 2026-08-09 alongside the
    local_cpu profile-selection bug: this previously ignored its caller's
    `--efficiency` argument entirely (main() called `_load_efficiency()`
    with no arguments), so that CLI flag silently did nothing -- the same
    class of dead-option bug as --env (see _non_canonical_profile_fragments)."""
    p = path or DEFAULT_EFFICIENCY
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _non_canonical_profile_fragments(effic: dict) -> list[str]:
    """Slug fragments belonging to the `local_cpu` dev-machine environment
    (step_writeups/step11.txt Section 4.1: "development artifact, not
    reported as an edge number"), derived the same way
    src/utils/efficiency.py:device_profile_slug() builds a profile key, so
    they can be recognised and excluded from the primary-axis pick below
    without needing environments[env_id]['device_profile_slug'] to be
    populated (it is None in every session observed so far).

    BUG THIS FIXES (found 2026-08-09, real Kaggle re-run data): without
    this exclusion, _primary_cost_profile did a bare `profile.startswith(
    prefer_prefix)` over per_image's dict-insertion order. Because
    scripts/efficiency_table.py merges NEW sessions onto the EXISTING file
    (never clobbers), local_cpu's cpu_1thread_* profile is always inserted
    first, so it always won that match over the canonical
    cpu_1thread_intel-...-2-00ghz Kaggle profile -- silently making every
    committed Pareto latency figure and pareto_frontier.json read from an
    uncontrolled personal laptop instead of the Kaggle CPU edge proxy
    (80.43ms local vs 62.38ms Kaggle for resnet18/bottleneck_parallel/
    evidential, a ~29% error on the RQ4 headline number)."""
    import re
    env = effic.get("environments", {}).get("local_cpu")
    if not env:
        return []
    cpu_model = env.get("host", {}).get("cpu_model")
    if not cpu_model:
        return []
    return [re.sub(r"[^a-z0-9]+", "-", cpu_model.lower()).strip("-")]


def _primary_cost_profile(effic: dict, key: str, *, prefer_prefix: str = "cpu_1thread_"
                          ) -> tuple[str, dict] | None:
    """Median per-image latency under the pre-registered primary profile
    (CPU, 1 thread) -- excluding the local_cpu dev machine's profile, which
    would otherwise win on dict-insertion order alone (see
    _non_canonical_profile_fragments's docstring for the incident this
    fixes). Falls back to any CPU profile, then anything present INCLUDING
    local_cpu, so a local_cpu-only file (Section 3's offline dev runs, no
    Kaggle session yet) still produces SOME figure rather than nothing."""
    per_image = effic.get("measured", {}).get(key, {}).get("per_image", {})
    if not per_image:
        return None
    excluded = _non_canonical_profile_fragments(effic)

    def is_excluded(profile: str) -> bool:
        return any(frag in profile for frag in excluded)

    for profile, timing in per_image.items():
        if profile.startswith(prefer_prefix) and not is_excluded(profile):
            return profile, timing
    for profile, timing in per_image.items():
        if profile.startswith("cpu_") and not is_excluded(profile):
            return profile, timing
    for profile, timing in per_image.items():
        if profile.startswith(prefer_prefix):
            return profile, timing
    for profile, timing in per_image.items():
        if profile.startswith("cpu_"):
            return profile, timing
    profile, timing = next(iter(per_image.items()))
    return profile, timing


def _cost_latency(effic: dict | None, key: str) -> tuple[float, str] | None:
    if effic is None:
        return None
    picked = _primary_cost_profile(effic, key)
    if picked is None:
        return None
    profile, timing = picked
    return timing["latency_ms"]["median"], profile


def _n_params(mvt, dataset, k_shot, backbone, adapter, head) -> float | None:
    node = _get(mvt, dataset, k_shot, backbone, adapter, head)
    stat = _stat(node, "n_params")
    return stat[0] if stat else None


def _episode_key_for(dataset: str, backbone: str, adapter: str) -> str:
    return f"{backbone}|{adapter}"


# --------------------------------------------------------------------------
# Frontier computation (shared by both figure families)
# --------------------------------------------------------------------------
def _build_panel_points(mvt, dataset: str, k_shot: int, *, cost: str,
                        quality_variant: str, effic: dict | None):
    """Returns list of dicts: {backbone, adapter, head, cost, quality,
    quality_ci95, label, is_baseline, x_plot}."""
    points = []
    for backbone, adapter in _cells(dataset):
        for head in HEADS:
            if cost == "params":
                c = _n_params(mvt, dataset, k_shot, backbone, adapter, head)
                x_plot = c if (c is not None and c > 0) else (1.0 if c == 0 else None)
            elif cost == "latency":
                key = _episode_key_for(dataset, backbone, adapter)
                # backbone/adapter's compute is dataset/shot-independent, but
                # the key must still carry the HEAD (params/latency differ
                # by ~2 params, and the key convention includes head).
                eff_key = f"{backbone}|{adapter}|{head}"
                lat = _cost_latency(effic, eff_key)
                c = lat[0] if lat else None
                x_plot = c
            else:
                raise ValueError(cost)
            if c is None:
                continue
            q = _quality_variant(mvt, dataset, k_shot, backbone, adapter, head,
                                 quality_variant)
            if q is None:
                continue
            points.append({
                "backbone": backbone, "adapter": adapter, "head": head,
                "cost": c, "x_plot": x_plot, "quality": q[0], "quality_ci95": q[1],
                "is_baseline": adapter in BASELINE_ADAPTERS,
                "label": f"{backbone}/{adapter}/{head}",
            })
    return points


def _panel_frontier(points: list[dict], *, tol_qual: float) -> dict:
    coords = [(p["cost"], p["quality"]) for p in points]
    strict = pareto_front(coords)
    eps_qual = [p["quality_ci95"] for p in points]
    eps_cost = [0.001 * p["cost"] for p in points]
    eps_front = pareto_front(coords, eps_cost=eps_cost, eps_qual=eps_qual)
    rec_idx = recommended_point(coords, [p["label"] for p in points], tol_qual=tol_qual) \
        if points else None
    return {
        "n_points": len(points),
        "n_baseline_points": sum(1 for p in points if p["is_baseline"]),
        "strict_front": sorted(points[i]["label"] for i in strict),
        "eps_front": sorted(points[i]["label"] for i in eps_front),
        "recommended_point": points[rec_idx]["label"] if rec_idx is not None else None,
        "strict_front_idx": strict,
        "eps_front_idx": eps_front,
        "recommended_idx": rec_idx,
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _annotate_offset(head: str) -> tuple[int, int]:
    return (8, 8) if head == "evidential" else (8, -12)


def _plot_panel(ax, points: list[dict], frontier: dict, *, xlabel: str,
                ylabel: str, logx: bool, title: str):
    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return
    strict_idx = set(frontier["strict_front_idx"])
    eps_idx = set(frontier["eps_front_idx"])

    by_adapter: dict[tuple, list[int]] = {}
    for i, p in enumerate(points):
        by_adapter.setdefault((p["backbone"], p["adapter"]), []).append(i)

    for (backbone, adapter), idxs in by_adapter.items():
        if len(idxs) == 2:
            xs = [points[i]["x_plot"] for i in idxs]
            ys = [points[i]["quality"] for i in idxs]
            ax.plot(xs, ys, color="grey", linewidth=0.6, zorder=1)

    for i, p in enumerate(points):
        marker = MARKERS.get(p["backbone"], "^")
        color = COLORS[p["head"]]
        edge = "black" if p["is_baseline"] else color
        lw = 1.6 if p["is_baseline"] else 0.6
        ax.scatter(p["x_plot"], p["quality"], s=90, marker=marker, color=color,
                  edgecolor=edge, linewidth=lw, zorder=3)
        if i in strict_idx:
            ax.scatter(p["x_plot"], p["quality"], s=260, marker="*",
                      color="none", edgecolor="gold", linewidth=1.4, zorder=4)
        if i in eps_idx and i not in strict_idx:
            ax.scatter(p["x_plot"], p["quality"], s=340, marker="o",
                      facecolors="none", edgecolor="grey", linewidth=1.0, zorder=2)
        label = p["label"].replace("bottleneck_parallel", "parallel")
        if p["is_baseline"]:
            label += "*"
        ax.annotate(label, (p["x_plot"], p["quality"]),
                   textcoords="offset points", xytext=_annotate_offset(p["head"]),
                   fontsize=6.5)

    # Achievable-region boundary as a step staircase over the strict frontier.
    front_pts = sorted((points[i]["x_plot"], points[i]["quality"])
                       for i in strict_idx)
    if len(front_pts) > 1:
        xs, ys = zip(*front_pts)
        ax.step(xs, ys, where="post", color="grey", linestyle="--",
                linewidth=1.0, zorder=1, alpha=0.6)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def _render_family(mvt, effic, *, cost: str, quality_variant: str,
                   xlabel: str, ylabel: str, logx: bool, stem: str,
                   out_dir: Path, tol_qual: float, frontier_json: dict) -> list[Path]:
    written = []
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        for ax, k_shot in zip(axes, SHOTS):
            points = _build_panel_points(mvt, dataset, k_shot, cost=cost,
                                         quality_variant=quality_variant, effic=effic)
            frontier = _panel_frontier(points, tol_qual=tol_qual)
            panel_key = f"{dataset}__{k_shot}shot"
            frontier_json.setdefault(stem, {})[panel_key] = {
                "cost_axis": cost, "quality_axis": quality_variant,
                **{k: v for k, v in frontier.items()
                   if k not in ("strict_front_idx", "eps_front_idx", "recommended_idx")},
                "points": [{"label": p["label"], "cost": p["cost"],
                           "quality": p["quality"], "is_baseline": p["is_baseline"]}
                          for p in points],
            }
            _plot_panel(ax, points, frontier, xlabel=xlabel, ylabel=ylabel,
                       logx=logx, title=f"{dataset} {k_shot}-shot")
        fig.suptitle(f"Step 11 -- {stem.replace('_', ' ')} ({dataset})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_png = out_dir / f"{stem}__{dataset}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        written.append(out_png)
        print(f"saved {out_png}")

    # Combined 2x2 (the literal spec filename): rows=dataset, cols=shots.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, dataset in enumerate(DATASETS):
        for col, k_shot in enumerate(SHOTS):
            ax = axes[row][col]
            points = _build_panel_points(mvt, dataset, k_shot, cost=cost,
                                         quality_variant=quality_variant, effic=effic)
            frontier = _panel_frontier(points, tol_qual=tol_qual)
            _plot_panel(ax, points, frontier, xlabel=xlabel, ylabel=ylabel,
                       logx=logx, title=f"{dataset} {k_shot}-shot")
    fig.suptitle(f"Step 11 -- {stem.replace('_', ' ')}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    combined_png = out_dir / f"{stem}.png"
    fig.savefig(combined_png, dpi=300)
    plt.close(fig)
    written.append(combined_png)
    print(f"saved {combined_png}")
    return written


# --------------------------------------------------------------------------
# Audit figures: quality-axis sensitivity
# --------------------------------------------------------------------------
def _render_quality_audit(mvt, out_dir: Path) -> list[dict]:
    variants = ["far_svhn_native", "near_dataset_native", "near_tin_best_score"]
    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, len(variants), figsize=(5 * len(variants), 9))
        for row, k_shot in enumerate(SHOTS):
            for col, variant in enumerate(variants):
                ax = axes[row][col]
                points = _build_panel_points(mvt, dataset, k_shot, cost="params",
                                             quality_variant=variant, effic=None)
                frontier = _panel_frontier(points, tol_qual=TOL_AUROC)
                _plot_panel(ax, points, frontier, xlabel="trainable params (log)",
                           ylabel=variant, logx=True,
                           title=f"{dataset} {k_shot}-shot / {variant}")
        fig.suptitle(f"Step 11 audit -- quality-axis sensitivity ({dataset})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = out_dir / f"quality_axis_variants__{dataset}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        manifest.append({"dataset": dataset, "dest": str(out_png.relative_to(REPO_ROOT)),
                         "status": "rendered", "variants": variants})
        print(f"saved {out_png}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvt", default=str(DEFAULT_MVT))
    ap.add_argument("--efficiency", default=str(DEFAULT_EFFICIENCY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    ap.add_argument("--env", default=None,
                    help="restrict latency lookups to profiles from a "
                        "specific environment's session id (not yet "
                        "enforced beyond documentation; efficiency_table.json's "
                        "measured block is keyed by hardware profile, not env, "
                        "so this is informational for now).")
    args = ap.parse_args()

    mvt_path = Path(args.mvt)
    if not mvt_path.exists():
        raise SystemExit(f"{mvt_path} not found -- run scripts/aggregate_grid.py first.")
    mvt = json.loads(mvt_path.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    efficiency_path = Path(args.efficiency)
    effic = _load_efficiency(efficiency_path)
    if effic is None:
        print(f"[pareto_plots] {efficiency_path} not found -- "
             "latency-vs-AUROC figures will render with 0 points "
             "(run scripts/efficiency_table.py first for real data).")

    if effic is not None and efficiency_path.is_relative_to(REPO_ROOT):
        efficiency_provenance = str(efficiency_path.relative_to(REPO_ROOT))
    elif effic is not None:
        efficiency_provenance = str(efficiency_path)
    else:
        efficiency_provenance = None

    frontier_json: dict = {"generated_from": {
        "mvt": str(mvt_path.relative_to(REPO_ROOT)) if mvt_path.is_relative_to(REPO_ROOT) else str(mvt_path),
        "efficiency": efficiency_provenance,
    }, "tol_accuracy": TOL_ACCURACY, "tol_auroc": TOL_AUROC}

    _render_family(mvt, None, cost="params", quality_variant="accuracy",
                   xlabel="trainable parameters (log)", ylabel="accuracy",
                   logx=True, stem="pareto_params_vs_accuracy", out_dir=out_dir,
                   tol_qual=TOL_ACCURACY, frontier_json=frontier_json)

    _render_family(mvt, effic, cost="latency", quality_variant="near_tin_native",
                   xlabel="CPU latency, 1 thread, per image (ms, log)",
                   ylabel="near-OOD AUROC (TinyImageNet, native score)",
                   logx=True, stem="pareto_latency_vs_auroc", out_dir=out_dir,
                   tol_qual=TOL_AUROC, frontier_json=frontier_json)

    frontier_path = out_dir / "pareto_frontier.json"
    frontier_path.write_text(json.dumps(frontier_json, indent=2, sort_keys=True))
    print(f"saved {frontier_path}")

    audit_dir = Path(args.audit_dir)
    manifest = _render_quality_audit(mvt, audit_dir)
    manifest_path = audit_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(
        {"n_rendered": len(manifest), "plots": manifest}, indent=2, sort_keys=True))
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
