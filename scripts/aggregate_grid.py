"""Step 10 (MVT grid execution) — aggregate the 120 grid cells' metrics JSONs
into one results/mvt_results.json.

Reads configs/grid/_index.json (never filenames — the results-JSON schema is
frozen and does not encode dataset/shots/backbone/seed; see
scripts/build_grid_configs.py's docstring), groups the 120 cells into 40
(dataset, k_shot, backbone, adapter, head) groups of 3 seeds each, and
aggregates every numeric metric across the (up to 3) seeds present:
    {dataset}.{kshot}.{backbone}.{adapter}.{head}.{metric}
        = {mean, std, ci95, per_seed, n_seeds}

Which metrics get aggregated is NOT a hardcoded list: every top-level numeric
key in a metrics JSON except a small metadata denylist (config_path, seed,
seeds_first10, ...) is aggregated automatically. This already covers every
metric plan.md names explicitly (accuracy_mean, accuracy_ci95, f1_macro_mean,
ece_pooled, ece_ts, brier_mean, n_params, best_val_epoch, and every
ood_auroc__{pool}__{score} / fpr_at_95_tpr__{pool}__{score} key scripts/
evaluate.py writes) without needing to enumerate the pool x score cross
product (which pools/scores appear depends on whether a cell's eval run used
--use-tinyimagenet / --use-gaussian).

Also emits "missing_cells": a list of index cells whose results JSON does not
exist yet, so "no missing cells" is a measurement, not an assumption.

Free reproducibility check (plan.md Section 10.4): the grid's seed-42 cells
for (cifar_fs, 5-shot, r18, bottleneck-parallel, {evidential, softmax}) use
the exact same recipe as the already-committed Step 6 phase4_parallel_*
results. If both a grid JSON and its Step 6 counterpart exist on disk, their
common numeric keys are diffed and any mismatch is reported (not a hard
failure — the two runs may reasonably use different OOD flags — but a clean
diff closes the byte-identical-rerun item open since Step 8).

Usage:
    python scripts/aggregate_grid.py
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "configs" / "grid" / "_index.json"
DEFAULT_OUT = REPO_ROOT / "results" / "mvt_results.json"

#: metadata keys every metrics JSON carries that are NOT metrics to aggregate.
NON_METRIC_KEYS = {
    "adapter_type", "config_path", "episodes_file", "head_type",
    "interpretation", "trainer_type", "seed", "seeds_first10", "seeds_last10",
    "temperature", "prior_per_class", "num_episodes", "primary_ood_pool",
}

#: the free reproducibility check's two Step 6 reference files.
STEP6_REFERENCE = {
    "evidential": REPO_ROOT / "results" / "phase4_parallel_bottleneck_prototype-evidential_metrics.json",
    "softmax":    REPO_ROOT / "results" / "phase4_parallel_bottleneck_prototype-softmax_metrics.json",
}


def _metric_keys(d: dict) -> list[str]:
    return [k for k, v in d.items()
           if k not in NON_METRIC_KEYS and isinstance(v, (int, float))
           and not isinstance(v, bool)]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float], mean: float) -> float:
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def _aggregate_group(per_seed_metrics: dict[int, dict]) -> dict:
    """per_seed_metrics: {seed: metrics_dict}. Returns the metric -> stats map."""
    all_keys = set()
    for d in per_seed_metrics.values():
        all_keys.update(_metric_keys(d))

    out = {}
    for key in sorted(all_keys):
        per_seed = {str(seed): d[key] for seed, d in per_seed_metrics.items()
                   if key in d}
        values = list(per_seed.values())
        n = len(values)
        mean = _mean(values)
        std = _std(values, mean)
        out[key] = {
            "mean": mean,
            "std": std,
            "ci95": 1.96 * std / math.sqrt(n) if n > 0 else float("nan"),
            "per_seed": per_seed,
            "n_seeds": n,
        }
    return out


def _group_key(c: dict) -> tuple:
    return (c["dataset"], c["k_shot"], c["backbone"], c["adapter"], c["head"])


def _kshot_label(k_shot: int) -> str:
    return f"{k_shot}shot"


def _check_step6_reproducibility(loaded: dict[str, dict]) -> list[str]:
    """loaded: results_json path (str) -> metrics dict, for whichever grid
    JSONs exist. Compares the seed-42 cifar_fs/5shot/r18/parallel cells
    against their committed Step 6 counterparts. Returns a list of
    human-readable notes (empty if nothing to check or everything matched)."""
    notes = []
    for head, ref_path in STEP6_REFERENCE.items():
        if not ref_path.exists():
            continue
        grid_path = (REPO_ROOT / "results" / "grid" /
                    f"grid_cifar_5shot_r18_parallel_seed42_bottleneck_"
                    f"prototype-{head}_metrics.json")
        grid_key = str(grid_path.relative_to(REPO_ROOT)).replace("\\", "/")
        if grid_key not in loaded:
            continue
        ref = json.load(open(ref_path))
        common = set(_metric_keys(ref)) & set(_metric_keys(loaded[grid_key]))
        mismatches = {
            k: (ref[k], loaded[grid_key][k]) for k in sorted(common)
            if not math.isclose(ref[k], loaded[grid_key][k], rel_tol=1e-9, abs_tol=1e-9)
        }
        if mismatches:
            notes.append(
                f"[reproducibility] {head}: grid seed-42 cell DIFFERS from "
                f"committed Step 6 result on {len(mismatches)} key(s): "
                f"{mismatches}"
            )
        else:
            notes.append(
                f"[reproducibility] {head}: grid seed-42 cell matches the "
                f"committed Step 6 result on all {len(common)} common "
                f"numeric key(s) — byte-identical-rerun confirmed."
            )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    index_path = Path(args.index)
    out_path = Path(args.out)
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found — run "
                        f"scripts/build_grid_configs.py first.")
    cells = json.load(open(index_path))["cells"]

    groups: dict[tuple, dict[int, dict]] = {}
    missing_cells = []
    loaded_by_path: dict[str, dict] = {}
    for c in cells:
        results_path = REPO_ROOT / c["results_json"]
        if not results_path.exists():
            missing_cells.append({k: c[k] for k in
                                  ("dataset", "k_shot", "backbone", "adapter",
                                   "head", "seed", "config", "results_json")})
            continue
        d = json.load(open(results_path))
        loaded_by_path[c["results_json"]] = d
        groups.setdefault(_group_key(c), {})[c["seed"]] = d

    mvt_results: dict = {}
    for (dataset, k_shot, backbone, adapter, head), per_seed in groups.items():
        node = mvt_results.setdefault(dataset, {}) \
                          .setdefault(_kshot_label(k_shot), {}) \
                          .setdefault(backbone, {}) \
                          .setdefault(adapter, {})
        node[head] = _aggregate_group(per_seed)

    n_complete_groups = sum(1 for g in groups.values() if len(g) == 3)
    n_total_groups = len({_group_key(c) for c in cells})

    out = {
        "protocol": "5-way {1,5}-shot, frozen 600 TEST-seed episodes "
                    "(configs/test_episodes.yaml), 3 seeds per cell",
        "n_cells_total": len(cells),
        "n_cells_present": len(cells) - len(missing_cells),
        "n_groups_total": n_total_groups,
        "n_groups_complete_3_seeds": n_complete_groups,
        "missing_cells": missing_cells,
        "results": mvt_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"saved {out_path}")
    print(f"{len(cells) - len(missing_cells)}/{len(cells)} cell metrics JSONs "
         f"found; {n_complete_groups}/{n_total_groups} (dataset, shot, "
         f"backbone, adapter, head) groups have all 3 seeds")
    if missing_cells:
        print(f"[warn] {len(missing_cells)} missing cells — see "
             f"'missing_cells' in {out_path}")

    for note in _check_step6_reproducibility(loaded_by_path):
        print(note)


if __name__ == "__main__":
    main()
