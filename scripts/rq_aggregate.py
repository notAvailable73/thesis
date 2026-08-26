"""Aggregation — RQ1's 2x4 factorial + variance attribution, RQ2's
before/after, RQ5's rank curves.

Reads only the per-cell JSONs the drivers wrote, so it can be re-run locally
after the session without a GPU.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

#: RQ1's score axis. `vacuity_valfit` is the headline vacuity (T1.0b option ii:
#: the affine is refit on VAL by the SAME procedure in both arms).
#: `vacuity_native` is carried as a sensitivity row, not a factorial column.
RQ1_SCORES = ("msp", "energy", "ts_msp", "vacuity_valfit")
FAR_POOLS = ("svhn_far", "gaussian_far")
NEAR_POOLS = ("cifar100_near", "mini_near", "tin_near")


def load_records(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(Path(out_dir).glob("*.json")):
        if p.name.startswith("_"):
            continue
        recs.append(json.load(open(p)))
    return recs


def design_key(rec: dict) -> str:
    return (f"{rec['dataset']}/{rec['k_shot']}shot/{rec['backbone']}/"
            f"{rec['meta'].get('adapter', rec['adapter_type'])}")


# =====================================================================
# RQ1
# =====================================================================
def factorial_observations(recs: list[dict]) -> list[dict]:
    """One row per (design, objective, score, pool, seed) -> AUROC."""
    rows = []
    for r in recs:
        s = r["summary"]
        for key, val in s.items():
            if not key.startswith("ood_auroc__"):
                continue
            _, pool, score = key.split("__")
            if score not in RQ1_SCORES:
                continue
            rows.append({
                "design": design_key(r),
                "objective": r["interpretation"],
                "score": score,
                "pool": pool,
                "pool_group": ("far" if pool in FAR_POOLS else
                               "near" if pool in NEAR_POOLS else "other"),
                "seed": r["seed"],
                "auroc": float(val),
            })
    return rows


def table_2x4(rows: list[dict], pool_group: str | None = None) -> dict:
    """The 2 (objective) x 4 (score) matrix RQ1 exists to fill in.

    Only the diagonal of this table existed before: evidential cells were
    scored with vacuity only, softmax cells with msp/energy/ts_msp only.
    """
    sel = [r for r in rows if pool_group is None or r["pool_group"] == pool_group]
    out = {}
    for obj in ("evidential", "softmax"):
        out[obj] = {}
        for sc in RQ1_SCORES:
            vals = [r["auroc"] for r in sel
                    if r["objective"] == obj and r["score"] == sc]
            out[obj][sc] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
                "n": len(vals),
            }
    return out


def eta_squared(rows: list[dict], factors: list[str], value: str = "auroc",
                interactions: bool = True) -> dict:
    """Classical SS decomposition -> eta^2 per factor.

    Exact for a balanced design; the returned `balanced` flag says whether the
    design actually is balanced, because the formula quietly stops being exact
    when it is not, and a silently-wrong variance attribution is precisely the
    failure mode T4.1 flagged in the existing decomposition.
    """
    y = np.array([r[value] for r in rows], dtype=float)
    if len(y) == 0:
        return {"error": "no observations"}
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    levels = {f: np.array([r[f] for r in rows], dtype=object) for f in factors}

    cell_counts: dict[tuple, int] = {}
    for i in range(len(y)):
        key = tuple(levels[f][i] for f in factors)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    balanced = len(set(cell_counts.values())) == 1

    def group_means(keys):
        idx: dict[tuple, list[int]] = {}
        for i in range(len(y)):
            k = tuple(levels[f][i] for f in keys)
            idx.setdefault(k, []).append(i)
        return {k: (y[v].mean(), len(v)) for k, v in idx.items()}

    ss: dict[str, float] = {}
    main = {f: group_means([f]) for f in factors}
    for f in factors:
        ss[f] = float(sum(n * (m - grand) ** 2 for m, n in main[f].values()))

    if interactions:
        for a, b in combinations(factors, 2):
            gm = group_means([a, b])
            tot = 0.0
            for (la, lb), (m, n) in gm.items():
                ma = main[a][(la,)][0]
                mb = main[b][(lb,)][0]
                tot += n * (m - ma - mb + grand) ** 2
            ss[f"{a}:{b}"] = float(tot)

    ss_resid = ss_total - sum(ss.values())
    out = {k: (v / ss_total if ss_total else float("nan")) for k, v in ss.items()}
    out["residual"] = ss_resid / ss_total if ss_total else float("nan")
    return {"eta_squared": out, "ss_total": ss_total, "n": int(len(y)),
            "balanced": balanced,
            "cell_counts": sorted(set(cell_counts.values()))}


def rq1_verdict(rows: list[dict]) -> dict:
    """Does the OBJECTIVE or the SCORE explain more of the AUROC variance?"""
    out = {}
    for group in ("far", "near"):
        sel = [r for r in rows if r["pool_group"] == group]
        if not sel:
            continue
        eta = eta_squared(sel, ["design", "objective", "score"])
        e = eta["eta_squared"]
        out[group] = {
            "eta_squared": e,
            "dominant": ("score" if e.get("score", 0) > e.get("objective", 0)
                         else "objective"),
            "ratio_score_over_objective": (
                e.get("score", 0) / e["objective"]
                if e.get("objective", 0) > 0 else float("inf")),
            "balanced": eta["balanced"],
            "n": eta["n"],
        }
    return out


# =====================================================================
# RQ2
# =====================================================================
def rq2_table(recs: list[dict]) -> list[dict]:
    """Per evidential cell: ECE and OOD-AUROC before/after the VAL refit.

    The headline is the JOINT outcome — an ECE improvement only matters if the
    OOD ranking survives it, which is why every row carries both.
    """
    rows = []
    for r in recs:
        if r["interpretation"] != "evidential":
            continue
        s = r["summary"]
        row = {
            "cell": r["meta"].get("cell") or design_key(r) + f"/seed{r['seed']}",
            "design": design_key(r), "seed": r["seed"],
            "ece_before": s.get("ece_pooled__evidential_native"),
            "ece_after": s.get("ece_pooled__evidential_valfit"),
            "ece_softmax_ref": s.get("ece_pooled__softmax"),
            "ece_ts_ref": s.get("ece_pooled__ts"),
            "brier_before": s.get("brier_mean__evidential_native"),
            "brier_after": s.get("brier_mean__evidential_valfit"),
            "affine_trained": r["affine_native"],
            "affine_refit": r["affine_valfit"],
            "affine_config_init": r["affine_config_init"],
        }
        row["ece_delta"] = ((row["ece_after"] - row["ece_before"])
                            if None not in (row["ece_after"], row["ece_before"])
                            else None)
        for key in s:
            if key.startswith("ood_auroc__") and key.endswith("__vacuity_native"):
                pool = key.split("__")[1]
                before = s[key]
                after = s.get(f"ood_auroc__{pool}__vacuity_valfit")
                row[f"auroc_before__{pool}"] = before
                row[f"auroc_after__{pool}"] = after
                if after is not None:
                    row[f"auroc_delta__{pool}"] = after - before
            if key.startswith("ranking_shift__"):
                row[f"rank_rho__{key.split('__')[1]}"] = s[key]["spearman_rho"]
                row[f"rank_discordant__{key.split('__')[1]}"] = \
                    s[key]["discordant_fraction"]
        rows.append(row)
    return rows


def rq2_verdict(rows: list[dict]) -> dict:
    if not rows:
        return {"error": "no evidential cells"}
    ece_d = [r["ece_delta"] for r in rows if r["ece_delta"] is not None]
    auroc_d = [v for r in rows for k, v in r.items()
               if k.startswith("auroc_delta__") and v is not None]
    rho = [v for r in rows for k, v in r.items() if k.startswith("rank_rho__")]
    return {
        "n_cells": len(rows),
        "ece_improved_in": int(sum(1 for d in ece_d if d < 0)),
        "ece_mean_delta": float(np.mean(ece_d)) if ece_d else None,
        "auroc_preserved_in": int(sum(1 for d in auroc_d if d >= -0.005)),
        "auroc_comparisons": len(auroc_d),
        "auroc_mean_delta": float(np.mean(auroc_d)) if auroc_d else None,
        "min_spearman_rho": float(np.min(rho)) if rho else None,
        "reordering_ever_observed": bool(any(
            r.get(k, 0) > 0 for r in rows for k in r
            if k.startswith("rank_discordant__"))),
    }


# =====================================================================
# RQ5
# =====================================================================
def rq5_curve(recs: list[dict]) -> dict:
    """Accuracy / ECE / AUROC vs trainable-parameter budget at FIXED
    architecture — the curve that tells RQ5's U-shape from an artefact of
    varying the adapter type at the same time."""
    by_rank: dict[int, dict] = {}
    for r in recs:
        rank = r["adapter_rank"]
        s = r["summary"]
        e = by_rank.setdefault(rank, {"rank": rank, "n_params": r["n_params"],
                                      "seeds": [], "acc": [], "ece_evid": [],
                                      "ece_softmax": [], "ece_ts": [],
                                      "auroc_far": [], "auroc_near": []})
        e["seeds"].append(r["seed"])
        e["acc"].append(s.get("accuracy_mean__softmax"))
        e["ece_evid"].append(s.get("ece_pooled__evidential_native"))
        e["ece_softmax"].append(s.get("ece_pooled__softmax"))
        e["ece_ts"].append(s.get("ece_pooled__ts"))
        far = [s[f"ood_auroc__{p}__vacuity_native"] for p in FAR_POOLS
               if f"ood_auroc__{p}__vacuity_native" in s]
        near = [s[f"ood_auroc__{p}__vacuity_native"] for p in NEAR_POOLS
                if f"ood_auroc__{p}__vacuity_native" in s]
        e["auroc_far"].append(float(np.mean(far)) if far else None)
        e["auroc_near"].append(float(np.mean(near)) if near else None)

    curve = []
    for rank in sorted(by_rank):
        e = by_rank[rank]
        row = {"rank": rank, "n_params": e["n_params"], "n_seeds": len(e["seeds"])}
        for k in ("acc", "ece_evid", "ece_softmax", "ece_ts",
                  "auroc_far", "auroc_near"):
            vals = [v for v in e[k] if v is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals)) if vals else None
        curve.append(row)

    def _argopt(key, mode):
        pts = [(r["rank"], r[key]) for r in curve if r.get(key) is not None]
        if not pts:
            return None
        return (min if mode == "min" else max)(pts, key=lambda t: t[1])[0]

    best_acc_rank = _argopt("acc_mean", "max")
    best_ece_rank = _argopt("ece_evid_mean", "min")
    ranks = [r["rank"] for r in curve]
    return {
        "curve": curve,
        "best_accuracy_rank": best_acc_rank,
        "best_ece_rank": best_ece_rank,
        "budgets_differ": (best_acc_rank != best_ece_rank),
        "ece_optimum_is_interior": (
            best_ece_rank not in (min(ranks), max(ranks)) if ranks else None),
    }
