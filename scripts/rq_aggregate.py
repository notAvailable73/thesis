"""Aggregation — RQ1's 2x4 factorial + variance attribution, RQ2's
before/after, RQ5's rank curves, RQ3's matched-budget adjudication.

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


# =====================================================================
# RQ3 — matched budget (docs/RQ3_MATCHED_BUDGET_PLAN.md §5/§6 Step 5)
# =====================================================================
# The aggregation and pre-registered adjudication for the matched-budget
# experiment. It lives here rather than in scripts/rq3_matched.py because
# the plan's §6 Step 5 asks for this module to be EXTENDED rather than for
# a parallel aggregator to be written; rq3_matched.py imports these names
# and keeps only the experiment's design constants and run harness.
#
# Everything below reads only the per-cell JSONs the driver wrote, so the
# verdict can be rebuilt locally, without a GPU, from a committed results
# directory.
#
# The RQ3M_ prefix is load-bearing: RQ3's near-OOD pool set is NOT RQ1's
# (no cifar100_near — the experiment is MiniImageNet-only), so these must
# not shadow the module-level NEAR_POOLS/FAR_POOLS above.

#: The design constants the adjudication depends on. They are duplicated
#: from scripts/rq3_matched.py's DESIGN by necessity (that module imports
#: this one, not the reverse); rq3_matched.py asserts at import time that
#: the two agree, so they cannot drift.
RQ3M_DATASET = "mini_imagenet"
RQ3M_K_SHOT = 5               # base.yaml's default; never overridden (§2.3)
RQ3M_BACKBONES = ("resnet18", "mobilenetv3_small")
RQ3M_HEADS = ("evidential", "softmax")
RQ3M_SEEDS = (42, 43, 44)

#: §7 — a level whose two arms differ by more than this is not "matched"
#: and the experiment is void. The binding case is MobileNetV3 Level L
#: (6,928 vs 6,720 = 3.10% of the smaller arm).
RQ3M_MAX_BUDGET_MISMATCH_PCT = 3.1

#: §5 — the unmatched ΔECE baselines from the existing grid (MiniImageNet
#: 5-shot), PRE-REGISTERED here so the collapse criterion cannot be computed
#: against a number chosen after seeing the result. `rq3m_unmatched_deltas()`
#: recomputes them from results/mvt_results.json and asserts they agree.
RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE = {
    ("resnet18", "evidential"): +0.1111,
    ("resnet18", "softmax"): +0.1080,
    ("mobilenetv3_small", "evidential"): -0.0124,
    ("mobilenetv3_small", "softmax"): -0.0152,
}

#: §5 — the collapse criterion for H3.2 and the power threshold both rules
#: use.
RQ3M_COLLAPSE_RATIO_MAX = 0.50
RQ3M_SIGMA_MULTIPLE = 2.0
RQ3M_MIN_CELLS_FIRING = 3     # of the 4 (backbone x head) cells


def budget_mismatch_pct(a: int, b: int) -> float:
    """Percent by which the larger arm exceeds the smaller. Denominator is
    the SMALLER count — the conservative reading, and the one
    RQ3M_MAX_BUDGET_MISMATCH_PCT is calibrated against."""
    return abs(a - b) / min(a, b) * 100.0

#: MiniImageNet's near-OOD pools (rq_core.build_ood_pools inserts them in this
#: order). RQ3's hypotheses concern near-OOD only; far-OOD is reported as a
#: control, not as evidence either way (§5 "pre-stated scope on OOD pools").
RQ3M_NEAR_POOLS = ("mini_near", "tin_near")
RQ3M_FAR_POOLS = ("svhn_far", "gaussian_far")

#: The probability set and OOD score each head is read under — the SAME
#: "native" convention results/mvt_results.json used, so a matched ΔECE is
#: comparable to the unmatched ΔECE it is being contrasted with.
RQ3M_NATIVE_PROB_SET = {"evidential": "evidential_native", "softmax": "softmax"}
RQ3M_NATIVE_OOD_SCORE = {"evidential": "vacuity_native", "softmax": "msp"}
#: The mvt_results.json spelling of the same score (Step 10 wrote the trained
#: affine's vacuity simply as `vacuity`).
RQ3M_MVT_OOD_SCORE = {"evidential": "vacuity", "softmax": "msp"}

#: Sample SD (ddof=1) throughout. n=3 seeds, so this is ~22% larger than the
#: population SD results/mvt_results.json reports; it is the conservative
#: choice for a power check and is applied identically to every rule, in both
#: directions. Stated here because the threshold is pre-registered and the
#: convention must not be chosen after seeing the numbers.
RQ3M_SD_DDOF = 1


def _rq3m_sd(values) -> float:
    v = [float(x) for x in values]
    if len(v) < 2:
        return float("nan")
    return float(np.std(v, ddof=RQ3M_SD_DDOF))


def rq3m_cell_metrics(rec: dict) -> dict:
    """The three outcomes, read off one per-cell factorial record.

    ECE is the POOLED ECE under the head's native probability set — the exact
    quantity `results/mvt_results.json`'s `ece_pooled` holds, which is what the
    unmatched baselines in §5 are.
    """
    s = rec["summary"]
    head = rec["meta"].get("head") or rec["interpretation"]
    pset = RQ3M_NATIVE_PROB_SET[head]
    score = RQ3M_NATIVE_OOD_SCORE[head]
    near = [s[f"ood_auroc__{p}__{score}"] for p in RQ3M_NEAR_POOLS
            if f"ood_auroc__{p}__{score}" in s]
    far = [s[f"ood_auroc__{p}__{score}"] for p in RQ3M_FAR_POOLS
           if f"ood_auroc__{p}__{score}" in s]
    out = {
        "ece": float(s[f"ece_pooled__{pset}"]),
        "accuracy": float(s[f"accuracy_mean__{pset}"]),
        "auroc_near": float(np.mean(near)) if near else float("nan"),
        "auroc_far": float(np.mean(far)) if far else float("nan"),
        "n_params": int(rec["n_params"]),
    }
    for p in RQ3M_NEAR_POOLS + RQ3M_FAR_POOLS:
        k = f"ood_auroc__{p}__{score}"
        if k in s:
            out[f"auroc__{p}"] = float(s[k])
    return out


def rq3m_arm_summary(recs: list[dict]) -> dict:
    """Mean / SD across seeds for one arm of one (backbone, level, head)."""
    per_seed = {int(r["seed"]): rq3m_cell_metrics(r) for r in recs}
    seeds = sorted(per_seed)
    out = {"n_seeds": len(seeds), "seeds": seeds,
           "n_params": per_seed[seeds[0]]["n_params"] if seeds else None,
           "per_seed": {str(s): per_seed[s] for s in seeds}}
    for m in ("ece", "accuracy", "auroc_near", "auroc_far"):
        vals = [per_seed[s][m] for s in seeds]
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_sd"] = _rq3m_sd(vals)
    return out


def rq3m_unmatched_deltas(repo_root: Path, *, tol: float = 1.5e-3) -> dict:
    """§5's unmatched baselines, RECOMPUTED from results/mvt_results.json.

    Recomputed rather than trusted so the collapse criterion is anchored to the
    committed grid rather than to a transcription; the pre-registered values are
    then asserted against it, which catches both a typo in the plan and a
    changed grid file. Returns per (backbone, head) the unmatched ΔECE, ΔAcc and
    Δ near-OOD AUROC, all as LoRA minus bottleneck (the same sign convention the
    matched deltas use).
    """
    path = repo_root / "results" / "mvt_results.json"
    if not path.exists():
        return {"available": False, "reason": f"{path} not found",
                "delta_ece": {f"{b}/{h}": v for (b, h), v
                              in RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE.items()},
                "source": "pre-registered (plan §5)"}
    grid = json.load(open(path))["results"][RQ3M_DATASET][f"{RQ3M_K_SHOT}shot"]

    out, mismatches = {}, []
    for backbone in RQ3M_BACKBONES:
        for head in RQ3M_HEADS:
            try:
                btl = grid[backbone]["bottleneck_parallel"][head]
                lora = grid[backbone]["lora"][head]
            except KeyError:
                continue
            sc = RQ3M_MVT_OOD_SCORE[head]

            def _near(cell):
                vals = [cell[f"ood_auroc__{p}__{sc}"]["mean"] for p in RQ3M_NEAR_POOLS
                        if f"ood_auroc__{p}__{sc}" in cell]
                return float(np.mean(vals)) if vals else float("nan")

            row = {
                "delta_ece": float(lora["ece_pooled"]["mean"] - btl["ece_pooled"]["mean"]),
                "delta_accuracy": float(lora["accuracy_mean"]["mean"]
                                        - btl["accuracy_mean"]["mean"]),
                "delta_auroc_near": float(_near(lora) - _near(btl)),
                "btl_params": int(btl["n_params"]["mean"]),
                "lora_params": int(lora["n_params"]["mean"]),
                "btl_ece": float(btl["ece_pooled"]["mean"]),
                "lora_ece": float(lora["ece_pooled"]["mean"]),
            }
            row["budget_mismatch_pct"] = round(
                budget_mismatch_pct(row["btl_params"], row["lora_params"]), 2)
            out[f"{backbone}/{head}"] = row

            pre = RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE.get((backbone, head))
            if pre is not None and abs(pre - row["delta_ece"]) > tol:
                mismatches.append(
                    f"{backbone}/{head}: plan §5 says {pre:+.4f}, "
                    f"mvt_results.json gives {row['delta_ece']:+.4f}")

    return {"available": True, "source": str(path.relative_to(repo_root)),
            "per_cell": out, "preregistration_ok": not mismatches,
            "preregistration_mismatches": mismatches}


def rq3m_matched_table(recs: list[dict]) -> dict:
    """Per (backbone, level, head): both arms, the three deltas, and σ.

    Sign convention throughout: **LoRA minus bottleneck**, matching §5's
    unmatched table, so a positive ΔECE means LoRA is worse calibrated on both
    sides of the comparison.
    """
    by_level: dict[tuple, dict] = {}
    for r in recs:
        m = r["meta"]
        key = (m["backbone"], m["level"], m["head"])
        by_level.setdefault(key, {}).setdefault(m["arm"], []).append(r)

    rows = {}
    for (backbone, level, head), arms in sorted(by_level.items()):
        row = {"backbone": backbone, "level": level, "head": head,
               "arms": {a: rq3m_arm_summary(rs) for a, rs in sorted(arms.items())}}
        have_both = set(row["arms"]) == {"btl", "lora"}
        row["complete"] = bool(
            have_both and all(row["arms"][a]["n_seeds"] == len(RQ3M_SEEDS)
                              for a in ("btl", "lora")))
        if have_both:
            btl, lora = row["arms"]["btl"], row["arms"]["lora"]
            row["btl_rank"] = arms["btl"][0]["meta"]["rank"]
            row["lora_rank"] = arms["lora"][0]["meta"]["rank"]
            row["budget_mismatch_pct"] = round(
                budget_mismatch_pct(btl["n_params"], lora["n_params"]), 4)
            row["budget_matched"] = (
                row["budget_mismatch_pct"] <= RQ3M_MAX_BUDGET_MISMATCH_PCT)
            for m in ("ece", "accuracy", "auroc_near", "auroc_far"):
                d = lora[f"{m}_mean"] - btl[f"{m}_mean"]
                # Pooled across-seed SD of the two arms: the scale the
                # pre-registered 2σ power check is expressed in.
                sigma = float(np.sqrt(
                    (btl[f"{m}_sd"] ** 2 + lora[f"{m}_sd"] ** 2) / 2.0))
                row[f"delta_{m}"] = float(d)
                row[f"sigma_{m}"] = sigma
                row[f"significant_{m}"] = bool(abs(d) > RQ3M_SIGMA_MULTIPLE * sigma)
        rows[f"{backbone}/{level}/{head}"] = row
    return rows


RQ3M_DECISION_RULES = {
    "budget": ("H3.2 — the sign flip disappears and gaps collapse toward zero "
               "on BOTH backbones: mean|ΔECE_matched| <= 50% of "
               "|ΔECE_unmatched| AND |ΔECE_matched| within 2σ of zero, in >= 3 "
               "of 4 (backbone x head) cells."),
    "architecture": ("H3.1 — one architecture is better calibrated regardless "
                     "of budget: ΔECE_matched keeps a CONSISTENT SIGN across "
                     "both backbones with |ΔECE| > 2σ, in >= 3 of 4 cells."),
    "backbone_intrinsic": ("H3.2-alt — the original per-backbone signs persist "
                           "at matched budget: ΔECE_matched retains its "
                           "unmatched sign per backbone (+ on ResNet-18, − on "
                           "MobileNetV3) with |ΔECE| > 2σ, in >= 3 of 4 cells."),
    "inconclusive": ("No rule fired, or more than one did. Recorded as-is: a "
                     "null is not evidence for the hypothesis that predicts a "
                     "null (§5)."),
}


def rq3m_adjudicate(table: dict, unmatched: dict) -> dict:
    """Apply §5's pre-registered rule. No post-hoc adjustment, by construction.

    The unit of decision is a (backbone x head) CELL — the two budget levels
    inside a cell are averaged, which is what §5's "mean |ΔECE_matched|" means
    and why the count is out of 4 rather than 8.
    """
    cells: dict[str, dict] = {}
    for key, row in table.items():
        if "delta_ece" not in row:
            continue
        ck = f"{row['backbone']}/{row['head']}"
        c = cells.setdefault(ck, {"backbone": row["backbone"],
                                  "head": row["head"], "levels": {},
                                  "complete": True})
        c["levels"][row["level"]] = {"delta_ece": row["delta_ece"],
                                     "sigma_ece": row["sigma_ece"],
                                     "delta_accuracy": row["delta_accuracy"],
                                     "sigma_accuracy": row["sigma_accuracy"],
                                     "delta_auroc_near": row["delta_auroc_near"],
                                     "sigma_auroc_near": row["sigma_auroc_near"],
                                     "complete": row["complete"]}
        c["complete"] &= bool(row["complete"])

    src = (unmatched.get("per_cell") or {})
    for ck, c in cells.items():
        lv = list(c["levels"].values())
        c["n_levels"] = len(lv)
        c["delta_ece_matched"] = float(np.mean([x["delta_ece"] for x in lv]))
        c["abs_delta_ece_matched"] = float(np.mean([abs(x["delta_ece"]) for x in lv]))
        c["sigma_ece"] = float(np.mean([x["sigma_ece"] for x in lv]))
        c["delta_accuracy_matched"] = float(np.mean([x["delta_accuracy"] for x in lv]))
        c["sigma_accuracy"] = float(np.mean([x["sigma_accuracy"] for x in lv]))
        c["delta_auroc_near_matched"] = float(np.mean([x["delta_auroc_near"] for x in lv]))
        c["sigma_auroc_near"] = float(np.mean([x["sigma_auroc_near"] for x in lv]))

        pre = RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE[(c["backbone"], c["head"])]
        unm = float(src.get(ck, {}).get("delta_ece", pre))
        c["delta_ece_unmatched"] = unm
        c["unmatched_sign"] = int(np.sign(unm))
        c["collapse_ratio"] = (c["abs_delta_ece_matched"] / abs(unm)
                               if unm else float("inf"))
        c["within_2sigma"] = bool(
            abs(c["delta_ece_matched"]) <= RQ3M_SIGMA_MULTIPLE * c["sigma_ece"])
        c["significant"] = bool(
            abs(c["delta_ece_matched"]) > RQ3M_SIGMA_MULTIPLE * c["sigma_ece"])
        c["sign"] = int(np.sign(c["delta_ece_matched"]))
        c["fires_budget"] = bool(c["collapse_ratio"] <= RQ3M_COLLAPSE_RATIO_MAX
                                 and c["within_2sigma"])
        c["fires_backbone_intrinsic"] = bool(c["significant"]
                                             and c["sign"] == c["unmatched_sign"])

    n_budget = sum(1 for c in cells.values() if c["fires_budget"])
    sig = [c for c in cells.values() if c["significant"]]
    signs = {c["sign"] for c in sig}
    backbones_covered = {c["backbone"] for c in sig}
    architecture_consistent = (len(signs) == 1 and len(backbones_covered) == 2)
    for c in cells.values():
        c["fires_architecture"] = bool(c["significant"] and architecture_consistent)
    n_architecture = sum(1 for c in cells.values() if c["fires_architecture"])
    n_backbone = sum(1 for c in cells.values() if c["fires_backbone_intrinsic"])

    fired = {
        "budget": n_budget >= RQ3M_MIN_CELLS_FIRING,
        "architecture": n_architecture >= RQ3M_MIN_CELLS_FIRING,
        "backbone_intrinsic": n_backbone >= RQ3M_MIN_CELLS_FIRING,
    }
    winners = [k for k, v in fired.items() if v]
    complete = bool(cells) and all(c["complete"] for c in cells.values()) \
        and len(cells) == len(RQ3M_BACKBONES) * len(RQ3M_HEADS)

    if not complete:
        verdict, reason = "inconclusive", (
            "the design is not complete — not every (backbone, level, head) "
            "cell has both arms at all 3 seeds. Nothing is adjudicated from a "
            "partial run; resume the sweep.")
    elif len(winners) == 1:
        verdict, reason = winners[0], RQ3M_DECISION_RULES[winners[0]]
    elif not winners:
        verdict, reason = "inconclusive", (
            "no pre-registered rule fired. Record it as inconclusive — do not "
            "narrate a preferred story from a null (§5).")
    else:
        verdict, reason = "inconclusive", (
            f"more than one rule fired ({winners}), which the pre-registration "
            f"does not resolve. Report the cell table, not a verdict.")

    return {
        "verdict": verdict,
        "reason": reason,
        "complete": complete,
        "cells": cells,
        "counts": {"budget": n_budget, "architecture": n_architecture,
                   "backbone_intrinsic": n_backbone,
                   "min_cells_to_fire": RQ3M_MIN_CELLS_FIRING,
                   "n_cells": len(cells)},
        "rules_fired": fired,
        "architecture_sign_consistent": architecture_consistent,
        "decision_rules": RQ3M_DECISION_RULES,
        "thresholds": {"collapse_ratio_max": RQ3M_COLLAPSE_RATIO_MAX,
                       "sigma_multiple": RQ3M_SIGMA_MULTIPLE,
                       "sd_ddof": RQ3M_SD_DDOF},
    }


def rq3m_secondary_outcomes(table: dict, unmatched: dict) -> dict:
    """§5's secondary tests: accuracy (a live risk to H3.1) and near-OOD AUROC.

    H3.1 predicts bottleneck still wins accuracy at matched budget on BOTH
    backbones. If the accuracy gap collapses too, then accuracy was tracking
    budget as well and RQ3's headline dissociation weakens substantially — so
    the collapse ratio is reported for accuracy exactly as it is for ECE,
    whichever way it falls.
    """
    src = unmatched.get("per_cell") or {}
    out = {}
    for metric in ("accuracy", "auroc_near"):
        rows = [r for r in table.values() if f"delta_{metric}" in r]
        btl_wins = sum(1 for r in rows if r[f"delta_{metric}"] < 0)
        sig_btl = sum(1 for r in rows
                      if r[f"delta_{metric}"] < 0 and r[f"significant_{metric}"])
        per_cell = {}
        for r in rows:
            ck = f"{r['backbone']}/{r['head']}"
            per_cell.setdefault(ck, []).append(abs(r[f"delta_{metric}"]))
        ratios = {}
        for ck, vals in per_cell.items():
            unm = abs(float(src.get(ck, {}).get(f"delta_{metric}", float("nan"))))
            ratios[ck] = (float(np.mean(vals)) / unm) if unm else float("nan")
        collapsed = [k for k, v in ratios.items()
                     if v == v and v <= RQ3M_COLLAPSE_RATIO_MAX]
        out[metric] = {
            "n_comparisons": len(rows),
            "bottleneck_wins": btl_wins,
            "bottleneck_wins_beyond_2sigma": sig_btl,
            "collapse_ratio_per_cell": ratios,
            "cells_collapsed": sorted(collapsed),
            "architecture_holds": bool(rows) and btl_wins == len(rows),
        }
    a = out["accuracy"]
    out["accuracy"]["interpretation"] = (
        "bottleneck still wins accuracy in every matched comparison — the "
        "architecture half of RQ3 survives budget equalisation."
        if a["architecture_holds"] else
        f"bottleneck wins accuracy in only {a['bottleneck_wins']}/"
        f"{a['n_comparisons']} matched comparisons — the accuracy half of RQ3 "
        f"does NOT survive budget equalisation, and RQ3's dissociation weakens.")
    return out
