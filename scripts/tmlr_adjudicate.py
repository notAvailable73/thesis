"""TMLR follow-up — the pre-specified decision rules (docs/TMLR_PRESPEC.md, plan A9).

This file IS the pre-specification in executable form. It is committed and
pushed before the first real wave; any later change is logged in the prespec's
changelog and reported in the paper.

Conventions (identical to the thesis's RQ3 rule, scripts/rq_aggregate.py):
  - sigma = pooled across-seed SD of the arms involved, sample SD (ddof=1);
    an effect FIRES when |effect| > SIGMA_MULTIPLE * sigma.
  - every arm difference is LoRA-type minus bottleneck-type, so a positive
    ECE difference means the LoRA arm is worse calibrated.
  - a cell = (dataset, backbone, level, head). Rules that average levels average
    L and H inside a (backbone, head) cell, exactly as the thesis did.
  - an incomplete design is never adjudicated: verdict "inconclusive".
Records are matched to design slots by ATTRIBUTES (dataset, k_shot, backbone,
arm, rank, head, seed, stream), never by experiment, so runs shared between
experiments (deduplicated in tmlr_design) are found wherever they live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tmlr_design as T  # noqa: E402

SIGMA_MULTIPLE = 2.0
SD_DDOF = 1
COLLAPSE_RATIO_MAX = 0.50
SHARE_FIRE = 0.75          # R-A: an effect "holds" in >= 75% of cells ...
SHARE_QUIET = 0.25         # ... while the other fires in <= 25%
FAMILY_SIGN = {"resnet18": 1, "resnet34": 1, "mobilenetv3_small": -1, "mobilenetv2": -1}
NATIVE_PSET = {"evidential": "evidential_native", "softmax": "softmax"}
NATIVE_SCORE = {"evidential": "vacuity_native", "softmax": "msp"}
NEAR_POOL = {"mini_imagenet": "mini_near", "fgvc_aircraft": "aircraft_near",
             "cifar_fs": "cifar100_near"}
FAR_POOLS = ("svhn_far", "gaussian_far")
RQ2_SCORES = ("msp", "ts_msp", "energy", "vacuity_valfit")
SEEDS = {"mini_imagenet": T.SEEDS_5, "fgvc_aircraft": T.SEEDS_3}


# =====================================================================
# Records
# =====================================================================
def outcomes_from_summary(head: str, dataset: str, summary: dict) -> dict:
    pset, score = NATIVE_PSET[head], NATIVE_SCORE[head]
    near = [summary.get(f"ood_auroc__{p}__{score}") for p in (NEAR_POOL[dataset], "tin_near")]
    far = [summary.get(f"ood_auroc__{p}__{score}") for p in FAR_POOLS]
    near = [x for x in near if x is not None]
    far = [x for x in far if x is not None]
    return {"accuracy": summary.get(f"accuracy_mean__{pset}"),
            "ece": summary.get(f"ece_pooled__{pset}"),
            "auroc_near": float(np.mean(near)) if near else None,
            "auroc_far": float(np.mean(far)) if far else None}


def load_records(results_root: Path = REPO / "results/tmlr") -> list[dict]:
    recs = []
    for s in T.build_design():
        p = results_root / s.experiment / s.run_id / "metrics.json"
        if not p.exists() or s.mode != "test":
            continue
        summary = json.load(open(p))["summary"]
        recs.append({"run_id": s.run_id, "experiment": s.experiment, "dataset": s.dataset,
                     "k_shot": s.k_shot, "backbone": s.backbone, "arm": s.arm,
                     "rank_spec": s.rank_spec, "head": s.head, "seed": s.seed,
                     "vary_stream": s.vary_stream, "loss": dict(s.loss_overrides),
                     "outcomes": outcomes_from_summary(s.head, s.dataset, summary),
                     "summary": summary})
    return recs


def _rank_key(r):
    return tuple(r) if isinstance(r, (list, tuple)) else int(r)


def _stream(seed: int, vary: bool) -> str:
    return "vary" if (vary and seed != 42) else "fixed"


def make_index(records: list[dict]) -> dict:
    idx = {}
    for r in records:
        if r.get("loss"):
            continue        # E5 recipe runs are only read by R-F
        key = (r["dataset"], r["k_shot"], r["backbone"], r["arm"], _rank_key(r["rank_spec"]),
               r["head"], r["seed"], _stream(r["seed"], r["vary_stream"]))
        idx[key] = r
    return idx


def arm_stats_from_values(values) -> dict:
    v = [float(x) for x in values]
    return {"mean": float(np.mean(v)), "n": len(v), "values": v,
            "sd": float(np.std(v, ddof=SD_DDOF)) if len(v) > 1 else float("nan")}


def arm_stats(idx, dataset, backbone, arm, rank, head, outcome, seeds, *, vary=True, k=5):
    vals = []
    for seed in seeds:
        r = idx.get((dataset, k, backbone, arm, _rank_key(rank), head, seed, _stream(seed, vary)))
        if r is None or r["outcomes"].get(outcome) is None:
            return None
        vals.append(r["outcomes"][outcome])
    return arm_stats_from_values(vals)


def _pooled_sigma(stats) -> float:
    return float(np.sqrt(np.mean([s["sd"] ** 2 for s in stats])))


# =====================================================================
# R-A — sites vs module (accuracy, near-OOD AUROC)
# =====================================================================
def verdict_RA(records, outcome: str, dataset: str, *, backbones=None, seeds=None) -> dict:
    idx = make_index(records)
    seeds = seeds or SEEDS[dataset]
    backbones = backbones or tuple(dict.fromkeys(b for (b, _l) in T.LEVELS))
    cells, complete = {}, True
    for bb in backbones:
        for lvl in ("L", "H"):
            for head in T.HEADS:
                st = {a: arm_stats(idx, dataset, bb, a, rk, head, outcome, seeds)
                      for a, rk in T.LEVELS[(bb, lvl)].items()}
                if any(v is None for v in st.values()):
                    complete = False
                    continue
                m = {a: v["mean"] for a, v in st.items()}
                sigma = _pooled_sigma(st.values())
                S = 0.5 * ((m["btl_ms"] - m["btl_ss"]) + (m["lora_ms"] - m["lora_ss"]))
                M = 0.5 * ((m["btl_ms"] - m["lora_ms"]) + (m["btl_ss"] - m["lora_ss"]))
                I = 0.5 * ((m["btl_ms"] - m["btl_ss"]) - (m["lora_ms"] - m["lora_ss"]))
                lim = SIGMA_MULTIPLE * sigma
                cells[f"{bb}/{lvl}/{head}"] = {
                    "site": S, "module": M, "interaction": I, "sigma": sigma,
                    "site_fires": abs(S) > lim, "module_fires": abs(M) > lim,
                    "interaction_fires": abs(I) > lim, "means": m}
    n = len(cells)
    c = {"n_cells": n,
         "site_pos": sum(1 for x in cells.values() if x["site_fires"] and x["site"] > 0),
         "site_any": sum(1 for x in cells.values() if x["site_fires"]),
         "module_pos": sum(1 for x in cells.values() if x["module_fires"] and x["module"] > 0),
         "module_any": sum(1 for x in cells.values() if x["module_fires"]),
         "interaction_any": sum(1 for x in cells.values() if x["interaction_fires"])}
    site = c["site_pos"] >= SHARE_FIRE * n
    module = c["module_pos"] >= SHARE_FIRE * n
    if not complete or n == 0:
        verdict = "inconclusive"
    elif site and module:
        verdict = "both"
    elif site and c["module_any"] <= SHARE_QUIET * n:
        verdict = "site"
    elif module and c["site_any"] <= SHARE_QUIET * n:
        verdict = "module"
    else:
        verdict = "inconclusive"
    return {"rule": "R-A", "outcome": outcome, "dataset": dataset, "verdict": verdict,
            "complete": complete, "counts": c, "cells": cells}


# =====================================================================
# R-B — budget / architecture / backbone (ECE, original pair btl_ms vs lora_ss)
# =====================================================================
def _pair_cells(idx, dataset, backbones, seeds, vary, lora_arm, unmatched_override):
    cells, complete = {}, True
    for bb in backbones:
        for head in T.HEADS:
            lv = []
            for lvl in ("L", "H"):
                arms = T.LEVELS[(bb, lvl)]
                b = arm_stats(idx, dataset, bb, "btl_ms", arms["btl_ms"], head, "ece", seeds, vary=vary)
                l = arm_stats(idx, dataset, bb, lora_arm, arms[lora_arm], head, "ece", seeds, vary=vary)
                if b is None or l is None:
                    complete = False
                    break
                lv.append({"delta": l["mean"] - b["mean"], "sigma": _pooled_sigma([b, l])})
            if len(lv) != 2:
                continue
            b16 = arm_stats(idx, dataset, bb, "btl_ms", 16, head, "ece", seeds, vary=vary)
            l16 = arm_stats(idx, dataset, bb, "lora_ss", 16, head, "ece", seeds, vary=vary)
            if b16 is not None and l16 is not None:
                unm = l16["mean"] - b16["mean"]
            elif unmatched_override and (bb, head) in unmatched_override:
                unm = float(unmatched_override[(bb, head)])
            else:
                complete = False
                continue
            delta = float(np.mean([x["delta"] for x in lv]))
            sigma = float(np.mean([x["sigma"] for x in lv]))
            absd = float(np.mean([abs(x["delta"]) for x in lv]))
            cells[f"{bb}/{head}"] = {
                "backbone": bb, "head": head, "delta_matched": delta, "abs_delta_matched": absd,
                "sigma": sigma, "delta_unmatched": unm,
                "collapse_ratio": absd / abs(unm) if unm else float("inf"),
                "significant": abs(delta) > SIGMA_MULTIPLE * sigma,
                "within_2sigma": abs(delta) <= SIGMA_MULTIPLE * sigma,
                "sign": int(np.sign(delta)), "unmatched_sign": int(np.sign(unm))}
    return cells, complete


def verdict_RB(records, backbones, dataset: str, *, min_cells: int, seeds=None, vary=True,
               unmatched_override: dict | None = None) -> dict:
    idx = make_index(records)
    seeds = seeds or SEEDS[dataset]
    cells, complete = _pair_cells(idx, dataset, backbones, seeds, vary, "lora_ss",
                                  unmatched_override)
    complete = complete and len(cells) == len(backbones) * len(T.HEADS)
    for c in cells.values():
        c["fires_budget"] = c["collapse_ratio"] <= COLLAPSE_RATIO_MAX and c["within_2sigma"]
        c["fires_backbone_intrinsic"] = c["significant"] and c["sign"] == c["unmatched_sign"]
    sig = [c for c in cells.values() if c["significant"]]
    consistent = len({c["sign"] for c in sig}) == 1 and {c["backbone"] for c in sig} == set(backbones)
    for c in cells.values():
        c["fires_architecture"] = c["significant"] and consistent
    counts = {k: sum(1 for c in cells.values() if c[f"fires_{k}"])
              for k in ("budget", "architecture", "backbone_intrinsic")}
    winners = [k for k, n in counts.items() if n >= min_cells]
    verdict = winners[0] if (complete and len(winners) == 1) else "inconclusive"
    return {"rule": "R-B", "dataset": dataset, "backbones": list(backbones), "verdict": verdict,
            "complete": complete, "counts": {**counts, "min_cells": min_cells, "n_cells": len(cells)},
            "cells": cells}


# =====================================================================
# R-C — does LoRA's site role explain the flip? (lora_ms vs btl_ms, ECE)
# =====================================================================
def verdict_RC(records, dataset: str, *, backbones=tuple(FAMILY_SIGN), min_cells: int = 6) -> dict:
    idx = make_index(records)
    cells, complete = _pair_cells(idx, dataset, backbones, SEEDS[dataset], True, "lora_ms",
                                  unmatched_override={(b, h): 0.0 for b in backbones for h in T.HEADS})
    complete = complete and len(cells) == len(backbones) * len(T.HEADS)
    sig = [c for c in cells.values() if c["significant"]]
    one_sign = len(sig) >= min_cells and len({c["sign"] for c in sig}) == 1
    quiet = sum(1 for c in cells.values() if c["within_2sigma"]) >= min_cells
    persists = sum(1 for c in sig if c["sign"] == FAMILY_SIGN[c["backbone"]]) >= min_cells
    if not complete:
        verdict = "inconclusive"
    elif one_sign or quiet:
        verdict = "site_role_explains"
    elif persists:
        verdict = "flip_persists"
    else:
        verdict = "inconclusive"
    return {"rule": "R-C", "dataset": dataset, "verdict": verdict, "complete": complete,
            "counts": {"significant": len(sig), "one_sign": one_sign, "within_2sigma_cells":
                       sum(1 for c in cells.values() if c["within_2sigma"]),
                       "family_pattern_cells": sum(1 for c in sig if c["sign"] == FAMILY_SIGN[c["backbone"]])},
            "cells": {k: {kk: v for kk, v in c.items() if kk != "delta_unmatched"}
                      for k, c in cells.items()}}


# =====================================================================
# R-D — family consistency of the original pair's calibration sign
# =====================================================================
def verdict_RD(records, dataset: str = "mini_imagenet") -> dict:
    rb = verdict_RB(records, tuple(FAMILY_SIGN), dataset, min_cells=6)
    cells = rb["cells"]
    checks = []
    for new, ref in (("resnet34", "resnet18"), ("mobilenetv2", "mobilenetv3_small")):
        for head in T.HEADS:
            a, b = cells.get(f"{new}/{head}"), cells.get(f"{ref}/{head}")
            ok = bool(a and b and a["significant"] and b["significant"] and a["sign"] == b["sign"])
            checks.append({"new": new, "ref": ref, "head": head, "pass": ok})
    n = sum(c["pass"] for c in checks)
    verdict = ("inconclusive" if not rb["complete"] else "family_consistent" if n == 4
               else "not_family_consistent" if n <= 2 else "partial")
    return {"rule": "R-D", "verdict": verdict, "passed": n, "checks": checks,
            "complete": rb["complete"]}


# =====================================================================
# Variance shares (balanced main effects)
# =====================================================================
def eta_squared(rows: list[dict], y: str, factors) -> dict:
    vals = np.array([r[y] for r in rows], dtype=float)
    grand, ss_tot = vals.mean(), float(((vals - vals.mean()) ** 2).sum())
    out = {}
    for f in factors:
        ss = 0.0
        for lvl in sorted({r[f] for r in rows}, key=str):
            g = np.array([r[y] for r in rows if r[f] == lvl], dtype=float)
            ss += len(g) * (g.mean() - grand) ** 2
        out[f] = ss / ss_tot if ss_tot else float("nan")
    return out


# =====================================================================
# R-E — Aircraft replication (E3a) ; R-F — alternative EDL recipe (E5)
# =====================================================================
def verdict_RE(records) -> dict:
    air = [r for r in records if r["experiment"] == "e3a"]
    expected = 2 * 2 * 2 * 2 * len(T.SEEDS_3)
    if len(air) != expected:
        return {"rule": "R-E", "verdict": "inconclusive", "complete": False, "n": len(air)}
    rows = [{"shots": r["k_shot"], "backbone": r["backbone"], "adapter": r["arm"],
             "head": r["head"], "ece": r["outcomes"]["ece"]} for r in air]
    eta_ece = eta_squared(rows, "ece", ("shots", "backbone", "adapter", "head"))
    head_largest = max(eta_ece, key=eta_ece.get) == "head"
    rq2 = {}
    for kind, pools in (("far", FAR_POOLS), ("near", ("aircraft_near", "tin_near"))):
        rr = [{"score": sc, "objective": r["head"], "auroc": r["summary"][f"ood_auroc__{p}__{sc}"]}
              for r in air for sc in RQ2_SCORES for p in pools]
        rq2[kind] = eta_squared(rr, "auroc", ("score", "objective"))
    rq2_ok = all(v["score"] > v["objective"] and v["objective"] < 0.02 for v in rq2.values())
    evid = [r for r in air if r["head"] == "evidential"]
    improved = sum(1 for r in evid if r["summary"]["ece_pooled__evidential_valfit"]
                   < r["summary"]["ece_pooled__evidential_native"])
    refit_ok = improved >= 0.9 * len(evid)
    wins = 0
    for k in (1, 5):
        for bb in ("resnet18", "mobilenetv3_small"):
            for head in T.HEADS:
                acc = {a: np.mean([r["outcomes"]["accuracy"] for r in air if r["k_shot"] == k
                                   and r["backbone"] == bb and r["head"] == head and r["arm"] == a])
                       for a in ("btl_ms", "lora_ss")}
                wins += acc["btl_ms"] > acc["lora_ss"]
    acc_ok = wins >= 0.75 * 8
    checks = {"head_largest_ece_factor": head_largest, "score_dominates_objective": rq2_ok,
              "refit_improves_90pct": refit_ok, "btl_beats_lora_75pct": acc_ok}
    return {"rule": "R-E", "verdict": "replicated" if all(checks.values()) else "not_replicated",
            "complete": True, "checks": checks, "eta_ece": eta_ece, "rq2_eta": rq2,
            "refit_improved": f"{improved}/{len(evid)}", "accuracy_wins": f"{wins}/8"}


def grid_view(records) -> list[dict]:
    """The original 2^5 grid cells (btl_ms/lora_ss r16, seeds 42-44, fixed stream),
    from E4 or from their deduplicated E1 seed-42 twins."""
    out = []
    for r in records:
        if r["arm"] not in ("btl_ms", "lora_ss") or r["rank_spec"] != 16 or r.get("loss"):
            continue
        if r["seed"] not in T.SEEDS_3 or _stream(r["seed"], r["vary_stream"]) != "fixed":
            continue
        if r["dataset"] in ("cifar_fs", "mini_imagenet") and r["experiment"] in ("e4", "e1"):
            out.append(r)
    return out


def verdict_RF(records, e5_selection: dict | None) -> dict:
    if e5_selection and e5_selection.get("skipped"):
        return {"rule": "R-F", "verdict": "not_run", "reason": "current recipe VAL-optimal"}
    grid = [r for r in grid_view(records) if r["head"] == "softmax"]
    e5 = [r for r in records if r["experiment"] == "e5"]
    if len(grid) != 48 or len(e5) != 48:
        return {"rule": "R-F", "verdict": "inconclusive", "complete": False,
                "n_softmax": len(grid), "n_e5": len(e5)}
    rows = [{"dataset": r["dataset"], "shots": r["k_shot"], "backbone": r["backbone"],
             "adapter": r["arm"], "head": r["head"], "ece": r["outcomes"]["ece"]} for r in grid + e5]
    eta = eta_squared(rows, "ece", ("dataset", "shots", "backbone", "adapter", "head"))
    worse = 0
    for d in ("cifar_fs", "mini_imagenet"):
        for k in (1, 5):
            for bb in ("resnet18", "mobilenetv3_small"):
                for a in ("btl_ms", "lora_ss"):
                    def m(src):
                        return np.mean([r["outcomes"]["ece"] for r in src if r["dataset"] == d
                                        and r["k_shot"] == k and r["backbone"] == bb and r["arm"] == a])
                    worse += m(e5) > m(grid)
    robust = max(eta, key=eta.get) == "head" and worse >= 0.75 * 16
    return {"rule": "R-F", "verdict": "head_dominance_robust" if robust else "not_robust",
            "complete": True, "eta_ece": eta, "evidential_worse_pairs": f"{worse}/16"}


def write_verdicts(dest: Path = REPO / "results/tmlr/verdicts.json") -> dict:
    import yaml
    recs = load_records()
    sel_p = REPO / "configs/tmlr/e5_selected.yaml"
    sel = yaml.safe_load(open(sel_p)) if sel_p.exists() else None
    v = {
        "RA_mini_accuracy": verdict_RA(recs, "accuracy", "mini_imagenet"),
        "RA_mini_near_ood": verdict_RA(recs, "auroc_near", "mini_imagenet"),
        "RA_air_accuracy": verdict_RA(recs, "accuracy", "fgvc_aircraft",
                                      backbones=("resnet18", "mobilenetv3_small")),
        "RA_air_near_ood": verdict_RA(recs, "auroc_near", "fgvc_aircraft",
                                      backbones=("resnet18", "mobilenetv3_small")),
        "RB_mini_r18_mnv3": verdict_RB(recs, ("resnet18", "mobilenetv3_small"), "mini_imagenet",
                                       min_cells=3),
        "RB_mini_all4": verdict_RB(recs, tuple(FAMILY_SIGN), "mini_imagenet", min_cells=6),
        "RB_air": verdict_RB(recs, ("resnet18", "mobilenetv3_small"), "fgvc_aircraft", min_cells=3),
        "RC_mini": verdict_RC(recs, "mini_imagenet"),
        "RD_mini": verdict_RD(recs, "mini_imagenet"),
        "RE_aircraft": verdict_RE(recs),
        "RF_edl_recipe": verdict_RF(recs, sel),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(v, indent=1, sort_keys=True, default=float))
    return v


if __name__ == "__main__":
    out = write_verdicts()
    for k, v in out.items():
        print(f"{k:<20} {v['verdict']}")
