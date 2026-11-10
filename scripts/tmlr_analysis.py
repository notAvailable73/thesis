"""TMLR follow-up — statistics upgrade, robustness checks and the verdicts
(plan Task 15; docs/superpowers/plans/2026-10-22-tmlr-publication.md).

Numbering: everything here uses the NEW four-RQ framing (RQ1 variance shares,
RQ2 objective vs score, RQ3 architecture vs budget, RQ4 post-hoc refit). The
pre-specified rules R-A ... R-F (docs/TMLR_PRESPEC.md) are adjudicated ONLY by
`tmlr_adjudicate.write_verdicts`, the authoritative executable form; nothing
in this module re-implements or overrides a rule.

What this module adds on top:
  - eta^2 / omega^2 with seed-bootstrap CIs for any balanced factorial
    (`eta_squared_with_ci`). Point estimates come from the thesis's own
    `rq_aggregate.eta_squared`; the bootstrap redraws the seeds of each
    configuration with replacement and holds the design fixed, exactly as
    `rq_completion.seed_bootstrap` does for the 2^5 grid. `rq_completion`'s
    OLS path is hard-wired to that 2-level 5-factor grid, so it cannot serve
    the Aircraft (4 factors) or RQ2 (multi-level design x score) designs;
    `test_e4_reproduces_thesis_rq1` pins this path to the thesis's numbers.
  - paired episode bootstrap over the 600 shared TEST episodes.
  - exact per-backbone sign tests (never pooled across backbones).
  - the reproduction gate of TMLR_PRESPEC section 6, which must pass before any
    verdict is written.

Usage:  uv run --frozen python scripts/tmlr_analysis.py
Writes: results/tmlr/analysis/*.json and results/tmlr/verdicts.json
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rq_aggregate as A  # noqa: E402
import rq_completion as C  # noqa: E402
import tmlr_adjudicate as J  # noqa: E402
import tmlr_design as T  # noqa: E402

TMLR = REPO / "results" / "tmlr"
OUT = TMLR / "analysis"
RQ1_FACTORS = C.FACTORS                     # ("dataset","k_shot","backbone","adapter","head")
RQ2_SCORES_THESIS = A.RQ1_SCORES            # msp, energy, ts_msp, vacuity_valfit (retired name)
RQ2_SCORES_SIX = ("msp", "ts_msp", "energy", "vacuity_valfit", "maxlogit", "mahalanobis")
N_BOOT = 2000
BOOT_SEED = 20261108
ARM_TO_GRID_ADAPTER = {"btl_ms": "bottleneck_parallel", "lora_ss": "lora"}

#: Reference files known to be defective, with the evidence that explains the
#: regression MISMATCH they cause (TMLR_PRESPEC section 6 + changelog 2026-11-08).
#: Problem B5 (docs/guide/05_problems_and_open_work.md): the committed grid JSON
#: for this cell is the Step 10a 20-episode smoke run. The 600-episode re-score
#: of the ORIGINAL thesis checkpoint (results/rq_factorial, Sept. 2026) is the
#: valid reference; the gate requires the new run to equal it exactly.
KNOWN_REFERENCE_DEFECTS = {
    "e4_cifar_5s_mnv3_lora_ss_grid_r16_evid_s42": {
        "run": TMLR / "e4" / "e4_cifar_5s_mnv3_lora_ss_grid_r16_evid_s42" / "metrics.json",
        "reference_600": REPO / "results" / "rq_factorial"
        / "cifar_fs_5shot_mobilenetv3_small_lora_evidential_seed42.json",
        "defective_reference": REPO / "results" / "grid"
        / "grid_cifar_5shot_mbnet_lora_seed42_lora_prototype-evidential_metrics.json",
        "problem": "B5",
    },
}


# =====================================================================
# Variance shares with bootstrap CIs
# =====================================================================
def _codes(rows, f):
    lv = sorted({r[f] for r in rows}, key=str)
    m = {v: i for i, v in enumerate(lv)}
    return np.array([m[r[f]] for r in rows]), len(lv)


def _group_stats(Y, codes, L):
    """Per-level means and counts of every row of Y (B, n)."""
    O = np.zeros((len(codes), L))
    O[np.arange(len(codes)), codes] = 1.0
    cnt = O.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (Y @ O) / cnt
    return means, cnt


def _eta_batch(Y, rows, factors, two_way):
    """The classical SS decomposition of rq_aggregate.eta_squared, vectorised
    over the rows of Y (B, n). Returns {term: (B,) share}."""
    grand = Y.mean(axis=1, keepdims=True)
    sst = ((Y - grand) ** 2).sum(axis=1)
    codes = {f: _codes(rows, f) for f in factors}
    main = {}
    ss = {}
    for f in factors:
        c, L = codes[f]
        means, cnt = _group_stats(Y, c, L)
        main[f] = means
        ss[f] = (cnt * (means - grand) ** 2).sum(axis=1)
    if two_way:
        for a, b in combinations(factors, 2):
            (ca, La), (cb, Lb) = codes[a], codes[b]
            pair = ca * Lb + cb
            means, cnt = _group_stats(Y, pair, La * Lb)
            present = cnt > 0
            la, lb = np.divmod(np.arange(La * Lb), Lb)
            dev = means - main[a][:, la] - main[b][:, lb] + grand
            ss[f"{a}:{b}"] = (cnt[present] * dev[:, present] ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = {t: v / sst for t, v in ss.items()}
        shares["residual"] = (sst - sum(ss.values())) / sst
    return shares, ss, sst


def _df(rows, term):
    return int(np.prod([len({r[f] for r in rows}) - 1 for f in term.split(":")]))


def eta_squared_with_ci(rows: list[dict], outcome: str, factors, *, two_way: bool = True,
                        n_boot: int = N_BOOT, seed: int = 0, group_by=None) -> dict:
    """eta^2 (+ two-way terms), omega^2 and 95% seed-bootstrap percentile CIs.

    `group_by` defines a configuration whose seeds are resampled together
    (default: the factors). Pass extra keys when a configuration holds more
    than one non-seed replicate, e.g. ("design","objective","score","pool").
    """
    factors = list(factors)
    point = A.eta_squared(rows, factors, value=outcome, interactions=two_way)
    shares = {k: float(v) for k, v in point["eta_squared"].items()}
    y = np.array([r[outcome] for r in rows], dtype=float)

    vec, ss, sst = _eta_batch(y[None, :], rows, factors, two_way)
    drift = max(abs(float(vec[k][0]) - shares[k]) for k in shares)
    if drift > 1e-9:
        raise RuntimeError(f"vectorised eta^2 disagrees with rq_aggregate.eta_squared by {drift:.2e}")

    n = len(rows)
    terms = [k for k in shares if k != "residual"]
    df_terms = {t: _df(rows, t) for t in terms}
    df_resid = n - 1 - sum(df_terms.values())
    ss_resid = shares["residual"] * float(sst[0])
    ms_err = ss_resid / df_resid if df_resid > 0 else float("nan")
    omega = {t: float((float(ss[t][0]) - df_terms[t] * ms_err) / (float(sst[0]) + ms_err))
             for t in terms}

    out = {"n": n, "factors": factors, "two_way": two_way, "balanced": bool(point["balanced"]),
           "method": ("classical SS (exact: balanced design)" if point["balanced"] else
                      "classical SS -- NOT exact on an unbalanced design, do not quote"),
           "shares": shares, "omega_squared": omega, "df": {**df_terms, "residual": df_resid},
           "ss_total": float(sst[0]), "n_boot": n_boot, "boot_seed": seed,
           "ci95": None, "boot_sd": None}
    if n_boot:
        keys = list(group_by or factors)
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(tuple(r[k] for k in keys), []).append(i)
        rng = np.random.default_rng(seed)
        idx = np.empty((n_boot, n), dtype=int)
        for members in groups.values():
            g = np.asarray(members)
            idx[:, g] = g[rng.integers(0, len(g), size=(n_boot, len(g)))]
        bshares, _, _ = _eta_batch(y[idx], rows, factors, two_way)
        out["ci95"] = {t: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                       for t, v in bshares.items()}
        out["boot_sd"] = {t: float(np.std(v)) for t, v in bshares.items()}
    return out


# =====================================================================
# Paired episode bootstrap
# =====================================================================
def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _per_episode(logits, targets, metric):
    logits = np.asarray(logits, dtype=np.float64)
    if metric == "accuracy":
        return (logits.argmax(axis=-1) == targets).mean(axis=1)
    if metric == "ece":
        # Softmax-of-logits ECE, i.e. for SOFTMAX-head models only: an
        # evidential model's native probabilities need PrototypeHead.to_evidence
        # and are never recomputed outside it.
        import torch
        from src.evaluators.calibration import expected_calibration_error
        p = _softmax(logits)
        return np.array([expected_calibration_error(torch.from_numpy(p[e]),
                                                    torch.from_numpy(np.asarray(targets[e])))
                         for e in range(len(p))])
    raise ValueError(f"unknown metric {metric!r}")


def episode_bootstrap_delta(logits_a, logits_b, targets, metric: str, *, n_boot: int = N_BOOT,
                            seed: int = 0) -> dict:
    """Mean per-episode metric(a) - metric(b), paired over the shared episodes,
    with a 95% percentile CI from resampling episodes."""
    la, lb, t = np.asarray(logits_a), np.asarray(logits_b), np.asarray(targets)
    if la.shape != lb.shape or la.shape[:2] != t.shape:
        raise ValueError(f"shape mismatch: {la.shape} vs {lb.shape} vs targets {t.shape}")
    d = _per_episode(la, t, metric) - _per_episode(lb, t, metric)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    delta = float(d.mean())
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if np.all(d == 0):
        delta, lo, hi = 0.0, 0.0, 0.0
    return {"metric": metric, "delta": delta, "ci95": [lo, hi], "n_episodes": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# =====================================================================
# Exact sign test, per backbone
# =====================================================================
def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def per_backbone_sign_test(pairs) -> dict:
    """Exact two-sided binomial sign test of `delta` per backbone. Zeros are
    dropped from n. Backbones are never pooled: RQ3's whole point is that the
    calibration sign depends on the backbone."""
    out = {}
    for bb in dict.fromkeys(p["backbone"] for p in pairs):
        d = [float(p["delta"]) for p in pairs if p["backbone"] == bb]
        pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
        out[bb] = {"n_pos": pos, "n_neg": neg, "n_zero": len(d) - pos - neg,
                   "n": pos + neg, "p_two_sided": _binom_two_sided(pos, pos + neg),
                   "median_delta": float(np.median(d)) if d else float("nan")}
    return out


# =====================================================================
# Tables
# =====================================================================
def underconfidence_table(records) -> list[dict]:
    """signed_calibration_gap = mean(max prob) - accuracy; < 0 = underconfident."""
    rows = []
    for r in records:
        if r["head"] != "evidential":
            continue
        s = r["summary"]
        g = s.get("signed_calibration_gap__evidential_native")
        gv = s.get("signed_calibration_gap__evidential_valfit")
        rows.append({"run_id": r["run_id"], "experiment": r.get("experiment"),
                     "dataset": r.get("dataset"), "k_shot": r.get("k_shot"),
                     "backbone": r.get("backbone"), "arm": r.get("arm"),
                     "gap_native": g, "gap_valfit": gv,
                     "underconfident_native": (g < 0) if g is not None else None,
                     "underconfident_valfit": (gv < 0) if gv is not None else None})
    return rows


def rq1_rows(records, outcome: str) -> list[dict]:
    """One row per grid observation, in rq_completion's factor names. Accepts
    fixture records (`values`) or loaded records (`summary`)."""
    rows = []
    for r in records:
        if "values" in r:
            v = r["values"][outcome]
        else:
            nat = "vacuity_native" if r["head"] == "evidential" else "msp"
            pset = J.NATIVE_PSET[r["head"]]
            key = {"accuracy": f"accuracy_mean__{pset}", "ece": f"ece_pooled__{pset}",
                   "far_ood_svhn": f"ood_auroc__svhn_far__{nat}",
                   "near_ood_tin": f"ood_auroc__tin_near__{nat}"}[outcome]
            v = r["summary"][key]
        rows.append({"dataset": r["dataset"], "k_shot": int(r["k_shot"]),
                     "backbone": r["backbone"], "adapter": ARM_TO_GRID_ADAPTER[r["arm"]],
                     "head": r["head"], "seed": int(r["seed"]), "value": float(v)})
    return rows


def matched_design_key(rec: dict) -> str:
    """For the matched-budget runs: one design per (backbone, arm, rank).
    rq_aggregate.design_key would merge an arm's ranks into one design and
    unbalance the crossing."""
    m = rec["meta"]
    return f"{rec['dataset']}/{m['backbone']}/{m['arm']}/{m['rank_spec']}"


def rq2_rows(raw_records, scores, near_pools=None, design_fn=None) -> list[dict]:
    """rq_aggregate.factorial_observations with an explicit score set."""
    near_pools = set(near_pools or A.NEAR_POOLS)
    design_fn = design_fn or A.design_key
    rows = []
    for r in raw_records:
        for key, val in r["summary"].items():
            if not key.startswith("ood_auroc__"):
                continue
            _, pool, score = key.split("__")
            if score not in scores:
                continue
            group = "far" if pool in A.FAR_POOLS else "near" if pool in near_pools else "other"
            rows.append({"design": design_fn(r), "objective": r["interpretation"],
                         "score": score, "pool": pool, "pool_group": group,
                         "seed": r["seed"], "auroc": float(val)})
    return rows


def rq2_decomposition(raw_records, scores, *, n_boot=N_BOOT, near_pools=None,
                      design_fn=None) -> dict:
    rows = rq2_rows(raw_records, scores, near_pools, design_fn)
    out = {"scores": list(scores)}
    for g in ("far", "near"):
        sel = [r for r in rows if r["pool_group"] == g]
        res = eta_squared_with_ci(sel, "auroc", ("design", "objective", "score"), n_boot=n_boot,
                                  seed=BOOT_SEED, group_by=("design", "objective", "score", "pool"))
        s = res["shares"]
        res["dominant"] = "score" if s["score"] > s["objective"] else "objective"
        res["score_over_objective"] = s["score"] / s["objective"] if s["objective"] > 0 else None
        res["mean_auroc_by_score"] = {
            sc: {o: float(np.mean([r["auroc"] for r in sel if r["score"] == sc
                                   and r["objective"] == o]))
                 for o in ("evidential", "softmax")} for sc in scores}
        out[g] = res
    return out


# =====================================================================
# Reproduction gate (TMLR_PRESPEC section 6)
# =====================================================================
def _numeric(d):
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def reproduction_gate(regression: dict, explained: dict) -> dict:
    """Pass iff every regression anchor is exact/within_tol, OR every failing
    anchor is a documented reference defect whose run equals the valid
    600-episode reference on every shared summary key. Tolerances are never
    widened: "equal" here means bit-identical."""
    bad = {k: v for k, v in regression.items() if k not in ("exact", "within_tol")}
    failing = [rid for ids in bad.values() for rid in ids]
    n_ok = sum(len(regression.get(k, [])) for k in ("exact", "within_tol"))
    if not failing:
        return {"status": "pass", "adjudicate": True, "n_ok": n_ok, "explained": {}}
    expl = {}
    for rid in failing:
        e = explained.get(rid)
        if e is None:
            expl[rid] = {"explained": False, "reason": "no documented reference defect"}
            continue
        new = _numeric(json.load(open(e["run"]))["summary"])
        ref_all = json.load(open(e["reference_600"]))
        ref = _numeric(ref_all.get("summary", ref_all))
        shared = sorted(set(new) & set(ref) - {"num_episodes"})
        diffs = {k: abs(new[k] - ref[k]) for k in shared}
        ok = (bool(shared) and int(ref.get("num_episodes", -1)) == C.EXPECTED_N_EPISODES
              and all(d == 0.0 for d in diffs.values()))
        info = {"explained": ok, "shared_keys": len(shared),
                "max_abs_diff": max(diffs.values()) if diffs else None,
                "reference_600": str(e["reference_600"]), "problem": e.get("problem")}
        if e.get("defective_reference") and Path(e["defective_reference"]).exists():
            info["defective_reference_num_episodes"] = json.load(
                open(e["defective_reference"])).get("num_episodes")
        expl[rid] = info
    ok = all(v["explained"] for v in expl.values())
    return {"status": "pass_with_explained_reference_defect" if ok else "fail",
            "adjudicate": ok, "n_ok": n_ok, "failing": failing, "explained": expl}


# =====================================================================
# Loading
# =====================================================================
def raw_metrics(rec) -> dict:
    return json.load(open(TMLR / rec["experiment"] / rec["run_id"] / "metrics.json"))


def grid120(records) -> list[dict]:
    """The thesis's 120-run grid as re-trained: E4 plus the E1 seed-42 twins."""
    base = J.grid_view(records)
    baselines = [r for r in records if r["experiment"] == "e4"
                 and r["arm"] in ("full_ft", "linear_probe")]
    return base + baselines


def rq3m_old_records() -> list[dict]:
    """results/rq3_matched (fixed stream, seeds 42-44) in tmlr_adjudicate's shape."""
    out = []
    for p in sorted((REPO / "results" / "rq3_matched").glob("rq3m_*_seed*.json")):
        d = json.load(open(p))
        m = d["meta"]
        out.append({"run_id": m["cell"], "experiment": "rq3_matched", "dataset": "mini_imagenet",
                    "k_shot": 5, "backbone": m["backbone"],
                    "arm": {"btl": "btl_ms", "lora": "lora_ss"}[m["arm"]],
                    "rank_spec": int(m["rank"]), "head": m["head"], "seed": int(m["seed"]),
                    "vary_stream": False, "loss": {},
                    "outcomes": J.outcomes_from_summary(m["head"], "mini_imagenet", d["summary"]),
                    "summary": d["summary"]})
    return out


def thesis_fixed_stream_records(records) -> list[dict]:
    """The thesis's matched-budget design as it was run (fixed stream, seeds
    42-44): results/rq3_matched plus the unmatched r16 pair, which the thesis
    read from the grid. The grid's MiniImageNet 5-shot r16 runs live in E4,
    except seed 42, which is trained once in E1 (fixed stream at seed 42)."""
    grid_mini5 = [r for r in J.grid_view(records)
                  if r["dataset"] == "mini_imagenet" and r["k_shot"] == 5]
    return rq3m_old_records() + grid_mini5


# =====================================================================
# Robustness: fixed-stream 3 seeds (thesis) vs varied-stream 5 seeds (E1)
# =====================================================================
def stream_robustness(old_records, new_records) -> dict:
    bbs = ("resnet18", "mobilenetv3_small")
    old = J.verdict_RB(old_records, bbs, "mini_imagenet", min_cells=3, seeds=T.SEEDS_3, vary=False)
    new = J.verdict_RB(new_records, bbs, "mini_imagenet", min_cells=3, seeds=T.SEEDS_5, vary=True)
    new3 = J.verdict_RB(new_records, bbs, "mini_imagenet", min_cells=3, seeds=T.SEEDS_3, vary=True)
    cells = {}
    for k in sorted(set(old["cells"]) | set(new["cells"])):
        o, n_ = old["cells"].get(k), new["cells"].get(k)
        if not (o and n_):
            continue
        cells[k] = {"delta_old": o["delta_matched"], "delta_new": n_["delta_matched"],
                    "sigma_old": o["sigma"], "sigma_new": n_["sigma"],
                    "sigma_ratio_new_over_old": n_["sigma"] / o["sigma"] if o["sigma"] else None,
                    "significant_old": o["significant"], "significant_new": n_["significant"],
                    "sign_agrees": o["sign"] == n_["sign"]}
    return {"old_fixed_3seed": {"verdict": old["verdict"], "counts": old["counts"]},
            "new_varied_5seed": {"verdict": new["verdict"], "counts": new["counts"]},
            "new_varied_first3seeds": {"verdict": new3["verdict"], "counts": new3["counts"]},
            "verdict_unchanged": old["verdict"] == new["verdict"], "cells": cells}


# =====================================================================
# Supplementary: paired episode CIs and per-backbone sign tests
# =====================================================================
def _id_logits(rec):
    z = np.load(TMLR / rec["experiment"] / rec["run_id"] / "logits.npz")
    return z["id_logits"].astype(np.float32), z["id_targets"].astype(np.int64)


def matched_pair_episode_cis(records, dataset) -> dict:
    """Seed by seed, btl_ms - lora_ss accuracy at matched budget, paired over
    the 600 TEST episodes. Shows how much of the across-seed sigma is episode
    noise; it adjudicates nothing."""
    idx = J.make_index(records)
    seeds = J.SEEDS[dataset]
    bbs = [b for b in dict.fromkeys(b for (b, _l) in T.LEVELS)
           if (dataset, 5, b, "btl_ms", J._rank_key(T.LEVELS[(b, "L")]["btl_ms"]),
               "softmax", seeds[0], "fixed") in idx]
    out = {}
    for bb in bbs:
        for lvl in ("L", "H"):
            arms = T.LEVELS[(bb, lvl)]
            for head in T.HEADS:
                per_seed = []
                for s in seeds:
                    st = J._stream(s, True)
                    b = idx.get((dataset, 5, bb, "btl_ms", J._rank_key(arms["btl_ms"]), head, s, st))
                    l = idx.get((dataset, 5, bb, "lora_ss", J._rank_key(arms["lora_ss"]), head, s, st))
                    if b is None or l is None:
                        continue
                    lb, t = _id_logits(b)
                    ll, t2 = _id_logits(l)
                    if not np.array_equal(t, t2):
                        raise RuntimeError(f"episode targets differ: {b['run_id']} vs {l['run_id']}")
                    r = episode_bootstrap_delta(lb, ll, t, "accuracy", seed=BOOT_SEED)
                    per_seed.append({"seed": s, **r})
                out[f"{bb}/{lvl}/{head}"] = {
                    "per_seed": per_seed,
                    "seeds_excluding_zero": sum(p["excludes_zero"] for p in per_seed),
                    "seeds_positive": sum(p["delta"] > 0 for p in per_seed)}
    return out


def matched_ece_sign_tests(records, dataset) -> dict:
    """Per backbone: sign of ECE(lora_ss) - ECE(btl_ms) at matched budget,
    paired by (level, head, seed)."""
    idx = J.make_index(records)
    pairs = []
    for (bb, lvl), arms in T.LEVELS.items():
        for head in T.HEADS:
            for s in J.SEEDS[dataset]:
                st = J._stream(s, True)
                b = idx.get((dataset, 5, bb, "btl_ms", J._rank_key(arms["btl_ms"]), head, s, st))
                l = idx.get((dataset, 5, bb, "lora_ss", J._rank_key(arms["lora_ss"]), head, s, st))
                if b and l:
                    pairs.append({"backbone": bb, "level": lvl, "head": head, "seed": s,
                                  "delta": l["outcomes"]["ece"] - b["outcomes"]["ece"]})
    return per_backbone_sign_test(pairs)


# =====================================================================
# Thesis reproduction on E4
# =====================================================================
def _max_diff(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    return max(abs(float(a[k]) - float(b[k])) for k in keys) if keys else float("nan")


def thesis_reproduction(records) -> dict:
    g = grid120(records)
    rq1_rows_src = J.grid_view(records)
    raw = [raw_metrics(r) for r in g]
    thesis_rq1 = json.load(open(REPO / "results/rq_completion/rq1_interactions_smoke_obs_replaced.json"))
    published = json.load(open(REPO / "results/rq_completion/rq1_interactions.json"))
    thesis_rq24 = json.load(open(REPO / "results/rq_completion/rq2_rq4_summary.json"))

    rq1 = {}
    for outcome in C.RQ1_OUTCOMES:
        rows = rq1_rows(rq1_rows_src, outcome)
        main = eta_squared_with_ci(rows, "value", RQ1_FACTORS, two_way=False, seed=BOOT_SEED)
        both = eta_squared_with_ci(rows, "value", RQ1_FACTORS, two_way=True, seed=BOOT_SEED)
        ref = thesis_rq1["outcomes"][outcome]
        rq1[outcome] = {
            "main_effects": main, "with_two_way": both,
            "max_abs_diff_vs_thesis_600ep_main": _max_diff(main["shares"], ref["main_effects_only"]),
            "max_abs_diff_vs_thesis_600ep_two_way": _max_diff(both["shares"], ref["with_two_way"]),
            "published_pp": C.PUBLISHED_MAIN_EFFECTS[outcome],
            "published_source_main": published["outcomes"][outcome]["main_effects_only"],
            "shift_vs_published_pp": {k: 100 * main["shares"][k] - v
                                      for k, v in C.PUBLISHED_MAIN_EFFECTS[outcome].items()}}

    rq2_thesis = rq2_decomposition(raw, RQ2_SCORES_THESIS)
    ref2 = thesis_rq24["rq2_after_complete_designs_only"]
    rq2_match = {gp: _max_diff(rq2_thesis[gp]["shares"], ref2[gp]["eta_squared"])
                 for gp in ("far", "near")}

    evid = [r for r in raw if r["interpretation"] == "evidential"]
    rq4 = A.rq2_verdict(A.rq2_table(evid))
    ref4 = thesis_rq24["rq4_after"]
    rq4_match = {k: (rq4[k], ref4[k]) for k in ref4}
    return {"n_models": len(g), "rq1": rq1,
            "rq2_four_scores": rq2_thesis, "rq2_max_abs_diff_vs_thesis": rq2_match,
            "rq4": rq4, "rq4_vs_thesis": rq4_match,
            "rq4_identical": all(a == b for a, b in rq4_match.values())}


# =====================================================================
# Aircraft (R-E support) and EDL recipe (R-F support)
# =====================================================================
def aircraft_analysis(records) -> dict:
    air = [r for r in records if r["experiment"] == "e3a"]
    rows = [{"shots": r["k_shot"], "backbone": r["backbone"], "adapter": r["arm"],
             "head": r["head"], "seed": r["seed"], **{k: r["outcomes"][k] for k in
                                                      ("accuracy", "ece", "auroc_near", "auroc_far")}}
            for r in air]
    fac = ("shots", "backbone", "adapter", "head")
    out = {"n": len(rows), "eta": {o: eta_squared_with_ci(rows, o, fac, seed=BOOT_SEED)
                                   for o in ("accuracy", "ece", "auroc_near", "auroc_far")}}
    raw = [raw_metrics(r) for r in air]
    out["rq2_six_scores"] = rq2_decomposition(raw, RQ2_SCORES_SIX,
                                              near_pools=("aircraft_near", "tin_near"))
    out["rq2_thesis_scores"] = rq2_decomposition(raw, RQ2_SCORES_THESIS,
                                                 near_pools=("aircraft_near", "tin_near"))
    evid = [r for r in raw if r["interpretation"] == "evidential"]
    out["rq4"] = A.rq2_verdict(A.rq2_table(evid))
    return out


def edl_recipe_analysis(records) -> dict:
    grid = J.grid_view(records)
    e5 = [r for r in records if r["experiment"] == "e5"]
    soft = [r for r in grid if r["head"] == "softmax"]
    evid_old = [r for r in grid if r["head"] == "evidential"]
    fac = ("dataset", "shots", "backbone", "adapter", "head")

    def rows(src):
        return [{"dataset": r["dataset"], "shots": r["k_shot"], "backbone": r["backbone"],
                 "adapter": r["arm"], "head": r["head"], "seed": r["seed"],
                 "ece": r["outcomes"]["ece"], "accuracy": r["outcomes"]["accuracy"]} for r in src]
    new = {o: eta_squared_with_ci(rows(soft + e5), o, fac, seed=BOOT_SEED) for o in ("ece", "accuracy")}
    old = {o: eta_squared_with_ci(rows(soft + evid_old), o, fac, seed=BOOT_SEED)
           for o in ("ece", "accuracy")}
    pairs = []
    for d in ("cifar_fs", "mini_imagenet"):
        for k in (1, 5):
            for bb in ("resnet18", "mobilenetv3_small"):
                for a in ("btl_ms", "lora_ss"):
                    def m(src, key="ece"):
                        return float(np.mean([r["outcomes"][key] for r in src if r["dataset"] == d
                                              and r["k_shot"] == k and r["backbone"] == bb
                                              and r["arm"] == a]))
                    pairs.append({"dataset": d, "k_shot": k, "backbone": bb, "adapter": a,
                                  "ece_softmax": m(soft), "ece_evid_orig": m(evid_old),
                                  "ece_evid_e5": m(e5), "acc_evid_orig": m(evid_old, "accuracy"),
                                  "acc_evid_e5": m(e5, "accuracy")})
    return {"eta_with_e5_recipe": new, "eta_with_original_recipe": old, "pairs": pairs,
            "e5_lowers_evid_ece_pairs": sum(p["ece_evid_e5"] < p["ece_evid_orig"] for p in pairs),
            "e5_still_worse_than_softmax_pairs": sum(p["ece_evid_e5"] > p["ece_softmax"] for p in pairs)}


# =====================================================================
# Driver
# =====================================================================
def _dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def run_all() -> None:
    regression = json.load(open(TMLR / "_regression.json"))
    gate = reproduction_gate(regression, KNOWN_REFERENCE_DEFECTS)
    _dump(gate, "reproduction_gate.json")
    print(f"reproduction gate: {gate['status']} ({gate['n_ok']} anchors ok)")
    if not gate["adjudicate"]:
        raise SystemExit("reproduction gate FAILED -- adjudication stops (TMLR_PRESPEC section 6)")

    records = J.load_records()
    print(f"{len(records)} test-mode records loaded")

    tr = thesis_reproduction(records)
    _dump(tr, "thesis_reproduction_e4.json")
    for o, v in tr["rq1"].items():
        print(f"  RQ1 {o:<13} vs thesis(600ep): main {v['max_abs_diff_vs_thesis_600ep_main']:.1e}"
              f"  two-way {v['max_abs_diff_vs_thesis_600ep_two_way']:.1e}")
    print(f"  RQ2 vs thesis: {tr['rq2_max_abs_diff_vs_thesis']}  RQ4 identical: {tr['rq4_identical']}")

    raw120 = [raw_metrics(r) for r in grid120(records)]
    _dump(rq2_decomposition(raw120, RQ2_SCORES_SIX), "rq2_six_scores_grid.json")
    mini = [r for r in records if r["dataset"] == "mini_imagenet" and r["experiment"] in ("e1", "e2")]
    _dump(rq2_decomposition([raw_metrics(r) for r in mini], RQ2_SCORES_SIX,
                            near_pools=("mini_near", "tin_near"), design_fn=matched_design_key),
          "rq2_six_scores_matched_mini.json")
    _dump(aircraft_analysis(records), "aircraft.json")
    _dump(edl_recipe_analysis(records), "edl_recipe.json")

    tmlr_ids = {r["run_id"] for r in records}
    under = underconfidence_table([r for r in records if r["run_id"] in tmlr_ids])
    _dump({"rows": under,
           "n": len(under),
           "underconfident_native": sum(1 for r in under if r["underconfident_native"]),
           "underconfident_valfit": sum(1 for r in under if r["underconfident_valfit"])},
          "underconfidence.json")

    _dump(stream_robustness(thesis_fixed_stream_records(records), records),
          "stream_robustness.json")

    _dump({"mini_imagenet": matched_pair_episode_cis(records, "mini_imagenet"),
           "fgvc_aircraft": matched_pair_episode_cis(records, "fgvc_aircraft")},
          "episode_cis_matched_accuracy.json")
    _dump({"mini_imagenet": matched_ece_sign_tests(records, "mini_imagenet"),
           "fgvc_aircraft": matched_ece_sign_tests(records, "fgvc_aircraft")},
          "ece_sign_tests_matched.json")

    v = J.write_verdicts()
    for k, x in v.items():
        print(f"  {k:<20} {x['verdict']}")


if __name__ == "__main__":
    run_all()
