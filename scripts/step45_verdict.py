"""Step 4.5 verdict: master comparison table + tiered decision rule.

Consumes the two Step 4.5 metrics JSONs (evidential + softmax) produced by
scripts/evaluate.py with the score x OOD-pool matrix, and decides which tier
of the decision rule (design spec §2) was reached:

  Tier 1 (headline stands): evidential beats BOTH softmax-MSP AND
      temperature-scaled softmax on NEAR-OOD AUROC by >= margin, and is not
      meaningfully worse-calibrated (ECE) than TS-softmax.
  Tier 2 (softened): evidential ties best baseline on near-OOD but wins ID
      calibration (evid ECE < TS-softmax ECE).
  Tier 3 (reframe): neither -> "parity at a calibration cost".

The logic is pure (dict in -> result out) so it is unit-tested locally without
any GPU run; the Colab notebook and the Step 4.5 writeup both call it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

NEAR_MARGIN = 0.03      # AUROC margin for a "real" near-OOD win (spec §2)
ECE_TOL = 0.02          # how much worse-calibrated evidential may be and still "not worse"


def _near_pool(metrics: dict) -> str:
    """Prefer TinyImageNet near-OOD; fall back to CIFAR-100-heldout near-OOD."""
    for pool in ("tin_near", "cifar100_near"):
        if any(k.startswith(f"ood_auroc__{pool}__") for k in metrics):
            return pool
    raise KeyError("no near-OOD pool (tin_near / cifar100_near) in metrics")


def _evid_near_auroc(evid: dict, pool: str) -> float:
    return float(evid[f"ood_auroc__{pool}__vacuity"])


def _soft_near_best(soft: dict, pool: str) -> Tuple[float, str]:
    """Best near-OOD AUROC among the softmax-side scores (msp/ts_msp/energy)."""
    candidates = {}
    for score in ("msp", "ts_msp", "energy"):
        key = f"ood_auroc__{pool}__{score}"
        if key in soft:
            candidates[score] = float(soft[key])
    best_score = max(candidates, key=candidates.get)
    return candidates[best_score], best_score


def _soft_fair_ece(soft: dict) -> float:
    """Temperature-scaled ECE if available, else raw pooled ECE."""
    return float(soft.get("ece_ts", soft.get("ece_pooled")))


def decide_tier(evid: dict, soft: dict, *, near_margin: float = NEAR_MARGIN,
                ece_tol: float = ECE_TOL) -> Tuple[int, str]:
    """Return (tier in {1,2,3}, human-readable reason)."""
    pool = _near_pool(evid)
    evid_near = _evid_near_auroc(evid, pool)
    soft_near, soft_best_score = _soft_near_best(soft, pool)
    evid_ece = float(evid.get("ece_pooled"))
    soft_ece = _soft_fair_ece(soft)

    beats_near = evid_near >= soft_near + near_margin
    not_worse_cal = evid_ece <= soft_ece + ece_tol
    ties_near = abs(evid_near - soft_near) < near_margin
    wins_cal = evid_ece < soft_ece

    base = (f"near-OOD[{pool}] evid(vacuity)={evid_near:.3f} vs "
            f"best softmax {soft_best_score}={soft_near:.3f} "
            f"(margin {near_margin}); ECE evid={evid_ece:.3f} vs "
            f"TS-softmax={soft_ece:.3f} (tol {ece_tol}).")

    if beats_near and not_worse_cal:
        return 1, "TIER 1 (headline stands): " + base
    if ties_near and wins_cal:
        return 2, "TIER 2 (competitive OOD, better calibrated): " + base
    return 3, "TIER 3 (reframe to parity-at-a-calibration-cost): " + base


def build_master_table(evid: dict, soft: dict) -> str:
    """Human-readable comparison table across the score x metric grid."""
    pool = _near_pool(evid)
    rows = []
    rows.append("Step 4.5 master comparison  (5-way 5-shot CIFAR-FS, 600 episodes)")
    rows.append("=" * 72)
    rows.append(f"{'metric':<30}{'evidential':>18}{'softmax':>18}")
    rows.append("-" * 72)

    def line(label, e, s, fmt="{:.3f}"):
        es = fmt.format(e) if e is not None else "-"
        ss = fmt.format(s) if s is not None else "-"
        rows.append(f"{label:<30}{es:>18}{ss:>18}")

    line("accuracy", evid.get("accuracy_mean"), soft.get("accuracy_mean"))
    line("macro-F1", evid.get("f1_macro_mean"), soft.get("f1_macro_mean"))
    line("ECE (pooled)", evid.get("ece_pooled"), soft.get("ece_pooled"))
    line("ECE (post-TS, softmax)", None, soft.get("ece_ts"))
    line("Brier", evid.get("brier_mean"), soft.get("brier_mean"))
    line("Brier (post-TS, softmax)", None, soft.get("brier_ts"))
    rows.append("-" * 72)
    rows.append("OOD AUROC by pool x score  (higher = better):")
    for p in ("svhn_far", "cifar100_near", "tin_near"):
        ev = evid.get(f"ood_auroc__{p}__vacuity")
        for sc in ("msp", "ts_msp", "energy"):
            sv = soft.get(f"ood_auroc__{p}__{sc}")
            if ev is not None or sv is not None:
                line(f"  {p}: evid=vacuity / soft={sc}", ev, sv)
    rows.append("-" * 72)
    tier, reason = decide_tier(evid, soft)
    rows.append(reason)
    rows.append(f"NEAR-OOD pool used for the verdict: {pool}")
    return "\n".join(rows)


def _load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidential", required=True, help="evidential metrics JSON")
    ap.add_argument("--softmax", required=True, help="softmax metrics JSON")
    args = ap.parse_args()
    evid, soft = _load(args.evidential), _load(args.softmax)
    print(build_master_table(evid, soft))


if __name__ == "__main__":
    main()
