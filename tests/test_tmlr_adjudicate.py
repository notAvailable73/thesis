"""TMLR plan Task 11 — pre-specified decision rules (docs/TMLR_PRESPEC.md).

Synthetic records are built so each rule's every verdict is hit exactly; the
rq3_matched reproduction test proves the generalised rule B still returns the
thesis verdict on the committed data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tmlr_adjudicate as J  # noqa: E402
import tmlr_design as T  # noqa: E402

Z = {3: (-1.0, 0.0, 1.0), 5: (-1.2649110640673518, -0.6324555320336759, 0.0,
                              0.6324555320336759, 1.2649110640673518)}   # sample SD = 1


def rec(dataset, backbone, arm, rank, head, seed, *, k=5, vary=True, **outcomes):
    return {"dataset": dataset, "k_shot": k, "backbone": backbone, "arm": arm,
            "rank_spec": rank, "head": head, "seed": seed, "vary_stream": vary,
            "outcomes": outcomes}


def arm_recs(dataset, backbone, arm, rank, head, seeds, outcome, mean, sd, **fixed):
    out = []
    for z, seed in zip(Z[len(seeds)], seeds):
        o = dict(fixed)
        o[outcome] = mean + sd * z
        out.append(rec(dataset, backbone, arm, rank, head, seed, **o))
    return out


def level_recs(backbone, level, head, outcome, means, sd=0.001, dataset="mini_imagenet",
               seeds=T.SEEDS_5):
    """means: arm -> mean outcome for that matched level."""
    recs = []
    for arm, rank in T.LEVELS[(backbone, level)].items():
        recs += arm_recs(dataset, backbone, arm, rank, head, seeds, outcome, means[arm], sd)
    return recs


# ------------------------------------------------------------------ R-A
def _ra_records(site, module, backbones=("resnet18", "mobilenetv3_small", "resnet34",
                                          "mobilenetv2")):
    recs = []
    for bb in backbones:
        for lvl in ("L", "H"):
            for head in T.HEADS:
                means = {"lora_ss": 0.80, "btl_ss": 0.80 + module,
                         "lora_ms": 0.80 + site, "btl_ms": 0.80 + site + module}
                recs += level_recs(bb, lvl, head, "accuracy", means)
    return recs


@pytest.mark.parametrize("site,module,verdict", [
    (0.05, 0.0, "site"), (0.0, 0.05, "module"), (0.05, 0.05, "both"), (0.0, 0.0, "inconclusive")])
def test_RA(site, module, verdict):
    v = J.verdict_RA(_ra_records(site, module), "accuracy", "mini_imagenet")
    assert v["complete"] and v["verdict"] == verdict, v["counts"]


def test_RA_incomplete_is_inconclusive():
    recs = _ra_records(0.05, 0.0)[:-1]
    v = J.verdict_RA(recs, "accuracy", "mini_imagenet")
    assert not v["complete"] and v["verdict"] == "inconclusive"


# ------------------------------------------------------------------ R-B / R-C / R-D
def _ece_records(delta_ss, delta_ms, unmatched, backbones, dataset="mini_imagenet",
                 seeds=T.SEEDS_5, sd=0.002):
    """Matched gaps per backbone: ECE(lora_ss) - ECE(btl_ms) = delta_ss and
    ECE(lora_ms) - ECE(btl_ms) = delta_ms at BOTH levels; unmatched gap
    ECE(lora_ss r16) - ECE(btl_ms r16) = unmatched, wherever those two arms sit.
    btl_ss = btl_ms throughout."""
    recs = []
    for bb in backbones:
        d, dms, u = delta_ss[bb], delta_ms[bb], unmatched[bb]
        L, H = T.LEVELS[(bb, "L")], T.LEVELS[(bb, "H")]
        a = 0.10
        if L["lora_ss"] == 16 and H["btl_ms"] == 16:      # ResNets: pair spans L and H
            b = a + d - u
        else:
            b = 0.10
        base = {"L": a, "H": b}
        for head in T.HEADS:
            for lvl in ("L", "H"):
                x = base[lvl]
                means = {"btl_ms": x, "btl_ss": x, "lora_ss": x + d, "lora_ms": x + dms}
                recs += level_recs(bb, lvl, head, "ece", means, sd=sd, dataset=dataset, seeds=seeds)
            if bb in T.UNMATCHED:
                arms = T.UNMATCHED[bb]
                btl16 = a if "btl_ms" not in arms else 0.10     # MNv3: btl_ms r16 is level L
                for arm, rank in arms.items():
                    m = btl16 + (u if arm == "lora_ss" else 0.0)
                    recs += arm_recs(dataset, bb, arm, rank, head, seeds, "ece", m, sd)
    return recs


BB4 = ("resnet18", "mobilenetv3_small", "resnet34", "mobilenetv2")
FAMILY = {"resnet18": 1, "resnet34": 1, "mobilenetv3_small": -1, "mobilenetv2": -1}


def test_RB_backbone_intrinsic():
    d = {b: 0.05 * FAMILY[b] for b in BB4}
    v = J.verdict_RB(_ece_records(d, d, d, BB4), BB4, "mini_imagenet", min_cells=6)
    assert v["verdict"] == "backbone_intrinsic", v["counts"]


def test_RB_budget():
    small = {b: 0.0 for b in BB4}
    unm = {b: 0.05 * FAMILY[b] for b in BB4}
    v = J.verdict_RB(_ece_records(small, small, unm, BB4), BB4, "mini_imagenet", min_cells=6)
    assert v["verdict"] == "budget", v["counts"]


def test_RB_architecture():
    d = {b: 0.05 for b in BB4}
    unm = {b: 0.05 * FAMILY[b] for b in BB4}
    v = J.verdict_RB(_ece_records(d, d, unm, BB4), BB4, "mini_imagenet", min_cells=6)
    assert v["verdict"] == "architecture", v["counts"]


def test_RC_explains_vs_persists():
    flip = {b: 0.05 * FAMILY[b] for b in BB4}
    same = {b: 0.05 for b in BB4}
    zero = {b: 0.0 for b in BB4}
    assert J.verdict_RC(_ece_records(flip, same, flip, BB4), "mini_imagenet")["verdict"] == "site_role_explains"
    assert J.verdict_RC(_ece_records(flip, zero, flip, BB4), "mini_imagenet")["verdict"] == "site_role_explains"
    assert J.verdict_RC(_ece_records(flip, flip, flip, BB4), "mini_imagenet")["verdict"] == "flip_persists"


def test_RD_family_consistent_vs_not():
    flip = {b: 0.05 * FAMILY[b] for b in BB4}
    assert J.verdict_RD(_ece_records(flip, flip, flip, BB4), "mini_imagenet")["verdict"] == "family_consistent"
    odd = dict(flip, resnet34=-0.05, mobilenetv2=0.05)
    assert J.verdict_RD(_ece_records(odd, odd, odd, BB4), "mini_imagenet")["verdict"] == "not_family_consistent"


def test_RB_reproduces_thesis_verdict():
    """Rule B, generalised, on the committed rq3_matched records (3 seeds, fixed
    stream) must return the thesis verdict: backbone_intrinsic, 3 of 4."""
    import rq_aggregate as A
    recs = []
    for p in sorted((REPO / "results/rq3_matched").glob("rq3m_*.json")):
        r = json.load(open(p))
        m = r["meta"]
        out = A.rq3m_cell_metrics(r)
        recs.append(rec("mini_imagenet", m["backbone"], {"btl": "btl_ms", "lora": "lora_ss"}[m["arm"]],
                        int(m["rank"]), m["head"], int(m["seed"]), vary=False,
                        ece=out["ece"], accuracy=out["accuracy"], auroc_near=out["auroc_near"]))
    pre = {k: v for k, v in A.RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE.items()
           if k[0] == "mobilenetv3_small"}
    v = J.verdict_RB(recs, ("resnet18", "mobilenetv3_small"), "mini_imagenet", min_cells=3,
                     seeds=T.SEEDS_3, unmatched_override=pre, vary=False)
    assert v["verdict"] == "backbone_intrinsic"
    assert v["counts"]["backbone_intrinsic"] == 3 and v["counts"]["budget"] == 0


def test_sigma_uses_ddof1():
    s = J.arm_stats_from_values([1.0, 2.0, 3.0])
    assert s["sd"] == pytest.approx(1.0)


# ------------------------------------------------------------------ R-E / R-F helpers
def test_eta_squared_main_effects():
    rows = [{"a": a, "b": b, "y": 10 * a + b} for a in (0, 1) for b in (0, 1) for _ in range(3)]
    eta = J.eta_squared(rows, "y", ("a", "b"))
    assert eta["a"] == pytest.approx(100 / 101) and eta["b"] == pytest.approx(1 / 101)
