"""TMLR plan Task 15 — statistics upgrade (scripts/tmlr_analysis.py) and the
rq_aggregate.eta_squared balance flag (problem B1).

CPU-only, no GPU, no real logits: every array here is synthetic except the E4
fixture, which holds 96 scalar outcomes copied from results/tmlr by the existing
loaders (tmlr_adjudicate.grid_view), and the thesis numbers it must reproduce,
read from the committed results/rq_completion JSON.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import rq_aggregate as A  # noqa: E402
import tmlr_analysis as TA  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "tmlr_e4_grid_outcomes.json"
THESIS_RQ1 = REPO / "results" / "rq_completion" / "rq1_interactions_smoke_obs_replaced.json"


def _toy_rows(seed=0, n_seeds=3, drop=None):
    """3 factors (2, 2, 3 levels) x n_seeds with real main effects and noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for a, b, c in itertools.product((0, 1), ("x", "y"), ("p", "q", "r")):
        if drop == (a, b, c):
            continue
        for s in range(n_seeds):
            v = 2.0 * a + (b == "y") * 1.0 + {"p": 0, "q": 0.5, "r": -0.5}[c] + rng.normal(0, 0.3)
            rows.append({"a": a, "b": b, "c": c, "seed": 42 + s, "value": float(v)})
    return rows


# ---------------------------------------------------------------- eta / CI
def test_eta_shares_and_residual_sum_to_one():
    res = TA.eta_squared_with_ci(_toy_rows(), "value", ("a", "b", "c"), n_boot=0)
    assert res["balanced"] is True
    total = sum(res["shares"].values())
    assert total == pytest.approx(1.0, abs=1e-12)
    assert set(res["shares"]) == {"a", "b", "c", "a:b", "a:c", "b:c", "residual"}
    main = TA.eta_squared_with_ci(_toy_rows(), "value", ("a", "b", "c"), two_way=False, n_boot=0)
    assert sum(main["shares"].values()) == pytest.approx(1.0, abs=1e-12)
    assert set(main["shares"]) == {"a", "b", "c", "residual"}
    # omega^2 shrinks eta^2 toward 0 and never exceeds it
    for t in ("a", "b", "c"):
        assert res["omega_squared"][t] <= res["shares"][t]


def test_ci_contains_point_estimate():
    res = TA.eta_squared_with_ci(_toy_rows(), "value", ("a", "b", "c"), n_boot=400, seed=1)
    for t in ("a", "b", "c"):
        lo, hi = res["ci95"][t]
        assert lo <= res["shares"][t] <= hi
        assert hi > lo
    # deterministic in its seed
    again = TA.eta_squared_with_ci(_toy_rows(), "value", ("a", "b", "c"), n_boot=400, seed=1)
    assert again["ci95"] == res["ci95"]


def test_bootstrap_resamples_within_configuration_only():
    """One seed per configuration: nothing to resample, so the CI is degenerate."""
    rows = _toy_rows(n_seeds=1)
    res = TA.eta_squared_with_ci(rows, "value", ("a", "b", "c"), n_boot=50, seed=0)
    for t in ("a", "b", "c"):
        lo, hi = res["ci95"][t]
        assert lo == pytest.approx(res["shares"][t]) and hi == pytest.approx(res["shares"][t])


def test_eta_squared_flags_missing_cell():
    """B1: dropping every seed of one configuration leaves the remaining cell
    counts all equal, which the old flag read as 'balanced'."""
    full = A.eta_squared(_toy_rows(), ["a", "b", "c"], value="value")
    assert full["balanced"] is True
    holed = A.eta_squared(_toy_rows(drop=(1, "y", "r")), ["a", "b", "c"], value="value")
    assert holed["balanced"] is False
    uneven = _toy_rows()[:-1]          # one seed missing in one cell
    assert A.eta_squared(uneven, ["a", "b", "c"], value="value")["balanced"] is False
    res = TA.eta_squared_with_ci(_toy_rows(drop=(1, "y", "r")), "value", ("a", "b", "c"), n_boot=0)
    assert res["balanced"] is False


# ---------------------------------------------------------------- episode bootstrap
def _episodes(rng, n_ep=40, q=15, way=5, signal=2.0):
    targets = rng.integers(0, way, size=(n_ep, q))
    logits = rng.normal(0, 1, size=(n_ep, q, way))
    logits[np.arange(n_ep)[:, None], np.arange(q)[None, :], targets] += signal
    return logits.astype(np.float32), targets


def test_episode_bootstrap_identical_inputs_zero_width():
    rng = np.random.default_rng(0)
    logits, targets = _episodes(rng)
    for metric in ("accuracy", "ece"):
        r = TA.episode_bootstrap_delta(logits, logits, targets, metric, n_boot=200, seed=0)
        assert r["delta"] == 0.0
        assert r["ci95"] == [0.0, 0.0]
        assert r["n_episodes"] == 40


def test_episode_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(1)
    strong, targets = _episodes(rng, signal=4.0)
    weak = rng.normal(0, 1, size=strong.shape).astype(np.float32)   # chance-level model
    r = TA.episode_bootstrap_delta(strong, weak, targets, "accuracy", n_boot=500, seed=0)
    assert r["delta"] > 0 and r["ci95"][0] > 0
    with pytest.raises(ValueError):
        TA.episode_bootstrap_delta(strong, weak[:-1], targets, "accuracy")


# ---------------------------------------------------------------- sign test, tables
def test_per_backbone_sign_test_never_pools():
    pairs = ([{"backbone": "resnet18", "delta": d} for d in (0.1, 0.2, 0.3, 0.1, 0.2)] +
             [{"backbone": "mobilenetv2", "delta": d} for d in (-0.1, 0.2, -0.3, 0.0)])
    out = TA.per_backbone_sign_test(pairs)
    assert set(out) == {"resnet18", "mobilenetv2"}
    r = out["resnet18"]
    assert (r["n_pos"], r["n_neg"], r["n_zero"]) == (5, 0, 0)
    assert r["p_two_sided"] == pytest.approx(2 * 0.5 ** 5)
    m = out["mobilenetv2"]
    assert (m["n_pos"], m["n_neg"], m["n_zero"]) == (1, 2, 1)   # ties dropped from n
    assert m["p_two_sided"] == pytest.approx(1.0)


def test_underconfidence_table_only_evidential():
    recs = [{"run_id": "a", "head": "evidential", "summary": {
                "signed_calibration_gap__evidential_native": -0.2,
                "signed_calibration_gap__evidential_valfit": -0.01}},
            {"run_id": "b", "head": "softmax", "summary": {
                "signed_calibration_gap__softmax": 0.03}}]
    t = TA.underconfidence_table(recs)
    assert [r["run_id"] for r in t] == ["a"]
    assert t[0]["gap_native"] == -0.2 and t[0]["underconfident_native"] is True


# ---------------------------------------------------------------- thesis reproduction
def test_e4_reproduces_thesis_rq1():
    """E4 (+ its E1 twins) re-trains the thesis grid. Its 96 balanced
    observations must give the thesis's RQ1 shares -- the 600-episode version,
    in which the one 20-episode smoke observation (problem B5) is replaced."""
    fx = json.load(open(FIX))["records"]
    thesis = json.load(open(THESIS_RQ1))["outcomes"]
    assert len(fx) == 96
    for outcome, ref in thesis.items():
        rows = TA.rq1_rows(fx, outcome)
        main = TA.eta_squared_with_ci(rows, "value", TA.RQ1_FACTORS, two_way=False, n_boot=0)
        both = TA.eta_squared_with_ci(rows, "value", TA.RQ1_FACTORS, two_way=True, n_boot=0)
        assert main["balanced"] and both["balanced"]
        for k, v in ref["main_effects_only"].items():
            assert main["shares"][k] == pytest.approx(v, abs=1e-9), (outcome, k)
        for k, v in ref["with_two_way"].items():
            assert both["shares"][k] == pytest.approx(v, abs=1e-9), (outcome, k)


# ---------------------------------------------------------------- reproduction gate
def test_gate_passes_only_an_explained_reference_defect(tmp_path):
    summary = {"accuracy_mean__evidential_native": 0.8757, "ece_pooled__evidential_native": 0.28}
    run = tmp_path / "e4" / "r1" / "metrics.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"summary": summary}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"summary": {**summary, "num_episodes": 600}}))
    reg = {"exact": ["x", "y"], "MISMATCH": ["r1"]}
    ok = TA.reproduction_gate(reg, explained={"r1": {"run": run, "reference_600": ref}})
    assert ok["status"] == "pass_with_explained_reference_defect" and ok["adjudicate"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"summary": {**summary, "ece_pooled__evidential_native": 0.3,
                                           "num_episodes": 600}}))
    nope = TA.reproduction_gate(reg, explained={"r1": {"run": run, "reference_600": bad}})
    assert nope["status"] == "fail" and not nope["adjudicate"]
    assert TA.reproduction_gate(reg, explained={})["status"] == "fail"
    assert TA.reproduction_gate({"exact": ["x"], "within_tol": ["z"]}, explained={})["status"] == "pass"


@pytest.mark.skipif(not (REPO / "results/tmlr/e1").exists(), reason="TMLR results not present")
def test_fixed_stream_side_reproduces_thesis_rq3_verdict():
    """The 'old' side of stream_robustness must be the thesis design: 4 cells,
    verdict backbone_intrinsic in 3 of 4 (results/rq3_matched/verdict.json)."""
    import tmlr_adjudicate as J
    recs = J.load_records()
    old = TA.stream_robustness(TA.thesis_fixed_stream_records(recs), recs)["old_fixed_3seed"]
    thesis = json.load(open(REPO / "results/rq3_matched/verdict.json"))
    assert old["counts"]["n_cells"] == 4
    assert old["verdict"] == "backbone_intrinsic"
    assert old["counts"]["backbone_intrinsic"] == 3
    assert thesis["verdict"] == "backbone_intrinsic"
