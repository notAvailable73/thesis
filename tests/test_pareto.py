"""Step 11 (RQ4) tests for src/utils/pareto.py's frontier mechanics.

Pure-Python module under test — no torch, no GPU, no efficiency measurement
required. Covers the dominance predicate's edge cases (empty / single point,
exact duplicates, cost=0, equal-cost ties, eps widening) plus two SCIENTIFIC
regression tests against the already-committed results/mvt_results.json:
Full-FT is dominated at CIFAR-FS 5-shot and on the frontier at 1-shot (see
step_writeups/step11.txt Section 0's pre-registered params-vs-accuracy
frontier reasoning). If a future data change flips either of those, this
suite is the alarm.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.utils.pareto import dominates, pareto_front, recommended_point

REPO_ROOT = Path(__file__).resolve().parents[1]
MVT_PATH = REPO_ROOT / "results" / "mvt_results.json"


# --------------------------------------------------------------------------
# dominates()
# --------------------------------------------------------------------------
def test_lower_cost_higher_quality_dominates():
    assert dominates((1.0, 0.9), (2.0, 0.8))


def test_higher_cost_lower_quality_does_not_dominate():
    assert not dominates((2.0, 0.8), (1.0, 0.9))


def test_exact_duplicates_never_dominate_each_other():
    a = (5.0, 0.5)
    b = (5.0, 0.5)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_cost_better_quality_dominates():
    # The latency-tie case: two heads of the same adapter share cost exactly.
    assert dominates((10.0, 0.95), (10.0, 0.90))
    assert not dominates((10.0, 0.90), (10.0, 0.95))


def test_equal_quality_lower_cost_dominates():
    assert dominates((1.0, 0.5), (2.0, 0.5))


def test_strictly_worse_on_both_axes_dominates():
    assert dominates((1.0, 0.9), (3.0, 0.5))


def test_incomparable_points_neither_dominates():
    # cheaper but lower quality: neither strictly beats the other.
    assert not dominates((1.0, 0.5), (2.0, 0.9))
    assert not dominates((2.0, 0.9), (1.0, 0.5))


def test_real_plus_two_param_pair_is_mutually_non_dominating():
    # The actual evidential-vs-softmax case: evidential costs 2 more params
    # but has higher accuracy, so neither dominates the other -- both belong
    # on the strict frontier (see test_strict_frontier_keeps_both_head_variants).
    softmax = (31744.0, 0.9144)
    evidential = (31746.0, 0.9158)
    assert not dominates(evidential, softmax)
    assert not dominates(softmax, evidential)


def test_lower_cost_equal_quality_dominates():
    assert dominates((100.0, 0.5), (102.0, 0.5))


# --------------------------------------------------------------------------
# pareto_front()
# --------------------------------------------------------------------------
def test_empty_input():
    assert pareto_front([]) == []


def test_single_point_is_always_on_the_frontier():
    assert pareto_front([(5.0, 0.5)]) == [0]


def test_all_dominated_by_one_point():
    points = [(1.0, 0.9), (2.0, 0.8), (3.0, 0.7), (4.0, 0.6)]
    assert pareto_front(points) == [0]


def test_all_collinear_non_dominated_chain():
    # cost and quality increase TOGETHER (a real trade-off curve) -> no point
    # has both lower-or-equal cost and higher-or-equal quality than another,
    # so every point is on the frontier (classic staircase).
    points = [(1.0, 0.5), (2.0, 0.7), (3.0, 0.9)]
    assert pareto_front(points) == [0, 1, 2]


def test_exact_duplicate_points_both_survive():
    points = [(5.0, 0.5), (5.0, 0.5), (10.0, 0.1)]
    front = pareto_front(points)
    assert set(front) == {0, 1}


def test_cost_zero_point_is_never_dominated_on_cost():
    # 0 params: nothing can be cheaper, so this point survives unless
    # something has cost <= 0 AND quality >= it.
    points = [(0.0, 0.70), (100.0, 0.95)]
    front = pareto_front(points)
    assert 0 in front
    assert 1 in front  # higher quality, higher cost: also non-dominated


def test_equal_cost_tie_resolves_to_better_quality_only():
    points = [(10.0, 0.90), (10.0, 0.95)]
    front = pareto_front(points)
    assert front == [1]


def test_eps_qual_keeps_a_head_pair_co_optimal():
    # softmax (cheaper, lower quality) vs evidential (pricier, higher quality):
    # already mutually non-dominating at eps=0 (both on the strict frontier).
    # Widening eps must not break that.
    points = [(31744.0, 0.9144), (31746.0, 0.9158)]
    strict = pareto_front(points)
    assert len(strict) == 2
    widened = pareto_front(points, eps_cost=0.1 * 31744, eps_qual=0.01)
    assert set(widened) == {0, 1}


def test_strict_frontier_keeps_both_head_variants():
    points = [(31744.0, 0.9144), (31746.0, 0.9158)]
    assert set(pareto_front(points)) == {0, 1}


def test_order_invariance():
    points = [(3.0, 0.5), (1.0, 0.9), (2.0, 0.7)]
    front_a = pareto_front(points)
    shuffled = [points[2], points[0], points[1]]
    front_b = pareto_front(shuffled)
    costs_a = sorted(points[i][0] for i in front_a)
    costs_b = sorted(shuffled[i][0] for i in front_b)
    assert costs_a == costs_b


def test_per_point_eps_uses_max_of_the_pair():
    # Equal cost, point 1 has slightly higher quality -> point 1 dominates
    # point 0 via the quality axis alone. Give ONLY point 0 a generous
    # eps_qual; because the pairwise tolerance is max(eps[i], eps[j]), that
    # is enough to neutralise the domination even though point 1's own eps
    # is 0.
    points = [(10.0, 0.80), (10.0, 0.82)]
    front_no_eps = pareto_front(points)
    assert front_no_eps == [1]  # point 1 strictly better on quality -> dominates
    front_with_eps = pareto_front(points, eps_cost=[0.0, 0.0], eps_qual=[0.05, 0.0])
    assert set(front_with_eps) == {0, 1}


# --------------------------------------------------------------------------
# recommended_point()
# --------------------------------------------------------------------------
def test_recommended_point_picks_cheapest_within_tolerance():
    points = [(10.0, 0.90), (20.0, 0.95), (30.0, 0.951)]
    labels = ["a", "b", "c"]
    # tol=0.06 -> all three qualify (best=0.951); cheapest is "a"
    idx = recommended_point(points, labels, tol_qual=0.06)
    assert labels[idx] == "a"


def test_recommended_point_tight_tolerance_forces_expensive_pick():
    points = [(10.0, 0.90), (20.0, 0.95), (30.0, 0.951)]
    labels = ["a", "b", "c"]
    # tol=0.0001 excludes "b" (0.951 - 0.95 = 0.001 > tol), leaving only "c".
    idx = recommended_point(points, labels, tol_qual=0.0001)
    assert labels[idx] == "c"


def test_recommended_point_rejects_unknown_rule():
    with pytest.raises(ValueError):
        recommended_point([(1.0, 0.5)], ["a"], tol_qual=0.1, prefer="max_cost")


def test_recommended_point_rejects_empty():
    with pytest.raises(ValueError):
        recommended_point([], [], tol_qual=0.1)


# --------------------------------------------------------------------------
# Scientific regressions against the committed Step-10 grid.
# --------------------------------------------------------------------------
def _cifar_fs_params_accuracy_points(kshot: str) -> list[tuple]:
    mvt = json.loads(MVT_PATH.read_text())
    node = mvt["results"]["cifar_fs"][kshot]
    points = []
    for backbone in node:
        for adapter in node[backbone]:
            for head in node[backbone][adapter]:
                m = node[backbone][adapter][head]
                points.append((
                    m["n_params"]["mean"],
                    m["accuracy_mean"]["mean"],
                    f"{backbone}/{adapter}/{head}",
                ))
    return points


@pytest.mark.skipif(not MVT_PATH.exists(), reason="results/mvt_results.json not present")
def test_full_ft_is_dominated_at_cifar_fs_5shot():
    points = _cifar_fs_params_accuracy_points("5shot")
    front_idx = pareto_front(points)
    front_labels = {points[i][2] for i in front_idx}
    assert "resnet18/full_ft/softmax" not in front_labels
    assert "resnet18/full_ft/evidential" not in front_labels


@pytest.mark.skipif(not MVT_PATH.exists(), reason="results/mvt_results.json not present")
def test_full_ft_is_on_the_frontier_at_cifar_fs_1shot():
    points = _cifar_fs_params_accuracy_points("1shot")
    front_idx = pareto_front(points)
    front_labels = {points[i][2] for i in front_idx}
    assert "resnet18/full_ft/softmax" in front_labels
