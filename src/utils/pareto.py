"""Step 11 (RQ4) — Pareto-frontier mechanics.

Pure Python: no torch, no matplotlib, no dependency on results/ existing on
disk. This is deliberate — it lets the frontier LOGIC be fully unit-tested
before any efficiency measurement exists (scripts/efficiency_table.py hasn't
run yet the first time this module is exercised), and it is testable purely
against the already-committed results/mvt_results.json.

A point is `(cost, quality)`: lower cost and higher quality are both wins.
`eps_cost` / `eps_qual` widen the dominance test so two points that differ
only by measurement/seed noise are treated as co-optimal rather than one
spuriously "beating" the other — see step_writeups/step11.txt Section 0 for
why this matters here specifically (evidential has exactly +2 trainable
params over softmax at every adapter, which produces a real-but-meaningless
Pareto step against the ±0.5pp episode-level CI).
"""
from __future__ import annotations

from typing import Sequence

__all__ = ["dominates", "pareto_front", "recommended_point"]


def dominates(q: Sequence[float], p: Sequence[float], *,
             eps_cost: float = 0.0, eps_qual: float = 0.0) -> bool:
    """True iff point `q` dominates point `p` (minimising cost, maximising
    quality), i.e. q is not worse on either axis (within eps) and strictly
    better on at least one.

    Exact coordinate duplicates never dominate each other — without this
    guard, two identical points would each "dominate" the other and both
    would be removed from the frontier.
    """
    if q[0] == p[0] and q[1] == p[1]:
        return False
    not_worse = (q[0] <= p[0] + eps_cost) and (q[1] >= p[1] - eps_qual)
    strictly_better = (q[0] < p[0] - eps_cost) or (q[1] > p[1] + eps_qual)
    return not_worse and strictly_better


def pareto_front(points: Sequence[Sequence[float]], *,
                 eps_cost: Sequence[float] | float | None = None,
                 eps_qual: Sequence[float] | float | None = None
                 ) -> list[int]:
    """Indices (into `points`, ascending by cost) of the non-dominated set.

    `points[i] = (cost, quality, ...)` — only the first two entries are read,
    so callers may carry extra payload (label, key) in each tuple.

    `eps_cost` / `eps_qual` may be a single float (applied to every point) or
    a per-point sequence; when per-point, the pairwise tolerance used for a
    given (i, j) comparison is `max(eps[i], eps[j])` so a noisy point cannot
    be shielded from comparisons where the OTHER point is the noisy one.

    O(n^2) pairwise comparison, deliberately: every panel in this thesis has
    at most 12 points, and an explicit nested-loop predicate is auditable in
    a way a sort-and-sweep implementation is not (same reasoning as
    scripts/grid_plots.py's `_pick_best_cell` being a plain loop rather than
    a max() one-liner).
    """
    n = len(points)

    def _eps_at(eps, i):
        if eps is None:
            return 0.0
        if isinstance(eps, (int, float)):
            return float(eps)
        return float(eps[i])

    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            ec = max(_eps_at(eps_cost, i), _eps_at(eps_cost, j))
            eq = max(_eps_at(eps_qual, i), _eps_at(eps_qual, j))
            if dominates(points[j], points[i], eps_cost=ec, eps_qual=eq):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: points[i][0])


def recommended_point(points: Sequence[Sequence[float]],
                      labels: Sequence[str], *,
                      tol_qual: float, prefer: str = "min_cost") -> int:
    """Index of the "recommended" point under a pre-registered rule.

    `prefer="min_cost"` (the only rule implemented, and the one Step 11
    pre-registers before any measurement exists — see
    step_writeups/step11.txt Section 0): the minimum-cost point whose
    quality is within `tol_qual` of the best quality in `points`. If no
    point qualifies (shouldn't happen since the best-quality point always
    qualifies against itself), falls back to the best-quality point.
    """
    if not points:
        raise ValueError("recommended_point: empty points")
    if prefer != "min_cost":
        raise ValueError(f"Unknown prefer rule: {prefer!r}")
    best_quality = max(p[1] for p in points)
    candidates = [i for i, p in enumerate(points)
                 if p[1] >= best_quality - tol_qual]
    return min(candidates, key=lambda i: points[i][0])
