"""TMLR follow-up — pack runs into Kaggle-session shards (plan Task 8, A10).

A shard = one Kaggle session on "GPU T4 x2" = two lanes (one run at a time per
GPU). Each lane is packed to <= LANE_BUDGET_MIN estimated minutes, leaving
headroom under Kaggle's 12 h cap. Packing unit = a comparison group (all arms
of one backbone x level x head x seed), so a session that dies early still
leaves whole comparisons, never half of one.

    python scripts/tmlr_shards.py --wave 1 [--timings results/tmlr/_timings.json]
    python scripts/tmlr_shards.py --wave 0 --min-shards 4
    python scripts/tmlr_shards.py --retry results/tmlr/_coverage.json --retry-n 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tmlr_design as T  # noqa: E402

LANES_PER_SHARD = 2
LANE_BUDGET_MIN = 540.0
ACCOUNTS = ("acct1", "acct2", "acct3", "acct4")
_BB_ORDER = {"resnet18": 0, "mobilenetv3_small": 1, "resnet34": 2, "mobilenetv2": 3}


def group_key(s: T.RunSpec) -> tuple:
    if s.wave == 0 or s.experiment in ("e4", "e5", "e5val"):
        return (s.experiment, s.run_id)
    return (s.experiment, s.dataset, s.k_shot, s.backbone, s.level or "-", s.head, s.seed)


def timing_key(s: T.RunSpec) -> str:
    kind = T.MODULE_OF.get(s.arm, s.arm)
    return f"{s.dataset}|{s.backbone}|{kind}|{s.k_shot}|{s.mode}"


def _est(s: T.RunSpec, timings: dict | None) -> float:
    return float((timings or {}).get(timing_key(s), s.est_minutes))


def _order(s: T.RunSpec) -> tuple:
    return (_BB_ORDER.get(s.backbone, 9), s.level or "-", s.head, s.seed, s.arm, s.run_id)


def _pack(specs: list[T.RunSpec], prefix: str, wave: int, *, timings, min_shards: int,
          lanes_per_shard: int, budget: float) -> list[dict]:
    if not specs:
        return []
    groups: dict = {}
    for s in sorted(specs, key=lambda x: x.run_id):
        groups.setdefault(group_key(s), []).append(s)
    sized = sorted(((sum(_est(s, timings) for s in g), k, g) for k, g in groups.items()),
                   key=lambda t: (-t[0], t[1]))
    total = sum(t[0] for t in sized)
    n_lanes = max(math.ceil(total / budget), min_shards * lanes_per_shard, 1)
    n_lanes = math.ceil(n_lanes / lanes_per_shard) * lanes_per_shard
    while True:
        lanes = [[] for _ in range(n_lanes)]
        loads = [0.0] * n_lanes
        for size, _k, g in sized:                 # LPT: biggest group -> lightest lane
            i = min(range(n_lanes), key=lambda j: (loads[j], j))
            lanes[i].extend(g)
            loads[i] += size
        over = [i for i in range(n_lanes)
                if loads[i] > budget and len({group_key(s) for s in lanes[i]}) > 1]
        if not over:
            break
        n_lanes += lanes_per_shard
    order = sorted(range(n_lanes), key=lambda i: (-loads[i], i))
    shards = []
    for si in range(n_lanes // lanes_per_shard):
        idx = order[si * lanes_per_shard:(si + 1) * lanes_per_shard]
        shards.append({
            "shard_id": f"{prefix}-{si + 1:02d}",
            "wave": wave,
            "lanes": [[s.run_id for s in sorted(lanes[i], key=_order)] for i in idx],
            "est_minutes": [round(loads[i], 2) for i in idx],
        })
    return [sh for sh in shards if any(sh["lanes"])]


def pack_shards(specs: list[T.RunSpec], *, wave: int, lanes_per_shard: int = LANES_PER_SHARD,
                lane_budget_min: float = LANE_BUDGET_MIN, timings: dict | None = None,
                min_shards: int = 1) -> list[dict]:
    return _pack([s for s in specs if s.wave == wave], f"W{wave}", wave, timings=timings,
                 min_shards=min_shards, lanes_per_shard=lanes_per_shard, budget=lane_budget_min)


def make_retry_shards(coverage: dict, specs: list[T.RunSpec], *, n: int,
                      timings: dict | None = None, min_shards: int = 1) -> list[dict]:
    want = set(coverage.get("missing", [])) | set(coverage.get("error", []))
    chosen = [s for s in specs if s.run_id in want]
    unknown = want - {s.run_id for s in chosen}
    if unknown:
        raise KeyError(f"retry run ids not in the design: {sorted(unknown)[:5]}")
    return _pack(chosen, f"R{n}", -1, timings=timings, min_shards=min_shards,
                 lanes_per_shard=LANES_PER_SHARD, budget=LANE_BUDGET_MIN)


def assign_accounts(shards: list[dict], accounts=ACCOUNTS) -> dict[str, list[str]]:
    """Longest shard first onto the account with the least queued wall time
    (a shard's wall time = its slowest lane)."""
    plan = {a: [] for a in accounts}
    load = {a: 0.0 for a in accounts}
    for sh in sorted(shards, key=lambda x: (-max(x["est_minutes"]), x["shard_id"])):
        a = min(accounts, key=lambda acc: (load[acc], acc))
        plan[a].append(sh["shard_id"])
        load[a] += max(sh["est_minutes"])
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int)
    ap.add_argument("--retry", type=Path, help="results/tmlr/_coverage.json")
    ap.add_argument("--retry-n", type=int, default=1)
    ap.add_argument("--timings", type=Path)
    ap.add_argument("--min-shards", type=int, default=1)
    ap.add_argument("--accounts", nargs="+", default=list(ACCOUNTS))
    a = ap.parse_args(argv)
    specs = T.build_design()
    timings = json.load(open(a.timings)) if a.timings and a.timings.exists() else None
    if a.retry:
        cov = json.load(open(a.retry))
        flat = {"missing": [r for e in cov.values() if isinstance(e, dict) for r in e.get("missing", [])],
                "error": [r for e in cov.values() if isinstance(e, dict) for r in e.get("error", [])]}
        shards = make_retry_shards(flat, specs, n=a.retry_n, timings=timings,
                                   min_shards=a.min_shards)
        out = REPO / f"configs/tmlr/shards_r{a.retry_n}.json"
    else:
        shards = pack_shards(specs, wave=a.wave, timings=timings, min_shards=a.min_shards)
        out = REPO / f"configs/tmlr/shards_w{a.wave}.json"
    plan = assign_accounts(shards, a.accounts)
    out.write_text(json.dumps({"shards": shards, "accounts": plan}, indent=1, sort_keys=True))
    n_runs = sum(len(l) for sh in shards for l in sh["lanes"])
    print(f"wrote {out.relative_to(REPO)}: {len(shards)} shards, {n_runs} runs")
    by_id = {sh["shard_id"]: sh for sh in shards}
    for acct, ids in plan.items():
        hours = sum(max(by_id[i]["est_minutes"]) for i in ids) / 60
        print(f"  {acct}: {', '.join(ids) or '-'}   (~{hours:.1f} h of sessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
