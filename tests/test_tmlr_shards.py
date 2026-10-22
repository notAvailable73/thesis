"""TMLR plan Task 8 — shard packer (pure, deterministic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tmlr_design as T  # noqa: E402
import tmlr_shards as S  # noqa: E402

DESIGN = T.build_design(e5_selection=None)


def _ids(shards):
    return [r for sh in shards for lane in sh["lanes"] for r in lane]


def test_every_run_in_exactly_one_shard():
    shards = S.pack_shards(DESIGN, wave=1)
    ids = _ids(shards)
    expected = {s.run_id for s in DESIGN if s.wave == 1}
    assert len(ids) == len(set(ids)) and set(ids) == expected


def test_lane_budget_respected():
    by_id = {s.run_id: s for s in DESIGN}
    for sh in S.pack_shards(DESIGN, wave=1):
        assert len(sh["lanes"]) == 2
        for lane, est in zip(sh["lanes"], sh["est_minutes"]):
            groups = {S.group_key(by_id[r]) for r in lane}
            assert est <= S.LANE_BUDGET_MIN or len(groups) == 1
            assert abs(est - sum(by_id[r].est_minutes for r in lane)) < 1e-6


def test_groups_not_split():
    by_id = {s.run_id: s for s in DESIGN}
    lane_of = {}
    for sh in S.pack_shards(DESIGN, wave=1):
        for li, lane in enumerate(sh["lanes"]):
            for r in lane:
                lane_of[r] = (sh["shard_id"], li)
    seen = {}
    for r, where in lane_of.items():
        g = S.group_key(by_id[r])
        assert seen.setdefault(g, where) == where, g


def test_deterministic():
    a = json.dumps(S.pack_shards(DESIGN, wave=2), sort_keys=True)
    b = json.dumps(S.pack_shards(list(reversed(DESIGN)), wave=2), sort_keys=True)
    assert a == b


def test_min_shards_spreads_wave0():
    shards = S.pack_shards(DESIGN, wave=0, min_shards=4)
    assert len(shards) == 4
    assert sorted(_ids(shards)) == sorted(T.WAVE0_RUN_IDS)


def test_timings_override_estimates():
    spec = next(s for s in DESIGN if s.wave == 1 and s.backbone == "resnet34")
    t = {S.timing_key(spec): 99.0}
    shards = S.pack_shards(DESIGN, wave=1, timings=t)
    by_id = {s.run_id: s for s in DESIGN}
    for sh in shards:
        for lane, est in zip(sh["lanes"], sh["est_minutes"]):
            if spec.run_id in lane:
                assert est >= 99.0


def test_retry_covers_missing_only():
    target = [s.run_id for s in DESIGN if s.wave == 1][:5]
    shards = S.make_retry_shards({"missing": target[:3], "error": target[3:]}, DESIGN, n=1)
    assert sorted(_ids(shards)) == sorted(target)
    assert all(sh["shard_id"].startswith("R1-") for sh in shards)


def test_assign_accounts_balances():
    shards = S.pack_shards(DESIGN, wave=1)
    plan = S.assign_accounts(shards, ["acct1", "acct2", "acct3", "acct4"])
    assert sorted(x for v in plan.values() for x in v) == sorted(sh["shard_id"] for sh in shards)
    wall = {a: sum(max(sh["est_minutes"]) for sh in shards if sh["shard_id"] in ids)
            for a, ids in plan.items()}
    assert max(wall.values()) - min(wall.values()) <= S.LANE_BUDGET_MIN
