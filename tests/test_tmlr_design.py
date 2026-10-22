"""TMLR plan Task 7 — experiment design, config generation, manifest.

The design is the single source of every run. These tests pin its counts,
its budgets (closed form vs the instantiated module), the recipe control, and
that nothing frozen or pre-existing is touched.
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tmlr_configs as C  # noqa: E402
import tmlr_design as T  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.utils.seed import effective_train_seed_offset  # noqa: E402


@pytest.fixture(scope="module")
def design():
    return T.build_design(e5_selection=None)


def test_run_counts(design):
    d = Counter(s.experiment for s in design)
    assert (d["e1"], d["e2"], d["e3a"], d["e3b"], d["e4"], d["e5val"], d["e5"]) == (
        170, 180, 48, 94, 112, 6, 48)
    assert len(design) == 658


def test_every_backbone_has_unmatched_pair(design):
    for exp, bbs in (("e1", ("resnet18", "mobilenetv3_small")),
                     ("e2", ("resnet34", "mobilenetv2")),
                     ("e3b", ("resnet18", "mobilenetv3_small"))):
        pool = [s for s in design if s.experiment in (exp, "e3a")]
        for bb in bbs:
            for arm in ("btl_ms", "lora_ss"):
                assert any(s.backbone == bb and s.arm == arm and s.rank_spec == 16
                           and s.k_shot == 5 for s in pool), (exp, bb, arm)


def test_run_ids_unique(design):
    ids = [s.run_id for s in design]
    assert len(ids) == len(set(ids))


def test_no_duplicate_trainings(design):
    """Two specs that would train the identical model must not both exist."""
    keys = [T.training_identity(s) for s in design if s.wave >= 0]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("key", sorted(T.LEVELS))
def test_level_spread_within_tolerance(key):
    backbone, _level = key
    params = [T.expected_params(backbone, arm, rs, "softmax")
              for arm, rs in T.LEVELS[key].items()]
    assert max(params) / min(params) - 1 <= T.MAX_LEVEL_SPREAD


def test_budget_table_matches_plan():
    assert T.expected_params("resnet18", "lora_ms", [1, 1, 2, 1], "softmax") == 12160
    assert T.expected_params("resnet18", "btl_ss", 40, "softmax") == 31272
    assert T.expected_params("mobilenetv2", "btl_ms", 16, "evidential") == 12490
    assert T.expected_params("mobilenetv2", "lora_ss", 16, "softmax") == 20480


def test_expected_params_match_instantiated(design):
    """Closed forms (hardcoded shape tables) vs the real modules, for every
    distinct (backbone, arm, rank_spec, head) in the design."""
    report = C.assert_budgets(design)
    assert report["ok"], report["failures"]
    assert report["n_checked"] >= 60


def test_control_guard(design):
    report = C.control_guard(design)
    assert report["ok"], report["failures"]


_BOOKKEEPING = ("output", "wandb")


def _strip(cfg) -> dict:
    d = {k: v for k, v in dict(cfg).items() if k not in _BOOKKEEPING}
    d["trainer"] = {k: v for k, v in dict(d["trainer"]).items()
                    if k != "vary_train_stream_with_seed"}
    return d


def test_e4_wraps_original_grid(design):
    e4 = [x for x in design if x.experiment == "e4"]
    assert len(e4) == 112
    for s in e4:
        assert _strip(C.merged_config(s)) == _strip(load_config(REPO / s.parent)), s.run_id


def test_regression_ref_recipe_identical(design):
    """Every run with a regression reference trains exactly the referenced
    config's recipe (all 130), not just the same stream offset."""
    import json
    grid = {c["results_json"]: c["config"] for c in T._grid_cells()}
    n = 0
    for s in design:
        if not s.regression_ref:
            continue
        ref_cfg = (grid[s.regression_ref] if s.regression_kind == "grid"
                   else json.load(open(REPO / s.regression_ref))["config"])
        assert _strip(C.merged_config(s)) == _strip(load_config(REPO / ref_cfg)), s.run_id
        n += 1
    assert n == 130


def test_regression_refs_assigned(design):
    refs = [s for s in design if s.regression_ref]
    # 112 E4 cells + 2 E1 MNv3 lora_ss r16 seed-42 twins (no rq3_matched record) vs grid;
    # 16 E1 seed-42 existing arms vs rq3_matched.
    assert Counter(s.regression_kind for s in refs) == Counter({"grid": 114, "factorial": 16})
    for s in refs:
        assert (REPO / s.regression_ref).exists(), s.regression_ref


def test_seed42_vary_matches_rq3_matched(design):
    for s in [x for x in design if x.regression_kind == "factorial"]:
        merged = C.merged_config(s)
        assert effective_train_seed_offset(merged) == 20000
        assert s.seed == 42 and s.vary_stream


def test_wave0_runs(design):
    by_id = {s.run_id: s for s in design}
    assert len(T.WAVE0_RUN_IDS) == 9
    assert all(by_id[r].wave == 0 for r in T.WAVE0_RUN_IDS)
    assert sum(1 for r in T.WAVE0_RUN_IDS if by_id[r].regression_ref) == 3
    assert not any(by_id[r].dataset == "fgvc_aircraft" for r in T.WAVE0_RUN_IDS)


def test_e5_blocked_until_selected(design):
    e5 = [s for s in design if s.experiment == "e5"]
    assert e5 and all(s.wave == -1 for s in e5)
    chosen = T.build_design(e5_selection={"kl_weight_max": 0.05, "prior_per_class": 0.5,
                                          "skipped": False})
    assert all(s.wave == 2 for s in chosen if s.experiment == "e5")
    skipped = T.build_design(e5_selection={"kl_weight_max": 0.1, "prior_per_class": 1.0,
                                           "skipped": True})
    assert not [s for s in skipped if s.experiment == "e5"]


def test_select_e5_rule():
    recs = [
        {"kl_weight_max": 0.1, "prior_per_class": 1.0, "val_ece": 0.300, "val_acc": 0.90},
        {"kl_weight_max": 0.05, "prior_per_class": 1.0, "val_ece": 0.251, "val_acc": 0.89},
        {"kl_weight_max": 0.0, "prior_per_class": 0.5, "val_ece": 0.2525, "val_acc": 0.91},
    ]
    sel = C.choose_e5(recs)
    assert (sel["kl_weight_max"], sel["prior_per_class"]) == (0.0, 0.5)   # tie -> higher acc
    assert sel["skipped"] is False
    only_current = C.choose_e5([recs[0]])
    assert only_current["skipped"] is True


def test_frozen_files_untouched(tmp_path, design):
    """write_configs writes only under <root>/configs/tmlr and never touches the
    frozen episode files (checked in the real repo, not just the tmp root)."""
    frozen = [REPO / "configs/test_episodes.yaml", REPO / "configs/val_episodes.yaml"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen]
    C.write_configs([s for s in design if s.experiment == "e1"][:4], root=tmp_path)
    after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen]
    assert before == after
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.is_relative_to(tmp_path / "configs/tmlr") for p in written)
    assert len(list((tmp_path / "configs/tmlr/e1").glob("*.yaml"))) == 4
