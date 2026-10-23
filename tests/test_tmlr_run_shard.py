"""TMLR plan Task 8 — lane runner. Training/evaluation are faked; this tests
the orchestration: resume, error isolation, time budget, parameter re-check,
VAL-only isolation, output layout, shard report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tmlr_design as T  # noqa: E402
import tmlr_run_shard as RS  # noqa: E402

DESIGN = {s.run_id: s for s in T.build_design(e5_selection=None)}
RUNS = [r for r in T.WAVE0_RUN_IDS if not r.startswith("e5val")][:3]
VAL_RUN = next(r for r in T.WAVE0_RUN_IDS if r.startswith("e5val"))


class Fakes:
    def __init__(self, tmp: Path, fail_on=(), n_params_off=0):
        self.tmp, self.fail_on, self.off = tmp, set(fail_on), n_params_off
        self.trained, self.evaluated, self.val = [], [], []

    def ckpt(self, spec):
        return self.tmp / "ckpt" / f"{spec.run_id}.pt"

    def train(self, spec):
        if spec.run_id in self.fail_on:
            raise RuntimeError("boom")
        p = self.ckpt(spec)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        self.trained.append(spec.run_id)

    def evaluate(self, spec, ckpt, out_json, logits_out, adapter_out):
        self.evaluated.append(spec.run_id)
        rec = {"n_params": spec.expected_n_params + self.off, "summary": {"x": 1.0},
               "regression_guard": {"status": "not_requested"}}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rec))
        adapter_out.write_text("a")
        return rec

    def val_only(self, spec, ckpt, out_json):
        self.val.append(spec.run_id)
        rec = {"n_params": spec.expected_n_params, "val_ece_pooled__evidential_native": 0.2}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rec))
        return rec


def _run(tmp, fakes, runs, **kw):
    return RS.run_lane([DESIGN[r] for r in runs], tmp / "out", train_fn=fakes.train,
                       eval_fn=fakes.evaluate, val_fn=fakes.val_only, ckpt_fn=fakes.ckpt,
                       log=lambda *a: None, lane=0, **kw)


def test_outputs_layout(tmp_path):
    f = Fakes(tmp_path)
    summary = _run(tmp_path, f, RUNS)
    assert summary["ok"] == 3
    for r in RUNS:
        d = tmp_path / "out" / DESIGN[r].experiment / r
        assert (d / "metrics.json").exists() and (d / "adapter.pt").exists()
        assert not f.ckpt(DESIGN[r]).exists()          # full checkpoint deleted after success
    log = [json.loads(l) for l in (tmp_path / "out/_lane0_runlog.jsonl").read_text().splitlines()]
    assert [e["status"] for e in log] == ["ok"] * 3


def test_skips_done_runs(tmp_path):
    f = Fakes(tmp_path)
    _run(tmp_path, f, RUNS[:1])
    f2 = Fakes(tmp_path)
    summary = _run(tmp_path, f2, RUNS)
    assert summary["skipped_done"] == 1 and f2.trained == RUNS[1:]


def test_error_in_one_run_does_not_stop_lane(tmp_path):
    f = Fakes(tmp_path, fail_on={RUNS[0]})
    summary = _run(tmp_path, f, RUNS)
    assert summary["error"] == 1 and summary["ok"] == 2
    assert not (tmp_path / "out" / DESIGN[RUNS[0]].experiment / RUNS[0] / "metrics.json").exists()


def test_budget_stops_new_runs(tmp_path):
    ticks = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    f = Fakes(tmp_path)
    summary = _run(tmp_path, f, RUNS, max_minutes=5 / 60, clock=lambda: next(ticks))
    assert summary["ok"] == 1 and summary["not_run"] == 2


def test_param_drift_marks_error(tmp_path):
    f = Fakes(tmp_path, n_params_off=1)
    summary = _run(tmp_path, f, RUNS[:1])
    assert summary["error"] == 1
    d = tmp_path / "out" / DESIGN[RUNS[0]].experiment / RUNS[0]
    assert not (d / "metrics.json").exists() and (d / "metrics.error.json").exists()


def test_val_only_routes_to_val_fn(tmp_path):
    f = Fakes(tmp_path)
    summary = _run(tmp_path, f, [VAL_RUN])
    assert summary["ok"] == 1 and f.val == [VAL_RUN] and f.evaluated == []


def test_factorial_guard(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"summary": {"a": 0.5, "b": 0.25, "logits_dump": "x"}}))
    assert RS.factorial_guard({"a": 0.5, "b": 0.25, "maxlogit_new": 3.0}, ref)["status"] == "exact"
    assert RS.factorial_guard({"a": 0.5, "b": 0.2500004}, ref)["status"] == "within_tol"
    assert RS.factorial_guard({"a": 0.6, "b": 0.25}, ref)["status"] == "MISMATCH"


def test_shard_report_lists_missing(tmp_path):
    f = Fakes(tmp_path, fail_on={RUNS[1]})
    _run(tmp_path, f, RUNS)
    shard = {"shard_id": "W0-01", "lanes": [RUNS, [VAL_RUN]]}
    rep = RS.shard_report(shard, tmp_path / "out", DESIGN)
    assert rep["done"] == sorted([RUNS[0], RUNS[2]])
    assert rep["error"] == [RUNS[1]]
    assert rep["missing"] == sorted([RUNS[1], VAL_RUN])


def test_provenance_recorded(tmp_path):
    f = Fakes(tmp_path)
    RS.run_lane([DESIGN[RUNS[0]]], tmp_path / "out", train_fn=f.train, eval_fn=f.evaluate,
                val_fn=f.val_only, ckpt_fn=f.ckpt, log=lambda *a: None, lane=0,
                sha_fn=lambda s: "sha-" + s.run_id)
    rec = json.loads((tmp_path / "out" / DESIGN[RUNS[0]].experiment / RUNS[0]
                      / "metrics.json").read_text())
    assert rec["meta"] == {"run_id": RUNS[0], "merged_config_sha256": "sha-" + RUNS[0]}


def test_real_sha_matches_manifest():
    """The sha the runner records must be exactly what the manifest holds."""
    import tmlr_configs as C
    man = {e["run_id"]: e for e in json.load(open(REPO / "configs/tmlr/manifest.json"))["runs"]}
    for rid in T.WAVE0_RUN_IDS:
        assert C.merged_sha256(DESIGN[rid]) == man[rid]["merged_config_sha256"]


def test_factorial_guard_on_committed_rq3m_record():
    """Reviewer finding 1: a committed rq3_matched record fed its own summary
    (plus new keys the new evaluator adds) must come back exact."""
    for s in [x for x in DESIGN.values() if x.regression_kind == "factorial"][:4]:
        ref = REPO / s.regression_ref
        summary = dict(json.load(open(ref))["summary"])
        summary["ood_auroc__svhn_far__maxlogit"] = 0.5
        assert RS.factorial_guard(summary, ref)["status"] == "exact", s.run_id


def test_prepare_data_once_per_dataset_and_serial(monkeypatch):
    """Every dataset is prepared once, in this process, before lanes start."""
    import rq_core as R
    calls = []
    monkeypatch.setattr(R, "cached_id_split", lambda d, split: calls.append(("id", d["name"], split)) or [0] * 3)
    import torch
    monkeypatch.setattr(R, "ood_image_sets", lambda cfg: calls.append(("ood", cfg.dataset.name))
                        or {"svhn_far": torch.zeros(2, 3, 4, 4)})
    specs = [DESIGN[r] for r in T.WAVE0_RUN_IDS]                       # cifar + mini, many runs
    out = RS.prepare_data(specs, log=lambda *a: None)
    assert set(out) == {"cifar_fs", "mini_imagenet"}
    assert sum(1 for c in calls if c[0] == "id") == 6                   # 2 datasets x 3 splits
    assert sorted(c[1] for c in calls if c[0] == "ood") == ["cifar_fs", "mini_imagenet"]


def test_ood_disk_cache_shared_and_identical(tmp_path, monkeypatch):
    """A second process (simulated by clearing the in-process cache) loads the
    disk copy instead of rebuilding, and gets identical bytes."""
    import rq_core as R
    import torch
    monkeypatch.setenv(R.OOD_DISK_CACHE_ENV, str(tmp_path))
    built = []
    def builder():
        built.append(1)
        return torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    R.clear_caches()
    a = R._cached_ood_images(("tin", "x", 1), builder)
    R.clear_caches()                                    # "another lane"
    b = R._cached_ood_images(("tin", "x", 1), builder)
    assert built == [1] and torch.equal(a, b)
    assert not list(tmp_path.glob("*.tmp*"))
    R.clear_caches()


def test_mini_cache_write_is_atomic(tmp_path, monkeypatch):
    import numpy as np
    import src.datasets.mini_imagenet as mi
    imgs = np.zeros((2, 84, 84, 3), dtype=np.uint8)
    monkeypatch.setattr(mi, "_build_split_cache", lambda root, split, wnids: (imgs, ["a", "b"], "test"))
    arr, labels = mi._ensure_cache(str(tmp_path), "val", {"val": ["a", "b"]})
    assert arr.shape == (2, 84, 84, 3) and labels == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp*"))                            # temp files renamed away
