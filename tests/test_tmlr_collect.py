"""TMLR plan Task 10 — merge shard outputs from 4 accounts into results/tmlr/."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tmlr_collect as CO  # noqa: E402
import tmlr_design as T  # noqa: E402

DESIGN = {s.run_id: s for s in T.build_design(e5_selection=None)}
R1, R2 = T.WAVE0_RUN_IDS[2], T.WAVE0_RUN_IDS[3]
MANIFEST = {"runs": [{"run_id": r, "experiment": DESIGN[r].experiment,
                      "merged_config_sha256": "sha-" + r, "est_minutes": 9.0,
                      "dataset": DESIGN[r].dataset, "backbone": DESIGN[r].backbone,
                      "arm": DESIGN[r].arm, "k_shot": DESIGN[r].k_shot, "mode": "test"}
                     for r in (R1, R2)]}


def _shard_dir(root: Path, name: str, runs: dict, *, guard="exact") -> Path:
    """runs: run_id -> config sha recorded in the run's metrics.json"""
    d = root / name
    for rid, sha in runs.items():
        rd = d / DESIGN[rid].experiment / rid
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(json.dumps({
            "meta": {"run_id": rid, "merged_config_sha256": sha},
            "regression_guard": {"status": guard}, "wall_seconds": 300.0, "n_params": 1}))
        (rd / "adapter.pt").write_bytes(b"a")
    return d


def _zip(folder: Path) -> Path:
    z = folder.with_suffix(".zip")
    lines = []
    with zipfile.ZipFile(z, "w") as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = p.relative_to(folder)
                zf.write(p, rel)
                lines.append(f"{rel}  sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
        zf.writestr("MANIFEST.txt", "\n".join(lines))
    return z


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST))
    return p


def test_coverage_counts(tmp_path, manifest):
    inc = tmp_path / "incoming"
    _zip(_shard_dir(inc, "W0-01", {R1: "sha-" + R1}))
    rep = CO.collect(inc, tmp_path / "dest", manifest)
    cov = json.load(open(tmp_path / "dest/_coverage.json"))
    assert cov["e1"]["done"] == [R1] and cov["e1"]["missing"] == [R2]
    assert (tmp_path / "dest/e1" / R1 / "adapter.pt").exists()
    assert rep["copied"] == 1


def test_identical_duplicate_ok(tmp_path, manifest):
    inc = tmp_path / "incoming"
    _shard_dir(inc, "acct1", {R1: "sha-" + R1})
    _shard_dir(inc, "acct2", {R1: "sha-" + R1})
    rep = CO.collect(inc, tmp_path / "dest", manifest)
    assert rep["copied"] == 1 and rep["duplicates_identical"] == 1


def test_conflicting_duplicate_raises(tmp_path, manifest):
    inc = tmp_path / "incoming"
    _shard_dir(inc, "acct1", {R1: "sha-" + R1})
    d2 = _shard_dir(inc, "acct2", {R1: "sha-" + R1})
    p = d2 / DESIGN[R1].experiment / R1 / "metrics.json"
    rec = json.loads(p.read_text()); rec["n_params"] = 2; p.write_text(json.dumps(rec))
    with pytest.raises(CO.ConflictError):
        CO.collect(inc, tmp_path / "dest", manifest)


def test_stale_config_flagged(tmp_path, manifest):
    inc = tmp_path / "incoming"
    _shard_dir(inc, "W0-01", {R1: "some-other-sha"})
    CO.collect(inc, tmp_path / "dest", manifest)
    cov = json.load(open(tmp_path / "dest/_coverage.json"))
    assert cov["e1"]["stale_code"] == [R1] and R1 not in cov["e1"]["done"]


def test_bad_zip_hash_rejected(tmp_path, manifest):
    inc = tmp_path / "incoming"
    z = _zip(_shard_dir(inc, "W0-01", {R1: "sha-" + R1}))
    import shutil
    shutil.rmtree(inc / "W0-01")
    bad = inc / "bad.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            dst.writestr(item, data + b" " if item.filename.endswith("metrics.json") else data)
    z.unlink()
    with pytest.raises(CO.IntegrityError):
        CO.collect(inc, tmp_path / "dest", manifest)


def test_timings_written(tmp_path, manifest):
    inc = tmp_path / "incoming"
    _shard_dir(inc, "W0-01", {R1: "sha-" + R1, R2: "sha-" + R2})
    CO.collect(inc, tmp_path / "dest", manifest)
    t = json.load(open(tmp_path / "dest/_timings.json"))
    assert all(v == pytest.approx(5.0) for v in t.values())     # 300 s -> 5 min
    reg = json.load(open(tmp_path / "dest/_regression.json"))
    assert reg["exact"] == sorted([R1, R2])
