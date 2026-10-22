"""TMLR follow-up — merge shard outputs (zips or folders, from any account) into
results/tmlr/ and report coverage (plan Task 10).

    python scripts/tmlr_collect.py --incoming incoming/

Refuses silently-wrong merges:
  - a zip whose MANIFEST.txt sha256 does not match its contents -> IntegrityError
  - a run whose recorded merged-config sha256 differs from configs/tmlr/manifest.json
    (wrong GIT_SHA / edited config) -> reported as `stale_code`, not copied
  - the same run present twice with different results -> ConflictError
Writes results/tmlr/_coverage.json, _timings.json (for re-packing shards) and
_regression.json (every reproduction guard by status).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tmlr_design as T  # noqa: E402
from tmlr_shards import timing_key  # noqa: E402

RUN_FILES = ("metrics.json", "adapter.pt", "logits.npz")
#: Keys that legitimately differ between two executions of the same run.
_VOLATILE = ("wall_seconds", "env")


class IntegrityError(RuntimeError):
    pass


class ConflictError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _unpack(zip_path: Path, staging: Path) -> Path:
    out = staging / zip_path.stem
    if out.exists():
        shutil.rmtree(out)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    man = out / "MANIFEST.txt"
    if man.exists():
        for line in man.read_text().splitlines():
            if "sha256=" not in line:
                continue
            rel, digest = line.split("  sha256=")
            rel = rel.split("  ")[0].strip()
            f = out / rel
            if not f.exists() or _sha(f) != digest.strip():
                raise IntegrityError(f"{zip_path.name}: {rel} does not match its MANIFEST hash")
    return out


def _canonical(metrics_path: Path) -> str:
    rec = json.load(open(metrics_path))
    for k in _VOLATILE:
        rec.pop(k, None)
    return json.dumps(rec, sort_keys=True)


def collect(incoming: Path, dest: Path = REPO / "results/tmlr",
            manifest: Path = REPO / "configs/tmlr/manifest.json") -> dict:
    runs = {e["run_id"]: e for e in json.load(open(manifest))["runs"]}
    design = {s.run_id: s for s in T.build_design(e5_selection=None)}
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / "_staging"
    roots = [p for p in sorted(incoming.iterdir()) if p.is_dir() and p != staging]
    for z in sorted(incoming.rglob("*.zip")):
        roots.append(_unpack(z, staging))

    found: dict[str, list[Path]] = {}
    errors: set[str] = set()
    for root in roots:
        for m in sorted(root.rglob("metrics.json")):
            rid = m.parent.name
            if rid in runs:
                found.setdefault(rid, []).append(m.parent)
        for log in sorted(root.rglob("_lane*_runlog.jsonl")):
            for line in log.read_text().splitlines():
                e = json.loads(line)
                if e.get("status") == "error":
                    errors.add(e["run_id"])

    report = {"copied": 0, "duplicates_identical": 0, "stale_code": []}
    for rid, dirs in sorted(found.items()):
        canon = {_canonical(d / "metrics.json") for d in dirs}
        existing = dest / runs[rid]["experiment"] / rid / "metrics.json"
        if existing.exists():
            canon.add(_canonical(existing))
        if len(canon) > 1:
            raise ConflictError(f"{rid}: differing results in {[str(d) for d in dirs]}")
        rec = json.load(open(dirs[0] / "metrics.json"))
        if (rec.get("meta") or {}).get("merged_config_sha256") != runs[rid]["merged_config_sha256"]:
            report["stale_code"].append(rid)
            continue
        report["duplicates_identical"] += len(dirs) - 1
        target = existing.parent
        target.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (dirs[0] / name).exists():
                shutil.copy2(dirs[0] / name, target / name)
        report["copied"] += 1

    coverage: dict = {}
    for rid, e in sorted(runs.items()):
        c = coverage.setdefault(e["experiment"], {"done": [], "missing": [], "error": [],
                                                  "stale_code": []})
        if (dest / e["experiment"] / rid / "metrics.json").exists():
            c["done"].append(rid)
        elif rid in report["stale_code"]:
            c["stale_code"].append(rid)
        else:
            c["missing"].append(rid)
            if rid in errors:
                c["error"].append(rid)
    (dest / "_coverage.json").write_text(json.dumps(coverage, indent=1, sort_keys=True))

    minutes: dict[str, list[float]] = {}
    regression: dict[str, list[str]] = {}
    for exp, c in coverage.items():
        for rid in c["done"]:
            rec = json.load(open(dest / exp / rid / "metrics.json"))
            if rid in design and "wall_seconds" in rec:
                minutes.setdefault(timing_key(design[rid]), []).append(rec["wall_seconds"] / 60)
            st = (rec.get("regression_guard") or {}).get("status")
            if st and st != "not_requested":
                regression.setdefault(st, []).append(rid)
    (dest / "_timings.json").write_text(json.dumps(
        {k: statistics.median(v) for k, v in sorted(minutes.items())}, indent=1))
    (dest / "_regression.json").write_text(json.dumps(
        {k: sorted(v) for k, v in sorted(regression.items())}, indent=1))
    report["coverage"] = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in coverage.items()}
    report["regression"] = {k: len(v) for k, v in regression.items()}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", type=Path, required=True)
    ap.add_argument("--dest", type=Path, default=REPO / "results/tmlr")
    a = ap.parse_args(argv)
    rep = collect(a.incoming, a.dest)
    print(json.dumps(rep, indent=1, sort_keys=True))
    bad = rep["stale_code"] or rep["regression"].get("MISMATCH")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
