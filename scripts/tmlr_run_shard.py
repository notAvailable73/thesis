"""TMLR follow-up — run one lane of one shard (plan Task 8). One process per GPU.

    CUDA_VISIBLE_DEVICES=0 python scripts/tmlr_run_shard.py \
        --shard-file configs/tmlr/shards_w1.json --shard-id W1-03 --lane 0 \
        --out-dir /kaggle/working/tmlr_out/W1-03 --max-minutes 600

Per run, in-process (train.py's main via rq_drivers.train_cell):
train -> evaluate (factorial: every objective x score, plus MaxLogit and Mahalanobis; or
VAL-only for E5-val) -> re-check the parameter budget -> regression guard
-> atomically publish <out>/<exp>/<run_id>/metrics.json (that file existing
== the run is done) -> keep the small adapter.pt, delete the full checkpoint.

A failing run never stops the lane; the time budget stops *starting* runs, so
a Kaggle session never dies in the middle of a run it could not finish.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tmlr_design as T  # noqa: E402

#: Same threshold as rq_core.regression_guard: float32 noise is ~1e-7, a real
#: logic change moves metrics by >= 1e-2.
GUARD_TOL = 1e-6
_NOT_COMPARED = ("logits_dump", "logits_dump_bytes")


def factorial_guard(summary: dict, ref_path: Path, *, tol: float = GUARD_TOL) -> dict:
    """Compare a new factorial summary with a committed rq3_matched record on
    every numeric key both contain (new keys such as maxlogit are ignored)."""
    ref = json.load(open(ref_path))["summary"]
    diffs = {}
    for k, v in ref.items():
        if k in _NOT_COMPARED or k not in summary:
            continue
        if isinstance(v, (int, float)) and isinstance(summary[k], (int, float)):
            diffs[k] = abs(float(v) - float(summary[k]))
    if not diffs:
        return {"status": "no_comparable_keys", "path": str(ref_path)}
    worst = max(diffs, key=diffs.get)
    n_exact = sum(1 for d in diffs.values() if d == 0.0)
    status = ("exact" if n_exact == len(diffs)
              else "within_tol" if diffs[worst] <= tol else "MISMATCH")
    return {"status": status, "tol": tol, "n_keys": len(diffs), "n_exact": n_exact,
            "max_abs_diff": diffs[worst], "max_abs_diff_key": worst, "path": str(ref_path)}


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**entry, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                           sort_keys=True) + "\n")


def run_lane(specs: list[T.RunSpec], out_dir: Path, *, train_fn: Callable, eval_fn: Callable,
             val_fn: Callable, ckpt_fn: Callable, lane: int, log=print,
             max_minutes: float | None = None, clock=time.monotonic,
             persist_logits: bool = True, sha_fn: Callable | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runlog = out_dir / f"_lane{lane}_runlog.jsonl"
    counts = {"ok": 0, "skipped_done": 0, "error": 0, "not_run": 0}
    t0 = clock()
    for i, s in enumerate(specs):
        run_dir = out_dir / s.experiment / s.run_id
        final = run_dir / "metrics.json"
        if final.exists():
            counts["skipped_done"] += 1
            continue
        if max_minutes is not None and (clock() - t0) / 60.0 > max_minutes:
            counts["not_run"] = len(specs) - i
            log(f"[lane {lane}] time budget {max_minutes} min reached; "
                f"{counts['not_run']} run(s) left for a retry shard")
            break
        log(f"[lane {lane}] ({i + 1}/{len(specs)}) {s.run_id}")
        partial = run_dir / "metrics.partial.json"
        started = time.monotonic()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_fn(s)
            if not Path(ckpt).exists():
                train_fn(s)
            if s.mode == "val_only":
                rec = val_fn(s, ckpt, partial)
            else:
                rec = eval_fn(s, ckpt, partial,
                              logits_out=(run_dir / "logits.npz") if persist_logits else None,
                              adapter_out=run_dir / "adapter.pt")
            if int(rec["n_params"]) != int(s.expected_n_params):
                raise RuntimeError(f"parameter budget drifted: built {rec['n_params']}, "
                                   f"expected {s.expected_n_params}")
            # Provenance the collector checks: which run, from which exact config.
            rec["meta"] = {**(rec.get("meta") or {}), "run_id": s.run_id,
                           "merged_config_sha256": sha_fn(s) if sha_fn else None}
            if s.regression_kind == "factorial":
                rec["regression_guard"] = factorial_guard(rec["summary"], REPO / s.regression_ref)
            partial.write_text(json.dumps(rec, indent=2, sort_keys=True))
            partial.replace(final)
            if s.arm != "full_ft" and Path(ckpt).exists():
                Path(ckpt).unlink()
            guard = (rec.get("regression_guard") or {}).get("status")
            counts["ok"] += 1
            _append(runlog, {"run_id": s.run_id, "status": "ok", "guard": guard,
                             "wall_seconds": round(time.monotonic() - started, 1)})
            log(f"[lane {lane}]   ok  guard={guard}  [{time.monotonic() - started:.0f}s]")
        except Exception as e:  # noqa: BLE001 — one bad run must not kill the lane
            counts["error"] += 1
            if partial.exists():
                partial.replace(run_dir / "metrics.error.json")
            _append(runlog, {"run_id": s.run_id, "status": "error", "error": repr(e),
                             "wall_seconds": round(time.monotonic() - started, 1)})
            log(f"[lane {lane}]   ERROR {e!r}")
    (out_dir / f"_lane{lane}_summary.json").write_text(json.dumps(counts, sort_keys=True))
    log(f"[lane {lane}] done: {counts}")
    return counts


def shard_report(shard: dict, out_dir: Path, design: dict) -> dict:
    expected = [r for lane in shard["lanes"] for r in lane]
    done = sorted(r for r in expected
                  if (out_dir / design[r].experiment / r / "metrics.json").exists())
    last_status = {}
    for p in sorted(out_dir.glob("_lane*_runlog.jsonl")):
        for line in p.read_text().splitlines():
            e = json.loads(line)
            last_status[e["run_id"]] = e["status"]
    error = sorted(r for r in expected if r not in done and last_status.get(r) == "error")
    guards: dict = {}
    for r in done:
        rec = json.load(open(out_dir / design[r].experiment / r / "metrics.json"))
        st = (rec.get("regression_guard") or {}).get("status")
        if st and st != "not_requested":
            guards.setdefault(st, []).append(r)
    return {"shard_id": shard["shard_id"], "expected": len(expected), "done": done,
            "error": error, "missing": sorted(set(expected) - set(done)), "guards": guards}


# =====================================================================
# Real wiring (Kaggle)
# =====================================================================
def _real_fns(device, num_episodes: int):
    import rq_drivers as D
    from scripts.train import _checkpoint_tag
    from src.utils import load_config

    def ckpt_fn(s):
        cfg = load_config(REPO / s.config)
        return REPO / cfg.output.checkpoint_dir / f"model_phase2_{_checkpoint_tag(cfg)}.pt"

    def train_fn(s):
        D.train_cell(REPO / s.config, wandb_mode="disabled")

    def eval_fn(s, ckpt, out_json, *, logits_out, adapter_out):
        committed = REPO / s.regression_ref if s.regression_kind == "grid" else None
        return D.factorial_run_one(
            REPO, REPO / s.config, ckpt, out_json, device=device, num_episodes=num_episodes,
            logits_out=logits_out, committed_metrics=committed, meta=s.to_json(),
            adapter_out=adapter_out, fit_mahalanobis=True)

    def val_fn(s, ckpt, out_json):
        return D.val_only_run_one(REPO, REPO / s.config, ckpt, out_json, device=device)

    return train_fn, eval_fn, val_fn, ckpt_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-file", type=Path, required=True)
    ap.add_argument("--shard-id", required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--num-episodes", type=int, default=600)
    ap.add_argument("--no-logits", action="store_true")
    a = ap.parse_args(argv)

    import torch
    os.chdir(REPO)       # train.py writes checkpoints relative to the cwd
    shards = {sh["shard_id"]: sh for sh in json.load(open(REPO / a.shard_file))["shards"]}
    lane_ids = shards[a.shard_id]["lanes"][a.lane]
    design = {s.run_id: s for s in T.build_design()}
    specs = [design[r] for r in lane_ids]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lane {a.lane}] shard {a.shard_id}: {len(specs)} runs on {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    train_fn, eval_fn, val_fn, ckpt_fn = _real_fns(device, a.num_episodes)
    from tmlr_configs import merged_sha256
    counts = run_lane(specs, a.out_dir, train_fn=train_fn, eval_fn=eval_fn, val_fn=val_fn,
                      ckpt_fn=ckpt_fn, lane=a.lane, max_minutes=a.max_minutes,
                      persist_logits=not a.no_logits, sha_fn=merged_sha256,
                      log=lambda *x: print(*x, flush=True))
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
