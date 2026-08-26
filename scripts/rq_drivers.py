"""Phase 0 / A / B drivers — checkpoint recovery, factorial evaluation, rank sweep.

Every driver is resumable (skips a cell whose output JSON already exists) and
budget-aware (`max_minutes` finishes the current cell then stops cleanly), for
the same reason scripts/run_mvt_grid.py is: a hosted-notebook session can be
killed at any moment and nothing already paid for should be lost.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import rq_core as R
from src.utils import load_config, set_seed, count_trainable_params
from src.models import build_model
from src.evaluators import fit_temperature
from scripts.evaluate import _load_test_seeds

#: The frozen VAL seeds. Hard-coded here ONLY as an assertion target: the
#: fitting code reads configs/val_episodes.yaml, and this is the tripwire that
#: proves it never silently drifted onto the 600 test seeds (T2.5).
EXPECTED_VAL_SEED_RANGE = (10000, 10099)


# =====================================================================
# Phase 0 — T0 checkpoint recovery audit
# =====================================================================
def recover_checkpoints(repo_root: Path, search_roots=("/kaggle/input",),
                        log=print) -> dict:
    """Find Step 10 checkpoints in whatever the session has attached.

    Handles both shapes the Step 10 notebooks could have produced:
      - the artifact ZIPs those notebooks pushed (step10a/b/c packed
        `checkpoints/model_phase2_*.pt` alongside `results/grid/*`), and
      - loose `.pt` files, if a dataset was built by copying rather than zipping.

    ZIPs are extracted to a staging directory and only `.pt` files are copied
    into `checkpoints/`. Committed `results/` files are deliberately NOT
    overwritten: they are the T1.6 regression baseline, and clobbering them
    with a copy of themselves would destroy the only independent check we have.
    """
    ckpt_dir = repo_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    staging = repo_root / "_recovered"
    staging.mkdir(parents=True, exist_ok=True)

    found_zips, loose_pt, copied, skipped = [], [], [], []
    for root in search_roots:
        rp = Path(root)
        if not rp.exists():
            log(f"  (no such path: {root})")
            continue
        for p in sorted(rp.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".zip":
                found_zips.append(p)
            elif p.suffix == ".pt" and p.name.startswith("model_phase2_"):
                loose_pt.append(p)

    log(f"  scanned {search_roots}: {len(found_zips)} zip(s), "
        f"{len(loose_pt)} loose checkpoint(s)")

    for z in found_zips:
        try:
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist()
                           if m.startswith("checkpoints/") and m.endswith(".pt")]
                if not members:
                    log(f"  {z.name}: no checkpoints/ members, skipped")
                    continue
                # Extract only the members we want, and never outside staging.
                for m in members:
                    dest = (staging / m).resolve()
                    if not str(dest).startswith(str(staging.resolve())):
                        log(f"  !! refusing unsafe zip member {m!r} in {z.name}")
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                log(f"  {z.name}: extracted {len(members)} checkpoint(s)")
        except zipfile.BadZipFile:
            log(f"  {z.name}: not a readable zip, skipped")

    candidates = list(staging.rglob("model_phase2_*.pt")) + loose_pt
    for src in candidates:
        dest = ckpt_dir / src.name
        if dest.exists():
            skipped.append(dest.name)
            continue
        shutil.copy2(src, dest)
        copied.append(dest.name)

    return {"zips_seen": [str(p) for p in found_zips],
            "loose_pt_seen": len(loose_pt),
            "copied": copied, "already_present": skipped}


def audit_checkpoints(repo_root: Path, log=print) -> dict:
    """T0.3 — the written verdict, computed rather than assumed."""
    index = json.load(open(repo_root / "configs" / "grid" / "_index.json"))["cells"]
    present, missing = [], []
    for c in index:
        (present if (repo_root / c["checkpoint"]).exists() else missing).append(c)

    by_seed: dict[int, int] = {}
    by_slice: dict[str, list[int]] = {}
    for c in present:
        by_seed[c["seed"]] = by_seed.get(c["seed"], 0) + 1
    for c in index:
        key = f"{c['dataset']}/{c['k_shot']}shot"
        got = by_slice.setdefault(key, [0, 0])
        got[1] += 1
        if (repo_root / c["checkpoint"]).exists():
            got[0] += 1

    n = len(present)
    seeds_present = sorted(by_seed)
    if n == len(index):
        verdict, note = "a", "All 120 recovered — RQ1/RQ2 are evaluation-only."
    elif n == 0:
        verdict, note = "c", ("None recovered — RQ1 needs the grid retrain; RQ2 "
                              "piggybacks on it rather than costing extra.")
    elif seeds_present == [42] and n == 40:
        verdict, note = "b", ("Seed-42 subset only — RQ1/RQ2 run at n=1 seed. "
                              "Report WITHOUT seed error bars and say so.")
    else:
        verdict, note = "b-partial", (f"Partial recovery ({n}/120). Usable, but "
                                      f"state the coverage explicitly.")

    log(f"\n  checkpoints present: {n}/{len(index)}")
    log(f"  by seed: {dict(sorted(by_seed.items())) or '(none)'}")
    for k in sorted(by_slice):
        got, tot = by_slice[k]
        log(f"    {k:<26} {got}/{tot}")
    log(f"\n  VERDICT ({verdict}): {note}")

    return {"n_present": n, "n_total": len(index), "verdict": verdict,
            "note": note, "by_seed": by_seed,
            "by_slice": {k: v for k, v in sorted(by_slice.items())},
            "missing_cells": [c["cell" if "cell" in c else "config"] for c in missing]}


# =====================================================================
# One factorial evaluation
# =====================================================================
def _rel(path, root: Path) -> str:
    """Repo-relative path, falling back to absolute.

    `Path.relative_to` RAISES when the target is outside `root`, and this is
    called while assembling the result record -- i.e. AFTER a 600-episode
    evaluation has already been paid for. Step 11's post-mortem
    (step_writeups/step11.txt Section 8) is exactly this: a crash in
    bookkeeping threw away a correct measurement before it could be saved.
    Bookkeeping must never be able to destroy a result.
    """
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(Path(path).resolve())


def factorial_run_one(repo_root: Path, config_path: Path, checkpoint_path: Path,
                      out_json: Path, *, device, num_episodes: int,
                      logits_out: Path | None, committed_metrics: Path | None,
                      meta: dict | None = None, log=print) -> dict:
    """Evaluate one trained cell under every (objective, score) combination."""
    t0 = time.monotonic()
    cfg = load_config(config_path)
    interp = cfg.head.get("interpretation", "evidential")
    K = int(cfg.dataset.n_way)
    prior_pc = float(cfg.loss.get("prior_per_class", 1.0))

    set_seed(int(cfg.seed))
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    n_params = int(count_trainable_params(model))
    affine_native = R.read_evidence_affine(model.head)

    # --- VAL-only fitting (T2.1 / T2.5) -------------------------------
    val_logits, val_targets, val_seeds = R.load_val_logits(
        model, cfg, device, repo_root)
    lo, hi = EXPECTED_VAL_SEED_RANGE
    if not (val_seeds[0] == lo and val_seeds[-1] == hi):
        raise RuntimeError(
            f"VAL seed guard failed: fitting would use seeds "
            f"{val_seeds[0]}..{val_seeds[-1]}, expected {lo}..{hi}. "
            f"Refusing to continue — this is the repo's hard convention.")
    test_seeds_all = _load_test_seeds(repo_root, cfg)
    if set(val_seeds) & set(test_seeds_all):
        raise RuntimeError("VAL and TEST seed sets overlap — refusing to fit.")

    T = fit_temperature(val_logits, val_targets)
    affine_valfit = R.fit_evidence_affine(
        val_logits, val_targets, num_classes=K, prior_per_class=prior_pc,
        scale_init=affine_native[0], bias_init=affine_native[1])
    log(f"    T={T:.4f}  affine native=({affine_native[0]:.4f}, "
        f"{affine_native[1]:.4f})  refit=({affine_valfit[0]:.4f}, "
        f"{affine_valfit[1]:.4f})")

    pools = R.build_ood_pools(model, cfg, device,
                              use_tinyimagenet=True, use_gaussian=True)
    test_seeds = test_seeds_all[:num_episodes]
    summary = R.factorial_evaluate(
        model, cfg, test_seeds=test_seeds, ood_pools=pools, device=device,
        temperature=T, affine_valfit=affine_valfit, prior_per_class=prior_pc,
        ece_bins=int(cfg.eval.ece_bins), logits_out=logits_out,
        log_every=200, logger_print=log)

    guard = (R.regression_guard(summary, committed_metrics, interp)
             if committed_metrics else {"status": "not_requested"})

    record = {
        "meta": meta or {},
        "config": _rel(config_path, repo_root),
        "checkpoint": _rel(checkpoint_path, repo_root),
        "interpretation": interp,
        "adapter_type": cfg.adapter.type,
        "adapter_rank": int(cfg.adapter.get("rank", -1)),
        "adapter_placement": str(cfg.adapter.get("placement", "post_pool")),
        "backbone": str(cfg.backbone.name),
        "dataset": str(cfg.dataset.get("name", "cifar_fs")),
        "k_shot": int(cfg.dataset.k_shot),
        "seed": int(cfg.seed),
        "n_params": n_params,
        "best_val_epoch": int(ckpt.get("best_val_epoch", -1)),
        "temperature": float(T),
        "affine_native": [float(affine_native[0]), float(affine_native[1])],
        "affine_valfit": [float(affine_valfit[0]), float(affine_valfit[1])],
        "affine_config_init": [float(cfg.head.get("evidence_scale_init", 1.0)),
                               float(cfg.head.get("evidence_bias_init", 0.0))],
        "val_seeds": {"first": int(val_seeds[0]), "last": int(val_seeds[-1]),
                      "n": len(val_seeds)},
        "summary": summary,
        "regression_guard": guard,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


def train_cell(config_path: Path, wandb_mode="disabled", log=print) -> None:
    """Train one cell by calling the REAL scripts/train.py main() in-process.

    In-process, not subprocess: Step 9 found subprocess output can silently
    vanish on hosted notebooks (scripts/run_mvt_grid.py's docstring), and this
    keeps exactly one source of truth for what "a run" means.
    """
    import scripts.train as train_mod
    old = sys.argv
    sys.argv = ["train.py", "--config", str(config_path), "--wandb-mode", wandb_mode]
    try:
        train_mod.main()
    finally:
        sys.argv = old


# =====================================================================
# Phase A — RQ1 + RQ2 over the grid
# =====================================================================
def run_phase_a(repo_root: Path, cells: list[dict], *, device, out_dir: Path,
                logits_dir: Path | None, num_episodes: int, allow_retrain: bool,
                wandb_mode: str, max_minutes: float | None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "_run_log.jsonl"
    started = time.monotonic()
    counts = {"ok": 0, "skipped_done": 0, "no_checkpoint": 0, "error": 0,
              "trained": 0}

    for i, c in enumerate(cells):
        if max_minutes is not None and (time.monotonic() - started) / 60 > max_minutes:
            log(f"[A] budget {max_minutes} min reached; "
                f"{len(cells) - i} cell(s) left for the next session.")
            break

        cell_id = (f"{c['dataset']}_{c['k_shot']}shot_{c['backbone']}_"
                   f"{c['adapter']}_{c['head']}_seed{c['seed']}")
        out_json = out_dir / f"{cell_id}.json"
        if out_json.exists():
            counts["skipped_done"] += 1
            continue

        log(f"[A] ({i + 1}/{len(cells)}) {cell_id}")
        ckpt = repo_root / c["checkpoint"]
        try:
            if not ckpt.exists():
                if not allow_retrain:
                    log("    no checkpoint and ALLOW_RETRAIN=False -> skipped")
                    counts["no_checkpoint"] += 1
                    _append(run_log, {"cell": cell_id, "status": "no_checkpoint"})
                    continue
                log("    no checkpoint -> training this cell first")
                train_cell(repo_root / c["config"], wandb_mode=wandb_mode, log=log)
                counts["trained"] += 1

            rec = factorial_run_one(
                repo_root, repo_root / c["config"], ckpt, out_json,
                device=device, num_episodes=num_episodes,
                logits_out=(logits_dir / f"{cell_id}.npz") if logits_dir else None,
                committed_metrics=repo_root / c["results_json"],
                meta=c, log=log)
            g = rec["regression_guard"]
            log(f"    guard: {g.get('status')} "
                f"({g.get('n_exact', 0)}/{g.get('n_keys', 0)} exact, "
                f"max|diff|={g.get('max_abs_diff', float('nan')):.2e})"
                f"  [{rec['wall_seconds']}s]")
            counts["ok"] += 1
            _append(run_log, {"cell": cell_id, "status": "ok",
                              "guard": g.get("status"),
                              "wall_seconds": rec["wall_seconds"]})
        except Exception as e:  # noqa: BLE001 — one bad cell must not kill the phase
            log(f"    ERROR: {e!r}")
            counts["error"] += 1
            _append(run_log, {"cell": cell_id, "status": "error", "error": repr(e)})

    log(f"[A] done: {counts}")
    return counts


# =====================================================================
# Phase B — RQ5 rank sweep
# =====================================================================
def run_phase_b(repo_root: Path, cells: list[dict], *, device, out_dir: Path,
                logits_dir: Path | None, num_episodes: int, wandb_mode: str,
                max_minutes: float | None, reuse_grid_rank16: bool = True,
                log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "_run_log.jsonl"
    started = time.monotonic()
    counts = {"ok": 0, "skipped_done": 0, "trained": 0, "reused_grid": 0, "error": 0}

    grid_index = json.load(open(repo_root / "configs/grid/_index.json"))["cells"]

    for i, c in enumerate(cells):
        if max_minutes is not None and (time.monotonic() - started) / 60 > max_minutes:
            log(f"[B] budget {max_minutes} min reached; "
                f"{len(cells) - i} cell(s) left for the next session.")
            break

        out_json = out_dir / f"{c['cell']}.json"
        if out_json.exists():
            counts["skipped_done"] += 1
            continue

        log(f"[B] ({i + 1}/{len(cells)}) {c['cell']}  (rank={c['rank']}, seed={c['seed']})")
        ckpt = repo_root / c["checkpoint"]
        try:
            if not ckpt.exists() and reuse_grid_rank16 and c["rank"] == 16:
                # rank 16 + parallel + r18 + CIFAR-FS 5-shot IS the Step 10 grid
                # recipe (verified: n_params 31,746 both ways). If that cell's
                # checkpoint was recovered, retraining it would burn ~18 min to
                # reproduce a model we already have.
                twin = _find_grid_twin(grid_index, c)
                if twin and (repo_root / twin["checkpoint"]).exists():
                    shutil.copy2(repo_root / twin["checkpoint"], ckpt)
                    log(f"    reused Step 10 checkpoint: {twin['checkpoint']}")
                    counts["reused_grid"] += 1

            if not ckpt.exists():
                log("    training…")
                train_cell(repo_root / c["config"], wandb_mode=wandb_mode, log=log)
                counts["trained"] += 1

            rec = factorial_run_one(
                repo_root, repo_root / c["config"], ckpt, out_json,
                device=device, num_episodes=num_episodes,
                logits_out=(logits_dir / f"{c['cell']}.npz") if logits_dir else None,
                committed_metrics=None, meta=c, log=log)
            log(f"    acc={rec['summary'].get('accuracy_mean__softmax', float('nan')):.4f} "
                f"ECE(evid)={rec['summary'].get('ece_pooled__evidential_native', float('nan')):.4f} "
                f"params={rec['n_params']}  [{rec['wall_seconds']}s]")
            counts["ok"] += 1
            _append(run_log, {"cell": c["cell"], "status": "ok",
                              "wall_seconds": rec["wall_seconds"]})
        except Exception as e:  # noqa: BLE001
            log(f"    ERROR: {e!r}")
            counts["error"] += 1
            _append(run_log, {"cell": c["cell"], "status": "error", "error": repr(e)})

    log(f"[B] done: {counts}")
    return counts


def _find_grid_twin(grid_index: list[dict], sweep_cell: dict) -> dict | None:
    for g in grid_index:
        if (g["dataset"] == "cifar_fs" and g["k_shot"] == 5
                and g["backbone"] == "resnet18"
                and g["adapter"] == "bottleneck_parallel"
                and g["head"] == sweep_cell["head"]
                and g["seed"] == sweep_cell["seed"]):
            return g
    return None


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
