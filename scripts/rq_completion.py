"""RQ2/RQ4 checkpoint completion + RQ1 two-way interactions.

Implements docs/TASK_RQ2_COMPLETION_AND_RQ1_INTERACTIONS.md. The source lives
in notebooks/rq2_completion_rq1_interactions.ipynb, which writes it to
scripts/rq_completion.py in the clone.

Part A  The 21 CIFAR-FS 5-shot grid cells whose checkpoints were never
        recovered are trained and scored by the EXISTING Phase A driver
        (`rq_drivers.run_phase_a(allow_retrain=True)`). RQ2 and RQ4 are then
        re-aggregated by the EXISTING `rq_aggregate` functions.
Part B  RQ1's two-way interaction terms, from the EXISTING
        `rq_aggregate.eta_squared` (which always computed them) over the
        per-seed table in results/mvt_results.json.

Retired-numbering warning: `rq1_verdict` in code = RQ2 in the thesis/deck
(objective vs score); `rq2_table`/`rq2_verdict` in code = RQ4 (post-hoc
refit). Every label this module PRINTS uses the thesis numbering.

No model is scored, no evidence mapped and no variance decomposed here by new
code, with three deliberate exceptions:

1. Smoke-baseline guard (task A.4). The committed baseline for
   cifar_5shot_mbnet_lora_evidential_seed42 is NOT a 600-episode evaluation:
   its results/grid JSON has num_episodes=20 and no tin_near/gaussian_far keys.
   It is the Step 10a pre-flight smoke test (`--num-episodes 20`, no OOD
   extras). results/grid/_run_log.jsonl shows it `ok` at line 2 and the real
   grid run `skipped_done` at line 22, because `--resume` found that JSON. It
   is the only one of the 120 committed grid JSONs like this, and it is where
   RQ1's "95/96 TinyImageNet observations" gap comes from. A 600-episode
   regression guard against it reads MISMATCH whatever happens, so the check
   that means something is the same checkpoint re-scored on the same first 20
   test seeds. `smoke_baseline_guards` does that (training does not depend on
   --num-episodes; evaluate.py and factorial_run_one both take seeds[:n]).

2. RQ2 completeness check (task A.5.2). eta_squared's `balanced` flag compares
   the counts of cells that are PRESENT, so a (design, objective) arm with zero
   observations is invisible to it. That is exactly the 99-cell state:
   cifar_fs/5shot/mobilenetv3_small/lora had softmax records and no
   evidential ones. `rq2_completeness` enumerates the full crossing instead.

3. OLS Type-II decomposition (Part B). The published TinyImageNet-near RQ1 row
   (1.64 / 42.60 / 0.61 / 21.98 / 20.98, residual 11.42) is a Type-II
   main-effects fit on the 95 unbalanced rows. eta_squared gives
   1.79 / 42.92 / 0.70 / 22.90 / 21.02, residual 10.67 on the same rows; its
   docstring says it is exact only when the design is balanced. With +-1 coding
   on a balanced 2-level design the two methods agree exactly, and
   `decompose_outcome` checks that on the three balanced outcomes before any
   OLS number is used. OLS also turns the seed bootstrap (task B.3) into one
   matrix product per model instead of thousands of Python loops.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np

import rq_aggregate as A

OUT_SUBDIR = "rq_completion"
EXPECTED_N_EPISODES = 600

# =====================================================================
# Part A: which cells
# =====================================================================
#: Verbatim from the task doc, and cross-checked against
#: results/rq_checkpoint_audit.json["missing_cells"] in `select_cells`.
MISSING_CONFIGS = frozenset(
    f"configs/grid/cifar_5shot_{bb}_{ad}_{head}_seed{seed}.yaml"
    for bb, ad, heads in (("r18", "parallel", ("evidential", "softmax")),
                          ("r18", "lora", ("evidential", "softmax")),
                          ("mbnet", "parallel", ("evidential", "softmax")),
                          ("mbnet", "lora", ("evidential",)))
    for head in heads
    for seed in (42, 43, 44)
)

#: Committed grid baselines known NOT to be 600-episode evaluations.
EXPECTED_SMOKE_BASELINES = frozenset(
    {"configs/grid/cifar_5shot_mbnet_lora_evidential_seed42.yaml"})

#: Run order. Each design (backbone x adapter) is finished, both heads x 3
#: seeds, before the next one starts. A session that stops on its time budget
#: then never leaves a design with only one objective arm, which is the
#: half-present state that made the 99-cell RQ2 ratio unstable.
#: mobilenetv3_small/lora goes first because it is the one design that is
#: half-present TODAY (softmax recovered, evidential not). Its 3 cells alone
#: restore RQ2's objective x score crossing. The rest follow in increasing
#: Step 10 wall time.
DESIGN_ORDER = (("mobilenetv3_small", "lora"),
                ("resnet18", "lora"),
                ("mobilenetv3_small", "bottleneck_parallel"),
                ("resnet18", "bottleneck_parallel"))


def cell_id(c: dict) -> str:
    """Identical to the id rq_drivers.run_phase_a gives each output JSON."""
    return (f"{c['dataset']}_{c['k_shot']}shot_{c['backbone']}_"
            f"{c['adapter']}_{c['head']}_seed{c['seed']}")


def design_of(c: dict) -> str:
    """Identical to rq_aggregate.design_key for a grid-index cell."""
    return f"{c['dataset']}/{c['k_shot']}shot/{c['backbone']}/{c['adapter']}"


def grid_index(repo_root: Path) -> list[dict]:
    return json.load(open(repo_root / "configs" / "grid" / "_index.json"))["cells"]


def select_cells(repo_root: Path, only: dict | None = None) -> list[dict]:
    """The 21 cells, in run order, optionally filtered by `only`."""
    cells = [c for c in grid_index(repo_root) if c["config"] in MISSING_CONFIGS]
    if len(cells) != 21:
        raise RuntimeError(f"expected 21 cells in configs/grid/_index.json, got "
                           f"{len(cells)} -- re-check the audit list")
    audit = json.load(open(repo_root / "results" / "rq_checkpoint_audit.json"))
    if set(audit["missing_cells"]) != MISSING_CONFIGS:
        raise RuntimeError("the task doc's 21 configs and "
                           "results/rq_checkpoint_audit.json disagree")
    order = {d: i for i, d in enumerate(DESIGN_ORDER)}
    cells.sort(key=lambda c: (order[(c["backbone"], c["adapter"])],
                              c["seed"], c["head"] != "evidential"))
    if only:
        cells = [c for c in cells if all(c.get(k) == v for k, v in only.items())]
    return cells


# =====================================================================
# Part A: pre-flight
# =====================================================================
#: Committed files this notebook must never change. `configs/` covers the two
#: frozen episode files and every grid recipe.
FROZEN_PATHS = ("configs", "results/grid", "results/mvt_results.json",
                "results/rq_summary.json", "results/rq_checkpoint_audit.json")


def frozen_files_untouched(repo_root: Path) -> dict:
    r = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain",
                        "--", *FROZEN_PATHS], capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": None, "error": r.stderr.strip(), "paths": list(FROZEN_PATHS)}
    changed = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return {"ok": not changed, "changed": changed, "paths": list(FROZEN_PATHS)}


def preflight(repo_root: Path, cells: list[dict], log=print) -> dict:
    """Checks that cost no GPU time. Any `problems` entry means do not train."""
    import yaml

    problems, notes = [], []
    val = list(yaml.safe_load(open(repo_root / "configs/val_episodes.yaml"))["seeds"])
    test = list(yaml.safe_load(open(repo_root / "configs/test_episodes.yaml"))["seeds"])
    if val != list(range(10000, 10100)):
        problems.append(f"VAL seeds are {val[0]}..{val[-1]} (n={len(val)}), "
                        f"expected 10000..10099")
    if test != list(range(0, 600)):
        problems.append(f"TEST seeds are {test[0]}..{test[-1]} (n={len(test)}), "
                        f"expected 0..599")
    if set(val) & set(test):
        problems.append("VAL and TEST seeds overlap")

    smoke = []
    for c in cells:
        if not (repo_root / c["config"]).exists():
            problems.append(f"missing config {c['config']}")
        committed = repo_root / c["results_json"]
        if not committed.exists():
            problems.append(f"missing committed baseline {c['results_json']}")
            continue
        n = int(json.load(open(committed)).get("num_episodes", -1))
        if n != EXPECTED_N_EPISODES:
            smoke.append({"config": c["config"], "committed_num_episodes": n})
    unexpected = {s["config"] for s in smoke} - EXPECTED_SMOKE_BASELINES
    if unexpected:
        notes.append(f"unexpected non-600-episode baselines: {sorted(unexpected)}")

    frozen = frozen_files_untouched(repo_root)
    if frozen["ok"] is False:
        problems.append(f"frozen paths already modified in this clone: {frozen['changed']}")

    log(f"  VAL seeds {val[0]}..{val[-1]} (n={len(val)}), "
        f"TEST seeds {test[0]}..{test[-1]} (n={len(test)}), disjoint={not set(val) & set(test)}")
    log(f"  {len(cells)} cell(s): configs + committed baselines present = "
        f"{not any('missing' in p for p in problems)}")
    for s in smoke:
        log(f"  NOTE baseline is a smoke run: {s['config']} "
            f"(num_episodes={s['committed_num_episodes']}) -> first-"
            f"{s['committed_num_episodes']}-episode guard applies (Section 8)")
    log(f"  frozen paths clean: {frozen['ok']}")
    for p in problems:
        log(f"  PROBLEM: {p}")
    for n_ in notes:
        log(f"  NOTE: {n_}")
    return {"ok": not problems, "problems": problems, "notes": notes,
            "smoke_baselines": smoke, "frozen": frozen}


# =====================================================================
# Part A: restore what an earlier session (or Step 10) already paid for
# =====================================================================
#: Directories never worth walking: huge image trees, VCS metadata.
_SKIP_DIRS = frozenset({".git", "tiny-imagenet-200", "cifar-100-python",
                        "wandb", "__pycache__"})


def restore_session_artifacts(repo_root: Path, cells: list[dict],
                              search_roots=("/kaggle/input",), log=print) -> dict:
    """Scan attached inputs (zips and loose files) and restore, for the given
    cells only:

      - checkpoints/model_phase2_*.pt   (task A.1 recovery, and resume)
      - results/rq_factorial/<cell>.json, accepted only if it is a full
        600-episode record, so a smoke output can never enter the aggregation
      - results/rq_completion/provenance/<cell>.json and logs/*.log

    Targeted on purpose: rq_drivers.recover_checkpoints extracts EVERY
    checkpoint in the Step 10 zips (99 files, ~2.6 GB, staged then copied) and
    none of those 99 are needed here. Their scores are already in the
    committed rq_factorial JSONs. Existing files are never overwritten, and
    results/grid/ is never written.
    """
    ckpt_dir = repo_root / "checkpoints"
    fac_dir = repo_root / "results" / "rq_factorial"
    comp_dir = repo_root / "results" / OUT_SUBDIR
    want_ckpt = {Path(c["checkpoint"]).name for c in cells}
    want_ids = {cell_id(c) for c in cells}
    got = {"checkpoints": [], "factorial": [], "provenance": [], "logs": [],
           "rejected": []}
    zips = []

    def _route(member: str):
        p = Path(member)
        if p.suffix == ".pt" and p.name in want_ckpt:
            return "checkpoints", ckpt_dir
        if p.suffix == ".json" and p.stem in want_ids:
            if p.parent.name == "rq_factorial":
                return "factorial", fac_dir
            if p.parent.name == "provenance" and p.parent.parent.name == OUT_SUBDIR:
                return "provenance", comp_dir / "provenance"
        if (p.suffix == ".log" and p.parent.name == "logs"
                and p.parent.parent.name == OUT_SUBDIR):
            return "logs", comp_dir / "logs"
        return None

    def _take(kind: str, dest_dir: Path, name: str, reader) -> None:
        dest = dest_dir / name
        if dest.exists():
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (name + ".part")
        with open(tmp, "wb") as out:
            shutil.copyfileobj(reader, out)
        if kind == "factorial":
            try:
                rec = json.load(open(tmp))
                ok = int(rec["summary"]["num_episodes"]) == EXPECTED_N_EPISODES
            except Exception:  # noqa: BLE001 -- unreadable means rejected
                ok = False
            if not ok:
                tmp.unlink()
                got["rejected"].append(name)
                return
        tmp.rename(dest)
        got[kind].append(name)

    for root in search_roots:
        rp = Path(root)
        if not rp.exists():
            log(f"  (no such path: {root})")
            continue
        for dirpath, dirnames, filenames in os.walk(rp):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.suffix == ".zip":
                    zips.append(str(p))
                    try:
                        with zipfile.ZipFile(p) as zf:
                            for m in zf.namelist():
                                route = _route(m)
                                if route:
                                    with zf.open(m) as fh:
                                        _take(route[0], route[1], Path(m).name, fh)
                    except zipfile.BadZipFile:
                        log(f"  {p.name}: not a readable zip, skipped")
                    continue
                route = _route(str(p))
                if route:
                    with open(p, "rb") as fh:
                        _take(route[0], route[1], p.name, fh)

    log(f"  scanned {list(search_roots)}: {len(zips)} zip(s)")
    log(f"  restored: {len(got['checkpoints'])} checkpoint(s), "
        f"{len(got['factorial'])} factorial record(s), "
        f"{len(got['provenance'])} provenance file(s), {len(got['logs'])} log(s)")
    if got["rejected"]:
        log(f"  rejected (not a 600-episode record): {got['rejected']}")
    return {"zips_seen": zips, **{k: sorted(v) for k, v in got.items()}}


def _step10_wall_seconds(repo_root: Path) -> dict[str, float]:
    """Measured train+eval wall time per config, from the Step 10 run log."""
    out: dict[str, float] = {}
    p = repo_root / "results" / "grid" / "_run_log.jsonl"
    if p.exists():
        for line in open(p):
            e = json.loads(line)
            if e.get("status") == "ok" and e.get("wall_seconds"):
                out[e["config"]] = float(e["wall_seconds"])
    return out


def coverage(repo_root: Path, cells: list[dict]) -> dict:
    """What this session actually has left to do, with a measured estimate."""
    fac_dir = repo_root / "results" / "rq_factorial"
    wall = _step10_wall_seconds(repo_root)
    out = {"n_cells": len(cells), "evaluated": [], "checkpoint_present": [],
           "needs_training": [], "estimate_minutes": 0.0}
    eval_s = 240.0  # measured: Phase A scored cifar 5-shot cells in ~190-230 s
    for c in cells:
        cid = cell_id(c)
        if (fac_dir / f"{cid}.json").exists():
            out["evaluated"].append(cid)
        elif (repo_root / c["checkpoint"]).exists():
            out["checkpoint_present"].append(cid)
            out["estimate_minutes"] += eval_s / 60
        else:
            out["needs_training"].append(cid)
            # Step 10 wall already includes one evaluation; factorial replaces it.
            out["estimate_minutes"] += wall.get(c["config"], 1500.0) / 60
    out["estimate_minutes"] = round(out["estimate_minutes"], 1)
    return out


def record_provenance(repo_root: Path, cells: list[dict], had_checkpoint: set,
                      session_info: dict, log=print) -> list[str]:
    """After run_phase_a: note, per newly evaluated cell, whether THIS session
    trained its checkpoint or found one on disk. The best_val_epoch check in
    the guard table only tests training reproducibility for trained cells."""
    fac_dir = repo_root / "results" / "rq_factorial"
    prov_dir = repo_root / "results" / OUT_SUBDIR / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in cells:
        cid = cell_id(c)
        if (prov_dir / f"{cid}.json").exists() or not (fac_dir / f"{cid}.json").exists():
            continue
        source = ("checkpoint_found_on_disk" if cid in had_checkpoint
                  else "trained_this_session")
        with open(prov_dir / f"{cid}.json", "w") as f:
            json.dump({"cell": cid, "checkpoint_source": source, **session_info},
                      f, indent=2, sort_keys=True)
        written.append(cid)
    log(f"  provenance written for {len(written)} cell(s)")
    return written


# =====================================================================
# Part A.4: regression guards
# =====================================================================
def smoke_baseline_guards(repo_root: Path, cells: list[dict], *, device,
                          log=print) -> list[dict]:
    """For every cell whose committed baseline is not a 600-episode run,
    re-score the checkpoint on exactly the committed number of test seeds and
    diff against it with the unchanged rq_core.regression_guard."""
    import rq_drivers as D

    out_dir = repo_root / "results" / OUT_SUBDIR / "smoke_baseline_guard"
    results = []
    for c in cells:
        committed_path = repo_root / c["results_json"]
        committed = json.load(open(committed_path))
        n = int(committed.get("num_episodes", EXPECTED_N_EPISODES))
        if n == EXPECTED_N_EPISODES:
            continue
        cid = cell_id(c)
        out_json = out_dir / f"{cid}__first{n}episodes.json"
        ckpt = repo_root / c["checkpoint"]
        if out_json.exists():
            rec = json.load(open(out_json))
        elif not ckpt.exists():
            log(f"  {cid}: no checkpoint yet -- smoke-baseline guard deferred")
            results.append({"cell": cid, "committed_num_episodes": n,
                            "guard": {"status": "no_checkpoint_yet"}})
            continue
        else:
            log(f"  {cid}: re-scoring the checkpoint on test seeds 0..{n - 1} "
                f"against the committed {n}-episode baseline")
            rec = D.factorial_run_one(
                repo_root, repo_root / c["config"], ckpt, out_json,
                device=device, num_episodes=n, logits_out=None,
                committed_metrics=committed_path,
                meta={**c, "purpose": "smoke_baseline_guard"}, log=log)
        g = rec["regression_guard"]
        log(f"    guard: {g.get('status')} ({g.get('n_exact', 0)}/{g.get('n_keys', 0)} "
            f"exact, max|diff|={g.get('max_abs_diff', float('nan')):.2e})")
        results.append({
            "cell": cid, "committed_num_episodes": n, "guard": g,
            "best_val_epoch": {"committed": committed.get("best_val_epoch"),
                               "checkpoint": rec.get("best_val_epoch")},
            "n_params": {"committed": committed.get("n_params"),
                         "checkpoint": rec.get("n_params")},
        })
    return results


def guard_table(repo_root: Path, cells: list[dict], smoke_guards: list[dict]) -> dict:
    """One row per target cell: the APPLICABLE guard (600-episode, or the
    first-n-episode one when the baseline is a smoke run), plus two checks
    that are independent of evaluation: best_val_epoch and n_params."""
    fac_dir = repo_root / "results" / "rq_factorial"
    prov_dir = repo_root / "results" / OUT_SUBDIR / "provenance"
    sg = {s["cell"]: s for s in smoke_guards}
    rows = []
    for c in cells:
        cid = cell_id(c)
        rec_path = fac_dir / f"{cid}.json"
        if not rec_path.exists():
            rows.append({"cell": cid, "guard_status": "not_run"})
            continue
        rec = json.load(open(rec_path))
        committed = json.load(open(repo_root / c["results_json"]))
        prov = (json.load(open(prov_dir / f"{cid}.json"))
                if (prov_dir / f"{cid}.json").exists() else {})
        n_comm = int(committed.get("num_episodes", -1))
        if n_comm == EXPECTED_N_EPISODES:
            g, baseline = rec["regression_guard"], "600-episode"
        else:
            g = sg.get(cid, {}).get("guard", {"status": "smoke_guard_missing"})
            baseline = f"first-{n_comm}-episode (committed JSON is a smoke run)"
        bve_c, bve_r = committed.get("best_val_epoch"), rec.get("best_val_epoch")
        rows.append({
            "cell": cid,
            "checkpoint_source": prov.get("checkpoint_source", "unknown"),
            "gpu": prov.get("gpu"),
            "baseline": baseline,
            "guard_status": g.get("status"),
            "n_exact": g.get("n_exact"), "n_keys": g.get("n_keys"),
            "max_abs_diff": g.get("max_abs_diff"),
            "max_abs_diff_key": g.get("max_abs_diff_key"),
            "guard_600_episode_status": rec["regression_guard"].get("status"),
            "best_val_epoch_committed": bve_c, "best_val_epoch_new": bve_r,
            "best_val_epoch_match": (bve_c == bve_r) if bve_c is not None else None,
            "n_params_match": committed.get("n_params") == rec.get("n_params"),
        })
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["guard_status"]] = counts.get(r["guard_status"], 0) + 1
    evaluated = [r for r in rows if r["guard_status"] != "not_run"]
    trained = [r for r in evaluated if r.get("checkpoint_source") == "trained_this_session"]
    return {
        "rows": rows,
        "status_counts": counts,
        "n_evaluated": len(evaluated),
        "n_reproduced": sum(1 for r in evaluated
                            if r["guard_status"] in ("exact", "within_tol")),
        "n_best_val_epoch_match": sum(1 for r in evaluated if r.get("best_val_epoch_match")),
        "n_trained_this_session": len(trained),
        "n_params_all_match": (all(r.get("n_params_match") for r in evaluated)
                               if evaluated else None),
    }


def guards_markdown(g: dict) -> str:
    L = ["### Regression guard against the committed Step 10 grid", "",
         f"Evaluated {g['n_evaluated']}/21. Guard status counts: "
         + ", ".join(f"`{k}` {v}" for k, v in sorted(g["status_counts"].items()))
         + f". Reproduced (exact or within 1e-6): **{g['n_reproduced']}/{g['n_evaluated']}**. "
         f"`best_val_epoch` matches: **{g['n_best_val_epoch_match']}/{g['n_evaluated']}**. "
         f"`n_params` all match: {g['n_params_all_match']}.", "",
         "| cell | checkpoint | baseline | guard | exact/keys | max abs diff | best_val_epoch committed → new |",
         "|---|---|---|---|---:|---:|---:|"]
    for r in g["rows"]:
        if r["guard_status"] == "not_run":
            L.append(f"| `{r['cell']}` | | | not run | | | |")
            continue
        mad = r.get("max_abs_diff")
        L.append(f"| `{r['cell']}` | {r['checkpoint_source']} | {r['baseline']} | "
                 f"`{r['guard_status']}` | {r.get('n_exact')}/{r.get('n_keys')} | "
                 f"{'' if mad is None else f'{mad:.2e}'} | "
                 f"{r['best_val_epoch_committed']} → {r['best_val_epoch_new']} |")
    return "\n".join(L)


# =====================================================================
# Part A.5: RQ2 (code: rq1_verdict) and RQ4 (code: rq2_verdict)
# =====================================================================
def rq2_completeness(recs: list[dict], grid_cells: list[dict]) -> dict:
    """Is the objective x score design FULLY crossed over the designs present?

    eta_squared's `balanced` only sees cells that have observations. This
    enumerates what should be there, so a wholly missing arm is reported
    rather than silently leaving the design out.
    """
    seeds = sorted({c["seed"] for c in grid_cells})
    designs = sorted({design_of(c) for c in grid_cells})
    present: dict[tuple, set] = {}
    for r in recs:
        present.setdefault((A.design_key(r), r["interpretation"]), set()).add(r["seed"])

    complete, absent, half, partial = [], [], [], []
    for d in designs:
        ev = present.get((d, "evidential"), set())
        sm = present.get((d, "softmax"), set())
        if ev == set(seeds) and sm == set(seeds):
            complete.append(d)
        elif not ev and not sm:
            absent.append(d)
        elif not ev or not sm:
            half.append({"design": d, "evidential_seeds": sorted(ev),
                         "softmax_seeds": sorted(sm)})
        else:
            partial.append({"design": d, "evidential_seeds": sorted(ev),
                            "softmax_seeds": sorted(sm)})

    rows = A.factorial_observations(recs)
    zero_cells, count_values = {}, {}
    for g in ("far", "near"):
        cnt: dict[tuple, int] = {}
        for r in rows:
            if r["pool_group"] == g:
                k = (r["design"], r["objective"], r["score"])
                cnt[k] = cnt.get(k, 0) + 1
        seen = sorted({k[0] for k in cnt})
        zero_cells[g] = [f"{d} | {o} | {s}" for d in seen
                         for o in ("evidential", "softmax") for s in A.RQ1_SCORES
                         if (d, o, s) not in cnt]
        count_values[g] = sorted(set(cnt.values()))
    return {
        "n_designs_expected": len(designs), "complete_designs": complete,
        "absent_designs": absent, "half_present_designs": half,
        "partial_seed_designs": partial, "zero_observation_cells": zero_cells,
        "observation_counts_per_cell": count_values,
        "fully_crossed": (not half and not partial
                          and not any(zero_cells.values())
                          and all(len(v) == 1 for v in count_values.values())),
    }


def _max_numeric_diff(a, b, path="") -> tuple[float, list[str]]:
    if isinstance(a, dict) and isinstance(b, dict):
        worst, bad = 0.0, []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: present on one side only")
                continue
            w, bb = _max_numeric_diff(a[k], b[k], f"{path}/{k}")
            worst, bad = max(worst, w), bad + bb
        return worst, bad
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return 0.0, ([] if a == b else [f"{path}: {a!r} != {b!r}"])
    if a == b:
        return 0.0, []
    if math.isnan(a) and math.isnan(b):
        return 0.0, []
    return abs(float(a) - float(b)), []


def rq2_rq4_aggregate(repo_root: Path, grid_cells: list[dict], new_ids: set,
                      committed_summary_path: Path) -> dict:
    """Task A.5, steps 1-3 and 5, over whatever records are on disk."""
    known = {cell_id(c) for c in grid_cells}
    recs = [r for r in A.load_records(repo_root / "results" / "rq_factorial")
            if cell_id(r["meta"]) in known]
    old = [r for r in recs if cell_id(r["meta"]) not in new_ids]
    new = [r for r in recs if cell_id(r["meta"]) in new_ids]
    committed = json.load(open(committed_summary_path))

    # 0. Before trusting any new number: the unchanged aggregator on the
    #    records that produced the committed summary must reproduce it.
    old_v2 = A.rq1_verdict(A.factorial_observations(old))
    old_v4 = A.rq2_verdict(A.rq2_table(old))
    d2, bad2 = _max_numeric_diff(old_v2, committed["rq1_verdict"])
    d4, bad4 = _max_numeric_diff(old_v4, committed["rq2_verdict"])

    comp = rq2_completeness(recs, grid_cells)
    rows = A.factorial_observations(recs)
    after = A.rq1_verdict(rows)
    keep = set(comp["complete_designs"])
    after_complete = A.rq1_verdict([r for r in rows if r["design"] in keep])

    rq4_rows = A.rq2_table(recs)
    return {
        "coverage": {
            "n_records": len(recs), "n_grid_cells": len(grid_cells),
            "n_committed_records": len(old), "n_new_records": len(new),
            "new_records": sorted(cell_id(r["meta"]) for r in new),
            "n_evidential_records": sum(1 for r in recs if r["interpretation"] == "evidential"),
        },
        "aggregator_reproduces_committed": {
            "rq2_max_abs_diff": d2, "rq2_non_numeric_mismatches": bad2,
            "rq4_max_abs_diff": d4, "rq4_non_numeric_mismatches": bad4,
            "ok": d2 <= 1e-12 and d4 <= 1e-12 and not bad2 and not bad4,
        },
        "completeness": comp,
        "rq2_before_committed": committed["rq1_verdict"],
        "rq2_after_all_records": after,
        "rq2_after_complete_designs_only": after_complete,
        "rq2_quotable": ("all_records" if comp["fully_crossed"]
                         else "complete_designs_only"),
        "rq2_tables_after": {g: A.table_2x4(rows, g) for g in ("far", "near")},
        "rq4_before_committed": committed["rq2_verdict"],
        "rq4_after": A.rq2_verdict(rq4_rows),
        "rq4_new_cells_only": A.rq2_verdict(A.rq2_table(new)) if new else None,
        "rq4_rows": rq4_rows,
    }


def rq2_rq4_markdown(agg: dict) -> str:
    cov, comp, rep = agg["coverage"], agg["completeness"], agg["aggregator_reproduces_committed"]
    L = ["## RQ2 / RQ4 re-aggregation", "",
         f"Phase A factorial records: **{cov['n_records']}/{cov['n_grid_cells']}** "
         f"({cov['n_committed_records']} committed + {cov['n_new_records']} new). "
         f"Evidential records: {cov['n_evidential_records']}.", "",
         f"Unchanged aggregator reproduces the committed `results/rq_summary.json` on the "
         f"committed records: **{rep['ok']}** (RQ2 max abs diff {rep['rq2_max_abs_diff']:.1e}, "
         f"RQ4 {rep['rq4_max_abs_diff']:.1e}).", "",
         f"Design completeness: {len(comp['complete_designs'])}/{comp['n_designs_expected']} "
         f"complete, {len(comp['absent_designs'])} absent, "
         f"{len(comp['half_present_designs'])} with one objective arm missing, "
         f"{len(comp['partial_seed_designs'])} with seeds missing. Zero-observation "
         f"objective×score cells: far {len(comp['zero_observation_cells']['far'])}, "
         f"near {len(comp['zero_observation_cells']['near'])}. "
         f"**Fully crossed: {comp['fully_crossed']}.**"]
    for h in comp["half_present_designs"]:
        L.append(f"- one arm missing: `{h['design']}` evidential seeds "
                 f"{h['evidential_seeds']}, softmax seeds {h['softmax_seeds']}")
    L += ["", "### RQ2: objective vs scoring rule (η² share of OOD-AUROC variance)", "",
          "| variant | pool | score η² | objective η² | score/objective | balanced | n |",
          "|---|---|---:|---:|---:|---|---:|"]
    variants = [("before: committed, 99 records", agg["rq2_before_committed"]),
                ("after: all records", agg["rq2_after_all_records"])]
    if not comp["fully_crossed"]:
        variants.append(("after: complete designs only", agg["rq2_after_complete_designs_only"]))
    for label, v in variants:
        for g in ("far", "near"):
            if g not in v:
                continue
            e = v[g]["eta_squared"]
            L.append(f"| {label} | {g} | {100 * e['score']:.2f}% | "
                     f"{100 * e['objective']:.3f}% | "
                     f"{v[g]['ratio_score_over_objective']:.1f}× | "
                     f"{v[g]['balanced']} | {v[g]['n']} |")
    L += ["", f"Quotable variant: **{agg['rq2_quotable'].replace('_', ' ')}**"
          + ("" if comp["fully_crossed"] else
             " (the all-records row still contains a half-present design; "
             "`balanced` does not detect that)."), ""]
    for g in ("far", "near"):
        t = agg["rq2_tables_after"][g]
        L += [f"Mean AUROC, {g}-OOD, after (rows = training objective):", "",
              "| trained as | " + " | ".join(A.RQ1_SCORES) + " |",
              "|---|" + "---:|" * len(A.RQ1_SCORES)]
        for obj in ("evidential", "softmax"):
            L.append(f"| {obj} | " + " | ".join(
                f"{t[obj][s]['mean']:.4f} (n={t[obj][s]['n']})" for s in A.RQ1_SCORES) + " |")
        L.append("")
    L += ["### RQ4: post-hoc evidence-affine refit (VAL only)", "",
          "| variant | evidential cells | ECE improved | AUROC preserved (Δ ≥ −0.005) | "
          "mean ΔECE | mean ΔAUROC | min Spearman ρ | reordering observed |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, v in (("before: committed", agg["rq4_before_committed"]),
                     ("after: all records", agg["rq4_after"]),
                     ("new cells only", agg["rq4_new_cells_only"])):
        if not v or "error" in v:
            continue
        L.append(f"| {label} | {v['n_cells']} | {v['ece_improved_in']}/{v['n_cells']} | "
                 f"{v['auroc_preserved_in']}/{v['auroc_comparisons']} | "
                 f"{v['ece_mean_delta']:+.4f} | {v['auroc_mean_delta']:+.4f} | "
                 f"{v['min_spearman_rho']:.4f} | {v['reordering_ever_observed']} |")
    return "\n".join(L)


# =====================================================================
# Part B: RQ1 two-way interactions
# =====================================================================
FACTORS = ("dataset", "k_shot", "backbone", "adapter", "head")
BASELINE_ADAPTERS = frozenset({"full_ft", "linear_probe"})

#: Outcome -> results/mvt_results.json key. Checked 2026-09-14 by reproducing
#: every published main-effect share in docs/RQ_RESULTS_SUMMARY.md §3.1:
#:  - ECE is `ece_pooled`, NOT `ece_per_episode_mean` as the task doc guessed.
#:    That key gives head 83.03 / residual 10.55, not the published 82.89 / 7.70.
#:  - "OOD AUROC (far)" is the SVHN pool alone under each head's native score
#:    (== `ood_auroc_mean`, the legacy primary pool), not an SVHN+Gaussian mean.
#:  - native score = vacuity (evidential) / msp (softmax): Step 10 scored each
#:    head with its own score only.
RQ1_OUTCOMES = {
    "accuracy": "accuracy_mean",
    "ece": "ece_pooled",
    "far_ood_svhn": "ood_auroc__svhn_far__{native}",
    "near_ood_tin": "ood_auroc__tin_near__{native}",
}

#: docs/RQ_RESULTS_SUMMARY.md §3.1, in percent. The regression target for
#: `flatten`: if these do not come back, the row table is wrong.
PUBLISHED_MAIN_EFFECTS = {
    "accuracy": {"dataset": 1.75, "k_shot": 76.05, "backbone": 3.92,
                 "adapter": 9.14, "head": 0.19, "residual": 8.95},
    "ece": {"dataset": 2.02, "k_shot": 2.13, "backbone": 4.63,
            "adapter": 0.63, "head": 82.89, "residual": 7.70},
    "far_ood_svhn": {"dataset": 1.18, "k_shot": 22.68, "backbone": 11.54,
                     "adapter": 1.73, "head": 39.01, "residual": 23.87},
    "near_ood_tin": {"dataset": 1.64, "k_shot": 42.60, "backbone": 0.61,
                     "adapter": 21.98, "head": 20.98, "residual": 11.42},
}

#: eta_squared names a pair in FACTORS order, so the task doc's
#: "adapter:backbone" is this key.
FOCUS_TERM = "backbone:adapter"

#: For an evidential factorial record, the keys holding the same quantity as
#: each outcome. These are exactly the pairs rq_core.regression_guard maps.
FACTORIAL_KEYS_EVIDENTIAL = {
    "accuracy": "accuracy_mean__evidential_native",
    "ece": "ece_pooled__evidential_native",
    "far_ood_svhn": "ood_auroc__svhn_far__vacuity_native",
    "near_ood_tin": "ood_auroc__tin_near__vacuity_native",
}
SMOKE_CELL = {"dataset": "cifar_fs", "k_shot": 5, "backbone": "mobilenetv3_small",
              "adapter": "lora", "head": "evidential", "seed": 42}


def _obs_key(dataset, k_shot, backbone, adapter, head, seed) -> tuple:
    return (dataset, int(k_shot), backbone, adapter, head, int(seed))


def flatten(mvt: dict, outcome: str, overrides: dict | None = None) -> list[dict]:
    """One row per (cell, seed) of the balanced 2^5 design (baselines excluded).

    `overrides` maps an `_obs_key` to {outcome: value}. It replaces or supplies
    that single observation.
    """
    tmpl = RQ1_OUTCOMES[outcome]
    rows = []
    for dataset, by_shot in mvt["results"].items():
        for shot_key, by_backbone in by_shot.items():
            k_shot = int(shot_key.replace("shot", ""))
            for backbone, by_adapter in by_backbone.items():
                for adapter, by_head in by_adapter.items():
                    if adapter in BASELINE_ADAPTERS:
                        continue
                    for head, leaf in by_head.items():
                        key = tmpl.format(native="vacuity" if head == "evidential" else "msp")
                        per_seed = {int(s): float(v)
                                    for s, v in (leaf.get(key) or {}).get("per_seed", {}).items()}
                        for ok, vals in (overrides or {}).items():
                            if ok[:5] == (dataset, k_shot, backbone, adapter, head) and outcome in vals:
                                per_seed[ok[5]] = float(vals[outcome])
                        for seed in sorted(per_seed):
                            rows.append({"dataset": dataset, "k_shot": k_shot,
                                         "backbone": backbone, "adapter": adapter,
                                         "head": head, "seed": seed,
                                         "value": per_seed[seed]})
    return rows


def _coded_columns(rows: list[dict], interactions: bool) -> tuple[dict, list[str]]:
    cols = {}
    for f in FACTORS:
        levels = sorted({r[f] for r in rows}, key=str)
        if len(levels) != 2:
            raise ValueError(f"factor {f!r} has {len(levels)} levels; this "
                             f"decomposition assumes the 2^5 design")
        cols[f] = np.array([1.0 if r[f] == levels[1] else -1.0 for r in rows])
    terms = list(FACTORS)
    if interactions:
        for a, b in combinations(FACTORS, 2):
            cols[f"{a}:{b}"] = cols[a] * cols[b]
            terms.append(f"{a}:{b}")
    return cols, terms


def ols_type2(rows: list[dict], *, interactions: bool, Y: np.ndarray | None = None) -> dict:
    """Type-II sums of squares, as shares of SS_total, for every row of Y.

    SS(T) = RSS(terms not containing T) - RSS(those terms + T). With +-1 coding
    on a balanced 2-level design every column is orthogonal and this equals
    the classical decomposition eta_squared computes, to machine precision.
    Returns (B,)-shaped arrays when Y is (B, n); a single y otherwise.
    """
    y = np.array([r["value"] for r in rows], dtype=float)
    n = len(y)
    Ys = y[None, :] if Y is None else np.asarray(Y, dtype=float)
    cols, terms = _coded_columns(rows, interactions)
    makers: dict[tuple, np.ndarray] = {}

    def rss(ts):
        key = tuple(sorted(ts))
        if key not in makers:
            X = np.column_stack([np.ones(n)] + [cols[t] for t in key])
            makers[key] = np.eye(n) - X @ np.linalg.pinv(X)
        return np.einsum("bi,ij,bj->b", Ys, makers[key], Ys)

    sst = ((Ys - Ys.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    rss_full = rss(terms)
    ss = {}
    for t in terms:
        parts = set(t.split(":"))
        reduced = [u for u in terms if not parts <= set(u.split(":"))]
        ss[t] = rss(reduced) - rss(reduced + [t])
    X_full = np.column_stack([np.ones(n)] + [cols[t] for t in terms])
    df_resid = n - int(np.linalg.matrix_rank(X_full))
    shares = {t: ss[t] / sst for t in terms}
    shares["residual"] = rss_full / sst
    return {"terms": terms, "shares": shares, "ss": ss, "rss_full": rss_full,
            "sst": sst, "n": n, "df_resid": df_resid}


def seed_bootstrap(rows: list[dict], n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resampled outcomes: the seeds of each of the 32 cells are
    redrawn with replacement, the design itself is held fixed (task B.3)."""
    rng = np.random.default_rng(seed)
    y = np.array([r["value"] for r in rows], dtype=float)
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r[f] for f in FACTORS), []).append(i)
    idx = np.empty((n_boot, len(rows)), dtype=int)
    for members in groups.values():
        g = np.asarray(members)
        idx[:, g] = g[rng.integers(0, len(g), size=(n_boot, len(g)))]
    return y[idx]


def _f_test_p(ss_term: float, rss_full: float, df_resid: int) -> float | None:
    try:
        from scipy import stats
    except ImportError:
        return None
    if df_resid <= 0 or rss_full <= 0:
        return None
    return float(stats.f.sf(ss_term / (rss_full / df_resid), 1, df_resid))


def decompose_outcome(rows: list[dict], *, n_boot: int, boot_seed: int) -> dict:
    eta_main = A.eta_squared(rows, list(FACTORS), value="value", interactions=False)
    eta_int = A.eta_squared(rows, list(FACTORS), value="value", interactions=True)
    ols_main = ols_type2(rows, interactions=False)
    ols_int = ols_type2(rows, interactions=True)
    balanced = bool(eta_int["balanced"])
    ols_int_pt = {k: float(v[0]) for k, v in ols_int["shares"].items()}
    ols_main_pt = {k: float(v[0]) for k, v in ols_main["shares"].items()}

    agreement = max(abs(eta_int["eta_squared"][k] - ols_int_pt[k]) for k in ols_int_pt)
    if balanced and agreement > 1e-9:
        raise RuntimeError(f"OLS Type II and eta_squared disagree by {agreement:.2e} "
                           f"on a BALANCED outcome -- the OLS path is wrong, stop")
    if balanced:
        point, main_only = dict(eta_int["eta_squared"]), dict(eta_main["eta_squared"])
        method = "eta_squared (exact: balanced design)"
    else:
        point, main_only = ols_int_pt, ols_main_pt
        method = "OLS Type II (eta_squared is not exact on unbalanced rows)"

    boot = (ols_type2(rows, interactions=True, Y=seed_bootstrap(rows, n_boot, boot_seed))
            if n_boot else None)
    interaction_terms = [f"{a}:{b}" for a, b in combinations(FACTORS, 2)]
    table = []
    for t in list(FACTORS) + interaction_terms + ["residual"]:
        row = {"term": t, "kind": ("residual" if t == "residual" else
                                   "interaction" if ":" in t else "main"),
               "share": float(point[t])}
        if boot is not None:
            b = boot["shares"][t]
            row.update(boot_lo=float(np.percentile(b, 2.5)),
                       boot_hi=float(np.percentile(b, 97.5)),
                       boot_sd=float(np.std(b)))
        if t != "residual":
            row["f_test_p"] = _f_test_p(float(ols_int["ss"][t][0]),
                                        float(ols_int["rss_full"][0]),
                                        ols_int["df_resid"])
        table.append(row)

    resid_main = float(main_only["residual"])
    resid_int = float(point["residual"])
    focus = float(point[FOCUS_TERM])
    return {
        "n": len(rows), "balanced": balanced, "method": method,
        "df_resid_with_interactions": ols_int["df_resid"],
        "ols_vs_eta_squared_max_abs_diff": float(agreement),
        "main_effects_only": {k: float(v) for k, v in main_only.items()},
        "with_two_way": {k: float(v) for k, v in point.items()},
        "eta_squared_with_two_way_raw": {k: float(v) for k, v in eta_int["eta_squared"].items()},
        "table": table,
        "residual_main_effects_only": resid_main,
        "residual_with_two_way": resid_int,
        "two_way_total": float(sum(point[t] for t in interaction_terms)),
        "focus_term": FOCUS_TERM, "focus_share": focus,
        "focus_share_of_main_effects_residual": (focus / resid_main if resid_main else None),
        "sum_of_shares_incl_residual": float(sum(point.values())),
    }


def check_published(outcome: str, main_only: dict) -> dict:
    pub = PUBLISHED_MAIN_EFFECTS[outcome]
    diffs = {k: abs(100 * main_only[k] - v) for k, v in pub.items()}
    # published values are rounded to 2 d.p.
    return {"ok": all(d <= 0.0051 for d in diffs.values()),
            "max_abs_diff_pp": max(diffs.values()), "diffs_pp": diffs}


def rq1_interactions(mvt_path: Path, *, overrides: dict | None = None,
                     n_boot: int = 2000, boot_seed: int = 20260914) -> dict:
    mvt = json.load(open(mvt_path))
    out = {"source": str(mvt_path), "factors": list(FACTORS),
           "baseline_adapters_excluded": sorted(BASELINE_ADAPTERS),
           "outcome_keys": RQ1_OUTCOMES, "focus_term": FOCUS_TERM,
           "n_boot": n_boot, "boot_seed": boot_seed,
           "overrides": ({"|".join(map(str, k)): v for k, v in overrides.items()}
                         if overrides else None),
           "outcomes": {}}
    for outcome in RQ1_OUTCOMES:
        rows = flatten(mvt, outcome, overrides)
        dec = decompose_outcome(rows, n_boot=n_boot, boot_seed=boot_seed)
        dec["published_main_effects_check"] = check_published(outcome, dec["main_effects_only"])
        out["outcomes"][outcome] = dec
    return out


def smoke_cell_overrides(repo_root: Path, guards: dict) -> dict | None:
    """The one 20-episode observation in the RQ1 table, replaced by the SAME
    model's 600-episode numbers. Offered only when the first-20-episode guard
    has shown the checkpoint on disk is that model."""
    cid = cell_id(SMOKE_CELL)
    rec_path = repo_root / "results" / "rq_factorial" / f"{cid}.json"
    row = next((r for r in guards["rows"] if r["cell"] == cid), None)
    if (not rec_path.exists() or row is None
            or row.get("guard_status") not in ("exact", "within_tol")):
        return None
    s = json.load(open(rec_path))["summary"]
    if int(s["num_episodes"]) != EXPECTED_N_EPISODES:
        return None
    key = _obs_key(*(SMOKE_CELL[f] for f in FACTORS), SMOKE_CELL["seed"])
    return {key: {o: float(s[k]) for o, k in FACTORIAL_KEYS_EVIDENTIAL.items()}}


def _pct(x):
    return "" if x is None else f"{100 * x:.2f}%"


def _p(p):
    if p is None:
        return ""
    return "<1e-6" if p < 1e-6 else f"{p:.2g}"


def rq1_markdown(res: dict, title: str) -> str:
    L = [f"## {title}", "",
         f"Factors {', '.join(res['factors'])}; baselines excluded "
         f"({', '.join(res['baseline_adapters_excluded'])}). Seed bootstrap: "
         f"{res['n_boot']} resamples, rng seed {res['boot_seed']}."]
    if res["overrides"]:
        L.append(f"Observations replaced: {list(res['overrides'])}")
    for outcome, d in res["outcomes"].items():
        chk = d["published_main_effects_check"]
        pub_line = (f"Published main effects reproduced: **{chk['ok']}** "
                    f"(max abs diff {chk['max_abs_diff_pp']:.3f} pp)."
                    if not res["overrides"] else
                    f"Main effects vs the published table (expected to move): "
                    f"max abs diff {chk['max_abs_diff_pp']:.2f} pp.")
        L += ["", f"### {outcome}: `{res['outcome_keys'][outcome]}`, n={d['n']}, "
                  f"balanced={d['balanced']}", "",
              f"Method: {d['method']}. {pub_line} Residual df with two-way terms: "
              f"{d['df_resid_with_interactions']}.", "",
              f"Residual **{_pct(d['residual_main_effects_only'])}** with main effects only → "
              f"**{_pct(d['residual_with_two_way'])}** after adding the ten two-way terms "
              f"(which together take {_pct(d['two_way_total'])}).", "",
              f"`{d['focus_term']}` = **{_pct(d['focus_share'])}**, which is "
              f"{_pct(d['focus_share_of_main_effects_residual'])} of the main-effects residual.",
              "", "| term | share | 95% seed bootstrap | bootstrap SD | F-test p |",
              "|---|---:|---:|---:|---:|"]
        rows = sorted([r for r in d["table"] if r["kind"] != "residual"],
                      key=lambda r: -r["share"])
        rows += [r for r in d["table"] if r["kind"] == "residual"]
        for r in rows:
            name = f"**{r['term']}**" if r["term"] == d["focus_term"] else r["term"]
            ci = (f"[{_pct(r['boot_lo'])}, {_pct(r['boot_hi'])}]" if "boot_lo" in r else "")
            L.append(f"| {name} | {_pct(r['share'])} | {ci} | "
                     f"{_pct(r.get('boot_sd'))} | {_p(r.get('f_test_p'))} |")
        if not d["balanced"]:
            L.append(f"\nType-II shares on unbalanced rows do not sum to 100% "
                     f"(here {_pct(d['sum_of_shares_incl_residual'])}).")
    return "\n".join(L)


def rq1_focus_markdown(before: dict, after: dict | None) -> str:
    """Compact side-by-side of the headline interaction numbers."""
    L = ["| outcome | n | residual (main only) | residual (+2-way) | all 2-way | "
         f"`{FOCUS_TERM}` [95% CI] | largest 2-way term |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for label, res in (("as committed", before), ("smoke obs. replaced", after)):
        if res is None:
            continue
        for outcome, d in res["outcomes"].items():
            f = next(r for r in d["table"] if r["term"] == FOCUS_TERM)
            top = max((r for r in d["table"] if r["kind"] == "interaction"),
                      key=lambda r: r["share"])
            ci = f" [{_pct(f['boot_lo'])}, {_pct(f['boot_hi'])}]" if "boot_lo" in f else ""
            L.append(f"| {outcome} ({label}) | {d['n']} | {_pct(d['residual_main_effects_only'])} | "
                     f"{_pct(d['residual_with_two_way'])} | {_pct(d['two_way_total'])} | "
                     f"{_pct(d['focus_share'])}{ci} | {top['term']} {_pct(top['share'])} |")
    return "\n".join(L)


# =====================================================================
# Report
# =====================================================================
def write_report(path: Path, *, session: dict, frozen: dict, guards: dict | None,
                 agg: dict | None, rq1: dict | None, rq1_corrected: dict | None) -> str:
    L = ["# RQ2/RQ4 completion + RQ1 interactions: session report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. GPU `{session.get('gpu')}`, "
         f"torch `{session.get('torch')}`, CUDA `{session.get('cuda')}`, "
         f"repo HEAD `{session.get('repo_head')}`.", "",
         "Thesis numbering throughout: code `rq1_verdict` = **RQ2**, code "
         "`rq2_verdict` = **RQ4**.", "",
         f"Frozen paths untouched ({', '.join(frozen.get('paths', []))}): "
         f"**{frozen.get('ok')}**" + (f" (changed: {frozen['changed']})" if frozen.get("changed") else ""),
         ""]
    if agg:
        cov = agg["coverage"]
        L += ["## The four deliverables", "",
              f"1. **Coverage:** {cov['n_records']}/{cov['n_grid_cells']} Phase A records "
              f"({cov['n_new_records']}/21 new)."]
        q = (agg["rq2_after_all_records"] if agg["rq2_quotable"] == "all_records"
             else agg["rq2_after_complete_designs_only"])
        parts = []
        for g in ("far", "near"):
            if g in q:
                e = q[g]["eta_squared"]
                parts.append(f"{g}: score {100 * e['score']:.2f}% / objective "
                             f"{100 * e['objective']:.3f}% = {q[g]['ratio_score_over_objective']:.1f}×")
        L.append(f"2. **RQ2 ({agg['rq2_quotable'].replace('_', ' ')}):** " + "; ".join(parts)
                 + f". Fully crossed: {agg['completeness']['fully_crossed']}.")
        v4 = agg["rq4_after"]
        if v4 and "error" not in v4:
            L.append(f"3. **RQ4:** ECE improved in {v4['ece_improved_in']}/{v4['n_cells']} evidential "
                     f"cells; AUROC preserved in {v4['auroc_preserved_in']}/{v4['auroc_comparisons']} "
                     f"comparisons (committed: {agg['rq4_before_committed']['ece_improved_in']}/"
                     f"{agg['rq4_before_committed']['n_cells']} and "
                     f"{agg['rq4_before_committed']['auroc_preserved_in']}/"
                     f"{agg['rq4_before_committed']['auroc_comparisons']}).")
    if rq1:
        L.append("4. **RQ1 interactions** (four full tables in the RQ1 section below). "
                 "Headline numbers:")
        L += ["", rq1_focus_markdown(rq1, rq1_corrected), ""]
    if guards:
        L += ["", guards_markdown(guards), ""]
        mism = [r for r in guards["rows"] if r["guard_status"] == "MISMATCH"]
        if mism:
            L += ["> **MISMATCH present.** These retrained cells do not reproduce the committed "
                  "grid numbers. Their Phase A records are still internally valid (each model is "
                  "scored under all four scores), but they are re-trainings, not the Step 10 "
                  "models, and `results/mvt_results.json` still holds the originals. Report "
                  "this before quoting byte-identical reproducibility for this slice.", ""]
    if agg:
        L += [rq2_rq4_markdown(agg), ""]
    if rq1:
        L += [rq1_markdown(rq1, "RQ1: two-way interactions (committed grid, as published)"), ""]
    if rq1_corrected:
        L += [rq1_markdown(rq1_corrected, "RQ1: sensitivity (the 20-episode smoke observation "
                                          "replaced by the same model's 600-episode scores)"), ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
