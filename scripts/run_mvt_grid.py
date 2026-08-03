"""Step 10 (MVT grid execution) — resumable, priority-ordered driver over the
120 cells in configs/grid/_index.json.

Runs scripts/train.py + scripts/evaluate.py IN-PROCESS (imports their main()
and patches sys.argv) rather than via subprocess, because Step 9 found
subprocess output can silently vanish on hosted notebooks (instructions.txt
section 4d / scripts/step45_val_sweep.py's docstring) — the same reason that
script is an in-process sweep instead of a shelled-out loop. Unlike that
script, this one does NOT re-implement train/eval logic: it calls the real
scripts.train.main() / scripts.evaluate.main(), so there is exactly one
source of truth for what a "run" does.

Resumability: a cell whose results JSON already exists is skipped entirely
(--resume). A cell whose checkpoint already exists but whose results JSON
does not (interrupted after train, before eval) skips straight to eval. This
mirrors implementation.txt's "skip on existing metrics JSON" spec, extended
one step further (skip on existing checkpoint too) so a mid-cell timeout
doesn't retrain from scratch.

One cell failing (collapse-guard fired, or any other exception) is logged and
the grid moves on to the next cell — a 120-run unattended driver should never
let one bad cell abort everything a session has already paid for; that is
exactly what --resume across sessions and results/grid/_run_log.jsonl are for.

Usage:
    python scripts/build_grid_configs.py            # once, or after any parent-config change
    python scripts/run_mvt_grid.py --resume --only "dataset=cifar_fs,shots=5"
    python scripts/run_mvt_grid.py --dry-run --priority
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INDEX_PATH = REPO_ROOT / "configs" / "grid" / "_index.json"
RUN_LOG_PATH = REPO_ROOT / "results" / "grid" / "_run_log.jsonl"

#: convenience aliases for --only, on top of the literal _index.json values.
_DATASET_ALIASES = {"cifar": "cifar_fs", "mini": "mini_imagenet"}
_BACKBONE_ALIASES = {"r18": "resnet18", "mbnet": "mobilenetv3_small"}


def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"{INDEX_PATH} not found — run `python scripts/build_grid_configs.py` "
            f"first."
        )
    return json.load(open(INDEX_PATH))["cells"]


def _parse_only(spec: str | None) -> dict:
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "shots":
            key = "k_shot"
            value = int(value)
        elif key == "seed":
            value = int(value)
        elif key == "dataset":
            value = _DATASET_ALIASES.get(value, value)
        elif key == "backbone":
            value = _BACKBONE_ALIASES.get(value, value)
        out[key] = value
    return out


def _matches(cell: dict, filt: dict) -> bool:
    return all(cell.get(k) == v for k, v in filt.items())


def _cell_id(c: dict) -> str:
    return (f"{c['dataset']}/{c['k_shot']}shot/{c['backbone']}/{c['adapter']}/"
            f"{c['head']}/seed{c['seed']}")


def _run_module_main(argv: list[str]):
    """Import + call scripts.train.main() / scripts.evaluate.main() with
    `argv` patched onto sys.argv (argparse reads sys.argv[1:]). Imported
    lazily so --dry-run never pays torch's import cost."""
    import scripts.train as train_mod
    import scripts.evaluate as evaluate_mod
    module = train_mod if argv[0] == "train.py" else evaluate_mod
    old_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _read_best_val_epoch(ckpt_path: Path) -> int | None:
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return int(ckpt.get("best_val_epoch", -1))
    except Exception:
        return None


def _append_log(entry: dict) -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _run_cell(cell: dict, args) -> dict:
    from src.trainers import EpisodicCollapse

    config_path = REPO_ROOT / cell["config"]
    results_json = REPO_ROOT / cell["results_json"]
    checkpoint = REPO_ROOT / cell["checkpoint"]

    entry = {"cell": _cell_id(cell), "config": cell["config"],
             "results_json": cell["results_json"]}
    start = time.monotonic()

    if args.resume and results_json.exists():
        entry.update(status="skipped_done", wall_seconds=0.0)
        _append_log(entry)
        return entry

    status, error = "ok", None
    try:
        if not checkpoint.exists():
            train_argv = ["train.py", "--config", str(config_path)]
            if args.wandb_mode:
                train_argv += ["--wandb-mode", args.wandb_mode]
            _run_module_main(train_argv)
        else:
            print(f"[grid] checkpoint exists, skipping train: {checkpoint}")

        eval_argv = ["evaluate.py", "--config", str(config_path),
                    "--num-episodes", str(args.num_episodes),
                    "--results-suffix", cell["results_suffix"]]
        if args.wandb_mode:
            eval_argv += ["--wandb-mode", args.wandb_mode]
        if args.use_tinyimagenet:
            eval_argv.append("--use-tinyimagenet")
        if args.use_gaussian:
            eval_argv.append("--use-gaussian")
        _run_module_main(eval_argv)
    except EpisodicCollapse as e:
        status, error = "collapsed", str(e)
    except Exception as e:  # noqa: BLE001 — one bad cell must not kill the grid
        status, error = "error", repr(e)

    entry["status"] = status
    entry["error"] = error
    entry["wall_seconds"] = round(time.monotonic() - start, 1)
    entry["best_val_epoch"] = (_read_best_val_epoch(checkpoint)
                               if checkpoint.exists() else None)
    _append_log(entry)

    if status == "ok" and args.keep_checkpoints and checkpoint.exists():
        kept_seed = int(args.keep_checkpoints.replace("seed", ""))
        if cell["seed"] != kept_seed:
            checkpoint.unlink()
            print(f"[grid] deleted checkpoint (seed {cell['seed']} != kept "
                 f"{kept_seed}): {checkpoint}")

    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Skip any cell whose results JSON already exists.")
    ap.add_argument("--priority", action="store_true",
                    help="Run cells in priority order (P1..P4) instead of "
                        "generation order.")
    ap.add_argument("--only", default=None,
                    help="Filter, e.g. 'dataset=cifar_fs,shots=5'. Keys: "
                        "dataset, shots, backbone, adapter, head, seed.")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Stop cleanly (finish the current cell, then exit) "
                        "once this many minutes have elapsed.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned cells; run nothing.")
    ap.add_argument("--keep-checkpoints", default=None, metavar="seedNN",
                    help="e.g. 'seed42': delete every OTHER seed's checkpoint "
                        "right after that cell's eval succeeds.")
    ap.add_argument("--num-episodes", type=int, default=600)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default=None)
    ap.add_argument("--use-tinyimagenet", action="store_true")
    ap.add_argument("--use-gaussian", action="store_true")
    args = ap.parse_args()

    cells = _load_index()
    filt = _parse_only(args.only)
    cells = [c for c in cells if _matches(c, filt)]
    if args.priority:
        cells = sorted(cells, key=lambda c: (
            c["priority"], c["dataset"], c["k_shot"], c["backbone"],
            c["adapter"], c["head"], c["seed"]))

    print(f"[grid] {len(cells)} cells selected"
         f"{' (filtered by ' + args.only + ')' if args.only else ''}")

    if args.dry_run:
        for c in cells:
            print(f"  [{c['priority']}] {_cell_id(c)} -> {c['config']}")
        return

    grid_start = time.monotonic()
    n_ok = n_collapsed = n_error = n_skipped = 0
    for i, cell in enumerate(cells):
        if args.max_minutes is not None:
            elapsed_min = (time.monotonic() - grid_start) / 60.0
            if elapsed_min > args.max_minutes:
                print(f"[grid] --max-minutes {args.max_minutes} reached "
                     f"({elapsed_min:.1f} min elapsed); stopping with "
                     f"{len(cells) - i} cell(s) remaining.")
                break

        print(f"[grid] ({i + 1}/{len(cells)}) {_cell_id(cell)}")
        result = _run_cell(cell, args)
        status = result["status"]
        n_ok += status == "ok"
        n_collapsed += status == "collapsed"
        n_error += status == "error"
        n_skipped += status == "skipped_done"
        print(f"[grid]   -> {status}"
             f"{' (' + result['error'] + ')' if result.get('error') else ''}"
             f"  [{result['wall_seconds']}s]")

    print(f"[grid] done: {n_ok} ok, {n_skipped} skipped (already done), "
         f"{n_collapsed} collapsed, {n_error} errored "
         f"(log: {RUN_LOG_PATH})")


if __name__ == "__main__":
    main()
