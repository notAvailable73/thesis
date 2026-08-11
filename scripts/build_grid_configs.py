"""Step 10 (MVT grid execution) — generates the 120 grid configs.

96 PEFT cells (2 datasets x 2 shots x 2 backbones x 2 adapters x 2 heads x
3 seeds) + 24 baseline cells (CIFAR-FS + ResNet-18 only: 2 adapters
{full_ft, linear_probe} x 2 shots x 2 heads x 3 seeds) = 120 total, per
plan.md / implementation.txt Section 10 / proposal.txt Section 6.

Every grid config is a 5-key override (`seed`, `dataset.k_shot`,
`output.run_tag`, `output.results_dir`, `wandb.{disabled,mode,group,tags}`)
on top of one of the existing hand-written parent configs from Steps 5/6/8/9
(plus the 6 new LoRA parent configs this step adds for the two combinations
that never had a parent config: mbnet x LoRA, mini x LoRA). Nothing about the
recipe itself (LR, rank, KL schedule, evidence affine) is touched here — the
grid recipe is FROZEN across all 120 cells (plan.md Section 3's "keep the
grid recipe FROZEN" decision), so every cell is a controlled comparison that
differs only on the grid's own six axes.

The (adapter.type, head.type/interpretation) pair — and therefore the
checkpoint tag / results-JSON filename convention — is derived from the
REAL merged parent config via scripts.train._head_descriptor /
_checkpoint_tag (imported, not re-implemented), so this generator can never
drift from what scripts/train.py and scripts/evaluate.py actually name their
outputs.

Because the metrics-JSON schema is frozen (instructions.txt: adding a key
breaks every earlier config's byte-identical rerun) and the results filename
itself does not encode dataset / shots / backbone / seed, `configs/grid/
_index.json` is the sidecar the grid runner (scripts/run_mvt_grid.py) and
aggregator (scripts/aggregate_grid.py) both read instead of parsing filenames.

Usage:
    python scripts/build_grid_configs.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.utils import load_config
from scripts.train import _head_descriptor, _checkpoint_tag

CONFIGS_DIR = REPO_ROOT / "configs"
GRID_DIR = CONFIGS_DIR / "grid"
RESULTS_GRID_DIR = "results/grid"
INDEX_PATH = GRID_DIR / "_index.json"

HEADS = ["evidential", "softmax"]
SEEDS = [42, 43, 44]
SHOTS = [1, 5]
DATASETS = ["cifar_fs", "mini_imagenet"]
BACKBONES = ["resnet18", "mobilenetv3_small"]
PEFT_ADAPTERS = ["parallel", "lora"]          # short names used in run_tag
BASELINE_ADAPTERS = ["full_ft", "linear_probe"]

DATASET_SHORT = {"cifar_fs": "cifar", "mini_imagenet": "mini"}
BACKBONE_SHORT = {"resnet18": "r18", "mobilenetv3_small": "mbnet"}
#: descriptive label for _index.json's "adapter" field (distinct from the
#: short name used in filenames/run_tag, and from cfg.adapter.type itself --
#: "parallel" is ambiguous between Bottleneck and a hypothetical future
#: adapter, "bottleneck_parallel" is not).
ADAPTER_LABEL = {
    "parallel": "bottleneck_parallel",
    "lora": "lora",
    "full_ft": "full_ft",
    "linear_probe": "linear_probe",
}

#: PEFT parent map (plan.md Section 6, "Full parent map" table): every grid
#: cell must `extends:` one of these, never retype a recipe.
PEFT_PARENTS = {
    ("cifar_fs", "resnet18", "parallel"): {
        "evidential": "exp_phase3_placement_parallel_evidential.yaml",
        "softmax":    "exp_phase3_placement_parallel_softmax.yaml",
    },
    ("cifar_fs", "resnet18", "lora"): {
        "evidential": "exp_phase3_lora_evidential.yaml",
        "softmax":    "exp_phase3_lora_softmax.yaml",
    },
    ("cifar_fs", "mobilenetv3_small", "parallel"): {
        "evidential": "exp_phase5_mbnet_parallel_evidential.yaml",
        "softmax":    "exp_phase5_mbnet_parallel_softmax.yaml",
    },
    ("cifar_fs", "mobilenetv3_small", "lora"): {
        "evidential": "exp_phase5_mbnet_lora_evidential.yaml",
        "softmax":    "exp_phase5_mbnet_lora_softmax.yaml",
    },
    ("mini_imagenet", "resnet18", "parallel"): {
        "evidential": "exp_phase5_mini_parallel_evidential.yaml",
        "softmax":    "exp_phase5_mini_parallel_softmax.yaml",
    },
    ("mini_imagenet", "resnet18", "lora"): {
        "evidential": "exp_phase5_mini_lora_evidential.yaml",
        "softmax":    "exp_phase5_mini_lora_softmax.yaml",
    },
    ("mini_imagenet", "mobilenetv3_small", "parallel"): {
        "evidential": "exp_phase5_mini_mbnet_parallel_evidential.yaml",
        "softmax":    "exp_phase5_mini_mbnet_parallel_softmax.yaml",
    },
    ("mini_imagenet", "mobilenetv3_small", "lora"): {
        "evidential": "exp_phase5_mini_mbnet_lora_evidential.yaml",
        "softmax":    "exp_phase5_mini_mbnet_lora_softmax.yaml",
    },
}

#: Baseline parents (proposal Section 6, non-negotiable minimum): CIFAR-FS +
#: ResNet-18 only.
BASELINE_PARENTS = {
    "full_ft": {
        "evidential": "exp_phase3_full_ft_evidential.yaml",
        "softmax":    "exp_phase3_full_ft_softmax.yaml",
    },
    "linear_probe": {
        "evidential": "exp_phase3_linear_probe_evidential.yaml",
        "softmax":    "exp_phase3_linear_probe_softmax.yaml",
    },
}

#: Priority order (implementation.txt Section 10 "RISKS" / plan.md Section
#: 10.3): if compute runs out, the least-important cells drop first.
PRIORITY = {
    ("cifar_fs", 5): 1,
    ("mini_imagenet", 5): 2,
    ("cifar_fs", 1): 3,
    ("mini_imagenet", 1): 4,
}


def _run_tag(dataset: str, k_shot: int, backbone: str, adapter_short: str) -> str:
    return (f"grid_{DATASET_SHORT[dataset]}_{k_shot}shot_"
            f"{BACKBONE_SHORT[backbone]}_{adapter_short}")


def _cell_filename(dataset: str, k_shot: int, backbone: str, adapter_short: str,
                   head: str, seed: int) -> str:
    return (f"{DATASET_SHORT[dataset]}_{k_shot}shot_{BACKBONE_SHORT[backbone]}_"
            f"{adapter_short}_{head}_seed{seed}.yaml")


def _wandb_group(dataset: str, k_shot: int) -> str:
    return f"{DATASET_SHORT[dataset]}-{k_shot}shot"


def _write_cell(dataset: str, k_shot: int, backbone: str, adapter_short: str,
                head: str, seed: int, parent_rel: str, is_baseline: bool) -> dict:
    """Write one grid config YAML + return its _index.json entry."""
    run_tag = _run_tag(dataset, k_shot, backbone, adapter_short)
    filename = _cell_filename(dataset, k_shot, backbone, adapter_short, head, seed)
    config_rel = f"configs/grid/{filename}"
    group = _wandb_group(dataset, k_shot)
    tags = ["phase5", "step10", "grid"] + (["baseline"] if is_baseline else [])

    body = {
        "extends": f"../{parent_rel}",
        "seed": seed,
        "dataset": {"k_shot": k_shot},
        "output": {"run_tag": run_tag, "results_dir": RESULTS_GRID_DIR},
        "wandb": {"disabled": False, "mode": "online", "group": group, "tags": tags},
    }
    header = (
        f"# AUTO-GENERATED by scripts/build_grid_configs.py — do not hand-edit.\n"
        f"# Step 10 MVT grid cell: dataset={dataset} k_shot={k_shot} "
        f"backbone={backbone} adapter={ADAPTER_LABEL[adapter_short]} head={head} "
        f"seed={seed}\n"
    )
    with open(GRID_DIR / filename, "w") as f:
        f.write(header)
        yaml.safe_dump(body, f, sort_keys=False)

    # Resolve the fully-merged cfg to derive the real adapter.type / checkpoint
    # tag / results filename — never re-derive this convention independently
    # (scripts/train.py / scripts/evaluate.py own it; a local re-implementation
    # here would silently drift the moment either script's naming changes).
    cfg = load_config(GRID_DIR / filename)
    adapter_type = str(cfg.adapter.type)
    head_descriptor = _head_descriptor(cfg)
    ckpt_tag = _checkpoint_tag(cfg)
    results_suffix = f"{run_tag}_seed{seed}"
    results_json = f"{RESULTS_GRID_DIR}/{results_suffix}_{adapter_type}_{head_descriptor}_metrics.json"
    checkpoint = f"checkpoints/model_phase2_{ckpt_tag}.pt"

    return {
        "dataset": dataset,
        "k_shot": k_shot,
        "backbone": backbone,
        "adapter": ADAPTER_LABEL[adapter_short],
        "adapter_type": adapter_type,
        "head": head,
        "seed": seed,
        "is_baseline": is_baseline,
        "priority": PRIORITY[(dataset, k_shot)],
        "run_tag": run_tag,
        "config": config_rel,
        "results_suffix": results_suffix,
        "results_json": results_json,
        "checkpoint": checkpoint,
        "wandb_group": group,
        "wandb_tags": tags,
    }


def build_cells() -> list[dict]:
    cells = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for adapter_short in PEFT_ADAPTERS:
                parent_map = PEFT_PARENTS[(dataset, backbone, adapter_short)]
                for head in HEADS:
                    parent_rel = parent_map[head]
                    for k_shot in SHOTS:
                        for seed in SEEDS:
                            cells.append(_write_cell(
                                dataset, k_shot, backbone, adapter_short, head,
                                seed, parent_rel, is_baseline=False))
    for adapter_short in BASELINE_ADAPTERS:
        parent_map = BASELINE_PARENTS[adapter_short]
        for head in HEADS:
            parent_rel = parent_map[head]
            for k_shot in SHOTS:
                for seed in SEEDS:
                    cells.append(_write_cell(
                        "cifar_fs", k_shot, "resnet18", adapter_short, head,
                        seed, parent_rel, is_baseline=True))
    return cells


def _assert_no_collisions(cells: list[dict]) -> None:
    """Every config path / results JSON / checkpoint path must be unique
    across the 120 cells, and none may already exist under results/ (hard
    constraint: nothing Step 10 writes may overwrite a Steps 1-9 result).

    `results_suffix` alone is NOT required to be unique: it is shared between
    a cell's evidential and softmax variants by design (scripts/evaluate.py
    appends `_{adapter.type}_{head_descriptor}` to it, exactly as every
    pre-Step-10 --results-suffix value already does for its own
    evidential/softmax pair) — `results_json` is the field that must be
    unique."""
    for key in ("config", "results_json", "checkpoint"):
        values = [c[key] for c in cells]
        dupes = {v for v in values if values.count(v) > 1}
        if dupes:
            raise AssertionError(f"duplicate {key} values in the grid: {dupes}")

    existing = {p.name for p in (REPO_ROOT / "results").glob("*.json")}
    for c in cells:
        name = Path(c["results_json"]).name
        if name in existing:
            raise AssertionError(
                f"grid results filename {name!r} collides with an existing "
                f"file under results/ — this must never overwrite a prior "
                f"step's result.")


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    for old in GRID_DIR.glob("*.yaml"):
        old.unlink()

    cells = build_cells()
    _assert_no_collisions(cells)

    n_peft = sum(1 for c in cells if not c["is_baseline"])
    n_baseline = sum(1 for c in cells if c["is_baseline"])
    assert n_peft == 96, f"expected 96 PEFT cells, got {n_peft}"
    assert n_baseline == 24, f"expected 24 baseline cells, got {n_baseline}"
    assert len(cells) == 120, f"expected 120 total cells, got {len(cells)}"

    with open(INDEX_PATH, "w") as f:
        json.dump({"cells": cells}, f, indent=2, sort_keys=True)

    by_priority = {}
    for c in cells:
        by_priority.setdefault(c["priority"], 0)
        by_priority[c["priority"]] += 1
    print(f"wrote {len(cells)} configs to {GRID_DIR}/ ({n_peft} PEFT + "
          f"{n_baseline} baseline)")
    print(f"wrote {INDEX_PATH}")
    for p in sorted(by_priority):
        print(f"  priority {p}: {by_priority[p]} cells")


if __name__ == "__main__":
    main()
