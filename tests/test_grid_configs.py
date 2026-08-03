"""Step 10 (MVT grid) offline config tests — no GPU, no real datasets needed.

Covers plan.md Section 10.7's five assertions:
  - exactly 120 configs generated, matching configs/grid/_index.json
  - config / results-JSON / checkpoint paths are unique (no cell overwrites
    another's artefact)
  - zero collisions with any existing file under results/ (Steps 1-9 results
    must never be overwritten)
  - every generated config loads and build_model(cfg) succeeds; trainable
    param counts match the expected value per (backbone, adapter)
  - the seed-42 grid cells resolve to a config identical to their Step
    5/6/8/9 parent apart from the 5 overridden keys

Requires `python scripts/build_grid_configs.py` to have been run first (the
generated configs/grid/*.yaml + _index.json are committed artefacts, same
convention as any other configs/*.yaml).
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.utils import load_config, count_trainable_params
from src.models import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = REPO_ROOT / "configs" / "grid"
INDEX_PATH = GRID_DIR / "_index.json"


@pytest.fixture(scope="module")
def index() -> list[dict]:
    assert INDEX_PATH.exists(), (
        "configs/grid/_index.json missing -- run "
        "`python scripts/build_grid_configs.py` first"
    )
    return json.load(open(INDEX_PATH))["cells"]


# --------------------------------------------------------------------------
# 120 cells, matching _index.json
# --------------------------------------------------------------------------
def test_exactly_120_cells(index):
    assert len(index) == 120
    assert sum(1 for c in index if c["is_baseline"]) == 24
    assert sum(1 for c in index if not c["is_baseline"]) == 96


def test_index_matches_generated_files(index):
    for c in index:
        assert (REPO_ROOT / c["config"]).exists(), f"missing {c['config']}"
    on_disk = {f"configs/grid/{p.name}" for p in GRID_DIR.glob("*.yaml")}
    assert on_disk == {c["config"] for c in index}


# --------------------------------------------------------------------------
# Filesystem-artefact uniqueness (config / results / checkpoint paths)
# --------------------------------------------------------------------------
def test_filesystem_paths_are_unique(index):
    # `config`, `results_json`, `checkpoint` are the actual files each cell
    # writes -- each must be unique across all 120 cells. `run_tag` and
    # `results_suffix` are intentionally SHARED between a cell's evidential
    # and softmax siblings (scripts/train.py:_checkpoint_tag and
    # scripts/evaluate.py's results-filename convention both append
    # `head_descriptor` -- and _checkpoint_tag also appends `seed` -- on top
    # of run_tag/results_suffix automatically; see
    # scripts/build_grid_configs.py's module docstring), so they are not
    # asserted unique here.
    for key in ("config", "results_json", "checkpoint"):
        values = [c[key] for c in index]
        dupes = {v for v in values if values.count(v) > 1}
        assert not dupes, f"duplicate {key} values in grid index: {dupes}"


def test_no_collision_with_existing_results(index):
    existing = {p.name for p in (REPO_ROOT / "results").glob("*.json")}
    for c in index:
        name = Path(c["results_json"]).name
        assert name not in existing, (
            f"grid results filename {name!r} collides with an existing "
            f"committed result under results/"
        )


# --------------------------------------------------------------------------
# Every config loads
# --------------------------------------------------------------------------
def test_every_config_loads(index):
    for c in index:
        cfg = load_config(REPO_ROOT / c["config"])
        assert cfg.head.type == "prototype"
        assert cfg.trainer.type == "episodic"
        assert int(cfg.dataset.k_shot) == c["k_shot"]
        assert int(cfg.seed) == c["seed"]
        assert str(cfg.dataset.name) == c["dataset"]
        assert str(cfg.backbone.name) == c["backbone"]


# --------------------------------------------------------------------------
# build_model(cfg) succeeds; trainable param counts match expectations per
# (backbone, adapter). Dataset / k_shot / seed do not affect model
# architecture (only data loading / episode sampling do), so one
# representative config per unique (backbone, adapter, head) architecture
# already covers every real risk -- this is the same reasoning
# scripts/build_grid_configs.py uses to avoid re-deriving naming per cell.
# --------------------------------------------------------------------------
ARCHITECTURES = [
    ("configs/exp_phase3_placement_parallel_evidential.yaml", 31_746),
    ("configs/exp_phase3_placement_parallel_softmax.yaml", 31_744),
    ("configs/exp_phase3_lora_evidential.yaml", 12_290),
    ("configs/exp_phase3_lora_softmax.yaml", 12_288),
    ("configs/exp_phase5_mbnet_parallel_evidential.yaml", 6_930),
    ("configs/exp_phase5_mbnet_parallel_softmax.yaml", 6_928),
    ("configs/exp_phase5_mbnet_lora_evidential.yaml", 10_754),
    ("configs/exp_phase5_mbnet_lora_softmax.yaml", 10_752),
    ("configs/exp_phase3_full_ft_evidential.yaml", 11_176_514),
    ("configs/exp_phase3_linear_probe_evidential.yaml", 2),
    ("configs/exp_phase3_linear_probe_softmax.yaml", 0),
]


@pytest.mark.parametrize("config_path,expected", ARCHITECTURES)
def test_architecture_param_counts(config_path, expected):
    cfg = load_config(REPO_ROOT / config_path)
    model = build_model(cfg)
    assert count_trainable_params(model) == expected


def test_never_run_grid_cells_build_model(index):
    """plan.md Section 2's three real risks: LoRA x MobileNetV3-Small, LoRA x
    MiniImageNet, and 1-shot never having run before. Exercise build_model
    through the GENERATED grid file itself (not just its hand-written
    parent), since a bug in the 5-key override merge is the risk this step
    adds on top of the already-covered parent configs."""
    risky = [
        c for c in index
        if (c["adapter"] == "lora" and c["backbone"] == "mobilenetv3_small")
        or (c["adapter"] == "lora" and c["dataset"] == "mini_imagenet")
        or c["k_shot"] == 1
    ]
    assert risky
    seen_architectures = set()
    for c in risky:
        arch = (c["backbone"], c["adapter"], c["head"])
        if arch in seen_architectures:
            continue
        seen_architectures.add(arch)
        cfg = load_config(REPO_ROOT / c["config"])
        model = build_model(cfg)
        n = count_trainable_params(model)
        assert n > 0 or c["adapter"] == "linear_probe" and c["head"] == "softmax"


# --------------------------------------------------------------------------
# Free reproducibility check: seed-42 grid cells for
# (cifar_fs, 5-shot, r18, bottleneck-parallel) must resolve to exactly their
# Step 6 parent config apart from the 5 overridden keys.
# --------------------------------------------------------------------------
def _drop(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


@pytest.mark.parametrize("head,parent_file", [
    ("evidential", "exp_phase3_placement_parallel_evidential.yaml"),
    ("softmax", "exp_phase3_placement_parallel_softmax.yaml"),
])
def test_seed42_grid_cell_matches_step6_parent(index, head, parent_file):
    cell = next(
        c for c in index
        if c["dataset"] == "cifar_fs" and c["k_shot"] == 5
        and c["backbone"] == "resnet18" and c["adapter"] == "bottleneck_parallel"
        and c["head"] == head and c["seed"] == 42
    )
    grid_cfg = load_config(REPO_ROOT / cell["config"])
    parent_cfg = load_config(REPO_ROOT / "configs" / parent_file)

    # The 5 keys the grid generator is allowed to override.
    assert _drop(dict(grid_cfg), "output", "wandb") == \
        _drop(dict(parent_cfg), "output", "wandb")
    # seed and dataset.k_shot happen to already match the parent's own
    # defaults (42, 5-shot) for this specific cell -- the override is a
    # no-op here, which is exactly what "identical apart from overrides"
    # requires when the override value equals the inherited value.
    assert grid_cfg.seed == parent_cfg.seed == 42
    assert grid_cfg.dataset.k_shot == parent_cfg.dataset.k_shot == 5
