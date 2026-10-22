"""TMLR follow-up — the complete experiment design (plan Part A, A2/A3/A6/A8).

Every run the paper relies on is a `RunSpec` produced by `build_design()`.
Nothing else invents runs: configs, the manifest, shards, collection and
adjudication all key off these specs.

Budgets: `expected_params` uses the hardcoded `SHAPES` tables below,
independent of the adapter code. tmlr_configs.assert_budgets checks them
against the instantiated modules before any GPU time is spent.

Duplicates: a run that would train an identical model to another run exists
once. Six E4 grid cells are E1's seed-42 existing arms; six E3b cells are E3a's
seed-42 5-shot default arms. Analysis matches runs by attributes
(dataset, k_shot, backbone, arm, rank_spec, head, seed), never by experiment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

REPO = Path(__file__).resolve().parents[1]

RankSpec = Union[int, tuple]

# =====================================================================
# Shapes (independent of src/adapters; verified on torchvision 2026-10-22)
# =====================================================================
SHAPES = {
    # stage widths at the stage-final residual blocks; the single-site conv
    # (Cin, Cout, k); lora_ms per-rank cost per site = Cin*k*k + Cout.
    "resnet18": {"short": "r18", "widths": (64, 128, 256, 512),
                 "ss": (256, 512, 1), "ms_units": (640, 1280, 2560, 5120)},
    "resnet34": {"short": "r34", "widths": (64, 128, 256, 512),
                 "ss": (256, 512, 1), "ms_units": (640, 1280, 2560, 5120)},
    "mobilenetv3_small": {"short": "mnv3", "widths": (24, 40, 48, 96),
                          "ss": (576, 96, 1), "ms_units": (112, 280, 192, 672)},
    "mobilenetv2": {"short": "mnv2", "widths": (24, 32, 64, 96, 160),
                    "ss": (960, 320, 1), "ms_units": (168, 224, 448, 672, 1120)},
}
#: Full-FT trains the whole ResNet-18 trunk (fc removed); baselines exist on R18 only.
FULL_FT_PARAMS = {"resnet18": 11_176_512}

ARMS = ("btl_ms", "lora_ss", "lora_ms", "btl_ss")
#: Plan A3 — max/min - 1 within a level (0.0379 worst, R18/R34 level L).
MAX_LEVEL_SPREAD = 0.040

#: Plan A3 table. (backbone, level) -> {arm: rank_spec}. "U" = MNv2 unmatched pair.
LEVELS = {
    ("resnet18", "L"): {"btl_ms": 6, "lora_ss": 16, "lora_ms": (1, 1, 2, 1), "btl_ss": 15},
    ("resnet18", "H"): {"btl_ms": 16, "lora_ss": 41, "lora_ms": (3, 3, 4, 3), "btl_ss": 40},
    ("resnet34", "L"): {"btl_ms": 6, "lora_ss": 16, "lora_ms": (1, 1, 2, 1), "btl_ss": 15},
    ("resnet34", "H"): {"btl_ms": 16, "lora_ss": 41, "lora_ms": (3, 3, 4, 3), "btl_ss": 40},
    ("mobilenetv3_small", "L"): {"btl_ms": 16, "lora_ss": 10, "lora_ms": (6, 6, 6, 5), "btl_ss": 10},
    ("mobilenetv3_small", "H"): {"btl_ms": 22, "lora_ss": 14, "lora_ms": (7, 7, 7, 8), "btl_ss": 14},
    ("mobilenetv2", "L"): {"btl_ms": 15, "lora_ss": 9, "lora_ms": (4, 4, 4, 4, 5), "btl_ss": 9},
    ("mobilenetv2", "H"): {"btl_ms": 22, "lora_ss": 13, "lora_ms": (7, 6, 6, 6, 7), "btl_ss": 13},
}
#: The default (grid, rank-16) arms that no matched level contains, run as level "U"
#: so every backbone's UNMATCHED pair (btl_ms r16 vs lora_ss r16) exists under the
#: same seeds/stream as its matched levels (R-B's collapse ratio needs it).
#: R18/R34: both default arms are already in L/H. MNv3: btl_ms r16 is level L.
UNMATCHED = {"mobilenetv3_small": {"lora_ss": 16},
             "mobilenetv2": {"btl_ms": 16, "lora_ss": 16}}

HEADS = ("evidential", "softmax")
SEEDS_5 = (42, 43, 44, 45, 46)
SEEDS_3 = (42, 43, 44)
DS_SHORT = {"cifar_fs": "cifar", "mini_imagenet": "mini", "fgvc_aircraft": "air"}
HEAD_SHORT = {"evidential": "evid", "softmax": "soft"}
MODULE_OF = {"btl_ms": "btl", "btl_ss": "btl", "lora_ss": "lora", "lora_ms": "lora"}

#: E5-val grid (plan A8) and the current recipe it is compared against.
E5_KL = (0.0, 0.05, 0.1)
E5_PRIOR = (1.0, 0.5)
CURRENT_EDL = {"kl_weight_max": 0.1, "prior_per_class": 1.0}

#: Plan A10 per-run minutes (MiniImageNet/Aircraft 5-shot); replaced by Wave 0 timings.
BASE_MINUTES = {"resnet18": 9.0, "mobilenetv3_small": 7.0, "resnet34": 13.0, "mobilenetv2": 8.0}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    experiment: str            # e1 e2 e3a e3b e4 e5val e5
    dataset: str
    k_shot: int
    backbone: str
    arm: str                   # btl_ms lora_ss lora_ms btl_ss full_ft linear_probe
    level: Optional[str]       # L H U or None (grid-style)
    rank_spec: RankSpec        # int, or tuple of per-site ranks for lora_ms
    head: str
    seed: int
    vary_stream: bool
    config: str                # repo-relative path of the generated config
    parent: str                # repo-relative path the config extends
    expected_n_params: int
    est_minutes: float
    wave: int                  # 0, 1, 2; -1 = blocked (E5 before selection)
    mode: str = "test"         # test | val_only
    regression_ref: Optional[str] = None
    regression_kind: Optional[str] = None   # grid | factorial
    loss_overrides: tuple = field(default_factory=tuple)   # ((key, value), ...)

    def to_json(self) -> dict:
        d = asdict(self)
        d["rank_spec"] = list(self.rank_spec) if isinstance(self.rank_spec, tuple) else self.rank_spec
        d["loss_overrides"] = dict(self.loss_overrides)
        return d


# =====================================================================
# Parameter closed forms
# =====================================================================
def expected_params(backbone: str, arm: str, rank_spec: RankSpec, head: str) -> int:
    """Trainable parameters from SHAPES alone (+2 evidence scalars if evidential)."""
    sh = SHAPES[backbone]
    if arm == "btl_ms":
        r = int(rank_spec)
        core = sum(2 * c * r + r + c for c in sh["widths"])
    elif arm == "lora_ss":
        cin, cout, k = sh["ss"]
        core = int(rank_spec) * (cin * k * k + cout)
    elif arm == "btl_ss":
        cin, cout, _k = sh["ss"]
        r = int(rank_spec)
        core = r * (cin + cout) + r + cout
    elif arm == "lora_ms":
        ranks = tuple(rank_spec)
        if len(ranks) != len(sh["ms_units"]):
            raise ValueError(f"{backbone} has {len(sh['ms_units'])} sites, got ranks {ranks}")
        core = sum(r * u for r, u in zip(ranks, sh["ms_units"]))
    elif arm == "full_ft":
        core = FULL_FT_PARAMS[backbone]
    elif arm == "linear_probe":
        core = 0
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return core + (2 if head == "evidential" else 0)


def adapter_block(backbone: str, arm: str, rank_spec: RankSpec) -> dict:
    """The full `adapter:` override written into a generated config."""
    n_stages = len(SHAPES[backbone]["widths"])
    if arm == "btl_ms":
        return {"type": "bottleneck", "placement": "parallel", "rank": int(rank_spec),
                "block_ids": list(range(n_stages))}
    if arm == "btl_ss":
        return {"type": "bottleneck", "placement": "conv_parallel", "rank": int(rank_spec)}
    if arm == "lora_ss":
        return {"type": "lora", "rank": int(rank_spec)}
    if arm == "lora_ms":
        ranks = [int(r) for r in rank_spec]
        return {"type": "lora", "rank": max(ranks), "lora_targets": "multisite",
                "lora_ranks": ranks}
    raise ValueError(f"no generated adapter block for arm {arm!r}")


# =====================================================================
# Naming, parents, estimates
# =====================================================================
def _rank_str(rank_spec: RankSpec) -> str:
    return ("r" + "-".join(str(r) for r in rank_spec)) if isinstance(rank_spec, tuple) \
        else f"r{rank_spec}"


def run_id(exp, dataset, k, backbone, arm, level, rank_spec, head, seed, suffix="") -> str:
    lvl = level if level else "grid"
    return (f"{exp}_{DS_SHORT[dataset]}_{k}s_{SHAPES[backbone]['short']}_{arm}_{lvl}_"
            f"{_rank_str(rank_spec)}_{HEAD_SHORT[head]}{suffix}_s{seed}")


def parent_for(dataset: str, backbone: str, arm: str, head: str) -> str:
    module = MODULE_OF[arm]
    if dataset == "mini_imagenet" and backbone == "resnet18":
        return f"configs/exp_phase5_mini_{'parallel' if module == 'btl' else 'lora'}_{head}.yaml"
    if dataset == "mini_imagenet" and backbone == "mobilenetv3_small":
        return f"configs/exp_phase5_mini_mbnet_{'parallel' if module == 'btl' else 'lora'}_{head}.yaml"
    prefix = {"mini_imagenet": "mini", "fgvc_aircraft": "air"}[dataset]
    return f"configs/tmlr/parents/{prefix}_{SHAPES[backbone]['short']}_{module}_{head}.yaml"


def estimate_minutes(dataset: str, k: int, backbone: str, arm: str, mode: str = "test") -> float:
    m = BASE_MINUTES[backbone]
    if k == 1:
        m *= 0.85
    if dataset == "cifar_fs":
        m *= 1.4
    if arm == "full_ft":
        m *= 3.0
    elif arm == "linear_probe":
        m *= 0.6
    if mode == "val_only":
        m *= 0.7
    return round(m, 2)


def _spec(exp, dataset, k, backbone, arm, level, rank_spec, head, seed, vary, wave, *,
          parent=None, mode="test", ref=None, ref_kind=None, loss=(), suffix="") -> RunSpec:
    rid = run_id(exp, dataset, k, backbone, arm, level, rank_spec, head, seed, suffix)
    return RunSpec(
        run_id=rid, experiment=exp, dataset=dataset, k_shot=k, backbone=backbone, arm=arm,
        level=level, rank_spec=rank_spec, head=head, seed=seed, vary_stream=vary,
        config=f"configs/tmlr/{exp}/{rid}.yaml",
        parent=parent or parent_for(dataset, backbone, arm, head),
        expected_n_params=expected_params(backbone, arm, rank_spec, head),
        est_minutes=estimate_minutes(dataset, k, backbone, arm, mode), wave=wave,
        mode=mode, regression_ref=ref, regression_kind=ref_kind, loss_overrides=tuple(loss))


def training_identity(s: RunSpec) -> tuple:
    """Two specs with the same identity would train the same model."""
    stream = s.seed if (s.vary_stream and s.seed != 42) else "fixed"
    return (s.dataset, s.k_shot, s.backbone, s.arm, s.rank_spec, s.head, s.seed, stream,
            s.loss_overrides, s.mode)


# =====================================================================
# Regression references
# =====================================================================
_RQ3M_SHORT = {"resnet18": "r18", "mobilenetv3_small": "mnv3"}
_RQ3M_ARM = {"btl_ms": "btl", "lora_ss": "lora"}


def _rq3m_ref(backbone, arm, rank, head) -> Optional[str]:
    if backbone not in _RQ3M_SHORT or arm not in _RQ3M_ARM:
        return None
    p = f"results/rq3_matched/rq3m_{_RQ3M_SHORT[backbone]}_{_RQ3M_ARM[arm]}_{head}_r{rank}_seed42.json"
    return p if (REPO / p).exists() else None


def _grid_cells() -> list[dict]:
    return json.load(open(REPO / "configs/grid/_index.json"))["cells"]


_GRID_ARM = {"bottleneck_parallel": "btl_ms", "lora": "lora_ss",
             "full_ft": "full_ft", "linear_probe": "linear_probe"}


# =====================================================================
# The design
# =====================================================================
WAVE0_RUN_IDS = (
    "e1_mini_5s_r18_btl_ms_H_r16_evid_s42",            # regression vs rq3_matched
    "e1_mini_5s_mnv3_lora_ss_L_r10_soft_s42",          # regression vs rq3_matched
    "e1_mini_5s_r18_lora_ms_H_r3-3-4-3_evid_s42",      # new arm: multi-site LoRA
    "e1_mini_5s_r18_btl_ss_H_r40_soft_s42",            # new arm: single-site bottleneck (stride 2)
    "e2_mini_5s_r34_btl_ms_H_r16_soft_s42",            # new backbone
    "e2_mini_5s_mnv2_lora_ss_U_r16_evid_s42",          # new backbone, 5-stage family
    "e2_mini_5s_mnv2_btl_ss_H_r13_evid_s42",           # btl_ss on a MobileNet projection conv
    "e4_cifar_1s_r18_btl_ms_grid_r16_soft_s42",        # regression vs results/grid
    "e5val_cifar_5s_r18_btl_ms_grid_r16_evid_kl0.0_p1.0_s42",   # val_only path
)


def _matched(exp, dataset, backbones, seeds, wave) -> list[RunSpec]:
    out = []
    for bb in backbones:
        levels = [(lvl, arms) for (b, lvl), arms in LEVELS.items() if b == bb]
        if bb in UNMATCHED:
            levels.append(("U", UNMATCHED[bb]))
        for lvl, arms in levels:
            for arm, rs in arms.items():
                for head in HEADS:
                    for seed in seeds:
                        ref = kind = None
                        if exp == "e1" and seed == 42 and not isinstance(rs, tuple):
                            ref = _rq3m_ref(bb, arm, rs, head)
                            kind = "factorial" if ref else None
                        out.append(_spec(exp, dataset, 5, bb, arm, lvl, rs, head, seed,
                                         True, wave, ref=ref, ref_kind=kind))
    return out


def build_design(e5_selection: Optional[dict] = "auto") -> list[RunSpec]:
    """All runs (plan A2). `e5_selection`: "auto" reads configs/tmlr/e5_selected.yaml;
    None means not selected yet (E5 emitted with wave -1); a dict is used as given."""
    if e5_selection == "auto":
        p = REPO / "configs/tmlr/e5_selected.yaml"
        import yaml
        e5_selection = yaml.safe_load(open(p)) if p.exists() else None

    specs: list[RunSpec] = []
    # E1 / E2 — matched budgets, sites x module, 5 seeds, varied stream.
    specs += _matched("e1", "mini_imagenet", ("resnet18", "mobilenetv3_small"), SEEDS_5, 1)
    specs += _matched("e2", "mini_imagenet", ("resnet34", "mobilenetv2"), SEEDS_5, 1)

    # E3a — Aircraft sub-grid with the original grid recipe (fixed stream).
    e3a = []
    for k in (1, 5):
        for bb in ("resnet18", "mobilenetv3_small"):
            for arm in ("btl_ms", "lora_ss"):
                for head in HEADS:
                    for seed in SEEDS_3:
                        e3a.append(_spec("e3a", "fgvc_aircraft", k, bb, arm, None, 16, head,
                                         seed, False, 2))
    specs += e3a
    # E3b — Aircraft matched budgets, 3 seeds, varied stream, minus E3a twins.
    e3b = _matched("e3b", "fgvc_aircraft", ("resnet18", "mobilenetv3_small"), SEEDS_3, 2)
    taken = {training_identity(s) for s in specs}
    specs += [s for s in e3b if training_identity(s) not in taken]

    # E4 — the original 120-run grid, wrapping configs/grid/*, minus E1 twins. A twin
    # that carries no rq3_matched reference inherits the grid reference instead, so
    # every grid cell is still checked against its committed August numbers.
    index = {training_identity(s): i for i, s in enumerate(specs)}
    for c in _grid_cells():
        arm = _GRID_ARM[c["adapter"]]
        rank = 16 if arm in ("btl_ms", "lora_ss") else 0
        s = _spec("e4", c["dataset"], int(c["k_shot"]), c["backbone"], arm, None, rank,
                  c["head"], int(c["seed"]), False, 2, parent=c["config"],
                  ref=c["results_json"], ref_kind="grid")
        twin = index.get(training_identity(s))
        if twin is None:
            specs.append(s)
        elif specs[twin].regression_ref is None:
            specs[twin] = RunSpec(**{**asdict(specs[twin]), "regression_ref": c["results_json"],
                                     "regression_kind": "grid"})

    # E5-val — VAL-only EDL recipe sweep (plan A8). Never touches TEST.
    for kl in E5_KL:
        for prior in E5_PRIOR:
            specs.append(_spec(
                "e5val", "cifar_fs", 5, "resnet18", "btl_ms", None, 16, "evidential", 42,
                False, 1, parent="configs/grid/cifar_5shot_r18_parallel_evidential_seed42.yaml",
                mode="val_only", loss=(("kl_weight_max", kl), ("prior_per_class", prior)),
                suffix=f"_kl{kl}_p{prior}"))

    # E5 — the evidential half of the 2^5 grid with the selected recipe.
    if not (e5_selection and e5_selection.get("skipped")):
        wave = 2 if e5_selection else -1
        loss = (() if not e5_selection else
                (("kl_weight_max", float(e5_selection["kl_weight_max"])),
                 ("prior_per_class", float(e5_selection["prior_per_class"]))))
        for c in _grid_cells():
            arm = _GRID_ARM[c["adapter"]]
            if c["head"] != "evidential" or arm not in ("btl_ms", "lora_ss"):
                continue
            specs.append(_spec("e5", c["dataset"], int(c["k_shot"]), c["backbone"], arm, None,
                               16, "evidential", int(c["seed"]), False, wave,
                               parent=c["config"], loss=loss))

    w0 = set(WAVE0_RUN_IDS)
    specs = [RunSpec(**{**asdict(s), "wave": 0}) if s.run_id in w0 else s for s in specs]
    missing = w0 - {s.run_id for s in specs}
    if missing:
        raise RuntimeError(f"WAVE0_RUN_IDS not in the design: {sorted(missing)}")
    return specs
