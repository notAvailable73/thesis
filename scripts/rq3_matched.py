"""RQ3 matched-budget experiment — config generation, guards, driver, verdict.

Implements docs/RQ3_MATCHED_BUDGET_PLAN.md. The question it exists to close:

    RQ3's 16 bottleneck-vs-LoRA pairs show the ACCURACY winner never changing
    when the parameter-budget ordering reverses between backbones, while the
    CALIBRATION winner changes exactly in step with it. But the reversal is
    welded to the backbone (bottleneck is the larger arm on ResNet-18, the
    smaller one on MobileNetV3-Small), so two accounts predict the identical
    16/16 pattern and the existing grid cannot separate them:

        H3.2      (budget)            — the larger-budget arm wins calibration
        H3.2-alt  (backbone-intrinsic)— something about the backbone does

    Building BOTH architectures at the SAME budget WITHIN each backbone breaks
    the weld: H3.1 (architecture), H3.2 and H3.2-alt then predict different
    things for the first time (see `DECISION_RULES` / `adjudicate`).

Design decisions, all deliberate and all from the plan:

§2.1  Both arms of a comparison inherit the SAME training recipe. Every
      generated config is a MINIMAL OVERRIDE (`adapter.rank` + bookkeeping) on
      the parent the Step 10 grid itself used, so LR / epochs / patience / KL
      schedule / evidence affine arrive by inheritance and can never be
      hand-copied out of sync. `assert_matched_arms_controlled` proves this on
      the MERGED configs, not the YAML source, so a difference inherited from a
      parent cannot hide.

§2.2  Bottleneck `rank` must not exceed the SHALLOWEST stage's channel count.
      `Conv1x1Bottleneck(channels, rank)` builds `Conv2d(channels, rank, 1)`;
      with `rank > channels` that is an over-complete projection, not a
      bottleneck. MobileNetV3-Small's shallowest placement stage is 24 channels
      (features.3), so its bottleneck ranks are capped at 23 — which is why
      Level H is rank 22/14 rather than the tighter-matching rank 25.

§2.3  MiniImageNet 5-shot only. RQ3's ECE effects clear 2x the across-seed SD
      in 8/8 MiniImageNet pairs but only 2/8 CIFAR-FS pairs, so a null on
      CIFAR-FS would be indistinguishable from insufficient power.
      `k_shot: 5` is base.yaml's default and is deliberately NOT overridden.

§2.4  Both head interpretations are TRAINED SEPARATELY, never read off one
      checkpoint. RQ3's original 16 pairs used separately trained evidential
      and softmax cells; reading softmax off an evidential parent would not be
      the same experiment. This is why the run count is x2.

Nothing scientific is reimplemented here: training goes through the real
`scripts/train.py:main()`, evaluation through `rq_core.factorial_evaluate`
(and therefore through `PrototypeHead.to_evidence`, the single source of truth
for the evidence map), and the checkpoint / results naming through
`scripts/train.py`'s own `_checkpoint_tag` / `_head_descriptor`.
"""
from __future__ import annotations

import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

from src.utils import load_config, count_trainable_params
from src.models import build_model
from scripts.train import _head_descriptor, _checkpoint_tag

# The aggregation and the pre-registered decision rule are NOT reimplemented
# here: plan §6 Step 5 asks for rq_aggregate.py to be extended rather than for
# a parallel aggregator, so they are imported from its `RQ3 — matched budget`
# section under their unprefixed names. `import rq_aggregate` (bare, not
# `scripts.rq_aggregate`) is the convention rq_core / rq_drivers already use;
# the notebook puts scripts/ on sys.path for exactly this reason.
import rq_aggregate as _A
from rq_aggregate import (                       # noqa: F401  (re-exported)
    budget_mismatch_pct,
    RQ3M_DATASET as DATASET,
    RQ3M_K_SHOT as K_SHOT,
    RQ3M_HEADS as HEADS,
    RQ3M_SEEDS as SEEDS,
    RQ3M_NEAR_POOLS as NEAR_POOLS,
    RQ3M_FAR_POOLS as FAR_POOLS,
    RQ3M_NATIVE_PROB_SET as NATIVE_PROB_SET,
    RQ3M_NATIVE_OOD_SCORE as NATIVE_OOD_SCORE,
    RQ3M_MVT_OOD_SCORE as MVT_OOD_SCORE,
    RQ3M_SD_DDOF as SD_DDOF,
    RQ3M_MAX_BUDGET_MISMATCH_PCT as MAX_BUDGET_MISMATCH_PCT,
    RQ3M_PREREGISTERED_UNMATCHED_DELTA_ECE as PREREGISTERED_UNMATCHED_DELTA_ECE,
    RQ3M_COLLAPSE_RATIO_MAX as COLLAPSE_RATIO_MAX,
    RQ3M_SIGMA_MULTIPLE as SIGMA_MULTIPLE,
    RQ3M_MIN_CELLS_FIRING as MIN_CELLS_FIRING,
    RQ3M_DECISION_RULES as DECISION_RULES,
    rq3m_cell_metrics as cell_metrics,
    rq3m_arm_summary as arm_summary,
    rq3m_unmatched_deltas as unmatched_deltas,
    rq3m_matched_table as matched_table,
    rq3m_adjudicate as adjudicate,
    rq3m_secondary_outcomes as secondary_outcomes,
)


# =====================================================================
# §3/§4 — the design, and the parameter formulas it is built from
# =====================================================================
#: Per-backbone facts the design depends on. Every number here was derived
#: from source (src/adapters/placement.py, src/adapters/lora.py) and validated
#: against the committed rank-16 `n_params` in results/mvt_results.json.
BACKBONES = {
    "resnet18": {
        "short": "r18",
        # resolve_stage_paths -> layer{1..4}.1; channels 64/128/256/512.
        "stage_channels": (64, 128, 256, 512),
        # _DEFAULT_LORA_TARGETS -> layer4.0.downsample.0 (256 -> 512).
        "lora_in_out": (256, 512),
        "parents": {"btl": "exp_phase5_mini_parallel_{head}.yaml",
                    "lora": "exp_phase5_mini_lora_{head}.yaml"},
        "common_ancestor": "exp_phase5_mini_base_{head}.yaml",
    },
    "mobilenetv3_small": {
        "short": "mnv3",
        # mobilenetv3_stage_paths -> features.{3,6,8,11}; widths 24/40/48/96.
        "stage_channels": (24, 40, 48, 96),
        # _mobilenetv3_default_lora_target -> features.11.block.3.0 (576 -> 96).
        "lora_in_out": (576, 96),
        "parents": {"btl": "exp_phase5_mini_mbnet_parallel_{head}.yaml",
                    "lora": "exp_phase5_mini_mbnet_lora_{head}.yaml"},
        "common_ancestor": "exp_phase5_mini_mbnet_postpool_{head}.yaml",
    },
}


#: `grid_adapter` is the label configs/grid/_index.json uses, so an existing
#: arm can be matched to its Step 10 twin without re-deriving the convention.
#: The design lives here; the adjudication's copy of the same protocol facts
#: lives in rq_aggregate. Neither can drift silently past this assertion.
assert tuple(BACKBONES) == _A.RQ3M_BACKBONES, (tuple(BACKBONES), _A.RQ3M_BACKBONES)


ARMS = {"btl": {"adapter_type": "bottleneck", "grid_adapter": "bottleneck_parallel"},
        "lora": {"adapter_type": "lora", "grid_adapter": "lora"}}

#: §4 — 2 backbones x 2 matched budget levels x 2 arms. `status` is "existing"
#: when the arm IS a Step 10 grid cell (rank 16 on the same parent), so its
#: checkpoint can be reused instead of retrained, and its metrics double as the
#: §7 sanity check that this harness measures the same thing the grid did.
DESIGN = [
    # backbone,             level, arm,    rank, status
    ("resnet18",            "L",   "lora", 16,   "existing"),
    ("resnet18",            "L",   "btl",   6,   "new"),
    ("resnet18",            "H",   "btl",  16,   "existing"),
    ("resnet18",            "H",   "lora", 41,   "new"),
    ("mobilenetv3_small",   "L",   "btl",  16,   "existing"),
    ("mobilenetv3_small",   "L",   "lora", 10,   "new"),
    ("mobilenetv3_small",   "H",   "btl",  22,   "new"),
    ("mobilenetv3_small",   "H",   "lora", 14,   "new"),
]

#: §5/§7 — the pre-registered thresholds, the unmatched baselines they are
#: measured against, and the protocol constants all live with the
#: adjudication in rq_aggregate.py; imported above under these names so the
#: rest of this module reads unchanged.


def n_affine(head: str) -> int:
    """Evidential configs set `head.evidence_affine: true` -> 2 learnable
    scalars on top of the adapter. Softmax configs inherit base.yaml's
    `false`, which stores the same two numbers as buffers instead."""
    return 2 if head == "evidential" else 0


def expected_trainable_params(backbone: str, arm: str, rank: int, head: str) -> int:
    """Closed form for a cell's trainable-parameter count (§3).

    bottleneck-parallel: per stage a 1x1 down conv (C->r, +r bias) and a 1x1 up
    conv (r->C, +C bias)  ->  sum_stages (2*C*r + r + C)
    lora:                A (in x r) + B (r x out), both bias-free  ->  r*(in+out)

    Both reproduce the committed rank-16 `n_params` exactly; `assert_parameter_
    budgets` re-checks against the INSTANTIATED model because these ranks have
    never been built before.
    """
    spec = BACKBONES[backbone]
    if arm == "btl":
        core = sum(2 * c * rank + rank + c for c in spec["stage_channels"])
    elif arm == "lora":
        cin, cout = spec["lora_in_out"]
        core = rank * (cin + cout)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return core + n_affine(head)


def max_bottleneck_rank(backbone: str) -> int:
    """§2.2 — rank must stay strictly under the shallowest stage's width, or
    the "bottleneck" is an over-complete projection at that stage."""
    return min(BACKBONES[backbone]["stage_channels"]) - 1



# =====================================================================
# §6 Step 1 — config generation
# =====================================================================
def cell_id(backbone: str, arm: str, head: str, rank: int, seed: int) -> str:
    return (f"rq3m_{BACKBONES[backbone]['short']}_{arm}_{head}_"
            f"r{rank}_seed{seed}")


def run_tag(backbone: str, arm: str, rank: int) -> str:
    return f"rq3m_{BACKBONES[backbone]['short']}_{arm}_r{rank}"


def build_matched_configs(repo_root: Path, *, heads=HEADS, seeds=SEEDS,
                          out_dir_name="rq3_matched",
                          results_dir="results/rq3_matched",
                          wandb_disabled: bool = True) -> list[dict]:
    """Write one minimal-override YAML per (design arm, head, seed).

    A generated config carries exactly four things beyond `extends`: the seed,
    `adapter.rank`, output bookkeeping, and wandb bookkeeping. No recipe key is
    ever written here — that is the §2.1 constraint, and writing one would make
    the two arms incomparable in a way no downstream check could detect.
    """
    cfg_dir = repo_root / "configs" / out_dir_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []

    for backbone, level, arm, rank, status in DESIGN:
        if arm == "btl" and rank > max_bottleneck_rank(backbone):
            raise ValueError(
                f"§2.2 violated: bottleneck rank {rank} > "
                f"{max_bottleneck_rank(backbone)} on {backbone} — the "
                f"shallowest stage has {min(BACKBONES[backbone]['stage_channels'])} "
                f"channels, so this would be an over-complete projection, not a "
                f"bottleneck.")
        for head in heads:
            parent = BACKBONES[backbone]["parents"][arm].format(head=head)
            for seed in seeds:
                cid = cell_id(backbone, arm, head, rank, seed)
                tag = run_tag(backbone, arm, rank)
                doc = {
                    "extends": f"../{parent}",
                    "seed": int(seed),
                    "adapter": {"rank": int(rank)},
                    "output": {"run_tag": tag, "results_dir": results_dir},
                    "wandb": {"disabled": bool(wandb_disabled),
                              "group": "rq3-matched-budget",
                              "tags": ["rq3", "matched_budget", head,
                                       BACKBONES[backbone]["short"], f"level_{level}"]},
                }
                path = cfg_dir / f"{cid}.yaml"
                with open(path, "w") as f:
                    f.write(
                        "# AUTO-GENERATED by rq3_matched.build_matched_configs "
                        "— do not hand-edit.\n"
                        f"# RQ3 matched-budget: backbone={backbone} level={level} "
                        f"arm={arm} rank={rank} head={head} seed={seed} "
                        f"({status} arm)\n"
                        "# Minimal override on the Step 10 parent: only "
                        "adapter.rank and bookkeeping.\n")
                    yaml.safe_dump(doc, f, sort_keys=False)

                merged = load_config(path)
                cells.append({
                    "cell": cid,
                    "backbone": backbone,
                    "backbone_short": BACKBONES[backbone]["short"],
                    "arm": arm,
                    "grid_adapter": ARMS[arm]["grid_adapter"],
                    "level": level,
                    "rank": int(rank),
                    "head": head,
                    "seed": int(seed),
                    "status": status,
                    "dataset": DATASET,
                    "k_shot": int(merged.dataset.k_shot),
                    "expected_n_params": expected_trainable_params(
                        backbone, arm, rank, head),
                    "config": str(path.relative_to(repo_root)),
                    "parent": f"configs/{parent}",
                    "run_tag": tag,
                    "results_suffix": cid,
                    "checkpoint": f"checkpoints/model_phase2_{_checkpoint_tag(merged)}.pt",
                    "results_json": (f"{results_dir}/{cid}_{merged.adapter.type}_"
                                     f"{_head_descriptor(merged)}_metrics.json"),
                })

    _assert_k_shot(cells)
    with open(cfg_dir / "_index.json", "w") as f:
        json.dump({"cells": cells,
                   "design": [list(d) for d in DESIGN],
                   "dataset": DATASET,
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, indent=2, sort_keys=True)
    return cells


def _assert_k_shot(cells: list[dict]) -> None:
    bad = sorted({c["k_shot"] for c in cells} - {K_SHOT})
    if bad:
        raise RuntimeError(
            f"§2.3 violated: k_shot {bad} appeared in the merged configs; this "
            f"experiment is MiniImageNet {K_SHOT}-shot only.")


# =====================================================================
# §6 Step 2 — the parameter assertion (BEFORE any GPU time is spent)
# =====================================================================
def assert_parameter_budgets(repo_root: Path, cells: list[dict], *,
                             device="cpu", log=print) -> dict:
    """Build every distinct arm and assert instantiated == intended (§6 Step 2).

    The §3 formulas were validated at rank 16 only; the five NEW ranks have
    never been instantiated. This is the one check that catches a formula that
    happens to be right at 16 and wrong at 6 — and it costs seconds, against
    8.8 GPU-hours of otherwise-wasted training.

    Also enforces §7's matched-budget criterion per (backbone, level): the two
    arms must differ by <= MAX_BUDGET_MISMATCH_PCT, or the level is not matched
    and the experiment is void.
    """
    import torch

    by_arm: dict[tuple, dict] = {}
    for c in cells:
        key = (c["backbone"], c["arm"], c["rank"], c["head"])
        by_arm.setdefault(key, c)

    rows, failures = [], []
    for (backbone, arm, rank, head), c in sorted(by_arm.items()):
        cfg = load_config(repo_root / c["config"])
        model = build_model(cfg).to(device)
        got = int(count_trainable_params(model))
        want = int(c["expected_n_params"])
        ok = got == want
        rows.append({"backbone": backbone, "arm": arm, "rank": rank,
                     "head": head, "level": c["level"], "status": c["status"],
                     "expected": want, "instantiated": got, "ok": ok})
        log(f"  {backbone:<18} {arm:<4} r{rank:<3} {head:<11} "
            f"expected={want:>7,}  built={got:>7,}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{backbone}/{arm}/r{rank}/{head}: "
                            f"expected {want}, built {got}")
        del model
        if str(device) != "cpu":
            torch.cuda.empty_cache()

    levels = {}
    for r in rows:
        levels.setdefault((r["backbone"], r["level"], r["head"]), {})[r["arm"]] = r
    level_rows = []
    for (backbone, level, head), arms in sorted(levels.items()):
        if set(arms) != {"btl", "lora"}:
            failures.append(f"level {backbone}/{level}/{head} has arms {sorted(arms)}")
            continue
        a, b = arms["btl"]["instantiated"], arms["lora"]["instantiated"]
        pct = budget_mismatch_pct(a, b)
        ok = pct <= MAX_BUDGET_MISMATCH_PCT
        level_rows.append({"backbone": backbone, "level": level, "head": head,
                           "btl_params": a, "lora_params": b,
                           "mismatch_pct": round(pct, 4), "ok": ok,
                           "btl_rank": arms["btl"]["rank"],
                           "lora_rank": arms["lora"]["rank"]})
        log(f"  level {backbone:<18} {level} {head:<11} btl={a:>7,} "
            f"lora={b:>7,}  mismatch={pct:5.2f}%  "
            f"{'OK' if ok else 'NOT MATCHED'}")
        if not ok:
            failures.append(f"level {backbone}/{level}/{head}: mismatch "
                            f"{pct:.2f}% > {MAX_BUDGET_MISMATCH_PCT}%")

    return {"ok": not failures, "failures": failures,
            "arms": rows, "levels": level_rows}


# =====================================================================
# §6 Step 3 — the control guard
# =====================================================================
#: Keys this experiment is ALLOWED to vary. Widened relative to
#: rq5_sweep.ALLOWED_DIFF_PATHS because two DIFFERENT ADAPTER TYPES are being
#: compared here, not one adapter at two ranks.
ALLOWED_DIFF_PATHS = {
    "seed",
    "adapter.type", "adapter.rank", "adapter.placement", "adapter.block_ids",
    "adapter.stage_paths", "adapter.lora_targets", "adapter.alpha",
    "adapter.dropout",
    "output.run_tag", "output.results_dir",
    "wandb.group", "wandb.tags", "wandb.mode", "wandb.disabled",
}

#: Keys that differ BY DESIGN along the head axis (evidential vs softmax).
#: Enumerated rather than waved through: each is a real recipe difference
#: inherited from exp_phase2_evidential_retuned vs exp_phase2_softmax, and the
#: guard proves the list is exactly this and nothing more.
HEAD_AXIS_PATHS = {
    "head.interpretation", "head.evidence_affine",
    "head.evidence_scale_init", "head.evidence_bias_init",
    "loss.kl_weight_max", "loss.kl_anneal_steps",
    "loss.prior_per_class", "loss.use_variance",
}

#: Keys that differ BY DESIGN along the backbone axis.
BACKBONE_AXIS_PATHS = {"backbone.name", "backbone.feature_dim"}


def _flatten(d, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _differing_keys(merged: dict[str, dict]) -> dict[str, list]:
    keys = set()
    for m in merged.values():
        keys |= set(m.keys())
    diffs = {}
    for key in sorted(keys):
        vals = {json.dumps(m.get(key, "<MISSING>"), sort_keys=True, default=str)
                for m in merged.values()}
        if len(vals) > 1:
            diffs[key] = sorted(vals)[:4]
    return diffs


def assert_matched_arms_controlled(repo_root: Path, cells: list[dict]) -> dict:
    """§6 Step 3 — prove nothing but the intended axes moves.

    Two checks, because the plan's literal "identical across all 30 configs"
    cannot hold as written: head interpretation and backbone are themselves
    design axes here, so some keys MUST differ globally. The honest decomposition:

      within  — inside each (backbone, head) group (the group a bottleneck-vs-
                LoRA comparison is actually made in), only ALLOWED_DIFF_PATHS
                may differ. This is the check the experiment's validity rests
                on: LR, epochs, patience, KL schedule, evidence affine,
                dataset, episode files, n_way and k_shot are all covered by it.
      global  — across ALL configs, every differing key must be accounted for
                by ALLOWED_DIFF_PATHS, HEAD_AXIS_PATHS or BACKBONE_AXIS_PATHS.
                An unlisted key differing anywhere is a hard failure.

    Both run on the MERGED configs, so a difference inherited from a parent
    cannot hide in a 4-key override file.
    """
    merged = {c["cell"]: _flatten(dict(load_config(repo_root / c["config"])))
              for c in cells}

    groups: dict[tuple, dict] = {}
    for c in cells:
        groups.setdefault((c["backbone"], c["head"]), {})[c["cell"]] = merged[c["cell"]]

    within = {}
    within_ok = True
    for key, group in sorted(groups.items()):
        offenders = {k: v for k, v in _differing_keys(group).items()
                     if k not in ALLOWED_DIFF_PATHS}
        within[f"{key[0]}/{key[1]}"] = {
            "n_configs": len(group), "ok": not offenders,
            "offending_keys": offenders,
        }
        within_ok &= not offenders

    accounted = ALLOWED_DIFF_PATHS | HEAD_AXIS_PATHS | BACKBONE_AXIS_PATHS
    global_diffs = _differing_keys(merged)
    unaccounted = {k: v for k, v in global_diffs.items() if k not in accounted}

    return {
        "ok": within_ok and not unaccounted,
        "n_configs": len(cells),
        "within_group": within,
        "global_differing_keys": sorted(global_diffs),
        "unaccounted_keys": unaccounted,
        "expected_head_axis_keys": sorted(
            k for k in global_diffs if k in HEAD_AXIS_PATHS),
        "expected_backbone_axis_keys": sorted(
            k for k in global_diffs if k in BACKBONE_AXIS_PATHS),
    }


# =====================================================================
# §6 Step 4 — recovery + the driver
# =====================================================================
def needed_checkpoint_names(repo_root: Path, cells: list[dict]) -> set[str]:
    """Basenames worth extracting from attached datasets: this experiment's own
    checkpoints, plus the Step 10 twins the `existing` arms reuse.

    Filtering matters: the Step 10 artifact zips hold 99 checkpoints (~2.6 GB)
    and only 18 of them are twins of an arm here.
    """
    want = {Path(c["checkpoint"]).name for c in cells}
    grid = _grid_index(repo_root)
    if grid:
        for c in cells:
            twin = find_grid_twin(grid, c)
            if twin:
                want.add(Path(twin["checkpoint"]).name)
    return want


def _grid_index(repo_root: Path) -> list[dict]:
    p = repo_root / "configs" / "grid" / "_index.json"
    return json.load(open(p))["cells"] if p.exists() else []


def find_grid_twin(grid_index: list[dict], cell: dict) -> dict | None:
    """The Step 10 grid cell that IS this arm, or None.

    Generalises rq_drivers._find_grid_twin, whose filter is hardcoded to
    CIFAR-FS / 5-shot / bottleneck_parallel. Only rank-16 arms can have a twin:
    16 is the grid's frozen rank, so any other rank is by construction a cell
    Step 10 never ran.
    """
    if cell["status"] != "existing" or cell["rank"] != 16:
        return None
    for g in grid_index:
        if (g["dataset"] == DATASET and g["k_shot"] == K_SHOT
                and g["backbone"] == cell["backbone"]
                and g["adapter"] == cell["grid_adapter"]
                and g["head"] == cell["head"]
                and g["seed"] == cell["seed"]):
            return g
    return None


def restore_session_artifacts(repo_root: Path, cells: list[dict],
                              search_roots=("/kaggle/input",),
                              out_dir_name="rq3_matched", log=print) -> dict:
    """Bring back what an earlier session already paid for.

    Two kinds of input are handled, from zips or loose files:
      - checkpoints (`checkpoints/*.pt`) — this experiment's own, plus the
        Step 10 twins the three `existing` arms reuse;
      - per-cell result JSONs (`results/rq3_matched/*.json`) — so a resumed
        session skips cells that are already evaluated.

    `results/grid/` is NEVER written: those committed metrics are the baseline
    the reused arms are checked against (§7), and overwriting them with a copy
    of themselves would destroy the only independent check there is.
    """
    ckpt_dir = repo_root / "checkpoints"
    res_dir = repo_root / "results" / out_dir_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    want_ckpt = needed_checkpoint_names(repo_root, cells)
    res_prefix = f"results/{out_dir_name}/"
    copied_ckpt, copied_res, zips = [], [], []

    def _take_ckpt(name: str, reader) -> None:
        if name not in want_ckpt or (ckpt_dir / name).exists():
            return
        with open(ckpt_dir / name, "wb") as out:
            shutil.copyfileobj(reader, out)
        copied_ckpt.append(name)

    def _take_res(name: str, reader) -> None:
        if (res_dir / name).exists():
            return
        with open(res_dir / name, "wb") as out:
            shutil.copyfileobj(reader, out)
        copied_res.append(name)

    for root in search_roots:
        rp = Path(root)
        if not rp.exists():
            log(f"  (no such path: {root})")
            continue
        for p in sorted(rp.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".zip":
                zips.append(str(p))
                try:
                    with zipfile.ZipFile(p) as zf:
                        for m in zf.namelist():
                            if m.endswith(".pt") and "checkpoints/" in m:
                                with zf.open(m) as fh:
                                    _take_ckpt(Path(m).name, fh)
                            elif m.startswith(res_prefix) and m.endswith(".json"):
                                with zf.open(m) as fh:
                                    _take_res(Path(m).name, fh)
                except zipfile.BadZipFile:
                    log(f"  {p.name}: not a readable zip, skipped")
            elif p.suffix == ".pt" and p.name.startswith("model_phase2_"):
                with open(p, "rb") as fh:
                    _take_ckpt(p.name, fh)
            elif (p.suffix == ".json" and p.parent.name == out_dir_name
                  and not p.name.startswith("_")):
                with open(p, "rb") as fh:
                    _take_res(p.name, fh)

    log(f"  scanned {list(search_roots)}: {len(zips)} zip(s)")
    log(f"  restored {len(copied_ckpt)} checkpoint(s), "
        f"{len(copied_res)} result JSON(s)")
    return {"zips_seen": zips, "checkpoints_restored": sorted(copied_ckpt),
            "results_restored": sorted(copied_res),
            "n_wanted_checkpoints": len(want_ckpt)}


def coverage(repo_root: Path, cells: list[dict], out_dir: Path) -> dict:
    """What this session would actually have to do."""
    grid = _grid_index(repo_root)
    done = trained = reusable = todo = 0
    for c in cells:
        if (out_dir / f"{c['cell']}.json").exists():
            done += 1
        elif (repo_root / c["checkpoint"]).exists():
            trained += 1
        else:
            twin = find_grid_twin(grid, c)
            if twin and (repo_root / twin["checkpoint"]).exists():
                reusable += 1
            else:
                todo += 1
    return {"n_cells": len(cells), "evaluated": done,
            "checkpoint_present": trained, "reusable_from_grid": reusable,
            "needs_training": todo}


def run_order(cells: list[dict]) -> list[dict]:
    """Order runs so a session that stops early still closes whole comparisons.

    A ΔECE needs BOTH arms at all three seeds, so the sort completes one
    (backbone, level, head) pair — 6 runs — before starting the next. ResNet-18
    first: it carries the largest unmatched gaps (ΔECE ~ 0.111) and the fewest
    new arms, so the most informative half of the experiment lands first.
    """
    bb_order = {"resnet18": 0, "mobilenetv3_small": 1}
    lvl_order = {"L": 0, "H": 1}
    head_order = {"evidential": 0, "softmax": 1}
    arm_order = {"btl": 0, "lora": 1}
    return sorted(cells, key=lambda c: (bb_order[c["backbone"]],
                                        lvl_order[c["level"]],
                                        head_order[c["head"]],
                                        c["seed"], arm_order[c["arm"]]))


def run_matched_sweep(repo_root: Path, cells: list[dict], *, device,
                      out_dir: Path, logits_dir: Path | None,
                      num_episodes: int, wandb_mode: str,
                      max_minutes: float | None, reuse_grid: bool = True,
                      log=print) -> dict:
    """Train (or reuse) then evaluate every cell. Resumable and budget-aware.

    Mirrors rq_drivers.run_phase_b: in-process training (subprocess output can
    silently vanish on hosted notebooks), skip-if-done, and a `max_minutes`
    budget that finishes the cell in flight and then stops cleanly so a Kaggle
    session ends without losing the run it is in the middle of.

    For `existing` arms the committed Step 10 metrics are passed to the
    regression guard, which turns §7's "reused arms reproduce the grid" from an
    assumption into a measured, per-cell number.
    """
    import rq_core as R
    import rq_drivers as D

    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "_run_log.jsonl"
    started = time.monotonic()
    counts = {"ok": 0, "skipped_done": 0, "trained": 0, "reused_grid": 0,
              "error": 0, "not_run": 0}
    grid = _grid_index(repo_root)
    ordered = run_order(cells)

    for i, c in enumerate(ordered):
        out_json = out_dir / f"{c['cell']}.json"
        if out_json.exists():
            counts["skipped_done"] += 1
            continue
        if max_minutes is not None and (time.monotonic() - started) / 60 > max_minutes:
            counts["not_run"] = len(ordered) - i
            log(f"[rq3m] budget {max_minutes} min reached; {counts['not_run']} "
                f"cell(s) left for the next session (re-run this notebook with "
                f"the artifact zip attached and it resumes here).")
            break

        log(f"[rq3m] ({i + 1}/{len(ordered)}) {c['cell']}  "
            f"[{c['backbone']} level {c['level']} {c['arm']} r{c['rank']} "
            f"{c['head']} seed{c['seed']}]")
        ckpt = repo_root / c["checkpoint"]
        try:
            if not ckpt.exists() and reuse_grid:
                twin = find_grid_twin(grid, c)
                if twin and (repo_root / twin["checkpoint"]).exists():
                    shutil.copy2(repo_root / twin["checkpoint"], ckpt)
                    log(f"    reused Step 10 checkpoint: {twin['checkpoint']}")
                    counts["reused_grid"] += 1

            if not ckpt.exists():
                log("    training…")
                D.train_cell(repo_root / c["config"], wandb_mode=wandb_mode, log=log)
                counts["trained"] += 1

            twin = find_grid_twin(grid, c)
            committed = None
            if twin and (repo_root / twin["results_json"]).exists():
                committed = repo_root / twin["results_json"]

            rec = D.factorial_run_one(
                repo_root, repo_root / c["config"], ckpt, out_json,
                device=device, num_episodes=num_episodes,
                logits_out=(logits_dir / f"{c['cell']}.npz") if logits_dir else None,
                committed_metrics=committed, meta=c, log=log)

            if int(rec["n_params"]) != int(c["expected_n_params"]):
                raise RuntimeError(
                    f"parameter budget drifted at eval time: built "
                    f"{rec['n_params']}, expected {c['expected_n_params']}")

            pset = ("evidential_native" if c["head"] == "evidential" else "softmax")
            g = rec["regression_guard"]
            log(f"    acc={rec['summary'].get(f'accuracy_mean__{pset}', float('nan')):.4f} "
                f"ECE={rec['summary'].get(f'ece_pooled__{pset}', float('nan')):.4f} "
                f"params={rec['n_params']}  guard={g.get('status')}  "
                f"[{rec['wall_seconds']}s]")
            counts["ok"] += 1
            _append(run_log, {"cell": c["cell"], "status": "ok",
                              "guard": g.get("status"),
                              "n_params": rec["n_params"],
                              "wall_seconds": rec["wall_seconds"]})
        except Exception as e:  # noqa: BLE001 — one bad cell must not kill the sweep
            log(f"    ERROR: {e!r}")
            counts["error"] += 1
            _append(run_log, {"cell": c["cell"], "status": "error", "error": repr(e)})

    log(f"[rq3m] done: {counts}")
    return counts


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


# =====================================================================
# §6 Step 5 — aggregate and adjudicate
# =====================================================================
# The aggregation, the pre-registered decision rule and the thresholds it
# uses live in scripts/rq_aggregate.py's `RQ3 — matched budget` section,
# imported at the top of this module under their unprefixed names. The
# plan's §6 Step 5 asks for that module to be extended rather than for a
# parallel aggregator to be written here; what remains below is only the
# assembly of verdict.json and its rendering, both of which need this
# module's run harness (`cells`, the parameter assertion, the guard).

def build_verdict(repo_root: Path, out_dir: Path, cells: list[dict],
                  *, params_check: dict | None = None,
                  control_guard: dict | None = None) -> dict:
    """Assemble results/rq3_matched/verdict.json (§6 Step 5 / §7).

    Everything the acceptance criteria ask for lands in one file: per-cell
    metrics with across-seed SD, matched and unmatched ΔECE side by side, which
    rule fired, the parameter assertion, the control guard, the residual budget
    mismatch, and the reused-arm reproduction check.
    """
    recs = [r for r in _A.load_records(out_dir) if r.get("meta", {}).get("arm")]
    table = matched_table(recs)
    unmatched = unmatched_deltas(repo_root)
    decision = adjudicate(table, unmatched)

    expected = {c["cell"] for c in cells}
    got = {r["meta"]["cell"] for r in recs}
    reused = [{"cell": r["meta"]["cell"],
               "status": r["regression_guard"].get("status"),
               "n_exact": r["regression_guard"].get("n_exact"),
               "n_keys": r["regression_guard"].get("n_keys"),
               "max_abs_diff": r["regression_guard"].get("max_abs_diff")}
              for r in recs
              if r["meta"].get("status") == "existing"
              and r.get("regression_guard", {}).get("status") not in
              (None, "not_requested")]

    verdict = {
        "experiment": "RQ3 matched-budget (docs/RQ3_MATCHED_BUDGET_PLAN.md)",
        "protocol": (f"{DATASET} {K_SHOT}-shot, 5-way, frozen 600 TEST-seed "
                     f"episodes, seeds {list(SEEDS)}, both head interpretations "
                     f"trained separately"),
        "sign_convention": "every delta is LoRA arm minus bottleneck arm",
        "verdict": decision["verdict"],
        "reason": decision["reason"],
        "decision": decision,
        "secondary": secondary_outcomes(table, unmatched),
        "matched_table": table,
        "unmatched_baseline": unmatched,
        "coverage": {
            "n_expected": len(expected), "n_present": len(got),
            "missing_cells": sorted(expected - got),
            "new_arm_runs_expected": sum(1 for c in cells if c["status"] == "new"),
            "new_arm_runs_present": sum(
                1 for r in recs if r["meta"].get("status") == "new"),
        },
        "parameter_assertion": params_check,
        "control_guard": control_guard,
        "reused_arm_reproduction": reused,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    acc = verdict["coverage"]
    verdict["acceptance"] = {
        "all_runs_complete": acc["n_present"] == acc["n_expected"],
        "parameter_assertion_passed": bool((params_check or {}).get("ok")),
        "budget_matched_everywhere": all(
            r.get("budget_matched", True) for r in table.values()),
        "control_guard_passed": bool((control_guard or {}).get("ok")),
        "reused_arms_reproduce_grid": all(
            r["status"] in ("exact", "within_tol") for r in reused) if reused
        else None,
        "verdict_named": decision["verdict"] in
        ("budget", "architecture", "backbone_intrinsic", "inconclusive"),
        "secondary_accuracy_reported": True,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True, default=str)
    return verdict


def print_verdict(v: dict, log=print) -> None:
    """Human-readable rendering of verdict.json."""
    log("\n" + "=" * 78)
    log("RQ3 MATCHED-BUDGET — RESULT")
    log("=" * 78)
    log(f"protocol : {v['protocol']}")
    log(f"signs    : {v['sign_convention']}")
    cov = v["coverage"]
    log(f"coverage : {cov['n_present']}/{cov['n_expected']} cells evaluated "
        f"({cov['new_arm_runs_present']}/{cov['new_arm_runs_expected']} new-arm runs)")

    log("\n--- per (backbone, level, head): matched arms ---")
    log(f"  {'backbone/level/head':<38}{'btl r':>6}{'lora r':>7}{'mism%':>7}"
        f"{'ECE btl':>9}{'ECE lora':>10}{'ΔECE':>9}{'2σ':>8}{'sig':>5}")
    for key in sorted(v["matched_table"]):
        r = v["matched_table"][key]
        if "delta_ece" not in r:
            log(f"  {key:<38}  (incomplete: arms {sorted(r['arms'])})")
            continue
        log(f"  {key:<38}{r['btl_rank']:>6}{r['lora_rank']:>7}"
            f"{r['budget_mismatch_pct']:>7.2f}"
            f"{r['arms']['btl']['ece_mean']:>9.4f}"
            f"{r['arms']['lora']['ece_mean']:>10.4f}"
            f"{r['delta_ece']:>+9.4f}"
            f"{SIGMA_MULTIPLE * r['sigma_ece']:>8.4f}"
            f"{'yes' if r['significant_ece'] else 'no':>5}")

    log("\n--- per (backbone x head): matched vs unmatched, and which rule fires ---")
    log(f"  {'cell':<34}{'ΔECE unm':>10}{'ΔECE mat':>10}{'collapse':>10}"
        f"{'2σ':>8}   {'rule fired':<34}")
    for ck in sorted(v["decision"]["cells"]):
        c = v["decision"]["cells"][ck]
        rules = [n for n in ("budget", "architecture", "backbone_intrinsic")
                 if c.get(f"fires_{n}")]
        log(f"  {ck:<34}{c['delta_ece_unmatched']:>+10.4f}"
            f"{c['delta_ece_matched']:>+10.4f}{c['collapse_ratio']:>10.2f}"
            f"{SIGMA_MULTIPLE * c['sigma_ece']:>8.4f}   "
            f"{(', '.join(rules) or '—'):<34}")
    log(f"\n  rule counts (need >= {v['decision']['counts']['min_cells_to_fire']} "
        f"of 4; {v['decision']['counts']['n_cells']} cell(s) present): "
        f"budget={v['decision']['counts']['budget']}, "
        f"architecture={v['decision']['counts']['architecture']}, "
        f"backbone_intrinsic={v['decision']['counts']['backbone_intrinsic']}")

    log(f"\n>>> VERDICT: {v['verdict'].upper()}")
    log(f"    {v['reason']}")

    log("\n--- secondary outcomes (control, and a live test of H3.1) ---")
    for metric, s in v["secondary"].items():
        log(f"  {metric:<12} bottleneck wins {s['bottleneck_wins']}/"
            f"{s['n_comparisons']} matched comparisons "
            f"({s['bottleneck_wins_beyond_2sigma']} beyond 2σ); "
            f"cells whose gap collapsed: {s['cells_collapsed'] or 'none'}")
    log(f"  -> {v['secondary']['accuracy']['interpretation']}")

    log("\n--- acceptance criteria (§7) ---")
    for k, ok in sorted(v["acceptance"].items()):
        mark = "n/a" if ok is None else ("PASS" if ok else "FAIL")
        log(f"  [{mark:>4}] {k}")
    log("=" * 78)
