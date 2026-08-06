"""Step 11 (RQ4) -- per-(backbone, adapter, head) efficiency measurement.

Measures trainable/total params, FLOPs (MACs, via fvcore per
implementation.txt 11.1), latency (GPU + CPU), and peak memory for every
(backbone, adapter, head) combination in the closed Step-10 grid, plus (an
addition, flagged as such) a training-step memory/time measurement. Writes
results/efficiency_table.json.

WHY 12 KEYS, NOT 40 CELLS. The grid's 40 mvt_results.json leaf cells vary
over (dataset, k_shot, backbone, adapter, head), but nothing about a model's
compute or parameter count depends on WHICH DATASET it trains on, and shot
regime only changes episode input shapes (handled by per-shot units below,
not by re-measuring anything dataset-specific). So the measurement-relevant
identity is (backbone, adapter, head): exactly the 12 tuples derived here
from configs/grid/_index.json (never hardcoded -- keys_from_index() derives
them the same way scripts/aggregate_grid.py derives its grouping key), which
together cover all 40 mvt cells.

TWO CLASSES OF NUMBER (see src/utils/efficiency.py's module docstring):
params/FLOPs are recomputed and diffed across sessions (the `static` block);
latency/memory are measured-and-session-dependent (the `measured` block) and
are NEVER byte-identical by design -- every measured leaf names its session.

Usage:
    python scripts/efficiency_table.py --check-params-only
    python scripts/efficiency_table.py --device cpu --cpu-threads 1,0 --env-id local_cpu
    python scripts/efficiency_table.py --device cuda --env-id kaggle_t4 --merge
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INDEX_PATH = REPO_ROOT / "configs" / "grid" / "_index.json"
OUT_PATH = REPO_ROOT / "results" / "efficiency_table.json"
LOG_PATH = REPO_ROOT / "results" / "_efficiency_log.jsonl"

PROTOCOL = (
    "Per-(backbone, adapter, head) efficiency for the CLOSED Step-10 MVT "
    "grid. 12 keys derived from configs/grid/_index.json cover all 40 "
    "mvt_results.json leaf cells. 5-way {1,5}-shot, 15 query, 224x224."
)

#: (n_warmup, n_measure) per device x unit class. GPU n_warmup raised above
#: the implementation.txt default (10 -> 50): 10 reps at ~a few ms each does
#: not cover a T4's clock-ramp time. CPU n_warmup raised 5 -> 10 (first reps
#: pay conv-workspace allocation + cold cache).
_TIMING_DEFAULTS = {
    "cuda": {"per_image": (50, 100), "support_1shot": (50, 100),
             "support_5shot": (50, 100), "query_batch": (50, 100),
             "score_stage_1shot": (50, 100), "score_stage_5shot": (50, 100),
             "episode_1shot": (10, 30), "episode_5shot": (10, 30)},
    "cpu": {"per_image": (10, 50), "support_1shot": (10, 50),
            "support_5shot": (10, 50), "query_batch": (3, 10),
            "score_stage_1shot": (10, 50), "score_stage_5shot": (10, 50),
            "episode_1shot": (2, 5), "episode_5shot": (2, 5)},
}

ALL_UNITS = ["per_image", "support_1shot", "support_5shot", "query_batch",
            "score_stage_1shot", "score_stage_5shot",
            "episode_1shot", "episode_5shot"]


def efficiency_key(backbone: str, adapter: str, head: str) -> str:
    return f"{backbone}|{adapter}|{head}"


def _architecture_signature(cfg) -> tuple:
    """Everything a model instance's COMPUTE depends on -- deliberately
    excludes dataset/k_shot/seed, which don't change the built model."""
    return (
        str(cfg["backbone"]["name"]), int(cfg["backbone"]["feature_dim"]),
        json.dumps(dict(cfg["adapter"]), sort_keys=True, default=str),
        json.dumps(dict(cfg["head"]), sort_keys=True, default=str),
        int(cfg["dataset"]["n_way"]), int(cfg["dataset"]["q_query"]),
        int(cfg["dataset"]["image_size"]),
        float(cfg["train"]["lr"]), float(cfg["train"].get("weight_decay", 0.0)),
    )


def keys_from_index(index_path: Path = INDEX_PATH) -> dict:
    """Derive the 12 (backbone, adapter, head) measurement keys from
    configs/grid/_index.json.

    Per key: backbone/adapter/head, `cells` (the matching index rows,
    sorted), `reference_config` (the seed-42 cell, or the lexicographically
    first if 42 is absent), `n_cells`, `mvt_cells` (["dataset/kshot", ...]),
    `architecture_signatures` (must be 1 -- asserted below).

    Raises ValueError if a key's cells do NOT share one architecture
    signature: that sharing is what lets one reference config stand in for
    every cell measurement-wise.
    """
    from src.utils import load_config

    raw_cells = json.loads(Path(index_path).read_text())["cells"]
    grouped: dict[str, list[dict]] = {}
    for c in raw_cells:
        key = efficiency_key(c["backbone"], c["adapter"], c["head"])
        grouped.setdefault(key, []).append(c)

    keys: dict[str, dict] = {}
    for key in sorted(grouped):
        cells = sorted(grouped[key], key=lambda c: (c["dataset"], c["k_shot"], c["seed"]))
        ref_cell = next((c for c in cells if c["seed"] == 42), cells[0])
        mvt_cells = sorted({f"{c['dataset']}/{c['k_shot']}shot" for c in cells})

        signatures = {_architecture_signature(load_config(REPO_ROOT / c["config"]))
                     for c in cells}
        if len(signatures) != 1:
            raise ValueError(
                f"{key}: {len(signatures)} distinct architecture signatures "
                f"across its {len(cells)} cells -- one reference config "
                f"cannot stand for all of them."
            )

        keys[key] = {
            "backbone": cells[0]["backbone"], "adapter": cells[0]["adapter"],
            "head": cells[0]["head"], "cells": cells,
            "reference_config": ref_cell["config"], "n_cells": len(cells),
            "mvt_cells": mvt_cells, "architecture_signatures": len(signatures),
        }
    return keys


def _shapes_for(cfg, k_shot: int) -> dict:
    """episode_input_shapes() with dataset.k_shot overridden to `k_shot`,
    regardless of the loaded cfg's own (fixed) k_shot -- every key is
    measured at BOTH shot regimes since episode-unit cost depends on it."""
    from src.utils.efficiency import episode_input_shapes
    d = dict(cfg["dataset"])
    d["k_shot"] = int(k_shot)
    return episode_input_shapes({"dataset": d})


def _synthetic_episode(shapes: dict, device):
    import torch
    sx = torch.zeros(*shapes["support"], device=device)
    sy = torch.arange(shapes["n_way"], device=device).repeat_interleave(shapes["k_shot"])
    qx = torch.zeros(*shapes["query"], device=device)
    return sx, sy, qx


def _score_from_logits(logits, head, interpretation: str, num_classes: int,
                       prior_per_class: float = 1.0):
    """The head's NATIVE uncertainty score (vacuity for evidential, msp for
    softmax) -- mirrors src/evaluators/episodic.py's `_id_score_set`, minus
    the extra softmax-side scores (energy/ts_msp), since this is a COST
    measurement of "the head + uncertainty stage", not an accuracy eval."""
    import torch
    from src.evaluators.ood import evidence_to_probs_and_vacuity

    if interpretation == "evidential":
        evidence = head.to_evidence(logits)
        _, vacuity = evidence_to_probs_and_vacuity(evidence, num_classes, prior_per_class)
        return 1.0 - vacuity
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def _build_unit_callable(model, cfg, unit: str, device, *, dtype=None):
    """Returns (fn, extra) where fn() runs the unit's forward and `extra`
    carries anything the caller needs post-hoc (e.g. shapes)."""
    import torch

    if unit == "per_image":
        x = torch.zeros(1, 3, int(cfg["dataset"]["image_size"]),
                        int(cfg["dataset"]["image_size"]), device=device)
        return (lambda: model.adapter_features(x)), {"shape": list(x.shape)}

    if unit in ("support_1shot", "support_5shot"):
        k_shot = 1 if unit.endswith("1shot") else 5
        shapes = _shapes_for(cfg, k_shot)
        x = torch.zeros(*shapes["support"], device=device)
        return (lambda: model.adapter_features(x)), {"shape": list(x.shape)}

    if unit == "query_batch":
        shapes = _shapes_for(cfg, 5)  # q_query is shot-independent
        x = torch.zeros(*shapes["query"], device=device)
        return (lambda: model.adapter_features(x)), {"shape": list(x.shape)}

    if unit in ("score_stage_1shot", "score_stage_5shot"):
        k_shot = 1 if unit.endswith("1shot") else 5
        shapes = _shapes_for(cfg, k_shot)
        sx, sy, qx = _synthetic_episode(shapes, device)
        with torch.inference_mode():
            sf = model.adapter_features(sx)
            qf = model.adapter_features(qx)
        interpretation = str(cfg["head"].get("interpretation", "softmax"))
        prior = float(cfg["loss"].get("prior_per_class", 1.0))
        n_way = shapes["n_way"]

        def fn():
            logits = model.head(sf, sy, qf)
            return _score_from_logits(logits, model.head, interpretation, n_way, prior)

        return fn, {"support_shape": list(sf.shape), "query_shape": list(qf.shape)}

    if unit in ("episode_1shot", "episode_5shot"):
        k_shot = 1 if unit.endswith("1shot") else 5
        shapes = _shapes_for(cfg, k_shot)
        sx, sy, qx = _synthetic_episode(shapes, device)
        return (lambda: model.forward_proto(sx, sy, qx)), {
            "support_shape": list(sx.shape), "query_shape": list(qx.shape)}

    raise ValueError(f"Unknown unit: {unit!r}")


def _measure_static(key: str, entry: dict) -> dict:
    """Deterministic block: params + FLOPs, ALWAYS on CPU (so it stays
    bit-identical regardless of which device a session measures latency
    on)."""
    from src.utils import load_config
    from src.models import build_model
    from src.utils.efficiency import params_report, count_flops_detailed

    cfg = load_config(REPO_ROOT / entry["reference_config"])
    model = build_model(cfg)
    model.eval()

    pr = params_report(model)
    per_image_flops = count_flops_detailed(model, input_shape=(
        1, 3, int(cfg["dataset"]["image_size"]), int(cfg["dataset"]["image_size"])))

    result = {
        "axes": {"backbone": entry["backbone"], "adapter": entry["adapter"],
                 "head": entry["head"]},
        "reference_config": entry["reference_config"],
        "n_cells": entry["n_cells"], "mvt_cells": entry["mvt_cells"],
        "weights": "IMAGENET1K_V1",
        "params": pr,
        "flops": {"per_image": per_image_flops},
    }
    del model
    gc.collect()
    return result


def _measure_train_step(key: str, entry: dict, *, device: str) -> dict:
    from src.utils import load_config
    from src.models import build_model
    from src.utils.efficiency import measure_train_step_peak_memory

    cfg = load_config(REPO_ROOT / entry["reference_config"])
    model = build_model(cfg)
    result = measure_train_step_peak_memory(model, cfg, device=device, n_steps=3)
    del model
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return result


def _measure_key_units(key: str, entry: dict, *, device: str,
                       cpu_threads: list[int], units: list[str],
                       env, session, warmup_override=None,
                       measure_override=None) -> dict:
    import torch
    from src.utils import load_config
    from src.models import build_model
    from src.utils.efficiency import (
        time_callable, measure_peak_memory, device_profile_slug,
    )

    cfg = load_config(REPO_ROOT / entry["reference_config"])
    model = build_model(cfg)
    model.eval()
    device_t = torch.device(device)
    model.to(device_t)

    out: dict[str, dict] = {}
    thread_settings = cpu_threads if device == "cpu" else [None]
    for unit in units:
        n_warmup, n_measure = _TIMING_DEFAULTS[device][unit]
        if warmup_override is not None:
            n_warmup = warmup_override
        if measure_override is not None:
            n_measure = measure_override
        out[unit] = {}
        for threads in thread_settings:
            num_threads = None if (threads in (None, 0)) else int(threads)
            fn, extra = _build_unit_callable(model, cfg, unit, device_t)
            if device == "cpu":
                from src.utils.efficiency import thread_count
                with thread_count(num_threads):
                    timing = time_callable(fn, device=device_t, n_warmup=n_warmup,
                                           n_measure=n_measure, grad_mode="inference")
            else:
                timing = time_callable(fn, device=device_t, n_warmup=n_warmup,
                                       n_measure=n_measure, grad_mode="inference")
            profile = device_profile_slug(env, num_threads=num_threads)
            mean_ms = timing["latency_ms"]["mean"]
            batch = extra.get("shape", extra.get("query_shape", [1]))[0]
            timing["throughput_img_per_s"] = (1000.0 * batch / mean_ms) if mean_ms > 0 else None
            timing["session"] = session
            timing["input"] = extra
            out[unit][profile] = timing

    memory = {}
    if device == "cuda":
        fn, extra = _build_unit_callable(model, cfg, "per_image", device_t)
        mem = measure_peak_memory(model, input_shape=tuple(extra["shape"]))
        mem["session"] = session
        memory["per_image"] = mem
    else:
        memory["per_image"] = {"status": "skipped",
                               "reason": "no CUDA device on this session"}

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"timing": out, "memory": memory}


def _reference_gate() -> dict:
    """Validation gate against published MAC figures, run BEFORE any other
    measurement (a gate whose failure is silently tolerated is not a gate).
    Uses bare torchvision backbones with weights=None -- offline, and
    architecture-only, so identical regardless of pretrained weights."""
    import torch.nn as nn
    from torchvision.models import resnet18, mobilenet_v3_small
    from src.utils.efficiency import count_flops, check_reference_flops

    r18 = resnet18(weights=None)
    r18.eval()
    r18_macs = count_flops(r18, forward="forward")
    r18_backbone = resnet18(weights=None)
    r18_backbone.fc = nn.Identity()
    r18_backbone.eval()
    r18_backbone_macs = count_flops(r18_backbone)
    r18_result = check_reference_flops("resnet18", r18_macs, r18_backbone_macs)

    mb = mobilenet_v3_small(weights=None)
    mb.eval()
    mb_macs = count_flops(mb, forward="forward")
    mb_backbone = mobilenet_v3_small(weights=None)
    mb_backbone.classifier = nn.Identity()
    mb_backbone.eval()
    mb_backbone_macs = count_flops(mb_backbone)
    mb_result = check_reference_flops("mobilenetv3_small", mb_macs, mb_backbone_macs)

    return {"resnet18": r18_result, "mobilenetv3_small": mb_result}


def _reference_backbones(*, device: str) -> dict:
    """ADDITION: architecture-only reference rows (never trained in this
    repo) for ViT-B/16 (torchvision, no new dependency) and DeiT-Tiny (timm,
    gated -- a missing `timm` must never break the run). Every row is
    labelled so it can never be mistaken for a thesis result."""
    import torch
    from src.utils.efficiency import (
        params_report, count_flops, time_callable, device_profile_slug,
    )

    out: dict = {}
    device_t = torch.device(device)

    from torchvision.models import vit_b_16
    vit = vit_b_16(weights=None)
    vit.eval().to(device_t)
    x = torch.zeros(1, 3, 224, 224, device=device_t)
    pr = params_report(vit)
    macs = count_flops(vit, forward="forward")
    timing = time_callable(lambda: vit(x), device=device_t, n_warmup=2, n_measure=5,
                           grad_mode="inference")
    out["vit_b_16"] = {
        "source": "torchvision.models.vit_b_16, weights=None",
        "disclaimer": "architecture-only, never trained here, not a thesis result",
        "params": pr, "flops": {"macs": macs},
        "latency_ms": timing["latency_ms"],
    }

    try:
        import timm
        deit = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        deit.eval().to(device_t)
        pr2 = params_report(deit)
        macs2 = count_flops(deit, forward="forward")
        timing2 = time_callable(lambda: deit(x), device=device_t, n_warmup=2,
                                n_measure=5, grad_mode="inference")
        out["deit_tiny_patch16_224"] = {
            "source": "timm.create_model('deit_tiny_patch16_224', pretrained=False)",
            "disclaimer": "architecture-only, never trained here, not a thesis result",
            "params": pr2, "flops": {"macs": macs2},
            "latency_ms": timing2["latency_ms"],
        }
    except ImportError:
        out["deit_tiny_patch16_224"] = {
            "status": "unavailable",
            "reason": "timm not installed; not a project dependency, gated behind --include-reference-backbones",
        }
    return out


def _deep_merge(base: dict, add: dict) -> dict:
    out = dict(base)
    for k, v in add.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu", "both"], default=None)
    ap.add_argument("--cpu-threads", default="1,0",
                    help="comma list; 0 means 'all threads' (no restriction)")
    ap.add_argument("--only", default=None,
                    help="filter, e.g. 'backbone=resnet18,adapter=lora'")
    ap.add_argument("--units", default=",".join(ALL_UNITS))
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--measure", type=int, default=None)
    ap.add_argument("--train-step", dest="train_step", action="store_true", default=True)
    ap.add_argument("--no-train-step", dest="train_step", action="store_false")
    ap.add_argument("--include-reference-backbones", action="store_true")
    ap.add_argument("--check-params-only", action="store_true",
                    help="rebuild all 12 models, assert trainable params match "
                        "mvt_results.json's n_params for all 40 cells; no timing.")
    ap.add_argument("--allow-reference-mismatch", action="store_true")
    ap.add_argument("--env-id", default=None,
                    help="label for this session's environment block, e.g. "
                        "'local_cpu' / 'kaggle_t4'. Defaults to the computed session id.")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    from src.utils import set_seed, load_config
    from src.models import build_model
    from src.utils.efficiency import collect_env, session_id, gpu_clock_snapshot

    set_seed(42, deterministic=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    devices = ["cuda", "cpu"] if device == "both" else [device]
    cpu_threads = [int(t) for t in args.cpu_threads.split(",") if t.strip() != ""]
    units = [u.strip() for u in args.units.split(",") if u.strip()]

    keys = keys_from_index()
    print(f"[efficiency] {len(keys)} keys derived from {INDEX_PATH.relative_to(REPO_ROOT)}")

    only = {}
    if args.only:
        for pair in args.only.split(","):
            k, _, v = pair.partition("=")
            only[k.strip()] = v.strip()
    selected = {k: e for k, e in keys.items()
               if all(e[axis] == val for axis, val in only.items())}
    print(f"[efficiency] {len(selected)} keys selected"
         f"{' (filtered by ' + args.only + ')' if only else ''}")

    if args.check_params_only:
        mvt_path = REPO_ROOT / "results" / "mvt_results.json"
        mvt = json.loads(mvt_path.read_text())["results"] if mvt_path.exists() else {}
        n_checked = n_match = 0
        for key, entry in selected.items():
            cfg = load_config(REPO_ROOT / entry["reference_config"])
            model = build_model(cfg)
            from src.utils import count_trainable_params
            trainable = count_trainable_params(model)
            # Dedupe to the unique (dataset, kshot) mvt cells this key covers
            # -- entry["cells"] has one row per SEED (up to 3x redundant),
            # but n_params is seed-invariant, so checking each aggregated
            # mvt cell once is the right count (40 total across all keys).
            for ds_kshot in entry["mvt_cells"]:
                dataset, kshot = ds_kshot.split("/")
                node = (mvt.get(dataset, {}).get(kshot, {})
                       .get(entry["backbone"], {}).get(entry["adapter"], {})
                       .get(entry["head"]))
                if node is None:
                    continue
                n_checked += 1
                if int(node["n_params"]["mean"]) == trainable:
                    n_match += 1
                else:
                    print(f"[efficiency]   MISMATCH {key} / {ds_kshot}: "
                         f"measured={trainable} mvt={node['n_params']['mean']}")
        print(f"[efficiency] param cross-check: {n_match}/{n_checked} mvt cells match")
        if n_checked and n_match != n_checked:
            raise SystemExit(1)
        return

    if args.dry_run:
        for key, entry in selected.items():
            print(f"  {key} -> {entry['reference_config']} "
                 f"({entry['n_cells']} cells, mvt: {entry['mvt_cells']})")
        return

    gate = _reference_gate()
    print(f"[efficiency] reference gate: resnet18={gate['resnet18']['status']} "
         f"mobilenetv3_small={gate['mobilenetv3_small']['status']}")
    if any(g["status"] == "fail" for g in gate.values()) and not args.allow_reference_mismatch:
        raise SystemExit(
            "[efficiency] reference gate FAILED -- measured MACs do not match "
            "the published reference within tolerance. Re-run with "
            "--allow-reference-mismatch to proceed anyway (not recommended)."
        )

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    static_block = dict(existing.get("static", {}))
    static_mismatches = []
    for key, entry in selected.items():
        print(f"[efficiency] static: {key}")
        fresh = _measure_static(key, entry)
        old = static_block.get(key)
        if old is not None:
            old_params, new_params = old["params"]["trainable"], fresh["params"]["trainable"]
            old_macs = old["flops"]["per_image"]["macs"]
            new_macs = fresh["flops"]["per_image"]["macs"]
            if old_params != new_params or old_macs != new_macs:
                static_mismatches.append({
                    "key": key, "old_trainable": old_params, "new_trainable": new_params,
                    "old_macs": old_macs, "new_macs": new_macs,
                })
                print(f"[efficiency]   STATIC MISMATCH {key}: trainable "
                     f"{old_params} -> {new_params}, macs {old_macs} -> {new_macs}")
        static_block[key] = fresh
    if static_mismatches and not args.allow_reference_mismatch:
        raise SystemExit(
            f"[efficiency] {len(static_mismatches)} key(s) have a DIFFERENT "
            "static (params/MACs) value than the committed file -- params "
            "and FLOPs are supposed to be byte-identical across sessions "
            "(src/utils/efficiency.py's determinism claim). This means "
            "either the code changed or an environment differs in a way "
            "that affects the architecture. Re-run with "
            "--allow-reference-mismatch to overwrite anyway."
        )

    result: dict = {
        "schema_version": "step11-efficiency-v1",
        "flop_convention": "macs",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "reference_gate": gate,
        "reproducibility": {
            "static_is_byte_identical": True,
            "measured_is_byte_identical": False,
            "measured_exemption_reason": (
                "Wall-clock latency and CUDA peak memory depend on host load, "
                "GPU clocks/thermals, driver version, cuDNN kernel selection "
                "and allocator history. The repo's byte-identical-rerun "
                "invariant is asserted on the `static` block only."
            ),
            "cross_session_static_check": {
                "status": "pass" if not static_mismatches else "fail",
                "keys_compared": sum(1 for k in selected if k in existing.get("static", {})),
                "mismatches": static_mismatches,
            },
        },
        "static": static_block,
        "measured": dict(existing.get("measured", {})),
        "training_step": dict(existing.get("training_step", {})),
        "environments": dict(existing.get("environments", {})),
        "collapse_check": {},
        "key_arithmetic": {
            "report_rows": len(keys),
            "note": (
                "implementation.txt 11.2 estimates '8 of the 12 combos "
                "collapse' for the efficiency loop. Measured against the "
                "closed grid: 12 keys derived; both heads of every "
                "(backbone, adapter) pair ARE measured (not collapsed) so "
                "the assumption is checked (see collapse_check), not assumed."
            ),
        },
    }

    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            print("[efficiency] --device cuda/both requested but CUDA is "
                 "unavailable on this host; skipping the CUDA measurements.")
            continue
        env = collect_env(device=dev)
        base_id = args.env_id or session_id(env)
        # --device both measures cuda AND cpu in the same invocation (the
        # notebook's call pattern): suffix by device so the second pass
        # doesn't overwrite the first session's environment block under the
        # same id (a real bug caught while implementing this step -- the GPU
        # env was silently clobbered by the CPU env written right after it).
        sid = base_id if len(devices) == 1 else f"{base_id}_{dev}"
        env["gpu_clocks"] = gpu_clock_snapshot() if dev == "cuda" else {
            "status": "n/a", "reason": "cpu session"}
        env["cpu_threads_measured"] = cpu_threads if dev == "cpu" else None
        result["environments"][sid] = env

        for key, entry in selected.items():
            print(f"[efficiency] measured [{dev}]: {key}")
            units_out = _measure_key_units(
                key, entry, device=dev, cpu_threads=cpu_threads, units=units,
                env=env, session=sid, warmup_override=args.warmup,
                measure_override=args.measure,
            )
            result["measured"].setdefault(key, {})
            for unit, by_profile in units_out["timing"].items():
                result["measured"][key].setdefault(unit, {})
                for profile, timing in by_profile.items():
                    result["measured"][key][unit][profile] = timing
            if dev == "cuda":
                result["measured"][key].setdefault("memory", {})
                result["measured"][key]["memory"]["per_image"] = units_out["memory"]["per_image"]

            if args.train_step:
                print(f"[efficiency] train_step [{dev}]: {key}")
                result["training_step"].setdefault(key, {})
                result["training_step"][key][dev] = _measure_train_step(key, entry, device=dev)

        # Drift check: re-measure the first key's per_image latency at the
        # end of the session; Kaggle/Colab give no clock locking on a shared
        # host, so this measures session drift instead of assuming it away.
        first_key = next(iter(selected), None)
        if first_key is not None:
            recheck = _measure_key_units(
                first_key, selected[first_key], device=dev, cpu_threads=cpu_threads[:1],
                units=["per_image"], env=env, session=sid,
            )
            profile = next(iter(recheck["timing"]["per_image"]))
            first_median = result["measured"][first_key]["per_image"][profile]["latency_ms"]["median"]
            last_median = recheck["timing"]["per_image"][profile]["latency_ms"]["median"]
            drift_pct = (100.0 * abs(last_median - first_median) / first_median
                        if first_median else 0.0)
            env["drift_check"] = {
                "unit": "per_image", "key": first_key,
                "median_ms_first": first_median, "median_ms_last": last_median,
                "drift_pct": drift_pct, "threshold_pct": 5.0,
                "status": "pass" if drift_pct <= 5.0 else "warn",
                "purpose": ("Kaggle/Colab give no clock locking or persistence "
                           "mode on a shared host; this measures session "
                           "drift instead of assuming it away."),
            }

    # Collapse check: per (backbone, adapter), compare the two heads'
    # per_image median -- turns implementation.txt 11.2's "heads collapse"
    # assumption into a measurement with a threshold, rather than assuming it.
    by_compute_key: dict[str, dict] = {}
    for key, entry in selected.items():
        compute_key = f"{entry['backbone']}|{entry['adapter']}"
        by_compute_key.setdefault(compute_key, {})[entry["head"]] = key
    for compute_key, heads in by_compute_key.items():
        if "evidential" not in heads or "softmax" not in heads:
            continue
        ev_key, sm_key = heads["evidential"], heads["softmax"]
        ev_units = result["measured"].get(ev_key, {}).get("per_image", {})
        sm_units = result["measured"].get(sm_key, {}).get("per_image", {})
        for profile in set(ev_units) & set(sm_units):
            ev_med = ev_units[profile]["latency_ms"]["median"]
            sm_med = sm_units[profile]["latency_ms"]["median"]
            rel_diff = 100.0 * abs(ev_med - sm_med) / sm_med if sm_med else 0.0
            result["collapse_check"][f"{compute_key}@{profile}"] = {
                "evidential_median_ms": ev_med, "softmax_median_ms": sm_med,
                "rel_diff_pct": rel_diff,
                "status": "pass" if rel_diff <= 5.0 else "warn",
            }

    if args.include_reference_backbones:
        ref_device = "cuda" if ("cuda" in devices and torch.cuda.is_available()) else "cpu"
        result["reference_backbones"] = _reference_backbones(device=ref_device)

    result["limitations"] = [
        "No Jetson Nano or any ARM edge device was available; CPU latency is "
        "used as the edge proxy (implementation.txt Step 11 exit criterion).",
        "The CPU proxy is a server/laptop core, not an edge SoC. Absolute "
        "milliseconds are not transferable to edge hardware; ratios between "
        "backbones/adapters are the portable signal.",
        "PyTorch eager, fp32, no quantization, no TFLite/TensorRT.",
        "MAC counts exclude BatchNorm, activations, pooling and residual "
        "adds (no MAC counter includes them) -- see uncounted_ops per key.",
    ]

    # Merge onto any existing file (multi-session workflow: CPU measured
    # locally, GPU measured on Kaggle, both land in one file).
    merged = _deep_merge(existing, result) if existing else result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str))
    try:
        printed_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        printed_path = out_path
    print(f"[efficiency] wrote {printed_path}")


if __name__ == "__main__":
    main()
