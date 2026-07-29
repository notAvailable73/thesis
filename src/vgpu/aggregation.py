"""Merge deterministic 50-episode vGPU shards into the frozen metrics schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def vgpu_validate_shards(
    shards: Iterable[dict[str, Any]],
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    ordered = sorted(shards, key=lambda shard: int(shard["seeds"][0]))
    observed: list[int] = []
    for shard in ordered:
        seeds = [int(seed) for seed in shard.get("seeds", [])]
        if not seeds:
            raise ValueError("evaluation shard has no seeds")
        if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            raise ValueError(f"non-contiguous evaluation shard: {seeds[:3]}...")
        observed.extend(seeds)
    if observed != expected_seeds:
        missing = sorted(set(expected_seeds) - set(observed))
        duplicate_count = len(observed) - len(set(observed))
        raise ValueError(
            "evaluation shards do not match the frozen seed list: "
            f"missing={missing[:10]}, duplicates={duplicate_count}"
        )
    return ordered


def vgpu_merge_evaluation_shards(
    shards: Iterable[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Return the same summary keys as ``scripts/evaluate.py``.

    Heavy numerical imports are intentionally lazy so manifest/upload tests can
    run on a controller machine before the ML environment is installed.
    """

    import numpy as np
    import torch

    from src.evaluators.calibration import expected_calibration_error, brier_score
    from src.evaluators.temperature import apply_temperature

    seeds = expected_seeds if expected_seeds is not None else list(range(600))
    ordered = vgpu_validate_shards(shards, seeds)
    per_names = ("accuracy", "f1_macro", "ece", "brier")
    per = {
        name: [
            float(value)
            for shard in ordered
            for value in shard["per_episode"][name]
        ]
        for name in per_names
    }
    probs = torch.tensor(
        [row for shard in ordered for row in shard["pooled_probs"]],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [row for shard in ordered for row in shard["pooled_logits"]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [value for shard in ordered for value in shard["pooled_targets"]],
        dtype=torch.long,
    )
    num_classes = int(metadata.get("num_classes", probs.shape[-1]))
    ece_bins = int(metadata.get("ece_bins", 15))

    def mean(values):
        return float(np.mean(values))

    def std(values):
        return float(np.std(values))

    def ci95(values):
        return float(1.96 * np.std(values) / np.sqrt(len(values)))

    summary: dict[str, Any] = {
        "accuracy_mean": mean(per["accuracy"]),
        "accuracy_std": std(per["accuracy"]),
        "accuracy_ci95": ci95(per["accuracy"]),
        "f1_macro_mean": mean(per["f1_macro"]),
        "f1_macro_std": std(per["f1_macro"]),
        "f1_macro_ci95": ci95(per["f1_macro"]),
        "ece_per_episode_mean": mean(per["ece"]),
        "ece_per_episode_std": std(per["ece"]),
        "ece_pooled": float(expected_calibration_error(probs, targets, num_bins=ece_bins)),
        "brier_mean": mean(per["brier"]),
        "brier_std": std(per["brier"]),
        "num_episodes": len(seeds),
    }

    ood_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for shard in ordered:
        for pool_name, score_map in shard.get("ood", {}).items():
            for score_name, values in score_map.items():
                cell = ood_values.setdefault(pool_name, {}).setdefault(
                    score_name, {"auroc": [], "fpr": []}
                )
                cell["auroc"].extend(float(v) for v in values["auroc"])
                cell["fpr"].extend(float(v) for v in values["fpr"])
    for pool_name, score_map in ood_values.items():
        for score_name, values in score_map.items():
            summary[f"ood_auroc__{pool_name}__{score_name}"] = mean(values["auroc"])
            summary[f"ood_auroc_std__{pool_name}__{score_name}"] = std(values["auroc"])
            summary[f"fpr_at_95_tpr__{pool_name}__{score_name}"] = mean(values["fpr"])

    interpretation = str(metadata["interpretation"])
    native = "vacuity" if interpretation == "evidential" else "msp"
    primary = str(metadata.get("primary_ood_pool", "svhn_far"))
    primary_cell = ood_values.get(primary, {}).get(native)
    if primary_cell:
        summary.update({
            "ood_auroc_mean": mean(primary_cell["auroc"]),
            "ood_auroc_std": std(primary_cell["auroc"]),
            "fpr_at_95_tpr_mean": mean(primary_cell["fpr"]),
            "fpr_at_95_tpr_std": std(primary_cell["fpr"]),
            "primary_ood_pool": primary,
        })
    else:
        summary.update({
            "ood_auroc_mean": 0.0,
            "ood_auroc_std": 0.0,
            "fpr_at_95_tpr_mean": 1.0,
            "fpr_at_95_tpr_std": 0.0,
        })

    temperature = float(metadata.get("temperature", 0.0))
    if interpretation == "softmax" and temperature > 0:
        ts_probs = apply_temperature(logits, temperature)
        summary["ece_ts"] = float(
            expected_calibration_error(ts_probs, targets, num_bins=ece_bins)
        )
        summary["brier_ts"] = float(
            brier_score(ts_probs, targets, num_classes)
        )

    for key in (
        "adapter_type", "config_path", "episodes_file", "head_type",
        "interpretation", "n_params", "seed", "trainer_type",
        "best_val_epoch", "temperature", "prior_per_class",
    ):
        if key in metadata:
            summary[key] = metadata[key]
    summary["seeds_first10"] = seeds[:10]
    summary["seeds_last10"] = seeds[-10:]
    return {
        "summary": summary,
        "pooled_probs": probs,
        "pooled_logits": logits,
        "pooled_targets": targets,
        "last_id_scores": ordered[-1].get("last_id_scores"),
        "last_ood_scores": ordered[-1].get("last_ood_scores"),
    }


def vgpu_write_metrics(path: str | Path, summary: dict[str, Any]) -> None:
    from .state import vgpu_atomic_write_json
    vgpu_atomic_write_json(Path(path), summary)
