"""Episodic evaluator for the Step 4 prototype-head meta-trained model.

Step 4.5 (W3) generalises this from a single OOD pool / single score to a
SCORE x OOD-POOL MATRIX so the decisive near-OOD comparison is fair:

  - multiple OOD pools (e.g. "svhn_far", "cifar100_near", "tin_near") passed
    as pre-computed backbone features;
  - multiple ID-ness scores per head:
      evidential -> {"vacuity"}                 (1 - u = 1 - K/S)
      softmax    -> {"msp", "energy"[, "ts_msp"]}
    where "ts_msp" is added when a temperature T (fit on the frozen VAL
    episodes, Guo et al.) is provided, and "energy" is Liu et al.'s logsumexp.

For each test episode the query set is prototype-classified against THIS
episode's support prototypes, and every OOD pool is scored against the SAME
prototypes (standard open-set few-shot protocol). AUROC + FPR@95 are
accumulated per (pool, score).

The returned `summary` keeps ALL Step-3/Step-4 keys (populated from a PRIMARY
pool + the head's native score, for back-compat) and ADDS:
  ood_auroc__{pool}__{score}, fpr_at_95_tpr__{pool}__{score}   (per cell)
  ece_ts, brier_ts                                             (softmax + T only)
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from .accuracy import accuracy, f1_macro
from .calibration import expected_calibration_error, brier_score
from .ood import (
    ood_auroc,
    fpr_at_95_tpr,
    evidence_to_probs_and_vacuity,
    energy_score,
)
from .temperature import apply_temperature


def _proto_logits_for_query(model, support_feats: torch.Tensor,
                            support_y: torch.Tensor,
                            query_feats: torch.Tensor) -> torch.Tensor:
    """Prototype-similarity logits for `query_feats` using THIS episode's
    support set. Both feature tensors are backbone outputs (B, D); the model
    applies the adapter internally. Returns (Q, n_way) logits."""
    return model.forward_proto_from_features(support_feats, support_y, query_feats)


def _id_score_set(logits: torch.Tensor, interpretation: str, head,
                  num_classes: int, temperature: Optional[float],
                  prior_per_class: float) -> Dict[str, torch.Tensor]:
    """Map prototype logits to a dict {score_name: (B,) id-ness score}.
    Higher score => looks more in-distribution. `head.to_evidence` supplies
    the SAME logits->evidence map used at train time (no train/test skew)."""
    if interpretation == "evidential":
        evidence = head.to_evidence(logits)
        _, vacuity = evidence_to_probs_and_vacuity(
            evidence, num_classes, prior_per_class,
        )
        return {"vacuity": 1.0 - vacuity}
    if interpretation == "softmax":
        probs = torch.softmax(logits, dim=-1)
        scores = {
            "msp": probs.max(dim=-1).values,
            "energy": energy_score(logits),
        }
        if temperature is not None:
            scores["ts_msp"] = apply_temperature(logits, temperature).max(dim=-1).values
        return scores
    raise ValueError(f"Unknown interpretation: {interpretation!r}")


def _logits_to_probs(logits: torch.Tensor, num_classes: int,
                     interpretation: str, head,
                     prior_per_class: float) -> torch.Tensor:
    """Probability vector per sample (for ECE / Brier / Macro-F1)."""
    if interpretation == "evidential":
        evidence = head.to_evidence(logits)
        probs, _vac = evidence_to_probs_and_vacuity(
            evidence, num_classes, prior_per_class,
        )
        return probs
    if interpretation == "softmax":
        return torch.softmax(logits, dim=-1)
    raise ValueError(f"Unknown interpretation: {interpretation!r}")


def _native_score(interpretation: str) -> str:
    """The head's own OOD score, used to populate the legacy single-score keys."""
    return "vacuity" if interpretation == "evidential" else "msp"


def evaluate_episodic(
    model,
    test_iterable: Iterable[Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]],
    ood_pools: Optional[Dict[str, torch.Tensor]],
    *,
    num_classes: int,
    interpretation: str,            # "softmax" | "evidential"
    ece_bins: int = 15,
    temperature: Optional[float] = None,
    prior_per_class: float = 1.0,
    device: torch.device | str = "cpu",
    logger=None,
    wandb_run=None,
) -> dict:
    """Run prototype-head evaluation over a stream of test episodes.

    Args:
        ood_pools: {name: (N, D) backbone features}. Pass None/empty to skip OOD.
        interpretation: "softmax" or "evidential".
        temperature: optional T (softmax only) enabling the "ts_msp" score and
            the ece_ts / brier_ts calibration keys.
        prior_per_class: R-EDL Dirichlet prior mass per class; MUST match the
            training loss so train/test Dirichlets agree (default 1.0 = "+1").

    Returns dict {summary, pooled_probs, pooled_targets, last_id_scores,
    last_ood_scores}.
    """
    model = model.to(device)
    model.eval()
    backbone = model.backbone
    head = model.head

    ood_pools = ood_pools or {}
    ood_pools = {name: feats.to(device) for name, feats in ood_pools.items()}
    pool_names = list(ood_pools.keys())
    primary_pool = pool_names[0] if pool_names else None
    native = _native_score(interpretation)

    per_ep_acc, per_ep_f1, per_ep_ece, per_ep_brier = [], [], [], []
    pooled_probs, pooled_targets, pooled_logits = [], [], []
    # auroc[pool][score] -> list over episodes
    auroc_acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    fpr_acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    last_id_scores: np.ndarray | None = None
    last_ood_scores: np.ndarray | None = None

    with torch.no_grad():
        for i, (sx, sy, qx, qy) in enumerate(test_iterable):
            sx = sx.to(device); sy = sy.to(device)
            qx = qx.to(device); qy = qy.to(device)

            sx_feats = backbone(sx)
            qx_feats = backbone(qx)
            q_logits = _proto_logits_for_query(model, sx_feats, sy, qx_feats)

            probs = _logits_to_probs(q_logits, num_classes, interpretation,
                                     head, prior_per_class)
            per_ep_acc.append(accuracy(probs, qy))
            per_ep_f1.append(f1_macro(probs, qy, num_classes=num_classes))
            per_ep_ece.append(expected_calibration_error(probs, qy, num_bins=ece_bins))
            per_ep_brier.append(brier_score(probs, qy, num_classes))
            pooled_probs.append(probs.cpu())
            pooled_targets.append(qy.cpu())
            pooled_logits.append(q_logits.cpu())

            if pool_names:
                id_scores = _id_score_set(q_logits, interpretation, head,
                                          num_classes, temperature, prior_per_class)
                for pname in pool_names:
                    ood_logits = _proto_logits_for_query(
                        model, sx_feats, sy, ood_pools[pname])
                    ood_scores = _id_score_set(ood_logits, interpretation, head,
                                               num_classes, temperature, prior_per_class)
                    for sname, id_s in id_scores.items():
                        id_np = id_s.cpu().numpy()
                        ood_np = ood_scores[sname].cpu().numpy()
                        auroc_acc[pname][sname].append(ood_auroc(id_np, ood_np))
                        fpr_acc[pname][sname].append(fpr_at_95_tpr(id_np, ood_np))
                        if pname == primary_pool and sname == native:
                            last_id_scores, last_ood_scores = id_np, ood_np

            if logger is not None:
                prim = (auroc_acc[primary_pool][native][-1]
                        if primary_pool else float("nan"))
                logger.info(
                    f"ep {i:3d}  acc={per_ep_acc[-1]:.3f}  F1={per_ep_f1[-1]:.3f}  "
                    f"ECE={per_ep_ece[-1]:.3f}  Brier={per_ep_brier[-1]:.3f}  "
                    f"AUROC[{primary_pool}/{native}]={prim:.3f}"
                )
            if wandb_run is not None:
                wandb_run.log({
                    "eval/episode": i,
                    "eval/accuracy": float(per_ep_acc[-1]),
                    "eval/ECE": float(per_ep_ece[-1]),
                    "eval/accuracy_running_mean": float(np.mean(per_ep_acc)),
                })

    n = len(per_ep_acc)
    if n == 0:
        raise RuntimeError("evaluate_episodic: test_iterable was empty")

    pooled_probs_t = torch.cat(pooled_probs, dim=0)
    pooled_targets_t = torch.cat(pooled_targets, dim=0)
    pooled_logits_t = torch.cat(pooled_logits, dim=0)
    pooled_ece = expected_calibration_error(
        pooled_probs_t, pooled_targets_t, num_bins=ece_bins)

    def _ci95(arr):
        return float(1.96 * np.std(arr) / np.sqrt(len(arr))) if len(arr) else 0.0

    summary = {
        "accuracy_mean": float(np.mean(per_ep_acc)),
        "accuracy_std":  float(np.std(per_ep_acc)),
        "accuracy_ci95": _ci95(per_ep_acc),
        "f1_macro_mean": float(np.mean(per_ep_f1)),
        "f1_macro_std":  float(np.std(per_ep_f1)),
        "f1_macro_ci95": _ci95(per_ep_f1),
        "ece_per_episode_mean": float(np.mean(per_ep_ece)),
        "ece_per_episode_std":  float(np.std(per_ep_ece)),
        "ece_pooled":           float(pooled_ece),
        "brier_mean":           float(np.mean(per_ep_brier)),
        "brier_std":            float(np.std(per_ep_brier)),
        "num_episodes":         int(n),
        "interpretation":       str(interpretation),
    }

    # Per-cell matrix keys.
    for pname in pool_names:
        for sname, vals in auroc_acc[pname].items():
            summary[f"ood_auroc__{pname}__{sname}"] = float(np.mean(vals))
            summary[f"ood_auroc_std__{pname}__{sname}"] = float(np.std(vals))
            summary[f"fpr_at_95_tpr__{pname}__{sname}"] = float(np.mean(fpr_acc[pname][sname]))

    # Legacy single-score keys (primary pool + native score).
    if primary_pool is not None:
        summary["ood_auroc_mean"] = float(np.mean(auroc_acc[primary_pool][native]))
        summary["ood_auroc_std"]  = float(np.std(auroc_acc[primary_pool][native]))
        summary["fpr_at_95_tpr_mean"] = float(np.mean(fpr_acc[primary_pool][native]))
        summary["fpr_at_95_tpr_std"]  = float(np.std(fpr_acc[primary_pool][native]))
        summary["primary_ood_pool"] = str(primary_pool)
    else:
        summary["ood_auroc_mean"] = 0.0
        summary["ood_auroc_std"]  = 0.0
        summary["fpr_at_95_tpr_mean"] = 1.0
        summary["fpr_at_95_tpr_std"]  = 0.0

    # Temperature-scaled calibration (softmax only).
    if temperature is not None and interpretation == "softmax":
        ts_probs = apply_temperature(pooled_logits_t, temperature)
        summary["ece_ts"] = float(expected_calibration_error(
            ts_probs, pooled_targets_t, num_bins=ece_bins))
        summary["brier_ts"] = float(brier_score(ts_probs, pooled_targets_t, num_classes))

    return {
        "summary": summary,
        "pooled_probs": pooled_probs_t,
        "pooled_targets": pooled_targets_t,
        "last_id_scores": last_id_scores,
        "last_ood_scores": last_ood_scores,
    }
