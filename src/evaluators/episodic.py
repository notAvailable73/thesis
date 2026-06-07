"""Episodic evaluator for the Step 4 prototype-head meta-trained model.

Step 1-3's evaluator (`evaluate_finetune` -- the per-episode inner-loop
fine-tuner) lives inline in scripts/evaluate.py. This module hosts the
new path:

    evaluate_episodic(model, test_iterable, ood_loader, head_type,
                      ece_bins, num_classes, interpretation, device)

For each test episode:
  1. Forward support+query through (frozen backbone + meta-trained
     adapter + parameter-free prototype head). No gradients, no inner
     fitting — the adapter is FROZEN at test time.
  2. Compute per-episode accuracy, Macro-F1, ECE, Brier.
  3. Compute per-episode OOD AUROC + FPR@95 against the SAME episode's
     prototypes — OOD images are scored against THIS episode's support
     prototypes, so the OOD score is "how in-distribution does this
     sample look relative to the 5 support classes drawn for this
     episode?". This is the standard open-set few-shot protocol.

Returns a dict in the Step 3 schema PLUS:
  f1_macro_mean, f1_macro_std, f1_macro_ci95
  fpr_at_95_tpr_mean, fpr_at_95_tpr_std
  best_val_epoch     (set by the caller -- the trainer wrote this into
                      the checkpoint)
"""
from __future__ import annotations
from typing import Iterable, Tuple, Optional

import numpy as np
import torch

from .accuracy import accuracy, f1_macro
from .calibration import expected_calibration_error, brier_score
from .ood import (
    ood_auroc,
    fpr_at_95_tpr,
    evidence_to_probs_and_vacuity,
    logits_to_probs_and_uncertainty,
)


def _proto_logits_for_query(model, support_feats: torch.Tensor,
                            support_y: torch.Tensor,
                            query_feats: torch.Tensor) -> torch.Tensor:
    """Compute prototype-similarity logits for `query_feats` using THIS
    episode's support set. Both feature tensors must already be the
    backbone outputs (B, D) on the same device.

    Returns (Q, n_way) logits.
    """
    return model.forward_proto_from_features(
        support_feats, support_y, query_feats,
    )


def _logits_to_id_score(logits: torch.Tensor,
                        num_classes: int,
                        interpretation: str,
                        head) -> torch.Tensor:
    """Map prototype-similarity logits to an in-distribution score per
    sample. Higher score => looks more in-distribution.

    - "evidential": head.to_evidence -> Dirichlet -> 1 - vacuity
    - "softmax":    softmax  -> max_p

    `head` supplies the SAME logits->evidence mapping used at train time
    (PrototypeHead.to_evidence); evaluating with a different mapping than
    training was the latent train/test-skew risk this signature removes.
    """
    if interpretation == "evidential":
        evidence = head.to_evidence(logits)
        _, vacuity = evidence_to_probs_and_vacuity(evidence, num_classes)
        return 1.0 - vacuity
    if interpretation == "softmax":
        probs, _unc = logits_to_probs_and_uncertainty(logits)
        return probs.max(dim=-1).values
    raise ValueError(f"Unknown interpretation: {interpretation!r}")


def _logits_to_probs(logits: torch.Tensor, num_classes: int,
                     interpretation: str, head) -> torch.Tensor:
    """Map prototype-similarity logits to a probability vector per
    sample (used for ECE / Brier / Macro-F1)."""
    if interpretation == "evidential":
        evidence = head.to_evidence(logits)
        probs, _vac = evidence_to_probs_and_vacuity(evidence, num_classes)
        return probs
    if interpretation == "softmax":
        return torch.softmax(logits, dim=-1)
    raise ValueError(f"Unknown interpretation: {interpretation!r}")


def evaluate_episodic(
    model,
    test_iterable: Iterable[Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]],
    ood_features: Optional[torch.Tensor],
    *,
    num_classes: int,
    interpretation: str,            # "softmax" | "evidential"
    ece_bins: int = 15,
    device: torch.device | str = "cpu",
    logger=None,
    wandb_run=None,
) -> dict:
    """Run prototype-head evaluation over a stream of test episodes.

    Args:
        model:           BPEFTModel with a PrototypeHead. Adapter weights
                         are FROZEN here; .eval() is applied internally.
        test_iterable:   yields (support_x, support_y, query_x, query_y)
                         per episode. Same shapes as sample_episode().
        ood_features:    pre-computed BACKBONE features for the OOD pool
                         (e.g. SVHN), shape (N_ood, D). Pass None to skip
                         OOD metrics.
        interpretation:  "softmax" or "evidential" -- decides how the
                         prototype-similarity logits become probabilities
                         and OOD scores.
        ece_bins:        bin count for ECE.
        device:          where to run forward passes.

    Returns:
        dict with the Phase 2 schema (see module docstring).
    """
    model = model.to(device)
    model.eval()
    backbone = model.backbone

    if ood_features is not None:
        ood_features = ood_features.to(device)

    per_ep_acc, per_ep_f1, per_ep_ece, per_ep_brier = [], [], [], []
    per_ep_auroc, per_ep_fpr95 = [], []
    pooled_probs, pooled_targets = [], []
    last_id_scores: np.ndarray | None = None
    last_ood_scores: np.ndarray | None = None

    with torch.no_grad():
        for i, (sx, sy, qx, qy) in enumerate(test_iterable):
            sx = sx.to(device); sy = sy.to(device)
            qx = qx.to(device); qy = qy.to(device)

            # Backbone features (frozen) -> not re-applied inside
            # forward_proto_from_features.
            sx_feats = backbone(sx)
            qx_feats = backbone(qx)

            q_logits = _proto_logits_for_query(model, sx_feats, sy, qx_feats)
            probs = _logits_to_probs(q_logits, num_classes, interpretation,
                                     model.head)

            acc = accuracy(probs, qy)
            f1  = f1_macro(probs, qy, num_classes=num_classes)
            ece = expected_calibration_error(probs, qy, num_bins=ece_bins)
            bri = brier_score(probs, qy, num_classes)

            per_ep_acc.append(acc)
            per_ep_f1.append(f1)
            per_ep_ece.append(ece)
            per_ep_brier.append(bri)
            pooled_probs.append(probs.cpu())
            pooled_targets.append(qy.cpu())

            # OOD scoring uses THIS episode's prototypes.
            auroc = float("nan")
            fpr95 = float("nan")
            if ood_features is not None:
                ood_logits = _proto_logits_for_query(
                    model, sx_feats, sy, ood_features,
                )
                id_score  = _logits_to_id_score(q_logits, num_classes,
                                                interpretation, model.head)
                ood_score = _logits_to_id_score(ood_logits, num_classes,
                                                interpretation, model.head)
                id_np  = id_score.cpu().numpy()
                ood_np = ood_score.cpu().numpy()
                last_id_scores, last_ood_scores = id_np, ood_np
                auroc = ood_auroc(id_np, ood_np)
                fpr95 = fpr_at_95_tpr(id_np, ood_np)
                per_ep_auroc.append(auroc)
                per_ep_fpr95.append(fpr95)

            if logger is not None:
                logger.info(
                    f"ep {i:3d}  acc={acc:.3f}  F1={f1:.3f}  "
                    f"ECE={ece:.3f}  Brier={bri:.3f}  "
                    f"AUROC={auroc:.3f}  FPR95={fpr95:.3f}"
                )
            if wandb_run is not None:
                wandb_run.log({
                    "eval/episode": i,
                    "eval/accuracy": float(acc),
                    "eval/f1_macro": float(f1),
                    "eval/ECE": float(ece),
                    "eval/Brier": float(bri),
                    "eval/OOD_AUROC": float(auroc) if auroc == auroc else 0.0,
                    "eval/FPR_at_95_TPR": float(fpr95) if fpr95 == fpr95 else 0.0,
                    "eval/accuracy_running_mean": float(np.mean(per_ep_acc)),
                    "eval/ECE_running_mean": float(np.mean(per_ep_ece)),
                    "eval/AUROC_running_mean": (
                        float(np.mean(per_ep_auroc)) if per_ep_auroc else 0.0
                    ),
                })

    n = len(per_ep_acc)
    if n == 0:
        raise RuntimeError("evaluate_episodic: test_iterable was empty")

    pooled_probs_t   = torch.cat(pooled_probs,   dim=0)
    pooled_targets_t = torch.cat(pooled_targets, dim=0)
    pooled_ece = expected_calibration_error(
        pooled_probs_t, pooled_targets_t, num_bins=ece_bins,
    )

    def _ci95(arr):
        if len(arr) == 0:
            return 0.0
        return float(1.96 * np.std(arr) / np.sqrt(len(arr)))

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
        "ood_auroc_mean":       float(np.mean(per_ep_auroc)) if per_ep_auroc else 0.0,
        "ood_auroc_std":        float(np.std(per_ep_auroc))  if per_ep_auroc else 0.0,
        "fpr_at_95_tpr_mean":   float(np.mean(per_ep_fpr95)) if per_ep_fpr95 else 1.0,
        "fpr_at_95_tpr_std":    float(np.std(per_ep_fpr95))  if per_ep_fpr95 else 0.0,
        "num_episodes":         int(n),
        "interpretation":       str(interpretation),
    }
    # Return pooled tensors too so the caller can dump reliability /
    # confusion-matrix plots without re-running the model.
    return {
        "summary": summary,
        "pooled_probs": pooled_probs_t,
        "pooled_targets": pooled_targets_t,
        "last_id_scores": last_id_scores,
        "last_ood_scores": last_ood_scores,
    }
