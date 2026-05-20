"""Accuracy + Macro-F1 (proposal §7 Performance metrics)."""
import torch


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    return (probs.argmax(dim=-1) == targets).float().mean().item()


def f1_macro(probs: torch.Tensor, targets: torch.Tensor,
             num_classes: int | None = None) -> float:
    """Unweighted (macro) F1 across classes.

    For each class c:
      precision_c = TP_c / (TP_c + FP_c)
      recall_c    = TP_c / (TP_c + FN_c)
      f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
    Then average f1_c over the classes present in (preds ∪ targets), so a
    5-way episode reports the mean over its 5 classes regardless of which
    class IDs they happen to be.

    Returns 0.0 for an empty batch. Single-class edge cases yield f1=1.0
    or f1=0.0 depending on whether the model agreed.

    This is the proposal §7 "Macro-F1" metric. It was missing from the
    Phase 1 metrics JSONs and is added here so every Step-4-onward
    evaluator can report it.
    """
    if probs.numel() == 0:
        return 0.0
    preds = probs.argmax(dim=-1)
    if num_classes is None:
        # Use the classes that actually appear so 5-way episodes work.
        classes = torch.unique(torch.cat([preds, targets])).tolist()
    else:
        classes = list(range(int(num_classes)))
    if not classes:
        return 0.0

    f1_scores = []
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        if tp + fp == 0 and tp + fn == 0:
            # Class did not appear in either preds or targets — skip.
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    if not f1_scores:
        return 0.0
    return float(sum(f1_scores) / len(f1_scores))
