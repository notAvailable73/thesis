"""Adapter-only checkpoints (TMLR plan Task 6).

A full checkpoint stores the frozen ImageNet backbone too (~45 MB for
ResNet-18). The frozen part is reproducible from the public torchvision
weights, so the release keeps only what training changed: every trainable
parameter plus the head's evidence scale/bias tensors (parameters on
evidential heads, buffers on softmax heads). That is a few KB per run, small
enough that adapters never again exist in only one place.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

#: Head tensors kept even when they are buffers (softmax heads store the
#: evidence affine as constants; RQ2 re-reads them).
_EVIDENCE_TENSOR_SUFFIXES = (
    "_evidence_raw_scale", "_evidence_bias",
    "_evidence_scale_const", "_evidence_bias_const",
)


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Every requires_grad parameter + the evidence-affine tensors, detached
    CPU copies, keyed exactly as in `model.state_dict()`."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    out: Dict[str, torch.Tensor] = {}
    for name, t in model.state_dict().items():
        if name in trainable or name.endswith(_EVIDENCE_TENSOR_SUFFIXES):
            out[name] = t.detach().cpu().clone()
    return out


def load_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load an adapter-only state into a freshly built model (same config).

    Raises KeyError on any key the model does not have; the frozen backbone is
    left as built (pretrained torchvision weights).
    """
    own = model.state_dict()
    unknown = sorted(k for k in state if k not in own)
    if unknown:
        raise KeyError(f"adapter state has keys the model lacks: {unknown[:5]}"
                       f"{' ...' if len(unknown) > 5 else ''}")
    # The converse matters as much: a state missing a trainable tensor would
    # load "fine" under strict=False and leave e.g. a zero-init lora_B/up, i.e.
    # an untrained adapter that silently evaluates as the frozen backbone.
    missing = sorted(k for k in trainable_state_dict(model) if k not in state)
    if missing:
        raise KeyError(f"adapter state is missing trainable tensors: {missing[:5]}"
                       f"{' ...' if len(missing) > 5 else ''}")
    model.load_state_dict(state, strict=False)
