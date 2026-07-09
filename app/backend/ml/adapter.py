"""Trained B-PEFT adapter — loads the Phase-2 episodic checkpoint.

This mirrors ``src/adapters/bottleneck.py`` bit-for-bit (the same tiny
down -> ReLU -> up residual module) so weights trained by
``scripts/train.py --config configs/exp_phase2_evidential_retuned.yaml``
load directly. It is a local copy rather than an import of ``src.adapters``
so the app stays self-contained and deployable without the research code
present (see ``backbone.py``'s docstring: "swapping it in here would mean
loading a trained checkpoint into this module" — this is that step).

Checkpoint schema (``scripts/train.py``'s episodic-trainer save path):
    state_dict["adapter.down.weight" / ".down.bias" / ".up.weight" / ".up.bias"]
    state_dict["head._evidence_raw_scale"]   -- learnable evidence-affine scale (pre-softplus)
    state_dict["head._evidence_bias"]        -- learnable evidence-affine bias
``PrototypeHead.cosine_scale`` is a plain Python float, not a tensor, so it
never lands in the state_dict; configs/exp_phase2_evidential_retuned.yaml
(via configs/exp_phase2_evidential.yaml) fixes it at 10.0 — see
``_HEAD_COSINE_SCALE`` below. If a checkpoint is ever trained from a config
with a different ``head.cosine_scale``, update this constant to match.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.backend.core.logging import get_logger

log = get_logger("ml.adapter")

_HEAD_COSINE_SCALE = 10.0


class BottleneckAdapter(nn.Module):
    """Houlsby-style bottleneck adapter: down -> ReLU -> up, with residual.

    Architecturally identical to ``src/adapters/bottleneck.py``.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(rank, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


@dataclass(frozen=True)
class TrainedAdapterBundle:
    adapter: BottleneckAdapter
    evidence_scale: float  # combined: head.cosine_scale * softplus(raw_scale)
    evidence_bias: float
    checkpoint_path: str
    best_val_acc: float


def load_trained_adapter(
    checkpoint_path: Path, dim: int = 512, rank: int = 16,
) -> TrainedAdapterBundle | None:
    """Load the Phase-2 trained adapter + evidence affine.

    Returns ``None`` (never raises) if the checkpoint is absent or fails to
    load — the app falls back to the untrained backbone-only pipeline so a
    missing checkpoint never blocks startup.
    """
    if not checkpoint_path.exists():
        log.info(
            "No trained adapter checkpoint at %s -- using untrained "
            "backbone-only pipeline.", checkpoint_path,
        )
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["state_dict"]

        adapter = BottleneckAdapter(dim=dim, rank=rank)
        adapter.down.weight.data.copy_(sd["adapter.down.weight"])
        adapter.down.bias.data.copy_(sd["adapter.down.bias"])
        adapter.up.weight.data.copy_(sd["adapter.up.weight"])
        adapter.up.bias.data.copy_(sd["adapter.up.bias"])
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)

        raw_scale = sd["head._evidence_raw_scale"]
        bias = sd["head._evidence_bias"]
        evidence_scale = float(F.softplus(raw_scale)) * _HEAD_COSINE_SCALE
        evidence_bias = float(bias)

        best_val_acc = float(ckpt.get("best_val_acc", float("nan")))
        log.info(
            "Loaded trained adapter from %s (best_val_acc=%.3f, "
            "evidence_scale=%.3f, evidence_bias=%.3f)",
            checkpoint_path, best_val_acc, evidence_scale, evidence_bias,
        )
        return TrainedAdapterBundle(
            adapter=adapter,
            evidence_scale=evidence_scale,
            evidence_bias=evidence_bias,
            checkpoint_path=str(checkpoint_path),
            best_val_acc=best_val_acc,
        )
    except Exception as exc:  # noqa: BLE001 - never let a bad checkpoint crash startup
        log.warning(
            "Could not load trained adapter from %s (%s); falling back to "
            "backbone-only.", checkpoint_path, exc,
        )
        return None
