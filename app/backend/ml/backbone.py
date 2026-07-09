"""Frozen ResNet18 feature extractor.

This is the "frozen backbone" of B-PEFT: an ImageNet-pretrained CNN with its
classifier head removed, producing a 512-d embedding per image. Weights are
frozen (``requires_grad = False``) and the module is kept in ``eval`` mode — we
never train it at runtime; we only read features out of it.

The app deliberately does NOT bolt on a trainable adapter at runtime: enrollment
uses the parameter-free prototype head (mean of embeddings), which needs no
gradient step. The adapter lives in the research code (`src/adapters/`); swapping
it in here would mean loading a trained checkpoint into this module.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from app.backend.core.logging import get_logger

log = get_logger("ml.backbone")

WeightsStatus = Literal["imagenet", "random"]


class FrozenResNet18(nn.Module):
    """ResNet18 truncated at the global-average-pool, output dim = 512."""

    output_dim = 512

    def __init__(self, allow_random_weights: bool = True) -> None:
        super().__init__()
        self.weights_status: WeightsStatus = self._build(allow_random_weights)
        # Freeze everything — inference only.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _build(self, allow_random_weights: bool) -> WeightsStatus:
        from torchvision.models import resnet18, ResNet18_Weights

        status: WeightsStatus
        try:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            status = "imagenet"
            log.info("Loaded ImageNet-pretrained ResNet18 weights.")
        except Exception as exc:  # noqa: BLE001 - offline / no cache
            if not allow_random_weights:
                raise
            log.warning(
                "Could not load pretrained weights (%s). Falling back to "
                "RANDOM init — the pipeline works but identification quality "
                "will be poor until weights are cached.",
                exc,
            )
            net = resnet18(weights=None)
            status = "random"

        # Drop the 1000-way classifier; keep everything up to and including avgpool.
        self.features = nn.Sequential(*list(net.children())[:-1])
        return status

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W) -> (B, 512)`` embeddings."""
        feats = self.features(x)          # (B, 512, 1, 1)
        return torch.flatten(feats, 1)    # (B, 512)
