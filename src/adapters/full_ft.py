"""Full Fine-Tuning baseline (proposal.txt §6) — Step 5.

Unfreezes the ENTIRE backbone. No adapter module is inserted; the adapted
feature is just the (now trainable) backbone's output. This is the proposal's
primary non-PEFT baseline: with ~11.7M trainable parameters against a 25-image
support set, it is EXPECTED to overfit catastrophically and score below the
PEFT methods on the query set. That collapse IS the result — it demonstrates
why parameter-efficient adaptation is necessary in the few-shot regime
(proposal §3, Problem 1). If Full-FT instead matches the PEFT methods, suspect
a data leak rather than a genuine win.

Note on BatchNorm: the episodic trainer keeps the backbone in eval() mode, so
BN running statistics stay frozen at their ImageNet values even for Full-FT;
the BN affine parameters and conv weights still receive gradients and update.
Freezing BN running stats avoids leaking query-set statistics through 25-image
support batches (standard few-shot practice) — documented in step5.txt.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FullFTAdapter(nn.Module):
    """Unfreeze the whole backbone; no post-pool adapter."""

    #: the entire backbone trains.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if backbone is None:
            raise ValueError("FullFTAdapter requires the backbone")
        num = 0
        for p in backbone.parameters():
            p.requires_grad = True
            num += p.numel()
        self.num_backbone_params = int(num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
