"""Linear Probing baseline (proposal.txt §6) — Step 5.

The backbone stays frozen and NO adapter is inserted; only the head trains.
Under the Step 4 parameter-free PrototypeHead this is the most extreme PEFT
case: with a softmax interpretation the head has zero parameters, so there is
literally nothing to train and all transfer comes from the frozen ImageNet
features (a training-free nearest-prototype classifier). With the evidential
interpretation the head's learnable evidence-affine adds 2 scalars, so a
linear-probe+evidential run tunes exactly those 2 parameters — still "~0".

The distinction from FullFTAdapter is one line: FullFT flips the backbone to
requires_grad=True; LinearProbe leaves it frozen (backbone_trainable = False),
so BPEFTModel runs it under no_grad exactly like the Bottleneck path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbeAdapter(nn.Module):
    """Frozen backbone, identity adapter; only the head (if any) trains."""

    #: backbone stays frozen — run under no_grad like Bottleneck.
    backbone_trainable = False

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
