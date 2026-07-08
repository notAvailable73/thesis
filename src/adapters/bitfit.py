"""BitFit (Ben-Zaken 2022) for the B-PEFT ResNet-18 backbone — Step 5.

BitFit freezes every weight matrix and gain, and trains ONLY the additive bias
vectors (plus the task head). For ResNet-18 the convolutions have bias=False
(they are followed by BatchNorm), so the only bias parameters are the
BatchNorm shift terms (beta). We therefore unfreeze exactly the `.bias`
parameters of the backbone and leave BatchNorm gains (weight / gamma) and all
conv weights frozen — faithful to "bias-only", not "affine-only".

For torchvision ResNet-18 this is ~4.8K trainable parameters (all BN betas),
matching the proposal's extreme "<0.1% params" selective baseline used for RQ2.

Like LoRAAdapter, the trainable parameters live inside the backbone subtree
(they ARE the backbone's bias tensors, now with requires_grad=True), so the
optimiser and count_trainable_params discover them automatically; this
wrapper's forward is the identity on the pooled feature vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BitFitAdapter(nn.Module):
    """Unfreeze only the backbone's bias parameters (BatchNorm betas)."""

    #: gradients must flow through the backbone to reach the bias params.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if backbone is None:
            raise ValueError("BitFitAdapter requires the backbone")
        num_bias = 0
        num_tensors = 0
        for name, p in backbone.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
                num_bias += p.numel()
                num_tensors += 1
        if num_tensors == 0:
            raise ValueError(
                "BitFitAdapter found no `.bias` parameters in the backbone; "
                "nothing to train. (ResNet-18 BatchNorm layers should expose "
                "bias terms — check the backbone.)"
            )
        self.num_bias_params = int(num_bias)
        self.num_bias_tensors = int(num_tensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
