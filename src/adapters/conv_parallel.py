"""TMLR plan A3 `btl_ss` — a parallel bottleneck wrapped around ONE conv.

Papers: Houlsby et al. 2019 (bottleneck form) and He et al. 2022 (the
serial/parallel vocabulary); PAPER SUMMARIES/PEFT_summaries.txt papers 1 and 7.
The module is the existing Conv1x1Bottleneck body (1x1 down -> ReLU -> 1x1 up,
with biases, up zero-initialised), but placed in parallel to a single frozen
conv instead of around a whole block:

    y = base(x) + up(relu(down(x)))

It sits exactly where the single-site LoRA arm does (`default_lora_targets`).
With the ReLU removed and both biases zero it computes the same function as
LoRAConv2d with alpha = rank (tests/test_conv_parallel.py), so btl_ss vs
lora_ss isolates the non-linearity + biases at an identical site. btl_ss vs
btl_ms isolates the number and location of insertion sites.

Restricted to 1x1, padding-0 base convs: both single sites in the design are
1x1 convs, and a 1x1 down-projection only matches a kxk base's output when the
base is itself 1x1. The down-projection copies the base's stride, so the R18 / R34
downsample site (stride 2) keeps its output shape.

Deviation flagged: on MobileNetV2/V3 the site is the LINEAR projection conv,
which MobileNetV2 keeps linear on purpose (Sandler et al. 2018). The parallel
ReLU branch adds a non-linearity beside it, not in series. The paper does not
address this; the comparison with lora_ss (linear) at the same site measures it.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .lora import _replace_submodule, default_lora_targets


class ConvParallelBottleneck(nn.Module):
    """Frozen 1x1 conv plus a trainable parallel bottleneck branch."""

    def __init__(self, base_conv: nn.Conv2d, rank: int):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(f"ConvParallelBottleneck expects nn.Conv2d, got "
                            f"{type(base_conv).__name__}")
        if (tuple(base_conv.kernel_size) != (1, 1)
                or tuple(base_conv.padding) != (0, 0)
                or tuple(base_conv.dilation) != (1, 1)
                or base_conv.groups != 1):
            raise ValueError(
                "ConvParallelBottleneck needs a dense 1x1 base conv with padding 0; "
                f"got kernel={base_conv.kernel_size} padding={base_conv.padding} "
                f"groups={base_conv.groups}")
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.down = nn.Conv2d(base_conv.in_channels, self.rank, kernel_size=1,
                              stride=base_conv.stride, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(self.rank, base_conv.out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.act(self.down(x)))


class SingleSiteBottleneckAdapter(nn.Module):
    """Replaces one frozen conv in the backbone with a ConvParallelBottleneck.

    Same contract as LoRAAdapter: the adaptation lives inside the backbone, so
    this module's forward is the identity on the pooled feature, and the
    trainable params are found via `p.requires_grad`.
    """

    #: gradients must flow through the backbone to reach the wrapped conv.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module, rank: int, target: Optional[str] = None):
        super().__init__()
        if backbone is None:
            raise ValueError("SingleSiteBottleneckAdapter requires a backbone")
        self.rank = int(rank)
        self.target: str = target or default_lora_targets(backbone)[0]
        base = backbone.get_submodule(self.target)
        _replace_submodule(backbone, self.target, ConvParallelBottleneck(base, self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
