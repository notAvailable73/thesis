"""Step 6 — serial / parallel in-block placement of a 1x1 Bottleneck adapter.

RQ1 isolates PLACEMENT: the adapter form is a fixed 1x1 channel bottleneck
(the conv analogue of the post-pool Bottleneck), inserted at the final
BasicBlock of each selected ResNet-18 stage via a forward hook:

    serial   : out' = out + body(out)      (sequential / Residual-Sequential)
    parallel : out' = out + body(input)    (parallel   / Residual-Parallel)

body = 1x1 down -> ReLU -> 1x1 up, with up zero-init so the placed model is
identical to the frozen backbone at init. The adapter's own forward is the
identity on the pooled vector — the adaptation happens INSIDE the backbone
(same pattern as LoRAAdapter), so backbone_trainable = True and the backbone
must run with autograd for gradients to reach the inserted bottlenecks.

Attaching to the FINAL block of a stage means the block input and output share
shape (B, C, H, W), so both serial and parallel compose with no shape mismatch.

Caveat (Conv-Adapter, Chen 2022): a 1x1/linear adapter "loses locality" — the
form Conv-Adapter reports as its core failure mode; depth-wise-separable K x K
is their preferred form. We accept the 1x1 form deliberately to keep the
adapter identical across placements so RQ1 measures PLACEMENT alone. See
docs/superpowers/specs/2026-07-21-step6-adapter-placement-design.md.
"""
from __future__ import annotations
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

#: ResNet-18 stage attribute names, indexed 0..3 by cfg.adapter.block_ids.
_STAGE_ATTRS = ["layer1", "layer2", "layer3", "layer4"]


class Conv1x1Bottleneck(nn.Module):
    """1x1 down -> ReLU -> 1x1 up. NO residual (the placement hook adds it).

    up is zero-init (weight AND bias) so body(x) == 0 at init, making the
    placed backbone byte-identical to the frozen one until training moves it.
    """

    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(rank, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def register_serial_adapter(block: nn.Module, mod: Conv1x1Bottleneck):
    """Forward hook: transform the block OUTPUT in-line (out + body(out))."""
    def hook(_module, _inp, out):
        return out + mod(out)
    return block.register_forward_hook(hook)


def register_parallel_adapter(block: nn.Module, mod: Conv1x1Bottleneck):
    """Forward hook: run body on the block INPUT, summed at the output."""
    def hook(_module, inp, out):
        return out + mod(inp[0])
    return block.register_forward_hook(hook)


class PlacementAdapter(nn.Module):
    """Places 1x1 Bottleneck adapters at the final block of each selected
    stage. Holds the trainable bottlenecks (`self.bodies`) and registers the
    forward hooks on the backbone. Its own forward is the identity on the
    pooled feature — the adaptation lives inside the backbone, so the trainable
    params are discovered by the standard
    `[p for p in model.parameters() if p.requires_grad]` optimiser wiring.
    """

    #: gradients must flow through the backbone to reach the placed bottlenecks.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module, rank: int, placement: str,
                 block_ids: Optional[Sequence[int]] = None):
        super().__init__()
        if placement not in ("serial", "parallel"):
            raise ValueError(
                f"placement must be 'serial' or 'parallel', got {placement!r}")
        if backbone is None:
            raise ValueError("PlacementAdapter requires a backbone")
        self.placement = placement
        self.rank = int(rank)
        ids: List[int] = ([int(i) for i in block_ids]
                          if block_ids is not None else [0, 1, 2, 3])
        self.block_ids = ids
        self.bodies = nn.ModuleList()
        self._handles = []
        register = (register_serial_adapter if placement == "serial"
                    else register_parallel_adapter)
        for sid in ids:
            if not 0 <= sid < len(_STAGE_ATTRS):
                raise ValueError(
                    f"block_id {sid} out of range 0..{len(_STAGE_ATTRS) - 1}")
            block = getattr(backbone, _STAGE_ATTRS[sid])[-1]
            channels = block.conv2.out_channels
            body = Conv1x1Bottleneck(channels, self.rank)
            self.bodies.append(body)
            self._handles.append(register(block, body))
        if len(self.bodies) == 0:
            raise ValueError("PlacementAdapter placed 0 adapters; check block_ids")
        self.num_placed = len(self.bodies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity on the pooled feature; the adaptation already happened
        # inside the backbone via the registered hooks.
        return x
