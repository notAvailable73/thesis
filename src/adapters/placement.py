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

Step 8 (Phase 5) makes the stage resolution BACKBONE-AGNOSTIC so the same
placement study transfers to MobileNetV3-Small: the stage list comes from
`resolve_stage_paths`, which dispatches on the backbone family (torchvision
ResNet -> layer1..layer4's last block; torchvision MobileNetV3 -> the
stage-final residual InvertedResidual blocks) and can always be overridden
explicitly with cfg.adapter.stage_paths. The ResNet path is unchanged so Step 6
stays reproducible.

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


def _is_resnet(backbone: nn.Module) -> bool:
    return all(hasattr(backbone, attr) for attr in _STAGE_ATTRS)


def _is_mobilenetv3(backbone: nn.Module) -> bool:
    features = getattr(backbone, "features", None)
    return features is not None and any(
        hasattr(m, "use_res_connect") for m in features
    )


def resolve_stage_paths(backbone: nn.Module) -> List[str]:
    """Dotted module paths of the per-stage insertion points, shallow -> deep.

    ResNet     : the LAST BasicBlock of layer1..layer4, i.e.
                 ["layer1.1", "layer2.1", "layer3.1", "layer4.1"] for ResNet-18
                 — the same module objects Step 6 hooked via `layerN[-1]`.
    MobileNetV3: the stage-final residual InvertedResidual blocks
                 (see src/backbones/mobilenetv3.py:mobilenetv3_stage_paths).
    """
    if _is_resnet(backbone):
        return [f"{attr}.{len(getattr(backbone, attr)) - 1}"
                for attr in _STAGE_ATTRS]
    if _is_mobilenetv3(backbone):
        from ..backbones.mobilenetv3 import mobilenetv3_stage_paths
        return mobilenetv3_stage_paths(backbone)
    raise ValueError(
        f"Cannot auto-derive adapter placement stages for "
        f"{type(backbone).__name__}. Set cfg.adapter.stage_paths to a list of "
        f"dotted module paths (e.g. ['features.11']) to place explicitly."
    )


def infer_block_channels(block: nn.Module) -> int:
    """Number of channels a block OUTPUTS (== its input channels, since only
    shape-preserving blocks are eligible placement sites).

    Ordered fallbacks, most specific first:
      1. `block.conv2.out_channels`  — torchvision ResNet BasicBlock (the Step 6
         path, kept first so ResNet behaviour is bit-for-bit unchanged).
      2. `block.out_channels`        — torchvision MobileNetV3 InvertedResidual.
      3. the last Conv2d / BatchNorm2d in module order — generic CNN blocks
         (the projection conv and its norm come last in an inverted residual, so
         SE-internal convs do not win).
    """
    conv2 = getattr(block, "conv2", None)
    if isinstance(conv2, nn.Conv2d):
        return int(conv2.out_channels)
    out_channels = getattr(block, "out_channels", None)
    if isinstance(out_channels, int):
        return int(out_channels)
    last: Optional[int] = None
    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            last = int(m.out_channels)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            last = int(getattr(m, "num_features", getattr(m, "num_channels", 0)))
    if not last:
        raise ValueError(
            f"could not infer the channel count of a "
            f"{type(block).__name__} placement site"
        )
    return last


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

    Stage sites are resolved by `resolve_stage_paths(backbone)` (ResNet or
    MobileNetV3) unless `stage_paths` names them explicitly; `block_ids` then
    selects which of those stages actually get an adapter (default: all).
    """

    #: gradients must flow through the backbone to reach the placed bottlenecks.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module, rank: int, placement: str,
                 block_ids: Optional[Sequence[int]] = None,
                 stage_paths: Optional[Sequence[str]] = None):
        super().__init__()
        if placement not in ("serial", "parallel"):
            raise ValueError(
                f"placement must be 'serial' or 'parallel', got {placement!r}")
        if backbone is None:
            raise ValueError("PlacementAdapter requires a backbone")
        self.placement = placement
        self.rank = int(rank)
        all_paths: List[str] = ([str(p) for p in stage_paths] if stage_paths
                                else resolve_stage_paths(backbone))
        ids: List[int] = ([int(i) for i in block_ids]
                          if block_ids is not None else list(range(len(all_paths))))
        self.block_ids = ids
        self.bodies = nn.ModuleList()
        self.stage_paths: List[str] = []
        self.stage_channels: List[int] = []
        self._handles = []
        register = (register_serial_adapter if placement == "serial"
                    else register_parallel_adapter)
        for sid in ids:
            if not 0 <= sid < len(all_paths):
                raise ValueError(
                    f"block_id {sid} out of range 0..{len(all_paths) - 1} "
                    f"(stages: {all_paths})")
            path = all_paths[sid]
            block = backbone.get_submodule(path)
            channels = infer_block_channels(block)
            body = Conv1x1Bottleneck(channels, self.rank)
            self.bodies.append(body)
            self.stage_paths.append(path)
            self.stage_channels.append(int(channels))
            self._handles.append(register(block, body))
        if len(self.bodies) == 0:
            raise ValueError("PlacementAdapter placed 0 adapters; check block_ids")
        self.num_placed = len(self.bodies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity on the pooled feature; the adaptation already happened
        # inside the backbone via the registered hooks.
        return x
