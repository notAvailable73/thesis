"""LoRA (Hu 2021) for the B-PEFT ResNet-18 backbone — Step 5 rewrite.

The Step 2 stub applied a low-rank residual to the POOLED feature vector
(really a low-rank bottleneck adapter). Step 5 makes LoRA what the proposal
means: a low-rank reparameterization of the backbone's OWN weight matrices,
in place, so it changes what the backbone computes.

    W_eff = W_frozen + (alpha / rank) * B @ A          (Hu 2021, eq. for ΔW)

with A initialised Kaiming-uniform and B initialised to zero, so ΔW = 0 at
init and training starts from the pristine pre-trained network.

Two concrete pieces:

  - LoRALayer:  the linear form (in_features -> out_features). Kept for the
                unit test (identity-at-init + analytical param count) and as
                the reference implementation of the A/B decomposition.
  - LoRAConv2d: wraps a FROZEN nn.Conv2d and adds the low-rank update on the
                channel dimension. For a 1x1 conv this is exactly LoRALayer on
                the channel matrix; for a k x k conv the down-projection keeps
                the base kernel so the spatial receptive field is preserved
                (ConvLoRA, Aleem 2024).
  - LoRAAdapter: the object placed in the model's `adapter` slot. It does NOT
                operate on pooled features (its forward is the identity);
                instead, at construction it INJECTS LoRAConv2d in place of the
                configured target conv modules inside the frozen backbone. The
                trainable A/B params therefore live in the backbone subtree,
                where the optimiser (`p.requires_grad`) and
                count_trainable_params pick them up automatically.

Analytical param count per target conv = rank * (in_channels + out_channels)
for a 1x1 conv (rank*in for A + out*rank for B). NOTE: implementation.txt 5.1
quotes "2 * rank * (in + out)"; the honest count for a standard A/B
decomposition is HALF that (rank*(in+out)). We implement the correct value and
assert it in tests/test_lora.py — flagged in step_writeups/step5.txt.
"""
from __future__ import annotations
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Linear LoRA: y = base(x) + (alpha/rank) * dropout(x) @ A @ B.

    If `base` is None the layer returns ONLY the low-rank delta (useful for
    the identity-at-init test, where the delta is exactly zero because B=0).
    A: (in_features, rank) Kaiming-uniform; B: (rank, out_features) zero.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: Optional[float] = None, dropout: float = 0.0,
                 base: Optional[nn.Module] = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.base = base
        if base is not None:
            for p in self.base.parameters():
                p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (self.dropout(x) @ self.lora_A @ self.lora_B)
        if self.base is None:
            return delta
        return self.base(x) + delta


class LoRAConv2d(nn.Module):
    """Wrap a frozen nn.Conv2d with a low-rank additive update.

        y = base(x) + (alpha/rank) * lora_B(lora_A(dropout(x)))

    lora_A keeps the base conv's kernel/stride/padding/dilation so its output
    has the SAME spatial size as the base output (receptive field preserved,
    ConvLoRA-style); lora_B is a 1x1 conv projecting rank -> out_channels.
    lora_A Kaiming-uniform, lora_B zero -> identity at init.
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int,
                 alpha: Optional[float] = None, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got "
                            f"{type(base_conv).__name__}")
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Conv2d(
            base_conv.in_channels, rank,
            kernel_size=base_conv.kernel_size, stride=base_conv.stride,
            padding=base_conv.padding, dilation=base_conv.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, base_conv.out_channels,
                                kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


# Default LoRA target: the 1x1 downsample conv of the final ResNet-18 stage.
# A 1x1 conv is a pure channel matrix, so LoRA on it is the clean
# rank*(in+out) reparameterization; restricting to the last block keeps the
# param count (12,288 at rank 16) comparable to Bottleneck-16 (~19K) — see
# implementation.txt 5.1 RISK.
_DEFAULT_LORA_TARGETS: List[str] = ["layer4.0.downsample.0"]

def _mobilenetv3_default_lora_target(features: nn.Module) -> List[str]:
    """Step 8: the MobileNetV3 analogue of the ResNet default — the 1x1 LINEAR
    PROJECTION conv of the DEEPEST inverted-residual block (for
    MobileNetV3-Small: features.11.block.3.0, 576 -> 96).

    Same reasoning as the ResNet target: a 1x1 dense (groups==1) conv is a pure
    channel matrix, so LoRA on it is the clean rank*(in+out) reparameterization
    — 10,752 params at rank 16, comparable to the ResNet target's 12,288. The
    1x1 expand / KxK depthwise / 1x1 project convs are exactly the insertion
    sites named in PAPER SUMMARIES/CNN_paper_summaries.txt §8.

    Discovered by traversal rather than a hardcoded index so a torchvision
    layout change surfaces as a clear error, not a silently wrong target. The
    projection conv is the LAST 1x1 dense conv in the block (the depthwise conv
    is grouped, and the squeeze-excite 1x1s come before the projection).
    """
    for i in range(len(features) - 1, -1, -1):
        block = features[i]
        if not hasattr(block, "use_res_connect"):
            continue
        last_name = None
        for name, m in block.named_modules():
            if (isinstance(m, nn.Conv2d) and tuple(m.kernel_size) == (1, 1)
                    and m.groups == 1):
                last_name = name
        if last_name:
            return [f"features.{i}.{last_name}"]
    raise ValueError(
        "found no 1x1 projection conv in the backbone's inverted-residual "
        "blocks; set cfg.adapter.lora_targets explicitly."
    )


def default_lora_targets(backbone: nn.Module) -> List[str]:
    """Family-appropriate default LoRA targets for `backbone`.

    Overridden per-experiment by cfg.adapter.lora_targets, which stays the
    single knob for widening or moving the target set.
    """
    if all(hasattr(backbone, a) for a in ("layer1", "layer4")):
        return list(_DEFAULT_LORA_TARGETS)
    features = getattr(backbone, "features", None)
    if features is not None and any(
            hasattr(m, "use_res_connect") for m in features):
        return _mobilenetv3_default_lora_target(features)
    raise ValueError(
        f"No default LoRA target known for {type(backbone).__name__}; set "
        f"cfg.adapter.lora_targets explicitly."
    )


def _replace_submodule(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    """Replace root.<dotted> (e.g. 'layer4.0.downsample.0') with new_module.

    Handles numeric path components (nn.Sequential / nn.ModuleList indices).
    """
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    key = parts[-1]
    if key.isdigit():
        parent[int(key)] = new_module
    else:
        setattr(parent, key, new_module)


class LoRAAdapter(nn.Module):
    """Injects LoRAConv2d into the frozen backbone's target conv layers.

    The `adapter` slot in BPEFTModel expects a module whose forward maps the
    POOLED feature vector -> adapted feature vector. LoRA changes the backbone
    itself, not the pooled feature, so this adapter's forward is the identity;
    the real adaptation is the in-place injection performed at construction.

    Because the injected LoRAConv2d modules live inside `backbone`, their
    trainable A/B params are discovered by the standard
    `[p for p in model.parameters() if p.requires_grad]` optimiser wiring and
    by count_trainable_params — LoRAAdapter itself holds no parameters.
    """

    #: gradients must flow through the backbone for the injected LoRA params.
    backbone_trainable = True

    def __init__(self, backbone: nn.Module, rank: int,
                 alpha: Optional[float] = None,
                 targets: Optional[Sequence[str]] = None,
                 dropout: float = 0.0):
        super().__init__()
        if backbone is None:
            raise ValueError("LoRAAdapter requires the backbone to inject into")
        self.rank = int(rank)
        self.alpha = alpha
        self.targets: List[str] = (list(targets) if targets
                                   else default_lora_targets(backbone))
        injected = 0
        for name in self.targets:
            base = backbone.get_submodule(name)
            if not isinstance(base, nn.Conv2d):
                raise TypeError(
                    f"LoRA target {name!r} is a {type(base).__name__}, not "
                    f"nn.Conv2d. cfg.adapter.lora_targets must name conv layers."
                )
            _replace_submodule(backbone, name,
                               LoRAConv2d(base, self.rank, alpha, dropout))
            injected += 1
        if injected == 0:
            raise ValueError("LoRAAdapter injected 0 layers; check lora_targets")
        self.num_injected = injected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity on the pooled feature: the adaptation already happened
        # inside the backbone.
        return x
