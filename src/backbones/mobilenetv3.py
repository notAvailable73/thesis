"""MobileNetV3-Small frozen backbone (Step 8 / Phase 5).

Second mandatory backbone from proposal §5A. The efficiency half of RQ4 needs a
genuinely small CNN: ResNet-18 is 11.7M params / 1.8 GFLOPs, MobileNetV3-Small
is ~2.5M params / ~57 MAdds (Howard 2019, reference numbers as printed in
`PAPER SUMMARIES/CNN_paper_summaries.txt` §6).

Feature contract (identical to build_frozen_resnet18)
-----------------------------------------------------
torchvision's MobileNetV3 forward is
    features -> avgpool -> flatten -> classifier
so replacing `classifier` with nn.Identity() makes `backbone(x)` return the
POOLED conv feature of shape (B, 576) — the same (B, D) contract the adapters,
PrototypeHead and the feature-caching evaluator already expect. 576 is
`6 * last_inverted_residual_out_channels` (6 * 96), i.e. the width of
`features[12]`, and is the number implementation.txt §8.1 specifies.

NOTE on the 1024-dim alternative (flagged, not used): MobileNetV3's paper
design moves the final feature-expansion layer AFTER global average pooling
(CNN summaries §3, hand design (i)); in torchvision that expansion is
`classifier[0]` (Linear 576 -> 1024). Keeping it would give a 1024-dim frozen
feature. We take the 576-dim conv feature because (a) implementation.txt §8.1
specifies 576, (b) `classifier[0]` is a 1000-class-tuned layer, and (c) a
smaller feature dim is the honest "small backbone" for the RQ4 comparison. The
1024-dim variant is left as a documented alternative, not an assumption.

Stage discovery for in-block adapter placement
----------------------------------------------
`mobilenetv3_stage_paths` derives the placement sites PROGRAMMATICALLY rather
than hardcoding indices: it returns the LAST residual InvertedResidual block of
each channel-width group in `features`. Only residual blocks (stride 1 and
in_channels == out_channels) are eligible because the PARALLEL hook composes
`out + body(block_input)` and therefore requires block input and output to
share a shape. For MobileNetV3-Small this yields 4 stages, the direct analogue
of ResNet-18's layer1..layer4:

    features.3  (24ch)   features.6  (40ch)
    features.8  (48ch)   features.11 (96ch)

The block internals (1x1 expand -> KxK depthwise -> SE -> 1x1 project) are
exactly the sites the CNN paper summary (§8) names for adapter/LoRA insertion.
"""
from __future__ import annotations
from typing import List

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

#: Pooled conv-feature dim with `classifier` replaced by Identity (6 * 96).
MOBILENETV3_SMALL_FEATURE_DIM = 576


def build_frozen_mobilenetv3_small(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained MobileNetV3-Small, classifier removed, fully frozen.

    Args:
        pretrained: load IMAGENET1K_V1 weights (default). `False` builds the
            same architecture with random weights and NO download — used by the
            offline unit tests, never by an experiment config.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # features -> avgpool -> flatten -> Identity  =>  forward(x) is (B, 576).
    model.classifier = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def mobilenetv3_stage_paths(backbone: nn.Module) -> List[str]:
    """Dotted paths of the stage-final RESIDUAL blocks of `backbone.features`.

    A block is eligible when torchvision's InvertedResidual set
    `use_res_connect = (stride == 1 and in_channels == out_channels)`, which is
    exactly the condition under which block input and output share a shape (so
    both serial and parallel placement compose). Blocks are grouped by their
    output width and the LAST block of each width is taken, giving one insertion
    point per "stage" in the sense of MobileNetV3's layer table (Howard 2019,
    Table 2), ordered from shallow to deep.

    Returns e.g. ["features.3", "features.6", "features.8", "features.11"] for
    MobileNetV3-Small.
    """
    features = getattr(backbone, "features", None)
    if features is None:
        raise ValueError(
            "mobilenetv3_stage_paths expects a torchvision MobileNetV3 "
            f"(no .features attribute on {type(backbone).__name__})"
        )
    last_by_width: dict = {}
    width_order: List[int] = []
    for i, block in enumerate(features):
        if not bool(getattr(block, "use_res_connect", False)):
            continue
        width = getattr(block, "out_channels", None)
        if width is None:
            continue
        width = int(width)
        if width not in last_by_width:
            width_order.append(width)
        last_by_width[width] = f"features.{i}"
    paths = [last_by_width[w] for w in width_order]
    if not paths:
        raise ValueError(
            "found no residual InvertedResidual blocks in backbone.features — "
            "cannot derive placement stages. Set cfg.adapter.stage_paths "
            "explicitly to override."
        )
    return paths
