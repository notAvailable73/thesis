"""MobileNetV2 frozen backbone (TMLR plan E2).

Second member of the inverted-residual family, next to MobileNetV3-Small
(Sandler et al. 2018; PAPER SUMMARIES/CNN_paper_summaries.txt). torchvision's
forward is features -> adaptive avg pool -> flatten -> classifier, so replacing
`classifier` with nn.Identity() returns the pooled (B, 1280) feature, i.e.
after the final 1x1 conv 320 -> 1280 (features.18).

Deliberate difference from our MobileNetV3-Small, reported in the paper:
MobileNetV3-Small's 576-d feature is taken BEFORE its final expansion
(that expansion sits in the classifier there), whereas MobileNetV2's 1280-d
expansion is part of `features`, so it is included here.

Placement sites come from the generic inverted-residual helpers
(`resolve_stage_paths` -> features.3/6/10/13/16, five width groups 24/32/64/96/160;
`default_lora_targets` -> features.17.conv.2, 960 -> 320).
"""
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

#: Pooled feature dim with `classifier` replaced by Identity.
MOBILENETV2_FEATURE_DIM = 1280


def build_frozen_mobilenetv2(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained MobileNetV2, classifier removed, fully frozen -> (B, 1280).

    `pretrained=False` builds the same architecture with random weights and no
    download (offline unit tests only).
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
