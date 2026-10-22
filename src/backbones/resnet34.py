"""ResNet-34 frozen backbone (TMLR plan E2).

Same feature contract as build_frozen_resnet18: fc -> Identity, every
parameter frozen, eval mode, forward(x) -> (B, 512). Same basic-block family
and stage widths (64/128/256/512) as ResNet-18 with 3/4/6/3 blocks per stage
(He et al. 2016, Table 1; PAPER SUMMARIES/CNN_paper_summaries.txt), so every
adapter budget matches ResNet-18's exactly. The existing placement and LoRA
helpers already resolve its sites (layer1.2 ... layer4.2; layer4.0.downsample.0).
"""
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


def build_frozen_resnet34(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet-34, fc removed, fully frozen -> (B, 512).

    `pretrained=False` builds the same architecture with random weights and no
    download (offline unit tests only).
    """
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet34(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
