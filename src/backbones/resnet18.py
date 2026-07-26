import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_frozen_resnet18(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet-18, fc removed, fully frozen -> (B, 512).

    `pretrained=False` builds the same architecture with random weights and no
    download (offline unit tests only). The default is unchanged from Steps 1-7.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
