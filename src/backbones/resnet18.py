import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_frozen_resnet18() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
