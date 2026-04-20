import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_frozen_resnet18() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


if __name__ == "__main__":
    backbone = build_frozen_resnet18()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy)
    assert out.shape == (2, 512), f"Bad shape: {out.shape}"
    frozen = all(not p.requires_grad for p in backbone.parameters())
    assert frozen, "Some parameters are not frozen!"
    print(f"Output shape: {out.shape}  |  All params frozen: {frozen}")
    print("OK")
