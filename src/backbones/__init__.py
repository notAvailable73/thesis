from .resnet18 import build_frozen_resnet18


def build_backbone(name: str):
    if name == "resnet18":
        return build_frozen_resnet18()
    raise ValueError(f"Unknown backbone: {name}")


__all__ = ["build_backbone", "build_frozen_resnet18"]
