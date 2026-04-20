import torch
import torch.nn as nn
from .backbone import build_frozen_resnet18
from .adapter import BottleneckAdapter


class BPEFTModel(nn.Module):
    """
    Full model: frozen ResNet18 -> Bottleneck Adapter -> Linear head.

    evidential mode: ReLU on logits -> non-negative evidence (alpha = evidence + 1 in loss)
    softmax mode:    raw logits (softmax applied inside loss)
    """
    def __init__(self, num_classes: int, feature_dim: int = 512,
                 adapter_rank: int = 16, mode: str = "evidential"):
        super().__init__()
        assert mode in ("evidential", "softmax")
        self.mode = mode
        self.backbone = build_frozen_resnet18()
        self.adapter = BottleneckAdapter(feature_dim, adapter_rank)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.adapter(feats)
        logits = self.head(feats)
        if self.mode == "evidential":
            return torch.relu(logits)
        return logits


if __name__ == "__main__":
    from src.utils import count_trainable_params

    model = BPEFTModel(num_classes=5, mode="evidential")
    dummy = torch.randn(8, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (8, 5), f"Bad shape: {out.shape}"
    assert (out >= 0).all(), "Evidential output must be non-negative!"
    n = count_trainable_params(model)
    print(f"Output shape: {out.shape}")
    print(f"All outputs >= 0: True")
    print(f"Trainable params: {n:,}  (expect < 20,000)")
    print("OK")
