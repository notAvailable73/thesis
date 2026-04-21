import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_frozen_resnet18
from .adapter import BottleneckAdapter
from .lora import LoRAAdapter


def build_adapter(adapter_type: str, dim: int, rank: int, lora_alpha: float = None):
    if adapter_type == "bottleneck":
        return BottleneckAdapter(dim, rank)
    if adapter_type == "lora":
        return LoRAAdapter(dim, rank, alpha=lora_alpha)
    raise ValueError(f"Unknown adapter_type: {adapter_type}")


class BPEFTModel(nn.Module):
    """
    Full model: frozen ResNet18 -> Adapter -> Linear head.

    adapter_type: "bottleneck" (Houlsby 2019) or "lora" (Hu 2021)

    evidential mode: Softplus on logits -> non-negative evidence (alpha = evidence + 1)
                     Trained with CE loss; Dirichlet probs used at evaluation time.
    softmax mode:    raw logits (softmax applied inside loss)
    """
    def __init__(self, num_classes: int, feature_dim: int = 512,
                 adapter_rank: int = 16, mode: str = "evidential",
                 adapter_type: str = "bottleneck", lora_alpha: float = None):
        super().__init__()
        assert mode in ("evidential", "softmax")
        self.mode = mode
        self.adapter_type = adapter_type
        self.backbone = build_frozen_resnet18()
        self.adapter = build_adapter(adapter_type, feature_dim, adapter_rank, lora_alpha)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.adapter(feats)
        logits = self.head(feats)
        if self.mode == "evidential":
            return F.softplus(logits)  # always > 0, smooth gradients (no dead neurons)
        return logits


if __name__ == "__main__":
    from src.utils import count_trainable_params

    for atype in ("bottleneck", "lora"):
        model = BPEFTModel(num_classes=5, mode="evidential", adapter_type=atype)
        dummy = torch.randn(8, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (8, 5), f"Bad shape: {out.shape}"
        assert (out >= 0).all(), "Evidential output must be non-negative!"
        n = count_trainable_params(model)
        print(f"[{atype}] shape={out.shape}  params={n:,}")
    print("OK")
