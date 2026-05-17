import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..adapters import build_adapter
from ..heads import build_head


class BPEFTModel(nn.Module):
    """Frozen backbone -> Adapter -> Head.

    Output semantics depend on the head:
      - LinearHead    -> raw logits (use softmax CE loss at train time)
      - EvidentialHead -> non-negative evidence (use evidential MSE+KL loss)
    """
    def __init__(self, backbone: nn.Module, adapter: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.adapter(feats)
        return self.head(feats)

    def forward_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Bypass the frozen backbone when features are pre-computed."""
        feats = self.adapter(feats)
        return self.head(feats)


def build_model(cfg) -> BPEFTModel:
    """Construct a BPEFTModel from a ConfigDict (or plain dict)."""
    backbone = build_backbone(cfg["backbone"]["name"])
    dim = int(cfg["backbone"].get("feature_dim", 512))
    adapter = build_adapter(dict(cfg["adapter"]), dim=dim)
    head = build_head(dict(cfg["head"]), in_dim=dim,
                      num_classes=int(cfg["dataset"]["n_way"]))
    return BPEFTModel(backbone, adapter, head)
