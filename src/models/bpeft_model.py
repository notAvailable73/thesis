"""Frozen backbone -> trainable Adapter -> Head.

Step 1-3 used LinearHead / EvidentialHead (parameterized) and the model's
`forward(x)` was the only inference path.

Step 4 (Phase 2) adds a PROTOTYPE head_type: a parameter-free
prototype-similarity classifier that consumes adapter features for
support+query in a single episode. The added entry point
`forward_proto(support_x, support_y, query_x)` runs:
    backbone(both)  ->  adapter(both)  ->  head(support_feats, support_y,
                                                 query_feats)
and returns query logits of shape (Q, n_way). The Dirichlet / softmax
interpretation of those logits stays in the loss/evaluator, same design
as the trained heads.

Both legacy entry points (`forward`, `forward_from_features`) are
preserved so the Step 1-3 reproduction tests and the legacy
single-episode evaluator keep working unchanged.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..adapters import build_adapter
from ..heads import build_head, PrototypeHead, ScaledPrototypeHead

_PROTOTYPE_HEAD_TYPES = (PrototypeHead, ScaledPrototypeHead)


class BPEFTModel(nn.Module):
    """Frozen backbone -> Adapter -> Head.

    Output semantics depend on the head:
      - LinearHead     -> raw logits (use softmax CE loss at train time)
      - EvidentialHead -> non-negative evidence (use evidential MSE+KL)
      - PrototypeHead  -> raw similarity logits; episode-conditional;
                          consumed via .forward_proto(support_x, support_y,
                                                       query_x)
    """

    def __init__(self, backbone: nn.Module, adapter: nn.Module,
                 head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    # ------------------------------------------------------------------
    # Step 1-3 legacy entry points (single-episode trainer; per-episode
    # fine-tune evaluator).
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.adapter(feats)
        return self.head(feats)

    def forward_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Bypass the frozen backbone when features are pre-computed."""
        feats = self.adapter(feats)
        return self.head(feats)

    # ------------------------------------------------------------------
    # Step 4 episodic entry point.
    # ------------------------------------------------------------------
    def adapter_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen backbone -> trainable adapter. Returns (B, D)."""
        with torch.no_grad():
            feats = self.backbone(x)
        return self.adapter(feats)

    def adapter_features_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Like adapter_features() but starting from pre-computed backbone
        features (skips the frozen backbone). Used when the trainer pre-
        caches backbone features once per episode."""
        return self.adapter(feats)

    def forward_proto(self, support_x: torch.Tensor,
                      support_y: torch.Tensor,
                      query_x: torch.Tensor) -> torch.Tensor:
        """Episodic forward through frozen backbone + trainable adapter +
        parameter-free prototype head. Returns query logits (Q, n_way).

        Raises if the model was not built with a PrototypeHead.
        """
        if not isinstance(self.head, _PROTOTYPE_HEAD_TYPES):
            raise TypeError(
                "forward_proto() requires a PrototypeHead; got "
                f"{type(self.head).__name__}. Check cfg.head.type == 'prototype'."
            )
        sf = self.adapter_features(support_x)
        qf = self.adapter_features(query_x)
        return self.head(sf, support_y, qf)

    def forward_proto_from_features(self, support_feats: torch.Tensor,
                                    support_y: torch.Tensor,
                                    query_feats: torch.Tensor) -> torch.Tensor:
        """Same as forward_proto() but starts from pre-computed BACKBONE
        features (skips the frozen backbone). The adapter is still applied.
        """
        if not isinstance(self.head, _PROTOTYPE_HEAD_TYPES):
            raise TypeError(
                "forward_proto_from_features() requires a PrototypeHead; got "
                f"{type(self.head).__name__}."
            )
        sf = self.adapter_features_from_features(support_feats)
        qf = self.adapter_features_from_features(query_feats)
        return self.head(sf, support_y, qf)


def build_model(cfg) -> BPEFTModel:
    """Construct a BPEFTModel from a ConfigDict (or plain dict).

    cfg.head.type ∈ {softmax, evidential, prototype}:
      - softmax / evidential land on LinearHead / EvidentialHead (Step 1-3).
      - prototype lands on the parameter-free PrototypeHead (Step 4+).
    """
    backbone = build_backbone(cfg["backbone"]["name"])
    dim = int(cfg["backbone"].get("feature_dim", 512))
    adapter = build_adapter(dict(cfg["adapter"]), dim=dim)
    head = build_head(
        dict(cfg["head"]),
        in_dim=dim,
        num_classes=int(cfg["dataset"]["n_way"]),
    )
    return BPEFTModel(backbone, adapter, head)
