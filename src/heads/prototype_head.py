"""Parameter-free prototype-similarity classifier for episodic meta-training.

Step 4 (Phase 2) introduces a head architecture deviation from proposal.txt §5B:
the trained Linear-then-Activation classifier becomes a parameter-free
PROTOTYPE head whose logits are produced by similarity between the adapter's
support-feature prototypes and the query features. This is the standard
CIFAR-FS / few-shot protocol (ProtoNet, R2D2, MetaOptNet).

The Dirichlet pipeline (softplus -> evidence -> alpha = evidence+1 -> p, S,
vacuity) is unchanged; it is applied OUTSIDE this head, by the loss /
evaluator. This file just emits a (Q, K) tensor of logits.

Two metric choices, selectable via cfg.head.prototype_metric:
  - "l2"     :  logit[q, k] = -||q - prototype_k||^2     (negative squared L2)
  - "cosine" :  logit[q, k] = cosine_similarity(q, prototype_k) * scale

L2 is the ProtoNet default and what Bertinetto 2019 uses. Cosine is gated
behind cfg for ablation.
"""
from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """Parameter-free; the only `init` arg is the similarity metric.

    forward(support_features, support_labels, query_features)
        -> query_logits of shape (Q, n_way)

    `support_labels` must be the EPISODE-LOCAL labels in range [0, n_way).
    n_way is inferred as (support_labels.max() + 1) at runtime so the
    same head works for 5-way at train time and 5-way at test time
    without re-instantiation.

    For evidential interpretation, the LOSS / EVALUATOR consumes these
    logits as:
        evidence = softplus(logits)
        alpha    = evidence + 1
    For softmax interpretation, the LOSS / EVALUATOR applies softmax.
    The head itself returns raw logits in both cases — same design as
    LinearHead.
    """

    SUPPORTED: tuple[str, ...] = ("l2", "cosine")

    def __init__(self, metric: Literal["l2", "cosine"] = "l2",
                 cosine_scale: float = 10.0):
        super().__init__()
        if metric not in self.SUPPORTED:
            raise ValueError(
                f"metric must be one of {self.SUPPORTED}, got {metric!r}"
            )
        self.metric = metric
        self.cosine_scale = float(cosine_scale)

    @staticmethod
    def _prototypes(support_features: torch.Tensor,
                    support_labels: torch.Tensor,
                    n_way: int) -> torch.Tensor:
        """Per-class mean over support features. Shape (n_way, D)."""
        D = support_features.shape[-1]
        protos = support_features.new_zeros((n_way, D))
        for k in range(n_way):
            mask = (support_labels == k)
            if not mask.any():
                raise ValueError(
                    f"Episode has no support examples for class {k}; "
                    f"check k_shot >= 1 and the sampler."
                )
            protos[k] = support_features[mask].mean(dim=0)
        return protos

    def forward(self, support_features: torch.Tensor,
                support_labels: torch.Tensor,
                query_features: torch.Tensor) -> torch.Tensor:
        """Return query logits of shape (Q, n_way)."""
        if support_features.dim() != 2 or query_features.dim() != 2:
            raise ValueError(
                "PrototypeHead expects 2-D feature tensors; got "
                f"support {support_features.shape}, query {query_features.shape}"
            )
        if support_labels.dim() != 1:
            raise ValueError(
                f"support_labels must be 1-D; got shape {support_labels.shape}"
            )

        n_way = int(support_labels.max().item()) + 1
        protos = self._prototypes(support_features, support_labels, n_way)

        if self.metric == "l2":
            # (Q, 1, D) - (1, K, D) -> (Q, K, D); ||.||^2 over D; negate.
            diff = query_features.unsqueeze(1) - protos.unsqueeze(0)
            return -(diff * diff).sum(dim=-1)

        # cosine
        q = F.normalize(query_features, p=2, dim=-1)
        p = F.normalize(protos,         p=2, dim=-1)
        return self.cosine_scale * (q @ p.t())
