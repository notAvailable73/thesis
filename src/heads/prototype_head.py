"""Prototype-similarity classifiers for episodic meta-training.

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
import math
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


class ScaledPrototypeHead(nn.Module):
    """Prototype head with learnable affine on instance-standardized logits.

    Wraps PrototypeHead and adds two learnable scalars:
        log_tau  : log of a positive scale factor  (init → tau = 1.0)
        bias     : additive offset to all classes  (init → 0.0)

    The forward pass:
        1. Compute raw L2/cosine prototype-similarity logits  (Q, K).
        2. Instance-standardize over the K-class dimension per query
           sample:  normed = (logits - mean) / std.clamp(1e-6)
           This makes initialization scale-invariant — raw L2 distances
           for 512-dim features can be in [-50000, 0], which maps
           softplus to ~0; after standardizing they are in a fixed range
           where softplus gives non-zero evidence regardless of backbone
           feature scale.
        3. Return  tau * normed + bias,  where tau = exp(log_tau) > 0.

    The evidential loss/evaluator then applies softplus to this output
    to get evidence.  With tau=1, bias=0 at init, the nearest class has
    a positive standardized score → evidence > 0 → gradients flow
    from step 1 of training.

    Used when head.type == "prototype" and head.interpretation == "evidential"
    (action 4.21 of the Step 4 remediation).
    """

    def __init__(self, metric: Literal["l2", "cosine"] = "l2",
                 cosine_scale: float = 10.0):
        super().__init__()
        self._inner = PrototypeHead(metric=metric, cosine_scale=cosine_scale)
        # log_tau = 0  →  tau = exp(0) = 1.0
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.bias    = nn.Parameter(torch.zeros(1))

    def forward(self, support_features: torch.Tensor,
                support_labels: torch.Tensor,
                query_features: torch.Tensor) -> torch.Tensor:
        """Return scaled query logits of shape (Q, n_way)."""
        logits = self._inner(support_features, support_labels, query_features)
        # Standardize over the n_way class dimension, per query sample.
        mu    = logits.mean(dim=-1, keepdim=True)
        sigma = logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
        normed = (logits - mu) / sigma
        tau = self.log_tau.exp()
        return tau * normed + self.bias
