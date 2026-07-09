"""Evidential (Dirichlet) uncertainty over cosine-prototype similarities.

This is the mathematical heart of the "honest AI" behaviour, following
Sensoy et al. (2018), *Evidential Deep Learning to Quantify Classification
Uncertainty*, adapted to the thesis Step-4 cosine PrototypeHead.

Pipeline, per query embedding vs. K class prototypes (all L2-normalised):

    sim_k   = cosine(query, prototype_k)            in [-1, 1]
    e_k     = softplus(scale * sim_k + bias)        >= 0   (evidence for class k)
    alpha_k = e_k + 1                               Dirichlet parameters
    S       = sum_k alpha_k = sum_k e_k + K         Dirichlet strength
    p_k     = alpha_k / S                           expected class probability
    b_k     = e_k / S                               belief mass for class k
    u       = K / S                                 *vacuity* (uncertainty mass)

Key property: ``sum_k b_k + u = 1``. When the query matches nothing (all sims
low), evidence -> 0, so S -> K and u -> 1: the model says "I don't know" instead
of forcing a confident label. High ``u`` ⇒ out-of-distribution.

The affine ``scale`` and ``bias`` mirror the thesis Step-4 "learnable evidence
affine". They matter in practice because a frozen ImageNet backbone gives
*inflated* cosine similarities (unrelated images still score ~0.5), so without a
negative ``bias`` to re-centre the operating point, evidence never drops and OOD
items are never flagged. ``bias`` shifts the similarity at which evidence crosses
zero to roughly ``-bias / scale``.

For contrast we also compute a plain **softmax** over the same scaled
similarities. Softmax has no "none of the above" outlet, so it stays
overconfident on unknown items — that gap is the thesis's whole point, and the
UI shows the two side by side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable softplus: log(1 + e^x) = max(x,0) + log(1 + e^-|x|)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass(frozen=True)
class EvidentialResult:
    """Full evidential + softmax breakdown for one query against K classes."""

    probabilities: np.ndarray      # (K,) Dirichlet mean p_k, sums to 1
    beliefs: np.ndarray            # (K,) belief mass b_k
    evidence: np.ndarray           # (K,) non-negative evidence e_k
    uncertainty: float             # vacuity u in [0, 1]
    pred_index: int                # argmax class
    confidence: float              # p_k at pred_index
    softmax_probabilities: np.ndarray  # (K,) baseline, sums to 1
    softmax_confidence: float          # max softmax prob (the "overconfident" number)

    @property
    def num_classes(self) -> int:
        return int(self.probabilities.shape[0])


def evaluate(similarities: np.ndarray, scale: float, bias: float = 0.0) -> EvidentialResult:
    """Turn cosine similarities into the full evidential result.

    Args:
        similarities: ``(K,)`` cosine similarities in [-1, 1], one per class.
        scale: sharpness multiplier (thesis ``cosine_scale``).
        bias: additive offset before softplus (thesis evidence affine). Negative
            values re-centre so unrelated items yield ~0 evidence.
    """
    sims = np.asarray(similarities, dtype=np.float64).reshape(-1)
    if sims.size == 0:
        raise ValueError("Need at least one class prototype to evaluate.")

    k = sims.size
    evidence = _softplus(scale * sims + bias)   # (K,)
    alpha = evidence + 1.0
    strength = float(alpha.sum())               # S
    probabilities = alpha / strength            # expected Dirichlet mean
    beliefs = evidence / strength
    uncertainty = k / strength                  # vacuity

    pred_index = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_index])

    softmax_probabilities = _softmax(scale * sims + bias)
    softmax_confidence = float(np.max(softmax_probabilities))

    return EvidentialResult(
        probabilities=probabilities,
        beliefs=beliefs,
        evidence=evidence,
        uncertainty=float(uncertainty),
        pred_index=pred_index,
        confidence=confidence,
        softmax_probabilities=softmax_probabilities,
        softmax_confidence=softmax_confidence,
    )
