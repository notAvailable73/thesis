"""DetectorService — the inspection decision for one query image.

Combines the three lower pieces:
    ModelService     query image  -> embedding
    PrototypeStore   enrolled products -> prototype matrix
    ml.evidential    similarities -> beliefs + vacuity

and applies the decision policy that makes this an *industrial* tool rather than
a bare classifier:

    UNKNOWN  vacuity >= uncertainty_threshold   -> out-of-distribution / foreign
             object / anomaly. Route to manual inspection. (The safety feature.)
    REVIEW   known but top-1 belief < floor     -> ambiguous; a human should confirm.
    MATCH    confident and in-distribution      -> auto-accept as that product.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.backend.core.config import Settings
from app.backend.core.logging import get_logger
from app.backend.ml import evidential
from app.backend.services.model_service import ModelService
from app.backend.services.prototype_store import PrototypeStore

log = get_logger("services.detector")


@dataclass
class ClassScore:
    product_id: str
    name: str
    similarity: float
    evidence: float
    probability: float           # evidential Dirichlet mean
    softmax_probability: float   # overconfident baseline


@dataclass
class Detection:
    decision: str                # MATCH | REVIEW | UNKNOWN
    predicted_id: str | None
    predicted_name: str | None
    confidence: float            # evidential top-1 belief/prob
    uncertainty: float           # vacuity u
    softmax_confidence: float    # baseline top-1 (for the honest-AI contrast)
    scores: list[ClassScore]
    threshold: float
    inference_ms: float
    weights_status: str
    adapter_status: str           # "trained" (B-PEFT checkpoint loaded) | "baseline"


class NoProductsRegistered(Exception):
    """Raised when a detection is attempted with an empty registry."""


class DetectorService:
    def __init__(
        self,
        settings: Settings,
        model: ModelService,
        store: PrototypeStore,
    ) -> None:
        self._settings = settings
        self._model = model
        self._store = store

    def detect(self, image: bytes) -> Detection:
        products, matrix = self._store.prototype_matrix()
        if not products:
            raise NoProductsRegistered(
                "No products enrolled yet — register at least one product first."
            )

        t0 = time.time()
        query = self._model.embed_one(image)          # (D,), unit norm
        # Both sides unit-normalised => dot product == cosine similarity.
        similarities = matrix @ query                 # (K,)
        result = evidential.evaluate(
            similarities,
            self._model.evidence_scale,
            self._model.evidence_bias,
        )
        inference_ms = (time.time() - t0) * 1000.0

        scores = [
            ClassScore(
                product_id=p.id,
                name=p.name,
                similarity=float(similarities[i]),
                evidence=float(result.evidence[i]),
                probability=float(result.probabilities[i]),
                softmax_probability=float(result.softmax_probabilities[i]),
            )
            for i, p in enumerate(products)
        ]
        scores.sort(key=lambda s: s.probability, reverse=True)

        decision, pred = self._decide(result, products)
        return Detection(
            decision=decision,
            predicted_id=pred.id if pred else None,
            predicted_name=pred.name if pred else None,
            confidence=result.confidence,
            uncertainty=result.uncertainty,
            softmax_confidence=result.softmax_confidence,
            scores=scores,
            threshold=self._settings.uncertainty_threshold,
            inference_ms=inference_ms,
            weights_status=self._model.weights_status,
            adapter_status=self._model.adapter_status,
        )

    def _decide(self, result: evidential.EvidentialResult, products):
        pred = products[result.pred_index]
        if result.uncertainty >= self._settings.uncertainty_threshold:
            return "UNKNOWN", None
        if result.confidence < self._settings.review_confidence_floor:
            return "REVIEW", pred
        return "MATCH", pred
