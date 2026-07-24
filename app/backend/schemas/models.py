"""API schemas. These define the JSON contract; the frontend depends on these
field names, so treat them as a stable interface.

Mapping helpers (``from_*``) convert internal dataclasses (services layer) into
these response models, keeping the service layer free of API concerns.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.backend.services.detector import Detection, ClassScore
from app.backend.services.prototype_store import Product


# --- Products ---------------------------------------------------------------
class ProductOut(BaseModel):
    id: str
    name: str
    n_shots: int
    created_at: float
    thumbnail: str

    @classmethod
    def from_product(cls, p: Product) -> "ProductOut":
        return cls(**p.public())


class ProductListOut(BaseModel):
    products: list[ProductOut]
    count: int


# --- Detection --------------------------------------------------------------
class ClassScoreOut(BaseModel):
    product_id: str
    name: str
    similarity: float
    evidence: float
    probability: float
    softmax_probability: float

    @classmethod
    def from_score(cls, s: ClassScore) -> "ClassScoreOut":
        return cls(
            product_id=s.product_id,
            name=s.name,
            similarity=s.similarity,
            evidence=s.evidence,
            probability=s.probability,
            softmax_probability=s.softmax_probability,
        )


class DetectionOut(BaseModel):
    decision: str = Field(description="MATCH | REVIEW | UNKNOWN")
    predicted_id: str | None
    predicted_name: str | None
    confidence: float
    uncertainty: float
    softmax_confidence: float
    scores: list[ClassScoreOut]
    threshold: float
    inference_ms: float
    weights_status: str
    adapter_status: str

    @classmethod
    def from_detection(cls, d: Detection) -> "DetectionOut":
        return cls(
            decision=d.decision,
            predicted_id=d.predicted_id,
            predicted_name=d.predicted_name,
            confidence=d.confidence,
            uncertainty=d.uncertainty,
            softmax_confidence=d.softmax_confidence,
            scores=[ClassScoreOut.from_score(s) for s in d.scores],
            threshold=d.threshold,
            inference_ms=d.inference_ms,
            weights_status=d.weights_status,
            adapter_status=d.adapter_status,
        )


# --- Health / config --------------------------------------------------------
class HealthOut(BaseModel):
    status: str
    version: str
    weights_status: str
    device: str
    embedding_dim: int
    product_count: int
    adapter_status: str = Field(
        description='"trained" if the B-PEFT checkpoint is loaded, else "baseline"'
    )
    checkpoint_val_accuracy: float | None = None


class ConfigOut(BaseModel):
    uncertainty_threshold: float
    review_confidence_floor: float
    cosine_scale: float
    evidence_bias: float
    image_size: int
    min_shots: int
    max_shots: int
