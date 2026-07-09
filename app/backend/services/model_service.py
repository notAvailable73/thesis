"""ModelService — owns the frozen backbone and turns images into embeddings.

One instance per process (created at startup). Thread-safety: PyTorch CPU
inference under ``no_grad`` is re-entrant for reads, but we guard the forward
pass with a lock so concurrent requests on the dev server behave predictably.
"""
from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np
import torch

from app.backend.core.config import Settings
from app.backend.core.logging import get_logger
from app.backend.ml.adapter import load_trained_adapter
from app.backend.ml.backbone import FrozenResNet18
from app.backend.ml.transforms import ImagePreprocessor

log = get_logger("services.model")


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = torch.device("cpu")  # low-spec target; CPU is the contract
        self._lock = threading.Lock()

        t0 = time.time()
        self._preprocess = ImagePreprocessor(settings.image_size)
        self._backbone = FrozenResNet18(settings.allow_random_weights).to(self._device)
        self._adapter_bundle = load_trained_adapter(
            settings.checkpoint_path, dim=self._backbone.output_dim,
        )
        log.info(
            "ModelService ready in %.2fs  (weights=%s, dim=%d, device=%s, adapter=%s)",
            time.time() - t0,
            self._backbone.weights_status,
            self._backbone.output_dim,
            self._device,
            self.adapter_status,
        )

    # --- Introspection ------------------------------------------------------
    @property
    def weights_status(self) -> str:
        return self._backbone.weights_status

    @property
    def embedding_dim(self) -> int:
        return self._backbone.output_dim

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def adapter_status(self) -> str:
        """``"trained"`` if the Phase-2 B-PEFT checkpoint loaded, else
        ``"baseline"`` (untrained backbone-only, fixed evidence constants)."""
        return "trained" if self._adapter_bundle is not None else "baseline"

    @property
    def evidence_scale(self) -> float:
        """Cosine-similarity -> evidence sharpness. The trained checkpoint's
        learned value if loaded, else the static config default."""
        if self._adapter_bundle is not None:
            return self._adapter_bundle.evidence_scale
        return self._settings.cosine_scale

    @property
    def evidence_bias(self) -> float:
        """Evidence-affine bias. The trained checkpoint's learned value if
        loaded, else the static config default."""
        if self._adapter_bundle is not None:
            return self._adapter_bundle.evidence_bias
        return self._settings.evidence_bias

    @property
    def checkpoint_val_accuracy(self) -> float | None:
        if self._adapter_bundle is None:
            return None
        return self._adapter_bundle.best_val_acc

    # --- Core: bytes -> unit-normalised embedding ---------------------------
    def embed_many(self, images: Sequence[bytes]) -> np.ndarray:
        """Embed a batch of raw image bytes. Returns ``(N, D)`` L2-normalised.

        L2-normalisation here means downstream cosine similarity is just a dot
        product, and prototype averaging stays on the unit hypersphere. When a
        trained adapter is loaded, it runs between the frozen backbone and
        normalisation -- exactly the ``backbone -> adapter -> (cosine) head``
        order used by ``BPEFTModel.adapter_features`` at training time.
        """
        if not images:
            raise ValueError("No images provided to embed.")
        tensors = [self._preprocess(raw) for raw in images]
        batch = torch.stack(tensors).to(self._device)

        with self._lock, torch.no_grad():
            feats = self._backbone(batch)                       # (N, D)
            if self._adapter_bundle is not None:
                feats = self._adapter_bundle.adapter(feats)
        feats = torch.nn.functional.normalize(feats, dim=1)     # unit vectors
        return feats.cpu().numpy().astype(np.float32)

    def embed_one(self, image: bytes) -> np.ndarray:
        """Embed a single image. Returns ``(D,)`` L2-normalised."""
        return self.embed_many([image])[0]
