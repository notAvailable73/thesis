"""PrototypeStore — the registry of enrolled products.

Each product is the thesis's *parameter-free prototype*: the mean of its
reference-image embeddings, re-normalised to the unit sphere. Enrolling a
product is therefore just averaging — no gradient training. This is what keeps
the app cheap enough to run on low-spec hardware.

Persistence layout (under ``data_dir``):
    registry.json                metadata for all products (+ a small thumbnail)
    prototypes/<id>.npy          the (D,) float32 prototype vector

All mutations are guarded by a lock and written atomically (tmp + os.replace),
so a crash mid-write can't corrupt the registry.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.backend.core.config import Settings
from app.backend.core.logging import get_logger

log = get_logger("services.prototypes")

_THUMB_PX = 128


@dataclass
class Product:
    id: str
    name: str
    n_shots: int
    created_at: float
    thumbnail: str  # data URI (small JPEG) for the UI

    def public(self) -> dict:
        return asdict(self)


def _make_thumbnail(raw: bytes) -> str:
    """Return a small square-ish JPEG as a base64 data URI, for the gallery."""
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((_THUMB_PX, _THUMB_PX))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:  # noqa: BLE001 - thumbnail is cosmetic, never fatal
        return ""


class PrototypeStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._products: dict[str, Product] = {}
        self._vectors: dict[str, np.ndarray] = {}

        settings.prototypes_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # --- Persistence --------------------------------------------------------
    def _load(self) -> None:
        reg = self._settings.registry_file
        if not reg.exists():
            log.info("No registry found; starting empty.")
            return
        try:
            meta = json.loads(reg.read_text())
        except Exception as exc:  # noqa: BLE001
            log.error("Registry unreadable (%s); starting empty.", exc)
            return
        for item in meta.get("products", []):
            vec_path = self._settings.prototypes_dir / f"{item['id']}.npy"
            if not vec_path.exists():
                log.warning("Prototype vector missing for %s; skipping.", item["id"])
                continue
            self._products[item["id"]] = Product(**item)
            self._vectors[item["id"]] = np.load(vec_path).astype(np.float32)
        log.info("Loaded %d product(s) from registry.", len(self._products))

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)

    def _persist_registry(self) -> None:
        payload = {"products": [p.public() for p in self._products.values()]}
        self._atomic_write_json(self._settings.registry_file, payload)

    # --- Queries ------------------------------------------------------------
    def list_products(self) -> list[Product]:
        with self._lock:
            return sorted(self._products.values(), key=lambda p: p.created_at)

    def count(self) -> int:
        with self._lock:
            return len(self._products)

    def get(self, product_id: str) -> Product | None:
        with self._lock:
            return self._products.get(product_id)

    def prototype_matrix(self) -> tuple[list[Product], np.ndarray]:
        """Return (ordered products, ``(K, D)`` prototype matrix).

        Order is stable (by creation time) so class indices are consistent
        between calls — the detector relies on this alignment.
        """
        with self._lock:
            products = self.list_products()
            if not products:
                return [], np.zeros((0, self._settings.feature_dim), dtype=np.float32)
            matrix = np.stack([self._vectors[p.id] for p in products])
            return products, matrix

    # --- Mutations ----------------------------------------------------------
    def register(self, name: str, embeddings: np.ndarray, first_image: bytes) -> Product:
        """Create a product from its reference embeddings.

        ``embeddings`` is ``(n_shots, D)`` L2-normalised. The prototype is the
        mean, re-normalised so it stays a unit vector (cosine-comparable).
        """
        name = name.strip()
        if not name:
            raise ValueError("Product name must not be empty.")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Need at least one reference embedding.")

        proto = embeddings.mean(axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        proto = proto.astype(np.float32)

        product = Product(
            id=uuid.uuid4().hex[:12],
            name=name,
            n_shots=int(embeddings.shape[0]),
            created_at=time.time(),
            thumbnail=_make_thumbnail(first_image),
        )
        with self._lock:
            self._products[product.id] = product
            self._vectors[product.id] = proto
            np.save(self._settings.prototypes_dir / f"{product.id}.npy", proto)
            self._persist_registry()
        log.info("Registered product %r (%s, %d shots).",
                 product.name, product.id, product.n_shots)
        return product

    def delete(self, product_id: str) -> bool:
        with self._lock:
            if product_id not in self._products:
                return False
            self._products.pop(product_id)
            self._vectors.pop(product_id, None)
            vec_path = self._settings.prototypes_dir / f"{product_id}.npy"
            if vec_path.exists():
                vec_path.unlink()
            self._persist_registry()
        log.info("Deleted product %s.", product_id)
        return True
