"""Application configuration.

Single source of truth for tunable knobs. Every value can be overridden with an
environment variable prefixed ``SENTINEL_`` so the same code runs unchanged in
dev, a demo laptop, or CI. Nothing else in the codebase should read
``os.environ`` directly — depend on :data:`settings` instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _env(key: str, default: str) -> str:
    return os.environ.get(f"SENTINEL_{key}", default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


# Repo root = two levels above this file's app/backend/core/ location -> app/ -> repo.
_APP_DIR = Path(__file__).resolve().parents[2]           # .../app
_REPO_ROOT = _APP_DIR.parent                             # repo root


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Instantiate once as :data:`settings`."""

    # --- Server -------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    reload: bool = field(default_factory=lambda: _env("RELOAD", "0") == "1")

    # --- Model --------------------------------------------------------------
    # 224 is what ImageNet ResNet18 expects; feature_dim is its penultimate width.
    image_size: int = field(default_factory=lambda: _env_int("IMAGE_SIZE", 224))
    feature_dim: int = 512
    # If pretrained weights aren't cached and can't be fetched, fall back to
    # random init so the app still boots (pipeline works, accuracy degraded).
    allow_random_weights: bool = field(
        default_factory=lambda: _env("ALLOW_RANDOM_WEIGHTS", "1") == "1"
    )

    # --- Detector / evidential decision policy ------------------------------
    # Cosine similarity -> evidence sharpness. Mirrors the thesis Step-4
    # PrototypeHead cosine_scale.
    cosine_scale: float = field(default_factory=lambda: _env_float("COSINE_SCALE", 10.0))
    # Additive evidence bias (thesis evidence affine). Negative re-centres the
    # inflated cosine baseline of a frozen backbone so foreign objects read as
    # OOD. Zero crossing ≈ -bias/scale ≈ 0.5 similarity with the defaults.
    evidence_bias: float = field(default_factory=lambda: _env_float("EVIDENCE_BIAS", -5.0))
    # Vacuity (u = K/S, Sensoy 2018) at/above which an item is declared UNKNOWN
    # (out-of-distribution) and routed to manual inspection.
    uncertainty_threshold: float = field(
        default_factory=lambda: _env_float("UNCERTAINTY_THRESHOLD", 0.30)
    )
    # Top-1 belief below this (but not OOD) triggers a REVIEW rather than MATCH.
    review_confidence_floor: float = field(
        default_factory=lambda: _env_float("REVIEW_CONFIDENCE_FLOOR", 0.60)
    )
    min_shots: int = field(default_factory=lambda: _env_int("MIN_SHOTS", 1))
    max_shots: int = field(default_factory=lambda: _env_int("MAX_SHOTS", 10))

    # --- Trained B-PEFT adapter (optional) -----------------------------------
    # If present, the app loads this Phase-2 episodic checkpoint (trained
    # bottleneck adapter + learnable evidence affine) and runs it after the
    # frozen backbone, instead of the untrained backbone-only pipeline. See
    # app/backend/ml/adapter.py. Missing file -> silent fallback, app still
    # boots.
    checkpoint_path: Path = field(
        default_factory=lambda: Path(_env(
            "CHECKPOINT_PATH",
            str(_REPO_ROOT / "checkpoints"
                / "model_phase2_bottleneck_prototype-evidential_seed42.pt"),
        ))
    )
    # Step-6 placement checkpoints don't record serial-vs-parallel in their
    # own metadata (both produce identical weight shapes); this is the
    # fallback used when the training config's filename can't be recovered
    # from the checkpoint either. See app/backend/ml/adapter.py:load_trained_adapter.
    adapter_placement_hint: str = field(
        default_factory=lambda: _env("ADAPTER_PLACEMENT", "parallel")
    )

    # --- Storage ------------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(_APP_DIR / "data")))
    )

    # --- Static frontend ----------------------------------------------------
    frontend_dir: Path = field(default_factory=lambda: _APP_DIR / "frontend")

    @property
    def prototypes_dir(self) -> Path:
        return self.data_dir / "prototypes"

    @property
    def registry_file(self) -> Path:
        return self.data_dir / "registry.json"

    def as_public_dict(self) -> dict[str, Any]:
        """Serialisable view for the /api/health and /api/config endpoints."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d


settings = Settings()
