"""Composition root — constructs and wires the services exactly once.

Keeping construction in one place (rather than module-level singletons scattered
about) makes the dependency graph explicit and the app testable: a test can
build a container with fakes and attach it to ``app.state.container``.
"""
from __future__ import annotations

from app.backend.core.config import Settings
from app.backend.services.detector import DetectorService
from app.backend.services.model_service import ModelService
from app.backend.services.prototype_store import PrototypeStore


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Order matters: model + store are independent; detector depends on both.
        self.model = ModelService(settings)
        self.store = PrototypeStore(settings)
        self.detector = DetectorService(settings, self.model, self.store)
