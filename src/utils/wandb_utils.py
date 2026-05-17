"""Thin wandb wrapper used by scripts/train.py and scripts/evaluate.py.

Step 3 (post-defence plan, action 3.1-3.5):
  - one entry point per run, project = "bpeft-thesis"
  - run name format: "{backbone}_{adapter}_{head}_{dataset}_{kshot}shot_seed{seed}"
  - disabled / offline modes are first-class so the same code runs in CI,
    on Colab with an API key, and on Colab without one.

The wrapper deliberately exposes a tiny surface (log, log_image, log_artifact,
finish). All wandb-specific imports live here so the rest of the codebase
never imports wandb directly.
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    import wandb
    _HAS_WANDB = True
except ImportError:  # pragma: no cover - wandb optional
    wandb = None  # type: ignore
    _HAS_WANDB = False


def make_run_name(cfg) -> str:
    """Spec format: {backbone}_{adapter}_{head}_{dataset}_{kshot}shot_seed{seed}."""
    return (
        f"{cfg.backbone.name}_{cfg.adapter.type}_{cfg.head.type}_"
        f"{cfg.dataset.name}_{int(cfg.dataset.k_shot)}shot_seed{int(cfg.seed)}"
    )


class WandbRun:
    """Wrap wandb.init/log/finish so callers don't have to special-case modes.

    Resolution order:
      1. disabled=True OR wandb not installed  ->  every method is a no-op.
      2. mode="offline"                        ->  wandb writes to ./wandb/, no upload.
      3. mode="online"                         ->  requires WANDB_API_KEY in env.

    `reinit=True` so multiple runs from the same process don't trip wandb's
    "run already in progress" guard (matters when the notebook drives both
    evidential + softmax in one Python session).
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict,
        mode: str = "online",
        disabled: bool = False,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        job_type: Optional[str] = None,
    ):
        self.disabled = bool(disabled) or not _HAS_WANDB
        self.run = None
        self.run_name = run_name
        self.mode = "disabled" if self.disabled else mode
        if self.disabled:
            return
        os.environ["WANDB_MODE"] = mode
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            group=group,
            tags=tags,
            job_type=job_type,
            reinit=True,
        )

    @property
    def url(self) -> Optional[str]:
        if self.disabled or self.run is None:
            return None
        return getattr(self.run, "url", None)

    def log(self, data: dict, step: Optional[int] = None) -> None:
        if self.disabled or self.run is None:
            return
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)

    def log_image(self, key: str, image_path: str, caption: str = "") -> None:
        if self.disabled or self.run is None:
            return
        wandb.log({key: wandb.Image(image_path, caption=caption)})

    def log_artifact(self, file_path: str, artifact_name: str,
                     artifact_type: str = "result") -> None:
        if self.disabled or self.run is None:
            return
        art = wandb.Artifact(artifact_name, type=artifact_type)
        art.add_file(file_path)
        self.run.log_artifact(art)

    def update_summary(self, data: dict) -> None:
        if self.disabled or self.run is None:
            return
        for k, v in data.items():
            wandb.run.summary[k] = v  # type: ignore[union-attr]

    def finish(self) -> None:
        if self.disabled or self.run is None:
            return
        wandb.finish()


__all__ = ["WandbRun", "make_run_name"]
