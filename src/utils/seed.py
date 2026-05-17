"""Seeding and determinism helpers.

Step 3 reproducibility (action 3.6): same config -> byte-identical
metrics.json. The headline knobs are cudnn.deterministic, the
CUBLAS_WORKSPACE_CONFIG env var (needed for fully deterministic CUDA matmul
since CUDA 10.2), and torch.use_deterministic_algorithms.

We default-on "deterministic" mode so scripts/* are reproducible by
construction. Code paths that genuinely need non-deterministic kernels can
pass deterministic=False.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch CPU + CUDA, and flip deterministic flags.

    Determinism flags (only applied when deterministic=True):
      - cudnn.deterministic = True, cudnn.benchmark = False
      - torch.use_deterministic_algorithms(True) (warn_only=True so old
        non-deterministic kernels degrade to a warning, not a crash)
      - CUBLAS_WORKSPACE_CONFIG set so deterministic cublas matmul works
      - PYTHONHASHSEED pinned so dict ordering of str keys is stable
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch builds don't support the warn_only kwarg; the cudnn
        # flags above are still applied which is sufficient for our ops.
        pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
