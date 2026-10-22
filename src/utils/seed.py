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


#: Per-seed shift of the training-episode stream when
#: `trainer.vary_train_stream_with_seed` is on (TMLR plan A6). Seed 42 maps to
#: shift 0, so it reproduces every pre-existing run exactly.
TRAIN_STREAM_STRIDE = 100_000
_TRAIN_STREAM_BASE_SEED = 42
#: Seed ranges reserved by the frozen episode files; the training stream must
#: never draw an episode seed from them.
_RESERVED_SEED_RANGES = ((0, 600), (10_000, 10_100))   # TEST, VAL (half-open)


def effective_train_seed_offset(cfg) -> int:
    """First episode seed of the training stream for this config.

    Default (flag absent or false): `trainer.train_seed_offset`, unchanged from
    Step 4 onwards, so every existing config keeps its stream byte-for-byte.
    With `trainer.vary_train_stream_with_seed: true` the stream is shifted by
    TRAIN_STREAM_STRIDE * (seed - 42), so extra seeds vary the episode order and
    not only the adapter initialisation.

    Raises ValueError for seed < 42 under the flag, or if the stream
    [offset, offset + num_epochs * episodes_per_epoch) would touch the TEST or
    VAL seed ranges.
    """
    trainer = cfg["trainer"]
    offset = int(trainer["train_seed_offset"])
    if bool(trainer.get("vary_train_stream_with_seed", False)):
        seed = int(cfg["seed"])
        if seed < _TRAIN_STREAM_BASE_SEED:
            raise ValueError(
                f"vary_train_stream_with_seed needs seed >= {_TRAIN_STREAM_BASE_SEED}, "
                f"got {seed}")
        offset += TRAIN_STREAM_STRIDE * (seed - _TRAIN_STREAM_BASE_SEED)
    end = offset + int(trainer.get("num_epochs", 30)) * int(trainer.get("episodes_per_epoch", 100))
    for lo, hi in _RESERVED_SEED_RANGES:
        if offset < hi and lo < end:
            raise ValueError(
                f"training stream seeds [{offset}, {end}) overlap the reserved "
                f"frozen range [{lo}, {hi})")
    return offset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
