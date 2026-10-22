"""TMLR plan Task 4 — seed-dependent training-episode stream.

`trainer.vary_train_stream_with_seed` (default false) shifts the training
episode stream by 100_000 per seed above 42, so extra seeds vary the episode
order as well as the adapter init. Seed 42 is unchanged, and every existing
config keeps its old stream byte-for-byte.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

from src.utils.config import load_config, ConfigDict
from src.utils.seed import effective_train_seed_offset

REPO = Path(__file__).resolve().parents[1]


def _cfg(seed: int, vary: bool | None) -> ConfigDict:
    trainer = {"train_seed_offset": 20000, "num_epochs": 30, "episodes_per_epoch": 100}
    if vary is not None:
        trainer["vary_train_stream_with_seed"] = vary
    return ConfigDict({"seed": seed, "trainer": trainer})


def test_default_unchanged():
    assert effective_train_seed_offset(_cfg(42, None)) == 20000
    assert effective_train_seed_offset(_cfg(43, None)) == 20000
    assert effective_train_seed_offset(_cfg(44, False)) == 20000


def test_vary():
    assert effective_train_seed_offset(_cfg(42, True)) == 20000
    assert effective_train_seed_offset(_cfg(43, True)) == 120000
    assert effective_train_seed_offset(_cfg(46, True)) == 420000


def test_rejects_seed_below_42():
    with pytest.raises(ValueError):
        effective_train_seed_offset(_cfg(41, True))


def test_rejects_overlap_with_frozen_seed_ranges():
    cfg = _cfg(42, None)
    cfg["trainer"]["train_seed_offset"] = 9_900   # stream would cover VAL 10000..10099
    with pytest.raises(ValueError):
        effective_train_seed_offset(cfg)
    cfg["trainer"]["train_seed_offset"] = 0       # stream would cover TEST 0..599
    with pytest.raises(ValueError):
        effective_train_seed_offset(cfg)


def test_old_configs_byte_identical_offset():
    paths = sorted(glob.glob(str(REPO / "configs/grid/*.yaml"))
                   + glob.glob(str(REPO / "configs/rq3_matched/*.yaml")))
    assert len(paths) == 168
    for p in paths:
        assert effective_train_seed_offset(load_config(p)) == 20000, p
