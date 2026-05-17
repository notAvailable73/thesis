import textwrap
from pathlib import Path
import pytest
from src.utils import load_config


def test_simple_load(tmp_path: Path):
    p = tmp_path / "a.yaml"
    p.write_text("seed: 7\nadapter:\n  type: lora\n  rank: 4\n")
    cfg = load_config(p)
    assert cfg.seed == 7
    assert cfg.adapter.type == "lora"
    assert cfg.adapter.rank == 4


def test_extends_merges(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(textwrap.dedent("""
        seed: 1
        adapter:
          type: bottleneck
          rank: 16
        train:
          lr: 1.0e-3
    """).strip())
    exp = tmp_path / "exp.yaml"
    exp.write_text(textwrap.dedent("""
        extends: base.yaml
        adapter:
          type: lora
          rank: 4
    """).strip())
    cfg = load_config(exp)
    assert cfg.seed == 1                     # inherited
    assert cfg.adapter.type == "lora"        # overridden
    assert cfg.adapter.rank == 4             # overridden
    assert cfg.train.lr == 1.0e-3            # inherited from base


def test_config_repo_yamls_load():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/exp_step1.yaml")
    assert cfg.head.type == "evidential"
    assert cfg.head.activation == "softplus"
    assert cfg.loss.kl_weight_max == 0.5
    assert cfg.adapter.type == "bottleneck"
    cfg2 = load_config(Path(__file__).resolve().parents[1] / "configs/exp_step1_softmax.yaml")
    assert cfg2.head.type == "softmax"
