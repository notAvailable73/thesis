"""Step 9 — get_id_split / get_heldout_near_ood dispatch tests.

scripts/train.py and scripts/evaluate.py used to call get_cifar_fs(...)
directly at five sites; Step 9 routes them through get_id_split /
get_heldout_near_ood so cfg.dataset.name selects the in-distribution dataset.
The REGRESSION tests below (monkeypatched, no data required) prove that for a
config that never sets `dataset.name` -- i.e. every pre-Step-9 config -- the
dispatch is a byte-identical pass-through to the old direct call, with the
same positional/keyword arguments. This is the property Steps 1-8's
byte-identical-rerun invariant depends on.
"""
import pytest

import src.datasets as ds
from src.utils import ConfigDict


def test_get_id_split_defaults_to_cifar_fs_when_name_absent(monkeypatch):
    calls = []

    def fake_get_cifar_fs(data_root, image_size, split, class_ids):
        calls.append((data_root, image_size, split, class_ids))
        return "CIFAR_SENTINEL"

    monkeypatch.setattr(ds, "get_cifar_fs", fake_get_cifar_fs)
    cfg = ConfigDict({"data_root": "data", "image_size": 224})  # no "name" key
    out = ds.get_id_split(cfg, split="val")
    assert out == "CIFAR_SENTINEL"
    assert calls == [("data", 224, "val", None)]


def test_get_id_split_cifar_fs_explicit_name(monkeypatch):
    calls = []

    def fake_get_cifar_fs(data_root, image_size, split, class_ids):
        calls.append((data_root, image_size, split, class_ids))
        return "CIFAR_SENTINEL"

    monkeypatch.setattr(ds, "get_cifar_fs", fake_get_cifar_fs)
    cfg = ConfigDict({"name": "cifar_fs", "data_root": "data",
                      "image_size": 32, "class_ids": [0, 1, 2]})
    out = ds.get_id_split(cfg, split="test")
    assert out == "CIFAR_SENTINEL"
    assert calls == [("data", 32, "test", [0, 1, 2])]


def test_get_id_split_dispatches_to_mini_imagenet(monkeypatch):
    calls = []

    def fake_get_mini_imagenet(data_root, image_size, split):
        calls.append((data_root, image_size, split))
        return "MINI_SENTINEL"

    monkeypatch.setattr(ds, "get_mini_imagenet", fake_get_mini_imagenet)
    cfg = ConfigDict({"name": "mini_imagenet", "data_root": "data",
                      "image_size": 224})
    out = ds.get_id_split(cfg, split="train")
    assert out == "MINI_SENTINEL"
    assert calls == [("data", 224, "train")]


def test_get_id_split_unknown_name_raises():
    cfg = ConfigDict({"name": "nonexistent_dataset"})
    with pytest.raises(ValueError, match="Unknown dataset"):
        ds.get_id_split(cfg, split="test")


def test_get_heldout_near_ood_pool_names_per_dataset(monkeypatch):
    monkeypatch.setattr(ds, "get_cifar_fs_heldout_ood",
                        lambda **kw: "CIFAR_HELDOUT")
    monkeypatch.setattr(ds, "get_mini_imagenet_heldout_ood",
                        lambda **kw: "MINI_HELDOUT")

    name, x = ds.get_heldout_near_ood(
        ConfigDict({"name": "cifar_fs"}), num_samples=10, seed=0)
    assert (name, x) == ("cifar100_near", "CIFAR_HELDOUT")

    name, x = ds.get_heldout_near_ood(
        ConfigDict({"name": "mini_imagenet"}), num_samples=10, seed=0)
    assert (name, x) == ("mini_near", "MINI_HELDOUT")


def test_get_heldout_near_ood_defaults_to_cifar_fs_when_name_absent(monkeypatch):
    monkeypatch.setattr(ds, "get_cifar_fs_heldout_ood",
                        lambda **kw: "CIFAR_HELDOUT")
    name, x = ds.get_heldout_near_ood(
        ConfigDict({}), num_samples=10, seed=0)
    assert name == "cifar100_near"


def test_get_heldout_near_ood_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        ds.get_heldout_near_ood(ConfigDict({"name": "nonexistent"}),
                                num_samples=10, seed=0)


def test_build_dataset_mini_imagenet_branch(monkeypatch):
    calls = []

    def fake_get_mini_imagenet(data_root, image_size, split):
        calls.append((data_root, image_size, split))
        return "MINI_SENTINEL"

    monkeypatch.setattr(ds, "get_mini_imagenet", fake_get_mini_imagenet)
    out = ds.build_dataset({"name": "mini_imagenet", "split": "test"})
    assert out == "MINI_SENTINEL"
    assert calls == [("data", 224, "test")]


# =====================================================================
# Real (non-mocked) regression guard: requires CIFAR-100 already downloaded
# locally, same requirement the existing cifar100-heldout tests in
# tests/test_tinyimagenet_ood.py already carry. Proves get_id_split's
# cifar_fs branch returns a dataset whose .targets match a direct
# get_cifar_fs(...) call exactly -- i.e. the Step 9 refactor changed nothing
# observable about the CIFAR-FS path.
# =====================================================================
def test_get_id_split_cifar_fs_matches_direct_call():
    direct = ds.get_cifar_fs(data_root="data", image_size=32, split="test")
    routed = ds.get_id_split(
        ConfigDict({"data_root": "data", "image_size": 32}), split="test")
    assert list(direct.targets) == list(routed.targets)
    assert len(direct) == len(routed)


def test_get_heldout_near_ood_cifar_fs_matches_direct_call():
    direct = ds.get_cifar_fs_heldout_ood(
        data_root="data", image_size=32, num_samples=8, seed=7,
        heldout_split="val")
    name, routed = ds.get_heldout_near_ood(
        ConfigDict({"data_root": "data", "image_size": 32}),
        num_samples=8, seed=7, heldout_split="val")
    assert name == "cifar100_near"
    import torch
    assert torch.allclose(direct, routed)
