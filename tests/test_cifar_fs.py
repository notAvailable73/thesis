"""Kaggle-staged CIFAR-100 discovery for src.datasets.cifar_fs.get_cifar_fs.

Mirrors tests/test_tinyimagenet_ood.py's staged-directory tests: exercises
the recursive `data_root` search branch of `_find_staged_cifar100_root`
directly (the /kaggle/input branch is the same glob logic rooted at a
different directory this sandbox can't write to).
"""
from src.datasets.cifar_fs import _find_staged_cifar100_root


def _make_fake_cifar100(root, has_train=True, has_test=True, has_meta=True,
                        wrap_levels=0):
    """A cifar-100-python/ dir with whichever of train/test/meta requested,
    optionally buried under extra wrapper folders to mirror a Kaggle
    dataset-name wrapper on top of the archive's own top-level folder."""
    d = root
    for i in range(wrap_levels):
        d = d / f"wrapper{i}"
    d = d / "cifar-100-python"
    d.mkdir(parents=True)
    if has_train:
        (d / "train").write_bytes(b"x")
    if has_test:
        (d / "test").write_bytes(b"x")
    if has_meta:
        (d / "meta").write_bytes(b"x")
    return d


def test_finds_staged_full_archive_nested_under_data_root(tmp_path):
    cifar_dir = _make_fake_cifar100(tmp_path, wrap_levels=2)
    root = _find_staged_cifar100_root(str(tmp_path))
    assert root == str(cifar_dir.parent)


def test_returns_none_when_train_is_missing(tmp_path):
    """torchvision's CIFAR100._check_integrity() verifies train+test
    together regardless of the `train=` flag, so a directory missing either
    partition can never be trusted as the staged root -- pointing it there
    anyway would crash trying to download the missing file into what may be
    a read-only /kaggle/input mount, instead of falling back to a writable
    data_root."""
    _make_fake_cifar100(tmp_path, has_train=False)
    assert _find_staged_cifar100_root(str(tmp_path)) is None


def test_returns_none_when_test_is_missing(tmp_path):
    _make_fake_cifar100(tmp_path, has_test=False)
    assert _find_staged_cifar100_root(str(tmp_path)) is None


def test_returns_none_when_meta_is_missing(tmp_path):
    _make_fake_cifar100(tmp_path, has_meta=False)
    assert _find_staged_cifar100_root(str(tmp_path)) is None


def test_returns_none_when_nothing_staged(tmp_path):
    assert _find_staged_cifar100_root(str(tmp_path)) is None
