"""Kaggle-staged SVHN discovery for src.datasets.svhn_ood.get_svhn_ood.

Mirrors tests/test_tinyimagenet_ood.py's staged-directory tests: exercises
the recursive `data_root` search branch of `_find_staged_svhn_root` directly
(the /kaggle/input branch is the same glob logic rooted at a different
directory this sandbox can't write to).
"""
from src.datasets.svhn_ood import _find_staged_svhn_root


def test_finds_mat_file_directly_in_data_root(tmp_path):
    (tmp_path / "test_32x32.mat").write_bytes(b"x")
    assert _find_staged_svhn_root(str(tmp_path)) == str(tmp_path)


def test_finds_mat_file_nested_under_data_root(tmp_path):
    nested = tmp_path / "datasets" / "notavailable73" / "bpeft-extra-data"
    nested.mkdir(parents=True)
    (nested / "test_32x32.mat").write_bytes(b"x")
    assert _find_staged_svhn_root(str(tmp_path)) == str(nested)


def test_returns_none_when_absent(tmp_path):
    assert _find_staged_svhn_root(str(tmp_path)) is None
