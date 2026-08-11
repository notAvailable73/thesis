"""Step 4.5 / W3 — near-OOD loaders (TinyImageNet + CIFAR-100-heldout)."""
import pytest
import torch


def test_cifar_fs_heldout_ood_shape():
    """Zero-download near-OOD: images from the CIFAR-FS val-split classes,
    disjoint from the 20 test-episode classes. CIFAR-100 is already local."""
    from src.datasets import get_cifar_fs_heldout_ood
    x = get_cifar_fs_heldout_ood(image_size=224, num_samples=16, seed=0)
    assert x.shape == (16, 3, 224, 224)
    assert x.dtype == torch.float32


def test_cifar_fs_heldout_ood_is_seed_deterministic():
    from src.datasets import get_cifar_fs_heldout_ood
    a = get_cifar_fs_heldout_ood(image_size=32, num_samples=8, seed=1)
    b = get_cifar_fs_heldout_ood(image_size=32, num_samples=8, seed=1)
    assert torch.allclose(a, b)


def test_tinyimagenet_ood_reads_from_zip_without_extract(tmp_path):
    """Exercise the real zip-reading path (no 240MB download) by pre-placing a
    tiny fake tiny-imagenet-200.zip. Guards the perf fix: the loader must read
    images straight out of the zip, never extract the ~120k-file archive."""
    import io
    import zipfile
    from PIL import Image
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    zpath = data_root / "tiny-imagenet-200.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(4):
            img = Image.new("RGB", (64, 64), (i * 60, 100, 200))
            buf = io.BytesIO()
            img.save(buf, "JPEG")
            z.writestr(
                f"tiny-imagenet-200/train/n{i:04d}/images/{i}.JPEG",
                buf.getvalue(),
            )

    x = get_tinyimagenet_ood(data_root=str(data_root), image_size=32,
                             num_samples=3, seed=0)
    assert x.shape == (3, 3, 32, 32)
    assert x.dtype == torch.float32
    # deterministic given the seed
    x2 = get_tinyimagenet_ood(data_root=str(data_root), image_size=32,
                              num_samples=3, seed=0)
    assert torch.allclose(x, x2)
    # the loader must NOT have extracted the archive into data_root
    assert not (data_root / "tiny-imagenet-200").exists()


def _make_fake_tin_extracted(data_root, n_classes, wrap_levels=0):
    """An already-extracted tiny-imagenet-200/ directory: wnids.txt + train/
    <wnid>/images/<f>.JPEG, optionally buried under extra wrapper folders to
    mirror Kaggle's observed double-nesting (dataset-name wrapper on top of
    the archive's own top-level folder)."""
    from PIL import Image

    root = data_root
    for i in range(wrap_levels):
        root = root / f"wrapper{i}"
    root = root / "tiny-imagenet-200"
    root.mkdir(parents=True)
    (root / "wnids.txt").write_text("\n".join(f"n{i:08d}" for i in range(n_classes)))
    for i in range(n_classes):
        wnid = f"n{i:08d}"
        img_dir = root / "train" / wnid / "images"
        img_dir.mkdir(parents=True)
        img = Image.new("RGB", (64, 64), (i * 20 % 255, 100, 200))
        img.save(img_dir / f"{wnid}_0.JPEG")
    return root


def test_tinyimagenet_ood_reads_from_extracted_dir_without_zip(tmp_path):
    """Step 9: a manually-staged, already-extracted directory (no zip at all)
    must be read directly -- the archive never needs to exist as a zip."""
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_fake_tin_extracted(data_root, n_classes=4, wrap_levels=2)

    x = get_tinyimagenet_ood(data_root=str(data_root), image_size=32,
                             num_samples=3, seed=0)
    assert x.shape == (3, 3, 32, 32)
    assert x.dtype == torch.float32
    x2 = get_tinyimagenet_ood(data_root=str(data_root), image_size=32,
                              num_samples=3, seed=0)
    assert torch.allclose(x, x2)
    # no zip was ever fetched/created for this path
    assert not (data_root / "tiny-imagenet-200.zip").exists()


def test_tinyimagenet_ood_extracted_dir_exclude_wnids(tmp_path):
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_fake_tin_extracted(data_root, n_classes=4)
    excl = {f"n{i:08d}" for i in range(3)}  # exclude classes 0,1,2 -- keep 3

    x = get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=10, seed=0, exclude_wnids=excl)
    assert x.shape[0] == 1  # only class 3's single image can remain

    with pytest.raises(RuntimeError, match="No train JPEGs"):
        get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=10, seed=0,
                             exclude_wnids={f"n{i:08d}" for i in range(4)})


def test_tinyimagenet_ood_prefers_extracted_dir_over_zip(tmp_path):
    """If both a staged zip and an already-extracted directory are present,
    the extracted directory wins -- proven by making the zip unreadable and
    confirming the loader still succeeds (it never had to open the zip)."""
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_fake_tin_extracted(data_root, n_classes=2)
    (data_root / "tiny-imagenet-200.zip").write_bytes(b"not a real zip")

    x = get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=2, seed=0)
    assert x.shape == (2, 3, 16, 16)


def _make_fake_tin_zip(data_root, n_classes):
    """Same fixture shape as the no-extract test above, but with distinct
    wnids per class so exclude_wnids has something to filter."""
    import io
    import zipfile
    from PIL import Image

    zpath = data_root / "tiny-imagenet-200.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(n_classes):
            img = Image.new("RGB", (64, 64), (i * 20 % 255, 100, 200))
            buf = io.BytesIO()
            img.save(buf, "JPEG")
            z.writestr(
                f"tiny-imagenet-200/train/n{i:08d}/images/{i}.JPEG",
                buf.getvalue(),
            )


def test_exclude_wnids_none_is_byte_identical_to_default(tmp_path):
    """Step 9: exclude_wnids=None (every pre-Step-9 call) must reproduce the
    exact same sample as calling with the parameter omitted entirely."""
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_fake_tin_zip(data_root, n_classes=8)

    a = get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=5, seed=3)
    b = get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=5, seed=3, exclude_wnids=None)
    assert torch.allclose(a, b)


def test_exclude_wnids_drops_the_named_classes(tmp_path):
    """A non-empty exclude_wnids must shrink the eligible pool: sampling all
    remaining images after excluding all-but-one class must always return the
    one surviving class's image, deterministically."""
    from src.datasets import get_tinyimagenet_ood

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_fake_tin_zip(data_root, n_classes=4)
    excl = {f"n{i:08d}" for i in range(3)}  # exclude classes 0,1,2 -- keep 3

    x = get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=10, seed=0, exclude_wnids=excl)
    # only 1 image (class 3) can possibly remain in the pool
    assert x.shape[0] == 1

    # excluding every class must raise the same "no train JPEGs" error the
    # loader already raises on a corrupt/empty archive.
    with pytest.raises(RuntimeError, match="No train JPEGs"):
        get_tinyimagenet_ood(data_root=str(data_root), image_size=16,
                             num_samples=10, seed=0,
                             exclude_wnids={f"n{i:08d}" for i in range(4)})
