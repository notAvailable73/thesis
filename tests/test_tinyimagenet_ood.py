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
