"""Near-OOD loader: TinyImageNet (OpenOOD near-OOD protocol; Step 7 plan).

Mirrors ``get_svhn_ood`` (same signature + ImageNet normalization) so the
episodic evaluator can treat every OOD pool uniformly. TinyImageNet is the
DECISIVE RQ3 test: far-OOD SVHN is easy for softmax-MSP, whereas evidential /
Dirichlet uncertainty is expected to differentiate on near-OOD where MSP
degrades (Malinin & Gales; OpenOOD near/far split).

PERFORMANCE (important on Colab+Drive): the archive holds ~120,000 tiny files.
Extracting or ImageFolder-scanning them on a Google-Drive FUSE mount does one
network round-trip PER FILE and can take HOURS. We therefore NEVER extract:
we cache the single 240MB zip (Drive-friendly: one big sequential file), copy
it once to local disk, and read only the ``num_samples`` images we actually
need straight out of the zip. This turns a multi-hour step into ~1-2 min.
"""
import io
import os
import random
import shutil
import tempfile
import zipfile
from typing import Iterable, Optional

import torch
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_TIN_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_ZIP_NAME = "tiny-imagenet-200.zip"


def get_tinyimagenet_ood(data_root: str = "data", image_size: int = 224,
                         num_samples: int = 500, seed: int = 42,
                         exclude_wnids: Optional[Iterable[str]] = None,
                         ) -> torch.Tensor:
    """Return (N, 3, image_size, image_size) ImageNet-normalized TinyImageNet
    images, sampled deterministically from the train partition's JPEGs. Reads
    the images directly from the zip (no extraction).

    exclude_wnids (Step 9): TinyImageNet-200 shares 25 wnids with
    MiniImageNet's 100-class split (5 of them MiniImageNet TEST classes) --
    when the near-OOD in-distribution dataset is MiniImageNet, pass its 100
    wnids here so the "OOD" pool cannot contain literal in-distribution
    classes. Default None leaves every existing (CIFAR-FS) call byte-
    identical: same `names` list, same rng.sample draw.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # Cache the zip once (Drive-friendly: a single large file downloads fast).
    from ._robust_download import ensure_archive
    ensure_archive(data_root, _ZIP_NAME, [_TIN_URL])
    drive_zip = os.path.join(data_root, _ZIP_NAME)

    # Read entries from a LOCAL copy: per-entry reads on a Drive mount are slow,
    # a single sequential copy to local disk is not. /content on Colab, else tmp.
    local_base = "/content" if os.path.isdir("/content") else tempfile.gettempdir()
    local_zip = os.path.join(local_base, _ZIP_NAME)
    if (not os.path.exists(local_zip)
            or os.path.getsize(local_zip) != os.path.getsize(drive_zip)):
        shutil.copyfile(drive_zip, local_zip)

    with zipfile.ZipFile(local_zip) as z:
        # train images live at tiny-imagenet-200/train/<wnid>/images/<f>.JPEG
        names = [n for n in z.namelist()
                 if "/train/" in n and n.lower().endswith(".jpeg")]
        if exclude_wnids:
            excl = set(exclude_wnids)
            names = [n for n in names
                     if n.split("/train/", 1)[1].split("/", 1)[0] not in excl]
        names.sort()  # deterministic order before sampling
        if not names:
            raise RuntimeError(
                "No train JPEGs found in the TinyImageNet zip; the download may "
                "be corrupt — delete "
                f"{drive_zip} and re-run.")
        rng = random.Random(seed)
        pick = rng.sample(names, min(num_samples, len(names)))
        imgs = []
        for n in pick:
            raw = z.read(n)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            imgs.append(transform(img))

    return torch.stack(imgs)
