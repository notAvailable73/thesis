"""Image preprocessing: raw bytes -> normalised model input tensor.

Kept separate from the backbone so the exact preprocessing (resize, crop,
ImageNet normalisation) is documented in one place and can be reused for both
enrollment and detection — they MUST match or embeddings won't be comparable.
"""
from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision import transforms as T

# ImageNet channel statistics — required because the backbone was pretrained
# with these; feeding un-normalised images silently degrades features.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> T.Compose:
    """Deterministic eval transform (no augmentation — inference must be stable)."""
    resize = int(round(image_size * 1.14))  # resize then centre-crop, standard eval recipe
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class ImagePreprocessor:
    """Turns uploaded image bytes into a ``(3, H, W)`` float tensor."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self._tf = build_transform(image_size)

    def __call__(self, raw: bytes) -> torch.Tensor:
        return self.from_bytes(raw)

    def from_bytes(self, raw: bytes) -> torch.Tensor:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - surface a clean API error upstream
            raise ValueError(f"Could not decode image: {exc}") from exc
        return self._tf(img)
