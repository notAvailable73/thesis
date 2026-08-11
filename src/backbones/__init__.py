from .resnet18 import build_frozen_resnet18
from .mobilenetv3 import (
    build_frozen_mobilenetv3_small,
    mobilenetv3_stage_paths,
    MOBILENETV3_SMALL_FEATURE_DIM,
)

#: Canonical pooled-feature dim per backbone. Used to validate
#: cfg.backbone.feature_dim so a config that switches the backbone but forgets
#: feature_dim fails loudly at build time instead of silently mis-sizing the
#: post-pool adapter (Step 8 / adding the second backbone made this a real
#: foot-gun: 512 vs 576).
BACKBONE_FEATURE_DIMS = {
    "resnet18": 512,
    "mobilenetv3_small": MOBILENETV3_SMALL_FEATURE_DIM,
}

#: Accepted config spellings -> canonical name.
_ALIASES = {
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
    "mobilenetv3_small": "mobilenetv3_small",
    "mobilenet_v3_small": "mobilenetv3_small",
    "mobilenetv3": "mobilenetv3_small",
    "mbnet": "mobilenetv3_small",
}


def canonical_backbone_name(name: str) -> str:
    """Normalise a cfg.backbone.name spelling; raise on an unknown backbone."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown backbone: {name!r}. Known: {sorted(set(_ALIASES))}"
        )
    return _ALIASES[key]


def backbone_feature_dim(name: str) -> int:
    """Canonical pooled-feature dim for a backbone name (accepts aliases)."""
    return BACKBONE_FEATURE_DIMS[canonical_backbone_name(name)]


def build_backbone(name: str, pretrained: bool = True):
    """Build a FROZEN, eval-mode backbone whose forward returns (B, D) pooled
    features.

    `pretrained=False` skips the ImageNet weight download — for offline unit
    tests only; every experiment config uses the default (True).
    """
    canonical = canonical_backbone_name(name)
    if canonical == "resnet18":
        return build_frozen_resnet18(pretrained=pretrained)
    if canonical == "mobilenetv3_small":
        return build_frozen_mobilenetv3_small(pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {name}")


__all__ = [
    "build_backbone",
    "build_frozen_resnet18",
    "build_frozen_mobilenetv3_small",
    "mobilenetv3_stage_paths",
    "backbone_feature_dim",
    "canonical_backbone_name",
    "BACKBONE_FEATURE_DIMS",
    "MOBILENETV3_SMALL_FEATURE_DIM",
]
