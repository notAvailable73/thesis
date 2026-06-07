from .linear_head import LinearHead, EvidentialHead
from .prototype_head import PrototypeHead, ScaledPrototypeHead


def build_head(spec: dict, in_dim: int, num_classes: int):
    """Construct a head from a config dict.

    spec keys:
      type: "softmax" | "evidential" | "prototype"

      For "evidential" the spec may also include:
        activation: "softplus" | "exp_clamp" | "relu"   (default: softplus)

      For "prototype" (Step 4 / Phase 2 — used with episodic meta-training):
        metric:         "l2" | "cosine"                 (default: l2)
        cosine_scale:   float                           (default: 10.0)
        interpretation: "softmax" | "evidential"

      When interpretation == "evidential", returns ScaledPrototypeHead,
      which adds learnable instance-standardization + affine (tau, bias)
      so that softplus inputs are non-zero from step 1 regardless of the
      raw L2 distance scale (action 4.21 of Step 4 remediation).
      When interpretation == "softmax" (or omitted), returns plain
      PrototypeHead (parameter-free).
    """
    htype = spec["type"]
    if htype == "softmax":
        return LinearHead(in_dim, num_classes)
    if htype == "evidential":
        return EvidentialHead(
            in_dim, num_classes,
            activation=spec.get("activation", "softplus"),
        )
    if htype == "prototype":
        metric       = spec.get("metric", "l2")
        cosine_scale = float(spec.get("cosine_scale", 10.0))
        interp       = spec.get("interpretation", "softmax")
        if interp == "evidential":
            return ScaledPrototypeHead(metric=metric, cosine_scale=cosine_scale)
        return PrototypeHead(metric=metric, cosine_scale=cosine_scale)
    raise ValueError(f"Unknown head type: {htype}")


__all__ = ["build_head", "LinearHead", "EvidentialHead",
           "PrototypeHead", "ScaledPrototypeHead"]
