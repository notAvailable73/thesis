from .linear_head import LinearHead, EvidentialHead
from .prototype_head import PrototypeHead


def build_head(spec: dict, in_dim: int, num_classes: int):
    """Construct a head from a config dict.

    spec keys:
      type: "softmax" | "evidential" | "prototype"

      For "evidential" the spec may also include:
        activation: "softplus" | "exp_clamp" | "relu"   (default: softplus)

      For "prototype" (Step 4 / Phase 2 — parameter-free, used with
      episodic meta-training) the spec may also include:
        metric:       "l2" | "cosine"                  (default: l2)
        cosine_scale: float                            (default: 10.0)

      The interpretation of the prototype-head logits (softmax vs
      evidential Dirichlet) is set by spec["interpretation"]
        - "softmax"     -> apply softmax for probabilities
        - "evidential"  -> softplus -> evidence -> Dirichlet
      The interpretation field is read by the LOSS / EVALUATOR, not by
      the head; this header just emits raw logits.
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
        return PrototypeHead(
            metric=spec.get("metric", "l2"),
            cosine_scale=float(spec.get("cosine_scale", 10.0)),
        )
    raise ValueError(f"Unknown head type: {htype}")


__all__ = ["build_head", "LinearHead", "EvidentialHead", "PrototypeHead"]
