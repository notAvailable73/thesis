from .linear_head import LinearHead, EvidentialHead


def build_head(spec: dict, in_dim: int, num_classes: int):
    """spec: {type: softmax} or {type: evidential, activation: softplus|exp_clamp|relu}."""
    htype = spec["type"]
    if htype == "softmax":
        return LinearHead(in_dim, num_classes)
    if htype == "evidential":
        return EvidentialHead(in_dim, num_classes, activation=spec.get("activation", "softplus"))
    raise ValueError(f"Unknown head type: {htype}")


__all__ = ["build_head", "LinearHead", "EvidentialHead"]
