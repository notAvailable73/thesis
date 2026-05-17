from .bottleneck import BottleneckAdapter
from .lora import LoRAAdapter


def build_adapter(spec: dict, dim: int):
    """spec: dict from config, e.g. {type: bottleneck, rank: 16} or {type: lora, rank: 4, alpha: 8}."""
    atype = spec["type"]
    rank = spec["rank"]
    if atype == "bottleneck":
        return BottleneckAdapter(dim=dim, rank=rank)
    if atype == "lora":
        return LoRAAdapter(dim=dim, rank=rank, alpha=spec.get("alpha"))
    raise ValueError(f"Unknown adapter type: {atype}")


__all__ = ["build_adapter", "BottleneckAdapter", "LoRAAdapter"]
