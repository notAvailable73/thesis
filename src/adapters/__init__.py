from .bottleneck import BottleneckAdapter
from .lora import LoRAAdapter, LoRALayer, LoRAConv2d, default_lora_targets
from .bitfit import BitFitAdapter
from .full_ft import FullFTAdapter
from .linear_probe import LinearProbeAdapter
from .placement import (
    PlacementAdapter, Conv1x1Bottleneck,
    register_serial_adapter, register_parallel_adapter,
    resolve_stage_paths, infer_block_channels,
)


def build_adapter(spec: dict, dim: int, backbone=None):
    """Construct an adapter from a config dict.

    spec examples:
      {type: bottleneck, rank: 16}
      {type: lora, rank: 16, alpha: null, lora_targets: [...]}
      {type: bitfit}
      {type: full_ft}
      {type: linear_probe}

    Post-pool adapters (bottleneck, linear_probe) operate on the frozen
    backbone's pooled feature vector and do NOT need the backbone here. The
    in-backbone adapters (lora, bitfit, full_ft) modify the backbone in place
    at construction, so they REQUIRE `backbone`. Each returned module carries a
    `backbone_trainable` flag telling BPEFTModel / the trainer whether the
    backbone must run with gradients.
    """
    atype = spec["type"]
    if atype == "bottleneck":
        # Step 6: the Bottleneck adapter can be PLACED. post_pool (default) is
        # the Step-4 pooled-feature adapter; serial/parallel insert a 1x1
        # bottleneck inside the backbone blocks (needs the backbone).
        placement = spec.get("placement", "post_pool")
        if placement == "post_pool":
            return BottleneckAdapter(dim=dim, rank=int(spec["rank"]))
        if placement in ("serial", "parallel"):
            if backbone is None:
                raise ValueError(
                    "serial/parallel bottleneck placement requires a backbone")
            # Step 8: stage sites are derived from the backbone family
            # (ResNet / MobileNetV3); adapter.stage_paths overrides explicitly.
            return PlacementAdapter(
                backbone=backbone, rank=int(spec["rank"]),
                placement=placement, block_ids=spec.get("block_ids"),
                stage_paths=spec.get("stage_paths"))
        raise ValueError(f"Unknown bottleneck placement: {placement!r}")
    if atype == "lora":
        if backbone is None:
            raise ValueError("lora adapter requires a backbone to inject into")
        return LoRAAdapter(
            backbone=backbone,
            rank=int(spec["rank"]),
            alpha=spec.get("alpha"),
            targets=spec.get("lora_targets"),
            dropout=float(spec.get("dropout", 0.0)),
        )
    if atype == "bitfit":
        if backbone is None:
            raise ValueError("bitfit adapter requires a backbone")
        return BitFitAdapter(backbone=backbone)
    if atype == "full_ft":
        if backbone is None:
            raise ValueError("full_ft adapter requires a backbone")
        return FullFTAdapter(backbone=backbone)
    if atype == "linear_probe":
        return LinearProbeAdapter()
    raise ValueError(f"Unknown adapter type: {atype}")


__all__ = [
    "build_adapter",
    "BottleneckAdapter",
    "LoRAAdapter",
    "LoRALayer",
    "LoRAConv2d",
    "default_lora_targets",
    "BitFitAdapter",
    "FullFTAdapter",
    "LinearProbeAdapter",
    "PlacementAdapter",
    "Conv1x1Bottleneck",
    "register_serial_adapter",
    "register_parallel_adapter",
    "resolve_stage_paths",
    "infer_block_channels",
]
