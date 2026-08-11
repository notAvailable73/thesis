"""Trained B-PEFT adapter — loads a Step 4/5 episodic checkpoint.

This mirrors ``src/adapters/*.py`` bit-for-bit (same modules, same forward
math) so weights trained by ``scripts/train.py`` load directly. Local copies
rather than imports of ``src.adapters`` so the app stays self-contained and
deployable without the research code present.

Three families of adapter, architecturally:
  - post-pool (bottleneck, placement="post_pool"): a small module bolted on
    AFTER the frozen backbone, operating on the pooled 512-d feature. Loading
    it back is just building the module and copying its own state-dict slice.
  - in-backbone weight change (lora, and eventually bitfit / full_ft): the
    adaptation changes the backbone's OWN weights (LoRA injects extra
    low-rank conv layers in place). Loading these means mutating the *live*
    backbone module passed in here, then loading the checkpoint's full
    ``backbone.*`` state-dict slice into it -- there is no separate post-pool
    module to run afterwards.
  - in-backbone forward hook (bottleneck, placement="serial"/"parallel",
    Step 6): the backbone's OWN weights never change; instead small
    Conv1x1Bottleneck modules are held OUTSIDE the backbone (in the
    adapter's own state-dict, `adapter.bodies.*`) and wired in via a forward
    hook on the last block of each stage. Loading these means constructing
    the bottleneck modules, copying their weights, and registering the same
    hook function (serial or parallel) on the live backbone.

Checkpoint schema (``scripts/train.py``'s episodic-trainer save path):
    ckpt["adapter_type"]                     -- e.g. "bottleneck", "lora"
                                                 (NOTE: "bottleneck" covers
                                                 post_pool AND serial/parallel
                                                 placement -- cfg.adapter.type
                                                 doesn't change with
                                                 placement, only
                                                 cfg.adapter.placement does,
                                                 which is NOT saved to the
                                                 checkpoint. See
                                                 _detect_placement_kind().)
    ckpt["config_path"]                      -- absolute path to the training
                                                 config, from wherever it was
                                                 trained (likely a Colab path
                                                 that doesn't exist locally --
                                                 only its basename is usable)
    ckpt["state_dict"]["backbone.*"]         -- full backbone (identity at
                                                 init for bottleneck/placement;
                                                 carries the trained LoRA A/B
                                                 mats for lora)
    ckpt["state_dict"]["adapter.*"]          -- post-pool adapter params
                                                 (bottleneck post_pool: down/
                                                 up; bottleneck serial/
                                                 parallel: bodies.0..3.down/up;
                                                 empty for lora)
    ckpt["state_dict"]["head._evidence_raw_scale"]  -- evidence-affine scale (pre-softplus)
    ckpt["state_dict"]["head._evidence_bias"]       -- evidence-affine bias
``PrototypeHead.cosine_scale`` is a plain Python float, not a tensor, so it
never lands in the state_dict; every Step 4/5/6 config fixes it at 10.0 — see
``_HEAD_COSINE_SCALE`` below. If a checkpoint is ever trained from a config
with a different ``head.cosine_scale``, update this constant to match.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.backend.core.logging import get_logger

log = get_logger("ml.adapter")

_HEAD_COSINE_SCALE = 10.0

# LoRA hyperparameters, hardcoded to match configs/exp_phase3_lora_evidential.yaml
# / exp_phase3_lora_softmax.yaml (rank=16, alpha=null -> rank, dropout=0.0,
# target=layer4.0.downsample.0). If a checkpoint is ever trained from a LoRA
# config with different values, update these to match.
_LORA_RANK = 16
_LORA_ALPHA = 16.0  # alpha: null in the config defaults to rank
_LORA_DROPOUT = 0.0
_LORA_TARGETS = ["layer4.0.downsample.0"]

# Placement (Step 6) hyperparameters, hardcoded to match
# configs/exp_phase3_placement_{serial,parallel}_{evidential,softmax}.yaml
# (rank=16, block_ids=[0,1,2,3] -> one bottleneck at the last block of every
# stage). If a checkpoint is ever trained with different block_ids, update
# _PLACEMENT_BLOCK_IDS to match.
_PLACEMENT_RANK = 16
_PLACEMENT_BLOCK_IDS = [0, 1, 2, 3]
_STAGE_ATTRS = ["layer1", "layer2", "layer3", "layer4"]


class BottleneckAdapter(nn.Module):
    """Houlsby-style bottleneck adapter: down -> ReLU -> up, with residual.

    Architecturally identical to ``src/adapters/bottleneck.py``.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(rank, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class LoRAConv2d(nn.Module):
    """Wrap a frozen nn.Conv2d with a low-rank additive update.

    Architecturally identical to ``src/adapters/lora.py:LoRAConv2d`` --
    ``y = base(x) + (alpha/rank) * lora_B(lora_A(x))``.
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int,
                 alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base_conv
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_A = nn.Conv2d(
            base_conv.in_channels, rank,
            kernel_size=base_conv.kernel_size, stride=base_conv.stride,
            padding=base_conv.padding, dilation=base_conv.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, base_conv.out_channels,
                                kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class Conv1x1Bottleneck(nn.Module):
    """1x1 down -> ReLU -> 1x1 up. NO residual (the hook below adds it).

    Architecturally identical to ``src/adapters/placement.py:Conv1x1Bottleneck``.
    """

    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, rank, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(rank, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def _register_serial_hook(block: nn.Module, mod: Conv1x1Bottleneck):
    """out' = out + body(out) -- mirrors src/adapters/placement.py."""
    def hook(_module, _inp, out):
        return out + mod(out)
    return block.register_forward_hook(hook)


def _register_parallel_hook(block: nn.Module, mod: Conv1x1Bottleneck):
    """out' = out + body(input) -- mirrors src/adapters/placement.py."""
    def hook(_module, inp, out):
        return out + mod(inp[0])
    return block.register_forward_hook(hook)


def _replace_submodule(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    """Replace root.<dotted> (e.g. 'layer4.0.downsample.0') with new_module."""
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    key = parts[-1]
    if key.isdigit():
        parent[int(key)] = new_module
    else:
        setattr(parent, key, new_module)


def _inject_lora(backbone: nn.Module, rank: int, alpha: float,
                 dropout: float, targets: Sequence[str]) -> None:
    """Replace each target conv in-place with a LoRAConv2d wrapper.

    Mirrors ``src/adapters/lora.py:LoRAAdapter.__init__`` exactly -- same
    target resolution order -- so the resulting module tree's state-dict
    keys match a checkpoint trained by that class.
    """
    for name in targets:
        base = backbone.get_submodule(name)
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRA target {name!r} is a {type(base).__name__}, not nn.Conv2d.")
        _replace_submodule(backbone, name, LoRAConv2d(base, rank, alpha, dropout))


@dataclass(frozen=True)
class TrainedAdapterBundle:
    adapter_type: str
    # "post_pool" | "serial" | "parallel" for bottleneck; None for lora.
    placement: Optional[str]
    # Post-pool module to run after the backbone; None when the adaptation
    # already happened inside the backbone's own forward (LoRA / placement).
    adapter: Optional[nn.Module]
    evidence_scale: float  # combined: head.cosine_scale * softplus(raw_scale)
    evidence_bias: float
    checkpoint_path: str
    best_val_acc: float


def _load_bottleneck(sd: dict, dim: int, rank: int) -> BottleneckAdapter:
    adapter = BottleneckAdapter(dim=dim, rank=rank)
    adapter.down.weight.data.copy_(sd["adapter.down.weight"])
    adapter.down.bias.data.copy_(sd["adapter.down.bias"])
    adapter.up.weight.data.copy_(sd["adapter.up.weight"])
    adapter.up.bias.data.copy_(sd["adapter.up.bias"])
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter


def _load_lora_into_backbone(sd: dict, backbone: nn.Module) -> None:
    """Inject LoRA layers into `backbone` (a FrozenResNet18's `.net`) and load
    the checkpoint's trained backbone weights (base convs + LoRA A/B mats)."""
    _inject_lora(backbone, _LORA_RANK, _LORA_ALPHA, _LORA_DROPOUT, _LORA_TARGETS)
    prefix = "backbone."
    backbone_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    backbone.load_state_dict(backbone_sd, strict=True)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()


def _detect_placement_kind(ckpt: dict, hint: str) -> str:
    """Best-effort disambiguation of serial vs parallel placement.

    The two placements produce IDENTICAL bottleneck weight shapes -- only the
    hook wiring (out+body(out) vs out+body(input)) differs, and that isn't
    recorded anywhere in the state-dict. Try the training config's filename
    (its basename is portable even though the full `config_path` is likely an
    absolute Colab path that doesn't exist locally); otherwise fall back to
    the caller-supplied hint (SENTINEL_ADAPTER_PLACEMENT).
    """
    name = Path(str(ckpt.get("config_path", ""))).name.lower()
    if "parallel" in name:
        return "parallel"
    if "serial" in name:
        return "serial"
    log.warning(
        "Checkpoint has bodies.* weights (Step 6 placement adapter) but "
        "config_path=%r doesn't say serial or parallel; using "
        "SENTINEL_ADAPTER_PLACEMENT=%r. Set that env var explicitly if "
        "this is wrong -- the two placements are NOT distinguishable from "
        "the weights alone.", str(ckpt.get("config_path", "")), hint,
    )
    return hint


def _load_placement_into_backbone(sd: dict, backbone: nn.Module, placement: str) -> None:
    """Build the Step 6 in-block bottlenecks, load their trained weights, and
    register the serial/parallel forward hook on the live backbone.

    Unlike LoRA, the backbone's OWN weights never change for this adapter
    type (it's a pure forward-hook side-branch), so there is no backbone.*
    state-dict to load here.
    """
    register = _register_serial_hook if placement == "serial" else _register_parallel_hook
    for i, stage_id in enumerate(_PLACEMENT_BLOCK_IDS):
        block = getattr(backbone, _STAGE_ATTRS[stage_id])[-1]
        channels = block.conv2.out_channels
        body = Conv1x1Bottleneck(channels, _PLACEMENT_RANK)
        body.down.weight.data.copy_(sd[f"adapter.bodies.{i}.down.weight"])
        body.down.bias.data.copy_(sd[f"adapter.bodies.{i}.down.bias"])
        body.up.weight.data.copy_(sd[f"adapter.bodies.{i}.up.weight"])
        body.up.bias.data.copy_(sd[f"adapter.bodies.{i}.up.bias"])
        body.eval()
        for p in body.parameters():
            p.requires_grad_(False)
        register(block, body)


def load_trained_adapter(
    checkpoint_path: Path, backbone: nn.Module, dim: int = 512, rank: int = 16,
    placement_hint: str = "parallel",
) -> TrainedAdapterBundle | None:
    """Load a trained adapter + evidence affine from a Step 4/5/6 checkpoint.

    `backbone` is the live `FrozenResNet18.net` module -- for in-backbone
    adapter types (LoRA, Step-6 placement) this function mutates it in place
    (injects layers or registers hooks, loads trained weights). For post-pool
    Bottleneck it is untouched. `placement_hint` is only used as a fallback
    when a Step-6 checkpoint's own metadata doesn't say serial vs parallel
    (see `_detect_placement_kind`); default "parallel" matches the thesis's
    own best/recommended result (step_writeups/step6.txt).

    Returns ``None`` (never raises) if the checkpoint is absent, its
    adapter_type isn't supported yet, or loading fails -- the app falls back
    to the untrained backbone-only pipeline so a missing/bad checkpoint never
    blocks startup.
    """
    if not checkpoint_path.exists():
        log.info(
            "No trained adapter checkpoint at %s -- using untrained "
            "backbone-only pipeline.", checkpoint_path,
        )
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["state_dict"]
        adapter_type = ckpt.get("adapter_type", "bottleneck")

        placement: Optional[str]
        if adapter_type == "bottleneck" and "adapter.bodies.0.down.weight" in sd:
            # Step 6: cfg.adapter.type stays "bottleneck" for post_pool AND
            # serial/parallel -- only the presence of bodies.* distinguishes
            # placement from the Step 4/5 post-pool form.
            placement = _detect_placement_kind(ckpt, placement_hint)
            _load_placement_into_backbone(sd, backbone, placement)
            post_pool_adapter: Optional[nn.Module] = None
        elif adapter_type == "bottleneck":
            placement = "post_pool"
            post_pool_adapter = _load_bottleneck(sd, dim, rank)
        elif adapter_type == "lora":
            placement = None
            _load_lora_into_backbone(sd, backbone)
            post_pool_adapter = None
        else:
            log.warning(
                "Checkpoint %s has adapter_type=%r, which this app build "
                "does not yet know how to load; falling back to "
                "backbone-only.", checkpoint_path, adapter_type,
            )
            return None

        raw_scale = sd["head._evidence_raw_scale"]
        bias = sd["head._evidence_bias"]
        evidence_scale = float(F.softplus(raw_scale)) * _HEAD_COSINE_SCALE
        evidence_bias = float(bias)

        best_val_acc = float(ckpt.get("best_val_acc", float("nan")))
        log.info(
            "Loaded trained adapter (%s%s) from %s (best_val_acc=%.3f, "
            "evidence_scale=%.3f, evidence_bias=%.3f)",
            adapter_type, f"/{placement}" if placement else "",
            checkpoint_path, best_val_acc, evidence_scale, evidence_bias,
        )
        return TrainedAdapterBundle(
            adapter_type=adapter_type,
            placement=placement,
            adapter=post_pool_adapter,
            evidence_scale=evidence_scale,
            evidence_bias=evidence_bias,
            checkpoint_path=str(checkpoint_path),
            best_val_acc=best_val_acc,
        )
    except Exception as exc:  # noqa: BLE001 - never let a bad checkpoint crash startup
        log.warning(
            "Could not load trained adapter from %s (%s); falling back to "
            "backbone-only.", checkpoint_path, exc,
        )
        return None
