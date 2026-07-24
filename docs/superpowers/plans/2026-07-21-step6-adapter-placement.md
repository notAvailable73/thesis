# Step 6 — Adapter Placement Study (RQ1) Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Implement task-by-task, running tests after each.

**Goal:** Add serial/parallel in-block placement of a 1×1 Bottleneck adapter to the frozen ResNet-18, plus configs, tests, a comparison plot, a writeup skeleton, and a Kaggle notebook — to answer RQ1 (placement vs accuracy/params).

**Architecture:** A new `PlacementAdapter` registers forward hooks on the final BasicBlock of each selected stage; the hook adds a zero-init 1×1 channel bottleneck (serial: on block output; parallel: on block input). It reuses the existing `backbone_trainable=True` wiring (identical to LoRA), so no trainer/evaluator changes are needed. Config selects placement via `adapter.placement`.

**Tech Stack:** PyTorch, torchvision ResNet-18, pytest, matplotlib. Spec: `docs/superpowers/specs/2026-07-21-step6-adapter-placement-design.md`.

## Global Constraints

- **Do NOT `git commit`.** Repo convention: a human reviews and commits. Each task ends by running its tests green and leaving the changes for review.
- **Frozen protocol:** do not change `image_size` (224), the 600 test seeds, `cudnn.deterministic`, or any file marked "DO NOT REGENERATE".
- Adapter form is **1×1 channel bottleneck**, rank **16**, held identical across placements (RQ1 isolates placement only). Zero-init `up` ⇒ identity at init.
- `block_ids` indexes **stages** `[0,1,2,3]` = layer1–4; adapter attaches to the **final BasicBlock** of each selected stage.
- Placement adapters set `backbone_trainable = True` (backbone weights stay frozen; only the bottlenecks train).
- Tests must run offline: build ResNet-18 with `weights=None` (mirror `tests/test_lora.py:_bare_frozen_resnet18`).

---

### Task 1: `PlacementAdapter` + hooks (`src/adapters/placement.py`)

**Files:**
- Create: `src/adapters/placement.py`
- Test: `tests/test_placement.py`

**Interfaces:**
- Produces: `Conv1x1Bottleneck(channels:int, rank:int)` (forward = `up(relu(down(x)))`, no residual); `register_serial_adapter(block, mod)->handle`; `register_parallel_adapter(block, mod)->handle`; `PlacementAdapter(backbone, rank:int, placement:str, block_ids:list|None)` with class attr `backbone_trainable=True`, `forward(x)->x`.

- [ ] **Step 1: Write failing tests** `tests/test_placement.py`:

```python
import torch, torch.nn as nn, pytest
from torchvision.models import resnet18
from src.adapters import PlacementAdapter, Conv1x1Bottleneck, build_adapter

def _bare_frozen_resnet18():
    m = resnet18(weights=None); m.fc = nn.Identity()
    for p in m.parameters(): p.requires_grad = False
    m.eval(); return m

def _pooled(backbone, x):
    return backbone(x)  # fc=Identity -> pooled (B,512)

@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_identity_at_init(placement):
    torch.manual_seed(0)
    bb = _bare_frozen_resnet18()
    x = torch.randn(2, 3, 224, 224)
    ref = _pooled(bb, x).clone()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[0,1,2,3])
    out = _pooled(bb, x)
    assert out.shape == ref.shape                       # shape preserved
    assert torch.allclose(out, ref, atol=1e-6)          # zero-init up -> identity

@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_only_adapter_trains(placement):
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[0,1,2,3])
    assert all(not p.requires_grad for p in bb.parameters() if p is not None and _in_backbone_core(bb, p))
    trainable = [p for p in ad.parameters() if p.requires_grad]
    # analytical: sum over stages of down(C*r + r) + up(r*C + C)
    exp = sum((C*16 + 16) + (16*C + C) for C in (64,128,256,512))
    assert sum(p.numel() for p in trainable) == exp

def _in_backbone_core(bb, p):
    return True  # bb params are all frozen conv/bn; sanity that none flipped

@pytest.mark.parametrize("placement", ["serial", "parallel"])
def test_gradients_reach_adapter(placement):
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement=placement, block_ids=[3])
    # perturb up weights so the adapter is not a no-op, then check grad flows
    for m in ad.bodies:
        nn.init.normal_(m.up.weight, std=0.01)
    x = torch.randn(2, 3, 224, 224)
    loss = bb(x).pow(2).sum()
    loss.backward()
    grads = [m.down.weight.grad for m in ad.bodies]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

def test_build_adapter_routes_placement():
    bb = _bare_frozen_resnet18()
    ad = build_adapter({"type": "bottleneck", "rank": 16,
                        "placement": "parallel", "block_ids": [0,1,2,3]},
                       dim=512, backbone=bb)
    assert isinstance(ad, PlacementAdapter)
    assert ad.backbone_trainable is True
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_placement.py -q` → ImportError (no `placement`).

- [ ] **Step 3: Implement** `src/adapters/placement.py`:

```python
"""Step 6 — serial / parallel in-block placement of a 1x1 Bottleneck adapter.

RQ1 isolates PLACEMENT: the adapter form is a fixed 1x1 channel bottleneck
(the conv analogue of the post-pool Bottleneck), inserted at the final
BasicBlock of each selected ResNet-18 stage via a forward hook.

  serial   : out' = out + body(out)      (sequential / Residual-Sequential)
  parallel : out' = out + body(input)    (parallel   / Residual-Parallel)

body = 1x1 down -> ReLU -> 1x1 up, with up zero-init so the placed model is
identical to the frozen backbone at init. The adapter's own forward is the
identity on the pooled vector — the adaptation happens inside the backbone
(same pattern as LoRAAdapter), so backbone_trainable = True.

Caveat (Conv-Adapter, Chen 2022): a 1x1 adapter "loses locality"; we accept it
to keep the adapter form fixed across placements. See the Step 6 design spec.
"""
from __future__ import annotations
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

_STAGE_ATTRS = ["layer1", "layer2", "layer3", "layer4"]


class Conv1x1Bottleneck(nn.Module):
    """1x1 down -> ReLU -> 1x1 up (NO residual; the placement hook adds it).
    up is zero-init so body(x) == 0 at init."""

    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(rank, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def register_serial_adapter(block: nn.Module, mod: Conv1x1Bottleneck):
    """Forward hook: transform the block OUTPUT in-line (out + body(out))."""
    def hook(_module, _inp, out):
        return out + mod(out)
    return block.register_forward_hook(hook)


def register_parallel_adapter(block: nn.Module, mod: Conv1x1Bottleneck):
    """Forward hook: run body on the block INPUT, sum at the output."""
    def hook(_module, inp, out):
        return out + mod(inp[0])
    return block.register_forward_hook(hook)


class PlacementAdapter(nn.Module):
    """Places 1x1 Bottleneck adapters at the final block of each selected
    stage. Holds the trainable bottlenecks; registers hooks on the backbone.
    Its forward is identity on the pooled feature (adaptation is in-backbone).
    """

    backbone_trainable = True

    def __init__(self, backbone: nn.Module, rank: int, placement: str,
                 block_ids: Optional[Sequence[int]] = None):
        super().__init__()
        if placement not in ("serial", "parallel"):
            raise ValueError(f"placement must be serial|parallel, got {placement!r}")
        if backbone is None:
            raise ValueError("PlacementAdapter requires a backbone")
        self.placement = placement
        self.rank = int(rank)
        ids: List[int] = ([int(i) for i in block_ids]
                          if block_ids is not None else [0, 1, 2, 3])
        self.block_ids = ids
        self.bodies = nn.ModuleList()
        self._handles = []
        register = (register_serial_adapter if placement == "serial"
                    else register_parallel_adapter)
        for sid in ids:
            if not 0 <= sid < len(_STAGE_ATTRS):
                raise ValueError(f"block_id {sid} out of range 0..3")
            block = getattr(backbone, _STAGE_ATTRS[sid])[-1]
            channels = block.conv2.out_channels
            body = Conv1x1Bottleneck(channels, self.rank)
            self.bodies.append(body)
            self._handles.append(register(block, body))
        if len(self.bodies) == 0:
            raise ValueError("PlacementAdapter placed 0 adapters; check block_ids")
        self.num_placed = len(self.bodies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity on pooled feature; adaptation is inside the backbone
```

- [ ] **Step 4: Run tests** — `pytest tests/test_placement.py -q` → PASS (after Task 2 exports exist; if run before, expect ImportError on `PlacementAdapter` — do Task 2 Step 3 first, then this).

---

### Task 2: Route placement through `build_adapter` (`src/adapters/__init__.py`)

**Files:** Modify `src/adapters/__init__.py`
**Interfaces:** Consumes Task 1's `PlacementAdapter`, `Conv1x1Bottleneck`. Produces updated `build_adapter` that maps `{type: bottleneck, placement: serial|parallel}` → `PlacementAdapter`; `post_pool`/absent → `BottleneckAdapter`.

- [ ] **Step 1:** add imports + exports:

```python
from .placement import PlacementAdapter, Conv1x1Bottleneck, \
    register_serial_adapter, register_parallel_adapter
```
(add these names to `__all__`.)

- [ ] **Step 2:** replace the `bottleneck` branch in `build_adapter`:

```python
    if atype == "bottleneck":
        placement = spec.get("placement", "post_pool")
        if placement == "post_pool":
            return BottleneckAdapter(dim=dim, rank=int(spec["rank"]))
        if placement in ("serial", "parallel"):
            if backbone is None:
                raise ValueError(
                    "serial/parallel bottleneck placement requires a backbone")
            return PlacementAdapter(backbone=backbone, rank=int(spec["rank"]),
                                    placement=placement,
                                    block_ids=spec.get("block_ids"))
        raise ValueError(f"Unknown bottleneck placement: {placement!r}")
```

- [ ] **Step 3: Run** — `pytest tests/test_placement.py tests/test_adapters.py -q` → PASS (existing post_pool bottleneck path unchanged: `placement` defaults to `post_pool`).

---

### Task 3: Configs (4 files)

**Files:** Create `configs/exp_phase3_placement_{serial,parallel}_{evidential,softmax}.yaml`

- [ ] **Step 1:** `exp_phase3_placement_serial_evidential.yaml`:

```yaml
# Step 6 (RQ1) — Bottleneck adapter placed SERIAL (in-line at each stage's
# final block) + evidential prototype head. Same 1x1 bottleneck form across
# placements so the comparison isolates PLACEMENT. See step_writeups/step6.txt.
extends: exp_phase2_evidential_retuned.yaml

adapter:
  type: bottleneck
  rank: 16
  placement: serial
  block_ids: [0, 1, 2, 3]

wandb:
  tags: [phase3, step6, placement, serial, evidential, prototype, cifar_fs_bertinetto]
```

- [ ] **Step 2:** `..._parallel_evidential.yaml` — identical but `placement: parallel` and tag `parallel`.
- [ ] **Step 3:** `..._serial_softmax.yaml` / `..._parallel_softmax.yaml` — `extends: exp_phase2_softmax.yaml`, same `adapter` block, softmax tags.
- [ ] **Step 4: Verify configs build a model** — `python -c "from src.utils import load_config; from src.models import build_model; m=build_model(load_config('configs/exp_phase3_placement_parallel_softmax.yaml')); from src.utils import count_trainable_params; print('trainable', count_trainable_params(m))"` → prints ~30,720 + head (0/2).

---

### Task 4: Comparison plot (`scripts/step6_placement_plot.py`)

**Files:** Create `scripts/step6_placement_plot.py`
**Interfaces:** Reads the 4 placement JSONs + 2 post_pool refs (`results/step45_bottleneck_prototype-{head}_metrics.json`); writes `results/step6_placement_comparison.png` (300 dpi).

- [ ] **Step 1:** implement (grouped bars: accuracy, ood_auroc_mean, ece_pooled; x = 3 placements; 2 subplots or hue by head). Skip any missing JSON with a printed warning. 300 dpi. Full code written during build.
- [ ] **Step 2: Run offline smoke** — after the run exists; otherwise assert it errors cleanly on missing files.

---

### Task 5: Writeup skeleton (`step_writeups/step6.txt`)

**Files:** Create `step_writeups/step6.txt`
- [ ] Same structure as `step5.txt`: §0 paper-grounded reasoning (Conv-Adapter parallel>serial prediction), §1 what was built, §2 analytical param counts per placement, §3 how-to-reproduce (Kaggle), §4 RESULTS `[FILL AFTER KAGGLE RUN]`, §5 caveats (1×1 locality, plain-add), §6 exit criteria.

---

### Task 6: Kaggle notebook (`notebooks/step6_placement.ipynb`)

**Files:** Create `notebooks/step6_placement.ipynb`
- [ ] Cells per design §8: GPU check; git clone + pip install; stage data from `/kaggle/input/bpeft-data` (md5 CIFAR-100, copy SVHN→`data/svhn/`, TinyImageNet download-at-runtime fallback); build split; optional pytest; run the 4 placement configs (resumable skip); summary table; `scripts/step6_placement_plot.py`; note Save Version.

---

### Task 7: Full verification

- [ ] **Step 1:** `python -u -m pytest -q` → all pass (91 existing + new placement tests).
- [ ] **Step 2:** print the analytical param counts for each placement/head for the writeup §2.
- [ ] **Step 3:** leave all changes for human review (do NOT commit).
