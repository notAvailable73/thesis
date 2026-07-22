# Step 7 — RQ3 far-OOD Gaussian + consolidation Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. Implement task-by-task, running tests after each.

**Goal:** Add a seeded Gaussian-noise far-OOD pool, wire it into `evaluate.py` (`--use-gaussian`), re-eval the Step-6 winner (parallel, both heads) across all 4 OOD pools, consolidate into `phase4_ood_table.json` + a bar chart, and write the RQ3 paragraph.

**Architecture:** `get_gaussian_ood` returns backbone-ready `N(0,1)` tensors; `evaluate.py` adds a `gaussian_far` pool exactly like `tin_near`; `evaluate_episodic` already emits AUROC + FPR@95 per pool (no evaluator change). A consolidation script reads the winner JSONs into the RQ3 table.

**Tech Stack:** PyTorch, pytest, matplotlib. Spec: `docs/superpowers/specs/2026-07-22-step7-far-ood-gaussian-design.md`.

## Global Constraints

- **Do NOT `git commit`** (human reviews/commits).
- Frozen protocol: no change to image_size 224, 600 test seeds, cudnn.deterministic.
- Gaussian pool: seeded `N(0,1)`, clamp `[-3,3]`, shape `(num_samples,3,224,224)`, deterministic (`torch.Generator`).
- `svhn_far` stays the primary pool; Gaussian is additive.
- Tests run offline (no network/GPU).

---

### Task 1: Gaussian OOD pool (`src/datasets/gaussian_noise_ood.py`)

**Files:** Create `src/datasets/gaussian_noise_ood.py`; Modify `src/datasets/__init__.py`; Test `tests/test_gaussian_ood.py`

**Interfaces:**
- Produces: `get_gaussian_ood(image_size=224, num_samples=500, seed=42, clamp=3.0) -> torch.Tensor` shape `(num_samples,3,image_size,image_size)`.

- [ ] **Step 1: failing tests** `tests/test_gaussian_ood.py`:

```python
import torch
from src.datasets import get_gaussian_ood

def test_shape_and_dtype():
    x = get_gaussian_ood(image_size=224, num_samples=500, seed=42)
    assert x.shape == (500, 3, 224, 224)
    assert x.dtype == torch.float32

def test_reproducible_same_seed():
    a = get_gaussian_ood(num_samples=16, seed=7)
    b = get_gaussian_ood(num_samples=16, seed=7)
    assert torch.equal(a, b)

def test_different_seed_differs():
    a = get_gaussian_ood(num_samples=16, seed=1)
    b = get_gaussian_ood(num_samples=16, seed=2)
    assert not torch.equal(a, b)

def test_clamped_range():
    x = get_gaussian_ood(num_samples=32, seed=3, clamp=3.0)
    assert float(x.min()) >= -3.0 and float(x.max()) <= 3.0
```

- [ ] **Step 2: run → fail** `pytest tests/test_gaussian_ood.py -q` → ImportError.

- [ ] **Step 3: implement** `src/datasets/gaussian_noise_ood.py`:

```python
"""Far-OOD Gaussian-noise pool (Step 7, RQ3).

Pure far-OOD sanity/ablation: seeded N(0,1) noise at 3x224x224, clamped to
[-clamp, clamp]. Returned in the same backbone-ready form as get_svhn_ood
(ImageNet-normalized real images sit at ~unit per-channel scale, so raw N(0,1)
is the right magnitude to feed the frozen backbone directly). Deterministic via
a torch.Generator so the pool is byte-identical across runs.
"""
import torch


def get_gaussian_ood(image_size: int = 224, num_samples: int = 500,
                     seed: int = 42, clamp: float = 3.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(int(num_samples), 3, int(image_size), int(image_size),
                    generator=g)
    if clamp is not None:
        x = x.clamp(-float(clamp), float(clamp))
    return x
```

- [ ] **Step 4:** add to `src/datasets/__init__.py`: `from .gaussian_noise_ood import get_gaussian_ood` and add `"get_gaussian_ood"` to `__all__`.

- [ ] **Step 5: run → pass** `pytest tests/test_gaussian_ood.py -q` → 4 passed.

---

### Task 2: Wire `--use-gaussian` into `scripts/evaluate.py`

**Files:** Modify `scripts/evaluate.py` (imports, pools block, argparse)
**Interfaces:** Consumes Task 1's `get_gaussian_ood`. Produces pool key `"gaussian_far"` in the OOD matrix (`ood_auroc__gaussian_far__*`, `fpr_at_95_tpr__gaussian_far__*`).

- [ ] **Step 1:** add `get_gaussian_ood` to the `from src.datasets import (...)` block.

- [ ] **Step 2:** after the `tin_near` block (right before the `logger.info("OOD pools: ...")` line), add:

```python
    # Gaussian far-OOD (Step 7): pure-noise sanity/ablation pool.
    use_gauss = bool(cfg.ood.get("use_gaussian", False)) or bool(
        getattr(args, "use_gaussian", False))
    if use_gauss:
        gauss_x = get_gaussian_ood(image_size=img_size, num_samples=n_ood,
                                   seed=ood_seed)
        pools["gaussian_far"] = _extract_features(model.backbone, gauss_x, device)
```

- [ ] **Step 3:** add the argparse flag next to `--use-tinyimagenet`:

```python
    parser.add_argument("--use-gaussian", action="store_true",
                        help="Add the Gaussian-noise far-OOD pool at eval time.")
```

- [ ] **Step 4: verify** the flag parses and the pool builds offline (small, CPU, 2 episodes):

```bash
python -c "
import sys; sys.argv=['x']
from src.datasets import get_gaussian_ood
p=get_gaussian_ood(num_samples=8); print('gaussian pool', tuple(p.shape))
"
```
Expected: `gaussian pool (8, 3, 224, 224)`.

---

### Task 3: Consolidation script (`scripts/step7_ood_consolidate.py`)

**Files:** Create `scripts/step7_ood_consolidate.py`
**Interfaces:** Reads `results/phase4_parallel_bottleneck_prototype-{head}_metrics.json`; writes `results/phase4_ood_table.json` + `results/step7_ood_comparison.png`.

- [ ] **Step 1:** implement — for each head read native-score AUROC/FPR@95 per pool
  (`vacuity` for evidential, `msp` for softmax) via keys
  `ood_auroc__{pool}__{score}` / `fpr_at_95_tpr__{pool}__{score}`; map pool→name
  (`svhn_far`→svhn, `gaussian_far`→gaussian, `cifar100_near`→cifar100,
  `tin_near`→tinyimagenet); write the table JSON; draw a far-vs-near grouped bar
  chart (AUROC), 300 dpi. Skip missing JSONs with a warning. Full code at build.

- [ ] **Step 2: smoke** — runs and warns cleanly when the phase4 JSONs don't exist yet.

---

### Task 4: Writeup skeleton (`step_writeups/step7.txt`)

**Files:** Create `step_writeups/step7.txt`
- [ ] Structure like step6.txt: §0 paper-grounded reasoning (near vs far, Gaussian = sanity), §1 what was built, §2 OOD pools table, §3 reproduce (Kaggle), §4 RESULTS `[FILL AFTER KAGGLE RUN]` (far: svhn+gaussian; near: cifar100+tin; both heads), §5 caveats (TinyImageNet uncurated → RQ3 rests on cifar100_near), §6 exit criteria.

---

### Task 5: Kaggle notebook (`notebooks/step7_ood.ipynb`)

**Files:** Create `notebooks/step7_ood.ipynb`
- [ ] Same cells as step6 (GPU check; git clone + pip; resilient CIFAR/SVHN staging from `bpeft-data`; build split; optional pytest incl. `test_gaussian_ood.py`), then: run the 2 parallel configs `train.py` + `evaluate.py --num-episodes 600 --use-tinyimagenet --use-gaussian --results-suffix phase4_parallel` (resumable skip); run `scripts/step7_ood_consolidate.py`; show `phase4_ood_table.json` + `step7_ood_comparison.png`; persist note.

---

### Task 6: Full verification

- [ ] `python -u -m pytest -q` → all pass (103 + 4 new gaussian tests = 107).
- [ ] Leave all changes for human review (no commit).
