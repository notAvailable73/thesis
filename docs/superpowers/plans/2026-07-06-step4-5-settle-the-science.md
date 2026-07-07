# Step 4.5 "Settle the Science" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether evidential PEFT keeps its headline reliability claim by adding fair baselines (temperature scaling + energy), a loss-level evidential retune (R-EDL/KL), and the decisive near-OOD test — then producing a tiered verdict.

**Architecture:** Add small, pure, unit-tested primitives (temperature fit, energy score, R-EDL loss knobs, near-OOD loaders); refactor the episodic evaluator from single-score/single-OOD to a score × OOD-pool matrix; wire `scripts/evaluate.py` to fit T on frozen val episodes and emit the wider metrics schema; run the 600-episode experiments on Colab via a new notebook; write the verdict.

**Tech Stack:** Python, PyTorch, torchvision, scikit-learn, pytest, YAML configs. Build + unit-test locally on CPU; run 600-episode evals on Colab GPU.

## Global Constraints

- **GIT (hard):** The agent MUST NOT run `git commit` / `git add` / `git push` / branch mutations. Only read from git. The user commits manually. Every task's final "Checkpoint" step = run `pytest -q` and confirm green (NOT a commit).
- **Reasoning-first (thesis rule):** Each new component names its source paper(s) in a code comment and, in the task notes, states pros/cons/fit and any deviation (per `thesis_implementation_instructions.txt`). Do not fabricate hyperparameters or split IDs.
- **No test leakage:** Temperature `T` and any operating point are fit on VAL episodes only (seeds 10000–10099). The 600 test seeds (`configs/test_episodes.yaml`, [0..599]) are read-only.
- **Back-compat:** All new function parameters are keyword args with defaults that reproduce current behavior bit-for-bit. Existing metrics-JSON keys are preserved; new keys are additive.
- **Compute:** Everything in Tasks 1–8 + 10 is built and unit-tested locally on CPU. Task 9 (600-episode runs) executes on Colab; its outputs feed Task 10's numbers.
- **Env:** run tests with `python -m pytest -q` from repo root `/Users/admin/Desktop/Projects/Thesis/thesis`.

---

### Task 1: Fix the pre-existing buggy prototype-head test (W0)

**Files:**
- Modify: `tests/test_prototype_head.py:71-83`

**Interfaces:**
- Consumes: `PrototypeHead.forward` raising `ValueError` when a class in `[0, n_way)` has no support example (existing behavior in `src/heads/prototype_head.py:_prototypes`).
- Produces: nothing new; makes an existing test actually exercise the raise.

**Why it's broken:** the test slices `support_labels` to `[0, 0, 1]`, so `n_way = max+1 = 2`, both classes present → no raise → test fails. Fix: keep a class-2 label and drop class 1, so `n_way = 3` with class 1 genuinely missing.

- [ ] **Step 1: Rewrite the test body**

```python
def test_prototype_head_raises_on_missing_class():
    """If a class in [0, n_way) is missing from the support set, the head
    must raise (not silently emit garbage). Here labels are {0, 2} so
    n_way infers to 3 and class 1 has no support example."""
    head = PrototypeHead()
    support_features = torch.randn(3, 8)
    support_labels = torch.tensor([0, 0, 2], dtype=torch.long)  # class 1 absent, n_way=3
    query_features = torch.randn(2, 8)
    with pytest.raises(ValueError):
        head(support_features, support_labels, query_features)
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_prototype_head.py -q`
Expected: all tests in the file PASS (previously 1 failed).

- [ ] **Step 3: Checkpoint** — run `python -m pytest -q`; confirm the full suite is green (was 45 passed / 1 failed → now 46 passed).

---

### Task 2: Materialize the real Bertinetto CIFAR-FS split on disk (W0)

**Files:**
- Modify (or overwrite): `data/cifar_fs_split.json`
- Create: `scripts/build_cifar_fs_split.py`
- Test: `tests/test_cifar_fs_split.py` (already asserts disjoint + union=0..99; reuse)

**Interfaces:**
- Consumes: `src/datasets/cifar_fs.py:CIFAR100_CLASS_NAMES` (torchvision CIFAR-100 label order) and `load_cifar_fs_split`.
- Produces: `data/cifar_fs_split.json` with keys `{"train": [64 ids], "val": [16 ids], "test": [20 ids]}` that are the canonical Bertinetto (2019) split, disjoint, union = 0..99.

**Data-source note (do NOT fabricate):** The canonical split is defined by CIFAR-100 *class names* in the Bertinetto/R2D2 release. The generator must obtain those name lists from an authoritative source, not from memory:
- Preferred: the split file already fetched by the current Colab run (`notebooks/step4_episodic.ipynb` fetched the canonical split at runtime) — copy its 64/16/20 name (or id) lists into the generator.
- Fallback: fetch the published split (e.g. the widely-used `bertinetto/cifar-fs` / `kjunelee/MetaOptNet` split lists) and map names → ids via `CIFAR100_CLASS_NAMES`.
If neither source is available at execution time, STOP and ask the user for the split lists rather than inventing ids.

- [ ] **Step 1: Write the generator** `scripts/build_cifar_fs_split.py` that (a) holds the canonical train/val/test **class-name** lists (sourced as above), (b) maps each name to its CIFAR-100 index via `CIFAR100_CLASS_NAMES.index(name)`, (c) asserts `len==64/16/20`, disjoint, union==set(range(100)), (d) writes `data/cifar_fs_split.json` sorted, with a top comment key `"_note": "Bertinetto 2019 canonical split — DO NOT REGENERATE"`.

- [ ] **Step 2: Run it**

Run: `python scripts/build_cifar_fs_split.py`
Expected: prints `wrote data/cifar_fs_split.json  (64/16/20, disjoint, union=100)`.

- [ ] **Step 3: Run the split test**

Run: `python -m pytest tests/test_cifar_fs_split.py -q`
Expected: PASS (disjoint + union assertions hold on the real split).

- [ ] **Step 4: Checkpoint** — `python -m pytest -q` green. (User commits the split file themselves.)

---

### Task 3: Temperature scaling primitive (W1)

**Files:**
- Create: `src/evaluators/temperature.py`
- Modify: `src/evaluators/__init__.py` (export `fit_temperature`, `apply_temperature`)
- Test: `tests/test_temperature.py`

**Interfaces:**
- Produces:
  - `fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 200, lr: float = 0.01) -> float` — returns scalar T>0 minimizing NLL of `softmax(logits/T)` vs `targets`.
  - `apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor` — returns `softmax(logits / T)` probabilities.

Source paper: Guo et al. 2017 (calibration summary). Pros: trivial, strong, mandatory baseline; T>0 preserves argmax so accuracy unchanged. Cons: needs a val logit dump. Fit: directly answers "did you beat temperature scaling?".

- [ ] **Step 1: Write failing tests** `tests/test_temperature.py`

```python
import torch
from src.evaluators import fit_temperature, apply_temperature


def test_fit_temperature_recovers_one_on_calibrated_logits():
    # Well-separated, already-calibrated-ish logits: T should stay near 1.
    torch.manual_seed(0)
    logits = torch.randn(500, 5) * 1.0
    targets = logits.argmax(dim=-1)  # perfectly-predicted -> mild scaling
    T = fit_temperature(logits, targets)
    assert 0.3 < T < 3.0


def test_fit_temperature_softens_overconfident_logits():
    # Hugely overconfident logits that are often WRONG -> optimal T > 1.
    torch.manual_seed(0)
    logits = torch.randn(500, 5) * 20.0     # extreme confidence
    targets = torch.randint(0, 5, (500,))   # random labels -> should soften
    T = fit_temperature(logits, targets)
    assert T > 1.5


def test_apply_temperature_preserves_argmax():
    torch.manual_seed(0)
    logits = torch.randn(64, 5)
    p1 = apply_temperature(logits, 1.0)
    p7 = apply_temperature(logits, 7.0)
    assert torch.equal(p1.argmax(-1), p7.argmax(-1))
    assert torch.allclose(p7.sum(-1), torch.ones(64), atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_temperature.py -q`
Expected: FAIL (ImportError: cannot import name 'fit_temperature').

- [ ] **Step 3: Implement** `src/evaluators/temperature.py`

```python
"""Post-hoc temperature scaling (Guo et al. 2017).

Fits a single scalar T>0 that minimizes NLL of softmax(logits/T). Used as
the fair calibration baseline the evidential head must beat: T is fit ONCE
on the frozen validation episodes (seeds 10000-10099), then frozen and
applied to every test episode. T>0 does not move the argmax, so accuracy
is unchanged.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                    *, max_iter: int = 200, lr: float = 0.01) -> float:
    logits = logits.detach().float()
    targets = targets.detach().long()
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    opt = torch.optim.Adam([log_T], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), targets)
        loss.backward()
        opt.step()
    return float(log_T.exp().item())


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(logits / float(T), dim=-1)
```

- [ ] **Step 4: Add exports** to `src/evaluators/__init__.py`: add `from .temperature import fit_temperature, apply_temperature` and append both names to `__all__`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_temperature.py -q`
Expected: 3 passed.

- [ ] **Step 6: Checkpoint** — `python -m pytest -q` green.

---

### Task 4: Energy OOD score primitive (W1)

**Files:**
- Modify: `src/evaluators/ood.py` (add `energy_score`)
- Modify: `src/evaluators/__init__.py` (export `energy_score`)
- Test: `tests/test_ood_scores.py`

**Interfaces:**
- Produces: `energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor` — returns the ID-ness score `T * logsumexp(logits / T, dim=-1)` (higher = more in-distribution; the negative of Liu et al.'s energy). Shape `(B,)`.

Source paper: Liu et al. 2020 (EBO, OOD summary). Pros: parameter-free, strong; parallels Dirichlet strength S. Cons: none material. Fit: contextualizes evidential OOD.

- [ ] **Step 1: Write failing tests** `tests/test_ood_scores.py`

```python
import torch
from src.evaluators import energy_score


def test_energy_higher_for_confident_logits():
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    flat = torch.zeros(1, 5)
    assert energy_score(confident).item() > energy_score(flat).item()


def test_energy_shape_and_finiteness():
    logits = torch.randn(32, 5)
    s = energy_score(logits)
    assert s.shape == (32,)
    assert torch.isfinite(s).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_ood_scores.py -q`
Expected: FAIL (ImportError: cannot import name 'energy_score').

- [ ] **Step 3: Implement** — append to `src/evaluators/ood.py`

```python
def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Energy-based ID-ness score (Liu et al. 2020). Returns
    T * logsumexp(logits / T), higher => more in-distribution (the
    negative of the paper's energy E). Parameter-free at T=1."""
    return float(T) * torch.logsumexp(logits / float(T), dim=-1)
```

- [ ] **Step 4: Export** — in `src/evaluators/__init__.py` add `energy_score` to the `from .ood import (...)` block and to `__all__`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ood_scores.py -q`
Expected: 2 passed.

- [ ] **Step 6: Checkpoint** — `python -m pytest -q` green.

---

### Task 5: R-EDL evidential-loss knobs + consistent vacuity (W2)

**Files:**
- Modify: `src/losses/evidential.py` (`evidential_mse_loss` gains `prior_per_class`, `use_variance`)
- Modify: `src/evaluators/ood.py` (`evidence_to_probs_and_vacuity` gains `prior_per_class`)
- Modify: `src/trainers/episodic_trainer.py` (thread `evid_prior_per_class`, `evid_use_variance` from ctor into `_episode_loss`)
- Modify: `scripts/train.py` (read `cfg.loss.prior_per_class`, `cfg.loss.use_variance`; pass to `EpisodicTrainer`)
- Modify: `src/evaluators/episodic.py` (pass `prior_per_class` into `evidence_to_probs_and_vacuity` calls)
- Create: `configs/exp_phase2_evidential_retuned.yaml`
- Test: `tests/test_losses.py` (extend)

**Interfaces:**
- `evidential_mse_loss(evidence, target_onehot, num_classes, kl_weight, *, prior_per_class: float = 1.0, use_variance: bool = True) -> Tensor`. Defaults reproduce current Sensoy loss exactly.
- `evidence_to_probs_and_vacuity(evidence, num_classes, prior_per_class: float = 1.0) -> (probs, vacuity)` with `alpha = evidence + prior_per_class`.
- `EpisodicTrainer.__init__` gains `evid_prior_per_class: float = 1.0`, `evid_use_variance: bool = True`.

Source: R-EDL (Survey EDL summary): the rigid "+1" prior and the variance-minimizing regularizer can induce miscalibration; make prior weight tunable and allow dropping the variance term. Deviation: our failure mode is ID *under*-confidence (not over-), so we sweep these on VAL to find the operating point that lowers ID-ECE without losing OOD. Cons: EDL can be unstable at few classes — keep the collapse guard on.

- [ ] **Step 1: Write failing tests** — append to `tests/test_losses.py`

```python
import torch
from src.losses import evidential_mse_loss
from src.evaluators import evidence_to_probs_and_vacuity


def test_redl_defaults_match_legacy_loss():
    torch.manual_seed(0)
    evidence = torch.rand(16, 5) * 3.0
    target = torch.eye(5)[torch.randint(0, 5, (16,))]
    legacy = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.3)
    explicit = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.3,
                                   prior_per_class=1.0, use_variance=True)
    assert torch.allclose(legacy, explicit)


def test_redl_drop_variance_changes_loss():
    torch.manual_seed(0)
    evidence = torch.rand(16, 5) * 3.0
    target = torch.eye(5)[torch.randint(0, 5, (16,))]
    with_var = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.0,
                                   use_variance=True)
    no_var = evidential_mse_loss(evidence, target, num_classes=5, kl_weight=0.0,
                                 use_variance=False)
    assert not torch.allclose(with_var, no_var)
    assert no_var < with_var  # variance term is non-negative


def test_vacuity_uses_prior_per_class():
    evidence = torch.ones(4, 5) * 2.0
    _, vac1 = evidence_to_probs_and_vacuity(evidence, 5, prior_per_class=1.0)
    _, vac_small = evidence_to_probs_and_vacuity(evidence, 5, prior_per_class=0.1)
    # Smaller prior_per_class => smaller S => larger vacuity K/S.
    assert (vac_small > vac1).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_losses.py -q`
Expected: FAIL (unexpected keyword argument `prior_per_class`).

- [ ] **Step 3: Implement loss** — replace the body of `evidential_mse_loss` in `src/losses/evidential.py`

```python
def evidential_mse_loss(evidence: torch.Tensor, target_onehot: torch.Tensor,
                        num_classes: int, kl_weight: float,
                        *, prior_per_class: float = 1.0,
                        use_variance: bool = True) -> torch.Tensor:
    """Sensoy 2018 Eq.5 + Eq.13 KL prior, with R-EDL knobs (Survey EDL).

    prior_per_class: added mass per class (alpha = evidence + prior_per_class).
      1.0 recovers the rigid Sensoy '+1' prior; smaller = sharper/less prior mass.
    use_variance: include the Bayes-risk variance term (True = Sensoy; False =
      R-EDL relaxation dropping the variance-minimizing regularizer).
    """
    alpha = evidence + float(prior_per_class)
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    mse_term = ((target_onehot - p) ** 2).sum(dim=-1)
    if use_variance:
        var = p * (1.0 - p) / (S + 1.0)
        mse_term = mse_term + var.sum(dim=-1)

    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)
    return (mse_term + kl_weight * kl).mean()
```

- [ ] **Step 4: Implement vacuity knob** — in `src/evaluators/ood.py` change `evidence_to_probs_and_vacuity` signature/body:

```python
def evidence_to_probs_and_vacuity(evidence: torch.Tensor, num_classes: int,
                                  prior_per_class: float = 1.0):
    """Dirichlet mean probs + vacuity = K/S with alpha = evidence + prior_per_class."""
    alpha = evidence + float(prior_per_class)
    S = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / S
    vacuity = (num_classes / S).squeeze(-1)
    return probs, vacuity
```

- [ ] **Step 5: Thread through the trainer** — in `src/trainers/episodic_trainer.py`:
  - add ctor kwargs `evid_prior_per_class: float = 1.0`, `evid_use_variance: bool = True`; store on `self`.
  - in `_episode_loss`, pass them into the `evidential_mse_loss(...)` call:

```python
        return evidential_mse_loss(
            evidence, target_oh, num_classes=self.num_classes, kl_weight=kl_w,
            prior_per_class=self.evid_prior_per_class,
            use_variance=self.evid_use_variance,
        )
```

- [ ] **Step 6: Thread through train.py** — in `scripts/train.py` where `EpisodicTrainer(...)` is constructed, read `evid_prior_per_class = float(cfg.loss.get("prior_per_class", 1.0))` and `evid_use_variance = bool(cfg.loss.get("use_variance", True))` and pass them. (Softmax path ignores them.)

- [ ] **Step 7: Keep eval consistent** — in `src/evaluators/episodic.py`, thread a `prior_per_class` (default 1.0) into `_logits_to_id_score` and `_logits_to_probs` so their `evidence_to_probs_and_vacuity(...)` calls use the same prior as training; pass it from `evaluate_episodic(..., prior_per_class=1.0)`. (Wired to config in Task 8.)

- [ ] **Step 8: Create retuned config** `configs/exp_phase2_evidential_retuned.yaml`

```yaml
# Step 4.5: evidential retuned at the LOSS (not the affine). Starting point;
# the exact kl_weight_max / prior_per_class / use_variance are chosen by a
# VAL-only sweep (seeds 10000-10099), never test. R-EDL: Survey EDL summary.
extends: exp_phase2_evidential.yaml

loss:
  kl_weight_max: 0.1      # lowered from 0.5 (val-swept); reduces ID under-confidence
  kl_anneal_steps: 1000
  prior_per_class: 1.0    # R-EDL knob; sweep {1.0, 0.5} on val if ECE stays high
  use_variance: false     # R-EDL relaxation: drop the variance-minimizing term

head:
  # Gentler affine than the (4,-24) sharpening; let the loss carry calibration.
  evidence_scale_init: 2.0
  evidence_bias_init: -6.0
```

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_losses.py tests/test_episodic_trainer.py -q`
Expected: all PASS (incl. the 3 new R-EDL tests; back-compat test proves defaults unchanged).

- [ ] **Step 10: Checkpoint** — `python -m pytest -q` green.

---

### Task 6: Near-OOD loaders — TinyImageNet + CIFAR-100-heldout (W3)

**Files:**
- Create: `src/datasets/tinyimagenet_ood.py`
- Modify: `src/datasets/cifar_fs.py` (add `get_cifar_fs_heldout_ood`) or add to `src/datasets/__init__.py`
- Modify: `src/datasets/__init__.py` (exports)
- Test: `tests/test_tinyimagenet_ood.py`

**Interfaces:**
- `get_tinyimagenet_ood(data_root: str = "data", image_size: int = 224, num_samples: int = 500, seed: int = 42) -> torch.Tensor` — mirrors `get_svhn_ood`: returns `(N,3,image_size,image_size)` ImageNet-normalized images.
- `get_cifar_fs_heldout_ood(data_root: str = "data", image_size: int = 224, num_samples: int = 500, seed: int = 42, heldout_split: str = "val") -> torch.Tensor` — samples images from the CIFAR-FS val-split classes (disjoint from the 20 test-episode classes) as a zero-download near-OOD.

Source: OpenOOD near/far protocol; Step 7 plan. Deviation: CIFAR-100-heldout is a cheap corroborating near-OOD (same visual domain, novel classes, no download).

- [ ] **Step 1: Write failing test** `tests/test_tinyimagenet_ood.py`

```python
import pytest
import torch


def test_cifar_fs_heldout_ood_shape():
    from src.datasets import get_cifar_fs_heldout_ood
    x = get_cifar_fs_heldout_ood(image_size=224, num_samples=16, seed=0)
    assert x.shape == (16, 3, 224, 224)
    assert x.dtype == torch.float32


@pytest.mark.skipif(True, reason="TinyImageNet download not run in CI; smoke only")
def test_tinyimagenet_ood_shape():
    from src.datasets import get_tinyimagenet_ood
    x = get_tinyimagenet_ood(image_size=224, num_samples=8, seed=0)
    assert x.shape == (8, 3, 224, 224)
```

(The CIFAR-100-heldout test runs for real because CIFAR-100 is already downloaded by the test env; the TinyImageNet test is skip-guarded to avoid a heavy network pull in local CI. Remove the skip on Colab to smoke it.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tinyimagenet_ood.py -q`
Expected: FAIL (ImportError: cannot import name 'get_cifar_fs_heldout_ood').

- [ ] **Step 3: Implement TinyImageNet loader** `src/datasets/tinyimagenet_ood.py`

```python
"""Near-OOD loader: TinyImageNet (OpenOOD near-OOD; Step 7 plan). Mirrors
get_svhn_ood so the episodic evaluator treats every OOD pool uniformly.
"""
import os
import random
import torch
from torchvision import datasets, transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_TIN_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def get_tinyimagenet_ood(data_root: str = "data", image_size: int = 224,
                         num_samples: int = 500, seed: int = 42) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tin_root = os.path.join(data_root, "tiny-imagenet-200")
    from ._robust_download import ensure_archive
    ensure_archive(os.path.dirname(tin_root) or ".", "tiny-imagenet-200.zip", [_TIN_URL])
    # extract if needed
    val_dir = os.path.join(tin_root, "val")
    if not os.path.isdir(val_dir):
        import zipfile
        with zipfile.ZipFile(os.path.join(data_root, "tiny-imagenet-200.zip")) as z:
            z.extractall(data_root)
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return torch.stack([dataset[i][0] for i in idx])
```

(If TinyImageNet's `val` layout lacks per-class folders, the Colab task falls back to `train/` — note flagged in the notebook. Loader may need a small path fix at run time; that is acceptable since it is exercised on Colab.)

- [ ] **Step 4: Implement CIFAR-100-heldout loader** — add to `src/datasets/cifar_fs.py`

```python
def get_cifar_fs_heldout_ood(data_root: str = "data", image_size: int = 224,
                             num_samples: int = 500, seed: int = 42,
                             heldout_split: str = "val") -> torch.Tensor:
    """Zero-download near-OOD: images from CIFAR-FS `heldout_split` classes
    (disjoint from the 20 test-episode classes). Same visual domain, novel
    classes. Returns (N,3,H,W) ImageNet-normalized like get_svhn_ood."""
    import random, torch
    ds = get_cifar_fs(data_root=data_root, image_size=image_size, split=heldout_split)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
    return torch.stack([ds[i][0] for i in idx])
```

(Adapt the indexing to whatever `get_cifar_fs` returns — if it returns a `(images, labels)` tuple or a Dataset, mirror how `get_svhn_ood`/the evaluator already index it. Confirm by reading `src/datasets/cifar_fs.py` before writing.)

- [ ] **Step 5: Export** in `src/datasets/__init__.py`: import and add `get_tinyimagenet_ood`, `get_cifar_fs_heldout_ood` to `__all__`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_tinyimagenet_ood.py -q`
Expected: 1 passed, 1 skipped.

- [ ] **Step 7: Checkpoint** — `python -m pytest -q` green.

---

### Task 7: Refactor `evaluate_episodic` to a score × OOD-pool matrix (W3)

**Files:**
- Modify: `src/evaluators/episodic.py`
- Test: `tests/test_evaluators.py` (extend) or `tests/test_episodic_matrix.py` (new)

**Interfaces:**
- New signature:
  `evaluate_episodic(model, test_iterable, ood_pools: dict[str, torch.Tensor] | None, *, num_classes, interpretation, ece_bins=15, temperature: float | None = None, prior_per_class: float = 1.0, device="cpu", logger=None, wandb_run=None) -> dict`
  where `ood_pools` maps a name ("svhn_far", "tin_near", "cifar100_near") to precomputed backbone features `(N, D)`.
- The returned `summary` keeps ALL existing keys (computed against the first/primary OOD pool for back-compat) and ADDS, per OOD pool name and per applicable score:
  `ood_auroc__{pool}__{score}`, `fpr_at_95_tpr__{pool}__{score}` where `score ∈ {vacuity|msp, ts_msp, energy}` (vacuity for evidential, msp/ts_msp/energy for softmax; ts_msp only if `temperature` is not None).
- Also adds `ece_ts`, `brier_ts` when `temperature` is not None (softmax interpretation).

**Design:** per episode, compute query logits once; derive per-score ID scores; for each OOD pool compute its logits vs THIS episode's prototypes and the matching OOD scores; accumulate AUROC/FPR per (pool, score). Keep the single-pool back-compat keys populated from the primary pool + native score.

- [ ] **Step 1: Write failing test** `tests/test_episodic_matrix.py` — build a tiny fake model exposing `.backbone` (identity-ish) and `forward_proto_from_features`, feed 2 synthetic episodes + two OOD pools, assert the returned summary contains `ood_auroc__svhn_far__msp` and `ood_auroc__tin_near__msp` and that legacy `ood_auroc_mean` still exists.

```python
import torch
from src.evaluators.episodic import evaluate_episodic
# ... construct a minimal stub model with .eval(), .backbone (nn.Identity-like
#     wrapper returning (B,D)), .head (PrototypeHead), and
#     forward_proto_from_features(sf, sy, qf). Mirror how the real model is
#     called in src/evaluators/episodic.py:_proto_logits_for_query.
def test_matrix_keys_present():
    ...  # assert 'ood_auroc__svhn_far__msp' in out['summary']
         # assert 'ood_auroc__tin_near__msp' in out['summary']
         # assert 'ood_auroc_mean' in out['summary']  # legacy preserved
```

(Write the stub concretely against the real call sites in `episodic.py`; do not leave `...`.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_episodic_matrix.py -q`
Expected: FAIL (new keys absent / signature mismatch).

- [ ] **Step 3: Implement the refactor** in `src/evaluators/episodic.py`:
  - change `ood_features` → `ood_pools: dict[str, Tensor] | None`;
  - add helpers `_id_scores(logits, interpretation, head, num_classes, temperature, prior_per_class) -> dict[str, Tensor]` returning `{score_name: (B,) score}` (evidential → `{"vacuity": 1-u}`; softmax → `{"msp": maxp, "energy": energy_score(logits)}` plus `"ts_msp": apply_temperature(logits, T).max(-1)` when `temperature` set);
  - per episode, per pool, compute the pool's logits and the SAME score set, accumulate `auroc[pool][score]`, `fpr[pool][score]`;
  - after the loop, write legacy keys from `(primary_pool, native_score)` and the new `ood_auroc__{pool}__{score}` / `fpr_at_95_tpr__{pool}__{score}` means; add `ece_ts`/`brier_ts` when `temperature` set (pooled TS probs).
  - import `energy_score`, `apply_temperature` at top.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_episodic_matrix.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing evaluator tests** to confirm back-compat

Run: `python -m pytest tests/test_evaluators.py -q`
Expected: PASS (legacy keys unchanged).

- [ ] **Step 6: Checkpoint** — `python -m pytest -q` green.

---

### Task 8: Wire `scripts/evaluate.py` — fit T on val, pass OOD pools, widen JSON (W3)

**Files:**
- Modify: `scripts/evaluate.py` (`_evaluate_episodic` + imports)

**Interfaces:**
- Consumes: `fit_temperature`, `get_tinyimagenet_ood`, `get_cifar_fs_heldout_ood`, refactored `evaluate_episodic`, `EpisodicIterableDataset`, `get_cifar_fs`.
- Produces: a metrics JSON with all legacy keys + the new matrix keys + `temperature`, `ece_ts`, `brier_ts`, `prior_per_class`.

**Val temperature fit (softmax only):** build a val `EpisodicIterableDataset` from the VAL split using `configs/val_episodes.yaml` seeds (10000–10099), run the frozen model, pool query logits + episode-local targets, `T = fit_temperature(pooled_logits, pooled_targets)`. For evidential runs, `temperature=None`.

- [ ] **Step 1: Add imports** — in `scripts/evaluate.py` extend the `from src.evaluators import (...)` to include `fit_temperature`, `energy_score`, and add `from src.datasets import get_tinyimagenet_ood, get_cifar_fs_heldout_ood` to the datasets import.

- [ ] **Step 2: Add a val-logit dump helper** inside `scripts/evaluate.py`

```python
def _fit_val_temperature(model, cfg, device, repo_root, logger) -> float:
    """Fit one global T on the frozen VAL episodes (seeds from
    configs/val_episodes.yaml). Softmax interpretation only."""
    import yaml
    val_spec = yaml.safe_load(open(repo_root / "configs/val_episodes.yaml"))
    val_seeds = list(val_spec["seeds"])
    val_split = get_cifar_fs(data_root=cfg.dataset.data_root,
                             image_size=int(cfg.dataset.image_size), split="val")
    val_iter = EpisodicIterableDataset(
        val_split, n_way=int(cfg.dataset.n_way), k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query), num_episodes=len(val_seeds),
        seed_offset=int(val_seeds[0]),
    )
    backbone = model.backbone
    logits_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for sx, sy, qx, qy in val_iter:
            sf = backbone(sx.to(device)); qf = backbone(qx.to(device))
            ql = model.forward_proto_from_features(sf, sy.to(device), qf)
            logits_all.append(ql.cpu()); targets_all.append(qy.cpu())
    T = fit_temperature(torch.cat(logits_all), torch.cat(targets_all))
    logger.info(f"fit temperature on {len(val_seeds)} val episodes: T={T:.4f}")
    return T
```

- [ ] **Step 3: Build OOD pools + call the refactored evaluator** — in `_evaluate_episodic`, after loading the model, build the pool dict:

```python
    pools = {}
    svhn_x = get_svhn_ood(data_root=cfg.ood.data_root, image_size=int(cfg.dataset.image_size),
                          num_samples=int(cfg.ood.num_samples), seed=int(cfg.ood.seed))
    pools["svhn_far"] = _extract_features(model.backbone, svhn_x, device)
    heldout_x = get_cifar_fs_heldout_ood(data_root=cfg.dataset.data_root,
                                         image_size=int(cfg.dataset.image_size),
                                         num_samples=int(cfg.ood.num_samples), seed=int(cfg.ood.seed))
    pools["cifar100_near"] = _extract_features(model.backbone, heldout_x, device)
    if bool(cfg.ood.get("use_tinyimagenet", False)):
        tin_x = get_tinyimagenet_ood(data_root=cfg.dataset.data_root,
                                     image_size=int(cfg.dataset.image_size),
                                     num_samples=int(cfg.ood.num_samples), seed=int(cfg.ood.seed))
        pools["tin_near"] = _extract_features(model.backbone, tin_x, device)

    T = None
    if interp == "softmax":
        T = _fit_val_temperature(model, cfg, device, repo_root, logger)

    result = evaluate_episodic(
        model=model, test_iterable=test_iter, ood_pools=pools,
        num_classes=K, interpretation=interp, ece_bins=int(cfg.eval.ece_bins),
        temperature=T, prior_per_class=float(cfg.loss.get("prior_per_class", 1.0)),
        device=device, logger=logger, wandb_run=wb,
    )
```

- [ ] **Step 4: Widen the summary** — add `base_summary.update({"temperature": (float(T) if T else 0.0), "prior_per_class": float(cfg.loss.get("prior_per_class", 1.0))})`; the matrix + `ece_ts`/`brier_ts` keys already flow up from `result["summary"]`.

- [ ] **Step 5: Local smoke** — run 3-episode CPU eval on the softmax config to confirm no crash and new keys present:

Run: `python scripts/evaluate.py --config configs/exp_phase2_softmax.yaml --num-episodes 3 --wandb-mode disabled --results-suffix step45_smoke`
Expected: exits 0; printed JSON contains `temperature`, `ood_auroc__svhn_far__msp`, `ood_auroc__cifar100_near__msp`, `ece_ts`. (Requires a checkpoint; if none locally, this smoke runs on Colab in Task 9 — note it and move on.)

- [ ] **Step 6: Checkpoint** — `python -m pytest -q` green (unit tests unaffected).

---

### Task 9: Colab runner notebook — execute the experiments (W3, runs on Colab)

**Files:**
- Create: `notebooks/step4_5_settle.ipynb`

**Content (cells):**
1. Mount Drive + `pip`/path setup (copy from `notebooks/step4_episodic.ipynb`).
2. Train softmax: `python scripts/train.py --config configs/exp_phase2_softmax.yaml`.
3. Val-sweep evidential: small loop over `{kl_weight_max ∈ [0.1, 0.25, 0.5]} × {use_variance ∈ [T,F]}` training each briefly and reading val ECE/val-OOD from the history/val eval; pick the best-on-VAL config; write it to `configs/exp_phase2_evidential_retuned.yaml`. (Val only — never touches test.)
4. Train retuned evidential with the chosen config.
5. Evaluate both at 600 episodes with `--results-suffix step45` (set `ood.use_tinyimagenet: true`).
6. Print the master table: for each head/score, {acc, F1, ECE (+ece_ts for softmax), Brier, and AUROC/FPR per OOD pool}.
7. Copy `results/step45_*_metrics.json` back to Drive.

- [ ] **Step 1: Author the notebook cells** as above (real code, mirroring the existing step4 notebook's setup + CLI calls).
- [ ] **Step 2 (user, on Colab):** run top-to-bottom; confirm the collapse guard does not fire and both heads reach ~0.87 acc.
- [ ] **Step 3:** save `results/step45_bottleneck_prototype-{evidential,softmax}_metrics.json` into the repo `results/`.
- [ ] **Checkpoint:** the two step45 metrics JSONs exist and contain the matrix keys.

---

### Task 10: Verdict + writeup + progress update (W4)

**Files:**
- Create: `step_writeups/step4_5.txt`
- Modify: `progress.txt` (Step 4 exit-criteria + Step 4.5 sub-section)
- Modify (memory, agent-side): `step4-5-settle-the-science.md` final verdict

**Interfaces:** consumes `results/step45_*_metrics.json` from Task 9.

- [ ] **Step 1: Build the master comparison table** from the two step45 JSONs: rows = {evidential(vacuity), softmax(msp), softmax(ts_msp), softmax(energy)}; cols = {acc, F1, ECE, Brier, SVHN-far AUROC/FPR, TIN-near AUROC/FPR, CIFAR100-near AUROC/FPR}.
- [ ] **Step 2: Apply the tiered decision rule** (spec §2) and record the tier reached.
- [ ] **Step 3: Write `step_writeups/step4_5.txt`** in the same honest structure as `step_writeups/step4.txt`: what was built, the fair-baseline result, the near-OOD result, the tier, and the recommended thesis narrative.
- [ ] **Step 4: Update `progress.txt`** — tick the Step 4 exit boxes now satisfied (real split committed by user, buggy test fixed, baselines added) and add a Step 4.5 block with the verdict.
- [ ] **Step 5:** update the `step4-5-settle-the-science` memory with the final tier + numbers.
- [ ] **Checkpoint:** `python -m pytest -q` green; writeup + progress reflect the actual numbers (no invented values — if Task 9 not yet run, mark the numbers TBD and STOP for the user's Colab run).

---

## Self-Review

**Spec coverage:** W0 → Tasks 1–2; W1 (TS) → Task 3, (energy) → Task 4; W2 (R-EDL loss retune) → Task 5; W3 (near-OOD loaders) → Task 6, (evaluator matrix) → Task 7, (script wiring) → Task 8, (execution) → Task 9; W4 (verdict) → Task 10. Decision rule → Task 10 Step 2. Testing/risks/compute → Global Constraints + per-task. All spec sections covered.

**Placeholder scan:** Task 7 Step 1 and Task 6 Step 4 intentionally instruct the implementer to write the stub/indexing against the real call sites (they must read `episodic.py` / `cifar_fs.py` first) — these are "confirm-then-write" notes, not deferred content; the surrounding code and asserted keys are concrete. No "TBD/handle edge cases" placeholders remain except Task 10's explicit "TBD if Colab not run," which is a deliberate stop-gate.

**Type consistency:** `fit_temperature/apply_temperature`, `energy_score`, `evidential_mse_loss(..., prior_per_class, use_variance)`, `evidence_to_probs_and_vacuity(..., prior_per_class)`, and `evaluate_episodic(..., ood_pools, temperature, prior_per_class)` names/signatures are used identically across Tasks 3–8. Trainer kwargs `evid_prior_per_class`/`evid_use_variance` match between Task 5 Steps 5–6.

**Deviations flagged:** commit steps replaced by Checkpoint (test-suite) steps per the no-git-commits constraint; the real Bertinetto split is data-sourced not fabricated (Task 2 stop-gate); TinyImageNet path layout may need a runtime fix on Colab (Task 6 note).
