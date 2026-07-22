# Step 7 — RQ3: far-OOD Gaussian + OOD consolidation — Design

**Date:** 2026-07-22
**Phase:** 4 · **RQ:** RQ3 (near-OOD is the hard case)
**Spec source:** implementation.txt §7 · progress.txt STEP 7
**Depends on:** Step 6 (CLOSED — parallel placement is the winner)

## 1. Goal & scope (decided: "tight consolidation")

Close RQ3 by adding a **far-OOD Gaussian-noise** sanity pool to the OOD matrix,
evaluating it on the **Step-6 winner (parallel placement, both heads)**, and
writing the RQ3 paragraph. The near-OOD infrastructure already exists
(TinyImageNet loader + multi-pool AUROC/FPR@95 evaluator), and RQ3's near-OOD
claim already rests on the **provably-disjoint `cifar100_near`** pool (16 val
classes, disjoint from the 20 test classes).

**In scope:** Gaussian pool, `--use-gaussian` wiring, re-eval of the 2 parallel
configs with the full OOD list, a consolidated `phase4_ood_table.json` + bar
chart, tests, writeup, Kaggle notebook.

**Out of scope (per the tight choice):** TinyImageNet class curation (kept
uncurated with a documented caveat); re-running Steps 4.5/5/6; Gaussian on
non-winner configs.

## 2. Paper grounding (thesis_implementation_instructions §5)

ODD summaries / OpenOOD near-vs-far split + Malinin & Gales: evidential
(Dirichlet) uncertainty is expected to hold on **near**-OOD where softmax-MSP
degrades; **far**-OOD (SVHN, Gaussian noise) is the easy sanity end. Gaussian
noise is the pure far-OOD ablation — if the model can't separate white noise
from CIFAR features, something is wrong. PROS: cheap, unambiguous far-OOD.
CONS/FIT: trivially easy, so it's a sanity check, not a discriminating test —
the discriminating RQ3 evidence stays the near-OOD pools.

## 3. Components

**3.1 `src/datasets/gaussian_noise_ood.py` (new)**
- `get_gaussian_ood(image_size=224, num_samples=500, seed=42, clamp=3.0)` →
  `(num_samples, 3, image_size, image_size)` float tensor of seeded `N(0,1)`
  noise clamped to `[-clamp, clamp]`. Same backbone-ready contract as
  `get_svhn_ood` (ImageNet-normalized real images sit at ~unit per-channel
  scale, so raw `N(0,1)` is the correct magnitude to feed the frozen backbone).
  Deterministic via `torch.Generator().manual_seed(seed)`. Export in
  `src/datasets/__init__.py`.

**3.2 `scripts/evaluate.py` (modify, additive — mirrors `--use-tinyimagenet`)**
- Add `--use-gaussian` flag + `cfg.ood.use_gaussian`. When set, add pool
  `"gaussian_far"` via `_extract_features`. `evaluate_episodic` already loops
  pools generically and emits AUROC + FPR@95 per pool → **no evaluator change**.
  `svhn_far` stays the primary pool (legacy single-score keys unchanged).

**3.3 Runs — parallel winner, both heads**
- Re-run `exp_phase3_placement_parallel_{evidential,softmax}` (train + eval)
  with `--use-tinyimagenet --use-gaussian --results-suffix phase4_parallel`.
  Kaggle wipes checkpoints, so this retrains — but parallel is cheap (no
  Full-FT), and with fixed seeds the svhn/cifar100/tin numbers reproduce Step 6
  exactly, now with an added `gaussian_far` column. Step-6 JSONs stay untouched.

**3.4 `scripts/step7_ood_consolidate.py` (new)**
- Read the 2 `results/phase4_parallel_*` JSONs → write
  `results/phase4_ood_table.json`:
  ```
  {"config": "exp_phase3_placement_parallel",
   "n_id": "600 episodes x 75 query", "n_ood": 500,
   "heads": {"evidential": {"svhn": {auroc,fpr_at_95}, "gaussian": {...},
                            "cifar100": {...}, "tinyimagenet": {...}},
             "softmax":    {... msp native ...}}}
  ```
  (native score = vacuity for evidential, msp for softmax). Plus a far-vs-near
  AUROC grouped bar chart → `results/step7_ood_comparison.png` (300 dpi).

**3.5 Tests — `tests/test_gaussian_ood.py`**
- shape `(500,3,224,224)` + float dtype; **reproducible** (same seed → identical
  tensor, `torch.equal`); values within `[-3,3]`; different seed → different pool.

**3.6 Writeup — `step_writeups/step7.txt`**
- Structure like step5/6. RQ3 paragraph: evidential-vs-softmax gap on **far**
  (SVHN, Gaussian) vs **near** (CIFAR100-clean, TinyImageNet), honest in either
  direction. Documented caveat: TinyImageNet uncurated → near-OOD claim rests on
  `cifar100_near`. `[FILL AFTER KAGGLE RUN]` for the Gaussian numbers.

**3.7 Kaggle notebook — `notebooks/step7_ood.ipynb`**
- Same pattern as Step 6 (git clone + `bpeft-data`, resilient CIFAR staging):
  run the 2 parallel configs with the extended OOD list, consolidate, plot.

## 4. Exit criteria (implementation.txt §7)

- [ ] AUROC + FPR@95 for ≥2 OOD datasets (SVHN + TinyImageNet) for the best
      config × both heads — plus Gaussian + CIFAR100 (4 pools total).
- [ ] RQ3 paragraph: far-OOD vs near-OOD evidential-vs-softmax gap, honest.
- [ ] `pytest tests/` still passes (`test_gaussian_ood.py` added).

## 5. Non-goals / YAGNI

- No TinyImageNet curation, no re-run of prior steps, no new evaluator code
  paths (Gaussian reuses the generic pool loop), not committed by the assistant.
