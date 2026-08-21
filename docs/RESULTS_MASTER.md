# B-PEFT — Master Results and Positioning Against the State of the Art

**Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**

| | |
|---|---|
| **Source of every number below** | `results/mvt_results.json` (produced by `scripts/aggregate_grid.py` from the 120 Step-10 grid runs) |
| **Grid completeness** | 120 / 120 cells present, 40 / 40 groups with all 3 seeds, `missing_cells: []` |
| **Run log** | `results/grid/_run_log.jsonl` — 123 `ok`, 1 `skipped_done`, 0 errors |
| **Protocol** | 5-way {1,5}-shot, 600 frozen TEST episodes (`configs/test_episodes.yaml`, seeds 0–599), 3 training seeds (42/43/44) per cell |
| **Generated** | 2026-08-06, from the aggregated JSON — no number in this file was typed by hand |

> **Reproducing this document:** run `python scripts/make_results_master.py`. Every table is emitted mechanically from `results/mvt_results.json`; the prose lives in `docs/RESULTS_MASTER_template.md`. **Edit the template, never this file** — the next run of the script overwrites it.

---

## 1. How to read these tables

### 1.1 The grid axes

| Axis | Values |
|---|---|
| Dataset | CIFAR-FS (Bertinetto split), MiniImageNet (Ravi & Larochelle split) |
| Shots | 1-shot, 5-shot (5-way throughout) |
| Backbone | ResNet-18, MobileNetV3-Small — both **ImageNet-pretrained and fully frozen** |
| Adapter | Bottleneck-parallel, LoRA, **Full-FT\***, **Linear-Probe\*** (\* = mandatory baselines, ResNet-18 / CIFAR-FS only) |
| Head interpretation | Evidential (Dirichlet) vs. Softmax — both on the same parameter-free `PrototypeHead` |
| Seeds | 42, 43, 44 |

40 unique configurations × 3 seeds = 120 runs.

### 1.2 Metric definitions

| Metric | Meaning | Direction |
|---|---|---|
| **Accuracy** | Mean 5-way episode accuracy over 600 frozen test episodes | higher better |
| **Macro-F1** | Per-episode macro-averaged F1 over the 5 episode classes, then averaged over episodes. In a balanced 5-way episode it tracks accuracy closely; the gap between them measures per-class imbalance in the errors | higher better |
| **ECE (pooled)** | Expected Calibration Error over all query predictions pooled across episodes | lower better |
| **ECE (per-episode)** | ECE computed inside each episode, then averaged — the harsher, small-sample variant | lower better |
| **ECE after TS** | ECE after post-hoc temperature scaling (softmax cells only — temperature scaling is not defined for a Dirichlet evidence mapping) | lower better |
| **Brier** | Multi-class Brier score of the predictive distribution | lower better |
| **OOD AUROC** | Area under ROC separating in-distribution query embeddings from an OOD set, using the head's uncertainty score | higher better |
| **FPR@95%TPR** | False-positive rate when 95 % of in-distribution samples are retained | lower better |

### 1.3 Uncertainty scores compared

| Head | Score | Definition |
|---|---|---|
| Evidential | `vacuity` | K / Σα, the Dirichlet vacuity from `PrototypeHead.to_evidence()` |
| Softmax | `msp` | maximum softmax probability |
| Softmax | `ts_msp` | maximum softmax probability after temperature scaling |
| Softmax | `energy` | logsumexp energy score (training-free, non-probabilistic) |

### 1.4 OOD sets

| Set | Type | Applies to |
|---|---|---|
| SVHN | far-OOD | both datasets |
| Gaussian noise | far-OOD | both datasets |
| CIFAR-100-heldout | near-OOD | CIFAR-FS |
| MiniImageNet-heldout | near-OOD | MiniImageNet |
| TinyImageNet | near-OOD | both datasets |

Cells marked `--` mean that OOD set does not apply to that dataset, or that the metric is undefined for that head (e.g. temperature scaling for evidential).

---

## 2. Full results

### Table 1 — Accuracy, macro-F1 and trainable parameters (all 40 configurations)

| Dataset | Shots | Backbone | Adapter | Head | Trainable params | Accuracy % (±95% CI over 600 episodes) | Acc. seed std % | Macro-F1 % (±95% CI) | F1 seed std % |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 79.19 ± 0.82 | 0.428 | 78.29 ± 0.87 | 0.407 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 78.57 ± 0.83 | 0.481 | 77.50 ± 0.90 | 0.450 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Evid. | 12,290 | 73.76 ± 0.87 | 1.090 | 72.61 ± 0.94 | 1.181 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Softmax | 12,288 | 75.44 ± 0.85 | 0.368 | 74.33 ± 0.91 | 0.417 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Evid. | 11,176,514 | 80.36 ± 0.82 | 0.000 | 79.14 ± 0.91 | 0.000 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Softmax | 11,176,512 | 81.14 ± 0.79 | 0.000 | 80.02 ± 0.88 | 0.000 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Evid. | 2 | 70.25 ± 0.84 | 0.000 | 69.10 ± 0.89 | 0.000 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Softmax | 0 | 70.25 ± 0.84 | 0.000 | 69.10 ± 0.89 | 0.000 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 78.88 ± 0.82 | 0.066 | 77.83 ± 0.89 | 0.107 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 78.80 ± 0.83 | 0.319 | 77.69 ± 0.90 | 0.414 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Evid. | 10,754 | 74.03 ± 0.88 | 0.519 | 72.47 ± 0.97 | 0.500 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Softmax | 10,752 | 75.43 ± 0.87 | 0.306 | 74.09 ± 0.95 | 0.382 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 91.58 ± 0.50 | 0.171 | 91.46 ± 0.51 | 0.171 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 91.44 ± 0.51 | 0.137 | 91.29 ± 0.52 | 0.132 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Evid. | 12,290 | 83.33 ± 0.63 | 1.071 | 82.96 ± 0.66 | 1.071 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Softmax | 12,288 | 86.25 ± 0.59 | 0.172 | 85.98 ± 0.62 | 0.191 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Evid. | 11,176,514 | 88.82 ± 0.54 | 0.000 | 88.56 ± 0.56 | 0.000 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Softmax | 11,176,512 | 90.47 ± 0.50 | 0.000 | 90.27 ± 0.52 | 0.000 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Evid. | 2 | 87.41 ± 0.56 | 0.000 | 87.28 ± 0.57 | 0.000 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Softmax | 0 | 87.41 ± 0.56 | 0.000 | 87.28 ± 0.57 | 0.000 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 90.24 ± 0.53 | 0.178 | 90.08 ± 0.54 | 0.184 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 90.74 ± 0.52 | 0.167 | 90.59 ± 0.54 | 0.186 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Evid. | 10,754 | 86.97 ± 1.82 | 0.738 | 86.73 ± 1.84 | 0.717 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Softmax | 10,752 | 88.05 ± 0.57 | 0.393 | 87.79 ± 0.60 | 0.423 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 84.81 ± 0.63 | 0.080 | 84.21 ± 0.69 | 0.040 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 85.03 ± 0.64 | 0.518 | 84.34 ± 0.71 | 0.594 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Evid. | 12,290 | 79.16 ± 0.74 | 0.371 | 78.11 ± 0.82 | 0.404 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Softmax | 12,288 | 80.29 ± 0.74 | 0.348 | 79.36 ± 0.82 | 0.358 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 75.61 ± 0.75 | 0.342 | 74.53 ± 0.82 | 0.354 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 74.92 ± 0.78 | 0.121 | 73.71 ± 0.86 | 0.130 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Evid. | 10,754 | 72.48 ± 0.81 | 0.156 | 70.97 ± 0.88 | 0.162 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Softmax | 10,752 | 72.45 ± 0.83 | 0.431 | 70.98 ± 0.91 | 0.385 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 95.88 ± 0.23 | 0.116 | 95.85 ± 0.24 | 0.118 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 95.56 ± 0.24 | 0.046 | 95.53 ± 0.24 | 0.046 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Evid. | 12,290 | 88.32 ± 0.41 | 0.353 | 88.20 ± 0.42 | 0.362 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Softmax | 12,288 | 91.56 ± 0.34 | 0.491 | 91.48 ± 0.35 | 0.492 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 90.64 ± 0.38 | 0.140 | 90.55 ± 0.38 | 0.135 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 90.10 ± 0.40 | 0.342 | 89.97 ± 0.41 | 0.343 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Evid. | 10,754 | 87.96 ± 0.46 | 0.272 | 87.79 ± 0.47 | 0.279 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Softmax | 10,752 | 87.96 ± 0.45 | 0.139 | 87.82 ± 0.46 | 0.146 |

**Reading notes for Table 1:**

- Two confidence measures are reported deliberately. The `± 95 % CI` is the standard few-shot reporting convention — the confidence interval over the 600 test episodes within a run, averaged across seeds; this is what compares to published few-shot numbers. The `seed std` column is the spread of the *mean* across the three training seeds, and is the number to use when asking whether two cells in this table differ.
- **Seed std is small almost everywhere** — it exceeds 1 pp in only 2 of the 40 configurations (CIFAR-FS × ResNet-18 × LoRA × evidential, at 1-shot 1.09 pp and 5-shot 1.07 pp; every other cell is ≤ 0.74 pp). Differences of >2 pp between cells in this table are therefore real and not seed noise; the LoRA-evidential cells specifically deserve more seeds before any sub-2 pp claim is made about them.
- Macro-F1 sits consistently ~0.5–1.5 pp **below** accuracy, and the gap widens at 1-shot. This is expected: with a single support image per class, a prototype that lands badly for one class produces recall collapse on that class only, which macro-F1 punishes and accuracy partly hides.
- `Linear-Probe*` evidential and softmax are identical in accuracy and F1 by construction — with no trainable adapter the logits are fixed, and the evidence affine is a monotone transform of them, so the argmax cannot change. Their calibration differs (Table 2); only their calibration can.
- **The two baselines have zero seed variance, and this is expected rather than a bug.** Both `Full-FT*` and `Linear-Probe*` return bit-identical metrics for seeds 42/43/44 (see Table 7 — even the early-stopping epoch matches). The grid configs vary only `seed:`, and `cfg.seed` reaches training solely through `set_seed()`, i.e. through random parameter initialisation; the training episode stream is derived from `cfg.trainer.train_seed_offset`, a fixed config value, independently of `cfg.seed` ([scripts/train.py:270](../scripts/train.py#L270), [scripts/train.py:287](../scripts/train.py#L287)). Full fine-tuning starts from pretrained ImageNet weights and the linear probe has no parameters at all, so neither has any randomly-initialised tensor for the seed to perturb — whereas the Bottleneck and LoRA cells do, which is exactly why only those show seed spread. **Practical consequence: for the two baseline rows, n = 3 seeds is effectively n = 1**, and any claim resting on a small margin over them should be read accordingly (see §4.7 claim 2 and §4.8 item 4).

### Table 2 — Calibration

| Dataset | Shots | Backbone | Adapter | Head | ECE (pooled) | ECE (per-episode) | ECE after temp. scaling | Brier | Brier after TS |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 0.2765 | 0.2885 | -- | 0.4050 | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 0.0560 | 0.1364 | 0.0297 | 0.3084 | 0.3033 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Evid. | 0.2970 | 0.3096 | -- | 0.4959 | -- |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Softmax | 0.0969 | 0.1592 | 0.0248 | 0.3579 | 0.3434 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Evid. | 0.3207 | 0.3290 | -- | 0.4254 | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Softmax | 0.0854 | 0.1468 | 0.0223 | 0.2850 | 0.2732 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Evid. | 0.4397 | 0.4397 | -- | 0.7038 | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Softmax | 0.2818 | 0.2944 | 0.0642 | 0.5283 | 0.4164 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 0.2540 | 0.2702 | -- | 0.3973 | -- |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 0.0499 | 0.1347 | 0.0322 | 0.3070 | 0.3042 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Evid. | 0.2156 | 0.2423 | -- | 0.4361 | -- |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Softmax | 0.0287 | 0.1357 | 0.0403 | 0.3461 | 0.3474 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 0.3010 | 0.3070 | -- | 0.2513 | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 0.0670 | 0.0989 | 0.0152 | 0.1332 | 0.1233 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Evid. | 0.3294 | 0.3366 | -- | 0.3888 | -- |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Softmax | 0.1016 | 0.1353 | 0.0158 | 0.2156 | 0.1962 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Evid. | 0.3383 | 0.3423 | -- | 0.3220 | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Softmax | 0.0728 | 0.1048 | 0.0135 | 0.1536 | 0.1419 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Evid. | 0.6234 | 0.6234 | -- | 0.7078 | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Softmax | 0.4476 | 0.4516 | 0.0297 | 0.4554 | 0.1824 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 0.3043 | 0.3117 | -- | 0.2708 | -- |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 0.0703 | 0.1039 | 0.0074 | 0.1464 | 0.1355 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Evid. | 0.2879 | 0.2993 | -- | 0.2991 | -- |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Softmax | 0.0616 | 0.1065 | 0.0060 | 0.1793 | 0.1710 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 0.3073 | 0.3158 | -- | 0.3523 | -- |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 0.1012 | 0.1416 | 0.0073 | 0.2352 | 0.2166 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Evid. | 0.3708 | 0.3785 | -- | 0.4909 | -- |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Softmax | 0.1748 | 0.2035 | 0.0077 | 0.3282 | 0.2816 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 0.2534 | 0.2664 | -- | 0.4402 | -- |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 0.0650 | 0.1433 | 0.0063 | 0.3551 | 0.3478 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Evid. | 0.2194 | 0.2417 | -- | 0.4557 | -- |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Softmax | 0.0242 | 0.1368 | 0.0118 | 0.3796 | 0.3787 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 0.2938 | 0.2993 | -- | 0.1883 | -- |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 0.0850 | 0.1014 | 0.0057 | 0.0856 | 0.0671 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Evid. | 0.4049 | 0.4093 | -- | 0.3977 | -- |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Softmax | 0.1930 | 0.2063 | 0.0187 | 0.1913 | 0.1258 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 0.3177 | 0.3229 | -- | 0.2817 | -- |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 0.1107 | 0.1339 | 0.0211 | 0.1699 | 0.1458 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Evid. | 0.3053 | 0.3119 | -- | 0.3095 | -- |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Softmax | 0.0955 | 0.1276 | 0.0221 | 0.1938 | 0.1765 |

### Table 3 — OOD detection AUROC (primary score per head: vacuity for evidential, MSP for softmax)

| Dataset | Shots | Backbone | Adapter | Head | Score | SVHN (far) | Gauss (far) | C100 (near) | MiniIN-held (near) | TIN (near) | Mean AUROC |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.8762 | 0.9482 | 0.7936 | -- | 0.8311 | 0.8623 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.7837 | 0.8265 | 0.6964 | -- | 0.7378 | 0.7611 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.8328 | 0.9398 | 0.7250 | -- | 0.7713 | 0.8172 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Softmax | msp | 0.7470 | 0.6858 | 0.6673 | -- | 0.7098 | 0.7025 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Evid. | vacuity | 0.8805 | 0.9899 | 0.7909 | -- | 0.8310 | 0.8731 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Softmax | msp | 0.8348 | 0.9060 | 0.7349 | -- | 0.7925 | 0.8171 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Evid. | vacuity | 0.6819 | 1.0000 | 0.5800 | -- | 0.8677 | 0.7824 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Softmax | msp | 0.6721 | 0.7330 | 0.6625 | -- | 0.7278 | 0.6988 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.8317 | 0.9546 | 0.8005 | -- | 0.8698 | 0.8642 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.7504 | 0.8608 | 0.7233 | -- | 0.7871 | 0.7804 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.8175 | 0.7680 | 0.7433 | -- | 0.7468 | 0.7689 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.7561 | 0.8025 | 0.6704 | -- | 0.6976 | 0.7317 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.9165 | 0.9814 | 0.9031 | -- | 0.9256 | 0.9317 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.8851 | 0.9363 | 0.8179 | -- | 0.8518 | 0.8728 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.8900 | 0.9810 | 0.8143 | -- | 0.8489 | 0.8835 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Softmax | msp | 0.8197 | 0.8600 | 0.7421 | -- | 0.7821 | 0.8010 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Evid. | vacuity | 0.9090 | 0.9954 | 0.8683 | -- | 0.8974 | 0.9175 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Softmax | msp | 0.8938 | 0.9098 | 0.8260 | -- | 0.8670 | 0.8741 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Evid. | vacuity | 0.7079 | 1.0000 | 0.5842 | -- | 0.8856 | 0.7944 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Softmax | msp | 0.8551 | 0.8115 | 0.7966 | -- | 0.8429 | 0.8265 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.9444 | 0.9624 | 0.8785 | -- | 0.9186 | 0.9260 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.8450 | 0.8623 | 0.8322 | -- | 0.8859 | 0.8564 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.9295 | 0.9898 | 0.8575 | -- | 0.8798 | 0.9141 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.8288 | 0.9099 | 0.7867 | -- | 0.8099 | 0.8338 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.9142 | 0.8902 | -- | 0.8521 | 0.8704 | 0.8817 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.8160 | 0.7581 | -- | 0.7894 | 0.8104 | 0.7935 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.9602 | 0.9623 | -- | 0.7795 | 0.7736 | 0.8689 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Softmax | msp | 0.8086 | 0.6974 | -- | 0.7403 | 0.7745 | 0.7552 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.8806 | 0.8946 | -- | 0.7796 | 0.8563 | 0.8528 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.7328 | 0.7551 | -- | 0.7138 | 0.7428 | 0.7361 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.8317 | 0.8646 | -- | 0.7573 | 0.7896 | 0.8108 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.6385 | 0.5999 | -- | 0.6683 | 0.6894 | 0.6490 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.9731 | 0.9576 | -- | 0.9414 | 0.9582 | 0.9576 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.9106 | 0.8614 | -- | 0.8973 | 0.9124 | 0.8954 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.9841 | 0.9831 | -- | 0.8501 | 0.8692 | 0.9216 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Softmax | msp | 0.9178 | 0.7171 | -- | 0.8223 | 0.8488 | 0.8265 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.9104 | 0.9283 | -- | 0.8761 | 0.9110 | 0.9065 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.7949 | 0.7747 | -- | 0.8199 | 0.8442 | 0.8084 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.9118 | 0.9128 | -- | 0.8484 | 0.8895 | 0.8906 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.7509 | 0.7466 | -- | 0.7663 | 0.7916 | 0.7639 |

### Table 4 — OOD FPR@95%TPR (lower is better; primary score per head)

| Dataset | Shots | Backbone | Adapter | Head | Score | SVHN (far) | Gauss (far) | C100 (near) | MiniIN-held (near) | TIN (near) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.5071 | 0.2916 | 0.6563 | -- | 0.6197 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.7423 | 0.6863 | 0.8389 | -- | 0.7992 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.6228 | 0.3217 | 0.7310 | -- | 0.6774 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Softmax | msp | 0.7698 | 0.8639 | 0.8536 | -- | 0.8174 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Evid. | vacuity | 0.4973 | 0.0607 | 0.6436 | -- | 0.6069 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Softmax | msp | 0.6469 | 0.4892 | 0.7752 | -- | 0.7143 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Evid. | vacuity | 0.8764 | 0.0000 | 0.8983 | -- | 0.5797 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Softmax | msp | 0.8865 | 0.7750 | 0.8889 | -- | 0.8270 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.5875 | 0.2342 | 0.6567 | -- | 0.5334 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.7607 | 0.5809 | 0.8064 | -- | 0.7317 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.6486 | 0.6245 | 0.7424 | -- | 0.7440 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.7725 | 0.7238 | 0.8494 | -- | 0.8250 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.4238 | 0.1201 | 0.4108 | -- | 0.3248 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.5326 | 0.3787 | 0.6599 | -- | 0.5986 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.4725 | 0.1172 | 0.5988 | -- | 0.5258 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Softmax | msp | 0.6377 | 0.6222 | 0.7483 | -- | 0.7063 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Evid. | vacuity | 0.3920 | 0.0164 | 0.4879 | -- | 0.4366 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Softmax | msp | 0.4342 | 0.4777 | 0.6045 | -- | 0.5335 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Evid. | vacuity | 0.8656 | 0.0000 | 0.8931 | -- | 0.5402 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Softmax | msp | 0.6554 | 0.7135 | 0.7340 | -- | 0.6519 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.2574 | 0.1972 | 0.4795 | -- | 0.3839 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.6099 | 0.5741 | 0.6243 | -- | 0.5276 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.3582 | 0.0817 | 0.5305 | -- | 0.4475 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.6400 | 0.4025 | 0.7250 | -- | 0.6827 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.4025 | 0.4074 | -- | 0.5577 | 0.5375 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.7015 | 0.7496 | -- | 0.7145 | 0.6993 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.2118 | 0.1530 | -- | 0.6837 | 0.7271 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Softmax | msp | 0.6901 | 0.7733 | -- | 0.7734 | 0.7536 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.3884 | 0.3628 | -- | 0.7131 | 0.5632 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.7696 | 0.6831 | -- | 0.7939 | 0.7703 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.5185 | 0.3901 | -- | 0.7252 | 0.7039 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.8499 | 0.8372 | -- | 0.8366 | 0.8334 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Evid. | vacuity | 0.1624 | 0.2708 | -- | 0.2729 | 0.2336 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Softmax | msp | 0.4828 | 0.5889 | -- | 0.4819 | 0.4582 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Evid. | vacuity | 0.0875 | 0.0986 | -- | 0.5501 | 0.5393 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Softmax | msp | 0.4249 | 0.6180 | -- | 0.6304 | 0.6199 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | vacuity | 0.4105 | 0.3490 | -- | 0.5291 | 0.4402 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | msp | 0.6489 | 0.6377 | -- | 0.6457 | 0.6146 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Evid. | vacuity | 0.3847 | 0.3513 | -- | 0.5916 | 0.5040 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Softmax | msp | 0.6278 | 0.5445 | -- | 0.7154 | 0.6893 |

### Table 5 — Evidential vacuity vs. every softmax-side OOD score, head-to-head

Each row pairs the evidential cell with the softmax cell that is identical in every other respect. `Δ` columns are evidential-vacuity AUROC minus that softmax score's AUROC; positive means evidential wins.

| Dataset | Shots | Backbone | Adapter | OOD set | Evid. vacuity | Softmax MSP | Softmax TS-MSP | Softmax energy | Δ vs MSP | Δ vs TS-MSP | Δ vs energy |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | SVHN (far) | 0.8762 | 0.7837 | 0.7792 | 0.8283 | 0.0926 | 0.0970 | 0.0479 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Gauss (far) | 0.9482 | 0.8265 | 0.8202 | 0.9547 | 0.1217 | 0.1279 | -0.0065 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | C100 (near) | 0.7936 | 0.6964 | 0.6934 | 0.8044 | 0.0972 | 0.1002 | -0.0108 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | TIN (near) | 0.8311 | 0.7378 | 0.7341 | 0.8215 | 0.0933 | 0.0970 | 0.0097 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | SVHN (far) | 0.8328 | 0.7470 | 0.7343 | 0.8582 | 0.0858 | 0.0985 | -0.0254 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Gauss (far) | 0.9398 | 0.6858 | 0.6762 | 0.8786 | 0.2540 | 0.2636 | 0.0612 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | C100 (near) | 0.7250 | 0.6673 | 0.6596 | 0.7377 | 0.0577 | 0.0654 | -0.0127 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | TIN (near) | 0.7713 | 0.7098 | 0.7000 | 0.7848 | 0.0614 | 0.0713 | -0.0135 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | SVHN (far) | 0.8805 | 0.8348 | 0.8235 | 0.9011 | 0.0456 | 0.0570 | -0.0206 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Gauss (far) | 0.9899 | 0.9060 | 0.8939 | 0.9952 | 0.0839 | 0.0961 | -0.0053 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | C100 (near) | 0.7909 | 0.7349 | 0.7266 | 0.7973 | 0.0561 | 0.0643 | -0.0063 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | TIN (near) | 0.8310 | 0.7925 | 0.7824 | 0.8374 | 0.0385 | 0.0486 | -0.0064 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | SVHN (far) | 0.6819 | 0.6721 | 0.6649 | 0.7268 | 0.0098 | 0.0169 | -0.0449 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Gauss (far) | 1.0000 | 0.7330 | 0.7055 | 1.0000 | 0.2670 | 0.2945 | 0.0000 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | C100 (near) | 0.5800 | 0.6625 | 0.6547 | 0.6307 | -0.0825 | -0.0747 | -0.0507 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | TIN (near) | 0.8677 | 0.7278 | 0.7100 | 0.8925 | 0.1400 | 0.1577 | -0.0248 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | SVHN (far) | 0.8317 | 0.7504 | 0.7477 | 0.8983 | 0.0813 | 0.0840 | -0.0666 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Gauss (far) | 0.9546 | 0.8608 | 0.8573 | 0.9975 | 0.0938 | 0.0973 | -0.0429 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | C100 (near) | 0.8005 | 0.7233 | 0.7211 | 0.7774 | 0.0771 | 0.0794 | 0.0231 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | TIN (near) | 0.8698 | 0.7871 | 0.7842 | 0.8629 | 0.0828 | 0.0857 | 0.0069 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | SVHN (far) | 0.8175 | 0.7561 | 0.7584 | 0.8465 | 0.0614 | 0.0591 | -0.0289 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Gauss (far) | 0.7680 | 0.8025 | 0.8059 | 0.9983 | -0.0345 | -0.0379 | -0.2303 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | C100 (near) | 0.7433 | 0.6704 | 0.6721 | 0.7497 | 0.0729 | 0.0712 | -0.0064 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | TIN (near) | 0.7468 | 0.6976 | 0.6996 | 0.7785 | 0.0493 | 0.0473 | -0.0317 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | SVHN (far) | 0.9165 | 0.8851 | 0.8720 | 0.9344 | 0.0314 | 0.0445 | -0.0179 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Gauss (far) | 0.9814 | 0.9363 | 0.9231 | 0.9894 | 0.0451 | 0.0583 | -0.0080 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | C100 (near) | 0.9031 | 0.8179 | 0.8073 | 0.9006 | 0.0852 | 0.0958 | 0.0026 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | TIN (near) | 0.9256 | 0.8518 | 0.8400 | 0.9288 | 0.0738 | 0.0856 | -0.0032 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | SVHN (far) | 0.8900 | 0.8197 | 0.7947 | 0.9323 | 0.0703 | 0.0953 | -0.0423 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Gauss (far) | 0.9810 | 0.8600 | 0.8318 | 0.9929 | 0.1210 | 0.1492 | -0.0119 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | C100 (near) | 0.8143 | 0.7421 | 0.7233 | 0.8374 | 0.0721 | 0.0910 | -0.0231 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | TIN (near) | 0.8489 | 0.7821 | 0.7606 | 0.8758 | 0.0669 | 0.0883 | -0.0268 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | SVHN (far) | 0.9090 | 0.8938 | 0.8796 | 0.9536 | 0.0152 | 0.0294 | -0.0446 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Gauss (far) | 0.9954 | 0.9098 | 0.8922 | 0.9901 | 0.0856 | 0.1033 | 0.0053 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | C100 (near) | 0.8683 | 0.8260 | 0.8127 | 0.8834 | 0.0424 | 0.0556 | -0.0151 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | TIN (near) | 0.8974 | 0.8670 | 0.8523 | 0.9137 | 0.0304 | 0.0451 | -0.0163 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | SVHN (far) | 0.7079 | 0.8551 | 0.8233 | 0.7955 | -0.1472 | -0.1154 | -0.0876 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Gauss (far) | 1.0000 | 0.8115 | 0.7607 | 1.0000 | 0.1885 | 0.2393 | 0.0000 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | C100 (near) | 0.5842 | 0.7966 | 0.7705 | 0.6629 | -0.2125 | -0.1863 | -0.0788 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | TIN (near) | 0.8856 | 0.8429 | 0.8078 | 0.9264 | 0.0428 | 0.0778 | -0.0408 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | SVHN (far) | 0.9444 | 0.8450 | 0.8310 | 0.9518 | 0.0995 | 0.1134 | -0.0074 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Gauss (far) | 0.9624 | 0.8623 | 0.8468 | 0.9770 | 0.1001 | 0.1156 | -0.0146 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | C100 (near) | 0.8785 | 0.8322 | 0.8193 | 0.8861 | 0.0463 | 0.0593 | -0.0075 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | TIN (near) | 0.9186 | 0.8859 | 0.8706 | 0.9508 | 0.0326 | 0.0480 | -0.0323 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | SVHN (far) | 0.9295 | 0.8288 | 0.8141 | 0.9405 | 0.1008 | 0.1154 | -0.0110 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Gauss (far) | 0.9898 | 0.9099 | 0.8937 | 0.9995 | 0.0799 | 0.0960 | -0.0097 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | C100 (near) | 0.8575 | 0.7867 | 0.7756 | 0.8285 | 0.0708 | 0.0819 | 0.0290 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | MiniIN-held (near) | -- | -- | -- | -- | -- | -- | -- |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | TIN (near) | 0.8798 | 0.8099 | 0.7980 | 0.8373 | 0.0699 | 0.0818 | 0.0425 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | SVHN (far) | 0.9142 | 0.8160 | 0.7978 | 0.9268 | 0.0983 | 0.1164 | -0.0126 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Gauss (far) | 0.8902 | 0.7581 | 0.7395 | 0.9140 | 0.1320 | 0.1507 | -0.0238 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | MiniIN-held (near) | 0.8521 | 0.7894 | 0.7736 | 0.8514 | 0.0627 | 0.0784 | 0.0007 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | TIN (near) | 0.8704 | 0.8104 | 0.7932 | 0.8564 | 0.0599 | 0.0771 | 0.0140 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | SVHN (far) | 0.9602 | 0.8086 | 0.7805 | 0.9425 | 0.1517 | 0.1797 | 0.0177 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Gauss (far) | 0.9623 | 0.6974 | 0.6760 | 0.9500 | 0.2649 | 0.2864 | 0.0123 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | MiniIN-held (near) | 0.7795 | 0.7403 | 0.7200 | 0.8027 | 0.0392 | 0.0595 | -0.0231 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | TIN (near) | 0.7736 | 0.7745 | 0.7508 | 0.7888 | -0.0009 | 0.0228 | -0.0151 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | SVHN (far) | 0.8806 | 0.7328 | 0.7200 | 0.9671 | 0.1478 | 0.1606 | -0.0866 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Gauss (far) | 0.8946 | 0.7551 | 0.7406 | 0.9545 | 0.1395 | 0.1540 | -0.0599 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | MiniIN-held (near) | 0.7796 | 0.7138 | 0.7055 | 0.7722 | 0.0658 | 0.0741 | 0.0074 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | TIN (near) | 0.8563 | 0.7428 | 0.7322 | 0.8608 | 0.1136 | 0.1241 | -0.0045 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | SVHN (far) | 0.8317 | 0.6385 | 0.6322 | 0.8194 | 0.1932 | 0.1994 | 0.0122 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Gauss (far) | 0.8646 | 0.5999 | 0.5940 | 0.8347 | 0.2647 | 0.2706 | 0.0300 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | MiniIN-held (near) | 0.7573 | 0.6683 | 0.6629 | 0.7656 | 0.0890 | 0.0944 | -0.0083 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | TIN (near) | 0.7896 | 0.6894 | 0.6831 | 0.7755 | 0.1002 | 0.1065 | 0.0140 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | SVHN (far) | 0.9731 | 0.9106 | 0.8942 | 0.9529 | 0.0625 | 0.0789 | 0.0202 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Gauss (far) | 0.9576 | 0.8614 | 0.8467 | 0.9372 | 0.0962 | 0.1109 | 0.0204 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | MiniIN-held (near) | 0.9414 | 0.8973 | 0.8830 | 0.9373 | 0.0441 | 0.0584 | 0.0041 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | TIN (near) | 0.9582 | 0.9124 | 0.8974 | 0.9623 | 0.0457 | 0.0608 | -0.0041 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | SVHN (far) | 0.9841 | 0.9178 | 0.8798 | 0.9876 | 0.0663 | 0.1043 | -0.0036 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Gauss (far) | 0.9831 | 0.7171 | 0.6807 | 0.9736 | 0.2660 | 0.3024 | 0.0095 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | MiniIN-held (near) | 0.8501 | 0.8223 | 0.7921 | 0.8759 | 0.0278 | 0.0580 | -0.0257 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | TIN (near) | 0.8692 | 0.8488 | 0.8154 | 0.8842 | 0.0204 | 0.0538 | -0.0150 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | SVHN (far) | 0.9104 | 0.7949 | 0.7720 | 0.9481 | 0.1156 | 0.1384 | -0.0377 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Gauss (far) | 0.9283 | 0.7747 | 0.7526 | 0.9641 | 0.1536 | 0.1757 | -0.0358 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | MiniIN-held (near) | 0.8761 | 0.8199 | 0.8027 | 0.8605 | 0.0563 | 0.0734 | 0.0156 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | TIN (near) | 0.9110 | 0.8442 | 0.8241 | 0.9154 | 0.0668 | 0.0869 | -0.0043 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | SVHN (far) | 0.9118 | 0.7509 | 0.7312 | 0.9702 | 0.1609 | 0.1806 | -0.0584 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Gauss (far) | 0.9128 | 0.7466 | 0.7305 | 0.9848 | 0.1662 | 0.1823 | -0.0720 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | C100 (near) | -- | -- | -- | -- | -- | -- | -- |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | MiniIN-held (near) | 0.8484 | 0.7663 | 0.7513 | 0.8076 | 0.0821 | 0.0971 | 0.0409 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | TIN (near) | 0.8895 | 0.7916 | 0.7739 | 0.8625 | 0.0979 | 0.1156 | 0.0270 |

**Win counts (evidential vacuity vs. each softmax score, over all matched dataset x shot x backbone x adapter x OOD-set comparisons), split by OOD difficulty:**

| Comparison | Far-OOD wins | Far-OOD mean Δ AUROC | Near-OOD wins | Near-OOD mean Δ AUROC | Overall win rate |
|---|---:|---:|---:|---:|---:|
| vacuity vs msp | 38/40 | +0.1108 | 37/40 | +0.0534 | 93.8% |
| vacuity vs ts_msp | 38/40 | +0.1272 | 38/40 | +0.0670 | 95.0% |
| vacuity vs energy | 10/40 | -0.0220 | 14/40 | -0.0067 | 30.0% |

### Table 6 — Parameter efficiency (5-shot, both datasets)

| Dataset | Backbone | Adapter | Head | Params | Accuracy % | Acc. per 1k params | vs Full-FT params | vs Full-FT accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|
| CIFAR-FS | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 91.58 | 2.9 | 0.284% | +1.12 pp |
| CIFAR-FS | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 91.44 | 2.9 | 0.284% | +0.98 pp |
| CIFAR-FS | ResNet-18 | LoRA | Evid. | 12,290 | 83.33 | 6.8 | 0.110% | -7.14 pp |
| CIFAR-FS | ResNet-18 | LoRA | Softmax | 12,288 | 86.25 | 7.0 | 0.110% | -4.21 pp |
| CIFAR-FS | ResNet-18 | Full-FT* | Evid. | 11,176,514 | 88.82 | 0.0 | 100.000% | -1.65 pp |
| CIFAR-FS | ResNet-18 | Full-FT* | Softmax | 11,176,512 | 90.47 | 0.0 | 100.000% | +0.00 pp |
| CIFAR-FS | ResNet-18 | Linear-Probe* | Evid. | 2 | 87.41 | 43706.7 | 0.000% | -3.05 pp |
| CIFAR-FS | ResNet-18 | Linear-Probe* | Softmax | 0 | 87.41 | n/a | 0.000% | -3.05 pp |
| CIFAR-FS | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 90.24 | 13.0 | 0.062% | -0.23 pp |
| CIFAR-FS | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 90.74 | 13.1 | 0.062% | +0.28 pp |
| CIFAR-FS | MobileNetV3-S | LoRA | Evid. | 10,754 | 86.97 | 8.1 | 0.096% | -3.49 pp |
| CIFAR-FS | MobileNetV3-S | LoRA | Softmax | 10,752 | 88.05 | 8.2 | 0.096% | -2.42 pp |
| MiniImageNet | ResNet-18 | Bottleneck-par | Evid. | 31,746 | 95.88 | 3.0 | -- | -- |
| MiniImageNet | ResNet-18 | Bottleneck-par | Softmax | 31,744 | 95.56 | 3.0 | -- | -- |
| MiniImageNet | ResNet-18 | LoRA | Evid. | 12,290 | 88.32 | 7.2 | -- | -- |
| MiniImageNet | ResNet-18 | LoRA | Softmax | 12,288 | 91.56 | 7.5 | -- | -- |
| MiniImageNet | MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 90.64 | 13.1 | -- | -- |
| MiniImageNet | MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 90.10 | 13.0 | -- | -- |
| MiniImageNet | MobileNetV3-S | LoRA | Evid. | 10,754 | 87.96 | 8.2 | -- | -- |
| MiniImageNet | MobileNetV3-S | LoRA | Softmax | 10,752 | 87.96 | 8.2 | -- | -- |

### Table 7 — Appendix: early-stopping epoch selected on VAL (mean over 3 seeds)

| Dataset | Shots | Backbone | Adapter | Head | Best VAL epoch (mean) | Per-seed |
|---|---|---|---|---|---:|---|
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 4.3 | 42:4, 43:4, 44:5 |
| CIFAR-FS | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 9.7 | 42:19, 43:4, 44:6 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Evid. | 6.7 | 42:1, 43:7, 44:12 |
| CIFAR-FS | 1-shot | ResNet-18 | LoRA | Softmax | 4.0 | 42:4, 43:3, 44:5 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Evid. | 4.0 | 42:4, 43:4, 44:4 |
| CIFAR-FS | 1-shot | ResNet-18 | Full-FT* | Softmax | 4.0 | 42:4, 43:4, 44:4 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Evid. | 1.0 | 42:1, 43:1, 44:1 |
| CIFAR-FS | 1-shot | ResNet-18 | Linear-Probe* | Softmax | 0.0 | 42:0, 43:0, 44:0 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 7.3 | 42:5, 43:12, 44:5 |
| CIFAR-FS | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 14.0 | 42:19, 43:12, 44:11 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Evid. | 5.3 | 42:5, 43:5, 44:6 |
| CIFAR-FS | 1-shot | MobileNetV3-S | LoRA | Softmax | 6.0 | 42:8, 43:5, 44:5 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 11.0 | 42:19, 43:2, 44:12 |
| CIFAR-FS | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 7.3 | 42:10, 43:2, 44:10 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Evid. | 4.7 | 42:4, 43:3, 44:7 |
| CIFAR-FS | 5-shot | ResNet-18 | LoRA | Softmax | 4.7 | 42:6, 43:4, 44:4 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Evid. | 6.0 | 42:6, 43:6, 44:6 |
| CIFAR-FS | 5-shot | ResNet-18 | Full-FT* | Softmax | 6.0 | 42:6, 43:6, 44:6 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Evid. | 1.0 | 42:1, 43:1, 44:1 |
| CIFAR-FS | 5-shot | ResNet-18 | Linear-Probe* | Softmax | 0.0 | 42:0, 43:0, 44:0 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 3.0 | 42:3, 43:3, 44:3 |
| CIFAR-FS | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 11.3 | 42:10, 43:10, 44:14 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Evid. | 2.7 | 42:3, 43:2, 44:3 |
| CIFAR-FS | 5-shot | MobileNetV3-S | LoRA | Softmax | 8.0 | 42:11, 43:11, 44:2 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Evid. | 4.3 | 42:3, 43:7, 44:3 |
| MiniImageNet | 1-shot | ResNet-18 | Bottleneck-par | Softmax | 4.3 | 42:7, 43:3, 44:3 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Evid. | 2.0 | 42:2, 43:2, 44:2 |
| MiniImageNet | 1-shot | ResNet-18 | LoRA | Softmax | 2.0 | 42:2, 43:2, 44:2 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6.0 | 42:6, 43:6, 44:6 |
| MiniImageNet | 1-shot | MobileNetV3-S | Bottleneck-par | Softmax | 6.0 | 42:6, 43:6, 44:6 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Evid. | 7.7 | 42:11, 43:7, 44:5 |
| MiniImageNet | 1-shot | MobileNetV3-S | LoRA | Softmax | 9.0 | 42:11, 43:9, 44:7 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Evid. | 10.3 | 42:16, 43:11, 44:4 |
| MiniImageNet | 5-shot | ResNet-18 | Bottleneck-par | Softmax | 5.0 | 42:4, 43:7, 44:4 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Evid. | 3.3 | 42:3, 43:4, 44:3 |
| MiniImageNet | 5-shot | ResNet-18 | LoRA | Softmax | 2.3 | 42:3, 43:2, 44:2 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Evid. | 6.0 | 42:6, 43:6, 44:6 |
| MiniImageNet | 5-shot | MobileNetV3-S | Bottleneck-par | Softmax | 7.7 | 42:11, 43:2, 44:10 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Evid. | 6.3 | 42:6, 43:6, 44:7 |
| MiniImageNet | 5-shot | MobileNetV3-S | LoRA | Softmax | 7.0 | 42:7, 43:7, 44:7 |

### Table 8 — Efficiency and Pareto frontier (Step 11, RQ4)

Environments measured: kaggle_t4_cpu, kaggle_t4_cuda, local_cpu. Params/MACs are deterministic (byte-identical across sessions, per `efficiency_table.json`'s `reproducibility` block); latency/memory are measured and session-dependent by design -- see step_writeups/step11.txt Section 5. Cost figures below use the pre-registered primary profile (CPU, 1 thread, median latency) where available.

| Backbone | Adapter | Head | Trainable params | Total params | GMACs | CPU ms/img (1 thr) | CPU ms/img (all thr) | GPU ms/img | Peak GPU MB | On frontier? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ResNet-18 | Bottleneck-par | Evid. | 31,746 | 11,208,258 | 1.8306 | 62.38 | 37.72 | 3.36 | 134.4 | yes |
| ResNet-18 | Bottleneck-par | Softmax | 31,744 | 11,208,256 | 1.8306 | 62.18 | 38.54 | 3.35 | 134.4 | yes |
| ResNet-18 | LoRA | Evid. | 12,290 | 11,188,802 | 1.8192 | 60.04 | 35.48 | 2.90 | 134.4 | no |
| ResNet-18 | LoRA | Softmax | 12,288 | 11,188,800 | 1.8192 | 59.76 | 36.82 | 2.87 | 134.4 | yes |
| ResNet-18 | Full-FT* | Evid. | 11,176,514 | 11,176,514 | 1.8186 | 59.76 | 34.53 | 3.02 | 134.3 | no |
| ResNet-18 | Full-FT* | Softmax | 11,176,512 | 11,176,512 | 1.8186 | 61.52 | 36.67 | 2.82 | 134.3 | yes |
| ResNet-18 | Linear-Probe* | Evid. | 2 | 11,176,514 | 1.8186 | 61.10 | 36.08 | 3.04 | 134.3 | no |
| ResNet-18 | Linear-Probe* | Softmax | 0 | 11,176,512 | 1.8186 | 60.63 | 35.48 | 2.81 | 134.3 | yes |
| MobileNetV3-S | Bottleneck-par | Evid. | 6,930 | 933,938 | 0.0593 | 11.86 | 13.35 | 6.53 | 38.7 | yes |
| MobileNetV3-S | Bottleneck-par | Softmax | 6,928 | 933,936 | 0.0593 | 12.14 | 14.52 | 6.58 | 70.7 | yes |
| MobileNetV3-S | LoRA | Evid. | 10,754 | 937,762 | 0.0586 | 11.55 | 13.62 | 6.01 | 70.7 | yes |
| MobileNetV3-S | LoRA | Softmax | 10,752 | 937,760 | 0.0586 | 11.44 | 13.76 | 5.88 | 70.7 | yes |

---

## 3. What the grid answers, research question by research question

### RQ1 — Adapter placement and the parameter/accuracy tradeoff

**Finding: the parallel bottleneck adapter beats LoRA in 16 / 16 matched comparisons**, by +2.13 pp to +8.26 pp, across both datasets, both shot regimes, both backbones and both heads. There is not a single configuration where LoRA wins.

**This confirms prior work rather than discovering the principle from scratch, and the closest prior work is closer than it first appeared.** [Task-Specific Adapters (TSA)](https://arxiv.org/abs/2107.00358) (Li, Liu & Bilen, CVPR 2022) already runs this near-exact comparison: **frozen ResNet-18**, serial vs. parallel/residual adapter connection, evaluated on **600 sampled episodic tasks** per dataset on held-out domains (Meta-Dataset), with a **parameter-free nearest-centroid classifier** — and finds *"residual [parallel] connections perform better than the serial one in almost all cases."* That is the same backbone, the same episode count, the same placement question, the same parameter-free-head design, and the same winning answer as this thesis's Step 6 result. TSA is the primary precedent for the serial-vs-parallel finding, not Conv-Adapter.

[Conv-Adapter](https://arxiv.org/abs/2208.07463) (Chen et al.) remains relevant for a narrower reason: it independently finds that adapters preserving spatial locality (its depth-wise-separable design) beat 1×1/linear-style adapters, whose "loss of locality" is its own named failure mode. **Do not equate this with the LoRA result stated below** — our LoRA arm targets a 1×1 downsample conv, and a 1×1 convolution (low-rank or not) still preserves per-pixel spatial locality; it does not obviously fall into Conv-Adapter's "loss of locality" failure mode the way a flattened linear adapter would. The LoRA-losing-16/16 result should be read as a separate, corroborating data point that a transformer-native reparameterisation underperforms a convolutional one on CNN backbones — not as an instance of a mechanism Conv-Adapter already named.

[FiT](https://arxiv.org/pdf/2206.08671) (Bateni et al.) is a third close relative: frozen CNN backbones (BiT-M-R50x1, EfficientNetV2-M), FiLM adapters as small as **11,648 parameters**, with a ProtoNets head as one classifier option — genuinely close in parameter scale, but it does not compare adapter placement or adapter type, and its evaluation is on fixed downstream benchmarks (VTAB-1k, CIFAR-100), not disjoint-class episodic testing.

**What remains new after all three are accounted for:** the specific LoRA-vs-bottleneck adapter-*type* comparison (neither TSA nor Conv-Adapter nor FiT runs it), trainable budgets down to **2** parameters in one baseline cell (below any of the three), a second backbone (MobileNetV3-Small, tested by none of them), and the evidential-uncertainty layer with calibration/OOD reported alongside accuracy — none of which any of the three test or report. That is a narrower claim than "the regime has never been tested," and it is the accurate one.

| Dataset | Shots | Backbone | Head | Parallel-bottleneck | LoRA | Δ |
|---|---|---|---|---|---|---:|
| CIFAR-FS | 1-shot | ResNet-18 | Evid. | 79.19 (31,746 p) | 73.76 (12,290 p) | **+5.43** |
| CIFAR-FS | 1-shot | ResNet-18 | Softmax | 78.57 (31,744 p) | 75.44 (12,288 p) | **+3.12** |
| CIFAR-FS | 5-shot | ResNet-18 | Evid. | 91.58 (31,746 p) | 83.33 (12,290 p) | **+8.26** |
| CIFAR-FS | 5-shot | ResNet-18 | Softmax | 91.44 (31,744 p) | 86.25 (12,288 p) | **+5.19** |
| CIFAR-FS | 1-shot | MobileNetV3-S | Evid. | 78.88 (6,930 p) | 74.03 (10,754 p) | **+4.85** |
| CIFAR-FS | 1-shot | MobileNetV3-S | Softmax | 78.80 (6,928 p) | 75.43 (10,752 p) | **+3.37** |
| CIFAR-FS | 5-shot | MobileNetV3-S | Evid. | 90.24 (6,930 p) | 86.97 (10,754 p) | **+3.26** |
| CIFAR-FS | 5-shot | MobileNetV3-S | Softmax | 90.74 (6,928 p) | 88.05 (10,752 p) | **+2.70** |
| MiniImageNet | 1-shot | ResNet-18 | Evid. | 84.81 (31,746 p) | 79.16 (12,290 p) | **+5.65** |
| MiniImageNet | 1-shot | ResNet-18 | Softmax | 85.03 (31,744 p) | 80.29 (12,288 p) | **+4.75** |
| MiniImageNet | 5-shot | ResNet-18 | Evid. | 95.88 (31,746 p) | 88.32 (12,290 p) | **+7.56** |
| MiniImageNet | 5-shot | ResNet-18 | Softmax | 95.56 (31,744 p) | 91.56 (12,288 p) | **+4.00** |
| MiniImageNet | 1-shot | MobileNetV3-S | Evid. | 75.61 (6,930 p) | 72.48 (10,754 p) | **+3.14** |
| MiniImageNet | 1-shot | MobileNetV3-S | Softmax | 74.92 (6,928 p) | 72.45 (10,752 p) | **+2.47** |
| MiniImageNet | 5-shot | MobileNetV3-S | Evid. | 90.64 (6,930 p) | 87.96 (10,754 p) | **+2.68** |
| MiniImageNet | 5-shot | MobileNetV3-S | Softmax | 90.10 (6,928 p) | 87.96 (10,752 p) | **+2.13** |

Two consequences worth stating separately:

1. **On MobileNetV3-Small the parallel bottleneck is a strict Pareto win** — it is both *cheaper* (6,930 vs 10,754 trainable parameters) and *more accurate* (by 2.1–4.9 pp) than LoRA. There is no tradeoff to negotiate on that backbone; LoRA is simply dominated.
2. **On ResNet-18 there is a genuine tradeoff**: the parallel bottleneck costs 2.6× the parameters of LoRA (31,746 vs 12,290) and buys 3.1–8.3 pp. Whether that is worth it is a deployment decision, but at these absolute magnitudes (≈ 32 k vs ≈ 12 k parameters — both negligible next to an 11.18 M-parameter backbone) the accuracy is almost always the binding constraint, not the 20 k parameter difference.

**Against the baselines** (CIFAR-FS, ResNet-18):

| Regime | Parallel bottleneck | Full fine-tuning | Linear probe |
|---|---|---|---|
| 5-shot accuracy | **91.44 %** @ 31,744 params | 90.47 % @ 11,176,512 params | 87.41 % @ 0 params |
| 1-shot accuracy | 78.57 % @ 31,744 params | **81.14 %** @ 11,176,512 params | 70.25 % @ 0 params |

At 5-shot, **a 31.7 k-parameter adapter beats full fine-tuning of an 11.18 M-parameter network by +0.97 pp while training 0.28 % of the parameters.** MobileNetV3-Small's parallel bottleneck reaches 90.74 % on the same benchmark with **6,928 trainable parameters — 0.06 % of full fine-tuning's budget — and still edges it out by +0.27 pp.**

At 1-shot the ordering flips and full fine-tuning wins by 2.6 pp, which is the honest limit of the claim: with a single support image per class, the extra capacity of full fine-tuning does buy something the adapter cannot recover.

### RQ2 — Does an evidential Dirichlet head calibrate better than softmax?

**Finding: no — decisively, and without a single exception in 20 matched comparisons.**

**Scope note on the proposal's wording, stated here rather than left implicit:** `proposal.txt` §4 phrases RQ2 as testing calibration *"when fine-tuning with <500 trainable parameters."* Only the Linear-Probe cells (**2** trainable parameters) literally fall in that range — the PEFT adapters that carry most of the grid sit at 6,928–31,746. The sub-500 case is reported and fails the *same* way (Linear-Probe evidential has the worst ECE in the entire grid, 0.4397–0.6234), but the headline RQ2 verdict below covers the full 2–31,746 parameter range actually tested, which is a widening of the proposal's stated scope.

Evidential pooled ECE is worse than plain softmax ECE in **20 / 20** matched pairs, by a factor of **1.4× to 9.1×**, and worse than temperature-scaled softmax by a factor of **5.3× to 51.2×**. The worst case (MiniImageNet 5-shot, ResNet-18, parallel bottleneck) is ECE 0.2938 evidential vs 0.0057 TS-softmax — a 51× gap.

| | Evidential ECE range | Softmax ECE range | TS-softmax ECE range |
|---|---|---|---|
| Across all 20 matched configurations | 0.2156 – 0.6234 | 0.0242 – 0.4476 | 0.0057 – 0.0642 |

Accuracy does not compensate: across the same 20 matched pairs the evidential head is on average **0.61 pp worse** in accuracy, winning only 7 / 20.

This confirms and substantially strengthens the Step 4.5 single-configuration verdict (`step_writeups/step4_5.txt`), which found the same gap at roughly 7× on one configuration. The grid shows it is not a configuration accident: it holds across 2 datasets × 2 shot regimes × 2 backbones × 4 adapters. **The answer to RQ2 as posed is a clean, well-powered negative result.**

### RQ3 — Does the Bayesian prior improve near-OOD detection in low-data regimes?

**Finding: yes against every probabilistic softmax score, no against the training-free energy score — and the advantage does grow as data shrinks.**

| Comparison | Far-OOD | Near-OOD | Overall |
|---|---|---|---|
| vacuity vs. MSP | **38 / 40 wins**, mean Δ AUROC **+0.111** | **37 / 40 wins**, mean Δ **+0.053** | 93.8 % |
| vacuity vs. TS-MSP | **38 / 40 wins**, mean Δ **+0.127** | **38 / 40 wins**, mean Δ **+0.067** | 95.0 % |
| vacuity vs. energy | 10 / 40 wins, mean Δ −0.022 | 14 / 40 wins, mean Δ −0.007 | 30.0 % |

The **low-data trend is real and in the predicted direction**: the near-OOD advantage of vacuity over MSP is **+0.0637 AUROC at 1-shot vs. +0.0431 at 5-shot**, and over TS-MSP **+0.0725 at 1-shot vs. +0.0614 at 5-shot**. The Bayesian prior helps *more* exactly where there is less data, which is the RQ3 hypothesis.

**But the honest counterpoint, which is stronger at grid scale than Step 4.5 suggested:** the energy score — which requires no evidential training, no Bayesian prior, and no extra parameters — beats vacuity in about 70 % of comparisons. Step 4.5's single configuration found evidential beating energy on far-OOD and CIFAR-100-near and losing only on TinyImageNet-near; **across the full grid that result does not generalise**, and energy is the better default OOD score in most cells. This is a correction to the previously recorded state of the science and should be reported as such rather than smoothed over.

The defensible claim is therefore narrower and more precise than "the Bayesian prior improves OOD detection": *among scores derived from the model's own predictive distribution, the Dirichlet vacuity is a substantially better OOD ranker than max-softmax-probability with or without temperature scaling, and its advantage increases in the lowest-data regime — but it does not beat a well-chosen logit-space score.*

### RQ4 — Latency vs. uncertainty-quality Pareto frontier

**STATUS: CLOSED 2026-08-09.** The canonical Kaggle T4 session (GPU + CPU) ran successfully; full account, including two real bugs found and fixed during closing verification, in `step_writeups/step11.txt` §8. Table 8 (§2) and the Pareto figures below are generated mechanically from the corrected `results/efficiency_table.json` / `results/pareto_frontier.json`.

Pre-registered (written before any number existed, so it could not be tuned after the fact — `step_writeups/step11.txt` §0): cost axis = CPU 1-thread per-image median latency (the edge proxy this repo has no Jetson Nano to measure on, matching Howard 2019's own "single large core" methodology); quality axis = `ood_auroc__tin_near__<native>` (TinyImageNet near-OOD, native score per head) — the only near-OOD pool common to both datasets, and the one RQ3 found the low-data trend on. The recommended operating point is the cheapest configuration within 1.0 pp accuracy / 0.010 AUROC of the panel best.

**Answer:** the recommended Pareto point is **MobileNetV3-Small + parallel bottleneck + evidential head** on CIFAR-FS (11.86 ms/image, 6,930 params, AUROC 0.870 / 0.919 at 1-/5-shot). On MiniImageNet the same point sits on the strict frontier but falls outside tolerance of the panel's best, so the recommended point becomes **ResNet-18 + parallel bottleneck + evidential** (62.38 ms, AUROC 0.870 / 0.958) — consistent with the ResNet-18-vs-MobileNetV3-Small accuracy gap this document already reports widening specifically on MiniImageNet (§3, RQ1).

**The most novel finding: backbone choice drives latency, adapter choice does not.** ResNet-18 vs. MobileNetV3-Small at matched adapter is a **5.12× latency difference** (62.18 ms vs 12.14 ms). Parallel-bottleneck vs. LoRA on ResNet-18 — a 2.58× difference in trainable *parameters* — is only a **3.9% latency difference**; on MobileNetV3-Small the adapter with *more* parameters (LoRA) is actually 5.8% *faster*. Both adapters' parameter deltas are swamped by the frozen trunk's forward-pass cost. Practical reading: the adapter decision is an accuracy decision (RQ1, up to +8.3 pp); the backbone decision is the latency decision (RQ4, 5.1×).

**"Evidential uncertainty is free at inference" — measured, not assumed.** Mean |latency delta| between evidential and softmax heads at matched (backbone, adapter), Kaggle CPU 1-thread: **1.29%**, smaller than this session's own measurement-noise floor (5.91% stability-rerun spread). The full uncertainty-scoring stage for a 75-query episode costs 1.09% of a single image's backbone forward pass.

**The energy-score correction (RQ3) is visible here too, not just in the accuracy-only OOD tables.** Under the primary native-score reading, evidential heads anchor every strict frontier. Under the softmax-gets-its-best-score reading, evidential's frontier presence on CIFAR-FS 5-shot collapses to zero — replaced by its softmax sibling at the same (backbone, adapter) point. The claim "put an evidential head on the Pareto-optimal edge config" is native-score-conditional, not universal.

A genuine correction surfaced during verification: with a data-selection bug that (silently, no error) sourced latency from this repo's own dev laptop instead of the canonical Kaggle CPU, `mobilenetv3_small/lora` appeared strictly dominated by `mobilenetv3_small/bottleneck_parallel`. With the corrected Kaggle-only data, LoRA's real latency on MobileNetV3-Small is slightly *lower* than the parallel bottleneck's despite having *more* parameters — both LoRA variants are genuinely on the CIFAR-FS strict frontier. Every panel's *recommended point* was unaffected by this bug; the *frontier membership* was not. Full incident: `step_writeups/step11.txt` §8.7.

---

## 4. Positioning against the state of the art

### 4.1 The comparability problem — read this before any number below

Our backbones are **ImageNet-pretrained and frozen**. The standard CIFAR-FS / MiniImageNet protocol trains the backbone from scratch on the 64 base classes. These are different problems, and MiniImageNet makes the difference acute: **MiniImageNet classes are ImageNet classes**, so every "novel" test class was seen during pretraining. Our 95.88 % MiniImageNet 5-shot number is not a few-shot learning result in the sense the benchmark was designed to measure.

This is not a discovery of ours; it is a known and documented property of the external-pretraining regime. The P>M>F authors place their supervised-ImageNet-1k pretraining row in supplemental material precisely because, in their words, *"supervised pre-training on ImageNet is only useful to check the upper bound performance"* ([Hu et al., CVPR 2022, supplemental](https://ar5iv.labs.arxiv.org/html/2204.07305)) — that row reaches **99.8 %** on MiniImageNet 5-shot, which is the reductio.

**Therefore: our accuracy numbers must never be tabulated next to from-scratch few-shot SOTA as if they competed.** The comparison below is split into two regimes accordingly, and the thesis's contribution claim rests on the axes where the comparison *is* fair — calibration, OOD detection, and trainable-parameter count.

### 4.2 Regime A — from-scratch few-shot protocol (**our results do not belong here**)

Reported by [Lee et al., "Meta-Learning with Differentiable Convex Optimization" (MetaOptNet), CVPR 2019](https://arxiv.org/abs/1904.03758):

| Method | Backbone | MiniImageNet 1-shot | MiniImageNet 5-shot | CIFAR-FS 1-shot | CIFAR-FS 5-shot |
|---|---|---:|---:|---:|---:|
| Matching Networks | Conv-4-64 | 43.56 ± 0.84 | 55.31 ± 0.73 | — | — |
| Meta-Learner LSTM | Conv-4-64 | 43.44 ± 0.77 | 60.60 ± 0.71 | — | — |
| MAML | Conv-4-32 | 48.70 ± 1.84 | 63.11 ± 0.92 | 58.9 ± 1.9 | 71.5 ± 1.0 |
| Prototypical Networks | Conv-4-64 | 49.42 ± 0.78 | 68.20 ± 0.66 | 55.5 ± 0.7 | 72.0 ± 0.6 |
| Relation Networks | Conv-4 var. | 50.44 ± 0.82 | 65.32 ± 0.70 | 55.0 ± 1.0 | 69.3 ± 0.8 |
| R2D2 | Conv-4 (wide) | 51.2 ± 0.6 | 68.8 ± 0.1 | 65.3 ± 0.2 | 79.4 ± 0.1 |
| TADAM | ResNet-12 | 58.50 ± 0.30 | 76.70 ± 0.30 | — | — |
| LEO | WRN-28-10 | 61.76 ± 0.08 | 77.59 ± 0.12 | — | — |
| MetaOptNet-SVM | ResNet-12 | 62.64 ± 0.61 | 78.63 ± 0.46 | 72.0 ± 0.7 | 84.2 ± 0.5 |

> Secondary comparison tables in later papers circulate a stronger ResNet-12 ProtoNet re-implementation (≈ 72.2 / 83.5 on CIFAR-FS). That number was seen in a secondary source during this literature check and has **not** been verified against a primary paper; do not cite it without checking the original.

### 4.3 Regime B — external-pretraining protocol (**this is our regime**)

Reported by [Hu et al., "Pushing the Limits of Simple Pipelines for Few-Shot Learning" (P>M>F), CVPR 2022](https://ar5iv.labs.arxiv.org/html/2204.07305), Table 4 of the supplemental material — all figures are with the backbone meta-trained and fine-tuned:

| Pretraining > method | Backbone | MiniImageNet 1-shot | MiniImageNet 5-shot | CIFAR-FS 1-shot | CIFAR-FS 5-shot | Trainable params |
|---|---|---:|---:|---:|---:|---|
| DINO > ProtoNet | ResNet-50 | 79.2 | 92.0 | — | — | full backbone (~25 M) |
| CLIP > ProtoNet | ResNet-50 | 78.9 | 92.2 | — | — | full backbone (~25 M) |
| DINO > ProtoNet | ViT-small | 93.1 | 98.0 | 81.1 | 92.5 | full backbone (~21 M) |
| DINO > ProtoNet | ViT-base | 95.3 | 98.4 | 84.3 | 92.2 | full backbone (~86 M) |
| CLIP > ProtoNet | ViT-base | 93.1 | 98.1 | 85.3 | 93.2 | full backbone (~86 M) |
| Sup-21k > ProtoNet | ViT-base | 97.2 | 99.2 | 92.3 | 96.7 | full backbone (~86 M) |
| Sup-1k > ProtoNet | ViT-base | 99.2 | 99.8 | 88.2 | 94.3 | full backbone (~86 M) |
| **B-PEFT (ours)**, parallel bottleneck, softmax | **frozen ResNet-18** | **85.03** *(F1 84.34)* | **95.56** *(F1 95.53)* | **78.57** *(F1 77.50)* | **91.44** *(F1 91.29)* | **31,744 (0.28 % of the backbone)** |
| **B-PEFT (ours)**, parallel bottleneck, softmax | **frozen MobileNetV3-S** | **74.92** *(F1 73.71)* | **90.10** *(F1 89.97)* | **78.80** *(F1 77.69)* | **90.74** *(F1 90.59)* | **6,928** |

**What this table actually shows.** Within the regime our results belong to, our frozen ResNet-18 with a 31.7 k-parameter adapter lands **above** the DINO/CLIP ResNet-50 ProtoNet variants on MiniImageNet 5-shot (95.56 vs 92.0 / 92.2) and **below** every ViT-based variant — while training **roughly three orders of magnitude fewer parameters than any row above it**, and updating none of the backbone.

That is the honest headline: *not* a new accuracy record, but a competitive point on a part of the accuracy-vs-trainable-parameters plane that the published pipelines do not occupy, on backbones (ResNet-18, MobileNetV3-Small) that fit on edge hardware.

### 4.4 The transformer era — accuracy, macro-F1 and parameter budget, side by side

The pre-defence objection to this thesis was that CNN backbones are outdated. [DEFENCE_BRIEF.md](DEFENCE_BRIEF.md) answers that in full; this section carries the numbers, because it is a results question. **Parameter efficiency is a trade against accuracy, and this section shows both sides of it.**

#### 4.4.1 Same protocol, so accuracy *is* comparable: 5-way few-shot episodic

P>M>F is evaluated on 5-way few-shot episodes on CIFAR-FS and MiniImageNet — the same task as ours. It differs in backbone and in what it trains (P>M>F meta-trains and fine-tunes the whole backbone; we freeze it and train an adapter). So accuracy here is a fair comparison, subject to the pretraining caveat in §4.1 which applies to both sides.

P>M>F figures from [arXiv:2204.07305](https://ar5iv.labs.arxiv.org/html/2204.07305) Table 4 (supplemental); ViT-S/16 = 21 M and ViT-B/16 = 86 M per the DINO model cards; ours from `results/mvt_results.json`.

| Method | Backbone | Trainable params | CIFAR-FS 1-shot | CIFAR-FS 5-shot | MiniIN 1-shot | MiniIN 5-shot | Macro-F1 reported? |
|---|---|---:|---:|---:|---:|---:|:---:|
| Sup-21k > ProtoNet | ViT-B/16 (86 M) | ~85,800,000 | 92.3 | **96.7** | 97.2 | **99.2** | ❌ |
| CLIP > ProtoNet | ViT-B/16 (86 M) | ~86,500,000 | 85.3 | 93.2 | 93.1 | 98.1 | ❌ |
| DINO > ProtoNet | ViT-B/16 (86 M) | ~85,800,000 | 84.3 | 92.2 | 95.3 | 98.4 | ❌ |
| DINO > ProtoNet | ViT-S/16 (21 M) | ~21,000,000 | 81.1 | 92.5 | 93.1 | 98.0 | ❌ |
| DINO > ProtoNet | ResNet-50 (25 M) | ~25,000,000 | — | — | 79.2 | 92.0 | ❌ |
| BEL (evidential few-shot) | ResNet-12 | not reported | 73.96 | 86.92 | 63.10 | 79.60 | ❌ |
| MetaOptNet-SVM *(from-scratch protocol)* | ResNet-12 | not reported | 72.0 | 84.2 | 62.64 | 78.63 | ❌ |
| **B-PEFT (ours)** parallel bottleneck, softmax | **ResNet-18 (11.7 M, frozen)** | **31,744** | 78.57 | **91.44** | 85.03 | **95.56** | ✅ **77.50 / 91.29 / 84.34 / 95.53** |
| **B-PEFT (ours)** parallel bottleneck, evidential | **ResNet-18 (11.7 M, frozen)** | **31,746** | 79.19 | **91.58** | 84.81 | **95.88** | ✅ **78.29 / 91.46 / 84.21 / 95.85** |
| **B-PEFT (ours)** parallel bottleneck, softmax | **MobileNetV3-S (2.5 M, frozen)** | **6,928** | 78.80 | **90.74** | 74.92 | **90.10** | ✅ **77.69 / 90.59 / 73.71 / 89.97** |

*(Our macro-F1 column lists CIFAR-FS 1-shot / CIFAR-FS 5-shot / MiniIN 1-shot / MiniIN 5-shot, in the same order as the accuracy columns. No other method in this table reports macro-F1 at all — see §4.6.)*

#### 4.4.2 The trade, stated plainly

| Ours vs. | Trainable-param saving | CIFAR-FS 5-shot | MiniIN 5-shot | CIFAR-FS 1-shot | MiniIN 1-shot |
|---|---:|---:|---:|---:|---:|
| ResNet-18 adapter (31,744) vs DINO>PN ViT-S | **662× fewer** | **−1.06 pp** | −2.44 pp | −2.53 pp | −8.07 pp |
| ResNet-18 adapter vs DINO>PN ViT-B | **2,703× fewer** | **−0.76 pp** | −2.84 pp | −5.73 pp | −10.27 pp |
| ResNet-18 adapter vs CLIP>PN ViT-B | 2,725× fewer | −1.76 pp | −2.54 pp | −6.73 pp | −8.07 pp |
| ResNet-18 adapter vs Sup-21k>PN ViT-B *(best published)* | 2,703× fewer | −5.26 pp | −3.64 pp | −13.73 pp | −12.17 pp |
| ResNet-18 adapter vs DINO>PN **ResNet-50** | 788× fewer | — | **+3.56 pp** | — | **+5.83 pp** |
| MobileNetV3-S adapter (6,928) vs DINO>PN ViT-S | **3,031× fewer** | −1.76 pp | −7.90 pp | −2.30 pp | −18.18 pp |

**How to read this — three separate conclusions, not one:**

1. **At 5-shot on CIFAR-FS the parameter saving is nearly free.** A 31,744-parameter adapter is **1.06 pp** behind a fully meta-trained DINO ViT-S and **0.76 pp** behind ViT-B, at 662× and 2,703× fewer trainable parameters. That is the thesis's strongest single trade.
2. **We beat the CNN-backboned foundation-model baseline outright.** Against DINO>ProtoNet on ResNet-50, our adapter is **+3.56 pp** (MiniIN 5-shot) and **+5.83 pp** (1-shot) *ahead*, with 788× fewer trainable parameters. When the comparison is backbone-family-matched, parameter efficiency costs nothing at all.
3. **The trade is real at 1-shot and on MiniImageNet, and gets worse on the small backbone.** MobileNetV3-Small loses 7.9 pp on MiniIN 5-shot and 18.2 pp at 1-shot against ViT-S. Do not present the 6,928-parameter cell as universally competitive — it is competitive **on CIFAR-FS**, where it is 1.76 pp behind ViT-S at 3,031× fewer parameters, and it is not on MiniImageNet. §4.1's pretraining-overlap caveat is the likely reason the MiniImageNet gap is so backbone-sensitive: a bigger ImageNet representation carries more of the answer there.

#### 4.4.3 Different protocol — PEFT method budgets on VTAB-1k

These are the standard ViT PEFT methods. **Their accuracies are on VTAB-1k (1,000 labelled examples per task, 19 tasks) and therefore cannot be compared to the few-shot numbers above or to ours** — they are included so the parameter budgets have their own protocol's accuracy attached rather than floating free. From the SSF paper's table on ViT-B/16, verified at [arXiv:2210.08823](https://ar5iv.labs.arxiv.org/html/2210.08823).

| Method | Backbone | Trainable params | VTAB-1k avg accuracy | × our MobileNetV3-S | × our ResNet-18 |
|---|---|---:|---:|---:|---:|
| Full fine-tuning | ViT-B/16 | 85,840,000 | 65.57 | 12,390× | 2,704× |
| SSF | ViT-B/16 | 240,000 | **73.10** | 34.6× | 7.6× |
| VPT-Deep | ViT-B/16 | 600,000 | 69.43 | 86.6× | 18.9× |
| VPT-Shallow | ViT-B/16 | 110,000 | 64.85 | 15.9× | 3.5× |
| Adapter | ViT-B/16 | 270,000 | 55.82 | 39.0× | 8.5× |
| Linear probing | ViT-B/16 | 40,000 | 52.94 | 5.8× | 1.3× |
| CoOp | CLIP | 8,192 | *(not on VTAB-1k)* | 1.2× | 0.3× |
| Full fine-tuning *(our measured baseline)* | ResNet-18 | 11,176,512 | *(different protocol)* | 1,613× | 352× |
| **B-PEFT (ours)** | ResNet-18 | **31,744** | *(few-shot; see §4.4.1)* | — | — |
| **B-PEFT (ours)** | MobileNetV3-S | **6,928** | *(few-shot; see §4.4.1)* | — | — |

**CoOp is the honest exception** on the parameter axis — 8,192 trainable parameters is *below* our ResNet-18 configuration. But its context vectors steer a frozen CLIP that must be resident at inference, so the deployed system is 34.6× larger than ours (§4.4.4), and CoOp reports neither calibration nor OOD detection ([verified at arXiv:2109.01134](https://ar5iv.labs.arxiv.org/html/2109.01134)).

#### 4.4.4 Deployment cost — what has to sit on the device

Trainable parameters are the *adaptation* cost; the frozen backbone is the *deployment* cost, and this is where the CNN choice does its work. Backbone sizes from the Oct 2025 comparative study's Table 1 ([arXiv:2510.04794](https://arxiv.org/html/2510.04794v1)); MobileNetV3-Small's 2.5 M from the MobileNetV3 paper as recorded in this repo's `PAPER SUMMARIES/CNN_paper_summaries.txt`.

| Backbone at inference | Parameters | × our MobileNetV3-Small |
|---|---:|---:|
| DINOv3 ViT-7B | 7,000,000,000 | 2,800× |
| CLIP-ViT-B/32 | 88,200,000 | 35.3× |
| CLIP-ViT-B/16 | 86,500,000 | 34.6× |
| DINO-ViT-B/16 | 85,800,000 | 34.3× |
| CLIP-ResNet-101 | 56,400,000 | 22.6× |
| DINO-ViT-S/16 | 21,000,000 | 8.4× |
| ResNet-18 *(ours)* | ~11,700,000 | 4.7× |
| **MobileNetV3-Small *(ours)*** | **2,500,000** | — |

**The combined statement:** our deployed system is a 2.5 M-parameter frozen backbone plus 6,928 adapted parameters, reaching 90.74 % on CIFAR-FS 5-shot — 1.76 pp behind a fully meta-trained DINO ViT-S that is 8.4× larger at inference and trains 3,031× more parameters.

Two findings from the 2025 literature bear on whether the CNN choice is defensible rather than merely cheap:

- **In low-data regimes CNNs match ViTs.** "In small data scenarios, the inductive bias and smaller capacity of CNNs improve their performance, allowing them to match that of a ViT… CNNs achieve comparable performance in low-data regimes even when the ViTs were pretrained on large-scaled datasets" ([arXiv:2510.04794](https://arxiv.org/html/2510.04794v1), Oct 2025). Few-shot learning is that regime, and §4.4.2's CIFAR-FS 5-shot row is this thesis's own instance of the finding.
- **Transformers do not fit the deployment target.** MCU-class hardware runs 32–512 kB SRAM, under ~1 MB flash and 20–200 MHz clocks; a memory-optimised transformer attention block still costs ~180 ms on an STM32F746 against ~8–12 ms for CNN inference ([arXiv:2506.18927](https://arxiv.org/html/2506.18927v2), 2025).

### 4.5 Is the research question current? — Bayesian PEFT, 2024–2026

The second half of the pre-defence objection was that the question is dated. The record says otherwise: *does uncertainty quantification survive parameter-efficient adaptation, and does it stay calibrated?* has been an active question continuously since 2024.

**What this section establishes, and what it does not.** The table below shows the *topic* is active — it is motivation for why the gap in §4.6 exists and is worth closing, not evidence that this thesis's specific *findings* are novel. "The question is trendy elsewhere" and "my answer to it is new" are separate claims; conflating them is a real risk when presenting this section out loud. The findings-level novelty claims are in §4.7, and they stand or fall on their own, independent of how active the surrounding topic is.

| Work | Year | Contribution | Domain |
|---|---|---|---|
| [Laplace-LoRA](https://proceedings.iclr.cc/paper_files/paper/2024/file/07c256a163a7559186ec1c71e95b9ec9-Paper-Conference.pdf) (ICLR) | 2024 | Post-hoc Laplace over LoRA params; ECE 31.2 % → 2.1 % (Winogrande-small, LLaMA2-7B), 1–5 % memory overhead | LLM |
| [BLoB](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf) (NeurIPS) | 2024 | Bayesian LoRA by backpropagation | LLM |
| [LoRA-Ensemble](https://arxiv.org/html/2405.14438v5) | 2024–25 | Parameter-efficient ensembling for uncertainty in self-attention networks | ViT |
| [Scalable Bayesian LoRA](https://arxiv.org/pdf/2506.21408) | 2025 | Stochastic variational subspace inference | LLM |
| [Calibrated Adaptation (Stiefel-Bayes)](https://arxiv.org/html/2602.17809) | 2026 | Geometry-aware Bayesian prior, <8 % wall-clock overhead vs deterministic LoRA | LLM |
| [BaRA](https://arxiv.org/pdf/2606.29184) | 2026 | Bayesian adaptive rank allocation | LLM |
| [Bayesian Sparse LoRA](https://arxiv.org/html/2607.02182v1) | 2026 | Sparse Bayesian posterior over LoRA | LLM |
| [Bayesian Adaptation Gym](https://arxiv.org/pdf/2606.22188) | 2026 | A *benchmark* for Bayesian low-rank adaptation | Multi-modal LM |
| [BayesAdapter](https://arxiv.org/abs/2412.09718) | 2025–26 | Variational Bayes over a linear CLIP few-shot adapter's weights; ~2.5 % ECE gain over deterministic | **Vision (CLIP, non-edge)** |
| **B-PEFT (this thesis)** | 2026 | Evidential Dirichlet uncertainty over a parameter-free prototype head on a frozen lightweight CNN, ≤31.7 k trainable params, few-shot episodic, calibration **and** OOD **and** parameter budget all reported | **Vision, edge** |

Every entry above except LoRA-Ensemble and BayesAdapter is on language models; those two are vision but neither is edge-deployable (ViT/CLIP-ResNet-50, not a small CNN). None is few-shot episodic vision on an edge-deployable backbone.

**A 2026 theory result converges with our RQ2 negative.** §3's finding — evidential worse-calibrated in 20/20 matched comparisons — is not contrarian. A 2026 analysis of second-order/evidential classification reports that standard reverse-KL EDL objectives yield non-vanishing epistemic uncertainty even in the infinite-data limit ([arXiv:2606.10777](https://arxiv.org/pdf/2606.10777)). Empirics and theory point the same way.

**The closest prior work, and the regime boundary it defines.** [Bayesian Evidential Learning for Few-Shot Classification](https://ar5iv.labs.arxiv.org/html/2207.13137) (BEL) runs evidential Dirichlet uncertainty on few-shot episodes with ResNet-12/Conv-4, reporting miniImageNet 63.10 / 79.60 and CIFAR-FS 73.96 / 86.92 (1-/5-shot) — and it reports ECE *improving* (3.59 % vs 14.69 %, miniImageNet 5-shot). It reports no OOD AUROC, no parameter counts, and its backbone is meta-trained rather than frozen.

BEL's calibration result therefore points opposite to ours, and the difference is the regime, not a contradiction: BEL meta-trains a backbone and fuses two networks' evidence; we freeze the backbone and cap the trainable budget at ≤31,744 parameters — including one configuration with **2**. Establishing where that boundary sits is a contribution in itself, and it is the edge-relevant side of the boundary, because meta-training a backbone is precisely what a 256 kB device cannot do.

**A second, independent data point points the same way as BEL.** BayesAdapter (above) finds a Bayesian method — variational Bayes over adapter weights, not evidential Dirichlet — *improving* calibration by ~2.5% ECE on a frozen CLIP backbone with up to 32 shots and a full linear-layer adapter, not a budget capped at a handful of parameters. That is now two independent papers, with two different Bayesian mechanisms, both finding calibration improves when the model has more capacity or data to work with than this thesis's grid provides. That makes the degradation this thesis reports a sharper, more specific finding — not an outlier result contradicted by the field, but the low-capacity end of a pattern the field has now shown on both sides of.

### 4.6 What the published work does **not** report — the actual gap

This is where the thesis's claim to novelty lives. Seven literatures — across every backbone family, not just CNNs — each cover part of the problem, and none covers the intersection:

| Literature | Representative work | Accuracy | Macro-F1 | Calibration (ECE) | OOD AUROC | Param budget | Few-shot episodic | Edge-deployable backbone |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical few-shot meta-learning | ProtoNet, MAML, R2D2, MetaOptNet | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot | P>M>F, [CoOp](https://ar5iv.labs.arxiv.org/html/2109.01134), [Tip-Adapter](https://arxiv.org/abs/2207.09519), [CLIP-Adapter](https://arxiv.org/abs/2110.04544), [DINOv3](https://arxiv.org/html/2508.10104v1) | ✅ | ❌ | ❌ | ❌ | partial | partial | ❌ |
| PEFT for vision transformers | VPT, AdaptFormer, [SSF](https://ar5iv.labs.arxiv.org/html/2210.08823), FacT, NOAH | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge | [Conv-Adapter](https://arxiv.org/abs/2208.07463), [LoRA-C](https://arxiv.org/abs/2410.16954), [LoRA-Edge](https://arxiv.org/abs/2511.03765), [CoLoRA](https://arxiv.org/html/2505.18315) | ✅ | ❌ | ❌ | ❌ | ✅ | partial | ✅ |
| Frozen-CNN episodic few-shot adapters | [TSA](https://arxiv.org/abs/2107.00358), [FiT](https://arxiv.org/pdf/2206.08671) | ✅ | ❌ | ❌ | ❌ | ✅ | partial | partial |
| Bayesian PEFT | Laplace-LoRA, BLoB, [BaRA](https://arxiv.org/pdf/2606.29184), [Stiefel-Bayes](https://arxiv.org/html/2602.17809), [BayesAdapter](https://arxiv.org/abs/2412.09718) | ✅ | ❌ | ✅ | partial | ✅ | ❌ | ❌ |
| Evidential few-shot | [BEL](https://ar5iv.labs.arxiv.org/html/2207.13137) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | partial |
| TinyML / TinyDL | [TinyDL survey](https://arxiv.org/html/2506.18927v2) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **B-PEFT (this thesis)** | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Notes on how this table was built, so it can be defended:

- **The macro-F1 column is ❌ for every prior work in the table.** None of the comparison methods in §4.4.1 reports macro-F1 — few-shot vision reports accuracy by convention. Since 5-way episodes are class-balanced this is defensible, but it hides per-class recall collapse, which is exactly what our 1-shot F1-vs-accuracy gap exposes (§2, Table 1 reading notes).
- **Two of the ❌ columns are documented absences, not inferences.** The PEFT-for-vision-transformers survey ([arXiv:2402.02242](https://arxiv.org/html/2402.02242v1)) — covering the entire ViT PEFT field — does not discuss calibration, uncertainty quantification, or OOD detection anywhere. The TinyML→TinyDL survey ([arXiv:2506.18927](https://arxiv.org/html/2506.18927v2), 2025) covers quantisation extensively and makes no mention of uncertainty estimation, calibration, or OOD detection. Two independent surveys, two different communities, the same blind spot.
- The PEFT-for-edge row reflects what those papers foreground: LoRA-Edge reports accuracy within 4.7 % of full fine-tuning while updating at most 1.49 % of parameters; LoRA-C reports accuracy gains on CIFAR-10/100/CIFAR-10-C/Icons-50; CoLoRA reports accuracy and AUC on an OCT classification task. Calibration error and OOD-detection AUROC are not headline metrics in any of them. **This claim was checked against their abstracts and summaries, not by exhaustively reading every table — verify before making it in the written thesis.**
- **The new "Frozen-CNN episodic few-shot adapters" row holds this thesis's single closest prior work, TSA, and RQ1's headline finding should be framed as confirming it, not discovering it.** [Task-Specific Adapters](https://arxiv.org/abs/2107.00358) (Li, Liu & Bilen, CVPR 2022) runs the serial-vs-parallel placement comparison on a **frozen ResNet-18**, over **600 sampled episodic tasks** on held-out domains, with a **parameter-free nearest-centroid head** — matching this thesis's backbone, episode count, placement question, and head design — and finds parallel wins, the same direction as Step 6. This is a closer match than Conv-Adapter, which is the secondary precedent (§ RQ1 above) for the separate locality-preservation angle. [FiT](https://arxiv.org/pdf/2206.08671) is the row's other entry: frozen CNN backbones with FiLM adapters as small as 11,648 parameters and a ProtoNets head option, but no placement or adapter-type comparison and no disjoint-class episodic evaluation. The row's "Few-shot episodic" cell is marked **partial**: TSA is genuinely episodic, FiT is not (VTAB-1k / fixed-benchmark evaluation). "Edge-deployable backbone" is also **partial**: both use ResNet-18/50-class backbones, not MobileNet-class ones, and FiT's EfficientNetV2-M/BiT-M-R50x1 are larger still. See the RQ1 section (§3) and claim 7 (§4.7 — claims are ordered by defensibility, not RQ number) for the full writeup.
- **Conv-Adapter** independently established that locality-preserving CNN adapters beat 1×1/linear-style ones. This should not be equated with this thesis's LoRA-vs-bottleneck result: a 1×1 convolution (this thesis's LoRA target) still preserves per-pixel spatial locality, so the LoRA finding is a separate, corroborating data point about transformer-native reparameterisations underperforming on CNNs, not an instance of Conv-Adapter's named "loss of locality" failure mode. Its "Few-shot episodic" cell stays marked **partial** for the reason already stated: Conv-Adapter evaluates 1/2/4/8-shot fine-tuning on FGVC datasets with a trainable per-task head, not disjoint-class episodic meta-testing.
- **BayesAdapter** ([Morales-Álvarez et al., IJCV 2025/2026, arXiv:2412.09718](https://arxiv.org/abs/2412.09718)) is added to the Bayesian-PEFT row as a **third** confirmed case — alongside BEL (§4.5) — of a Bayesian uncertainty method *improving* calibration rather than degrading it: variational Bayes over a linear CLIP adapter's weights, ~2.5 % ECE gain over the deterministic baseline. It is also the first entry in that row that is **vision, not language** — the "all of them are on language models" summary in §4.5 needs updating accordingly, though BayesAdapter is still not vision-*at-the-edge* (CLIP ResNet-50/ViT-B/16, not a small CNN). It reports no OOD-AUROC and no placement/adapter-type comparison, and strengthens rather than weakens this thesis's RQ2 boundary claim: a third data point where Bayesian calibration improves in a higher-capacity or higher-shot setting makes the frozen, extreme-low-parameter degradation this thesis reports a sharper, more distinctive boundary case, not a contradicted one.
- **The Foundation-model-few-shot row's "Few-shot episodic" cell is marked partial, not ✅ — verified by reading the full text of Tip-Adapter and CLIP-Adapter, not assumed from their category.** P>M>F genuinely runs N-way K-shot episodic testing on disjoint held-out classes, the same protocol this thesis uses (§4.3/§4.4). CoOp, Tip-Adapter and CLIP-Adapter do not: Tip-Adapter's own paper states it explicitly, contrasting itself with meta-learning protocols that *"split the same dataset into three sub-sets of different categories"* — Tip-Adapter instead trains and tests on the **same** classes (e.g. all 1,000 ImageNet classes), reporting one accuracy number on the full fixed test set rather than an average over sampled episodes. Neither paper reports ECE, Brier, or OOD-detection AUROC (CLIP-Adapter reports accuracy-under-distribution-shift on ImageNet-A/R/Sketch/V2, which is related but not the same measurement). Their adapters are also larger than this thesis's budget: CLIP-Adapter's visual adapter alone is 0.52 M parameters on a ResNet-50 backbone (17–75× this thesis's 6,928–31,746 range); Tip-Adapter-F's fine-tuned cache runs into the millions on ImageNet. The frozen-backbone-plus-small-adapter *idea* is shared with this line of work; the disjoint-class episodic protocol, the parameter scale, and the calibration/OOD measurement are not.
- The Bayesian-PEFT row is marked "partial" on OOD because several of those works evaluate selective prediction or OOD-adjacent tasks rather than reporting OOD AUROC against held-out datasets in the way this thesis does; check the individual paper before citing it as an OOD result.
- CoOp is marked "partial" on parameter budget because it states its context-vector count but does not frame parameter efficiency as a contribution; and ❌ on edge-deployability because it requires a resident CLIP at inference (§4.4).
- The evidential row reflects that EDL work overwhelmingly operates in the full-data training regime, where the entire network is trained with the evidential loss. The regime this thesis studies — an evidential interpretation layered on a *parameter-free prototype head* over a *frozen* backbone, with the entire trainable budget under ~32 k parameters and as low as **2** — is not a setting the EDL literature reports on. BEL (§4.5) is the nearest miss: few-shot and evidential, but meta-trained backbone, no OOD, no parameter accounting.

### 4.7 The publishable claims

Stated as a supervisor would want to see them — each one falsifiable and backed by a specific table above. **Ordered by defensibility, not by RQ number**: rigor (is the measurement solid) and novelty (is the finding new, or does it confirm/extend prior work) are two separate axes, and this list leads with the claims that are strongest on *both*. Claims 1–4 are genuinely new findings; claims 5–7 are solid, useful, but either replicate or closely extend prior work — say so plainly if asked, rather than presenting all seven as equally novel.

1. **Evidential calibration degrades specifically when the backbone is frozen and the parameter budget is starved — contradicting the nearest prior work, and explaining why.** [BEL](https://ar5iv.labs.arxiv.org/html/2207.13137) found evidential calibration *improving* (3.59 % vs 14.69 % baseline) when it meta-trains the whole backbone. [BayesAdapter](https://arxiv.org/abs/2412.09718) (a different Bayesian mechanism — variational Bayes over adapter weights, not evidential Dirichlet) independently found calibration *improving* by ~2.5% ECE on a frozen CLIP backbone with up to 32 shots and a full linear adapter. We find calibration *degrading*, 0/20, when the backbone is frozen **and** the trainable budget is capped at ≤31,744 parameters (down to 2 in one cell). All three results are true — the difference is capacity and data, not a contradiction — and locating exactly where the boundary sits is itself the contribution, not a failure to reproduce prior work. *(§3 RQ2, §4.5.)*
2. **Calibration quality and OOD-ranking quality are empirically decoupled.** The same evidential head that is 5–51× *worse* calibrated is simultaneously a *better* OOD detector than max-softmax-probability in 93.8 % of comparisons and than temperature-scaled MSP in 95.0 %. Reporting ECE alone would have hidden this entirely; so would reporting AUROC alone. Nobody in the surveyed literature reports both together on the same runs. *(§3 RQ2 + RQ3, Tables 2, 3, 5.)*
3. **A training-free energy score beats the Bayesian one in ~70 % of comparisons — a correction of this project's own earlier claim.** Step 4.5 (one configuration) found evidential roughly on par with energy; the 120-run grid overturns that. Volunteering a self-correction at scale is itself evidence the grid-level results can be trusted over the single-configuration pilot. *(§3 RQ3, Table 5.)*
4. **The evidential advantage over probabilistic softmax scores held between the two shot regimes tested, in the predicted direction.** Near-OOD Δ AUROC +0.064 at 1-shot vs +0.043 at 5-shot (only two shot levels were tested, so "held between the two points measured" is the accurate description — not a continuous trend traced across many). *(§3 RQ3.)*
5. **A ~7 k-parameter adapter on a frozen MobileNetV3-Small matches full fine-tuning of an 11.18 M-parameter ResNet-18** on CIFAR-FS 5-shot (90.74 ± 0.17 % vs 90.47 %) — 0.06 % of the trainable budget. Phrase this as *matches*, not *beats*: the +0.27 pp margin is smaller than the MobileNet cell's own seed spread, and the Full-FT baseline has no seed spread to compare against (Table 1 reading notes). The safe version of the claim is that a 6,928-parameter adapter reaches full-fine-tuning accuracy, which is the interesting result anyway. The ResNet-18 parallel bottleneck's **+0.98 pp** over Full-FT at 31,744 parameters (91.44 ± 0.14 % vs 90.47 %) is the margin that comfortably clears seed noise. *(Table 6.)*
6. **The parameter/accuracy trade is favourable and quantified, not asserted.** At 5-shot on CIFAR-FS a 31,744-parameter adapter is **1.06 pp** behind a fully meta-trained DINO ViT-S at **662× fewer trainable parameters**, and **0.76 pp** behind ViT-B at **2,703× fewer**; against the backbone-matched DINO>ProtoNet ResNet-50 it is **+3.56 pp ahead** on MiniIN 5-shot at 788× fewer. The trade turns unfavourable at 1-shot and on MiniImageNet with the small backbone (up to −18.2 pp) — stated, not hidden. Meanwhile the adaptation budget is 35–87× smaller than standard ViT PEFT methods and the backbone 34.6× smaller at inference than CLIP-ViT-B/16, while reporting calibration, OOD *and* macro-F1 that none of those methods report at all. *(§4.4, §4.6.)*
7. **Parallel convolutional bottleneck adapters strictly dominate LoRA for few-shot CNN adaptation** — 16 / 16 matched comparisons, +2.1 to +8.3 pp, and on MobileNetV3-Small they win while using *fewer* parameters. **This confirms prior work rather than discovering it, and the closest precedent is closer than a first check suggested**: [Task-Specific Adapters](https://arxiv.org/abs/2107.00358) (Li, Liu & Bilen, CVPR 2022) already runs the serial-vs-parallel comparison on a frozen ResNet-18 with a parameter-free nearest-centroid head over 600 sampled episodic tasks, and finds parallel wins — same backbone, same episode count, same question, same answer. Conv-Adapter independently established that locality-preserving adapters beat 1×1/linear-style ones, but that finding should not be equated with the LoRA-vs-bottleneck result here: a 1×1 convolution (our LoRA target) preserves spatial locality, so it is a separate, corroborating data point, not the same named failure mode. What's new after both are accounted for is the LoRA-vs-bottleneck adapter-*type* comparison itself (neither paper runs it), a second backbone (MobileNetV3-Small), and the evidential-uncertainty/calibration/OOD layer. *(§3 RQ1, Tables 1 and 6.)*

### 4.8 What would strengthen the publication case further

Honest gaps, in the order a reviewer would raise them:

1. **A from-scratch control.** Running the same grid with a randomly-initialised backbone trained on the 64 base classes would let the thesis report one number in Regime A and remove the "your accuracy is contaminated" objection entirely. This is the single highest-value addition.
2. **A transformer arm under our own protocol.** §4.4 argues from published ViT PEFT parameter counts rather than from a ViT we ran ourselves. A ViT-Tiny or DeiT-Small arm inside this grid would convert that argument into a measurement, and is the most direct answer to the "CNNs are outdated" objection ([DEFENCE_BRIEF.md](DEFENCE_BRIEF.md) §7).
3. **Latency measurement (RQ4 / Step 11)** — code-complete (`src/utils/efficiency.py`, `scripts/efficiency_table.py`, `scripts/pareto_plots.py`); the canonical Kaggle T4 GPU + Kaggle CPU session that would fill Table 8 and the RQ4 paragraph above has not yet run.
4. **Per-cell hyperparameter tuning.** The entire grid runs one frozen recipe (§5). A reviewer will ask whether the evidential calibration gap is a tuning artefact; the Step 4.5 VAL sweep says no (the ECE surface is flat at ≈ 0.285–0.296), but that sweep covered one configuration, not the grid.
5. **More than three seeds** for the cells where a headline claim rests on a <1 pp difference (notably claim 2 above, +0.27 pp).

---

## 5. Limitations that apply to every number in this document

1. **Frozen recipe.** All 120 cells use one training recipe — LR 5e-3, LoRA rank 16, `kl_weight_max` 0.1, evidence affine (scale 2, bias −6) — VAL-tuned once on ResNet-18 × CIFAR-FS × 5-shot in Step 4.5 and carried unchanged across two backbones, two datasets and two shot regimes it was never re-tuned for. The grid answers *"how do these axes compare under one fixed recipe"*, not *"what is the best achievable number per cell"*. This was a deliberate design decision (`plan.md` §3) so that all 40 configurations remain a controlled comparison rather than 40 independently-tuned numbers.
2. **ImageNet-pretraining overlap** (§4.1). Central to any accuracy claim, acute on MiniImageNet.
3. **Three seeds, and effectively one for the baselines.** Seed spread exceeds 1 pp in only 2 of 40 configurations, but n = 3 is thin for differences under ~1 pp — and for `Full-FT*` and `Linear-Probe*` the seed axis is inert by construction (Table 1 reading notes), so those two rows are single measurements with no variance estimate at all.
4. **Evidential calibration is not temperature-scalable.** Softmax cells get a post-hoc TS column; evidential cells have no equivalent post-hoc correction in this codebase, so the TS comparison is inherently favourable to softmax. It remains the right comparison — TS is cheap and standard — but the asymmetry should be stated.
5. **Inherited open issue: TinyImageNet OOD-set exclusion.** `step_writeups/step9.txt` §5 documents a test-isolation bug in the TinyImageNet class-exclusion check that was traced by code inspection but never confirmed by a logged exclusion count from a real run. All TinyImageNet near-OOD numbers in Tables 3–5 carry that caveat.
6. **Plot resolution.** The 300-dpi figures in `results/grid_plots/` are a mechanical raster upscale of `scripts/evaluate.py`'s 200-dpi output, not a vector re-render.
7. **Latency and peak-memory numbers (Step 11) are the only ones in this document that are NOT byte-reproducible by design.** Unlike every other number here, re-running `scripts/efficiency_table.py` on the same hardware yields a different (though close) value each time — host load, GPU clocks/thermals, driver version, cuDNN kernel selection and allocator history all vary session to session. `results/efficiency_table.json`'s `reproducibility` block states this explicitly and asserts byte-identical reproducibility only for the deterministic `static` block (trainable/total params, FLOPs). Every measured leaf names the session and hardware profile that produced it so a number can never be read out of context.

---

## 6. Provenance and file map

| Artefact | Path |
|---|---|
| Aggregated results (source of this document) | `results/mvt_results.json` |
| Per-cell metrics, 120 runs | `results/grid/*_metrics.json` |
| Run log (status, wall time, errors per cell) | `results/grid/_run_log.jsonl` |
| Grid config index (cell → config → results path) | `configs/grid/_index.json` |
| Master tables, LaTeX + PNG | `results/mvt_table_{accuracy,calibration,ood_auroc}.{tex,png}` |
| 16 reliability / OOD-histogram plots | `results/grid_plots/` + `_manifest.json` |
| Step 10 write-up | `step_writeups/step10.txt` |
| Step 10 explainer | `docs/explainers/step10.md` |
| Pre-defence brief ("isn't a CNN backdated?") | `docs/DEFENCE_BRIEF.md` |
| Step 11 efficiency measurement (params/FLOPs/latency/memory) | `results/efficiency_table.json` |
| Step 11 Pareto-frontier membership under every axis variant | `results/pareto_frontier.json` |
| Step 11 Pareto figures | `results/pareto_{params_vs_accuracy,latency_vs_auroc}*.png` |
| Step 11 quality-axis sensitivity audit | `results/pareto_audit/` + `_manifest.json` |
| Step 11 write-up | `step_writeups/step11.txt` |

**Sources cited in §4:**

- [Lee et al., Meta-Learning with Differentiable Convex Optimization (MetaOptNet), CVPR 2019 — arXiv:1904.03758](https://arxiv.org/abs/1904.03758)
- [Hu et al., Pushing the Limits of Simple Pipelines for Few-Shot Learning (P>M>F), CVPR 2022 — arXiv:2204.07305](https://ar5iv.labs.arxiv.org/html/2204.07305)
- [Li, Liu & Bilen, Cross-domain Few-shot Learning with Task-specific Adapters (TSA), CVPR 2022 — arXiv:2107.00358](https://arxiv.org/abs/2107.00358) (primary closest precedent for RQ1's serial-vs-parallel finding)
- [Chen et al., Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets — arXiv:2208.07463](https://arxiv.org/abs/2208.07463) (secondary precedent, locality-preservation angle)
- [Bateni et al., FiT: Parameter Efficient Few-shot Transfer Learning — arXiv:2206.08671](https://arxiv.org/pdf/2206.08671)
- [Morales-Álvarez et al., BayesAdapter: Enhanced Uncertainty Estimation in CLIP Few-Shot Adaptation, IJCV 2025–26 — arXiv:2412.09718](https://arxiv.org/abs/2412.09718)
- [Zhang et al., Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification — arXiv:2207.09519](https://arxiv.org/abs/2207.09519) (verified: same-class train/test, single fixed test-set accuracy, no ECE/OOD reported — §4.6 note)
- [Gao et al., CLIP-Adapter: Better Vision-Language Models with Feature Adapters — arXiv:2110.04544](https://arxiv.org/abs/2110.04544) (verified: 0.52M-parameter visual adapter, no ECE/OOD reported — §4.6 note)
- [Zhou et al., LoRA-C: Parameter-Efficient Fine-Tuning of Robust CNN for IoT Devices — arXiv:2410.16954](https://arxiv.org/abs/2410.16954)
- [LoRA-Edge: Tensor-Train-Assisted LoRA for Practical CNN Fine-Tuning on Edge Devices — arXiv:2511.03765](https://arxiv.org/abs/2511.03765)
- [CoLoRA: Parameter-Efficient Fine-Tuning for Convolutional Models — arXiv:2505.18315](https://arxiv.org/html/2505.18315)
- [Gao et al., A Comprehensive Survey on Evidential Deep Learning and Its Applications — arXiv:2409.04720](https://arxiv.org/pdf/2409.04720)
