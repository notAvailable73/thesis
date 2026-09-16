# RQ2/RQ4 completion + RQ1 interactions: session report

Generated 2026-09-14 22:51:33. GPU `Tesla T4`, torch `2.10.0+cu128`, CUDA `12.8`, repo HEAD `8d706ea`.

Thesis numbering throughout: code `rq1_verdict` = **RQ2**, code `rq2_verdict` = **RQ4**.

Frozen paths untouched (configs, results/grid, results/mvt_results.json, results/rq_summary.json, results/rq_checkpoint_audit.json): **True**

## The four deliverables

1. **Coverage:** 120/120 Phase A records (21/21 new).
2. **RQ2 (all records):** far: score 42.69% / objective 0.170% = 250.5×; near: score 14.72% / objective 0.634% = 23.2×. Fully crossed: True.
3. **RQ4:** ECE improved in 60/60 evidential cells; AUROC preserved in 192/240 comparisons (committed: 48/48 and 150/192).
4. **RQ1 interactions** (four full tables in the RQ1 section below). Headline numbers:

| outcome | n | residual (main only) | residual (+2-way) | all 2-way | `backbone:adapter` [95% CI] | largest 2-way term |
|---|---:|---:|---:|---:|---:|---|
| accuracy (as committed) | 96 | 8.95% | 1.31% | 7.64% | 0.73% [0.53%, 0.95%] | dataset:backbone 5.27% |
| ece (as committed) | 96 | 7.70% | 2.05% | 5.65% | 3.28% [2.52%, 4.15%] | backbone:adapter 3.28% |
| far_ood_svhn (as committed) | 96 | 23.87% | 9.60% | 14.27% | 0.18% [0.00%, 0.73%] | dataset:backbone 9.18% |
| near_ood_tin (as committed) | 95 | 11.42% | 6.23% | 5.18% | 0.00% [0.00%, 0.06%] | dataset:backbone 3.81% |
| accuracy (smoke obs. replaced) | 96 | 9.04% | 1.28% | 7.76% | 0.77% [0.58%, 0.98%] | dataset:backbone 5.37% |
| ece (smoke obs. replaced) | 96 | 7.68% | 2.03% | 5.64% | 3.23% [2.49%, 4.05%] | backbone:adapter 3.23% |
| far_ood_svhn (smoke obs. replaced) | 96 | 23.75% | 9.53% | 14.22% | 0.20% [0.00%, 0.75%] | dataset:backbone 9.11% |
| near_ood_tin (smoke obs. replaced) | 96 | 11.49% | 6.21% | 5.28% | 0.00% [0.00%, 0.05%] | dataset:backbone 3.90% |


### Regression guard against the committed Step 10 grid

Evaluated 21/21. Guard status counts: `exact` 21. Reproduced (exact or within 1e-6): **21/21**. `best_val_epoch` matches: **21/21**. `n_params` all match: True.

| cell | checkpoint | baseline | guard | exact/keys | max abs diff | best_val_epoch committed → new |
|---|---|---|---|---:|---:|---:|
| `cifar_fs_5shot_mobilenetv3_small_lora_evidential_seed42` | trained_this_session | first-20-episode (committed JSON is a smoke run) | `exact` | 8/8 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_mobilenetv3_small_lora_evidential_seed43` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 2 → 2 |
| `cifar_fs_5shot_mobilenetv3_small_lora_evidential_seed44` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_resnet18_lora_evidential_seed42` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 4 → 4 |
| `cifar_fs_5shot_resnet18_lora_softmax_seed42` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 6 → 6 |
| `cifar_fs_5shot_resnet18_lora_evidential_seed43` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_resnet18_lora_softmax_seed43` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 4 → 4 |
| `cifar_fs_5shot_resnet18_lora_evidential_seed44` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 7 → 7 |
| `cifar_fs_5shot_resnet18_lora_softmax_seed44` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 4 → 4 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_evidential_seed42` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_softmax_seed42` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 10 → 10 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_evidential_seed43` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_softmax_seed43` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 10 → 10 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_evidential_seed44` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 3 → 3 |
| `cifar_fs_5shot_mobilenetv3_small_bottleneck_parallel_softmax_seed44` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 14 → 14 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_evidential_seed42` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 19 → 19 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_softmax_seed42` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 10 → 10 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_evidential_seed43` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 2 → 2 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_softmax_seed43` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 2 → 2 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_evidential_seed44` | trained_this_session | 600-episode | `exact` | 12/12 | 0.00e+00 | 12 → 12 |
| `cifar_fs_5shot_resnet18_bottleneck_parallel_softmax_seed44` | trained_this_session | 600-episode | `exact` | 13/13 | 0.00e+00 | 10 → 10 |

## RQ2 / RQ4 re-aggregation

Phase A factorial records: **120/120** (99 committed + 21 new). Evidential records: 60.

Unchanged aggregator reproduces the committed `results/rq_summary.json` on the committed records: **True** (RQ2 max abs diff 0.0e+00, RQ4 0.0e+00).

Design completeness: 20/20 complete, 0 absent, 0 with one objective arm missing, 0 with seeds missing. Zero-observation objective×score cells: far 0, near 0. **Fully crossed: True.**

### RQ2: objective vs scoring rule (η² share of OOD-AUROC variance)

| variant | pool | score η² | objective η² | score/objective | balanced | n |
|---|---|---:|---:|---:|---|---:|
| before: committed, 99 records | far | 43.68% | 0.267% | 163.4× | True | 792 |
| before: committed, 99 records | near | 12.96% | 0.601% | 21.6× | True | 792 |
| after: all records | far | 42.69% | 0.170% | 250.5× | True | 960 |
| after: all records | near | 14.72% | 0.634% | 23.2× | True | 960 |

Quotable variant: **all records**

Mean AUROC, far-OOD, after (rows = training objective):

| trained as | msp | energy | ts_msp | vacuity_valfit |
|---|---:|---:|---:|---:|
| evidential | 0.7993 (n=120) | 0.9185 (n=120) | 0.7868 (n=120) | 0.9134 (n=120) |
| softmax | 0.8014 (n=120) | 0.9342 (n=120) | 0.7850 (n=120) | 0.9307 (n=120) |

Mean AUROC, near-OOD, after (rows = training objective):

| trained as | msp | energy | ts_msp | vacuity_valfit |
|---|---:|---:|---:|---:|
| evidential | 0.7916 (n=120) | 0.8414 (n=120) | 0.7805 (n=120) | 0.8347 (n=120) |
| softmax | 0.7770 (n=120) | 0.8371 (n=120) | 0.7634 (n=120) | 0.8233 (n=120) |

### RQ4: post-hoc evidence-affine refit (VAL only)

| variant | evidential cells | ECE improved | AUROC preserved (Δ ≥ −0.005) | mean ΔECE | mean ΔAUROC | min Spearman ρ | reordering observed |
|---|---:|---:|---:|---:|---:|---:|---|
| before: committed | 48 | 48/48 | 150/192 | -0.1373 | +0.0040 | 0.9212 | True |
| after: all records | 60 | 60/60 | 192/240 | -0.1387 | +0.0029 | 0.8658 | True |
| new cells only | 12 | 12/12 | 42/48 | -0.1441 | -0.0016 | 0.8658 | True |

## RQ1: two-way interactions (committed grid, as published)

Factors dataset, k_shot, backbone, adapter, head; baselines excluded (full_ft, linear_probe). Seed bootstrap: 2000 resamples, rng seed 20260914.

### accuracy: `accuracy_mean`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Published main effects reproduced: **True** (max abs diff 0.004 pp). Residual df with two-way terms: 80.

Residual **8.95%** with main effects only → **1.31%** after adding the ten two-way terms (which together take 7.64%).

`backbone:adapter` = **0.73%**, which is 8.15% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| k_shot | 76.05% | [74.90%, 77.39%] | 0.63% | <1e-6 |
| adapter | 9.14% | [8.49%, 9.80%] | 0.34% | <1e-6 |
| dataset:backbone | 5.27% | [4.78%, 5.73%] | 0.25% | <1e-6 |
| backbone | 3.92% | [3.46%, 4.48%] | 0.26% | <1e-6 |
| dataset | 1.75% | [1.46%, 2.05%] | 0.15% | <1e-6 |
| k_shot:backbone | 0.97% | [0.75%, 1.23%] | 0.12% | <1e-6 |
| **backbone:adapter** | 0.73% | [0.53%, 0.95%] | 0.10% | <1e-6 |
| adapter:head | 0.33% | [0.21%, 0.47%] | 0.07% | 2.2e-05 |
| head | 0.19% | [0.10%, 0.29%] | 0.05% | 0.0012 |
| dataset:k_shot | 0.15% | [0.07%, 0.25%] | 0.05% | 0.0036 |
| backbone:head | 0.08% | [0.03%, 0.16%] | 0.04% | 0.028 |
| dataset:adapter | 0.03% | [0.00%, 0.08%] | 0.02% | 0.19 |
| dataset:head | 0.03% | [0.00%, 0.08%] | 0.02% | 0.2 |
| k_shot:head | 0.03% | [0.00%, 0.08%] | 0.02% | 0.21 |
| k_shot:adapter | 0.02% | [0.00%, 0.06%] | 0.02% | 0.3 |
| residual | 1.31% | [1.06%, 1.46%] | 0.10% |  |

### ece: `ece_pooled`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Published main effects reproduced: **True** (max abs diff 0.005 pp). Residual df with two-way terms: 80.

Residual **7.70%** with main effects only → **2.05%** after adding the ten two-way terms (which together take 5.65%).

`backbone:adapter` = **3.28%**, which is 42.60% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| head | 82.89% | [81.97%, 83.87%] | 0.50% | <1e-6 |
| backbone | 4.63% | [3.76%, 5.55%] | 0.46% | <1e-6 |
| **backbone:adapter** | 3.28% | [2.52%, 4.15%] | 0.42% | <1e-6 |
| k_shot | 2.13% | [1.58%, 2.80%] | 0.31% | <1e-6 |
| dataset | 2.02% | [1.42%, 2.70%] | 0.32% | <1e-6 |
| k_shot:backbone | 0.89% | [0.49%, 1.38%] | 0.22% | <1e-6 |
| adapter | 0.63% | [0.32%, 1.04%] | 0.18% | 3.7e-06 |
| dataset:backbone | 0.60% | [0.31%, 0.99%] | 0.18% | 5.9e-06 |
| dataset:adapter | 0.34% | [0.12%, 0.64%] | 0.14% | 0.0005 |
| k_shot:adapter | 0.20% | [0.05%, 0.45%] | 0.11% | 0.0063 |
| k_shot:head | 0.19% | [0.05%, 0.44%] | 0.10% | 0.0073 |
| dataset:head | 0.09% | [0.01%, 0.28%] | 0.07% | 0.065 |
| backbone:head | 0.02% | [0.00%, 0.14%] | 0.04% | 0.36 |
| adapter:head | 0.02% | [0.00%, 0.13%] | 0.04% | 0.41 |
| dataset:k_shot | 0.01% | [0.00%, 0.11%] | 0.03% | 0.48 |
| residual | 2.05% | [1.29%, 2.39%] | 0.28% |  |

### far_ood_svhn: `ood_auroc__svhn_far__{native}`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Published main effects reproduced: **True** (max abs diff 0.004 pp). Residual df with two-way terms: 80.

Residual **23.87%** with main effects only → **9.60%** after adding the ten two-way terms (which together take 14.27%).

`backbone:adapter` = **0.18%**, which is 0.75% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| head | 39.01% | [34.59%, 43.81%] | 2.33% | <1e-6 |
| k_shot | 22.68% | [19.24%, 26.40%] | 1.85% | <1e-6 |
| backbone | 11.54% | [8.93%, 14.75%] | 1.50% | <1e-6 |
| dataset:backbone | 9.18% | [6.68%, 11.79%] | 1.30% | <1e-6 |
| dataset:head | 2.07% | [1.00%, 3.50%] | 0.64% | 8.1e-05 |
| adapter | 1.73% | [0.77%, 2.85%] | 0.54% | 0.00029 |
| backbone:head | 1.35% | [0.55%, 2.49%] | 0.51% | 0.0012 |
| dataset | 1.18% | [0.43%, 2.25%] | 0.47% | 0.0024 |
| k_shot:head | 0.62% | [0.15%, 1.41%] | 0.33% | 0.025 |
| adapter:head | 0.39% | [0.04%, 1.05%] | 0.27% | 0.076 |
| k_shot:backbone | 0.21% | [0.00%, 0.80%] | 0.21% | 0.19 |
| **backbone:adapter** | 0.18% | [0.00%, 0.73%] | 0.20% | 0.22 |
| dataset:k_shot | 0.13% | [0.00%, 0.62%] | 0.17% | 0.31 |
| dataset:adapter | 0.10% | [0.00%, 0.52%] | 0.15% | 0.36 |
| k_shot:adapter | 0.03% | [0.00%, 0.35%] | 0.10% | 0.61 |
| residual | 9.60% | [7.09%, 10.56%] | 0.88% |  |

### near_ood_tin: `ood_auroc__tin_near__{native}`, n=95, balanced=False

Method: OLS Type II (eta_squared is not exact on unbalanced rows). Published main effects reproduced: **True** (max abs diff 0.004 pp). Residual df with two-way terms: 79.

Residual **11.42%** with main effects only → **6.23%** after adding the ten two-way terms (which together take 5.18%).

`backbone:adapter` = **0.00%**, which is 0.00% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| k_shot | 42.86% | [40.41%, 45.00%] | 1.18% | <1e-6 |
| adapter | 21.67% | [20.20%, 23.11%] | 0.73% | <1e-6 |
| head | 21.18% | [19.50%, 23.14%] | 0.94% | <1e-6 |
| dataset:backbone | 3.81% | [3.10%, 4.55%] | 0.37% | <1e-6 |
| dataset | 1.63% | [1.14%, 2.16%] | 0.26% | 2e-05 |
| backbone:head | 0.72% | [0.42%, 1.08%] | 0.17% | 0.0035 |
| backbone | 0.62% | [0.34%, 0.99%] | 0.16% | 0.0062 |
| adapter:head | 0.23% | [0.07%, 0.46%] | 0.10% | 0.095 |
| k_shot:head | 0.16% | [0.04%, 0.35%] | 0.08% | 0.16 |
| dataset:adapter | 0.14% | [0.03%, 0.33%] | 0.08% | 0.18 |
| k_shot:adapter | 0.08% | [0.01%, 0.23%] | 0.06% | 0.33 |
| dataset:k_shot | 0.02% | [0.00%, 0.11%] | 0.03% | 0.63 |
| k_shot:backbone | 0.02% | [0.00%, 0.11%] | 0.03% | 0.64 |
| dataset:head | 0.01% | [0.00%, 0.09%] | 0.03% | 0.71 |
| **backbone:adapter** | 0.00% | [0.00%, 0.06%] | 0.02% | 0.94 |
| residual | 6.23% | [5.06%, 7.19%] | 0.55% |  |

Type-II shares on unbalanced rows do not sum to 100% (here 99.37%).

## RQ1: sensitivity (the 20-episode smoke observation replaced by the same model's 600-episode scores)

Factors dataset, k_shot, backbone, adapter, head; baselines excluded (full_ft, linear_probe). Seed bootstrap: 2000 resamples, rng seed 20260914.
Observations replaced: ['cifar_fs|5|mobilenetv3_small|lora|evidential|42']

### accuracy: `accuracy_mean`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Main effects vs the published table (expected to move): max abs diff 0.26 pp. Residual df with two-way terms: 80.

Residual **9.04%** with main effects only → **1.28%** after adding the ten two-way terms (which together take 7.76%).

`backbone:adapter` = **0.77%**, which is 8.50% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| k_shot | 76.31% | [75.19%, 77.57%] | 0.60% | <1e-6 |
| adapter | 8.98% | [8.35%, 9.60%] | 0.32% | <1e-6 |
| dataset:backbone | 5.37% | [4.88%, 5.82%] | 0.24% | <1e-6 |
| backbone | 3.82% | [3.38%, 4.34%] | 0.24% | <1e-6 |
| dataset | 1.69% | [1.42%, 1.95%] | 0.14% | <1e-6 |
| k_shot:backbone | 1.02% | [0.80%, 1.26%] | 0.12% | <1e-6 |
| **backbone:adapter** | 0.77% | [0.58%, 0.98%] | 0.10% | <1e-6 |
| adapter:head | 0.31% | [0.19%, 0.43%] | 0.06% | 3.8e-05 |
| head | 0.16% | [0.09%, 0.27%] | 0.05% | 0.0019 |
| dataset:k_shot | 0.13% | [0.06%, 0.22%] | 0.04% | 0.0057 |
| backbone:head | 0.10% | [0.04%, 0.18%] | 0.04% | 0.017 |
| dataset:adapter | 0.02% | [0.00%, 0.06%] | 0.02% | 0.26 |
| dataset:head | 0.02% | [0.00%, 0.06%] | 0.02% | 0.27 |
| k_shot:head | 0.02% | [0.00%, 0.06%] | 0.02% | 0.27 |
| k_shot:adapter | 0.01% | [0.00%, 0.05%] | 0.01% | 0.39 |
| residual | 1.28% | [1.04%, 1.43%] | 0.10% |  |

### ece: `ece_pooled`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Main effects vs the published table (expected to move): max abs diff 0.08 pp. Residual df with two-way terms: 80.

Residual **7.68%** with main effects only → **2.03%** after adding the ten two-way terms (which together take 5.64%).

`backbone:adapter` = **3.23%**, which is 42.06% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| head | 82.97% | [82.05%, 83.95%] | 0.49% | <1e-6 |
| backbone | 4.56% | [3.72%, 5.46%] | 0.45% | <1e-6 |
| **backbone:adapter** | 3.23% | [2.49%, 4.05%] | 0.41% | <1e-6 |
| k_shot | 2.16% | [1.63%, 2.82%] | 0.31% | <1e-6 |
| dataset | 1.98% | [1.39%, 2.63%] | 0.32% | <1e-6 |
| k_shot:backbone | 0.91% | [0.52%, 1.40%] | 0.23% | <1e-6 |
| adapter | 0.65% | [0.34%, 1.05%] | 0.18% | 2.6e-06 |
| dataset:backbone | 0.62% | [0.33%, 1.02%] | 0.18% | 4e-06 |
| dataset:adapter | 0.32% | [0.11%, 0.61%] | 0.13% | 0.00063 |
| k_shot:adapter | 0.21% | [0.06%, 0.47%] | 0.11% | 0.0049 |
| k_shot:head | 0.21% | [0.05%, 0.45%] | 0.10% | 0.0057 |
| dataset:head | 0.10% | [0.01%, 0.29%] | 0.07% | 0.054 |
| backbone:head | 0.02% | [0.00%, 0.12%] | 0.03% | 0.41 |
| adapter:head | 0.01% | [0.00%, 0.12%] | 0.03% | 0.45 |
| dataset:k_shot | 0.01% | [0.00%, 0.10%] | 0.03% | 0.53 |
| residual | 2.03% | [1.27%, 2.39%] | 0.28% |  |

### far_ood_svhn: `ood_auroc__svhn_far__{native}`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Main effects vs the published table (expected to move): max abs diff 0.17 pp. Residual df with two-way terms: 80.

Residual **23.75%** with main effects only → **9.53%** after adding the ten two-way terms (which together take 14.22%).

`backbone:adapter` = **0.20%**, which is 0.83% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| head | 38.93% | [34.51%, 43.77%] | 2.33% | <1e-6 |
| k_shot | 22.59% | [19.16%, 26.35%] | 1.84% | <1e-6 |
| backbone | 11.71% | [9.09%, 14.96%] | 1.50% | <1e-6 |
| dataset:backbone | 9.11% | [6.61%, 11.70%] | 1.29% | <1e-6 |
| dataset:head | 2.13% | [1.05%, 3.55%] | 0.65% | 6.1e-05 |
| adapter | 1.79% | [0.82%, 2.92%] | 0.55% | 0.00022 |
| backbone:head | 1.31% | [0.53%, 2.44%] | 0.50% | 0.0014 |
| dataset | 1.23% | [0.47%, 2.33%] | 0.48% | 0.0019 |
| k_shot:head | 0.66% | [0.16%, 1.45%] | 0.34% | 0.021 |
| adapter:head | 0.37% | [0.03%, 1.00%] | 0.26% | 0.083 |
| **backbone:adapter** | 0.20% | [0.00%, 0.75%] | 0.21% | 0.2 |
| k_shot:backbone | 0.19% | [0.00%, 0.77%] | 0.21% | 0.21 |
| dataset:k_shot | 0.11% | [0.00%, 0.61%] | 0.16% | 0.33 |
| dataset:adapter | 0.11% | [0.00%, 0.55%] | 0.16% | 0.33 |
| k_shot:adapter | 0.02% | [0.00%, 0.33%] | 0.09% | 0.65 |
| residual | 9.53% | [7.06%, 10.52%] | 0.88% |  |

### near_ood_tin: `ood_auroc__tin_near__{native}`, n=96, balanced=True

Method: eta_squared (exact: balanced design). Main effects vs the published table (expected to move): max abs diff 0.70 pp. Residual df with two-way terms: 80.

Residual **11.49%** with main effects only → **6.21%** after adding the ten two-way terms (which together take 5.28%).

`backbone:adapter` = **0.00%**, which is 0.00% of the main-effects residual.

| term | share | 95% seed bootstrap | bootstrap SD | F-test p |
|---|---:|---:|---:|---:|
| k_shot | 43.30% | [40.94%, 45.56%] | 1.21% | <1e-6 |
| adapter | 21.68% | [20.26%, 23.07%] | 0.73% | <1e-6 |
| head | 21.44% | [19.73%, 23.42%] | 0.91% | <1e-6 |
| dataset:backbone | 3.90% | [3.20%, 4.71%] | 0.38% | <1e-6 |
| dataset | 1.54% | [1.07%, 2.06%] | 0.25% | 2.7e-05 |
| backbone:head | 0.75% | [0.45%, 1.11%] | 0.17% | 0.0026 |
| backbone | 0.55% | [0.29%, 0.89%] | 0.16% | 0.0095 |
| adapter:head | 0.21% | [0.07%, 0.42%] | 0.09% | 0.1 |
| k_shot:head | 0.14% | [0.03%, 0.35%] | 0.08% | 0.18 |
| dataset:adapter | 0.13% | [0.03%, 0.33%] | 0.08% | 0.2 |
| k_shot:adapter | 0.09% | [0.01%, 0.24%] | 0.06% | 0.29 |
| dataset:k_shot | 0.02% | [0.00%, 0.12%] | 0.03% | 0.58 |
| k_shot:backbone | 0.02% | [0.00%, 0.12%] | 0.03% | 0.59 |
| dataset:head | 0.01% | [0.00%, 0.11%] | 0.03% | 0.67 |
| **backbone:adapter** | 0.00% | [0.00%, 0.05%] | 0.01% | 0.98 |
| residual | 6.21% | [5.02%, 7.14%] | 0.55% |  |
