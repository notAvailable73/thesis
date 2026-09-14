# 9. Methodology and Experimental Setup (thesis Chapters 3–4 draft)

Source material for **Chapter 3 (Methodology)** and **Chapter 4 (Experimental Setup)**. Every equation,
hyperparameter and procedure here was checked against the code and the resolved grid configs on
2026-09-14. The file and line to cite is given wherever a reader might want to check.

**Rule for writing from this file:** if the thesis text and this file disagree, check the code, not memory.
Section 9.12 lists every place where what ran differs from the proposal or from a cited paper.

Notation: K = number of classes per episode (= 5); N_S = shots per class (1 or 5); N_Q = queries per class
(= 15).

---

## 9.1 Overview

```
 support images (K·N_S)        query / OOD images
          │                            │
          ▼                            ▼
 ┌───────────────────────────────────────────────┐
 │ frozen ImageNet-pretrained backbone f_θ       │  ResNet-18 (512-d) | MobileNetV3-Small (576-d)
 │   + trainable adapter (inside the backbone)   │  bottleneck-parallel | LoRA | (Full FT | none)
 └───────────────────────────────────────────────┘
          │ pooled features                    │
          ▼                                    ▼
   class prototypes c_k  ───────────►  cosine-similarity logits z ∈ ℝ^K   (parameter-free head)
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     ▼                                                   ▼
            SOFTMAX readout                                   EVIDENTIAL readout
         p = softmax(z)                          e = softplus(a·z + b), α = e + 1, S = Σα
         scores: MSP, TS-MSP, energy             p = α/S, vacuity u = K/S
```

*Thesis figure to draw from this (listed as missing in `docs/DEFENCE_SLIDE_PLAN.md`, slide 15).*

---

## 9.2 Problem formulation

The 100 classes of each dataset are split into disjoint sets: base C_train (64), validation C_val (16) and
novel C_test (20). An **episode** 𝒯 = (𝒮, 𝒬) draws K classes from one split, then:

- a support set 𝒮 = {(x_i, y_i)} with N_S images per class;
- a query set 𝒬 with N_Q = 15 further images per class (75 queries).

Labels are re-indexed to {0,…,K−1} within each episode.

**Goal.** Learn trainable parameters φ (adapter, plus two evidence scalars for evidential runs) on episodes
from C_train, such that on episodes from the unseen C_test the model:

1. classifies queries accurately;
2. gives confidences that match its accuracy (calibration);
3. assigns lower "in-distribution-ness" to images from outside the episode's classes than to its queries
   (OOD detection).

The backbone weights θ stay fixed, except in the Full FT baseline.

---

## 9.3 Model

### 9.3.1 Backbones

| Backbone | Source | Feature | Frozen parameters | Notes |
|---|---|---|---|---|
| ResNet-18 [he2016resnet] | torchvision ImageNet weights | 512-d global-average-pooled | 11,176,512 (fc removed) | 4 stages, 64/128/256/512 channels |
| MobileNetV3-Small [howard2019mobilenetv3] | torchvision ImageNet weights | **576-d pooled conv feature** (classifier removed) | 927,008 trunk | The 1024-d post-pool expansion layer `classifier[0]` is *not* used (decision log 2026-07-26) |

**BatchNorm is always in `eval()` mode**, so running statistics stay at their ImageNet values for every
method, Full FT included (`src/trainers/episodic_trainer.py:217`). Reason: estimating BN statistics from
5–25-image support batches is unstable and would leak query statistics.

### 9.3.2 Adapters

**Bottleneck-parallel** (`src/adapters/placement.py`; Houlsby-style bottleneck [houlsby2019adapters], parallel
insertion per He et al. [he2022unified] and Conv-Adapter [chen2024convadapter]).

At the final residual block B_s of each stage s, with block input x_s and output B_s(x_s), the output is
replaced by

$$ y_s = B_s(x_s) + W^{up}_s\,\mathrm{ReLU}(W^{down}_s\, x_s), $$

where:

- W^down_s is a 1×1 convolution C_s → r and W^up_s is a 1×1 convolution r → C_s, both with bias;
- W^up is **zero-initialised**, so the adapted network equals the frozen one at initialisation.

Insertion sites:

- ResNet-18: `layer1.1 … layer4.1`, C = (64, 128, 256, 512);
- MobileNetV3-Small: the stage-final residual inverted-residual blocks `features.3/6/8/11`, C = (24, 40, 48, 96).

Only residual blocks qualify, because the input and output shapes must match. MobileNetV3's first stage is
stride-2 and has no eligible site.

$$ |\phi_{btl}(r)| = \sum_s (2 C_s r + r + C_s) = \begin{cases} 1924\,r + 960 & \text{ResNet-18} \\ 420\,r + 208 & \text{MobileNetV3-Small} \end{cases} $$

**LoRA** (`src/adapters/lora.py`) [hu2022lora], in the conv form of ConvLoRA [aleem2024convlora]. For a frozen
1×1 convolution W₀ ∈ ℝ^{C_out×C_in}:

$$ W = W_0 + \tfrac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times C_{in}},\; B \in \mathbb{R}^{C_{out} \times r}, $$

with α = r (scaling 1), A Kaiming-uniform and **B = 0** at initialisation. There is one target per backbone:

- ResNet-18: `layer4.0.downsample.0` (256 → 512);
- MobileNetV3-Small: `features.11.block.3.0`, the last block's 1×1 projection (576 → 96).

$$ |\phi_{LoRA}(r)| = r\,(C_{in} + C_{out}) = \begin{cases} 768\,r & \text{ResNet-18} \\ 672\,r & \text{MobileNetV3-Small} \end{cases} $$

*(The early spec's "2r(in+out)" was wrong by a factor of 2; tests assert the formula above.)*

**Baselines** (ResNet-18 + CIFAR-FS only):

- **Full FT** — all 11,176,512 backbone weights train. LR 1e-5, weight decay 1e-4, collapse guard disabled.
- **Linear Probe** — frozen backbone, identity adapter. 0 trainable parameters with softmax; only the 2
  evidence scalars with evidential. The 0-parameter case skips optimisation entirely.

**Grid parameter counts at r = 16** (softmax; evidential +2):

| | ResNet-18 | MobileNetV3-Small |
|---|---:|---:|
| Bottleneck-par. | 31,744 | 6,928 |
| LoRA | 12,288 | 10,752 |

The ordering reverses between backbones. RQ3 is built on this (§9.9.4).

**Used only in earlier steps:** serial bottleneck, y_s = B_s(x_s) + W^up ReLU(W^down B_s(x_s)) (Step 6);
post-pool bottleneck on the pooled vector (Steps 4–9); BitFit, biases only, 4,800 parameters (Step 5).

### 9.3.3 Prototype head

The head is parameter-free (`src/heads/prototype_head.py`) [snell2017protonet]. With g(·) the adapted backbone
feature:

$$ c_k = \frac{1}{N_S}\sum_{(x_i,y_i)\in\mathcal{S},\,y_i=k} g(x_i), \qquad z_k(x) = \tau\,\frac{g(x)^\top c_k}{\lVert g(x)\rVert\,\lVert c_k\rVert}, \quad \tau = 10. $$

**Why cosine, and why this needs a sentence in the thesis.** The ProtoNet summary [snell2017protonet] **[S]**
reports squared Euclidean distance clearly outperforming cosine, because it is a Bregman divergence. This
thesis uses cosine anyway (Step 4 decision, `step_writeups/step4.txt` §fix-2):

1. **Bounded logits.** Cosine logits lie in [−τ, τ]. Raw L2 logits are large and negative for ResNet-18
   features, which drove softplus(z) ≈ 0 everywhere and caused the Step 4 evidential collapse.
2. **A controlled comparison.** The same metric is used for both readouts, so the softmax-vs-evidential
   comparison is not confounded with L2-vs-cosine.
3. **Precedent.** Cosine prototype classifiers are a standard few-shot baseline (Baseline++
   [chen2019closerlook] **[S]**).

*Flag for the thesis:* no L2 run was repeated under the final recipe, so the cost of this choice relative to
ProtoNet's recommended metric is unmeasured.

**Why a prototype head rather than the proposal's linear head.** A linear classifier trained on the 64 base
classes cannot score the 20 disjoint test classes. The prototype head is the standard CIFAR-FS protocol
(decision log 2026-05-19).

### 9.3.4 Two readouts of the same logits

**Softmax:** p = softmax(z); confidence = max_k p_k.

**Evidential** [sensoy2018edl], through `PrototypeHead.to_evidence()`, the only implementation used by
training, evaluation and every analysis script:

$$ e_k = \mathrm{softplus}(a\, z_k + b), \quad a = \mathrm{softplus}(\tilde a) > 0, \quad \alpha_k = e_k + 1, \quad S = \sum_k \alpha_k, \quad p_k = \frac{\alpha_k}{S}, \quad u = \frac{K}{S}. $$

**(a, b) are learnable.** `evidence_affine: true` in every evidential grid config; they are initialised at
(a, b) = (2, −6) and optimised jointly with the adapter. Over the 48 recovered evidential checkpoints the
trained values are a ∈ [1.51, 4.52] (median 2.93) and b ∈ [−9.27, −5.58] (median −6.30)
(`results/rq_summary.json` → `rq2_rows[*].affine_trained`). The positivity reparameterisation of a keeps
the map monotone in each logit.

**Why softplus and not the proposal's ReLU:** ReLU gives zero gradient for negative logits, so dead classes
cannot recover (Step 1; the EDL survey also lists softplus as a standard evidence function [gao2024edlsurvey]
**[S]**). **Why the affine:** it moves cosine logits into the region where softplus is both informative and
differentiable (the Step 4 collapse fix).

---

## 9.4 Training objectives

**Softmax:** cross-entropy on the 75 query logits of each episode, L_CE = −(1/|𝒬|) Σ log p_{y}(x).

**Evidential** (`src/losses/evidential.py`): the Sensoy squared-error loss with the **variance term
removed** (R-EDL relaxation [chen2024reedl, gao2024edlsurvey]; `use_variance: false`), plus the annealed KL
term:

$$ \mathcal{L}_{EDL} = \frac{1}{|\mathcal{Q}|}\sum_{x\in\mathcal{Q}}\Big[\sum_k (y_k - p_k)^2 \;+\; \lambda_t\,\mathrm{KL}\big(\mathrm{Dir}(\tilde\alpha)\,\Vert\,\mathrm{Dir}(\mathbf 1)\big)\Big], \qquad \tilde\alpha = y + (1-y)\odot\alpha, $$

$$ \mathrm{KL} = \log\Gamma\!\big(\textstyle\sum_k\tilde\alpha_k\big) - \log\Gamma(K) - \sum_k \log\Gamma(\tilde\alpha_k) + \sum_k(\tilde\alpha_k - 1)\big(\psi(\tilde\alpha_k) - \psi(\textstyle\sum_j\tilde\alpha_j)\big), $$

$$ \lambda_t = \lambda_{max}\cdot\min(1,\; t / T_{KL}), \qquad \lambda_{max} = 0.1,\; T_{KL} = 1000 \text{ training episodes}. $$

The KL term penalises evidence on wrong classes only, because ᾱ strips the true class. Differences from
Sensoy: softplus instead of ReLU; annealing per episode over 1,000 episodes instead of min(1, t/10) per
epoch; variance term dropped; λ_max = 0.1 instead of 1.

---

## 9.5 Episodic meta-training

**Algorithm (per run):**

```
init adapter (zero-init up/B), evidence affine (2, −6) if evidential
for epoch = 1..30:
    for i = 1..100:                                   # training episodes from C_train
        episode seed = 20000 + (epoch−1)·100 + i      # independent of the training seed
        z = head(g(support), labels, g(query))
        loss = L_CE or L_EDL(λ_t);  Adam step;  t += 1
    val_acc = mean accuracy on the 100 fixed VAL episodes (seeds 10000–10099, C_val)
    keep the parameters with the best val_acc so far
    if epoch == 1 and val_acc ≤ 0.25: abort (collapse guard; disabled for Full FT)
    if no improvement for 5 epochs: stop
return best-val-accuracy parameters
```

**Hyperparameters (identical across all 120 grid runs unless noted):**

| Setting | Value | Source |
|---|---|---|
| Optimiser | Adam | `scripts/train.py:340` |
| Learning rate / weight decay | 5e-3 / 0 (Full FT: 1e-5 / 1e-4) | resolved grid configs |
| Adapter rank | 16 (RQ3 matched arms and rank sweep vary it) | configs |
| Episodes per epoch / max epochs / patience | 100 / 30 / 5 | `configs/base.yaml` |
| Model selection | best mean VAL-episode accuracy | `episodic_trainer.py` |
| KL max / anneal length | 0.1 / 1,000 episodes | `exp_phase2_evidential_retuned.yaml` |
| Dirichlet prior per class / variance term | 1.0 / off | same |
| Evidence affine init | (2, −6), learnable | same |
| Cosine scale τ | 10 | `exp_phase2_*.yaml` |
| Training seeds | 42, 43, 44 | grid index |
| Image size / augmentation | 224×224 / **none** | dataset loaders |

**What a training seed changes.** Only the adapter's random initialisation, via `set_seed`. Episode sampling
uses fixed seed offsets, so all three seeds see identical episodes. Full FT and Linear Probe have no random
initialisation, so their three seeds are identical runs.

**Where the recipe came from.** It was set on ResNet-18 + CIFAR-FS + 5-shot in Step 4.5 and never re-tuned
per cell. The Step 4.5 VAL-only sweep over kl_weight_max ∈ {0.05, 0.1, 0.25} × use_variance ranked 0.05 first
by VAL ECE (0.252 vs 0.260 for 0.1, 0.296 for 0.25; `step_writeups/step4_5.txt` §4). The config file the grid
inherits kept 0.1, and no decision-log entry explains why. The two values differ by 0.008 VAL ECE on a
surface the same sweep called flat. **Disclose this; do not describe 0.1 as "the VAL-selected value".**

---

## 9.6 Post-hoc components (fitted on VAL episodes only)

Both are fitted on the pooled query logits of the 100 VAL episodes (seeds 10000–10099, classes C_val). Test
seeds are never used.

**Temperature scaling** (`src/evaluators/temperature.py`) [guo2017calibration]: T = exp(τ̃) minimises NLL of
softmax(z/T), with Adam (lr 0.01, 200 iterations on τ̃ from 0). This gives TS-MSP and TS-ECE.

**Evidence-affine refit, RQ4** (`scripts/rq_core.py:fit_evidence_affine`): (a, b) minimise

$$ -\frac{1}{n}\sum_i \log p_{y_i}(x_i;\,a,b), \qquad p = \frac{\mathrm{softplus}(a z + b) + 1}{\sum_k(\cdot)}, $$

with Adam (lr 0.05, 500 iterations), starting from the checkpoint's trained (a, b). The optimisation runs on
a throwaway head, so it still goes through `to_evidence()`. The procedure is deterministic.

---

## 9.7 OOD scoring

Every score is oriented so that **higher means more in-distribution**:

| Score | Definition | Computed on |
|---|---|---|
| MSP [hendrycks2017baseline] | max_k softmax(z)_k | softmax runs (grid); all runs (RQ2) |
| TS-MSP | max_k softmax(z/T)_k | same |
| Energy [liu2020energy] | log Σ_k exp(z_k), i.e. −E(x) at T = 1, on **raw prototype logits** | same; on evidential models the raw logits are used too, so the score is the same function in both arms |
| Vacuity [sensoy2018edl] | 1 − K/S | evidential runs (grid, native affine); RQ2 uses the **VAL-refit affine in both arms** so softmax-trained models are not handed an untuned mapping |

**Protocol per test episode** (`src/evaluators/episodic.py`):

1. Compute the prototypes from this episode's support set.
2. The ID set is the 75 queries.
3. Each OOD pool (500 images) is scored against the **same prototypes**.
4. AUROC and FPR@95 are computed per episode, then averaged over the 600 episodes.

Because every OOD image is scored by the same 5-way head as the queries, K_ID = K_OOD = 5 always. The
vacuity cardinality artefact reported by [mcnamara2026vacuity] cannot arise.

---

## 9.8 Evaluation metrics

| Metric | Definition as implemented | Aggregation |
|---|---|---|
| Accuracy | fraction of correct query argmax | mean over 600 episodes; CI95 = 1.96·SD/√600 |
| Macro-F1 | unweighted mean of per-class F1 within an episode | mean over episodes |
| **ECE (pooled)** — headline | 15 equal-width bins over max-probability confidence, Σ_m (\|B_m\|/n)\|acc(B_m) − conf(B_m)\| [guo2017calibration], over all 45,000 test queries | one value per run |
| ECE (per-episode) | same, per episode | mean over episodes (stored, not headline) |
| Brier | mean over queries of Σ_k (p_k − y_k)² | mean over episodes |
| AUROC | ROC area, ID = positive class | per episode, then mean |
| FPR@95 | FPR on OOD at the first threshold with TPR ≥ 0.95 | per episode, then mean |
| Trainable params | Σ numel of `requires_grad` parameters | exact |
| FLOPs | fvcore `FlopCountAnalysis` total (= MACs) at 224×224 | deterministic |
| Latency | median of timed repetitions after warm-up (e.g. 5 warm-up + 20 timed for a full episode); single-thread CPU (Kaggle Xeon @ 2.00 GHz) as edge proxy, plus T4 GPU; session drift check ≤ 5% | not byte-reproducible |

**Seed aggregation.** Each run yields one value per metric; tables report the mean and SD across the 3 seeds
(`results/mvt_results.json`).

---

## 9.9 Analysis design for each research question

### 9.9.1 The main grid (Step 10), the data for RQ1 and RQ3's first evidence

2 datasets × 2 shots × 2 backbones × 2 adapters (bottleneck-par., LoRA) × 2 readouts = **32 balanced cells**,
plus 8 baseline cells (Full FT, Linear Probe × 2 readouts × 2 shots, ResNet-18 + CIFAR-FS). 40 cells × 3
seeds = **120 runs**. Configs are generated by `scripts/build_grid_configs.py`.

### 9.9.2 RQ1: variance attribution

On the 96 per-seed observations of the 32 balanced cells, for each outcome y (accuracy, pooled ECE, SVHN-far
AUROC, TinyImageNet-near AUROC) and each factor f with levels l:

$$ \eta^2_f = \frac{SS_f}{SS_{total}} = \frac{\sum_l n_l(\bar y_l - \bar y)^2}{\sum_i (y_i - \bar y)^2}, \qquad \text{residual} = 1 - \sum_f \eta^2_f . $$

The analysis uses **main effects only**, so the residual absorbs all interactions. Per-seed rather than
cell-mean observations keep a real error term; the recomputation moved the residual by only 0.3–3.6 points.
Supporting analysis: Spearman ρ between ECE and AUROC across the 40 cells, then again within each readout.

### 9.9.3 RQ2: training objective × scoring rule

- **Data:** 99 of 120 grid checkpoints recovered; all 21 missing are CIFAR-FS 5-shot adapter models.
- **Rescoring:** every checkpoint is re-scored with all four scores, giving a 2 (objective) × 4 (score)
  factorial.
- **Observations:** one AUROC per (design, objective, score, pool, seed). Pools are grouped into far (SVHN,
  Gaussian) and near (held-out classes, TinyImageNet).
- **Variance model:** η² for design, objective and score with pairwise interactions, computed separately for
  the far and near groups (`scripts/rq_aggregate.py:eta_squared`).
- **Regression guard:** the native scores of all 99 re-scored checkpoints must equal the committed Step 10
  metrics exactly. They do.

⚠️ The η² code marks the design "balanced" even though one design has only one objective
(`05_problems_and_open_work.md` B1). Report η² shares, not their ratio.

### 9.9.4 RQ3: matched-budget experiment (pre-registered)

**Design** (`docs/RQ3_MATCHED_BUDGET_PLAN.md` §4, fixed before any deciding run): MiniImageNet, 5-shot, 2
backbones × 2 budget levels × 2 architectures × 2 readouts (trained separately) × 3 seeds = **48 runs**, all
trained fresh.

| Backbone | Level | Bottleneck rank → params | LoRA rank → params | Mismatch |
|---|---|---|---|---:|
| ResNet-18 | L | 6 → 12,504 | 16 → 12,288 | 1.76% |
| ResNet-18 | H | 16 → 31,744 | 41 → 31,488 | 0.81% |
| MobileNetV3-S | L | 16 → 6,928 | 10 → 6,720 | 3.10% |
| MobileNetV3-S | H | 22 → 9,448 | 14 → 9,408 | 0.43% |

**Primary outcome.** ΔECE = mean_seeds ECE(LoRA) − mean_seeds ECE(bottleneck), averaged over the two levels
for each (backbone, readout) cell. The noise scale is σ = √((SD²_btl + SD²_LoRA)/2), the pooled across-seed SD.

**Decision rule** (verbatim thresholds; a hypothesis is supported if its condition holds in ≥ 3 of 4 cells):

| Hypothesis | Condition per cell |
|---|---|
| H3.2 — budget | \|ΔECE_matched\| ≤ 50% of \|ΔECE_unmatched\| **and** \|ΔECE_matched\| ≤ 2σ |
| H3.1 — architecture | ΔECE_matched has the **same sign on both backbones** and \|ΔECE\| > 2σ |
| H3.2-alt — backbone-intrinsic | ΔECE_matched keeps its **unmatched per-backbone sign** (+ ResNet-18, − MobileNetV3) and \|ΔECE\| > 2σ |
| none fires | record *inconclusive* |

**Secondary outcomes.** ΔAccuracy and Δnear-OOD AUROC at matched budget (the architecture half of RQ3).

**Seven acceptance guards:**

1. all runs OK;
2. exact parameter counts;
3. mismatch ≤ 3.1%;
4. only permitted keys differ between merged configs (learning rate, epochs, patience, KL schedule, dataset,
   episode files, n-way and k-shot are identical everywhere);
5. the 18 runs repeating grid cells reproduce the grid exactly;
6. the verdict follows the rule;
7. the secondary outcome is reported.

### 9.9.5 RQ4: post-hoc evidence-affine refit

For each of the 48 recovered evidential checkpoints, refit (a, b) on VAL (§9.6) and re-evaluate on the 600
test episodes.

- **H4.1:** ECE_after < ECE_before.
- **H4.2:** ΔAUROC ≥ −0.005 per (checkpoint, pool), giving 48 × 4 = 192 comparisons.

Ranking change is also measured as Spearman ρ between pre- and post-refit vacuity orderings. Both arms are
compared against the same model's softmax ECE and TS-ECE.

### 9.9.6 Former RQ5: rank sweep

CIFAR-FS, ResNet-18, bottleneck-parallel, evidential training (softmax and TS ECE read from the same logits),
rank ∈ {1, 2, 4, 8, 16, 32, 64} × 3 seeds = 21 runs (2,886 to 124,098 parameters). The question is whether
ECE reaches its minimum at an intermediate rank, and whether that rank differs from the accuracy-optimal one.

### 9.9.7 Efficiency (Orig-RQ4)

The 12 (backbone, adapter, readout) combinations were measured for parameters, FLOPs, GPU/CPU latency and
peak memory, with no training (`scripts/efficiency_table.py`). Pareto frontiers of latency against accuracy,
ECE and AUROC are built by `scripts/pareto_plots.py`.

---

## 9.10 Datasets and OOD pools (Experimental Setup)

### 9.10.1 In-distribution data

| Dataset | Origin | Classes (train/val/test) | Native size | Split file |
|---|---|---|---|---|
| CIFAR-FS | CIFAR-100 [krizhevsky2009cifar], split of [bertinetto2019r2d2] | 64 / 16 / 20 | 32×32 | `data/cifar_fs_split.json` (frozen) |
| MiniImageNet | ImageNet subset [vinyals2016matching], split of [ravi2017optimization]; images from the learn2learn Zenodo record 7978538 pickles | 64 / 16 / 20 | 84×84 | `data/mini_imagenet_split.json` (frozen) |

### 9.10.2 OOD pools (500 images each, sampled with seed 42)

| Pool | Type | Content | Detail |
|---|---|---|---|
| `svhn_far` | far | SVHN test split [netzer2011svhn] | house-number photos |
| `gaussian_far` | far | N(0, 1) noise in normalised input space, clamped to ±3 | sanity pool; added in Step 7 |
| `cifar100_near` / `mini_near` | near | images from the dataset's **16 validation classes** | disjoint from the 20 test classes; same visual domain (semantic shift only, as in OpenOOD) |
| `tin_near` | near | TinyImageNet train images [le2015tinyimagenet] | for MiniImageNet runs the 25 classes TinyImageNet shares with MiniImageNet are excluded |

⚠️ **Disclose in the thesis:** the held-out-class near-OOD pools use the same 16 classes as the VAL
episodes. Those episodes drive early stopping, temperature fitting and the RQ4 refit. No test class is
involved, but model selection and post-hoc fitting have seen those classes as *in-distribution* classes.
The TinyImageNet pool has no such overlap. It is reported alongside in the grid tables, and it is RQ1's
near-OOD outcome.

### 9.10.3 Preprocessing

Every image (ID and OOD) is resized to **224×224**, converted to a tensor and normalised with ImageNet
mean/std. There is **no data augmentation** at training or test time. *A Closer Look*
[chen2019closerlook] **[S]** warns that omitting augmentation underestimates baselines; this applies equally
to every cell of the grid, but should be stated.

---

## 9.11 Implementation, hardware and reproducibility

- **Software.** PyTorch + torchvision. Configs are YAML with `extends:` inheritance
  (`src/utils/config.py`), and components are chosen by factories in each subpackage's `__init__.py`. fvcore
  counts FLOPs. W&B was disabled for all reported runs.
- **Hardware.**
  - Step 10 grid, RQ3 and Phase A/B: Kaggle **NVIDIA Tesla T4** sessions (grid 36.3 GPU-h; RQ3 6.2 h;
    Phase A 5.51 GPU-h).
  - Earlier steps: Google Colab (Steps 1–5) and Kaggle T4 (Steps 6–9).
  - Latency: single-thread CPU on a Kaggle Xeon host, plus the T4. No Jetson Nano was available.
- **Determinism.** `set_seed` seeds Python, NumPy and torch (CPU and CUDA), and sets
  `cudnn.deterministic = True`. Metrics are written with `json.dump(sort_keys=True)`. Rerunning a config
  produces a **byte-identical** metrics file; this was checked in Step 3 and, indirectly, by the 99/99
  (Phase A) and 18/18 (RQ3) exact reproductions.
- **Frozen evaluation data.** `configs/test_episodes.yaml` holds 600 TEST seeds (0–599);
  `configs/val_episodes.yaml` holds 100 VAL seeds (10000–10099). Both are marked *DO NOT REGENERATE*.
  **Tuning and post-hoc fitting use VAL seeds only.** This is enforced by convention and checked in Phase A,
  not enforced in code.
- **Reproducibility statement for the thesis:**

  > All configurations, episode seeds and class splits are version-controlled
  > (`configs/test_episodes.yaml`, `configs/val_episodes.yaml`, `data/*_split.json`); every reported number
  > is transcribed from a committed result file (`results/mvt_results.json`, `results/rq_summary.json`,
  > `results/rq3_matched/verdict.json`), and re-running any configuration reproduces its metrics file
  > byte-for-byte.

---

## 9.12 Deviations and disclosed choices

Every point where what ran differs from the proposal or from the paper it cites. Chapter 3 should contain a
table like this one.

| # | Proposal / paper says | What ran | Why | Consequence to state |
|---|---|---|---|---|
| 1 | Linear classifier head (proposal §5) | Parameter-free prototype head | A linear head on 64 base classes cannot score 20 disjoint test classes | Standard CIFAR-FS protocol; head adds no parameters |
| 2 | Evidence = ReLU(f) (proposal §5B, Sensoy) | softplus(a·z + b) | ReLU dead units (Step 1); softplus(z) collapse on prototype logits (Step 4) | Adds 2 learnable parameters |
| 3 | — | Evidence affine **learnable**, init (2, −6) | Operating point must adapt across datasets and backbones (Step 4) | RQ4 refits a *trained* affine, not a default |
| 4 | Squared Euclidean distance (ProtoNet summary) | Cosine × 10 | Bounded logits for softplus; same metric for both readouts | Cost vs L2 not measured under the final recipe |
| 5 | Sensoy loss with variance term; KL weight → 1 per epoch | Variance term dropped; λ_max 0.1; linear over 1,000 episodes | R-EDL relaxation; the Step 4 KL 0.5 over-regularised | Loss is squared error + KL |
| 6 | VAL sweep winner kl_weight_max = 0.05 | Grid used 0.1 | Not recorded | Within a flat ECE surface; disclose |
| 7 | Vanilla Full FT updates BN statistics | BN in eval mode for all methods | Unstable with 25-image batches; query leakage | Full FT is "all weights, frozen BN stats" |
| 8 | LoRA on 1×1 and 3×3 kernels (proposal) | One 1×1 conv per backbone | A 3×3 target exceeds 160k parameters; keeps LoRA budget comparable | LoRA touches one layer, the bottleneck touches four stages |
| 9 | Depthwise-separable adapters preferred (Conv-Adapter) | 1×1 bottleneck | Keeps the adapter form identical across placements | Possible loss-of-locality penalty, applied to every bottleneck cell |
| 10 | Augmentation is critical (A Closer Look) | None | Keeps runs deterministic and comparable | Absolute accuracies may be lower than achievable |
| 11 | Near-OOD from held-out classes | 16 VAL classes | Zero-download, disjoint from test | Same classes as VAL episodes (§9.10.2) |
| 12 | ResNet-18, MobileNetV3-S, **ConvNeXt-Nano** | First two only | Scope and time | Two backbones cannot identify RQ3's mechanism |
| 13 | CIFAR-FS, MiniImageNet, **CUB-200, ISIC**, **CIFAR-10-C** | First two only | Scope and time | No fine-grained, medical or corruption evaluation |
| 14 | Jetson Nano latency | Single-thread CPU proxy | No device | Latency is relative, not absolute |
| 15 | "< 500 trainable parameters" (Orig-RQ2) | 6,928–31,746 (Linear Probe: 0–2) | Real adapters need more | Orig-RQ2 answered at the actual budgets |
| 16 | Serial vs parallel in the grid | Parallel only (serial tested once, Step 6) | Parallel won or tied in Step 6 | Placement evidence is single-seed, single-setting |

---

## 9.13 Literature grounding per component (pros / cons / fit)

A compact form of the reasoning `thesis_implementation_instructions.txt` §3 requires, for the methodology
chapter's justification paragraphs.

| Component | Grounded in | Pros | Cons / conflict | Fit |
|---|---|---|---|---|
| Bottleneck adapter, parallel | Houlsby [S]; He et al. unified view [S]; Conv-Adapter [S] (parallel > sequential) | Near-identity init; placement vocabulary; CNN evidence for parallel | Conv-Adapter prefers depthwise K×K over 1×1 | Good; 1×1 disclosed (#9) |
| LoRA on 1×1 conv | LoRA [S]; ConvLoRA [S] (low-data conv LoRA) | Zero-init; mergeable (no latency); proven in 10-image CNN adaptation | ConvLoRA adapts the whole encoder; here one layer | Adequate for a budget-comparable arm (#8) |
| Prototype head | ProtoNet [S]; Matching Nets [S]; Closer Look [S] | Parameter-free; transfers to novel classes; standard protocol | ProtoNet recommends L2 (#4) | Good |
| Evidential readout + loss | Sensoy EDL [S]; EDL survey / R-EDL [S] | Single-pass uncertainty; OOD-data-free | Idealised Dirichlet shapes not guaranteed (Ulmer survey [S]); critiques [T/R] | Central to the thesis; calibration outcome negative |
| Temperature scaling | Guo et al. [S] | One scalar, accuracy-preserving, strongest simple baseline | Needs a same-distribution VAL set | VAL episodes serve as that set |
| Energy score | Liu et al. [S] | Parameter-free, logsumexp on existing logits | Cosine logits bounded by τ, not the paper's setting | Strongest cheap competitor; decisive for RQ2 |
| Near/far pools, VAL-only tuning | OpenOOD v1/v1.5 [S] | Accepted protocol; leakage-free tuning | Pool size 500 is small; VAL-class overlap (#11) | Good |
| Variance decomposition | One Model, Many Behaviors [R] | Precedent for ANOVA on OOD | Main effects only here | Good |
| Not used: ODIN, Mahalanobis | [S] | Mahalanobis is robust in small data | ODIN costs a backward pass; neither is implemented | Named as future work |
