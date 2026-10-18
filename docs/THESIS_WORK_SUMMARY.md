# B-PEFT — Complete Record of the Thesis Work

**Thesis title:** *Reliability in Few-Shot Classification with Frozen CNN Adapters: A Controlled Study*

**What this document is:** a single, factual account of everything the project actually built, ran, found and
contributed, from the first experiments (April 2026) to the final completion run (15 September 2026). Every number is
taken from the repository's result files and guide (`docs/guide/`, `docs/RQ_SUPERVISOR_REPORT.md`,
`results/rq_completion/REPORT.md`). Numbers marked *current* reflect full 120/120 coverage. Older documents
contain some superseded figures; Section 9 lists them.

**Status:** all experiments are complete. What remains is thesis writing and the defence.

---

## 1. The problem in one paragraph

Lightweight CNNs (ResNet-18, MobileNetV3-Small) are still the practical choice for on-device vision. Adapting
them to new classes from only 1 or 5 labelled images raises two problems. First, retraining millions of
weights on so few images is wasteful. Second, the adapted model must still say when it is unreliable. We
**freeze** an ImageNet-pretrained backbone, train only a **small adapter** (about 7k–32k parameters), classify
with a **parameter-free prototype head**, and read that head's output in two ways: **softmax** and
**evidential (Dirichlet)**. We then measure three kinds of reliability together: **accuracy**,
**calibration** (is the confidence honest?) and **out-of-distribution (OOD) detection** (does it notice
images from classes it was never taught?).

---

## 2. What we built

### 2.1 The model pipeline

```
image (resized to 224×224, ImageNet normalisation, no augmentation)
  → frozen backbone        ResNet-18 (512-d) or MobileNetV3-Small (576-d), ImageNet-pretrained, BN stats frozen
  → trainable adapter      parallel Bottleneck or LoRA (rank 16 in the main grid)
  → prototype head         no weights; cosine similarity × 10 to each class's mean support feature
  → logits ─┬─ softmax reading     → probabilities, MSP / TS-MSP / energy scores
            └─ evidential reading  → evidence = softplus(logit·scale + bias), α = evidence + 1,
                                      p_k = α_k / S, vacuity = K / S   (scale, bias learnable, init (2, −6))
```

### 2.2 Components implemented

| Category | What we implemented | Notes |
|---|---|---|
| Backbones | ResNet-18, MobileNetV3-Small | Frozen trunks: 11,176,512 and 927,008 params |
| Adapters | Parallel bottleneck, serial bottleneck, post-pool bottleneck, LoRA, BitFit, Full Fine-Tuning, Linear Probe | Main grid uses parallel bottleneck and LoRA; Full FT and Linear Probe are baselines |
| Heads | Prototype head (main), linear head, evidential head (legacy) | Prototype head chosen because a linear head trained on 64 classes cannot transfer to 20 unseen test classes |
| Losses | Cross-entropy; evidential loss (Sensoy et al. 2018) with linear KL annealing (0 → 0.1 over 1,000 episodes), variance term dropped | |
| Trainer | Episodic meta-training: 100 episodes/epoch, 100 fixed VAL episodes, early stopping (patience 5, max 30 epochs) | Adam, LR 5e-3 (Full FT: 1e-5, WD 1e-4) |
| Datasets | CIFAR-FS (Bertinetto split), MiniImageNet (Ravi & Larochelle split), both 64/16/20 classes | Split files frozen |
| OOD pools | SVHN, Gaussian noise (far); CIFAR-100 / MiniImageNet held-out classes, TinyImageNet (near) | TinyImageNet's 25 MiniImageNet-overlap classes removed for MiniImageNet runs |
| OOD scores | MSP, temperature-scaled MSP, energy, evidential vacuity | |
| Metrics | Accuracy (+95% CI), macro-F1, ECE (pooled, 15 bins), Brier, AUROC, FPR@95, params, FLOPs, GPU/CPU latency, peak memory | |
| Post-hoc tools | Temperature scaling (VAL-fitted); evidence-affine refit (VAL-fitted) | |
| Analysis | η² variance decomposition with two-way interactions and bootstrap CIs; objective × score factorial; pre-registered matched-budget verdict engine; Pareto frontiers | `scripts/rq_*.py`, `scripts/rq3_matched.py`, `scripts/rq_completion.py` |

### 2.3 Engineering and reproducibility

- **Config-driven codebase.** Every component is chosen by a YAML key and built by a factory. Configs inherit
  through `extends:`. Entry points are `scripts/train.py` and `scripts/evaluate.py`.
- **Frozen evaluation protocol.** 600 fixed TEST episodes (seeds 0–599) and 100 fixed VAL episodes (seeds
  10000–10099). All tuning and post-hoc fitting used VAL only. No test seed was used for any choice.
- **Byte-identical reruns.** Full seeding and deterministic cuDNN, and `metrics.json` is written with sorted
  keys. Rerunning a config reproduces the file exactly. We checked this repeatedly.
- **Regression guards.** Every later analysis first re-scores the saved models and checks them against the
  committed grid numbers. Phase A matched 99/99, the RQ3 re-runs 18/18 and the completion run 21/21, all
  with max abs diff 0.0.
- **Test suite.** 33 test modules. The count grew from 28 tests (Step 2) to 294 (Step 11).
- **Generated results.** Grid tables, plots and `docs/RESULTS_MASTER.md` are regenerated by script, never
  edited by hand.
- **Compute.** Google Colab and Kaggle (mostly NVIDIA T4). The main grid alone took 36.3 GPU-hours.

### 2.4 Demo application (Sentinel)

`app/` is a FastAPI and web front-end demo for industrial product inspection. A user registers a product
from a few photos (few-shot prototype enrolment). The app then identifies items, reports an evidential
uncertainty and routes high-vacuity (unknown) items to manual inspection, showing softmax side by side.
It is a **demonstration only**. It uses a frozen ResNet-18 with its own simple model and is not connected to
the trained thesis adapters. It has no accuracy evaluation.

---

## 3. Everything we ran (chronological)

| # | Experiment | Date (2026) | Size | Role in the thesis |
|---|---|---|---|---|
| 1 | Step 1 — evidential calibration fix | Apr 21 | 20 episodes, stand-in data | History: found 2 loss bugs; ECE 0.526 → 0.167 |
| 2 | Step 2 — repo restructure | May 13 | Reproduction of Step 1 | Codebase, configs, 28 tests |
| 3 | Step 3 — W&B + reproducibility | May 16 | 2 configs × 600 episodes | Froze the 600 test episodes |
| 4 | Step 4 — real CIFAR-FS + episodic training | May–Jun | 2 configs | Negative result: OOD edge shrank to +0.006 |
| 5 | Step 4.5 — fair baselines (TS, energy, near-OOD, VAL sweep) | Jul 6–7 | 2 configs + VAL sweep | Fixed the training recipe used everywhere after |
| 6 | Step 5 — LoRA, BitFit, Full FT, Linear Probe | Jul 21 | 8 configs | Superseded by grid |
| 7 | Step 6 — adapter placement | Jul 21 | 4 configs | Only serial-vs-parallel comparison |
| 8 | Step 7 — Gaussian far-OOD | Jul 22 | 2 configs | Superseded by grid |
| 9 | Step 8 — MobileNetV3-Small | Jul 26 | 4 configs | Superseded by grid |
| 10 | Step 9 — MiniImageNet | Jul 30 | 10 configs | Superseded by grid |
| 11 | **Step 10 — full factorial grid** | Aug 2–6 | **120 runs, 36.3 GPU-h** | **Main dataset** |
| 12 | **Step 11 — efficiency / Pareto** | Aug 7–9 | 12 measured settings | Cost results |
| 13 | **Phase A — objective × score factorial + evidence refit** | Aug 25–26 | 99 recovered models, 5.51 GPU-h | RQ2, RQ4 |
| 14 | **Phase B — adapter rank sweep** | Aug 26 | 21 runs (ranks 1–64 × 3 seeds) | Tested and rejected an interior-optimum hypothesis |
| 15 | **RQ3 matched-budget experiment (pre-registered)** | Aug 26–27 | **48 runs, 6.2 h** | **Strongest result** |
| 16 | **Coverage completion + RQ1 interactions** | Sep 14–15 | 21 re-trained cells | Took RQ2/RQ4 to 120/120 |

**The main grid (Step 10).** It crosses 2 datasets (CIFAR-FS, MiniImageNet) × 2 shot counts (1, 5) × 2
backbones × 2 adapters (parallel bottleneck, LoRA) × 2 readings (softmax, evidential), which gives 32
settings. Eight baseline settings (Full FT and Linear Probe, ResNet-18 + CIFAR-FS) bring the total to **40
settings × 3 seeds (42, 43, 44) = 120 runs**. All 120 completed with zero errors, under one fixed recipe.

---

## 4. Research questions

The thesis asks four **attribution** questions of the balanced experimental data. They ask *what drives
what* rather than *does A beat B*. Direct method comparisons are reported as supporting results in
Section 5.

| RQ | Question | Answer | Novelty | Strength |
|---|---|---|---|---|
| RQ1 | How is variance in accuracy, calibration and OOD distributed across the design axes? | Shot count drives accuracy; head reading drives calibration | Partly new | 3rd |
| RQ2 | Is OOD detection caused by the training objective or by the scoring rule? | The scoring rule, decisively | Partly new | 2nd |
| RQ3 | Does accuracy follow adapter architecture and calibration follow parameter budget? | Accuracy follows architecture; calibration follows the **backbone** | **New** | **1st** |
| RQ4 | Can evidential calibration be fixed post hoc without breaking OOD ranking? | Improved in 60/60 cells, ranking kept in 80%; still worse than softmax | Partly new | 4th |

### 4.1 RQ1 — What drives what (variance decomposition)

The design is 32 balanced settings × 3 seeds = 96 runs, analysed with η² per factor.

| Outcome | Dataset | Shots | Backbone | Adapter | Head reading | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.8% | **76.1%** | 3.9% | 9.1% | 0.2% | 9.0% |
| ECE | 2.0% | 2.1% | 4.6% | 0.6% | **82.9%** | 7.7% |
| Far-OOD AUROC | 1.2% | 22.7% | 11.5% | 1.7% | **39.0%** | 23.9% |
| Near-OOD AUROC (TinyImageNet) | 1.6% | **42.6%** | 0.6% | 22.0% | 21.0% | 11.4% |

- The head reading explains about 83% of calibration but only 0.2% of accuracy.
- Calibration and OOD detection are nearly independent once the reading is fixed. Across all settings
  ρ ≈ +0.43 to +0.48, but within each reading it is only +0.15 to +0.26.
- **Two-way interactions (computed 15 Sep).** Adding them shrinks the residual (accuracy 8.95% → 1.31%;
  ECE 7.70% → 2.05%). The `backbone:adapter` term is significant **only for calibration**: 3.28% of ECE
  variance, 95% CI [2.52, 4.15], p < 1e-6. On accuracy and both OOD pools it is about zero. This
  independently corroborates RQ3.

### 4.2 RQ2 — Training objective or scoring rule?

In the original grid, evidential models were scored only with vacuity and softmax models only with softmax
scores, so every comparison changed two things at once. We scored **every** model with **all four** scores
(MSP, TS-MSP, energy, vacuity) under both objectives. That gives a 2 × 4 factorial with **full 120/120
coverage**.

| OOD type | Variance explained by score | Variance explained by training objective |
|---|---:|---:|
| Far-OOD | **42.7%** | **0.17%** |
| Near-OOD | **14.7%** | **0.63%** |

- Energy gives almost the same far-OOD AUROC on evidential-trained models (0.9185) as on softmax-trained
  models (0.9342). MSP stays near 0.80 under both objectives.
- Changing the score moves AUROC by about 0.13. Changing the objective moves it by at most about 0.02.
- Energy beats vacuity head-to-head in most comparisons (162/198 far, 183/198 near on the 99-record set).
- **Quote the shares, not a ratio.** Completing coverage moved the ratio from 163× to 250× while the shares
  barely changed.

### 4.3 RQ3 — Architecture, budget, or backbone? *(strongest result)*

**Stage 1 — grid evidence.** In 16 matched bottleneck-vs-LoRA pairs, bottleneck won accuracy 16/16 and
near-OOD 16/16. The *larger* adapter won calibration 16/16. That looked like "calibration follows budget".
But the bottleneck is larger on ResNet-18 and smaller on MobileNetV3-Small, so on the grid budget and
backbone were confounded.

**Stage 2 — pre-registered matched-budget experiment.** We wrote the decision rule down
(`docs/RQ3_MATCHED_BUDGET_PLAN.md`) **before** running. We then built both adapters at the **same** budget
within each backbone, with mismatch ≤ 3.1% (the original pairs differed by 55–158%).

| Backbone | Budget level | Bottleneck (rank → params) | LoRA (rank → params) |
|---|---|---|---|
| ResNet-18 | Low | 6 → 12,504 | 16 → 12,288 |
| ResNet-18 | High | 16 → 31,744 | 41 → 31,488 |
| MobileNetV3-S | Low | 16 → 6,928 | 10 → 6,720 |
| MobileNetV3-S | High | 22 → 9,448 | 14 → 9,408 |

The run covered MiniImageNet 5-shot × 2 backbones × 2 levels × 2 adapters × 2 readings × 3 seeds = **48
runs**, all trained fresh.

| Cell | ECE gap (LoRA − bottleneck) before matching | Gap at matched budget | Hypothesis fired |
|---|---:|---:|---|
| ResNet-18 / evidential | +0.111 | +0.112 | backbone |
| ResNet-18 / softmax | +0.108 | +0.099 | backbone |
| MobileNetV3-S / evidential | −0.012 | −0.006 | backbone |
| MobileNetV3-S / softmax | −0.015 | −0.006 | none (below threshold) |

- **Verdict: `backbone_intrinsic`.** The backbone hypothesis fired in 3/4 cells, meeting the pre-set
  threshold of 3. The budget hypothesis fired in 0/4.
- Equalising the budget left ResNet-18's gap intact and only halved MobileNetV3-Small's. The budget changes
  the gap's *size*; the backbone sets its *direction*.
- The architecture half held up: at matched budget, bottleneck still won accuracy 8/8 and near-OOD 8/8, all
  beyond 2σ.
- All 7 acceptance criteria passed. Parameter counts were exact, only the allowed config keys differed, and
  the 18 runs that repeated grid settings matched the grid exactly.
- The verdict went **against** the hypothesis we had favoured, which is why it is the most trustworthy result.
- **Limit:** with two backbones we cannot say *which* backbone property (depth, width, normalisation, block
  type, feature scale) is responsible.

### 4.4 RQ4 — Post-hoc refit of the evidential head

We refit only the two evidence parameters `(scale, bias)` on VAL episodes, then re-evaluated on the 600
test episodes.

| Result (current, 60 evidential cells) | Value |
|---|---|
| ECE improved | **60/60** cells; mean 0.323 → 0.185 (−0.139) |
| OOD AUROC preserved (Δ ≥ −0.005) | **192/240** comparisons (80%); mean Δ +0.003; worst drop −0.032 |
| Worst-case Spearman rank correlation | 0.866 |
| Still worse than plain softmax | **60/60** (mean 2.18×, best 1.07×, worst 5.8×) |
| Still worse than temperature-scaled softmax | **60/60** (mean 13.6×, best 2.27×) |

- The refitted scales (median 8.3) are far from the values learned in training (median 2.9). Joint training
  under the evidential loss leaves the head far from its NLL-optimal operating point.
- This corrects our own Step 4.5 conclusion that calibration "could not be tuned". That search tuned the loss,
  but the evidence affine is the parameter that controls calibration.
- Completing coverage made the ranking-preservation result slightly **weaker**. The 12 new cells preserved
  42/48 comparisons with mean ΔAUROC −0.0016. We report it in that direction.

### 4.5 Tested hypothesis that failed

The hypothesis was that calibration is best at an intermediate parameter budget. A controlled rank sweep
(CIFAR-FS, ResNet-18, parallel bottleneck; ranks 1–64 × 3 seeds, 21 runs, 2,886–124,098 params) showed
**no interior optimum**. Evidential ECE is lowest at rank 1 (0.291) and drifts up to rank 64 (0.309).
Softmax ECE moves the other way (0.093 → 0.081). The hypothesis was dropped and is reported as a negative
result.

---

## 5. Supporting results

| Question | Answer at grid scale |
|---|---|
| **Adapter choice and placement** | Parallel bottleneck beats LoRA in **16/16** matched pairs (+2.1 to +8.3 accuracy points). Serial and parallel tie on accuracy (Step 6, one setting: 0.9145 vs 0.9146), and parallel is better on OOD. A 31,744-param adapter beats Full FT at 5-shot on ResNet-18 (91.44% vs 90.47%). |
| **Does the evidential head calibrate better?** | **No, in 0 of 20 pairs.** Evidential ECE is worse than softmax by 1.4–9.1× and worse than TS-softmax by 5.3–51×. |
| **Does the evidential reading help OOD detection?** | **Split.** Vacuity beats MSP/TS-MSP in 37–38 of 40 settings on every pool, more so at 1-shot. But **energy beats vacuity in about 70%** of comparisons. |
| **Speed vs uncertainty trade-off** | The backbone drives latency (5.12×); the adapter barely does (3.9–5.8%). Evidential scoring is free (1.29%, below the 5.91% noise floor). Recommended edge point: MobileNetV3-Small + parallel + evidential, 11.86 ms/image, 6,930 params (CIFAR-FS). |

**Headline accuracies (5-way, 600 episodes, 3 seeds).** ResNet-18 + parallel bottleneck scores **91.44%**
on CIFAR-FS 5-shot and **95.56%** on MiniImageNet 5-shot. In Step 5, adapters reached about 98% of full
fine-tuning's accuracy with 660–2,330× fewer trainable parameters. In the grid, the 31,744-param parallel
adapter (352× fewer than Full FT) slightly exceeds it. Against published methods under the same protocol, it
is 1.06 points behind a ViT-S model that trains 662× more parameters (CIFAR-FS 5-shot). It is 3.56 points
ahead of a backbone-matched ResNet-50 (MiniImageNet 5-shot). Caveat: our backbones are ImageNet-pretrained,
so absolute accuracies are not comparable with from-scratch methods.

---

## 6. Our contributions

1. **A reliability-complete benchmark** of PEFT on frozen lightweight CNNs for few-shot classification. It
   has 120 runs (40 configurations × 3 seeds) and reports accuracy, macro-F1, ECE, Brier, near- and far-OOD
   AUROC and FPR@95, parameters, FLOPs and latency, all on the same 600 fixed episodes. The individual parts
   have precedent; we did not find them reported together in this setting.
2. **A variance attribution of reliability across five design axes (RQ1) — partly novel.** Shots explain
   76.1% of accuracy; the head reading explains 82.9% of ECE and 0.2% of accuracy. Calibration and OOD
   become nearly independent once the reading is fixed. The `backbone:adapter` interaction matters only for
   calibration.
3. **A factorial separation of training objective and scoring rule for OOD, extended to Dirichlet models
   (RQ2) — partly novel.** The score explains 42.7% (far) and 14.7% (near) of AUROC variance; the objective
   explains under 1%. We found no precedent for applying the energy score to evidential-trained models, where
   it works almost as well as on softmax models.
4. **A pre-registered matched-budget experiment separating adapter architecture from parameter budget (RQ3)
   — novel, the strongest contribution.** Accuracy and near-OOD follow architecture (8/8); calibration
   follows the backbone (budget hypothesis 0/4, backbone 3/4).
5. **A two-parameter post-hoc refit of the evidential prototype head with measured OOD-ranking preservation
   (RQ4) — partly novel.** ECE improves in 60/60 cells and ranking is kept in 80% of comparisons. The gap
   to softmax is honestly reported as not closed.
6. **Documented negative and corrected results:** evidential calibrates worse than softmax (0/20); energy
   beats vacuity in about 70% of comparisons, which overturns our own earlier single-setting finding; there
   is no interior calibration optimum over ranks 1–64; and the calibration-follows-budget reading was
   overturned by our own pre-registered test.
7. **Engineering artefacts:** a reproducible, config-driven codebase with byte-identical reruns, frozen
   VAL/TEST episode seeds, regression guards that re-verify the committed grid before every new analysis,
   generated results tables, Pareto/efficiency tooling, and a demo app.

---

## 7. Self-corrections made during the project

Many early expectations turned out to be wrong. Each was caught by our own follow-up experiments and is
reported openly.

| # | We expected / claimed | What the evidence showed | Caught at |
|---|---|---|---|
| 1 | Evidential head calibrates better (initial single-episode experiment, ECE 0.526) | The number came from two loss bugs. Even after the fix, evidential is worse than softmax in 20/20 | Step 1, Step 10 |
| 2 | ReLU evidence | Dead outputs; switched to softplus | Step 1 |
| 3 | A linear classifier head | Cannot transfer to unseen classes; switched to a prototype head | Step 4 |
| 4 | Softplus of raw prototype logits | "Evidential collapse" (no evidence, dead gradients); fixed with a learnable scale/bias | Step 4 |
| 5 | Large evidential OOD advantage | Shrank to +0.006 on the real benchmark | Step 4 |
| 6 | Full fine-tuning overfits on 25 images | It was the most accurate method (episodic meta-training prevents overfitting) | Step 5 |
| 7 | MobileNetV3-Small loses 7–8 points | It lost 0.3–1.6 on CIFAR-FS (but 4.7–5.7 on MiniImageNet) | Steps 8–9 |
| 8 | Vacuity is on par with energy | Energy wins about 70% of comparisons at grid scale | Step 10 |
| 9 | Evidential calibration cannot be tuned | Refitting scale/bias improves it in 60/60 cells | RQ4 |
| 10 | Calibration has an optimal middle budget | No interior optimum over ranks 1–64 | Phase B |
| 11 | Calibration follows the parameter budget | It follows the backbone (pre-registered test) | RQ3 |
| 12 | Silent latency bug | Three scripts used laptop instead of Kaggle CPU latency (errors up to 47%); fixed, recommendations unchanged | Step 11 |
| 13 | One committed grid file was a 20-episode smoke run | Measured impact ≤ 0.70 pp on any η²; no conclusion changes | Completion run |

---

## 8. Limitations (to state in the thesis)

1. **ImageNet pretraining overlap.** MiniImageNet test classes are ImageNet classes, so absolute accuracy
   is inflated. Our claims are relative comparisons.
2. **One fixed training recipe**, tuned once on ResNet-18 + CIFAR-FS + 5-shot. In addition, `kl_weight_max`
   = 0.1 was used although the VAL sweep ranked 0.05 first (the difference is small and the surface flat).
3. **Three seeds.** The seed changes only the adapter initialisation. Full FT and Linear Probe have no seed
   spread.
4. **Two backbones.** RQ3 identifies the backbone as the cause but not the mechanism.
5. **Baselines only on ResNet-18 + CIFAR-FS.**
6. **No real edge device.** Latency is hardware-dependent and not byte-reproducible.
7. **Held-out-class near-OOD pools are the VAL classes.** They are disjoint from the test classes but were
   seen during model selection. TinyImageNet has no such overlap.
8. **Cosine prototype head**, not ProtoNet's L2; **no data augmentation**.
9. **Three-way and higher interactions** are not modelled.
10. **Novelty searches** were thorough but cannot be exhaustive.

---

## 9. Wording rules and superseded numbers

| Do not write | Write instead |
|---|---|
| "Calibration follows the parameter budget." | "Calibration follows the backbone; budget changes the size of the gap, the backbone its direction." |
| "Evidential uncertainty is on par with energy." | "Vacuity beats softmax-probability scores, but energy still beats vacuity in most cells." |
| "Evidential training improves OOD detection." | "The OOD gain comes from the scoring rule, not the training objective." |
| "The score matters 163× more than the objective." | "The score explains 42.7% / 14.7% (far / near); the objective under 1%." |
| "Refitting fixes evidential calibration." | "Refitting improves it in 60/60 cells but it stays worse than softmax in all 60." |
| "The evidence scale/bias are frozen at (2, −6)." | "They are learnable, initialised at (2, −6)." |
| "We know why the backbone controls calibration." | "The responsible backbone property is not identified." |

**Superseded figures still present in some older documents** (`RQ_SUPERVISOR_REPORT.md`/`.pdf`,
`07_introduction.md`). Use the current values:

| Item | Old (99 records / 48 cells) | Current (120 / 60) |
|---|---|---|
| RQ2 far-OOD shares | 43.7% / 0.27% | **42.7% / 0.17%** |
| RQ2 near-OOD shares | 13.0% / 0.60% | **14.7% / 0.63%** |
| RQ4 ECE improved | 48/48 | **60/60** |
| RQ4 ranking preserved | 150/192 (78%) | **192/240 (80%)** |
| RQ4 worst ρ | 0.921 | **0.866** |
| RQ4 worse than TS-softmax (mean) | 11.9× | **13.6×** |

---

## 10. Where the evidence lives

| What | File |
|---|---|
| RQ statements, answers, novelty | `docs/RQ_SUPERVISOR_REPORT.md`, `docs/RQ_RESULTS_SUMMARY.md` |
| All 40-cell grid tables | `docs/RESULTS_MASTER.md` (generated), `results/mvt_results.json` |
| RQ3 pre-registration and verdict | `docs/RQ3_MATCHED_BUDGET_PLAN.md`, `results/rq3_matched/verdict.json`, `step_writeups/rq3_matched_budget.txt` |
| RQ2/RQ4 at full coverage, RQ1 interactions | `results/rq_completion/REPORT.md`, `rq2_rq4_summary.json`, `rq1_interactions.json` |
| Rank sweep | `results/rq5/`, `results/rq5_rank_sweep.png` |
| Efficiency / Pareto | `results/efficiency_table.json`, `results/pareto_frontier.json`, `step_writeups/step11.txt` |
| Per-step history | `step_writeups/step1–11.txt`, `progress.txt` |
| Plain-language guide and chapter drafts | `docs/guide/01–10` |
| Figures | `docs/figures/` |
| Defence material | `docs/DEFENCE_DECK_25_SLIDES.md`, `docs/DEFENCE_SLIDE_PLAN.md`, `docs/DEFENCE_BRIEF.md` |
| Bibliography | `docs/refs.bib`, `docs/CITATION_AUDIT.md` |
