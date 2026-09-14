# 10. Discussion, Conclusion and Future Work (thesis Chapters 6–7 draft)

Source material for **Chapter 6 (Discussion)** and **Chapter 7 (Conclusion)**. `03_results.md` reports
*what* was found. This file interprets it: how the four answers fit together, how they relate to prior
work, what a practitioner should take away, and what could make them wrong.

Every number is from `03_results.md`. Text marked **Interpretation** goes beyond what an experiment tested;
keep that label (or "we suggest", "is consistent with") in the thesis.

---

## 10.1 The four answers as one picture

The overarching question asked what governs reliability and whether the deficits can be repaired. Read
together, the four RQs give a single, simple answer: **the three reliability properties are controlled by
different parts of the system.**

| Property | Controlled mainly by | Barely affected by | Evidence |
|---|---|---|---|
| **Accuracy** | Information available (shots, 76%) and **adapter architecture** (bottleneck wins 16/16, and 8/8 at matched budget) | Head interpretation (0.2%) | RQ1, RQ3 |
| **Calibration** | **Head interpretation** (83%) and, between adapters, the **backbone** | Adapter parameter budget (budget hypothesis 0/4) | RQ1, RQ3 |
| **OOD ranking** | **Scoring rule** (far 43.7%, near 13.0%) | Training objective (< 1%) | RQ2 |
| **Repairability** | Calibration of the evidential head responds to its two evidence parameters (48/48) | OOD ranking is largely unaffected by the refit (78% within 0.005) | RQ4 |

**Why this matters.** It turns "choose a method" into three largely separable decisions:

- representation (backbone and adapter) decides accuracy;
- readout decides calibration;
- score decides OOD detection.

The separation is not perfect:

- near-OOD AUROC also depends on shots (42.6%) and adapter (22.0%);
- the calibration gap between adapters depends on the backbone.

But the dominant term differs for each property.

**A consequence for the thesis title.** The titular "Bayesian" component, the evidential Dirichlet head,
contributes to reliability through its **uncertainty score**, not through its **training objective** or its
**probabilities**:

- its probabilities are worse calibrated than softmax's in every matched comparison (0/20);
- the same logits scored with energy rank OOD inputs better than vacuity in most comparisons;
- evidential training itself explains under 1% of OOD variance.

State this plainly. It is the main scientific message, and a committee will reach it independently.

---

## 10.2 Interpreting each result

### 10.2.1 RQ1: separation of concerns

The head explains 83% of ECE variance but 0.2% of accuracy variance. Both readouts use the **same logits**
and the same argmax, so they almost always make the same predictions; they differ only in how confidence
is assigned. A near-zero accuracy effect is therefore expected. The informative part is the other half:
nothing else in the design (dataset, shots, backbone, adapter) moves calibration by more than 5%.

**Calibration–OOD orthogonality.** Across all 40 cells, worse-calibrated configurations appear to detect
OOD better (ρ ≈ +0.43/+0.48). Within a single readout this falls to +0.15–0.26. The apparent trade-off is a
head effect: the evidential cells are both worse calibrated and scored with vacuity. **Practical reading:**
tuning a head for calibration should not be expected to help or hurt its OOD ranking, and vice versa. RQ4
is a direct test of that prediction, and it largely held.

### 10.2.2 RQ2: the score, not the objective

Energy computed on evidential-trained logits reaches 0.911 far-OOD AUROC, against 0.929 on softmax-trained
logits. MSP stays around 0.79 under both objectives. Changing the score moves AUROC by about 0.13; changing
the objective moves it by at most about 0.02, and the far-OOD difference goes against the evidential
objective.

**Relation to prior work:**

- Agrees with the post-hoc-score literature: energy beats MSP [liu2020energy], and post-hoc methods are no
  worse than training-based ones [yang2022openood, bitterwolf2022breaking].
- Extends it to Dirichlet-trained models, which no work found had done.
- Consistent with the theoretical result that softmax is a special case of an evidential classifier
  [hayta2026pluginlosses]. If the two objectives define closely related functions of the same logits,
  similar logit geometry is to be expected.

**Interpretation.** On this reading, vacuity's advantage over MSP (37–38/40 wins in the grid) is mostly a
property of *reading the logits through a sum-of-evidence statistic* rather than a softmax maximum. Energy
is another such statistic. The Liu et al. summary itself notes that Dirichlet strength and negative energy
both measure "amount of support".

### 10.2.3 RQ3: architecture for accuracy, backbone for calibration

**Accuracy follows architecture, robustly.** The bottleneck wins accuracy 16/16 in the grid and 8/8 at
matched budget, all beyond 2σ, on a backbone where it has fewer parameters (MobileNetV3-Small) and on one
where it has more (ResNet-18).

**Interpretation, not tested:** the bottleneck adapts four stages at several depths, while LoRA modifies
one 1×1 convolution in the last stage. Coverage of the network may matter more than parameter count. This is
consistent with ConvLoRA's finding that adapting the whole encoder beats adapting part of it
[aleem2024convlora] and with Houlsby's observation of where adapters concentrate [houlsby2019adapters]. It
remains a hypothesis: no experiment varied the number of adapted layers at fixed budget.

**Calibration follows the backbone.** Equalising budget left ResNet-18's gap intact (collapse ratios 1.01,
0.92) and roughly halved MobileNetV3-Small's (0.50, 0.61). Budget changes the size of the gap; the backbone
sets its direction. **The mechanism is not identified.** Depth, width, block type, normalisation and feature
scale at the insertion sites all differ between the two backbones. The one observation available is that
ResNet-18 LoRA stops training 3–4× earlier than the other arms (best epoch 1–4); report it only as
something to look at next (`04_experiments.md` §4.7).

**Why this is the strongest result.** It was pre-registered: rule, thresholds and an "inconclusive" outcome
were written before the deciding runs. It also went *against* the reading the project favoured. Emphasise
both points.

### 10.2.4 RQ4: repairable, but not repaired

The refit improves ECE in every cell (mean 0.327 → 0.190). Refitted evidential ECE remains 2.18× plain
softmax on average and 11.9× temperature-scaled softmax, in 48/48 cells.

**The evidential head is underconfident, not overconfident.** The grid's reliability diagrams for ResNet-18 +
bottleneck (seed 42; `results/grid_plots/*_evidential_reliability.png`) show accuracy well **above**
confidence in every bin:

- on CIFAR-FS 5-shot, bins at ~0.2–0.45 confidence are ~65–70% accurate;
- no prediction exceeds ~0.93 confidence;
- the matching softmax diagram lies much closer to the diagonal, still slightly under it.

*Check the MobileNetV3 and LoRA cells before generalising beyond these plotted cells.*

**Interpretation (a plausible contributing mechanism, untested).** Three design choices bound how confident
the evidential head can become:

- cosine logits are bounded to [−10, 10];
- evidence is an affine-softplus of those logits;
- α = e + 1 adds one unit of prior mass to every class.

Together these limit how much of the Dirichlet mass the top class can hold. RQ4's refit consistently
**raises** the scale (median 2.9 → 8.3), which sharpens evidence and is consistent with correcting
underconfidence. It also agrees with the R-EDL observation that the "+1" prior is a non-essential,
calibration-relevant choice [chen2024reedl, gao2024edlsurvey].

**Relation to prior work:**

- *Be Confident in What You Know* reports that PEFT-adapted foundation models are **underconfident**
  [pandey2024bayesianpeft]. The direction matches; the setting (ViT foundation models) does not.
- BEL [linghu2022bel] and BayesAdapter [moralesalvarez2025bayesadapter] report Bayesian or evidential
  calibration **improving**. This thesis's negative result sits at the low-capacity end: a frozen backbone,
  6.9k–31.7k trainable parameters, and 1–5 shots.
- Theory reporting non-vanishing epistemic uncertainty in reverse-KL EDL objectives [hoarau2026epistemic]
  points the same way, although the loss here is the squared-error form.

**The Step 4.5 lesson.** A VAL sweep over the loss found a flat ECE surface and concluded calibration could
not be tuned. The refit shows the lever was the evidence mapping, which that sweep never varied. A flat
surface along the searched axes says nothing about the axes not searched.

### 10.2.5 The hypothesis that did not survive

With the architecture fixed, varying rank from 1 to 64 produced two opposite monotone trends: evidential ECE
drifts up and softmax ECE drifts down. There is no interior optimum. The earlier "U-shape" came from
comparing different adapter *types* at different budgets. That is the same confound RQ3 later removed,
which is why the two results support each other.

LoRA-Ensemble's rank-32 deterioration [muhlematter2026loraensemble] is a partial contradiction. It is
answered by the differences in adaptation target (attention LoRA in a ViT vs 1×1 conv adapters), head (a
trained linear head vs a parameter-free prototype head) and relative capacity (`03_results.md` §3.7).

---

## 10.3 Practical guidance (scoped)

**Scope line to keep attached:** *under one fixed recipe, two CNN backbones, CIFAR-FS and MiniImageNet,
5-way 1/5-shot.*

| If you need… | Choose | Because |
|---|---|---|
| Accuracy at a small budget | Bottleneck-parallel adapter over single-layer LoRA | 16/16 grid, 8/8 matched |
| The lowest latency | MobileNetV3-Small | The backbone drives latency 5.12×; the adapter only 3.9–5.8% |
| Calibrated probabilities | Softmax readout + temperature scaling fitted on validation episodes | TS-ECE 0.006–0.064 across the grid |
| OOD ranking | Energy on the prototype logits, whichever objective trained the model | RQ2; energy > vacuity in ~70% of grid comparisons |
| An explicit "don't know" output | Evidential readout, then **refit (a, b) on validation episodes** | Vacuity > MSP (37–38/40); refit improves ECE 48/48 at negligible cost |
| Uncertainty without latency cost | Either readout | Evidential scoring costs 1.29% (below the 5.91% noise floor) |

**Recommended edge operating point (Step 11):** MobileNetV3-Small + bottleneck-parallel + evidential,
11.86 ms/image on one CPU thread, 6,930 trainable parameters (CIFAR-FS). Given RQ2 and RQ4, a thesis can
reasonably add: *score it with energy for OOD, and refit its affine if its probabilities are used.*

---

## 10.4 Relation to the original proposal

| Proposal expectation | Outcome | How to present it |
|---|---|---|
| Evidential regularisation gives better calibration | Worse in 20/20 | A boundary case of a known effect (§10.2.4) |
| Evidential regularisation improves near-OOD detection | Vacuity beats softmax-probability scores, but the gain belongs to the score, not the regulariser | RQ2 reframes it |
| Serial vs parallel placement is a key design choice | Tie on accuracy (Step 6); TSA already answered it | Supporting result |
| Full FT overfits on 25 images | Full FT was among the most accurate runs, because episodic meta-training sees many episodes | Reported as a corrected expectation |
| Pareto frontier for edge deployment | Measured; the backbone dominates | Supporting result |

The shift from comparison questions to attribution questions (`08_literature_review.md` §8.9.3) is a
strength to explain, not a weakness to hide: every experiment was kept, and the new questions are ones the
balanced design can answer more convincingly.

---

## 10.5 Threats to validity

Consolidates `05_problems_and_open_work.md` §C and adds three items found on 2026-09-14 (marked **new**).

### Internal validity: could something other than the factor explain the result?

- **One fixed recipe.** Tuned once (ResNet-18, CIFAR-FS, 5-shot). Differences between cells could partly
  reflect how well the recipe suits each cell, especially MobileNetV3 and LoRA. ResNet-18 LoRA's very early
  best epochs are a symptom to note.
- **new — kl_weight_max = 0.1 is not the VAL-sweep winner (0.05).** The gap is small on a flat surface, but
  it must be disclosed (`09_methodology.md` §9.5).
- **new — near-OOD held-out-class pools use the validation classes**, which also drive early stopping,
  temperature fitting and the RQ4 refit. TinyImageNet is unaffected and is reported everywhere.
- **Seeds change only the adapter initialisation**, not episode order, so seed SD understates run-to-run
  variability. Full FT and Linear Probe have none.
- **RQ2/RQ4 cover 99 of 120 models.** The missing 21 are all CIFAR-FS 5-shot adapter models.

### Construct validity: do the metrics measure what we claim?

- **ECE is binned** (15 equal-width bins, pooled). Its absolute values depend on binning; the comparisons
  here are relative and use the same binning everywhere.
- **OOD pools are 500 images each**, and AUROC is averaged per episode; Gaussian noise is a sanity pool,
  not a realistic shift.
- **Latency comes from a single-thread CPU proxy**, not an edge device, and is not byte-reproducible.
- **new — the prototype metric is cosine, not ProtoNet's recommended squared Euclidean distance.** The
  results describe cosine prototype heads.

### External validity: how far do the results generalise?

- **Two backbones, two datasets, one task format.** RQ3 in particular cannot generalise beyond "the backbone
  matters" without more backbones.
- **ImageNet pretraining overlap** inflates MiniImageNet accuracy; conclusions are about relative
  differences.
- **No augmentation, no fine-grained or medical data, no corruption shift** (CUB, ISIC and CIFAR-10-C were
  not run).
- **Transformers were not trained** under the protocol.

### Statistical conclusion validity

- **RQ1 fits main effects only**; interactions sit in the residual.
- **RQ3's 16/16 sign consistency** has p ≈ 3×10⁻⁵ under a random-direction null, but magnitudes clear 2σ in
  only 10/16 unmatched pairs. The matched experiment's per-cell 2σ test is the stronger evidence.
- **Former RQ5:** 3 seeds × 7 ranks may be underpowered to exclude a subtle interior optimum. Use "not
  observed in the tested range".
- **The RQ2 score/objective ratio is unstable** (163× vs 394× depending on one unbalanced design). Report η²
  shares.
- **Novelty searches were extensive but not exhaustive.**

---

## 10.6 Conclusion

This thesis asked what governs the reliability of parameter-efficient adaptation of frozen lightweight CNNs
for few-shot image classification, and whether the deficits can be repaired after training. On ResNet-18
and MobileNetV3-Small, across CIFAR-FS and MiniImageNet, a balanced 120-run factorial and three follow-up
experiments give the following answers:

1. **Different parts of the system govern different reliability properties (RQ1).** The number of shots
   explains most accuracy variance (76%). The interpretation of the head explains most calibration
   variance (83%) and almost none of the accuracy variance. Once the head is fixed, calibration and OOD
   detection are nearly independent.
2. **OOD detection is determined by the scoring rule, not the training objective (RQ2).** The score
   explains 43.7% (far) and 13.0% (near) of AUROC variance, the objective under 1%. The energy score works
   almost as well on evidential-trained models as on softmax-trained ones.
3. **Accuracy follows adapter architecture; the calibration difference between adapters follows the
   backbone (RQ3).** In a pre-registered matched-budget experiment, the bottleneck adapter won accuracy and
   near-OOD ranking in 8/8 comparisons. The parameter-budget explanation of calibration was rejected (0/4),
   and a backbone-intrinsic explanation was supported (3/4). Which backbone property is responsible remains
   open.
4. **The evidential head's calibration deficit is partly repairable (RQ4).** Refitting two parameters on
   validation episodes improved ECE in 48/48 cells while largely preserving OOD ranking. The refitted head
   still does not match softmax with temperature scaling.

Several expectations from the proposal did not hold, most importantly that evidential heads calibrate
better than softmax. Two of the project's own earlier conclusions were overturned by its later experiments.
These results are reported alongside the positive ones. For a practitioner deploying a small adapted CNN,
the findings reduce to three largely independent choices:

- the representation for accuracy;
- the readout, with post-hoc fitting, for calibration;
- the score for OOD detection.

---

## 10.7 Future work

Ordered by how directly each item addresses a stated limitation.

| # | Direction | Addresses | Notes |
|---|---|---|---|
| 1 | **More backbones** (e.g. a third and fourth CNN differing in one property at a time) to identify *which* backbone property drives the adapter calibration gap | RQ3 mechanism | "More backbones, not more seeds" |
| 2 | **Retrain the 21 missing CIFAR-FS 5-shot models** and rerun Phase A | RQ2/RQ4 coverage | ~7 GPU-h (`05_problems_and_open_work.md` E2) |
| 3 | **Per-cell VAL tuning** (LR, KL weight incl. 0.05, patience), especially for MobileNetV3 and LoRA | Fixed-recipe threat | Planned side study 10.9, never run |
| 4 | **Same-backbone baselines**: Full FT and Linear Probe on every backbone × dataset | Cross-backbone comparisons | 72 runs, ~18–19 GPU-h (Step 12.F) |
| 5 | **Coverage vs budget**: adapt a varying number of stages or layers at a fixed parameter budget | Tests the interpretation in §10.2.3 | New experiment |
| 6 | **Confidence-ceiling test**: vary the prior mass (R-EDL `prior_per_class`), cosine scale τ, and L2 vs cosine | Tests the underconfidence interpretation in §10.2.4 | Knobs already exist in the code |
| 7 | **Refit objectives that target OOD as well as calibration**, and other post-hoc scores (Mahalanobis, KNN, ODIN) on Dirichlet models | Extends RQ2/RQ4 | Mahalanobis is robust with small data [lee2018mahalanobis†] |
| 8 | **A Transformer arm** (ViT-Tiny / DeiT-Small) under the same protocol | "Why not ViTs?" | Only size/latency measured so far |
| 9 | **A from-scratch (non-ImageNet) backbone control** | Pretraining overlap | Step 12.J |
| 10 | **Real edge hardware** (Jetson-class or MCU) | Latency proxy | — |
| 11 | **Harder shifts**: CUB-200 (fine-grained), ISIC (medical), CIFAR-10-C (corruptions) | External validity | From the proposal |
| 12 | **Data augmentation** during meta-training | A Closer Look's warning | Would change absolute numbers; relative conclusions need re-checking |
