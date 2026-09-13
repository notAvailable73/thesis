# B-PEFT — Research Questions, Answers, and Novelty Assessment

**Thesis:** Parameter-efficient fine-tuning of frozen lightweight CNN backbones for few-shot image
classification, evaluated for accuracy, calibration, and out-of-distribution (OOD) detection.

**Prepared for:** supervisor review · **Date:** 2026-08-27 · **Status:** all research questions answered;
no experiments outstanding.

**Purpose of this document.** It states each research question, the answer obtained, the closest prior work
found, and an explicit assessment of what is novel and what is not. It is written to be self-contained — no
repository access is required to follow it. Where a claim is weaker than it might first appear, that is
stated in the text rather than left for the reader to discover.

**Novelty labels used throughout:**

| Label | Meaning |
|---|---|
| **[NOVEL]** | No prior work found that makes this claim. |
| **[PARTLY NOVEL]** | The method or the general pattern has precedent; a specific component does not. |
| **[CONFIRMATORY]** | The finding is established; the contribution is rigour or quantification in a new setting. |

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Experimental setup](#2-experimental-setup)
3. [RQ1 — Attribution of outcome variance](#3-rq1--attribution-of-outcome-variance)
4. [RQ2 — Training objective versus scoring rule](#4-rq2--training-objective-versus-scoring-rule)
5. [RQ3 — Architecture, budget, or backbone](#5-rq3--architecture-budget-or-backbone)
6. [RQ4 — Post-hoc remediation of the calibration deficit](#6-rq4--post-hoc-remediation-of-the-calibration-deficit)
7. [A tested hypothesis that did not survive](#7-a-tested-hypothesis-that-did-not-survive)
8. [Appendix A — the original proposal questions](#appendix-a--the-original-proposal-questions)
9. [Appendix B — limitations applying to all results](#appendix-b--limitations-applying-to-all-results)
10. [Appendix C — how novelty was checked](#appendix-c--how-novelty-was-checked)

---

## 1. Executive summary

**Overarching question**

> What governs reliability — predictive accuracy, confidence calibration, and out-of-distribution
> detection — in parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image
> classification, and can the reliability deficits so identified be remediated post-hoc?

| # | Question, in one line | Answer, in one line | Novelty |
|---|---|---|---|
| **RQ1** | Which design axis drives which outcome? | Shot count drives accuracy; head type drives calibration; the two are near-independent. | **[PARTLY NOVEL]** |
| **RQ2** | Is OOD detection caused by the training objective or the scoring rule? | The scoring rule, decisively — by a factor of 163 on far-OOD. | **[PARTLY NOVEL]** |
| **RQ3** | Does accuracy follow adapter architecture and calibration follow parameter budget? | Accuracy follows architecture. Calibration follows *neither* — it follows the backbone. | **[NOVEL]** |
| **RQ4** | Can the evidential head's calibration be fixed after training without breaking OOD ranking? | Yes in 48/48 cells; ranking survives in 78% of comparisons, degrades gracefully in the rest. | **[PARTLY NOVEL]** |

**Ranked by strength of contribution: RQ3 > RQ2 > RQ1 > RQ4.**

**The single most defensible result** is RQ3, because the hypothesis was pre-registered — the decision rule,
its statistical thresholds, and an explicit "inconclusive" outcome were all committed to writing before any
of the deciding runs existed — and the experiment then returned a verdict **against** the hypothesis the
project had been favouring. That correction is reported here rather than quietly dropped.

**Two earlier findings were corrected during the work, and both corrections are reported:** the claim that
evidential uncertainty is competitive with the energy score (superseded at grid scale), and the claim that
calibration follows the parameter budget (refuted by direct experiment). Neither correction was forced by a
reviewer; both were found by the project's own follow-up tests.

---

## 2. Experimental setup

### 2.1 The model

```
frozen ImageNet-pretrained CNN backbone
        |
        v
small trainable adapter            <- the only trained parameters
        |
        v
parameter-free prototype head      <- classifies by similarity to support-set class means
```

The backbone is never updated (except in the Full-Fine-Tuning baseline). The head has no trainable weights:
classification logits are the similarity between a query embedding and the mean embedding of each class's
support images. A configuration's trainable-parameter count is therefore essentially the adapter's size.

**Head "interpretation" is a separate axis from head type.** The prototype head always emits raw similarity
logits, which are then read in one of two ways:

- **Softmax** — logits to softmax to predictive probabilities. Confidence = maximum softmax probability.
- **Evidential (Dirichlet)** — `evidence = softplus(logits x scale + bias)`, `alpha = evidence + 1`,
  `S = sum(alpha)`. Probability = `alpha/S`. Uncertainty = *vacuity* = `K/S`, with K = 5.

The evidential mapping adds exactly **two trainable parameters** (`scale`, `bias`). These are frozen at
(2, −6) across the main grid; RQ4 is the experiment that refits them.

### 2.2 Protocol

| | |
|---|---|
| Task | 5-way {1, 5}-shot episodic classification |
| Test episodes | 600 fixed episodes, seeds frozen in a version-controlled file |
| Training seeds | 3 per configuration (42, 43, 44) |
| Datasets | CIFAR-FS (Bertinetto split), MiniImageNet (Ravi & Larochelle split) |
| Backbones | ResNet-18 (11.7M params), MobileNetV3-Small (2.5M params) — ImageNet-pretrained, frozen |
| Hyperparameters | **One frozen recipe across all 40 configurations.** Not re-tuned per cell. |

Test classes are disjoint from the classes the adapter meta-trains on.

### 2.3 The grid

**40 unique configurations × 3 seeds = 120 runs**, all completed.

- 2 datasets × 2 shot regimes × 2 backbones × 2 adapters × 2 head interpretations = 32 cells
  (a **balanced full factorial**, which is what makes the RQ1 variance decomposition possible)
- plus 8 baseline cells (Full-FT, Linear-Probe) on CIFAR-FS × ResNet-18 only (unbalanced, therefore
  excluded from the variance decomposition)

**Adapters compared:** *bottleneck-parallel* (1×1 down-projection, ReLU, 1×1 up-projection, run on the block
input and summed at its output, at the final block of each of four stages); *LoRA* (a low-rank update
injected into one 1×1 convolution); *Full-FT* and *Linear-Probe* as baselines.

**Metrics.** Accuracy and macro-F1 (mean over 600 episodes); **ECE (pooled)** — lower is better; Brier;
**OOD AUROC** — higher is better, over SVHN and Gaussian noise (far-OOD) and CIFAR-100-heldout /
MiniImageNet-heldout and TinyImageNet (near-OOD).

**Uncertainty scores.** Evidential cells are scored with *vacuity*; softmax cells with *MSP* (maximum
softmax probability), *TS-MSP* (after temperature scaling), and *energy*. This asymmetry is precisely what
RQ2 exists to dissolve.

### 2.4 A structural accident that makes RQ3 possible

Trainable parameter counts (softmax variant; evidential adds 2):

| Backbone | Bottleneck-parallel | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck, by 2.58× |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA**, by 1.55× |

**The parameter ordering reverses between the two backbones.** This was not designed — it falls out of the
two backbones' channel widths. It partially separates adapter *architecture* from adapter *parameter
budget*, which is normally impossible in a PEFT study because the better-performing adapter is usually also
the larger one everywhere. RQ3 begins from this accident; Section 5.3 is the experiment that finishes the
job the accident could not.

---

## 3. RQ1 — Attribution of outcome variance

> **How is variance in accuracy, calibration, and out-of-distribution detection distributed across the
> principal design axes of a parameter-efficient few-shot classification system — dataset, shot count,
> backbone, adapter type, and uncertainty-head interpretation?**

### 3.1 Answer

Main-effects eta-squared computed over the balanced 32-cell factorial, on **per-seed observations**
(96 observations, not cell-averaged, so a genuine residual term is retained):

| Metric | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.75% | **76.05%** | 3.92% | 9.14% | 0.19% | 8.95% |
| ECE | 2.02% | 2.13% | 4.63% | 0.63% | **82.89%** | 7.70% |
| OOD AUROC (far) | 1.18% | 22.68% | 11.54% | 1.73% | **39.01%** | 23.87% |
| OOD AUROC (TinyImageNet near) | 1.64% | **42.60%** | 0.61% | 21.98% | 20.98% | 11.42% |

**Head interpretation explains roughly 83% of calibration variance and 0.2% of accuracy variance** — close
to a clean separation of concerns. How many shots are available dominates accuracy; which head is used
dominates calibration; the two barely interact.

**Supporting result — calibration and OOD detection are orthogonal, not in tension.** Across all 40 cells
the rank correlation between ECE and OOD AUROC is *positive* (Spearman rho = +0.433 for SVHN-far, +0.477 for
TinyImageNet-near), which would naively suggest that worse-calibrated configurations detect OOD better. This
is a head effect: stratifying by head interpretation collapses it to rho = +0.150/+0.195 (evidential) and
+0.263/+0.242 (softmax). **The defensible statement is orthogonality — once the head is fixed, calibration
carries almost no information about OOD detection.**

**Supporting measurement — uncertainty is free at inference.** Evidential versus softmax mean absolute
latency difference at matched (backbone, adapter), on a single-thread edge-proxy CPU: **1.29%**, below the
measurement session's own 5.91% noise floor. Uncertainty scoring for a 75-query episode costs 1.09% of a
single image's backbone forward pass.

**Robustness check, now closed.** An earlier version of this table averaged seeds before decomposing, which
risks suppressing the error term and manufacturing apparent cleanliness. Recomputing on all 96 raw per-seed
observations moved the residual by only **0.3 to 3.6 percentage points** — nowhere near enough to explain
the split as an averaging artefact. The claim holds on per-seed data.

### 3.2 Nearest prior work, and why it is not this

| Prior work | What it does | Why it is not this result |
|---|---|---|
| **A Benchmark Study on Calibration** (ICLR 2024, arXiv:2308.11838) | The largest calibration study in existence — 117,702 architectures. | Read in full. It runs **no** ANOVA, eta-squared, or variance decomposition of any kind — box plots and rank correlations only. This thesis's decomposition is statistically more precise than the largest existing calibration benchmark. |
| **One Model, Many Behaviors** (WACV 2026, arXiv:2601.10836) | A genuine three-way ANOVA on OOD-AUROC, at much larger scale (56 models × 21 detectors × 8 OOD sets), with a real per-observation residual (10.6%). | Closest methodological precedent, and the reason the per-seed recomputation above was performed. It does not cover PEFT, frozen backbones, few-shot episodic evaluation, or calibration as an outcome. **Cite it proactively.** |
| **Guo et al. 2017** (arXiv:1706.04599); **Minderer et al. 2021** (arXiv:2106.07998) | Established that calibration and accuracy are governed by different factors. | The *qualitative pattern* here is not new. Only the formal decomposition and the PEFT/few-shot setting are. Present as re-confirmation, not discovery. |
| **Sensoy et al. 2018** (arXiv:1806.01768) | The original evidential deep learning paper; frames negligible inference overhead as EDL's selling point over MC-dropout and ensembles. | **The inference-cost claim is not novel.** Present it as rigorous measurement in this specific setting, never as a finding. |

### 3.3 Novelty assessment — **[PARTLY NOVEL]**

- **Novel:** the balanced five-axis factorial with a formal eta-squared decomposition of accuracy, ECE, and
  AUROC in a PEFT / few-shot setting. No matching methodology found in this regime.
- **Novel:** the correlation-collapse-under-control result (ECE and AUROC become orthogonal once head
  interpretation is held fixed) was **not found anywhere else**. This is the headline addition.
- **Not novel:** the qualitative pattern that calibration and accuracy have different drivers (Guo et al.
  2017), and the negligible inference cost of evidential heads (Sensoy et al. 2018).
- **Honest limit:** main effects only; interactions are not modelled. Conclusions are relative contributions
  under one fixed hyperparameter recipe.

---

## 4. RQ2 — Training objective versus scoring rule

> **Is out-of-distribution detection performance attributable to the training objective under which the
> model was adapted, or to the scoring rule by which its outputs are read — and can the two be separated
> experimentally?**

### 4.1 The confound being dissolved

The main grid scores evidential runs with vacuity only, and softmax runs with MSP / TS-MSP / energy only.
Every comparison in the original results therefore varies objective and score *together*. When the grid
reports "energy beats vacuity in about 70% of cells," it is not possible to tell whether energy is a better
score or whether softmax training produces better-separated features. The fix computes all four scores on
all runs, yielding a clean **2 objectives × 4 scores** factorial.

### 4.2 Answer

**The score dominates the objective, decisively.** Eta-squared over the full 2×4 factorial
(n = 792 comparisons per pool):

| Pool | Scoring rule | Training objective | Ratio |
|---|---:|---:|---:|
| Far-OOD | **43.7%** | 0.27% | **163×** |
| Near-OOD | **13.0%** | 0.60% | **22×** |

The raw means make it concrete: the energy score achieves 0.911 AUROC on evidential-trained logits and
0.929 on softmax-trained logits (far-OOD) — near-identical performance from the *same score* regardless of
which objective produced the logits — while MSP remains around 0.79 under either objective. Both pools
return `dominant = "score"`.

**Coverage caveat.** Computed on 99 of 120 recoverable checkpoints. All included cells are matched
evidential-versus-softmax pairs. A regression guard confirmed that the code change adding the missing cross
terms did not perturb the original results: 99/99 cells match the committed metrics exactly.

### 4.3 Nearest prior work, and why it is not this

| Prior work | What it does | Why it is not this result |
|---|---|---|
| **A Systematic Comparison of Training Objectives for OOD Detection** (2026, arXiv:2603.07571) | Names this exact gap as an open limitation: objective and scoring rule "are not fully factorized, since each objective is evaluated with the confidence measure most natural to its output space." | Quote verified against the paper's raw text. It compares Cross-Entropy / Triplet / Prototype / Average-Precision losses — **no evidential or Dirichlet objective at all** — and does not cover frozen backbones, few-shot episodic evaluation, or PEFT. A strong motivating analogy, **not** existing coverage. |
| **One Model, Many Behaviors** (WACV 2026, arXiv:2601.10836) | Runs a comparable training-method x scoring-method ANOVA at much larger scale, in the same year. | **This is the most important paper to cite up front.** The ANOVA technique is therefore not unprecedented. What it does not do is cross scores onto Dirichlet-parameterised models. |
| **arXiv:2605.22746** (theory) | Proves that softmax is a mathematical special case of an evidential classifier. | **This pre-explains why the objective barely matters.** Cite proactively — do not leave it for a committee member to raise. It strengthens the result's interpretation rather than undermining it. |
| **OpenOOD / OpenOOD-1.5; Bitterwolf et al. 2022; Liu et al. 2020** (arXiv:2010.03759) | Establish that post-hoc scores often beat training-method changes, and that energy beats MSP. | The qualitative pattern is old news. Confirmation at scale in a new regime, not discovery. |
| **arXiv:2605.06382** | Critiques vacuity-based AUROC as inflated when ID and OOD class cardinality differ. | Checked and ruled out as a confound here: every OOD sample is scored under the same 5-way episode head, so K_ID = K_OOD = 5 always. Stated explicitly in Appendix B. |

### 4.4 Novelty assessment — **[PARTLY NOVEL]**

- **Not novel:** the ANOVA methodology (concurrent WACV 2026 work at larger scale), and the qualitative
  finding that post-hoc scores beat training-method changes.
- **Novel:** cross-applying the energy score to Dirichlet-parameterised logits and finding near-equivalence
  to softmax-trained energy. **No precedent found anywhere.** That evidential-specific angle — not the
  ANOVA technique — is the surviving contribution.
- **Defence framing:** present as *confirmation and precise quantification in a new regime*, with the
  concurrent work and the theory paper cited up front. Under a defence bar this is a strength: the result
  is unambiguous and the prior art is acknowledged rather than missed.

---

## 5. RQ3 — Architecture, budget, or backbone

> **Within the adapter axis, are accuracy and calibration governed by the same property of the adapter, or
> does accuracy follow adapter architecture while calibration follows trainable-parameter budget?**

**Relation to RQ1 — state this first, or it looks like a contradiction.** RQ1 finds the adapter axis
explains only 0.63% of calibration *variance*. RQ3 asks a different question: conditional on comparing two
adapters directly, which property predicts the winner. An axis can contribute little total variance while
exhibiting a highly consistent direction. **RQ3 is a claim about direction, not magnitude.**

### 5.1 First evidence — 16 matched comparisons

Bottleneck versus LoRA, across 2 datasets × 2 shot regimes × 2 backbones × 2 head interpretations, with
head interpretation held fixed within each pair. Parameter counts are as given in Section 2.4.

| Dataset | Shots | Backbone | Head | Larger arm | ECE btl | ECE LoRA | Better calibrated |
|---|---|---|---|---|---:|---:|---|
| CIFAR-FS | 1 | MobileNetV3-S | evid. | LoRA | 0.2540 | **0.2156** | LoRA |
| CIFAR-FS | 1 | MobileNetV3-S | softmax | LoRA | 0.0499 | **0.0287** | LoRA |
| CIFAR-FS | 1 | ResNet-18 | evid. | bottleneck | **0.2765** | 0.2970 | bottleneck |
| CIFAR-FS | 1 | ResNet-18 | softmax | bottleneck | **0.0560** | 0.0969 | bottleneck |
| CIFAR-FS | 5 | MobileNetV3-S | evid. | LoRA | 0.3043 | **0.2879** | LoRA |
| CIFAR-FS | 5 | MobileNetV3-S | softmax | LoRA | 0.0703 | **0.0616** | LoRA |
| CIFAR-FS | 5 | ResNet-18 | evid. | bottleneck | **0.3010** | 0.3294 | bottleneck |
| CIFAR-FS | 5 | ResNet-18 | softmax | bottleneck | **0.0670** | 0.1016 | bottleneck |
| MiniImageNet | 1 | MobileNetV3-S | evid. | LoRA | 0.2534 | **0.2194** | LoRA |
| MiniImageNet | 1 | MobileNetV3-S | softmax | LoRA | 0.0650 | **0.0242** | LoRA |
| MiniImageNet | 1 | ResNet-18 | evid. | bottleneck | **0.3073** | 0.3708 | bottleneck |
| MiniImageNet | 1 | ResNet-18 | softmax | bottleneck | **0.1012** | 0.1748 | bottleneck |
| MiniImageNet | 5 | MobileNetV3-S | evid. | LoRA | 0.3177 | **0.3053** | LoRA |
| MiniImageNet | 5 | MobileNetV3-S | softmax | LoRA | 0.1107 | **0.0955** | LoRA |
| MiniImageNet | 5 | ResNet-18 | evid. | bottleneck | **0.2938** | 0.4049 | bottleneck |
| MiniImageNet | 5 | ResNet-18 | softmax | bottleneck | **0.0850** | 0.1930 | bottleneck |

**Summary across the same 16 pairs:**

| Outcome | Winner is the bottleneck architecture | Winner is the larger-budget arm |
|---|---:|---:|
| Accuracy | **16 / 16** | 8 / 16 |
| OOD AUROC (near, native score) | **16 / 16** | 8 / 16 |
| OOD AUROC (TinyImageNet near) | **16 / 16** | 8 / 16 |
| OOD AUROC (SVHN far) | 11 / 16 | 7 / 16 |
| ECE (pooled) | 8 / 16 | **16 / 16** |

**Stated so it cannot be over-read.** The "8/16" entries are *arithmetically implied*, not independent
evidence: because the budget ordering reverses between backbones, any outcome perfectly consistent with
architecture is necessarily 8/16 on budget, and vice versa. The real content is two directly observed facts:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both
   backbones despite holding 2.58× *more* parameters on one and 1.55× *fewer* on the other.
2. **The ECE winner does change, exactly in step with the budget ordering.** LoRA is better calibrated in
   all 8 MobileNetV3-Small pairs (where LoRA is larger); bottleneck in all 8 ResNet-18 pairs.

Each direction is a 16/16 sign consistency, two-sided p ≈ 3.05 × 10⁻⁵ under a null of random direction.
**Magnitudes are weaker than the directions:** |ΔECE| exceeds twice the pooled across-seed standard
deviation in only 10/16 pairs — all 8 MiniImageNet pairs, but only 2 of 8 CIFAR-FS pairs.

**Far-OOD is reported as a non-finding, not excluded post hoc.** Separation is near-chance under both
accounts (11/16 architecture, 7/16 budget), and RQ1 attributes far-OOD predominantly to head interpretation
(39.0% head versus 1.7% adapter).

### 5.2 The identification problem this created

The budget ordering reverses **with the backbone**. Budget ordering and backbone identity are therefore
perfectly collinear across all 16 pairs, and two explanations predict the identical pattern:

- **H3.2 (budget)** — whichever adapter has more trainable parameters wins calibration.
- **H3.2-alt (backbone-intrinsic)** — something about ResNet-18 versus MobileNetV3-Small itself drives it.

No number of extra seeds, episodes, or datasets separates these, because the confound is structural rather
than statistical. On the 16-pair evidence alone, the budget account was **preferred on parsimony, never
demonstrated**. Publishing it as a causal claim would have been unsound.

### 5.3 The matched-budget experiment — the answer

**Design.** Build both adapter architectures at the *same* trainable-parameter budget within each backbone,
which breaks the collinearity. MiniImageNet 5-shot (chosen because RQ3's ECE effects clear 2 sigma in 8/8
MiniImageNet pairs but only 2/8 CIFAR-FS pairs — a null on CIFAR-FS would have been uninterpretable), 3
seeds, both head interpretations **trained separately**. 48 runs. **Residual budget mismatch at most 3.10%,
against 55–158% in the unmatched comparison.**

**The decision rule was pre-registered** — thresholds, the power check, and an explicit "inconclusive"
outcome all committed to writing before any deciding run existed.

| Backbone / level | Head | btl: rank -> params | LoRA: rank -> params | mism. | ECE btl | ECE LoRA | ΔECE | > 2 sigma |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ResNet-18 / L | evid. | 6 -> 12,506 | 16 -> 12,290 | 1.76% | **0.2973** | 0.4049 | +0.1076 | yes |
| ResNet-18 / L | softmax | 6 -> 12,504 | 16 -> 12,288 | 1.76% | **0.0827** | 0.1930 | +0.1103 | yes |
| ResNet-18 / H | evid. | 16 -> 31,746 | 41 -> 31,490 | 0.81% | **0.2938** | 0.4109 | +0.1170 | yes |
| ResNet-18 / H | softmax | 16 -> 31,744 | 41 -> 31,488 | 0.81% | **0.0850** | 0.1728 | +0.0878 | yes |
| MobileNetV3-S / L | evid. | 16 -> 6,930 | 10 -> 6,722 | 3.09% | 0.3177 | **0.3121** | −0.0056 | yes |
| MobileNetV3-S / L | softmax | 16 -> 6,928 | 10 -> 6,720 | 3.10% | **0.1107** | 0.1140 | +0.0033 | no |
| MobileNetV3-S / H | evid. | 22 -> 9,450 | 14 -> 9,410 | 0.43% | 0.3165 | **0.3097** | −0.0068 | yes |
| MobileNetV3-S / H | softmax | 22 -> 9,448 | 14 -> 9,408 | 0.43% | 0.1137 | **0.0983** | −0.0153 | yes |

*Every delta is LoRA minus bottleneck, matching the sign convention of the unmatched table. "L" and "H" are
the two matched budget levels within each backbone. The per-cell 2-sigma thresholds are given in the
adjudication table below.*

**Adjudication** (the decision unit is a backbone × head cell, averaging its two budget levels — 4 cells):

| Cell | ΔECE unmatched | ΔECE matched | collapse ratio | 2 sigma | rule fired |
|---|---:|---:|---:|---:|---|
| ResNet-18 / evidential | +0.1111 | **+0.1123** | 1.01 | 0.0353 | backbone-intrinsic |
| ResNet-18 / softmax | +0.1080 | **+0.0991** | 0.92 | 0.0120 | backbone-intrinsic |
| MobileNetV3-S / evidential | −0.0124 | **−0.0062** | 0.50 | 0.0016 | backbone-intrinsic |
| MobileNetV3-S / softmax | −0.0152 | −0.0060 | 0.61 | 0.0147 | none (not > 2 sigma) |

**Verdict: `backbone_intrinsic` (H3.2-alt), 3 of 4 cells against a pre-registered threshold of 3.**
H3.2 (budget) fired in **0 of 4**. H3.1 (one architecture better calibrated regardless) fired in **0 of 4**,
because the matched ΔECE sign remains positive on ResNet-18 and negative on MobileNetV3-Small.

**What this does and does not say.**

1. **Budget is not the mechanism.** On ResNet-18 the calibration gap does not move at all when budget is
   equalised (collapse ratios 1.01 and 0.92 — the matched gap is as large as, or larger than, the unmatched
   one). A gap caused by a budget difference should have vanished along with it.
2. **Budget is not irrelevant either.** On MobileNetV3-Small the matched gaps are roughly half the unmatched
   ones (0.50, 0.61). Budget modulates the *magnitude*; the backbone determines the *sign*.
3. **The two directions are not symmetric.** ResNet-18's matched |ΔECE| is about 0.10–0.12; MobileNetV3's is
   about 0.006 — an order of magnitude smaller. Real and statistically resolvable, but they must not be
   presented as equal halves of one phenomenon.
4. **The mechanism is not identified.** "Backbone-intrinsic" names where the evidence points; it is not an
   explanation. Depth, width, normalisation placement, the inverted-residual block, and feature-norm scale
   at the adapter sites all differ between the two backbones simultaneously, and two backbones cannot
   separate them. **This is the honest remaining limit of RQ3.** Resolving it needs more *backbones*, not
   more seeds.

**Secondary outcomes — the architecture half of RQ3, which was genuinely at risk.** Had the accuracy gap
also collapsed at matched budget, accuracy would have been tracking budget too and the dissociation would
have weakened substantially. It did not:

| Outcome at matched budget | Bottleneck wins | Beyond 2 sigma | Cells whose gap collapsed |
|---|---:|---:|---|
| Accuracy | **8 / 8** | 8 / 8 | none (ratios 0.68–1.22) |
| Near-OOD AUROC | **8 / 8** | 8 / 8 | none (ratios 0.61–1.19) |

**Harness validation.** All 48 runs completed with no errors. The instantiated trainable-parameter count
matched the intended value for all 16 arms. A control guard confirmed that only permitted keys differ
across all 48 *merged* configurations — learning rate, epochs, patience, KL schedule, dataset, episode
files, n-way and k-shot are identical everywhere. The 18 re-runs of existing grid arms reproduced the
committed earlier metrics **exactly** (maximum absolute difference 0.0 across 12–13 metric keys per cell).

### 5.4 The restated claim

> **Accuracy and near-OOD ranking follow adapter architecture, invariant to budget — now tested *at* matched
> budget, not merely at a reversed one. Calibration follows neither architecture nor budget: it follows the
> backbone.**

This is narrower than "calibration follows the parameter budget," and it is the claim the data supports. It
is also arguably more interesting: the dissociation is no longer architecture-versus-size but
adapter-property-versus-backbone-property.

### 5.5 Nearest prior work, and why it is not this

| Prior work | What it does | Why it is not this result |
|---|---|---|
| **CP tensor adapters** (2026, arXiv:2606.00428) | A controlled study of the accuracy–budget curve with finer parameter granularity; explicitly uses matched-budget runs. | Verified: accuracy-only, **no calibration measurement at all**, single backbone, no reversal design. The closest work on matched-budget *methodology*, but it never asks the calibration question. |
| **LoRA-Ensemble** (arXiv:2405.14438) | Sweeps LoRA rank under a frozen ViT backbone and reports ECE degrading at high rank. Also reports that its advantage over explicit ensembles grows with backbone size. | Budget-versus-ECE **within one architecture** — never architecture-versus-architecture. Its backbone finding concerns backbone *size* changing an ensembling method's advantage, a different question entirely. |
| **On Fairness of Low-Rank Adaptation** (arXiv:2405.17512) | Multi-backbone calibration analysis. | Compares LoRA versus full fine-tuning, not one adapter architecture against another at matched budget. |
| **Robust Calibration of Large Vision-Language Adapters** (ECCV 2024) | CLIP adapters, prompt learning, and test-time adaptation degrade calibration under distribution shift; identifies growing logit ranges as the cause. Uses ViT-B/16 and ResNet-50. | Read and confirmed distinct. Different mechanism (logit range), different setting (CLIP zero-shot adaptation, not frozen-backbone few-shot PEFT), and no budget-matched design. |
| **Be Confident in What You Know: Bayesian PEFT of Vision Foundation Models** (NeurIPS 2024) | PEFT-adapted models are severely *underconfident*; fixes this with base-rate adjustment and an evidential ensemble. | Genuinely distinct — ViT foundation models, no architecture-versus-budget dissociation. **But close enough in name ("Bayesian-PEFT") that the thesis must cite it and include an explicit differentiation paragraph.** |
| **Conv-Adapter** (arXiv:2208.07463) | Parameter-efficient transfer learning designed specifically for ConvNets. | No calibration measurement. |
| *(False friend, worth pre-empting)* | Several sources report "ResNet calibrates better than MobileNet." | This is from the **quantization** literature, where "calibration" means calibrating quantization ranges using a calibration dataset. A different meaning of the word, unrelated to confidence calibration or ECE. |

### 5.6 Novelty assessment — **[NOVEL]** — the strongest contribution

- **No prior work found**, in vision or NLP PEFT, that dissociates adapter architecture from adapter
  parameter budget by outcome metric.
- The standard convention in PEFT research is to *equalize* budgets for a fair comparison. **Deliberately
  exploiting a reversed budget ordering as an identification strategy appears to have no precedent.**
- **No prior work found** showing that the calibration ordering between two PEFT adapter architectures at
  matched budget is determined by backbone identity rather than by parameter count.
- **No direct contradiction found** — nothing in the literature shows calibration tracking architecture, or
  accuracy tracking budget, in a way that conflicts with this.
- Verified across three independent internal passes plus 27 fresh targeted searches on 2026-08-27, including
  direct re-checks of the three closest-named papers above.
- **Methodological strength worth emphasising:** the pre-registration. The rule was fixed before the data
  existed, and the verdict went against the project's own leading hypothesis. That is the strongest form
  this result could have taken.
- **Honest limit:** two backbones establish *that* the backbone matters, not *which property* of it does.
  State this as a limit; do not narrate a plausible-sounding mechanism.

---

## 6. RQ4 — Post-hoc remediation of the calibration deficit

> **Can the calibration of an evidential prototype head be improved post-hoc by refitting only the two
> parameters of its evidence-affine transform on held-out validation episodes, and is the head's
> out-of-distribution ranking preserved under that refitting?**

**Relation to RQ1.** RQ1 identifies head interpretation as the dominant source of calibration variance, and
the evidential head as the poorly-calibrated arm. RQ4 tests whether that deficit is *intrinsic to the head*
or merely an artefact of a fixed default parameterisation. It is the remediation half of the thesis's
overarching question.

**Why the answer is not obvious.** The transform is monotone in each individual logit, which might suggest
that ranking preservation follows automatically. But vacuity is `K / sum(alpha)`, a function of *all K
logits jointly*, so a per-logit monotone transform does **not** guarantee preservation of the induced
sample ordering. Preservation is an empirical question, not an analytic consequence.

**Operationalisation.** Parameters fitted **exclusively on the validation episode split**, evaluated on the
frozen 600-episode test split.

### 6.1 Answer

- **ECE improved in 48/48 evidential cells (100%)**, by a mean of **−0.137 absolute** — many cells drop from
  0.25–0.44 down to 0.10–0.31.
- Refit values cluster around scale 7–14, against the frozen default of (2, −6), implying the default sat
  well off the optimum throughout.
- **OOD-AUROC preserved** (delta at least −0.005) in **150/192 comparisons (78%)**; mean delta-AUROC ≈ +0.004
  (essentially flat).
- Reordering **does** occur — confirming the affine is not automatically rank-preserving for vacuity — but
  the **worst-case** cell across every pool still had Spearman rho = 0.921. Ranking degrades gracefully
  rather than collapsing.

**Honest caveat.** AUROC dropped by more than 0.5 percentage points in the remaining 22% of comparisons
(worst single-pool drops around −0.03). State this as *"survives in the large majority of cases, with a
measured minority exception"* — never as *"always."*

**A substantive correction of the project's own earlier finding.** The (2, −6) defaults were not arbitrary:
they were tuned once on validation episodes in an earlier phase, and that sweep found a *flat* ECE surface
(approximately 0.285–0.296), which is why they were frozen. The result above shows that the earlier sweep
was searching the wrong parameter. This is a correction with 48 cells behind it, not a configuration typo.

### 6.2 Nearest prior work, and why it is not this

| Prior work | What it does | Why it is not this result |
|---|---|---|
| **Kull et al. 2019** (NeurIPS), "Dirichlet calibration" | A generic calibration map applicable to any softmax classifier. | Verified as a **different use of the word "Dirichlet"** — a calibration map over softmax outputs, unrelated to evidential deep learning's Dirichlet evidence. Does not pre-empt this claim, but the name collision must be addressed explicitly in the thesis. |
| **Guo et al. 2017** (arXiv:1706.04599) | Temperature scaling; reports ECE drops of comparable or larger size (CIFAR-100 / ResNet-110: 16.53% to 1.26%). | **The magnitude of this thesis's improvement is not remarkable.** Present as confirmation, not as an impressive number. Note also that temperature scaling divides logits by one scalar before a softmax; this refits scale *and* bias inside a softplus producing Dirichlet evidence — a different transform in a different output space. |
| **Bayesian Evidential Learning** (arXiv:2207.13137) | A directly analogous few-shot evidential paper; already reports −11 to −15 percentage-point ECE drops. | Same ballpark as this thesis's −13.7. The *size* of the effect is not the contribution. |
| **Accuracy-preserving post-hoc calibration via invertible logit transforms** (arXiv:2608.10372) | Post-hoc calibration preserving accuracy. | Confirmed to apply only to standard softmax classifiers; contains no evidential or vacuity content. |
| **Density-informed EDL recalibration** (arXiv:2602.01477); **Invascal** (arXiv:2606.00069) | Recalibration approaches for evidential models. | Both are architectural or training-time interventions, **not post-hoc two-parameter refits.** |
| **arXiv:2510.08938; arXiv:2410.00393** | Document that EDL default hyperparameters are frequently poorly tuned. | "The defaults were bad" is already a known complaint in the EDL literature. The contribution must not be framed as discovering this. |

### 6.3 Novelty assessment — **[PARTLY NOVEL]** — the narrowest claim, scope it carefully

- **The claim is NOT "first post-hoc calibration."** Post-hoc calibration is a mature field.
- **The claim is:** a two-parameter recalibration of an *evidential prototype head* in a frozen-backbone
  few-shot regime, with OOD-ranking preservation **measured rather than assumed**.
- **Novel:** no paper found proving or disproving ranking preservation for a sum-of-affine-transformed-logits
  construction like vacuity. The mechanism (the evidence-affine) and the systematic 48-cell / 192-comparison
  quantification are the genuinely new parts.
- **Not novel:** the magnitude of the improvement, the fact that ranking is imperfectly preserved under a
  non-scalar transform (textbook calibration theory), and the observation that EDL defaults are poorly tuned.
- **Required framing:** *"we confirm and precisely quantify a known class of calibration/ranking trade-off
  for a new mechanism"* — never *"we discovered that refitting helps."*

**Why this question earns its place despite being the least novel.** It is the second clause of the thesis's
overarching question. Without it, RQ1's finding — that the head axis owns 83% of calibration variance and
the evidential head is the bad arm — is a diagnosis with no treatment, and the thesis's titular Bayesian
method loses on every axis with nothing done about it. RQ4 establishes that the calibration deficit is *not
intrinsic to the evidential head*, which converts a flat negative result into a diagnosed and partially
remediated one. It is also low-risk: it makes a modest claim that is straightforwardly true in 48 of 48
cells, so the only way it goes badly is if it is oversold.

---

## 7. A tested hypothesis that did not survive

Retained for completeness and because the correction is itself informative. **Not presented as a
contribution.**

**Original hypothesis.** Calibration error reaches an optimum at an intermediate trainable-parameter budget,
and that budget differs from the accuracy-optimal one.

**The superseded evidence** (CIFAR-FS × ResNet-18 — ECE fell then rose in 4 of 4 curves):

| Head, shots | 0–2 params (Linear-Probe) | 12.3k (LoRA) | 31.7k (bottleneck) | 11.18M (Full-FT) |
|---|---:|---:|---:|---:|
| Softmax, 1-shot | 0.2818 | 0.0969 | **0.0560** | 0.0854 |
| Softmax, 5-shot | 0.4476 | 0.1016 | **0.0670** | 0.0728 |
| Evidential, 1-shot | 0.4397 | 0.2970 | **0.2765** | 0.3207 |
| Evidential, 5-shot | 0.6234 | 0.3294 | **0.3010** | 0.3383 |

This confounded budget with adapter type *and* with which weights are trainable at all.

**Controlled result.** A rank sweep with architecture, backbone, dataset and shots all held fixed and only
the adapter rank moving — ranks {1, 2, 4, 8, 16, 32, 64} × 3 seeds, 21 of 21 runs completed. **No U-shape.**
Evidential ECE is lowest at rank 1 (0.291) and drifts noisily upward to rank 64 (0.309); softmax ECE moves
the *opposite* way, 0.093 down to 0.081. The accuracy/calibration mismatch survives (best ECE at rank 1,
best accuracy at rank 64) but via two roughly monotonic opposing trends, not an interior optimum.

**Required wording.** Three seeds × seven ranks is plausibly underpowered to exclude a subtle U-shape.
Always write **"no interior optimum observed in the tested range."**

### 7.1 A published partial contradiction, and the response

**LoRA-Ensemble** (arXiv:2405.14438), verified directly from its PDF, reports an interior-optimum-like
reversal in single-network ECE on **CIFAR-100** — the same dataset family: *"at rank 32, the calibration of
a single network augmented with LoRA begins to deteriorate."* Their tested range was {1 … 32}; this thesis's
softmax-read curve shows the **opposite** pattern in the overlapping region. **This must be addressed
explicitly in the thesis text — a supervisor or examiner who knows this paper will ask.**

Three legitimate distinguishing factors:

1. **Adaptation target.** LoRA-Ensemble injects rank into multi-head self-attention projections in a Vision
   Transformer; this sweep uses bottleneck-parallel adapters on 1×1 convolutional channels in a CNN.
2. **Head design.** They use a trainable linear head over 100 classes; this thesis uses a parameter-free
   nearest-centroid prototype head, whose similarity-based logits are intrinsically bounded in a way a free
   linear layer's are not — a plausible reason high-rank overconfidence would appear later here.
3. **Relative capacity.** Rank 64 here is roughly 63k parameters against an 11.7M frozen backbone (0.56%);
   the comparable point in their setup injects far more of the model's representational capacity.

**One sub-argument must be cut, not softened.** An earlier draft also argued the turning point "exists, just
further out," citing Full-FT's ECE (0.0854) against the 31.7k bottleneck's (0.0560). That reuses the
superseded four-point table above — the exact confounded comparison the sweep was built to eliminate.
Full-FT is not "more rank on the same architecture"; it is a different architecture entirely. The three
factors above stand without it.

**Why this still earns a place in the thesis.** "The interior-optimum story does not survive when
architecture is held fixed" is itself a finding, and it anticipated the RQ3 result: budget's effect on
calibration is direction-dependent even within one fixed architecture, which is exactly what a weak and
non-causal budget effect looks like. The matched-budget experiment later confirmed that reading.

---

## Appendix A — the original proposal questions

The proposal posed four *comparison* questions ("does A beat B"). **All four were answered and all results
are retained.** A literature review conducted after the grid completed found that each had close precedent.
The four questions above reformulate the same completed experiments as *attribution* and *remediation*
questions, which the balanced factorial design supports and for which precedent is substantially thinner.
**No experiment was discarded.**

| # | Question | Result obtained | Prior work that pre-empts it |
|---|---|---|---|
| Orig-RQ1 | Adapter placement: serial versus parallel | Parallel wins **16/16** (+2.1 to +8.3 pp). Strict Pareto win on MobileNetV3-Small. At 5-shot a 31.7k-parameter adapter beats Full-FT (91.44% versus 90.47%) at 0.28% of the parameter cost. | **TSA** (CVPR 2022, arXiv:2107.00358) — frozen ResNet-18, 600 episodic tasks, parameter-free nearest-centroid head, finds parallel wins "in almost all cases." Same backbone, same protocol, same answer. |
| Orig-RQ2 | Does an evidential head calibrate better than softmax under a tiny budget? | **No — 0/20.** Evidential ECE worse than softmax by 1.4×–9.1×, and worse than temperature-scaled softmax by 5.3×–51×. Accuracy does not compensate (7/20). | Well studied. **BEL** (arXiv:2207.13137) and **BayesAdapter** (arXiv:2412.09718) both report Bayesian calibration *improving*; this negative reads as a boundary case of a known effect. |
| Orig-RQ3 | Does a Bayesian prior improve near-OOD detection in low-data regimes? | Vacuity beats MSP and TS-MSP in roughly 37–38 of 40 cells (mean +0.05 to +0.13 AUROC), and the advantage **grows as shots shrink** — the predicted low-data trend holds. But training-free **energy** beats vacuity in about 70% of comparisons, reversing an earlier single-configuration finding. | Vacuity beating MSP is the standard EDL claim; energy beating MSP is **Liu et al. 2020** (arXiv:2010.03759). Confirmation at scale rather than discovery. |
| Orig-RQ4 | Latency versus uncertainty-quality Pareto frontier | Backbone drives latency (5.12×); adapter choice does not (3.9%). Evidential heads are effectively free at inference (1.29%, below the 5.91% noise floor). Recommended edge operating point: MobileNetV3-Small + parallel bottleneck + evidential (11.86 ms, 6,930 parameters). | Pareto reporting is a standard presentational device, not a research finding. |

**Note on the Orig-RQ3 correction.** An earlier single-configuration result suggested evidential vacuity was
roughly on par with the energy score. At grid scale this did **not** generalise: vacuity beats energy in only
10/40 far-OOD and 14/40 near-OOD matched comparisons. The defensible claim is narrower — vacuity is a
substantially better OOD ranker than softmax-probability scores (37–38 of 40 wins), but a well-chosen
logit-space score still beats it. This correction is reported wherever the claim appears.

---

## Appendix B — limitations applying to all results

1. **ImageNet-pretraining overlap.** The backbones are ImageNet-pretrained, whereas the standard
   CIFAR-FS / MiniImageNet protocol trains from scratch on base classes. MiniImageNet classes *are* ImageNet
   classes, so "novel" test classes were seen during pretraining. **Absolute accuracies are therefore not
   comparable to from-scratch few-shot literature.** The research questions concern relative differences
   between design choices, which mitigates but does not eliminate this.
2. **One frozen hyperparameter recipe** across all 40 configurations, tuned once on a single configuration.
   The grid answers "how do these axes compare under one fixed recipe," not "what is each cell's best
   achievable number."
3. **Three seeds**, and effectively one for the Full-FT and Linear-Probe baselines whose seed axis is inert
   by construction.
4. **A known vacuity evaluation artefact does not apply here, and this should be stated explicitly rather
   than left implicit.** Rethinking Vacuity for OOD Detection in EDL (arXiv:2605.06382) shows that
   vacuity-based AUROC can be inflated when class cardinality differs between in-distribution and OOD
   evaluation. In this protocol every OOD sample is scored under the same 5-way episode head, so
   K_ID = K_OOD = 5 always.
5. **Novelty claims rest on a non-exhaustive search.** See Appendix C. Absence of found prior art is not
   proof of absence, and this document does not claim otherwise.
6. **RQ3's mechanism is unidentified.** Two backbones establish that backbone identity matters; they cannot
   establish which property of the backbone is responsible.
7. **One data point of 96 is missing** in RQ1's per-seed table (a single seed's TinyImageNet near-OOD AUROC
   for one cell). It does not change any conclusion, but it is recorded rather than glossed.

---

## Appendix C — how novelty was checked

Novelty was assessed across five independent passes. Each is recorded so the work does not have to be
repeated:

| Pass | Date | Scope |
|---|---|---|
| Initial literature review | 2026-08-21 | 12 targeted searches, 5 full paper fetches |
| Independent re-verification | 2026-08-23 | 5 parallel agents, 60+ searches, per-question verdicts |
| Adversarial stress-test | 2026-08-26 | Full-PDF reads of the closest competitors, seeking contradictions |
| Independent deep-research pass | post-2026-08-26 | Confirmed internal verdicts rather than overturning them |
| Post-experiment re-check | 2026-08-27 | 27 targeted searches against RQ3's revised claim specifically |

**What the independent passes concluded:** RQ3 "fully novel — strongest contribution"; RQ2 "novel, scoped to
the Dirichlet-energy cross-application"; RQ1 "methodologically novel," with a per-seed recomputation required
(since completed); RQ4 "conditionally novel — a systematic re-quantification"; the demoted hypothesis
"original claim disproven, reframe as a negative result." **No new prior-art contradictions surfaced in any
pass.**

**Two prior open items were closed by direct reading:** "Robust Calibration of Large Vision-Language
Adapters" (ECCV 2024) and the NeurIPS 2024 "Bayesian-PEFT" paper were both confirmed distinct.

### Outstanding writing tasks (not experiments)

- Write an explicit differentiation paragraph against the NeurIPS 2024 "Bayesian-PEFT" paper — the name
  similarity requires it.
- Cite as prior grounding wherever the corresponding claim appears: arXiv:2601.10836, arXiv:2605.22746,
  arXiv:2608.10372, Sensoy et al. 2018, Guo et al. 2017, Minderer et al. 2021.
- Address the LoRA-Ensemble partial contradiction in the text (Section 7.1 has the drafted response).

---

## Traceability

Every number in this document is transcribed from a committed result file, never estimated:

| Content | Source |
|---|---|
| The 120-run grid; RQ1 and RQ3's 16-pair evidence | `results/mvt_results.json` |
| RQ2 and RQ4 results | `results/rq_factorial/`, `results/rq_summary.json` |
| RQ3 matched-budget adjudication, guards, acceptance criteria | `results/rq3_matched/verdict.json` (48 per-cell JSONs alongside) |
| RQ3 pre-registration (design, formulas, decision rule, thresholds) | `docs/RQ3_MATCHED_BUDGET_PLAN.md` |
| Demoted hypothesis rank sweep | `results/rq5/`, `results/rq5_rank_sweep.png` |
| Full grid tables (accuracy, ECE, Brier, AUROC, FPR@95, Pareto) | `docs/RESULTS_MASTER.md` |
| Execution write-up with all deviations and caveats | `step_writeups/rq3_matched_budget.txt` |
| Canonical status tracker and decisions log | `progress.txt` |

**Reproducibility invariant.** Running the same configuration twice produces a byte-identical metrics file.
The RQ3 matched-budget run re-trained 18 cells that already existed in the main grid; all 18 reproduced the
committed values exactly (maximum absolute difference 0.0), confirming the two harnesses measure the same
quantity.
