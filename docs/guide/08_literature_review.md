# 8. Literature Review and Research Gap (thesis Chapter 2 draft)

Source material for **Chapter 2**. It reviews the seven areas this thesis draws on, then puts them together
into the research gap that motivates RQ1–RQ4.

**Where each claim comes from.** The repo's rule (`thesis_implementation_instructions.txt` §6) is that
claims about papers must come from what was actually read, never from memory. Every paper below carries a
source tag:

| Tag | Meaning | How much to trust it |
|---|---|---|
| **[S]** | Summarised from the PDF in `PAPER SUMMARIES/*.txt` | Claims here are safe to paraphrase |
| **[R]** | Read or checked during the novelty passes; the claim is as recorded in `docs/RQ_SUPERVISOR_REPORT.md`, `docs/RQ_RESULTS_SUMMARY.md` or `docs/DEFENCE_BRIEF.md` | Use only the claim as recorded; re-open the paper before adding detail |
| **[T]** | Title or metadata only (`docs/refs.bib`); no summary exists | Cite for existence or topic only. Read it before saying more |
| **†** | Not yet in `docs/refs.bib` | Add the entry before citing |

Bracketed keys such as `[hu2022lora]` are `docs/refs.bib` keys.

---

## 8.1 Chapter structure

1. Lightweight CNN backbones (§8.2)
2. Few-shot image classification (§8.3)
3. Parameter-efficient fine-tuning (§8.4)
4. Confidence calibration (§8.5)
5. Predictive uncertainty and Bayesian PEFT (§8.6)
6. Evidential deep learning (§8.7)
7. Out-of-distribution detection (§8.8)
8. Synthesis and research gap (§8.9)
9. Citation hygiene (§8.10)

---

## 8.2 Lightweight CNN backbones

**Residual networks.** ResNet-18 [he2016resnet] **[T]** is the primary backbone: 11.7M parameters, a 512-d
pooled feature, four stages of residual BasicBlocks with 64/128/256/512 channels. It is also the reference
backbone of OpenOOD's CIFAR benchmarks [yang2022openood, zhang2024openoodv15] **[S]** and of TSA, the
closest few-shot adapter study [li2022tsa] **[R]**.

**MobileNetV3** [howard2019mobilenetv3] **[S]** combines platform-aware architecture search (MnasNet-style)
with NetAdapt layer refinement and hand-designed changes:

- a cheap h-swish nonlinearity;
- squeeze-and-excite modules sized at 1/4 of the expansion channels;
- the last feature-expansion layer moved after global pooling.

Its building block is the MobileNetV2 inverted residual: 1×1 expand → depthwise K×K → SE → 1×1 linear
projection, with a residual connection when input and output shapes match. MobileNetV3-Small reaches about
67.4–67.5% ImageNet top-1 with 2.5M parameters and about 56–57M MAdds, and its latency was measured on Pixel
phones. **Relevance.** These inverted-residual blocks are the adapter insertion sites for the second
backbone. The summary gives no ResNet-18 top-1, so the thesis must not quote a ResNet-vs-MobileNet accuracy
gap from this source (`progress.txt`, Step 8 decision log).

**ConvNeXt** [liu2022convnext†] **[S]** shows that much of the Transformer advantage in vision comes from
training recipe and macro/micro design choices rather than attention. It modernises a ResNet-50 step by step
(patchify stem, depthwise 7×7 convolutions, inverted bottleneck, LayerNorm, GELU) and matches Swin at equal
FLOPs. The proposal named "ConvNeXt-Nano", but that variant is not defined in the paper (it comes from
timm), and it was never run. **Use:** background for "CNNs are not obsolete" only.

**Why a CNN in 2026.** The TinyML→TinyDL survey [somvanshi2025tinydl] **[R]** reports:

- microcontroller-class targets have 32–512 kB SRAM and under ~1 MB flash;
- CNNs (MobileNet, SqueezeNet, MCUNet) dominate that space;
- even memory-optimised attention costs ~180 ms on an STM32F746, against ~8–12 ms for CNN inference.

⚠️ Do **not** cite arXiv:2510.04794 [kaya2025cnnvit] for "CNNs stay competitive in few-shot
classification". That paper is about few-shot *geometric estimation* (`docs/CITATION_AUDIT.md` §7.3).

---

## 8.3 Few-shot image classification

### 8.3.1 The episodic protocol

**Matching Networks** [vinyals2016matching] **[S]** introduced two lasting ideas:

- **"Test and train conditions must match."** Training is organised into N-way K-shot *episodes* that mimic
  the test task.
- **The miniImageNet benchmark.** 100 classes, 600 images each, 84×84.

Classification is attention over cosine similarities between a query embedding and the support embeddings.

**Standard splits.** The thesis uses:

- the CIFAR-FS split of Bertinetto et al. [bertinetto2019r2d2] **[T]**, with 64/16/20 classes from CIFAR-100
  [krizhevsky2009cifar] **[T]**;
- the Ravi & Larochelle 64/16/20 split of miniImageNet [ravi2017optimization] **[T]**.

Both split files are frozen in `data/`. **Evaluation convention.** Mean accuracy over 600 test episodes with
95% confidence intervals, as used by A Closer Look and ProtoNet [chen2019closerlook, snell2017protonet]
**[S]**.

### 8.3.2 Metric-based and optimisation-based methods

**Prototypical Networks** [snell2017protonet] **[S]**:

- represent each class by the mean of its support embeddings, c_k = (1/|S_k|) Σ f(x_i);
- classify with a softmax over negative distances;
- use **squared Euclidean distance** deliberately, because it is a Bregman divergence: this makes the class
  mean the optimal prototype and the classifier linear in feature space. The paper reports Euclidean
  distance clearly outperforming cosine.

It trains with a higher "way" than at test time and matches train and test shot.

**MAML** [finn2017maml] **[S]** takes the optimisation-based route. It learns an initialisation from which a
few gradient steps solve a new task, using a bi-level inner/outer update. A first-order approximation
reaches nearly the same accuracy about 33% faster.

**A Closer Look at Few-shot Classification** [chen2019closerlook] **[S]** re-implemented the main methods in
one codebase and found:

- (a) deeper backbones shrink the gaps between methods;
- (b) a simple "Baseline++" — a frozen feature extractor with a cosine-similarity classifier — is
  competitive with meta-learners;
- (c) under domain shift, a plain fine-tuned baseline wins;
- (d) earlier baselines were underestimated because data augmentation was omitted.

**MetaOptNet** [lee2019metaoptnet] **[T/R]** supplies the classical comparison rows in
`docs/RESULTS_MASTER.md` §4.2.

### 8.3.3 Pretrained backbones and adapters for few-shot learning

**P>M>F** [hu2022pmf] **[R]** chains pretraining, meta-training and fine-tuning with ViT backbones, on the
same 5-way CIFAR-FS / MiniImageNet protocol. Its supplemental material notes that supervised ImageNet
pretraining "is only useful to check the upper bound performance" and reaches 99.8% on MiniImageNet 5-shot.
This is the reason this thesis treats its own absolute accuracies as inflated.

**Task-Specific Adapters (TSA)** [li2022tsa] **[R]** is the closest prior work on placement. It uses a
frozen ResNet-18, 600 sampled episodic tasks and a parameter-free nearest-centroid head, and finds residual
(parallel) adapter connections better than serial ones "in almost all cases". It reports no ECE and no OOD
AUROC, and its adapters (~175k–1.22M parameters) are far larger than this thesis's 6,928–31,746.

**FiT** [shysheya2023fit] **[R]** uses frozen CNN backbones with FiLM adapters as small as 11,648 parameters
and a ProtoNets head option. It evaluates on fixed downstream benchmarks, not disjoint-class episodes, and
compares neither adapter types nor placements. *(The authors are Shysheya et al., not Bateni et al. — see
the citation audit.)*

**CLIP-based recipes.** Tip-Adapter [zhang2022tipadapter], CLIP-Adapter [gao2025clipadapter] and CoOp
[zhou2022coop] **[R]** freeze a vision-language model and train a small adapter or prompt. Tip-Adapter
explicitly trains and tests on the *same* classes, unlike the disjoint-class episodic protocol. None reports
ECE or OOD AUROC.

**Takeaway for the thesis.** Freezing a pretrained backbone and training a small adapter for few-shot
classification is established. What is not established is measuring the *reliability* of that recipe on
lightweight CNNs.

---

## 8.4 Parameter-efficient fine-tuning (PEFT)

### 8.4.1 Taxonomy

The PEFT survey of Han et al. [han2024peftsurvey†] **[S]** sorts methods into four families:

- **additive** — adapters, soft prompts, (IA)³, SSF;
- **selective** — BitFit, masking;
- **reparameterised** — LoRA and its variants;
- **hybrid** — MAM Adapter, UniPELT.

The thesis's three PEFT methods therefore cover the first three families: bottleneck adapter (additive),
BitFit (selective, Step 5 only) and LoRA (reparameterised).

### 8.4.2 Adapters, LoRA and BitFit

- **Adapters** [houlsby2019adapters] **[S].** A bottleneck h ← h + f(h·W_down)·W_up is inserted into a frozen
  Transformer, with near-identity initialisation. With about 3% extra parameters per task it stays within
  about 0.4% of full fine-tuning on GLUE. Adapter tuning leans on later layers, and adapters add inference
  latency because they run in-line.
- **LoRA** [hu2022lora] **[S].** It hypothesises that the weight *update* is low-rank:
  W = W₀ + BA with B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, B initialised to zero, and the update scaled by α/r. The update
  merges into W₀, so LoRA adds no inference latency.
- **BitFit** [benzaken2022bitfit] **[S].** Only the bias terms train (~0.08–0.09% of parameters). It matches
  full fine-tuning on small and medium data, the advantage reverses as data grows, and a random subset of the
  same size does far worse.
- **Unified view** [he2022unified] **[S].** Adapters, prefix tuning and LoRA all learn a modification Δh.
  They differ along four design dimensions: functional form, modified representation, **insertion form
  (sequential vs parallel)** and composition function. Prefix tuning is approximately a parallel adapter.
  This paper provides the vocabulary for "serial vs parallel".

### 8.4.3 PEFT for convolutional networks

- **Conv-Adapter** [chen2024convadapter] **[S]** is the first systematic PEFT design for CNNs. It uses a
  depthwise-separable convolutional bottleneck with learnable per-channel scaling and tests four insertion
  schemes. Findings:
  - **parallel beats sequential** for classification;
  - **1×1/linear adapters "lose locality"**, the paper's main failure mode, and depthwise-separable K×K
    adapters work better;
  - few-shot gains are largest at 1–2 shots.

  *This thesis uses 1×1 bottlenecks anyway, so that placement is the only thing varied (see
  `09_methodology.md` §9.12). That is a known departure from Conv-Adapter's preferred form.*
- **ConvLoRA + AdaBN** [aleem2024convlora] **[S]** applies LoRA to the convolutional encoder of a U-Net for
  multi-target domain adaptation from only 10 target images. It cuts trainable parameters by 99.8% while
  matching full fine-tuning, and finds that adapting the whole encoder beats adapting early layers.
- **Conv-LoRA for SAM** [zhong2024convlorasam] **[S]** injects lightweight convolutions inside the LoRA
  bottleneck through a mixture of experts. It shows a loss-level regulariser can be layered onto a PEFT
  module (the MoE balancing term).
- **Edge-oriented CNN PEFT.** LoRA-C [ding2024lorac], LoRA-Edge [kwak2026loraedge] and CoLoRA
  [rivera2025colora] (an OCT medical-imaging study) **[R]** target CNNs for deployment. None reports
  calibration or OOD detection (`docs/DEFENCE_BRIEF.md` §5).

### 8.4.4 PEFT for vision Transformers

VPT [jia2022vpt], AdaptFormer [chen2022adaptformer], SSF [lian2022ssf], FacT [jie2023fact] and NOAH
[zhang2022noah] **[T/R]** are the ViT-side PEFT methods named in the field-coverage matrix
(`docs/RESULTS_MASTER.md` §4.6). **Documented absence:** the survey of PEFT for pretrained vision models
[xin2024vpeftsurvey] **[R]** does not discuss calibration, uncertainty quantification or OOD detection at all.

---

## 8.5 Confidence calibration

**Definition and metrics.** Guo et al. [guo2017calibration] **[S]**:

- define calibration through reliability diagrams;
- measure it with **Expected Calibration Error**, ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|, reported with
  M = 15 bins, and **MCE** (the worst-bin gap);
- show modern deep networks are overconfident, with miscalibration linked to depth/width, Batch
  Normalization and reduced weight decay;
- find **temperature scaling** — one scalar T fitted by NLL on a held-out validation set — the most effective
  post-hoc fix on most vision datasets. It never changes the argmax, so accuracy is unchanged.

Vector scaling collapses to temperature scaling ("miscalibration is intrinsically low-dimensional"), and
matrix scaling overfits when there are many classes.

**The Brier score** is a proper scoring rule, used as a training loss and as a metric in Deep Ensembles
[lakshminarayanan2017ensembles] **[S]**.

**Later large-scale studies:**

- **Minderer et al.** [minderer2021revisiting] **[R]** revisit calibration across modern architectures and,
  with Guo et al., establish that calibration and accuracy are governed by different factors. RQ1's
  qualitative pattern is therefore not new.
- **A Benchmark Study on Calibration** [tao2024benchmarkcalibration] **[R]** covers 117,702 architectures but
  runs no variance decomposition: box plots and rank correlations only.
- **On Fairness of Low-Rank Adaptation** [ding2024fairness] **[R]** analyses calibration across backbones for
  LoRA versus full fine-tuning, not one adapter architecture against another.
- **Robust Calibration of Large Vision-Language Adapters** (ECCV 2024) [†] **[R]** finds that CLIP adapters and
  prompt learning degrade calibration under distribution shift because logit ranges grow. It is a different
  setting and mechanism from this thesis.
- **Accuracy-preserving calibration via invertible logit transforms** [zhao2026invertible] **[R]** applies to
  softmax classifiers only.

**Name collision to address explicitly.** *Dirichlet calibration* [kull2019dirichlet] **[R]** is a post-hoc
calibration map over softmax outputs. It is unrelated to the evidential Dirichlet head used here.

**False friend.** Sources saying "ResNet calibrates better than MobileNet" come from the *quantization*
literature, where "calibration" means choosing quantization ranges. It has nothing to do with ECE.

---

## 8.6 Predictive uncertainty and Bayesian PEFT

### 8.6.1 Ensembles and single-pass methods

**Deep Ensembles** [lakshminarayanan2017ensembles] **[S]** train M (default 5) randomly initialised networks
with a proper scoring rule and average their predictions. They match or beat MC-dropout on calibration and
especially on dataset shift, and give markedly higher entropy on unknown classes. The cost is M-fold
parameters and latency, which motivates single-pass alternatives for edge devices.

### 8.6.2 Bayesian PEFT

Key conceptual point from the summaries **[S]**: *"Bayesian" means different things across these papers*,
and none of the five read uses evidential Dirichlet output uncertainty.

| Work | Mechanism | Setting | Relevance |
|---|---|---|---|
| **Laplace-LoRA** [yang2024laplacelora] **[S]** | Post-hoc Laplace approximation (KFAC, low-rank factor) over LoRA weights only; linearised predictive with Monte-Carlo logits | LLaMA2-7B / RoBERTa, common-sense reasoning, ECE/NLL, near/far OOD | Canonical post-hoc Bayesian PEFT; tunes prior on the train set |
| **BLoB** [wang2024blob] **[S]** | Bayes-by-backprop on LoRA's A matrix only (asymmetric), Flipout, KL re-weighting; mean and covariance learned jointly | LLMs; ACC/ECE/NLL with small- and large-shift OOD | Train-time Bayesian LoRA; shows naive BBB+LoRA diverges |
| **TFB** [shi2025tfb†] **[S]** | Training-free: finds the largest isotropic low-rank posterior variance within a performance tolerance by binary search; equivalent to generalised VI | LLMs | Plug-in calibrator on a trained LoRA |
| **BH-PEFT** [chai2025bhpeft†] **[S]** | Mean-field VI over hybrid prefix + scaled parallel adapter weights; posterior-as-prior for streaming data | RoBERTa, business text tasks | Bayesian hybrid adapters |
| **Bayesian-LoRA** [meo2024bayesianlora†] **[S]** | Bayesian *gates* choosing LoRA rank and quantization bit-width | DeBERTaV3, GLUE | **Compression, not uncertainty.** Do not cite as an uncertainty method |
| **BayesAdapter** [moralesalvarez2025bayesadapter] **[R]** | Variational Bayes over a linear CLIP adapter, up to 32 shots | Vision (CLIP) | Reports ECE improving by ~2.5% |
| **LoRA-Ensemble** [muhlematter2026loraensemble] **[R]** | Parameter-efficient ensembling in self-attention; rank sweep | ViT, CIFAR-100 | Reports single-network ECE starting to deteriorate at rank 32. A partial contradiction to the thesis's rank sweep (`03_results.md` §3.7) |
| **"Be Confident in What You Know": Bayesian PEFT of vision foundation models** (NeurIPS 2024) [pandey2024bayesianpeft] **[R]** ⚠️ bib unverified | Finds PEFT-adapted foundation models *underconfident*; base-rate adjustment + evidential ensemble | ViT foundation models | **Closest-named paper. The thesis must include a differentiation paragraph** |

Recent LLM-side work (Scalable Bayesian LoRA [samplawski2025scalabl], Stiefel-manifold priors
[shihab2026stiefelbayes], BaRA [duan2026bara], Bayesian Sparse LoRA [zhang2026bayesiansparselora]) and a
2026 benchmark, Bayesian Adaptation Gym [samplawski2026bagym] **[R]**, show the question "does uncertainty
survive PEFT?" is active. These are motivation for the topic, not evidence that this thesis's findings are
new (`docs/DEFENCE_BRIEF.md` §4).

**Pattern across §8.6:** Bayesian PEFT is almost entirely weight-space (Gaussian posteriors via VI or
Laplace), on Transformer or LLM backbones, and outside the true 1–5-shot vision regime.

---

## 8.7 Evidential deep learning (EDL)

### 8.7.1 The foundational model

**Sensoy et al.** [sensoy2018edl] **[S]** reinterpret classification through subjective logic:

- the network outputs non-negative **evidence** e_k (ReLU in the paper);
- α_k = e_k + 1, S = Σα_k;
- belief b_k = e_k/S, **uncertainty (vacuity) u = K/S**, expected probability p̂_k = α_k/S.

Three losses are derived by integrating a base loss over the Dirichlet. The paper **chooses the Bayes risk of
squared error** as the most stable:

L_i = Σ_k (y_k − α_k/S)² + α_k(S − α_k)/(S²(S+1)).

A KL term towards a uniform Dirichlet, on the evidence with the true class removed, is annealed as
λ_t = min(1, t/10) over epochs. It lets misclassified samples recover rather than collapse to "don't know".
The whole prediction comes from one deterministic forward pass.

### 8.7.2 Prior and posterior networks

- **Prior Networks** [malinin2018priornetworks] **[S]** separate data, model and *distributional*
  uncertainty. A Dirichlet Prior Network is trained with a multi-task KL towards sharp in-distribution and
  flat OOD targets. **It requires OOD data during training**, and differential entropy is its best OOD
  measure.
- **Prior and Posterior Networks survey** [ulmer2023priorposterior†] **[S]** unifies the family. Prior
  networks predict α and often need OOD data or distillation. Posterior networks predict pseudo-counts and are
  OOD-free; this group includes Sensoy's EDL and PostNet [charpentier2020postnet]. **Key caution:** the
  idealised Dirichlet shapes are not produced automatically by empirical risk minimisation; the loss must
  force them.
- **EDL survey** [gao2024edlsurvey] **[S]** notes:
  - softplus is a standard evidence function ("ReLU also usable") and the choice matters for stability;
  - EDL "evidence" is not Bayesian model evidence;
  - vacuity (lack of evidence) differs from dissonance (conflicting evidence);
  - **R-EDL** shows the rigid "+1" prior and the variance-minimising regulariser are non-essential and can
    induce overconfidence [chen2024reedl].
- **Deep Evidential Regression** [amini2020der] **[S]** is the regression counterpart, with a Normal-Inverse-
  Gamma prior and an error-scaled evidence regulariser. Background only, because the thesis is classification.

### 8.7.3 Critiques and recent analyses

- **Titles only [T]:** *Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation*
  [bengs2022pitfalls] and *Are Uncertainty Quantification Capabilities of EDL a Mirage?* [shen2024mirage]
  question whether loss-minimisation-based second-order methods quantify epistemic uncertainty faithfully.
  Read them before stating more than this.
- **Epistemic calibration in second-order classification** [hoarau2026epistemic] **[R]:** standard reverse-KL
  EDL objectives yield non-vanishing epistemic uncertainty even in the infinite-data limit. *(This thesis's
  loss is the squared-error form, not reverse-KL; cite as convergent, not identical.)*
- **Plug-in losses for EDL** [hayta2026pluginlosses] **[R]:** proves softmax is a special case of an
  evidential classifier. This pre-explains RQ2's finding that the objective barely matters. **Cite
  proactively.**
- **Poorly tuned EDL defaults.** [chen2024reedl] and [yang2025bilevel] **[R]** both report that EDL default
  hyperparameters are frequently poorly tuned. RQ4 must not be framed as discovering this.
- **Rethinking Vacuity for OOD Detection** [mcnamara2026vacuity] **[R]:** vacuity-based AUROC can be inflated
  when in-distribution and OOD class counts differ. It does not apply here, because every OOD sample is scored
  under the same 5-way head. State this.
- **EDL recalibration.** Density-informed pseudo-counts [carlotti2026density] and Invascal
  [turacan2026invascal] **[R]** are training-time or architectural interventions, not post-hoc two-parameter
  refits.

### 8.7.4 Evidential few-shot learning

**Bayesian Evidential Learning (BEL)** [linghu2022bel] **[R]** is the nearest evidential few-shot work. It
uses ResNet-12/Conv-4 backbones meta-trained on episodes and **improves** ECE (3.59% vs 14.69%, miniImageNet
5-shot). It reports no OOD AUROC and no parameter counts, and its backbone is not frozen. Its calibration
result points in the **opposite** direction to this thesis's Orig-RQ2 negative, as does BayesAdapter's
(§8.6.2). Treat the thesis result as a boundary case, not a contradiction.

---

## 8.8 Out-of-distribution detection

### 8.8.1 Definitions and benchmarks

**OpenOOD** [yang2022openood] **[S]** unified 35 methods across anomaly detection, open-set recognition, OOD
detection and uncertainty. It defines:

- **near-OOD** as semantic shift only;
- **far-OOD** as semantic plus covariate or domain shift.

It removes in-distribution-overlapping images from OOD sets (for example 1,207 TinyImageNet images for
CIFAR-10). Findings: post-hoc methods are generally no worse than training-based ones, and extra outlier data
"seems not necessary".

**OpenOOD v1.5** [zhang2024openoodv15] **[S]** fixes evaluation pitfalls:

- strictly disjoint OOD train/val/test sets;
- hyperparameters tuned on validation data only, never test;
- 3 training runs instead of 1;
- CIFAR near-OOD = CIFAR-10/100 + TinyImageNet; far-OOD includes SVHN.

It finds no single winner across benchmarks.

**The thesis follows this protocol.** Held-out classes and TinyImageNet serve as near-OOD, SVHN and Gaussian
noise as far-OOD [netzer2011svhn, le2015tinyimagenet] **[T]**. Temperature and evidence-affine fitting use
VAL episodes only.

### 8.8.2 Scoring rules

| Score | Source | Idea | Used here? |
|---|---|---|---|
| **MSP** | [hendrycks2017baseline] **[S]** | Maximum softmax probability. Misleading as an absolute confidence, but its *relative* ordering separates ID from OOD. Introduced the AUROC/AUPR evaluation reused since | ✅ |
| **ODIN** | [liang2018odin†] **[S]** | Temperature-scaled softmax (T = 1000) plus a small input perturbation; needs an extra forward+backward pass | ❌ never implemented |
| **Mahalanobis** | [lee2018mahalanobis†] **[S]** | Class-conditional Gaussians with tied covariance in feature space; robust with small or noisy training sets; multi-layer ensemble | ❌ never implemented |
| **Energy** | [liu2020energy] **[S]** | −E(x) = T·log Σ exp(f_i/T). Aligned with input density, whereas softmax confidence is energy shifted by the max logit. T = 1 is best; parameter-free post hoc | ✅ |
| **TS-MSP** | [guo2017calibration] **[S]** | MSP after temperature scaling fitted on validation data | ✅ |
| **Vacuity** | [sensoy2018edl] **[S]** | u = K/S from the evidential head | ✅ |

The energy summary itself notes a conceptual parallel **[S]**: Dirichlet strength S and negative energy both
measure the "amount of support" for an input.

### 8.8.3 Training objective versus scoring rule

**One Model, Many Behaviors** (WACV 2026) [krumpl2026onemodel] **[R]** runs a genuine ANOVA over training
method × scoring method on OOD AUROC, at large scale (56 models × 21 detectors × 8 OOD sets) with a 10.6%
residual. It is the **closest methodological precedent** for RQ1 and RQ2. It does not cover PEFT, frozen
backbones, few-shot evaluation or calibration.

**A Systematic Comparison of Training Objectives for OOD Detection** (2026) [genc2026objectives] **[R]**
states the gap directly: objective and scoring rule "are not fully factorized, since each objective is
evaluated with the confidence measure most natural to its output space". It includes no evidential
objective.

**Bitterwolf et al.** [bitterwolf2022breaking] **[R]**, together with OpenOOD and Liu et al., establish that
post-hoc scores often matter more than training changes.

---

## 8.9 Synthesis and research gap

### 8.9.1 What each line of work reports

Adapted from `docs/DEFENCE_BRIEF.md` §5, where each row was checked against its sources.

| Line of work | Representative | Accuracy | ECE | OOD AUROC | Param budget | Few-shot episodic | Edge-class backbone |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical few-shot | ProtoNet, MAML, R2D2, MetaOptNet | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot | P>M>F, CoOp, Tip-Adapter, CLIP-Adapter | ✅ | ❌ | ❌ | partial | partial | ❌ |
| ViT PEFT | VPT, AdaptFormer, SSF, FacT, NOAH | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| CNN PEFT (edge) | Conv-Adapter, LoRA-C, LoRA-Edge, CoLoRA | ✅ | ❌ | ❌ | ✅ | partial | ✅ |
| Frozen-CNN few-shot adapters | TSA, FiT | ✅ | ❌ | ❌ | ✅ | partial | partial |
| Bayesian PEFT | Laplace-LoRA, BLoB, BayesAdapter | ✅ | ✅ | partial | ✅ | ❌ | ❌ |
| Evidential few-shot | BEL | ✅ | ✅ | ❌ | ❌ | ✅ | partial |
| TinyML / TinyDL | TinyDL survey | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **This thesis** | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 8.9.2 The gaps, and the RQ that addresses each

| Gap | What the literature leaves open | Evidence that it is open | Addressed by |
|---|---|---|---|
| **G1 — Joint reliability evaluation** | No study reports accuracy, calibration, OOD detection and budget together for PEFT on frozen lightweight CNNs with disjoint-class episodic evaluation | §8.9.1 table; two surveys omit uncertainty [xin2024vpeftsurvey, somvanshi2025tinydl] | Step 10 grid + Step 11 (Contribution 1) |
| **G2 — Attribution** | Which design axis drives which reliability outcome has not been decomposed in PEFT or few-shot settings | Largest calibration benchmark runs no ANOVA [tao2024benchmarkcalibration]; the WACV 2026 ANOVA excludes PEFT, few-shot and calibration [krumpl2026onemodel] | **RQ1** |
| **G3 — Objective vs score** | Evidential gains in OOD detection have always been measured with vacuity, so the objective and the score are confounded | Named as an open limitation [genc2026objectives]; no work found applying energy to Dirichlet-trained logits | **RQ2** |
| **G4 — Architecture vs budget** | Whether an adapter outcome follows its architecture or its parameter count has not been tested; convention equalises budgets | Matched-budget PEFT study is accuracy-only [wang2026cptensor]; rank sweeps vary budget within one architecture [muhlematter2026loraensemble] | **RQ3** |
| **G5 — Post-hoc evidential repair** | Post-hoc calibration exists for softmax; for evidential heads, fixes change training or architecture, and whether a refit preserves vacuity ranking is untested | [guo2017calibration, zhao2026invertible] softmax-only; [carlotti2026density, turacan2026invascal] not post-hoc | **RQ4** |

### 8.9.3 How the gap changed during the project (report this honestly)

The proposal framed the gap as "PEFT + evidential uncertainty on lightweight CNNs has not been studied". A
literature check after the grid (2026-08-21 to 08-23) showed each original *comparison* question already
had close precedent:

- **Placement:** TSA [li2022tsa], with the same backbone, head type and episode count.
- **Evidential or Bayesian calibration:** BEL [linghu2022bel] and BayesAdapter [moralesalvarez2025bayesadapter].
- **Energy vs MSP:** Liu et al. [liu2020energy].
- **Pareto plots:** a presentation device, not a finding.

The thesis therefore moved from *comparison* ("does A beat B") to *attribution and remediation* ("what
drives what, and can it be fixed"). No experiment was discarded; the balanced factorial design supported
the new questions. The move itself shows the literature review working as intended and belongs in
Chapter 2's closing paragraph.

**Novelty searches were thorough but not exhaustive** (five passes, `docs/RQ_SUPERVISOR_REPORT.md` Appendix
C). Write "no prior work was found", never "no prior work exists".

---

## 8.10 Citation hygiene before the bibliography is frozen

**Papers summarised in `PAPER SUMMARIES/` but missing from `docs/refs.bib`** (marked † above). Add entries
if the lit review cites them; they are background, not method grounding (`docs/CITATION_AUDIT.md` §4):

| Suggested key | Paper | Summary file |
|---|---|---|
| `liu2022convnext` | A ConvNet for the 2020s | `CNN_paper_summaries.txt` |
| `han2024peftsurvey` | PEFT for Large Models: A Comprehensive Survey | `PEFT_summaries.txt` |
| `ulmer2023priorposterior` | Prior and Posterior Networks survey | `EDL_paper_summaries.txt` |
| `liang2018odin` | ODIN | `ODD_paper_summaries.txt` |
| `lee2018mahalanobis` | Mahalanobis detector | `ODD_paper_summaries.txt` |
| `shi2025tfb` | Training-Free Bayesianization | `B-peft_paper_summaries.txt` |
| `chai2025bhpeft` | BH-PEFT (venue/year not printed in the PDF) | `B-peft_paper_summaries.txt` |
| `meo2024bayesianlora` | Bayesian-LoRA (compression) | `B-peft_paper_summaries.txt` |
| *(none)* | Robust Calibration of Large Vision-Language Adapters, ECCV 2024 | not summarised; read before citing |

**Unresolved or risky citations** (from `docs/CITATION_AUDIT.md` §7):

- `pandey2024bayesianpeft`: metadata **unverified**, and it is the closest-named paper. A human must read it.
- `shuttleworth2024illusion` (arXiv:2410.21228): metadata fine; the *use* of its claim needs re-checking.
- `lei2023riggedlottery` (arXiv:2302.09369): **not** a calibration double-descent paper. Do not cite it for that.
- `kaya2025cnnvit` (arXiv:2510.04794): geometric estimation, not classification.
- FiT: authors are **Shysheya et al.**; three older docs still say "Bateni".
- 15 `[CHECK]` entries in `refs.bib`. Spot-check first the five that the methodology depends on:
  `he2016resnet`, `deng2009imagenet`, `krizhevsky2009cifar`, `netzer2011svhn`, `le2015tinyimagenet`.
