# 7. Introduction, Research Gap and Contributions (thesis Chapter 1 draft)

This file is **source material for Chapter 1** and for the abstract. It is written in thesis register, but
it is a draft to adapt, not final text. Every number comes from `03_results.md`; every claim about prior
work comes from `08_literature_review.md`, which records its source.

Citation keys in square brackets (e.g. `[sensoy2018edl]`) are keys in `docs/refs.bib`.

---

## 7.0 Abstract (draft, ~250 words)

> Lightweight convolutional networks remain the practical choice for on-device vision, but deploying them on
> new tasks with only a handful of labelled images raises two problems at once: adapting millions of
> weights to one or five examples per class is wasteful, and the adapted model must still signal when its
> predictions are unreliable. This thesis studies parameter-efficient adaptation of frozen ImageNet-pretrained
> ResNet-18 and MobileNetV3-Small backbones for 5-way 1- and 5-shot classification. It compares a parallel
> bottleneck adapter with LoRA (6,928–31,746 trainable parameters) and reads a parameter-free prototype head
> either as a softmax or as an evidential Dirichlet distribution. A balanced 120-run factorial on CIFAR-FS and
> MiniImageNet, evaluated on 600 fixed episodes for accuracy, calibration and near/far out-of-distribution
> (OOD) detection, shows that the reliability properties separate cleanly. Shot count explains 76% of
> accuracy variance, while head interpretation explains 83% of calibration variance and 0.2% of accuracy
> variance. Crossing every OOD score with both training objectives shows that the scoring rule, not the
> training objective, determines OOD detection. A pre-registered matched-budget experiment shows that
> accuracy and near-OOD ranking follow adapter architecture, while the calibration difference between
> adapters follows the backbone rather than the parameter budget. Finally, refitting the evidential head's
> two evidence parameters on validation episodes improves its calibration in 48 of 48 cells and largely
> preserves OOD ranking, but does not close the gap to temperature-scaled softmax. Several expected results,
> including better evidential calibration, were not observed, and these negative results are reported
> alongside the positive ones.

*(Word count ≈ 245. Numbers: RQ1 η² table, RQ2 verdict, RQ3 verdict, RQ4 48/48 — all in `03_results.md`.)*

---

## 7.1 Motivation

**Vision models increasingly run on small devices, and new classes arrive with few labels.** An inspection
camera that must recognise a newly introduced product, or a field device that must learn a new category,
rarely has thousands of labelled images. It usually has one to five. Lightweight CNNs such as
MobileNetV3-Small were designed for exactly this deployment setting: the MobileNetV3 paper optimises the
accuracy–latency trade-off directly on mobile CPUs [howard2019mobilenetv3]. The TinyML → TinyDL survey
records that CNNs still dominate microcontroller-class targets, where attention layers remain a latency
bottleneck [somvanshi2025tinydl].

**Adapting such a model is itself a problem.** Updating every weight of a pretrained network from 25 images
risks overfitting and is expensive to store per task. Parameter-efficient fine-tuning (PEFT) freezes the
backbone and trains a small module instead. Adapters [houlsby2019adapters], LoRA [hu2022lora] and BitFit
[benzaken2022bitfit] reach close to full fine-tuning accuracy with a small fraction of the parameters.

**Accuracy alone is not enough.** A model deployed without supervision must also tell a user when not to
trust it. Modern networks are systematically overconfident [guo2017calibration]. Softmax confidence stays
high on inputs that are wrong or unfamiliar: in the MSP baseline paper, wrong CIFAR-100 predictions still
average ~66% confidence, and Gaussian noise fed to an MNIST network receives ~91% [hendrycks2017baseline].
Two properties are needed on top of accuracy:

- **calibration**: confidence should match the probability of being correct;
- **out-of-distribution (OOD) detection**: the model should flag inputs from outside the classes it was
  adapted to.

**Evidential deep learning promises both at almost no cost.** It replaces the softmax with a Dirichlet
distribution over class probabilities, giving an explicit "I don't know" mass (*vacuity*) from a single
forward pass [sensoy2018edl]. Ensembles and weight-space Bayesian methods need several passes or several
models [lakshminarayanan2017ensembles, yang2024laplacelora, wang2024blob], which is exactly what a small
device cannot afford. Combining a small adapter, a frozen lightweight CNN and an evidential head is therefore
a natural candidate for reliable few-shot vision at the edge. It was the starting point of this thesis
(`proposal.txt`).

---

## 7.2 Problem statement

The thesis addresses four connected problems. The first three come from the proposal (`proposal.txt` §3);
the fourth emerged from the literature review and the first round of experiments.

| # | Problem | Why it matters |
|---|---|---|
| **P1** | **Parameter inefficiency.** Full fine-tuning updates millions of weights from a few samples. | Costly per-task storage and a risk of overfitting in the 1–5-shot regime. |
| **P2** | **The reliability gap.** Standard PEFT methods inherit the base model's overconfidence, and PEFT studies rarely report calibration or OOD detection. | A confident wrong prediction on a device with no human in the loop is the failure that matters. Two surveys from different communities omit uncertainty entirely [xin2024vpeftsurvey, somvanshi2025tinydl]. |
| **P3** | **Architectural bias.** PEFT research concentrates on Transformers; how adapters behave inside convolutional blocks is less studied [chen2024convadapter]. | Edge deployment is still CNN-dominated. |
| **P4** | **Confounded comparisons.** Published comparisons change several things at once. OOD studies pair each training objective with the score natural to it [genc2026objectives]. PEFT studies compare adapters whose parameter budgets differ. | A result of the form "method A beats method B" cannot say *what* caused the difference, so it cannot guide design. |

**Problem statement (one paragraph, for the thesis).**

> When a frozen lightweight CNN is adapted to a few-shot task with a small trainable module, it is not known
> which design decisions govern its accuracy, which govern its calibration, and which govern its ability to
> detect out-of-distribution inputs, nor whether the reliability deficits that result can be repaired
> without retraining. Existing work either reports accuracy alone, studies uncertainty for large
> Transformer models, or compares complete methods in ways that confound the factor of interest with others.

---

## 7.3 Research challenges

1. **Few labels make every estimate noisy.** A 5-way 1-shot support set has five images. Results must be
   averaged over many episodes (600 here) and several training seeds (3 here) before a difference means
   anything.
2. **Reliability metrics disagree with each other.** A head can rank OOD inputs well and still be badly
   calibrated (RQ1 shows these are near-independent once the head is fixed), so no single number summarises
   reliability.
3. **Confounds are structural, not statistical.** Evidential models are normally scored with vacuity and
   softmax models with MSP. On these two backbones, the larger adapter is also a particular architecture.
   More seeds cannot separate factors that always move together; only a design change can.
4. **Evidential heads are fragile to implement.** In this project, applying softplus directly to prototype
   logits produced zero evidence everywhere and no gradient (the Step 4 "evidential collapse",
   `02_research_story.md` §2.2). The Dirichlet shapes that make EDL useful are not produced automatically by
   empirical risk minimisation [ulmer2023priorposterior†].

† = not yet in `docs/refs.bib` (it was excluded as "read but unused"); add it before citing. See
`08_literature_review.md` §8.10.
5. **ImageNet pretraining leaks into few-shot benchmarks.** MiniImageNet classes are ImageNet classes, so
   absolute accuracy is inflated. Conclusions must rest on relative comparisons [hu2022pmf].
6. **Latency on real edge hardware was not available.** Efficiency had to be measured on a single CPU thread
   as a proxy.

---

## 7.4 Research gap (summary)

The full argument, with sources, is in `08_literature_review.md` §8.9. In short:

- **G1: no joint reliability evaluation in this regime.** No work found reports accuracy, calibration, OOD
  detection and parameter budget together for PEFT on frozen lightweight CNNs under disjoint-class episodic
  evaluation. Classical few-shot methods report accuracy only. Bayesian PEFT targets Transformers and LLMs.
  The one evidential few-shot method found (BEL) reports calibration but not OOD detection or parameter
  counts, and meta-trains its backbone [linghu2022bel].
- **G2: no attribution of reliability outcomes to design axes.** Large calibration and OOD benchmarks
  compare models but do not decompose variance for PEFT or few-shot settings [tao2024benchmarkcalibration,
  krumpl2026onemodel].
- **G3: the objective–score confound is open.** A 2026 comparison of OOD training objectives names it as a
  limitation, and no work found crosses logit-space scores such as energy onto Dirichlet-trained models
  [genc2026objectives].
- **G4: adapter architecture and parameter budget have never been dissociated.** PEFT convention equalises
  budgets for fairness, which hides whether an outcome follows architecture or size
  [wang2026cptensor, muhlematter2026loraensemble].
- **G5: no post-hoc repair of evidential prototype heads with measured ranking preservation.** Temperature
  scaling is standard for softmax [guo2017calibration]. Existing EDL recalibration methods change training or
  architecture [carlotti2026density, turacan2026invascal].

**The gap the proposal originally claimed was narrower than expected.** The proposal's four comparison
questions (placement, evidential calibration, Bayesian near-OOD, latency Pareto) each had close precedent,
for example TSA for placement [li2022tsa]. They were still answered (Orig-RQ1…4) but are reported as
supporting results, not contributions (`03_results.md` §3.8).

---

## 7.5 Aim and objectives

**Aim.** To determine what governs the reliability (accuracy, calibration and OOD detection) of
parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image classification, and
whether the deficits identified can be remediated post hoc.

**Objectives.**

1. Build a reproducible pipeline with frozen backbones, interchangeable adapters, a prototype head, and
   softmax and evidential readouts, evaluated on fixed episodes (`09_methodology.md` §9.2–9.5).
2. Run a balanced factorial over dataset, shot count, backbone, adapter and head interpretation, and measure
   accuracy, calibration, OOD detection and cost (Step 10, Step 11).
3. Decompose outcome variance across those design axes (RQ1).
4. Separate the effect of the training objective from the effect of the OOD scoring rule (RQ2).
5. Separate adapter architecture from parameter budget with a pre-registered matched-budget experiment (RQ3).
6. Test a two-parameter post-hoc refit of the evidential head for calibration and OOD-ranking preservation
   (RQ4).
7. Report every overturned expectation alongside the confirmed ones.

---

## 7.6 Research questions

**Overarching question.**

> What governs reliability (predictive accuracy, confidence calibration and out-of-distribution
> detection) in parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image
> classification, and can the reliability deficits so identified be remediated post hoc?

| RQ | Formal wording (from `docs/RQ_SUPERVISOR_REPORT.md`) | Hypotheses tested |
|---|---|---|
| **RQ1** | How is variance in accuracy, calibration and OOD detection distributed across the principal design axes: dataset, shot count, backbone, adapter type and head interpretation? | Head interpretation dominates calibration but not accuracy |
| **RQ2** | Is OOD detection performance attributable to the training objective under which the model was adapted, or to the scoring rule by which its outputs are read, and can the two be separated experimentally? | Score effect > objective effect |
| **RQ3** | Within the adapter axis, are accuracy and calibration governed by the same property of the adapter, or does accuracy follow adapter architecture while calibration follows trainable-parameter budget? | H3.1 architecture · H3.2 budget · H3.2-alt backbone-intrinsic (pre-registered) |
| **RQ4** | Can the calibration of an evidential prototype head be improved post hoc by refitting only the two parameters of its evidence-affine transform on held-out validation episodes, and is its OOD ranking preserved? | H4.1 ECE decreases · H4.2 ranking preserved within ΔAUROC ≥ −0.005 |

The proposal's original questions (Orig-RQ1…4) are answered in `03_results.md` §3.8. Their results appear in
the thesis as supporting findings. **Never mix the two numbering schemes** (see `README.md`).

---

## 7.7 Contributions

Each contribution carries its novelty label from `docs/RQ_SUPERVISOR_REPORT.md`. Use this wording, or
narrower wording; do not use broader wording.

1. **A reliability-complete benchmark of PEFT on frozen lightweight CNNs for few-shot classification.** It
   covers 120 runs (40 configurations × 3 seeds) and reports accuracy, macro-F1, ECE, Brier, near- and
   far-OOD AUROC and FPR@95, trainable parameters, FLOPs and latency on 600 fixed episodes. *Positioning:
   the individual components have precedent; reporting all of them together in this regime was not found
   (G1).*
2. **A variance attribution of reliability outcomes across five design axes (RQ1), [PARTLY NOVEL].** Shot
   count explains 76.1% of accuracy variance. Head interpretation explains 82.9% of ECE variance and 0.2% of
   accuracy variance. Once the head is fixed, calibration carries little information about OOD detection.
3. **A factorial separation of training objective and scoring rule for OOD detection, extended to Dirichlet
   models (RQ2), [PARTLY NOVEL].** The score explains 43.7% (far) and 13.0% (near) of AUROC variance; the
   objective explains under 1%. The energy score computed on evidential-trained logits performs almost the
   same as on softmax-trained logits (0.911 vs 0.929 far-OOD AUROC).
4. **A pre-registered matched-budget experiment that dissociates adapter architecture from parameter budget
   (RQ3), [NOVEL], the strongest contribution.** Accuracy and near-OOD ranking follow architecture
   (bottleneck wins 8/8 at matched budget, all beyond 2σ). The calibration gap between adapters follows the
   backbone, not the budget: budget hypothesis 0/4, backbone hypothesis 3/4 against a pre-set threshold of 3.
   The verdict went against the hypothesis the project had favoured.
5. **A two-parameter post-hoc refit of the evidential head with measured OOD-ranking preservation (RQ4),
   [PARTLY NOVEL].** ECE improves in 48/48 cells (mean −0.137). OOD AUROC stays within 0.005 in 150/192
   comparisons, with worst-case rank correlation 0.921. The refitted head remains worse calibrated than
   softmax in 48/48 cells.
6. **Documented negative and corrected results.** These include:
   - evidential heads calibrate worse than softmax (0/20);
   - energy beats vacuity in about 70% of matched comparisons, overturning an earlier single-configuration
     finding;
   - no interior calibration optimum appears over adapter ranks 1–64.

   *Positioning: these are not claims of novelty. They are boundary results for known effects, reported so
   the thesis is not read as selective.*
7. **Engineering artefacts.** These include:
   - a reproducible, config-driven codebase with byte-identical reruns;
   - frozen episode seeds with VAL/TEST separation;
   - a regression guard confirming that later analyses reproduce the committed grid numbers exactly.

**Supporting results (not claimed as contributions).** These are the proposal's original questions:

- parallel bottleneck beats LoRA in 16/16 matched pairs;
- a 31.7k-parameter adapter matches full fine-tuning;
- the backbone drives latency (5.12×) while the adapter barely does (3.9–5.8%);
- evidential scoring adds no measurable latency (1.29%, below the 5.91% noise floor).

---

## 7.8 Scope and delimitations

**In scope:**

- two frozen ImageNet-pretrained CNN backbones: ResNet-18 and MobileNetV3-Small;
- two adapters (parallel bottleneck and LoRA), plus Full FT and Linear Probe baselines;
- one head (parameter-free prototype) with two interpretations;
- 5-way 1/5-shot episodes on CIFAR-FS and MiniImageNet;
- four OOD pools (SVHN, Gaussian noise, held-out classes, TinyImageNet);
- one fixed training recipe.

**Out of scope, with reasons (details in `02_research_story.md` §2.4):**

- **ConvNeXt-Nano, CUB-200, ISIC, CIFAR-10-C.** Proposed but not run.
- **Transformer backbones.** Not trained; only an architecture-level speed and size measurement exists.
- **Per-cell hyperparameter tuning.** The grid compares design axes under one recipe; it does not seek the
  best number per cell.
- **Weight-space Bayesian PEFT (Laplace-LoRA, BLoB) as trained baselines.** Discussed in the literature
  review only.
- **Real edge hardware.** A single CPU thread stands in.
- **The proposal's "fewer than 500 trainable parameters" target.** Met only by Linear Probe.

---

## 7.9 Thesis layout

| Chapter | Content | Guide source |
|---|---|---|
| 1 Introduction | Motivation, problem, gap, objectives, RQs, contributions | this file |
| 2 Literature Review | Backbones, few-shot learning, PEFT, calibration, uncertainty, EDL, OOD detection, gap synthesis | `08_literature_review.md` |
| 3 Methodology | Problem formulation, model, losses, training, scoring, metrics, analysis design per RQ | `09_methodology.md` §9.1–9.9 |
| 4 Experimental Setup | Datasets, OOD pools, preprocessing, hyperparameters, hardware, reproducibility | `09_methodology.md` §9.10–9.12; `04_experiments.md` |
| 5 Results | Grid tables, RQ1–RQ4, former RQ5, original questions, efficiency | `03_results.md` |
| 6 Discussion | Synthesis, relation to prior work, practical guidance, threats to validity | `10_discussion_and_conclusion.md` §10.1–10.5 |
| 7 Conclusion and Future Work | Answers, limitations, future work | `10_discussion_and_conclusion.md` §10.6–10.7 |
| References | IEEE numeric, from `docs/refs.bib` | `08_literature_review.md` §8.10 |
