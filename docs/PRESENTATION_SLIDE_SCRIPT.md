# B-PEFT — Presentation Slide Script (29 slides)

**Project:** B-PEFT — Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision
with Lightweight CNN Backbones.
**Purpose of this document:** a page-by-page, presentation-ready script — actual on-slide
text, image/diagram placeholders, and speaker notes — built from this project's own
tracked sources: [`proposal.txt`](../proposal.txt), [`progress.txt`](../progress.txt),
[`docs/RESULTS_MASTER.md`](RESULTS_MASTER.md), [`docs/DEFENCE_BRIEF.md`](DEFENCE_BRIEF.md),
and `PAPER SUMMARIES/*.txt`. Every number below is transcribed, not invented.
**Status this reflects:** RQ1–RQ3 closed (Step 10, grid CLOSED 2026-08-06); RQ4 code-complete,
canonical measurement pending (Step 11, as of 2026-08-11 — see [`progress.txt`](../progress.txt)).
If you present before Step 11 lands, Slide 23 says so explicitly rather than guessing a number.

## Fill in before presenting

- **Slide 1** needs your name(s), student ID(s), and supervisor's name — none of that is
  recorded anywhere in the tracked repo files, so I've left it as a placeholder rather than
  guess.
- **Slide 8** flags a citation-naming issue worth being aware of: `proposal.txt` §9 lists a
  paper as "**Bayesian Parameter-Efficient Fine-Tuning for Large Vision-Language Models**"
  and another as "**BayesLoRA**." Per `B-peft_paper_summaries.txt` (the authoritative
  paper-summary file this project's own conventions say to defer to — see `CLAUDE.md`),
  "BayesLoRA" is actually **Laplace-LoRA** (Yang, Robeyns, Wang, Aitchison, ICLR 2024,
  arXiv:2308.13111) — the proposal used an informal name. The "Large Vision-Language
  Models" paper doesn't cleanly match a single title in the summaries file (closest
  candidates: B-LoRA, BH-PEFT, TFB — all LLM-only, not VLM). Worth a two-minute check
  against your own PDFs before a defense committee does it for you.
- **All arXiv links** below were carried over from `docs/DEFENCE_BRIEF.md`, which records
  them as "verified during this session — fetched and read, not recalled." If you present
  significantly later than 2026-08, spot-check that nothing has been updated/retracted.

---

## Slide map (29 total)

| # | Title |
|---|---|
| 1 | Title |
| 2 | Agenda |
| 3 | Motivation & Background |
| 4 | Goal, Research Objective & Problem Statement |
| 5 | Objectives / Expected Contributions |
| 6 | Literature Review: Evidential Deep Learning (Sensoy et al., 2018) |
| 7 | Literature Review: Conv-Adapter — PEFT for CNNs |
| 8 | Literature Review: Bayesian PEFT for Large Models |
| 9 | Literature Review: Closest Prior Work — BEL (Bayesian Evidential Few-Shot Learning) |
| 10 | Literature Review: Foundation-Model Few-Shot Baselines |
| 11 | Literature Review Summary — The Gap |
| 12 | Research Challenges & Research Questions (RQ1–RQ4) |
| 13 | Proposed Architecture — Pipeline Overview |
| 14 | Proposed Architecture — PEFT Adapters & Placement |
| 15 | Proposed Architecture — Evidential Uncertainty Head |
| 16 | Datasets — In-Distribution (CIFAR-FS & MiniImageNet) |
| 17 | Datasets — Out-of-Distribution Sets |
| 18 | Experimental Setup — The MVT Grid |
| 19 | Evaluation Metrics |
| 20 | Results — RQ1: Adapter Placement |
| 21 | Results — RQ2: Calibration |
| 22 | Results — RQ3: OOD Detection |
| 23 | Results — RQ4: Efficiency & Pareto Frontier (status) |
| 24 | Results — Positioning vs. State of the Art |
| 25 | Results Discussion & Key Findings |
| 26 | Challenges & Limitations |
| 27 | Conclusion & Future Work |
| 28 | References |
| 29 | Thank You / Q&A |

---

## SLIDE 1 — Title

**Layout:** Title slide, centered, institutional branding (swap in your own university's crest/logo).

**On-slide text:**
> **B-PEFT: Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with
> Lightweight CNN Backbones**
>
> [Your Name] — [Student ID]
>
> Department of Computer Science and Engineering
> [Your University]
>
> Supervised by
> [Supervisor Name]
>
> Repository: github.com/notAvailable73/thesis

**Image placeholder:** `[PLACEHOLDER: University crest/logo]`

**Speaker notes:** One sentence framing before moving on: "This is a masters thesis on
making small, edge-deployable CNNs both parameter-efficient *and* honest about their own
uncertainty when learning new visual categories from just a few examples."

---

## SLIDE 2 — Agenda

**Layout:** Numbered list.

**On-slide text:**
1. Motivation & Problem
2. Objectives
3. Literature Review
4. Research Gap & Research Questions
5. Proposed Architecture — B-PEFT
6. Datasets
7. Experimental Setup & Metrics
8. Results (RQ1–RQ4)
9. Challenges, Conclusion & Future Work

**Image placeholder:** none required.

**Speaker notes:** ~15 seconds, orient the audience.

---

## SLIDE 3 — Motivation & Background

**Layout:** Text left, illustrative figure right.

**On-slide text:**
> **Three problems motivate this work** (proposal §3):

- **Parameter inefficiency.** Full fine-tuning of CNNs on small datasets causes
  catastrophic overfitting — updating millions of parameters from 5–10 examples per class
  memorizes the examples, not the concept.
- **The reliability gap.** Standard PEFT methods (LoRA, Adapters) inherit the
  overconfidence of the base model. A model that says "95% confident" when it is actually
  60% correct is dangerous in safety-critical settings — medical imaging, autonomous
  driving on edge devices.
- **Architectural bias.** Most PEFT research targets Vision Transformers. CNNs have very
  different properties (locality, translation invariance, hierarchical features) — how
  adapters behave *inside* convolutional blocks is comparatively under-studied.

**Image placeholder:** `[PLACEHOLDER: simple graphic — a small frozen CNN icon + a tiny
"adapter" puzzle-piece icon + a "?" confidence-gauge icon, representing the three problems above]`

**Speaker notes:** Anchor problem 2 concretely: an "I don't know" answer is often more
useful — and safer — than a confident wrong one.

---

## SLIDE 4 — Goal, Research Objective & Problem Statement

**Layout:** Callout box for the goal, formal statement below.

**On-slide text:**
> **Research Objective (proposal §2):** develop a framework that enables lightweight CNNs
> (e.g., MobileNetV3, ResNet-18) to adapt to new tasks using only 1–10 samples per class
> while maintaining high calibration and out-of-distribution (OOD) robustness, through
> evidential uncertainty regularization.

**In one sentence:** *freeze a small pretrained CNN, attach a tiny trainable "adapter,"
and swap the usual softmax output for an evidential one that can express genuine
uncertainty — then measure, rigorously, whether that combination is actually better.*

**Image placeholder:** none required.

**Speaker notes:** This slide is the thesis's thesis statement — everything after this is
either building it or measuring whether it's true.

---

## SLIDE 5 — Objectives / Expected Contributions

**Layout:** Numbered list.

**On-slide text:** (proposal §8, "Expected Contributions")
1. First systematic study of PEFT + evidential uncertainty on lightweight CNNs for
   few-shot learning.
2. Empirical analysis of adapter placement (serial vs. parallel) in convolutional
   architectures.
3. Demonstration of whether/where evidential regularization improves OOD detection in
   parameter-efficient fine-tuning settings.
4. Practical guidelines for deploying uncertainty-aware few-shot models on edge hardware
   (Pareto-frontier analysis).

**Image placeholder:** none required.

**Speaker notes:** Flag honestly, right here, that objective 3 is worded as a *question*
in this script ("whether/where") rather than a guaranteed win — because the results (Slide
21) answer it partly negatively. Don't let the audience assume a positive answer is coming
just because it's phrased as a contribution in the proposal.

---

## SLIDE 6 — Literature Review: Evidential Deep Learning (Sensoy et al., 2018)

**Layout:** Text left, Dirichlet-distribution illustration right.

**On-slide text:**
> **"Evidential Deep Learning to Quantify Classification Uncertainty."**
> Murat Sensoy, Lance Kaplan, Melih Kandemir. NeurIPS 2018 (arXiv:1806.01768). *The
> foundational EDL paper — this thesis's core uncertainty mechanism is built on it.*

**Core idea:**
- Reinterprets classifier output via Dempster-Shafer / Subjective-Logic theory: replace
  softmax with a non-negative **evidence** output; place a **Dirichlet distribution** over
  class probabilities.
- Predictions become "subjective opinions" with an explicit **"I don't know" (vacuity)**
  mass, instead of a forced 100%-confident guess.
- Single deterministic network — no ensembling or MC sampling needed — yet shows strong
  OOD and adversarial robustness versus Bayesian baselines in the original paper.

**Advantages:** principled uncertainty without ensembles · cheap at inference ·
directly gives a usable OOD score (vacuity).

**Limitations (found empirically in this thesis, not the original paper):** a naive ReLU
evidence activation causes "dead neurons" (negative logit → zero evidence → zero gradient)
— fixed here with softplus (Step 1, documented deviation). The KL-regularizer needs
careful annealing or it overwhelms the training signal on tiny support sets.

**Image placeholder:** `[PLACEHOLDER: diagram — Dirichlet simplex, illustrating a
"confident" corner-concentrated distribution vs. a "vacuous" spread-out one]`

**Speaker notes:** This is the single most-cited paper in the whole thesis — every RQ2/RQ3
result traces back to this mechanism.

---

## SLIDE 7 — Literature Review: Conv-Adapter — PEFT for CNNs

**Layout:** Text left, architecture figure right.

**On-slide text:**
> **"Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets."**
> Hao Chen, Ran Tao, Han Zhang, Yidong Wang, Xiang Li, Wei Ye, Jindong Wang, Guosheng Hu,
> Marios Savvides. arXiv:2208.07463 (v4, 2024).

**Core idea:**
- The first systematic PEFT design specifically for CNNs (prior PEFT work was largely
  Transformer/NLP-focused).
- A depth-wise-separable convolutional bottleneck (preserving spatial locality, unlike a
  1×1/linear adapter): down-project → nonlinearity → up-project, added back with a
  learnable per-channel scale.
- Tests **four insertion schemes** crossing (representation) × (parallel/sequential) —
  finds **Convolution-Parallel gives the best accuracy/parameter trade-off.**

**Advantages:** matches or beats full fine-tuning on 23 classification tasks with ~3.5% of
backbone params; especially strong in few-shot (up to +11% at 1-shot on ConvNeXt-B).

**Limitations:** 1×1/linear adapters suffer "loss of locality" — their core failure mode;
on large domain gaps, adapting only a K×K conv isn't enough — whole residual blocks may
need adapting.

**Image placeholder:** `[PLACEHOLDER: diagram — the four Conv-Adapter insertion schemes:
Convolution-Parallel / Convolution-Sequential / Residual-Parallel / Residual-Sequential]`

**Speaker notes:** This is the paper RQ1 directly descends from — its "parallel beats
sequential" finding for classification is exactly what Steps 6/8/10 test on a *bottleneck*
adapter (not depth-wise-separable) inside a frozen backbone, and it replicates (Slide 20).

---

## SLIDE 8 — Literature Review: Bayesian PEFT for Large Models

**Layout:** Small table + text.

**On-slide text:**
> A fast-growing cluster of 2024–2026 work asks this thesis's exact question —
> *"does uncertainty quantification survive parameter-efficient adaptation?"* — but almost
> entirely on **language models**, not vision.

| Work | Year | What it does | Domain |
|---|---|---|---|
| Laplace-LoRA *(proposal calls this "BayesLoRA")* — Yang, Robeyns, Wang, Aitchison, ICLR 2024, arXiv:2308.13111 | 2024 | Post-hoc Laplace approximation over LoRA params; cuts ECE from 31.2%→2.1% on Winogrande-small (LLaMA2-7B) | LLM |
| BLoB (NeurIPS 2024) | 2024 | Bayesian LoRA by backpropagation | LLM |
| B-LoRA (Meo et al., WANT@ICML 2024, arXiv:2406.13046) | 2024 | Learns optimal rank + quantization bit-width per LoRA block via Bayesian gates | LLM |
| BH-PEFT (Chai et al.) | ~2024–25 | Makes hybrid Adapter+LoRA+Prefix-tuning PEFT Bayesian | LLM |
| LoRA-Ensemble (arXiv:2405.14438) | 2024–25 | Parameter-efficient ensembling for uncertainty | **ViT** |
| Calibrated Adaptation / Stiefel-Bayes (arXiv:2602.17809) | 2026 | Geometry-aware Bayesian prior, <8% overhead | LLM |
| BaRA (arXiv:2606.29184) | 2026 | Bayesian rank allocation | LLM |
| Bayesian Adaptation Gym (arXiv:2606.22188) | 2026 | A dedicated *benchmark* for Bayesian low-rank adaptation | Multi-modal LM |
| **B-PEFT (this thesis)** | 2026 | Evidential Dirichlet uncertainty on a **frozen lightweight CNN**, ≤31.7k params, few-shot episodic | **Vision, edge** |

**Image placeholder:** none required (table-only slide).

**Speaker notes:** The point: a dedicated *benchmark paper* for Bayesian PEFT appeared in
2026 — fields don't build benchmarks for dated questions. Almost the entire cluster is
LLM-only; this thesis is the field's current question, applied to the one setting
(lightweight CNN, edge, vision) it hasn't reached yet.

---

## SLIDE 9 — Literature Review: Closest Prior Work — BEL

**Layout:** Text left, comparison callout right.

**On-slide text:**
> **"Bayesian Evidential Learning for Few-Shot Classification" (BEL).**
> arXiv:2207.13137. *The nearest neighbour to this thesis in the literature.*

**What it does:**
- Evidential Dirichlet uncertainty on few-shot episodes, ResNet-12 / Conv-4 backbones.
- Reports MiniImageNet 63.10 / 79.60 and CIFAR-FS 73.96 / 86.92 (1-shot / 5-shot).
- **Does report ECE — and improves it** (3.59% vs. 14.69% baseline, MiniImageNet 5-shot).

**What it doesn't do — the gap this thesis fills:**
- No OOD AUROC reported.
- No parameter counts reported.
- Its backbone is **meta-trained**, not frozen.

**The key nuance (state this carefully — it looks like a contradiction otherwise):**
> BEL shows evidential uncertainty *improving* calibration when the backbone is
> meta-trained and two networks' evidence is fused. This thesis shows it *degrading*
> calibration when the backbone is frozen and the entire trainable budget is ≤31,744
> parameters (down to just 2, for the evidential linear-probe cell). **Both can be true —
> they are different regimes.** Establishing where that boundary lies is itself a
> contribution, and it's the regime that matters for edge deployment, since meta-training a
> backbone is exactly what a 256 kB device cannot do.

**Image placeholder:** none required.

**Speaker notes:** Have this slide ready verbatim if a committee member raises "but BEL
already showed evidential heads calibrate well" — it converts an apparent contradiction
into the thesis's actual finding, rather than a rebuttal you have to invent on the spot.

---

## SLIDE 10 — Literature Review: Foundation-Model Few-Shot Baselines

**Layout:** Text left, small table right.

**On-slide text:**
> These aren't part of the thesis's own method — they're the strongest published few-shot
> numbers, used later (Slide 24) as the honest external yardstick for the accuracy/parameter
> trade-off.

| Method | Backbone | Trainable params | Role here |
|---|---|---|---|
| P>M>F (CVPR 2022, arXiv:2204.07305) | ViT-B/16, ViT-S/16, ResNet-50 (DINO/CLIP/Sup-21k pretrained) | ~21M – ~86M | Same 5-way few-shot *protocol* as ours — directly comparable accuracy |
| DINOv3 (arXiv:2508.10104) | ViT (~7B) | full model | Illustrates transformer-at-scale ceiling; deployment-cost contrast |
| SSF (arXiv:2210.08823) | ViT-B/16 | 240K | VTAB-1k PEFT param-budget reference (different protocol — 1000 labels/task vs. our 25) |
| CoOp (arXiv:2109.01134) | CLIP (frozen) | 8,192 | Closest param-count peer — but steers a frozen CLIP that must ship at inference |

**Image placeholder:** none required.

**Speaker notes:** Emphasize *now*, before the results slide, that P>M>F is the only one
of these on an apples-to-apples protocol — the others come with an explicit caveat that
gets restated on Slide 24.

---

## SLIDE 11 — Literature Review Summary — The Gap

**Layout:** Full-width checklist table (the payoff slide of the literature review).

**On-slide text:**

| Literature family | Representative work | Accuracy | Macro-F1 | Calibration (ECE) | OOD AUROC | Param budget | Few-shot episodic | Edge-deployable backbone |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical few-shot meta-learning | ProtoNet, MAML, R2D2, MetaOptNet | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot | P>M>F, CoOp, Tip-Adapter, DINOv3 | ✅ | ❌ | ❌ | ❌ | partial | ✅ | ❌ |
| PEFT for vision transformers | VPT, AdaptFormer, SSF, FacT, NOAH | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge | LoRA-C, LoRA-Edge, CoLoRA | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Bayesian PEFT | Laplace-LoRA, BLoB, BaRA, Stiefel-Bayes | ✅ | ❌ | ✅ | partial | ✅ | ❌ | ❌ |
| Evidential few-shot | BEL (arXiv:2207.13137) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | partial |
| TinyML / TinyDL | TinyDL survey (arXiv:2506.18927) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **B-PEFT (this thesis)** | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Image placeholder:** none required.

**Speaker notes:** Two of those ❌ columns are *documented absences*, not inferences: the
PEFT-for-vision survey (arXiv:2402.02242) does not discuss calibration/UQ/OOD at all; the
TinyML→TinyDL survey (arXiv:2506.18927) covers quantization extensively with zero mention
of uncertainty estimation. Two independent surveys, two different communities, the same
blind spot — that blind spot is this thesis's contribution.

---

## SLIDE 12 — Research Challenges & Research Questions

**Layout:** Two stacked sections.

**On-slide text:**

**Research Challenges:**
- Data scarcity by design — few-shot means training signal is inherently thin.
- Overconfidence in standard PEFT methods, inherited from the base model.
- CNN-specific adapter behavior is under-studied relative to Transformers.
- No existing work reports accuracy + calibration + OOD + parameter-budget together for
  this setting (Slide 11).

**Research Questions (proposal §4):**
- **RQ1:** How does the placement of adapter modules (serial vs. parallel) within a CNN
  bottleneck affect few-shot accuracy vs. parameter count?
- **RQ2:** Can Evidential Dirichlet Networks provide superior calibration compared to
  standard softmax when fine-tuning with <500 trainable parameters?
- **RQ3:** Does the integration of a Bayesian loss prior improve the detection of near-OOD
  samples in low-data regimes?
- **RQ4:** What is the Pareto frontier between inference latency and uncertainty quality on
  edge hardware?

**Image placeholder:** none required.

**Speaker notes:** State plainly and early that RQ2 is answered **no** (Slide 21) — don't
let four confidently-worded questions imply four confirmed "yes" answers are coming.

---

## SLIDE 13 — Proposed Architecture: Pipeline Overview

**Layout:** Simple left-to-right block diagram, minimal text.

**On-slide text:**
> **Pipeline:** frozen backbone → trainable adapter → classification head, assembled by a
> single `build_model(cfg)` factory (`src/models/bpeft_model.py`). Every piece — backbone,
> adapter, head, loss, dataset — is chosen entirely by a YAML config key.

**Conceptual flow:**
`Support/Query Images → Frozen CNN Backbone (ResNet-18 or MobileNetV3-Small) → Trainable
Adapter (Bottleneck / LoRA / BitFit / Full-FT / Linear-Probe) → Prototype Head → Softmax
or Evidential Interpretation → Prediction + Uncertainty`

**Image placeholder:** `[PLACEHOLDER: 4-box flow diagram — Input → Frozen Backbone →
Adapter → Head → Output, one icon per box, snowflake icon on the backbone to signal "frozen"]`

**Speaker notes:** Emphasize the **frozen backbone** — nothing in the large pretrained
network changes; only the small adapter (and, trivially, the parameter-free head) is
learned per task.

---

## SLIDE 14 — Proposed Architecture: PEFT Adapters & Placement

**Layout:** Table + small placement diagram.

**On-slide text:**
> Five adapter types compared head-to-head, at two candidate placements.

| Adapter | What's trained | ResNet-18 params |
|---|---|---:|
| Bottleneck (parallel) | small down→up projection block | 31,744 |
| LoRA | two low-rank matrices on one 1×1 conv | 12,288 |
| BitFit | backbone bias terms only | ~4,800 |
| Full Fine-Tuning *(baseline)* | entire backbone | 11,176,512 |
| Linear Probe *(baseline)* | nothing (0-param w/ prototype head) | 0 |

**Placement (RQ1):**
- **post_pool** — bolted on after the backbone's final pooled feature.
- **serial** — inserted directly in the data's path inside the backbone.
- **parallel** — runs alongside a backbone stage; output added back in, main path
  never blocked.

**Image placeholder:** `[PLACEHOLDER: diagram — three placement variants (post_pool /
serial / parallel) inside one ResNet stage, matching the style of Conv-Adapter's Fig. from
Slide 7]`

**Speaker notes:** Bottleneck-parallel is the configuration every later results slide
centers on — it's the RQ1 winner (Slide 20).

---

## SLIDE 15 — Proposed Architecture: Evidential Uncertainty Head

**Layout:** Text + formulas left, Dirichlet illustration right (reuse Slide 6's asset if helpful).

**On-slide text:**
> Instead of softmax probabilities, the model outputs non-negative **evidence** per class
> (proposal §5B; softplus activation — documented deviation from ReLU, see Slide 6):

$$\text{evidence}_k = \text{softplus}(f(x_i;\theta)_k)$$
$$\alpha_k = \text{evidence}_k + 1 \qquad S = \sum_k \alpha_k \qquad p_k = \alpha_k / S$$
$$\text{vacuity (uncertainty)} = K / S$$

- **High S** (lots of evidence) → low vacuity → confident prediction.
- **Low S** (little evidence) → high vacuity → "I don't know."

**Training:** Type-II Maximum Likelihood (evidential) loss + KL-divergence regularizer,
**annealed** from 0 over the first ~1,000 outer steps (prevents the KL term from
overwhelming the MSE signal on 25-image support sets — Step 1 finding).

**Design note:** the classification head itself (`PrototypeHead`) always emits raw
similarity logits; whether those are read as softmax logits or mapped to Dirichlet
evidence is decided by `cfg.head.interpretation`, kept in one shared function
(`to_evidence()`) so train/eval can never silently drift apart.

**Image placeholder:** `[PLACEHOLDER: reuse Slide 6's Dirichlet simplex diagram, or a
simple bar chart contrasting a "confident" evidence vector vs. a near-zero "vacuous" one]`

**Speaker notes:** This slide sets up both RQ2 (does this calibrate better than softmax —
Slide 21) and RQ3 (does vacuity detect OOD better — Slide 22).

---

## SLIDE 16 — Datasets: In-Distribution (CIFAR-FS & MiniImageNet)

**Layout:** Two-column split.

**On-slide text:**

**Left — CIFAR-FS:**
- 100 classes carved from CIFAR-100; **Bertinetto 2019 canonical split**, 64 train / 16
  val / 20 test classes — frozen (`data/cifar_fs_split.json`, "DO NOT REGENERATE").
- 5-way, 1-shot and 5-shot episodes; 600 frozen test-episode seeds (`configs/test_episodes.yaml`).

**Right — MiniImageNet:**
- 100 classes; **Ravi & Larochelle** 64/16/20 wnid split, cross-checked byte-for-byte
  against the canonical CSVs.
- 84×84 native images, resized to 224 to match the frozen ImageNet-pretrained backbones.
- Added (Step 9) specifically to test whether CIFAR-FS accuracy was inflated by
  ImageNet-pretraining overlap — **confirmed it partially was** (Slide 25).

**Image placeholder:** `[PLACEHOLDER: sample-image grid — a handful of CIFAR-FS and
MiniImageNet example classes side by side]`

**Speaker notes:** Both use the same episodic protocol (Slide 18) — the only thing that
changes between them is the source images and class split.

---

## SLIDE 17 — Datasets: Out-of-Distribution Sets

**Layout:** Small table, four rows.

**On-slide text:**

| OOD set | Far or Near? | Role |
|---|---|---|
| SVHN | Far-OOD | House-number photos — obviously different domain from natural-object photos |
| Gaussian Noise | Far-OOD (sanity check) | Seeded N(0,1) synthetic noise — if a model can't separate this from real photos, something is badly broken |
| CIFAR-100-heldout | Near-OOD | Classes disjoint from CIFAR-FS's 100, but visually similar (same source dataset family) |
| TinyImageNet | Near-OOD | 200 classes, subtly similar to the training domain — the harder, more clinically-relevant OOD case |

**Image placeholder:** `[PLACEHOLDER: sample-image grid — one example each from SVHN,
Gaussian noise, CIFAR-100-heldout, TinyImageNet]`

**Speaker notes:** RQ3 specifically hypothesizes the evidential edge should be *larger* on
near-OOD than far-OOD — that's exactly what near/far is testing for (confirmed, Slide 22).

---

## SLIDE 18 — Experimental Setup: The MVT Grid

**Layout:** Text + small grid-dimension diagram.

**On-slide text:**
> **Episodic meta-training** (ProtoNet-style): many sampled episodes per epoch update a
> shared adapter; validated each epoch on a fixed val-episode stream; early-stopped on
> val-accuracy plateau (patience 5).

**The MVT Grid (Step 10) — the thesis's largest single experiment:**
- **M**odel: 2 backbones (ResNet-18, MobileNetV3-Small)
- **V**ariant: adapter × placement (Bottleneck-parallel, LoRA, + Full-FT/Linear-Probe baselines)
- **T**ask: 2 datasets (CIFAR-FS, MiniImageNet) × 2 shot regimes (1-shot, 5-shot)
- × **3 random seeds** each
- = **120 total runs**, ~36.3 GPU-hours, **zero errors, 120/120 cells complete**

**Reproducibility discipline:** hyperparameter search only ever touches the 100 VAL
episode seeds (10000–10099); the 600 frozen TEST seeds (0–599) are never used for tuning.
Same config run twice → byte-identical `metrics.json`.

**Hardware:** Google Colab / Kaggle (T4 GPU).

**Image placeholder:** `[PLACEHOLDER: simple 4-axis grid diagram — Backbone × Adapter ×
Dataset × Shots, visualizing "120 cells"]`

**Speaker notes:** This grid is what elevates RQ1–RQ3 from single-configuration findings
(Steps 4–9) to well-powered, cross-checked verdicts (Step 10) — make that escalation in
rigor explicit.

---

## SLIDE 19 — Evaluation Metrics

**Layout:** Grouped list under three headers.

**On-slide text:**

**Performance:** Top-1 Accuracy · Macro-F1 Score

**Calibration:**
- **ECE** (Expected Calibration Error): $\text{ECE} = \sum_m \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$
- **Brier Score:** $\frac{1}{N}\sum (p_i - y_i)^2$
- **Temperature Scaling (TS):** a strong, cheap post-hoc calibration baseline

**Reliability (OOD):** AUROC · FPR@95%TPR · Energy score (a strong non-Bayesian baseline)

**Efficiency:** Trainable parameters · FLOPs/MACs · Inference latency (ms) · Peak memory (MB)

**Image placeholder:** none required.

**Speaker notes:** Note that no competing method in the literature reports *all* of these
together for this setting (callback to Slide 11) — that completeness is itself part of the
contribution.

---

## SLIDE 20 — Results: RQ1 — Adapter Placement

**Layout:** Table (subset — full 16-row table available in appendix/backup slide if needed).

**On-slide text:**
> **Finding: the parallel bottleneck adapter beats LoRA in 16 / 16 matched comparisons**,
> by **+2.13 pp to +8.26 pp**, across both datasets, both shot regimes, both backbones, both
> heads. Not a single configuration where LoRA wins.

| Dataset | Shots | Backbone | Parallel-bottleneck | LoRA | Δ |
|---|---|---|---|---|---:|
| CIFAR-FS | 5-shot | ResNet-18 | 91.44% (31,744p) | 86.25% (12,288p) | **+5.19** |
| CIFAR-FS | 5-shot | MobileNetV3-S | 90.74% (6,928p) | 88.05% (10,752p) | **+2.70** |
| MiniImageNet | 5-shot | ResNet-18 | 95.56% (31,744p) | 91.56% (12,288p) | **+4.00** |
| MiniImageNet | 1-shot | ResNet-18 | 85.03% (31,744p) | 80.29% (12,288p) | **+4.75** |

**Against baselines (CIFAR-FS, ResNet-18, 5-shot):** 31.7k-parameter parallel bottleneck
(**91.44%**) beats full fine-tuning of 11.18M parameters (**90.47%**) by **+0.97 pp** while
training **0.28%** of the parameters. On MobileNetV3-Small, 6,928 params reaches 90.74% —
**0.06%** of full fine-tuning's budget — and still edges it by +0.27 pp.

**The honest limit:** at **1-shot**, full fine-tuning wins by **2.57 pp** — the extra
capacity buys something the adapter can't recover with only one support image per class.

**Image placeholder:** `[PLACEHOLDER: bar chart or scatter — accuracy vs. trainable
params, parallel-bottleneck vs. LoRA vs. Full-FT vs. Linear-Probe, CIFAR-FS 5-shot]`

**Speaker notes:** On MobileNetV3-Small specifically, parallel is a **strict Pareto win**
— cheaper AND more accurate than LoRA, no tradeoff to argue.

---

## SLIDE 21 — Results: RQ2 — Calibration

**Layout:** Small range table + headline statement.

**On-slide text:**
> **Finding: no — decisively, 20 / 20 matched comparisons, no exceptions.**

| | Evidential ECE | Plain-Softmax ECE | TS-Softmax ECE |
|---|---|---|---|
| Range across all 20 configs | 0.2156 – 0.6234 | 0.0242 – 0.4476 | 0.0057 – 0.0642 |

- Evidential ECE is worse than plain softmax by **1.4×–9.1×**, and worse than
  temperature-scaled softmax by **5.3×–51.2×**.
- Worst case: MiniImageNet 5-shot, ResNet-18, parallel bottleneck — ECE 0.294 evidential
  vs. **0.0057** TS-softmax (a **51×** gap).
- Accuracy doesn't compensate either: evidential is **0.61 pp worse on average**, winning
  only 7 of 20 pairs.

**Independent 2026 theory support:** a theoretical analysis of second-order/evidential
classification (arXiv:2606.10777) shows standard reverse-KL EDL objectives yield
**non-vanishing epistemic uncertainty even with infinite data** — the same direction as
this empirical result.

**Image placeholder:** `[PLACEHOLDER: grouped bar chart or box plot — ECE distribution,
evidential vs. softmax vs. TS-softmax, across the 20 matched configs]`

**Speaker notes:** State this as a **well-powered negative result**, not a failed
experiment — 2 datasets × 2 shots × 2 backbones × 4/5 adapters, with an independent 2026
theory paper pointing the same direction, is strong, presentable evidence.

---

## SLIDE 22 — Results: RQ3 — OOD Detection

**Layout:** Table + two-sentence headline.

**On-slide text:**
> **Finding: yes against every softmax-based score, no against the training-free energy
> score — and the advantage grows exactly as the hypothesis predicts, as data shrinks.**

| Comparison | Far-OOD | Near-OOD | Overall win rate |
|---|---|---|---|
| vacuity vs. MSP | 38/40 wins, Δ +0.111 | 37/40 wins, Δ +0.053 | 93.8% |
| vacuity vs. TS-MSP | 38/40 wins, Δ +0.127 | 38/40 wins, Δ +0.067 | 95.0% |
| vacuity vs. energy | 10/40 wins, Δ −0.022 | 14/40 wins, Δ −0.007 | 30.0% |

**Low-data trend confirmed:** near-OOD advantage over MSP is **+0.0637 AUROC at 1-shot**
vs. **+0.0431 at 5-shot** — the Bayesian prior helps *more* exactly where there's less data.

**The honest counterpoint:** the training-free **energy score** — no evidential training,
no extra parameters — beats vacuity in **~70%** of comparisons overall. This is a
**correction** to an earlier single-configuration finding (Step 4.5) that suggested
near-parity; at grid scale, energy is the better default OOD score in most cells.

**Image placeholder:** `[PLACEHOLDER: AUROC comparison chart — vacuity vs. MSP vs. TS-MSP
vs. energy, grouped by far-OOD/near-OOD]`

**Speaker notes:** The defensible claim is narrower than "Bayesian priors improve OOD
detection" — it's *"among scores derived from the model's own predictive distribution,
Dirichlet vacuity beats max-softmax-probability, and more so at 1-shot — but it does not
beat a well-chosen logit-space score (energy)."* State it that precisely.

---

## SLIDE 23 — Results: RQ4 — Efficiency & Pareto Frontier

**Layout:** Status callout + preliminary (labeled non-canonical) table.

**On-slide text:**
> **STATUS: code-complete, canonical measurement PENDING.** All measurement/plotting
> infrastructure is built and tested (56 new tests, zero regressions); a `--check-params-only`
> mode independently reproduced all 40 grid cells' parameter counts from scratch. What
> remains is the actual Kaggle T4 GPU + CPU measurement session.

**Pre-registered axes** (fixed *before* any number exists, so they can't be tuned after
the fact): cost = CPU 1-thread per-image median latency (edge-hardware proxy); quality =
TinyImageNet near-OOD AUROC (the pool RQ3's low-data trend was found on).

**Development-only preview** (local CPU, NOT the canonical edge number — do not present as final):

| Backbone | Adapter | Trainable params | GMACs | Dev CPU ms/img |
|---|---|---:|---:|---:|
| ResNet-18 | Bottleneck-parallel | 31,744 | 1.83 | 54.6 |
| MobileNetV3-S | Bottleneck-parallel | 6,928 | 0.059 | 7.8 |

**Image placeholder:** `[PLACEHOLDER: leave blank / "pending" watermark until the
canonical Pareto plot (results/pareto_frontier.json) exists]`

**Speaker notes:** Say plainly: "this is the one research question still open — the
infrastructure is done and verified, but I don't yet have a number I'd stand behind as the
final answer, and I'd rather tell you that than show you a number from the wrong hardware."

---

## SLIDE 24 — Results: Positioning vs. State of the Art

**Layout:** Full-width comparison table.

**On-slide text:**
> Same 5-way few-shot protocol as P>M>F (arXiv:2204.07305) — directly comparable accuracy.

| Method | Backbone | Trainable | CIFAR-FS 5-shot | MiniIN 5-shot |
|---|---|---:|---:|---:|
| Sup-21k > ProtoNet | ViT-B/16 | ~85.8M | 96.7 | 99.2 |
| DINO > ProtoNet | ViT-S/16 | ~21M | 92.5 | 98.0 |
| DINO > ProtoNet | ResNet-50 | ~25M | — | 92.0 |
| **Ours**, parallel bottleneck | **ResNet-18 (frozen)** | **31,744** | **91.44** | **95.56** |
| **Ours**, parallel bottleneck | **MobileNetV3-S (frozen)** | **6,928** | **90.74** | 90.10 |

**Three conclusions — keep them separate:**
1. **At 5-shot on CIFAR-FS the saving is close to free:** 1.06 pp behind a fully
   meta-trained DINO ViT-S at **662× fewer** trainable parameters.
2. **Backbone-family-matched, we simply win:** vs. DINO>ProtoNet on ResNet-50, **+3.56 pp**
   (MiniIN 5-shot) at **788× fewer** parameters — the remaining gap is a ViT gap, not a
   parameter-efficiency gap.
3. **The trade is genuinely worse at 1-shot / MiniImageNet with the small backbone** — up
   to **−18.2 pp** against ViT-S. Volunteer this rather than let it be found.

**Image placeholder:** `[PLACEHOLDER: scatter plot — accuracy vs. trainable params
(log scale), our two configs vs. every row above]`

**Speaker notes:** Have the "what it costs" framing ready verbatim (`DEFENCE_BRIEF.md` §6)
— this will very likely be the first follow-up question.

---

## SLIDE 25 — Results Discussion & Key Findings

**Layout:** Bullet synthesis, no new table.

**On-slide text:**
- **RQ1 is a clean win:** placement matters independent of size; parallel bottleneck is
  Pareto-optimal or near it in every configuration tested.
- **RQ2 is a clean, well-powered negative** — and a 2026 theory paper independently
  supports why (Slide 21). This should be reported as a finding, not hidden.
- **RQ3 is real but narrower than hoped:** a genuine, low-data-amplified edge over
  probability-based scores, but not over the best non-Bayesian alternative (energy).
  Calibration quality and OOD-ranking quality are **decoupled** — the same head that
  calibrates badly is simultaneously a *better* OOD detector than max-softmax in 93.8% of
  comparisons. You only see that by measuring both.
- **The pretraining-overlap confound is real and measured** (Step 9): the adapter's
  accuracy uplift over a 0-parameter linear-probe baseline shrinks by roughly a
  third-to-a-half on MiniImageNet vs. CIFAR-FS — part of the headline accuracy numbers are
  inflated by ImageNet pretraining overlap, not pure few-shot learning ability.
- **RQ4 is the one open question** — infrastructure ready, canonical numbers pending.

**Image placeholder:** none required.

**Speaker notes:** This slide is where you make the actual argument rather than just
re-reading tables — decide in advance how you want to frame the RQ2/RQ3 mixed picture, since
it's the part most likely to draw follow-up questions.

---

## SLIDE 26 — Challenges & Limitations

**Layout:** Numbered list, five items (from `docs/DEFENCE_BRIEF.md` §7 — volunteer these,
don't wait to be asked).

**On-slide text:**
1. **No latency measurement yet.** The edge-deployment argument currently rests on
   parameter counts + published literature figures, not our own hardware timings —
   Step 11, the most exposed open gap.
2. **ImageNet-pretraining confound.** Accuracies aren't comparable to from-scratch
   few-shot SOTA; MiniImageNet's classes literally come from ImageNet. Fix: a
   from-scratch-backbone control run (planned, Step 12).
3. **No transformer arm in our own grid.** We argue from *published* ViT PEFT numbers
   rather than running a ViT under our own exact protocol. A trained ViT-Tiny/DeiT-Small
   arm would convert an argument into a measurement.
4. **The energy score beats our Bayesian one in ~70% of OOD comparisons.** Volunteered
   already on Slide 22 — repeated here because it belongs on this list too.
5. **A protocol mismatch in the SOTA comparison table.** VTAB-1k parameter budgets (Slide
   24's appendix data) use 1,000 labels/task vs. our 25 — parameter counts transfer,
   accuracies do not, and must never be tabulated against each other.

**Image placeholder:** none required.

**Speaker notes:** Presenting your own weaknesses first, clearly and specifically, is
consistently the strongest move in a defense — a reviewer who finds one of these unprompted
weighs it far more heavily than one you raised yourself.

---

## SLIDE 27 — Conclusion & Future Work

**Layout:** Two stacked sections.

**On-slide text:**

**Conclusion:**
- Built and validated a full frozen-backbone + PEFT-adapter + evidential-head pipeline
  across 2 backbones, 5 adapter types, 2 datasets, 2 shot regimes — 120 fully-run grid
  cells, zero errors.
- **RQ1 (placement):** confirmed — parallel adapter placement wins on accuracy-per-parameter,
  16/16 matched comparisons.
- **RQ2 (calibration):** answered negatively, decisively — evidential heads do not
  calibrate better here, and shouldn't be presented as if they do.
- **RQ3 (near-OOD):** answered with real but bounded support — beats probability-based
  scores, does not clearly beat the strongest non-Bayesian baseline.
- **RQ4 (efficiency Pareto):** infrastructure complete; canonical measurement is the one
  remaining open item.

**Future Work (Step 12 backlog, priority order):**
1. Trained ViT-Tiny/DeiT-Small arm under the exact same protocol — most direct answer to
   "why not just use a transformer."
2. From-scratch-backbone control run — removes the ImageNet-pretraining-overlap objection.
3. Energy score computed on evidential logits too — closes a scoring asymmetry, cheapest
   open follow-up (no retraining needed).
4. Full baseline coverage (Full-FT/Linear-Probe × every backbone × every dataset).
5. ConvNeXt-Nano backbone; CUB-200 / ISIC datasets; CIFAR-10-C corruption robustness
   (lower priority, explicit drop-order if time runs short).
6. Thesis writing (Step 13) — not started.

**Image placeholder:** none required.

**Speaker notes:** The single next action, ahead of everything else: get
`notebooks/step11_efficiency.ipynb` to complete successfully on a real Kaggle GPU session
— every RQ4 claim and the whole edge-deployment narrative depends on that number existing.

---

## SLIDE 28 — References

**Layout:** Two-column, small font. Compress to one slide (split to two if legibility
requires it in your actual deck).

**On-slide text:**
1. M. Sensoy, L. Kaplan, M. Kandemir, "Evidential Deep Learning to Quantify
   Classification Uncertainty," *NeurIPS*, 2018, arXiv:1806.01768.
2. H. Chen et al., "Conv-Adapter: Exploring Parameter Efficient Transfer Learning for
   ConvNets," arXiv:2208.07463, 2024.
3. A. X. Yang, M. Robeyns, X. Wang, L. Aitchison, "Bayesian Low-Rank Adaptation for Large
   Language Models" (Laplace-LoRA), *ICLR*, 2024, arXiv:2308.13111.
4. [BLoB: Bayesian Low-Rank Adaptation by Backpropagation], *NeurIPS*, 2024.
5. C. Meo, K. Sycheva, A. Goyal, J. Dauwels, "Bayesian-LoRA: LoRA-based PEFT using
   Optimal Quantization and Rank via Differentiable Bayesian Gates," WANT@ICML,
   arXiv:2406.13046, 2024.
6. [Bayesian Evidential Learning for Few-Shot Classification (BEL)], arXiv:2207.13137.
7. C. Zhang, N. Song, et al. [P>M>F], "Pushing the Limits of Simple Pipelines for
   Few-Shot Learning," *CVPR*, 2022, arXiv:2204.07305.
8. [DINOv3], arXiv:2508.10104, 2025.
9. D. Lian et al., "Scaling & Shifting Your Features" (SSF), arXiv:2210.08823.
10. K. Zhou et al., "CoOp: Learning to Prompt for Vision-Language Models,"
    arXiv:2109.01134.
11. [From Tiny Machine Learning to Tiny Deep Learning: A Survey], arXiv:2506.18927, 2025.
12. [A Comparative Study of Vision Transformers and CNNs for Few-Shot Learning],
    arXiv:2510.04794, Oct. 2025.
13. [PEFT for Pre-Trained Vision Models: A Survey], arXiv:2402.02242.
14. [Epistemic calibration in second-order classification], arXiv:2606.10777, 2026.

**Image placeholder:** none required.

**Speaker notes:** ⚠️ Entries in square brackets are missing a full author list in the
sources this script was built from (`docs/DEFENCE_BRIEF.md`'s link list) — fill in exact
author names from your own reference manager / the original PDFs before finalizing your
slides; do not present these bracketed placeholders as-is.

---

## SLIDE 29 — Thank You / Q&A

**Layout:** Centered, minimal.

**On-slide text:**
> **Thank you.**
> Questions?

**Image placeholder:** `[PLACEHOLDER: matching branding from title slide]`

**Speaker notes:** Have Slides 21, 22, 24, and 26 ready to jump back to — those are the
four most likely to generate follow-up questions (negative calibration result, energy-score
counterpoint, accuracy-vs-parameter trade, and the ViT-arm/latency gaps respectively).

---

*End of script. 29 slides. All numbers, citations, and dataset/grid statistics above are
transcribed from `progress.txt`, `docs/RESULTS_MASTER.md`, `docs/DEFENCE_BRIEF.md`, and
`PAPER SUMMARIES/*.txt` — nothing quantitative was invented. Resolve the two flags at the
top of this document (title-slide placeholders, the BayesLoRA/Laplace-LoRA and
VLM-paper naming check) before finalizing, and fill in the bracketed reference entries on
Slide 28 from your own source PDFs.*
