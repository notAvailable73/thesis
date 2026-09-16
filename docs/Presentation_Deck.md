# B-PEFT — Presentation Deck (23 slides + references)

The slide deck to present at the defence. Built from `docs/DEFENCE_DECK_25_SLIDES.md` after the supervisor's
review: less text on each slide, one message per slide, no defensive material on screen.

**How to read this file**

- Everything under a slide heading is **projected content**. Nothing here is a speaking script.
- `###` lines = **large headline type**. These are the numbers the committee should remember.
- `>` blocks = **boxed / highlighted** text.
- *Footer* = small type at the bottom of the slide.
- **Diagram** lines are build instructions, **not** slide text. IDs (F1–F16) refer to
  `docs/DIAGRAM_PLAN.md` and `docs/DIAGRAMS_MERMAID.md`. The slide numbers in those two files refer to the
  **old** deck, so use the map below instead.
- Numbers are rounded for the slides. Exact values and sources are in `docs/DEFENCE_DECK_25_SLIDES.md`
  and `results/rq_completion/`.

**Slide map**

| # | Slide | Section | Diagram |
|---:|---|---|---|
| 1 | Title | Opening | — |
| 2 | One message | Opening | — |
| 3 | Accuracy is not reliability | Problem | new simple illustration |
| 4 | Problem statement | Problem | — |
| 5 | Four key concepts | Background | F2 (compact) |
| 6 | Research gap | Background | — |
| 7 | The system | Method | F4 |
| 8 | Two adapters | Method | F5(a) |
| 9 | One head, two readouts | Method | F6(a) |
| 10 | Experimental design | Method | F7 |
| 11 | Datasets and evaluation | Method | — |
| 12 | Four research questions | Method | F1 |
| 13 | Parameter efficiency | Results | new big-number graphic (or F9) |
| 14 | RQ1 | Results | F11 |
| 15 | RQ2 | Results | F12 |
| 16 | RQ3 — the confound | Results | F13 + F5(b) |
| 17 | RQ3 — matched budget | Results | F14 |
| 18 | RQ4 | Results | F15 |
| 19 | Deployment cost | Implications | — |
| 20 | Limitations | Implications | — |
| 21 | Contributions | Closing | — |
| 22 | Conclusion | Closing | F16 (optional) |
| 23 | Thank you | Closing | — |

---

## Slide 1 — Title

- **Bayesian Parameter-Efficient Fine-Tuning (B-PEFT) for Reliable Few-Shot Vision with Lightweight CNN
  Backbones**
- *A controlled study of which design choices govern which aspects of reliability*
- [Presenter names · Student IDs]
- [Supervisor name and title]
- Department of Computer Science and Engineering · Islamic University of Technology
- [Defence date]

**Diagram:** none.

---

## Slide 2 — Different aspects of reliability are governed by different design choices

### No single design choice makes a few-shot model reliable on every axis.

| Reliability aspect | Governed mainly by |
|---|---|
| **Accuracy** | shot count + adapter architecture |
| **Calibration** | head type + backbone |
| **OOD detection** | the uncertainty scoring rule |
| **Calibration repair** | a post-hoc refit: it improves calibration but does not fully fix it |

*Footer:* 189 training runs · 600 fixed test episodes · 3 seeds

**Diagram:** none. The table is the visual.

---

## Slide 3 — A model can be accurate and still unreliable

- Edge devices must recognise **new classes from 1–5 examples** each.
- **Full fine-tuning** updates and stores all of the backbone's millions of weights, which is expensive in
  parameters and compute.
- **Parameter-efficient adapters** train a tiny fraction instead and keep accuracy, but what happens to the
  model's **confidence** is rarely measured.

> ### Accuracy ≠ reliability
> A model can be accurate while being **poorly calibrated**, or **unable to recognise unfamiliar inputs**.

*Footer:* Why lightweight CNNs: ResNet-18 [1] and MobileNetV3-Small [2] fit the memory and compute budgets
of edge and mobile devices [3].

**Diagram:** a new, simple illustration (not in DIAGRAM_PLAN), on the right half. Two small panels:
(1) a known-class image predicted wrong at "95% confident"; (2) an unfamiliar input (e.g. a street-number
digit) still labelled "cat, 90%". No data, no axes.

---

## Slide 4 — Can a tiny adapter make a frozen lightweight CNN reliable?

### Can a tiny trainable adapter make a frozen lightweight CNN reliable for few-shot image classification?

Three boxes in a row:

| Accurate | Calibrated | OOD-aware |
|---|---|---|
| correct on new classes it was not adapted on | its confidence matches how often it is right | flags inputs from outside the task |

| Challenge | Our response |
|---|---|
| Full fine-tuning updates millions of weights | Freeze the backbone; train at most **31.7K** adapter parameters once. New classes require no gradient-based adaptation |
| PEFT studies rarely measure reliability | Measure accuracy, calibration **and** OOD detection on the same runs |
| PEFT research focuses on transformers | Study adapters **inside lightweight CNNs** |

**Diagram:** none.

---

## Slide 5 — Four ideas the talk builds on

Four boxes, one idea each:

**1 · Few-shot episode** [4]
5 new classes, 1 or 5 labelled images each; classify the rest. Test classes are never used to train the adapter.

**2 · Parameter-efficient fine-tuning** [5, 6]
Freeze the pretrained network. Train only a small add-on module: an adapter or LoRA.

**3 · Evidential (Dirichlet) head** [7]
Outputs **evidence** per class instead of a single probability.
Low total evidence → high **vacuity** → "I don't know".

**4 · Measuring reliability**
**ECE** [8]: when the model says 90%, is it right 90% of the time? (lower is better)
**OOD AUROC**: are unfamiliar inputs scored as more uncertain than familiar ones? (higher is better)

*Footer:* "Bayesian" in the title refers to this evidential (Dirichlet) uncertainty, not to a posterior over
network weights.

**Diagram:** F2, compact. Use only the "episode" half (5 × 5 support grid + query block) inside box 1.
Leave out the 64/16/20 split half; it would crowd the slide.

---

## Slide 6 — The gap: properties studied separately, factors changed together

| | Frozen CNN backbone | Parameter-efficient adapter | Calibration | OOD detection |
|---|:-:|:-:|:-:|:-:|
| TSA [9] | ✓ | ✓ | – | – |
| BEL [10] | – *(backbone meta-trained)* | – | ✓ | – |
| BayesAdapter [11] | – *(frozen CLIP ViT)* | ✓ | ✓ | – |
| **This work** | **✓** | **✓** | **✓** | **✓** |

> ### Published comparisons usually change two things at once. We separate them:
> - **training objective** vs. **OOD score**
> - **adapter architecture** vs. **parameter budget**

*Small line:* The literature we reviewed does not combine all four properties in a controlled frozen-CNN
few-shot setting.

**Diagram:** none. The table is the visual.

---

## Slide 7 — The system: a frozen CNN, a tiny adapter, a parameter-free head

Pipeline, left to right, with three labels:

- **FROZEN backbone:** ResNet-18 or MobileNetV3-Small, never updated
- **TRAINABLE adapter:** 6.9K–31.7K parameters, the main trainable part (+2 scalars in the evidential head)
- **PARAMETER-FREE prototype head** [12]: a class = the average embedding of its support images

**Diagram:** F4, full width. For the slide, keep the colour code (grey = frozen, orange = trainable,
dashed = parameter-free) and the adapters drawn **inside** the backbone stages. Shrink the formula boxes to
short labels (softmax → MSP · evidential → vacuity).

---

## Slide 8 — Two adapter designs: a side branch vs. a low-rank weight update

| Parallel bottleneck [5] | LoRA [6] |
|---|---|
| A small side branch (down-project → ReLU → up-project), added at the end of **every stage** | A low-rank update **inside one frozen 1×1 convolution**: W = W₀ + (α/r)·BA |

*Footer:* Both start as "no change" (zero-initialised). Rank 16 in the main grid.

**Diagram:** F5(a), full width, the two zooms side by side. Do **not** add the parameter bars (F5(b))
here; they belong on Slide 16.

---

## Slide 9 — One head, two ways to read it

The prototype head produces one set of similarity scores (logits). We read them two ways:

| | Softmax | Evidential (Dirichlet) |
|---|---|---|
| Output | class probabilities | evidence → Dirichlet → probabilities |
| Uncertainty signal | max probability (MSP) | vacuity ("I don't know") |
| Extra parameters | 0 | **2** (learnable scale and bias) |

> A matched softmax / evidential pair differs by only **2 parameters**, so the head is a clean
> experimental factor.

**Diagram:** F6(a), across the top of the slide, with the table below it.

---

## Slide 10 — A controlled factorial design, not one-off comparisons

A top-to-bottom flow:

**5 design factors, 2 levels each**
Dataset {CIFAR-FS, MiniImageNet} · Shots {1, 5} · Backbone {ResNet-18, MobileNetV3-Small} ·
Adapter {bottleneck, LoRA} · Head {softmax, evidential}

↓

**32 factorial configurations**

↓

**+ 8 baseline configurations** (full fine-tuning, linear probe)

↓

**40 configurations × 3 seeds = 120 runs** → all 120 models re-scored with 4 OOD scores, plus a post-hoc
refit (no retraining)

↓

**Matched-budget experiment:** 48 runs · **pre-registered**

↓

**Rank sweep:** 21 runs · looks for a best adapter size (none found in ranks 1–64)

> Varying every factor systematically lets us **attribute** differences in each outcome to specific
> design choices, instead of just ranking models.

*Footer:* 189 training runs in total

**Diagram:** F7, redrawn as the vertical flow above so it reads in ten seconds. Make this the most
polished figure in the deck. For the slide version, drop the Phase A detail ("99 recovered + 21
retrained"); that belongs in the thesis figure only.

---

## Slide 11 — Two few-shot benchmarks, five OOD sets, four reliability measures

**Left column: data**

- **In-distribution:** CIFAR-FS [13] · MiniImageNet [14]
  - 5-way, 1-shot or 5-shot
  - 64 / 16 / 20 train / val / test classes (disjoint)
- **Far-OOD:** SVHN [15] · Gaussian noise
- **Near-OOD:** TinyImageNet [16] · held-out CIFAR-100 / MiniImageNet classes

**Right column: measures**

| Measure | Question it answers |
|---|---|
| Accuracy | Is it correct? |
| ECE | Does its confidence match its correctness? |
| OOD AUROC | Does it recognise unfamiliar inputs? |
| Latency · memory · parameters | Can it be deployed? |

*Footer:* 600 fixed test episodes · 3 seeds · test episodes never used for model selection

**Diagram:** none.

---

## Slide 12 — Four research questions

| | Question | Answered by |
|---|---|---|
| **RQ1** | Which design choices most strongly explain variation in accuracy, calibration and OOD detection? | 120-run factorial grid |
| **RQ2** | Is OOD detection explained more by the training objective or by the scoring rule? | All 120 models re-scored with 4 scores |
| **RQ3** | Does accuracy follow adapter architecture, and calibration follow parameter budget? | Grid + matched-budget experiment |
| **RQ4** | Can evidential calibration be repaired after training without breaking OOD detection? | Post-hoc refit of 60 evidential models |

**Diagram:** F1 (research roadmap) can replace this table. Slide version: keep the four RQ boxes and their
experiments, and drop the "Ch5 section" row and the long overarching-question box. Use the table or F1,
not both.

---

## Slide 13 — A 31.7K-parameter adapter achieves accuracy comparable to full fine-tuning

**Trainable parameters**

| Full fine-tuning | Bottleneck adapter |
|---:|---:|
| 11.18M | **31.7K** |

### 352× fewer trainable parameters

**Accuracy**

| Full fine-tuning | Bottleneck adapter | Linear probe (no adaptation) |
|---:|---:|---:|
| 90.47% | **91.44%** | 87.41% |

*Footer:* CIFAR-FS · 5-way 5-shot · ResNet-18 · mean of 3 seeds. At 1-shot, full fine-tuning leads
(81.1% vs 78.6%).

**Diagram:** a new big-number graphic as the main visual: two horizontal bars for trainable parameters on a
log scale, with the accuracy printed beside each bar. If you prefer a data plot, use F9 (5-shot panel only).
Do not add the MobileNetV3-Small row; it compares across backbones.

---

## Slide 14 — RQ1 · Shots explain accuracy; the head explains calibration

### Accuracy → shot count · 76%
### Calibration → head type · 83%
### OOD detection → no single dominant factor

*Small line under the headlines:* % = share of the variation across the 32 configurations explained by
that factor.

> The head changes calibration while barely touching accuracy (0.2% of accuracy variation).

**Diagram:** F11 heatmap, small, on the right. Outline the two headline cells (shots × accuracy,
head × ECE) so they stand out. Main effects only; no interaction table on this slide.

---

## Slide 15 — RQ2 · In our setting, OOD detection depends far more on how uncertainty is scored than on how the model was trained

Every model is scored 4 ways, so training objective and score are separated:
MSP [17] · temperature-scaled MSP [8] · energy [18] · vacuity [7]

### Far-OOD variation explained: scoring rule 42.7% · training objective 0.17%

*Small line:* Near-OOD: 14.7% vs 0.63%

- The same score gives nearly the same AUROC, whichever objective trained the model.
- Strongest score: **energy**. It beats vacuity in ~70% of configurations; vacuity beats
  softmax-probability scores in 37–38 of 40.

**Diagram:** F12, two panels (far | near): paired bars per score, one bar per training objective. The two
bars in each pair should look almost equal. That is the finding.

---

## Slide 16 — RQ3 · Our first experiment could not separate architecture from budget

Across the 16 matched bottleneck-vs-LoRA pairs in the grid:

- **Accuracy:** bottleneck wins **16/16**, whether it is the larger or the smaller adapter → follows
  **architecture**
- **Calibration:** the winner **flips between backbones**, and so does which adapter is larger

| Backbone | Bottleneck | LoRA | Larger adapter |
|---|---:|---:|---|
| ResNet-18 | **31.7K** | 12.3K | bottleneck |
| MobileNetV3-Small | 6.9K | **10.8K** | LoRA |

> Budget and backbone change **together**, so the grid cannot tell which one explains the calibration difference.
> → We designed a new experiment.

**Diagram:** F13 on the left (16 pairs: Δaccuracy all on one side; ΔECE flips with the backbone). F5(b)
parameter bars on the right; they can replace the table.

---

## Slide 17 — RQ3 · Matching the budget does not support parameter count as the main explanation

*Design line:* both adapters rebuilt at the **same budget within each backbone** · MiniImageNet 5-shot ·
48 runs · decision rule fixed **before** running

**Accuracy & near-OOD → architecture**

### Bottleneck still wins 8/8, all beyond 2σ

**Calibration → backbone**

### Budget hypothesis: 0 of 4 cells · Backbone dependence: 3 of 4 cells

*Small line:* The calibration gap is unchanged on ResNet-18 and halved (same direction) on
MobileNetV3-Small. Which backbone property matters is not yet identified.

*Footer:* σ = pooled across-seed SD of the two adapters

**Diagram:** F14(a) as the main visual: for each of the 4 cells, an unmatched bar next to a matched bar,
±2σ. The ResNet-18 bars stay the same height, and that is the point. Optional small inset: F8(a), showing the
budgets were matched within ~3%.

---

## Slide 18 — RQ4 · Refitting two parameters improves evidential calibration, but does not fix it

*Method line:* refit only the head's **2 evidence parameters** on validation episodes · no retraining

### ECE 0.323 → 0.185 · improved in 60 / 60 models
### Still worse than softmax (0.112) in 60 / 60
### OOD ranking preserved in 80% of comparisons (192 / 240)

**Diagram:** F15(a): a before → after dumbbell per model, with a softmax tick on each row so the remaining
gap is visible. Show F15(b) (ΔAUROC strip) only as a small panel, or leave it out and keep the 80% headline.

---

## Slide 19 — Measured inference cost is dominated by the backbone

| Design choice | Effect on latency |
|---|---:|
| **Backbone** (ResNet-18 vs MobileNetV3-Small) | **5.1×** |
| Adapter type | 4–6% |
| Evidential vs softmax head | ~1%, within measurement noise |

### In our measurements: backbone dominates latency; adapter choice matters more for accuracy.

*Footer:* Measured on a Kaggle T4 GPU and a single CPU thread. These are a hardware proxy, not an edge device.

**Diagram:** none needed. Optional: a simple three-bar chart of the effects above (not in DIAGRAM_PLAN).

---

## Slide 20 — Limitations and what remains

1. **Pretraining overlap:** ImageNet pretraining overlaps MiniImageNet's test classes, which limits
   comparison with few-shot results from other pretraining setups.
2. **Two backbones:** we show that calibration depends on the backbone, not why.
3. **One shared hyperparameter recipe:** a controlled comparison, not per-configuration tuning.
4. **Three seeds:** small differences should be read cautiously.

> ### Most important next experiment
> Add more backbones to identify which backbone property explains the calibration effect.

**Diagram:** none.

---

## Slide 21 — Contributions

1. **Separated adapter architecture from parameter budget** with a pre-registered matched-budget
   experiment
2. **Separated the training objective from the scoring rule** for OOD detection, across all 120 models
3. **Decomposed the variation** in accuracy, calibration and OOD detection across five design factors
4. **Measured a two-parameter post-hoc recalibration** of the evidential head: how much it repairs, and
   what it costs OOD ranking
*Small type, visually set apart below items 1–4:*
5. A reproducible experimental framework: 189 runs, fixed test episodes, and retrained runs reproduce the
   original results exactly

**Diagram:** none.

---

## Slide 22 — What did we learn?

| Reliability aspect | Main factor | Evidence |
|---|---|---|
| **Accuracy** | Shots + adapter architecture | 76% of variation · bottleneck wins 8/8 at matched budget |
| **Calibration** | Head + backbone | 83% of variation · backbone dependence 3/4 cells, budget 0/4 |
| **OOD detection** | Scoring rule | 42.7% vs 0.17% of far-OOD variation (score vs objective) |
| **Calibration repair** | Post-hoc evidence refit | ECE improved 60/60 · still worse than softmax 60/60 |

### No single design choice optimises every reliability axis.

> The main contribution is not a universally superior model, but a controlled decomposition of which
> design choices govern which aspects of reliability in this few-shot PEFT regime.

**Diagram:** optional. The evidence column is what makes this slide different from Slide 2, so keep the
table. If there is room, add F16 small beside it: thick edges for effects found, crossed edges for
"budget → calibration" and "objective → OOD" (tested, not supported).

---

## Slide 23 — Thank you

### Thank you
### Questions?

**Diagram:** none.

---

# References (not counted in the 23 slides)

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[2] A. Howard *et al.*, "Searching for MobileNetV3," *ICCV*, 2019.

[3] S. Somvanshi *et al.*, "From Tiny Machine Learning to Tiny Deep Learning: A Survey," *ACM Computing
Surveys*, 2025.

[4] O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra, "Matching Networks for One Shot
Learning," *NIPS*, 2016.

[5] N. Houlsby *et al.*, "Parameter-Efficient Transfer Learning for NLP," *ICML*, 2019.

[6] E. J. Hu *et al.*, "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.

[7] M. Sensoy, L. Kaplan, and M. Kandemir, "Evidential Deep Learning to Quantify Classification
Uncertainty," *NIPS*, 2018.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *ICML*,
2017.

[9] W.-H. Li, X. Liu, and H. Bilen, "Cross-domain Few-shot Learning with Task-specific Adapters," *CVPR*,
2022.

[10] X. Linghu *et al.*, "Bayesian Evidential Learning for Few-Shot Classification," arXiv:2207.13137, 2022.

[11] P. Morales-Álvarez, S. Christodoulidis, M. Vakalopoulou, P. Piantanida, and J. Dolz, "BayesAdapter:
Enhanced Uncertainty Estimation in CLIP Few-Shot Adaptation," arXiv:2412.09718, 2025.

[12] J. Snell, K. Swersky, and R. S. Zemel, "Prototypical Networks for Few-shot Learning," *NIPS*, 2017.

[13] L. Bertinetto, J. F. Henriques, P. H. S. Torr, and A. Vedaldi, "Meta-learning with Differentiable
Closed-form Solvers," *ICLR*, 2019.

[14] S. Ravi and H. Larochelle, "Optimization as a Model for Few-Shot Learning," *ICLR*, 2017.

[15] Y. Netzer *et al.*, "Reading Digits in Natural Images with Unsupervised Feature Learning," *NIPS
Workshop on Deep Learning and Unsupervised Feature Learning*, 2011.

[16] Y. Le and X. Yang, "Tiny ImageNet Visual Recognition Challenge," Stanford CS231N, 2015.

[17] D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples
in Neural Networks," *ICLR*, 2017.

[18] W. Liu, X. Wang, J. D. Owens, and Y. Li, "Energy-based Out-of-distribution Detection," *NeurIPS*, 2020.

*All entries are in `docs/refs.bib`.*
