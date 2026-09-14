# 2. The Research Story

This page tells the story in order: what we believed at each stage, what the experiments showed, and why
we changed direction. Many early expectations were wrong. That is normal, and the thesis reports it
openly.

---

## 2.1 Timeline at a glance

| When (2026) | Stage | What happened, in one line |
|---|---|---|
| Up to Apr 21 | Proposal + pre-defence demo | One test episode suggested evidential was well calibrated and better at OOD detection. The calibration number came from a bug. |
| Apr 21 | Step 1 | Found and fixed two bugs in the evidential loss. ECE went from 0.526 to 0.167. |
| May 13–16 | Steps 2–3 | Cleaned up the code, added configs, W&B logging, reproducibility, and 600 fixed test episodes. |
| May–Jun | Step 4 | Moved to the real CIFAR-FS benchmark and episodic training. **The OOD advantage mostly disappeared.** |
| Jul 7 | Step 4.5 | Re-tested fairly with stronger baselines. Evidential beat softmax-probability scores at OOD, but calibration stayed poor. |
| Jul 21 | Steps 5–6 | Added LoRA, BitFit and the baselines, then tested adapter placement. Full fine-tuning did **not** overfit as predicted. Parallel placement won. |
| Jul 22 | Step 7 | Added Gaussian-noise OOD. The evidential advantage was bigger on near-OOD than far-OOD. |
| Jul 26 | Step 8 | Added MobileNetV3-Small. It lost far less accuracy than predicted. |
| Jul 30 | Step 9 | Added MiniImageNet. Found that ImageNet pretraining inflates accuracy. |
| Aug 6 | Step 10 | Ran the full 120-run grid. **Overturned** "evidential is on par with energy". Evidential calibration lost 20 of 20 comparisons. |
| Aug 9 | Step 11 | Measured speed and cost. The backbone drives latency; the adapter barely does. Found and fixed a silent data bug. |
| Aug 21–23 | Literature check | Found that **all four original research questions already had close prior work**. Drafted new questions. |
| Aug 25–26 | New experiments A + B | Ran the objective × score factorial, the calibration refit, and a rank sweep. One new hypothesis (RQ5) failed and was dropped. |
| Aug 27 | RQ3 matched-budget experiment | Pre-registered and ran 48 runs. **Overturned** "calibration follows parameter budget". |
| Sep 13 | Reporting | Supervisor report (PDF), defence slide plan, citation audit and bibliography committed. |
| Sep 14 | Clean-up | Removed stale documents; wrote this guide. |

---

## 2.2 Stage by stage

### Stage 0 — The proposal and the pre-defence demo

**What we believed.** Our planned contributions were:

- adding an evidential (Bayesian) head to a small adapter on a frozen CNN would give **better calibration**
  than softmax, even with under 500 trainable parameters;
- the same head would **improve near-OOD detection**;
- serial vs parallel adapter placement would be a key design choice;
- we would map a speed-vs-uncertainty trade-off for edge hardware.

The proposal also planned three backbones (plus ConvNeXt-Nano), four datasets (plus CUB-200 and ISIC), and
CIFAR-10-C robustness tests.

**What the demo showed.** One 5-way 5-shot episode on CIFAR-100 test images, ResNet-18 + bottleneck +
evidential head. Its calibration number (ECE 0.526) was broken by bugs.

### Stage 1 — Fix and build the foundation (Steps 1–3)

- **Step 1** found two bugs: the loss used cross-entropy on softplus outputs instead of the evidential
  loss, and the KL prior was switched off. After the fix, ECE fell from 0.526 to 0.167.
  - We also switched evidence from **ReLU to softplus**, because ReLU caused "dead" outputs that could not
    recover. This is a documented change from the proposal.
  - Evidential still could not beat softmax on calibration, but it was far better at OOD (+14 points
    AUROC).
- **Step 2** reorganised the code into packages, with YAML configs and 28 tests.
- **Step 3** added W&B logging, fixed seeds so reruns are byte-identical, and froze the 600 test episodes.
  - Numbers at this stage (still a stand-in setup): evidential accuracy 0.849 / AUROC 0.951, softmax
    accuracy 0.823 / AUROC 0.850.

### Stage 2 — The real benchmark breaks the first story (Steps 4 and 4.5)

**Step 4** replaced the stand-in with the proper CIFAR-FS split (64/16/20 classes) and true episodic
meta-training. Two forced design changes came with it:

- **Prototype head instead of a linear head.** A layer trained on 64 classes cannot classify 20 different
  test classes.
- **Evidential collapse bug.** Raw prototype scores are large negative numbers, so `softplus(score)` was
  about 0 everywhere. The model had no evidence and learned nothing. The fix was the `scale`/`bias` step
  before softplus.

**Result: the original claim did not survive.**

| | Accuracy | SVHN AUROC | ECE |
|---|---:|---:|---:|
| Evidential | 0.870 | 0.843 | 0.344 |
| Softmax | 0.875 | 0.838 | 0.119 |

The OOD gap was only +0.006, below our 0.05 bar, and evidential was worse calibrated.

**Step 4.5 ("settle the science")** asked whether that was a fair test. It added the two strongest
softmax baselines (temperature scaling, energy), a better-tuned evidential loss chosen on validation
episodes only, and two near-OOD sets.

- Evidential accuracy 0.884, ECE 0.285. Softmax ECE 0.082, or 0.041 after temperature scaling.
- Evidential's uncertainty beat MSP and TS-MSP on **every** OOD set, by +0.076 to +0.141 AUROC.
- Against energy it won 2 of 3 sets. *(This later turned out not to generalise — see Step 10.)*
- The tuning search found the calibration surface **flat** (ECE ≈ 0.285–0.296). We concluded calibration
  could not be fixed by tuning. *(RQ4 later showed that search was tuning the wrong knob.)*

### Stage 3 — More methods and placement (Steps 5–7)

- **Step 5** added LoRA, BitFit, Full Fine-Tuning and Linear Probe.
  - **Wrong prediction:** the proposal said full fine-tuning would overfit on 25 images. Instead it was
    the **most accurate** run (softmax 0.905). The reason: episodic meta-training learns from many
    training episodes, not from one 25-image task.
  - Changed story: adapters reach about 98% of full fine-tuning's accuracy with 660–2,330× fewer
    trainable parameters.
  - LoRA was the weakest method.
- **Step 6** tested *where* to put the bottleneck adapter.
  - Inside the blocks (serial or parallel) beat after-pooling by 3–4 points of accuracy.
  - Serial and parallel **tied** on accuracy (0.9145 vs 0.9146).
  - Parallel was better on OOD, so parallel became the default.
  - The 31,744-parameter parallel adapter slightly beat Full Fine-Tuning.
- **Step 7** added Gaussian-noise OOD. Evidential beat MSP on every set, and by **more on near-OOD** than
  far-OOD — what the proposal predicted.

### Stage 4 — New backbone and dataset (Steps 8–9)

- **Step 8, MobileNetV3-Small.**
  - **Wrong prediction:** we expected about a 7–8 point accuracy drop. It lost only 0.3–1.6 points on
    CIFAR-FS, with a 12× smaller frozen trunk.
  - Parallel placement became cheaper *and* better there (6,928 vs 19,024 parameters).
  - **Warning sign:** energy beat evidential uncertainty on 3 of 4 OOD sets.
- **Step 9, MiniImageNet.**
  - Accuracy was 4–9 points higher than on CIFAR-FS. MiniImageNet classes are ImageNet classes, which the
    backbone already saw in pretraining. This inflation is a limit we must state.
  - **Wrong prediction:** Step 8's "small backbone is almost free" did not hold here. The gap was
    4.7–5.7 points.
  - Energy beat evidential uncertainty in 9 of 16 comparisons.

### Stage 5 — The full grid (Step 10) and cost (Step 11)

**Step 10** ran every combination: 2 datasets × 2 shot counts × 2 backbones × 2 adapters × 2 output
readings, plus baselines. That is 40 settings × 3 seeds = **120 runs, all complete, 36.3 GPU-hours**.
It answered the original questions:

| Original question | Answer at grid scale |
|---|---|
| Orig-RQ1: which adapter | Parallel bottleneck beat LoRA in 16 of 16 matched comparisons (+2.1 to +8.3 points) |
| Orig-RQ2: evidential calibrates better? | **No — 0 of 20.** Evidential ECE was worse than plain softmax in 20 of 20 pairs |
| Orig-RQ3: evidential improves OOD? | **Split.** Beats MSP / TS-MSP in 37–38 of 40. But energy beats it in about 70% of comparisons |

**Correction.** Step 4.5's "evidential is on par with energy" was true for one setting only. It does not
generalise.

Also found: Full Fine-Tuning and Linear Probe give identical results for all 3 seeds. They have no random
starting weights, so their "3 seeds" are really 1.

**Step 11** measured parameters, FLOPs, latency and memory:

- The **backbone** changes latency by 5.12×. The **adapter** changes it by only 3.9–5.8%.
- Evidential scoring costs essentially nothing at inference (1.29%, below the 5.91% measurement noise).
- Recommended edge setting: MobileNetV3-Small + parallel + evidential on CIFAR-FS, 11.86 ms per image,
  6,930 parameters.
- **A silent bug:** three scripts picked a noisy laptop latency instead of the real Kaggle CPU number.
  Errors reached 47% on single cells, with no crash and no failed test. It was fixed, and the recommended
  points did not change.

### Stage 6 — The research questions change (Aug 21–26)

**Why.** A literature check found that each original question already had close prior work:

- placement was already answered by **TSA** (CVPR 2022) with the same backbone and protocol;
- the calibration and OOD patterns were already known (BEL, BayesAdapter, Liu et al. 2020);
- a Pareto plot is a presentation device, not a finding.

**What we did.** We kept every experiment and asked new questions of the same data. The balanced grid
lets us ask *what drives what* (attribution) instead of *does A beat B* (comparison). A five-question draft
was checked for novelty on Aug 23, then tested with two new Kaggle sessions:

- **Phase A (Aug 25–26).** We recovered 99 of the 120 saved models. For each, we computed all four OOD
  scores (the objective × score factorial, now RQ2) and refit the evidential `scale`/`bias` on validation
  episodes (now RQ4).
- **Phase B (Aug 26).** A rank sweep (ranks 1–64, 21 runs) tested an old idea: that calibration is best at
  a middle parameter budget (then "RQ5"). **No such middle optimum appeared.** RQ5 was demoted to "a tested
  hypothesis that did not survive".

That left **four** research questions, the final set.

### Stage 7 — Settling RQ3 (Aug 27)

RQ3's first evidence looked like **"calibration follows the larger parameter budget"**. But on the grid,
"bigger adapter" and "which backbone" always changed together, so the data could not tell them apart.

We wrote the decision rule down first (`docs/RQ3_MATCHED_BUDGET_PLAN.md`), then built both adapters at the
**same** budget on each backbone. 48 runs, 6.2 hours.

**Verdict: `backbone_intrinsic`.** The budget explanation fired in 0 of 4 cells; the backbone explanation
fired in 3 of 4. We had been leaning towards the budget explanation, and the pre-registered test ruled
against it. That makes it the most trustworthy result in the thesis.

---

## 2.3 Things we expected that turned out wrong

| # | What we expected | What actually happened | Where recorded |
|---|---|---|---|
| 1 | Evidential head calibrates **better** than softmax | Worse in 20 of 20 grid comparisons | Step 10; Orig-RQ2 |
| 2 | Evidential training is what makes OOD detection good | The **scoring rule** matters; the training objective barely does | RQ2 |
| 3 | Big OOD edge from the demo / Steps 1–3 | Almost gone on the real benchmark (+0.006) | Step 4 |
| 4 | A linear classifier head would work | Could not transfer to new classes; switched to prototype head | Step 4 |
| 5 | ReLU evidence as in the proposal | Dead outputs; switched to softplus | Step 1 |
| 6 | Full fine-tuning overfits on 25 images | It was the most accurate run | Step 5 |
| 7 | MobileNetV3-Small loses ~7–8 accuracy points | Lost 0.3–1.6 on CIFAR-FS (but 4.7–5.7 on MiniImageNet) | Steps 8–9 |
| 8 | Evidential uncertainty is on par with energy | Energy wins ~70% of comparisons | Step 10 |
| 9 | Calibration can't be improved by tuning (flat surface) | Refitting `scale`/`bias` improved ECE in 48 of 48 cases | RQ4 |
| 10 | Calibration is best at a middle parameter budget | No middle optimum in ranks 1–64 | Former RQ5 |
| 11 | Calibration follows the adapter's parameter budget | It follows the **backbone** | RQ3 |
| 12 | The worst evidential ECE was caused by an early checkpoint during KL warm-up | Did not reproduce in the grid's 4 matching cells | Step 10 |
| 13 | "3 seeds" gives a spread for every row | Full FT and Linear Probe have zero seed spread by design | Step 10 |
| 14 | The original four questions were new | Each had close prior work | Literature check, Aug 21–23 |

---

## 2.4 What was planned but never done

| Planned in the proposal | Status |
|---|---|
| ConvNeXt-Nano backbone | Not done |
| CUB-200 and ISIC datasets | Not done |
| CIFAR-10-C corruption robustness | Not done |
| Jetson Nano measurement | Not done; a single CPU thread used as a stand-in |
| "Under 500 trainable parameters" (Orig-RQ2) | Real adapters use 6,928–31,744. Only Linear Probe (0–2) is under 500 |
| Serial placement in the full grid | Tested only in Step 6 (one setting) |
| BitFit in the full grid | Tested only in Step 5 |
| Baselines (Full FT, Linear Probe) on every backbone and dataset | Only ResNet-18 + CIFAR-FS |
| A trained ViT comparison arm | Only a speed/size measurement (ViT-B/16, DeiT-Tiny), no accuracy |

These are listed as optional "Step 12" items in `progress.txt`. None were run, and none was formally
dropped. See `05_problems_and_open_work.md`.
