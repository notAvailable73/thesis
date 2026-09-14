# 3. Results

The final answers to every research question, in simple words, with the key numbers, what is and is not
new, and the exact wording that is safe to use.

---

## 3.1 The overarching question

> What decides reliability (accuracy, calibration, OOD detection) when you adapt a frozen lightweight
> CNN with a small adapter for few-shot image classification? And can the weak spots be fixed after
> training?

## 3.2 Summary of the four final research questions

| RQ | Question in one line | Answer in one line | Novelty | Strength rank |
|---|---|---|---|---|
| **RQ1** | Which design choice drives which outcome? | Number of shots drives accuracy. Output reading (softmax vs evidential) drives calibration. They barely interact. | Partly new | 3rd |
| **RQ2** | Is OOD detection caused by the training objective or the scoring rule? | The scoring rule, by far | Partly new | 2nd |
| **RQ3** | Does accuracy follow adapter design, and calibration follow parameter budget? | Accuracy follows design. Calibration follows **the backbone**, not the budget. | **New** | **1st** |
| **RQ4** | Can evidential calibration be fixed after training without breaking OOD ranking? | Improved in 48/48 cases; OOD ranking kept in 78% of comparisons. Still worse than plain softmax. | Partly new | 4th |

Main sources: `docs/RQ_SUPERVISOR_REPORT.md` and `docs/RQ_RESULTS_SUMMARY.md`.

---

## 3.3 RQ1 — What drives what

**Question.** Across the five design choices (dataset, shots, backbone, adapter, output reading), how much
of the variation in each outcome does each choice explain?

**What we did.** We used the 32 balanced grid settings × 3 seeds = 96 runs and computed η² (share of
variation explained) per factor, using per-seed data.

**Answer.**

| Outcome | Dataset | Shots | Backbone | Adapter | Output reading | Left over |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.8% | **76.1%** | 3.9% | 9.1% | 0.2% | 9.0% |
| ECE (calibration) | 2.0% | 2.1% | 4.6% | 0.6% | **82.9%** | 7.7% |
| Far-OOD AUROC | 1.2% | 22.7% | 11.5% | 1.7% | **39.0%** | 23.9% |
| Near-OOD AUROC (TinyImageNet) | 1.6% | **42.6%** | 0.6% | 22.0% | 21.0% | 11.4% |

- The output reading explains about **83% of calibration** but only **0.2% of accuracy**.
- Calibration and OOD detection are **independent** once the output reading is fixed. Across all 40
  settings the correlation looks positive (ρ = +0.43 / +0.48), but inside each reading it shrinks to
  +0.15 to +0.26.
- Evidential scoring is free at inference: a 1.29% latency difference, below the 5.91% measurement noise.

**Not new.** That accuracy and calibration have different causes (Guo et al. 2017; Minderer et al. 2021).
That evidential heads are cheap at inference (Sensoy et al. 2018).

**New.** The formal five-factor breakdown in a PEFT few-shot setting. The finding that calibration and OOD
become independent once the reading is fixed.

**Limits.** Main effects only (no interactions). One fixed training recipe. One of 96 values is missing
(see `05_problems_and_open_work.md`).

**Closest prior work to cite.** *One Model, Many Behaviors* (WACV 2026, arXiv:2601.10836) does a similar
ANOVA at larger scale, but not for PEFT, few-shot or calibration.

---

## 3.4 RQ2 — Training objective or scoring rule?

**Question.** Is good OOD detection caused by *how the model was trained* (softmax vs evidential loss), or
by *how the output is scored* (MSP, TS-MSP, energy, vacuity)?

**Why it matters.** In the grid, evidential runs were only scored with vacuity and softmax runs only with
softmax scores. Every comparison changed both things at once.

**What we did.** On 99 recovered models, we computed all 4 scores for both training objectives. That
gives a 2 × 4 design.

**Answer: the scoring rule dominates.**

| OOD type | Share explained by score | Share explained by training objective |
|---|---:|---:|
| Far-OOD | 43.7% | 0.27% |
| Near-OOD | 13.0% | 0.60% |

- Energy scores 0.911 AUROC on evidential-trained models and 0.929 on softmax-trained models (far-OOD):
  almost the same. MSP stays around 0.79 under both.
- Changing the score moves AUROC by about 0.13. Changing the training objective moves it by at most about
  0.02, and on far-OOD in the *wrong* direction.
- Energy beats vacuity head-to-head in 162 of 198 far-OOD and 183 of 198 near-OOD comparisons.

⚠️ **Do not quote "163×".** The reports give the score/objective ratio as 163× (far) and 22× (near).
The objective's share is so close to zero that the ratio is unstable. We recomputed it after removing the
one setting that is missing half its data (`cifar_fs/5shot/mobilenetv3_small/lora`): far-OOD becomes
**394×** and near-OOD **19×**. The conclusion does not change. Quote the η² percentages, or say "two to
three orders of magnitude".

**Not new.** Post-hoc scores often beat training changes, and energy beats MSP (Liu et al. 2020;
OpenOOD). The ANOVA method itself (WACV 2026).

**New.** Applying the energy score to Dirichlet (evidential) models, and finding it works almost the same
as on softmax models. No precedent found.

**Must cite.** arXiv:2603.07571 (names this exact gap as open), arXiv:2601.10836, and arXiv:2605.22746
(proves softmax is a special case of evidential, which explains why the objective barely matters).

**Coverage limit.** Only 99 of 120 models were recovered. All 21 missing ones are CIFAR-FS 5-shot adapter
models, which is the slice holding the project's most-quoted setting.

---

## 3.5 RQ3 — Adapter design, parameter budget, or backbone? *(strongest result)*

**Question.** When comparing two adapters, is the accuracy winner decided by adapter *design*, and the
calibration winner by adapter *size*?

**Step 1 — first evidence (16 matched pairs from the grid).** Bottleneck vs LoRA, with everything else
equal:

| Outcome | Bottleneck (design) wins | Larger adapter wins |
|---|---:|---:|
| Accuracy | **16 / 16** | 8 / 16 |
| Near-OOD AUROC | **16 / 16** | 8 / 16 |
| ECE (calibration) | 8 / 16 | **16 / 16** |

This looked like "calibration follows budget". But the bigger adapter is bottleneck on ResNet-18 and LoRA
on MobileNetV3-Small. So "bigger adapter" and "which backbone" always change together, and the grid could
not separate them.

**Step 2 — the matched-budget experiment (Aug 27).** We fixed the decision rule in writing first. Then we
built both adapters at the **same** size on each backbone, with sizes within 3.1% of each other (the
original pairs differed by 55–158%). MiniImageNet, 5-shot, 2 backbones × 2 budget levels × 2 adapters ×
2 readings × 3 seeds = **48 runs**.

| Cell | Calibration gap before matching | Gap at matched budget | Explanation that fired |
|---|---:|---:|---|
| ResNet-18 / evidential | +0.111 | **+0.112** | backbone |
| ResNet-18 / softmax | +0.108 | **+0.099** | backbone |
| MobileNetV3-S / evidential | −0.012 | **−0.006** | backbone |
| MobileNetV3-S / softmax | −0.015 | −0.006 | none (too small) |

*(Gap = LoRA ECE minus bottleneck ECE. Positive means LoRA is worse calibrated.)*

**Verdict: `backbone_intrinsic`**, in 3 of 4 cells (the pre-set threshold was 3).

- The budget explanation fired in **0 of 4**.
- "One design is always better calibrated" also fired in 0 of 4.

**What this means, simply.**

1. **Budget is not the cause.** On ResNet-18, equalising size did not shrink the gap at all.
2. **Budget is not irrelevant.** On MobileNetV3-Small the gap roughly halved. Budget changes *how big*
   the gap is; the backbone decides *which way* it goes.
3. **The two sides are not equal.** The ResNet-18 gap (~0.10) is about 10× bigger than the
   MobileNetV3-Small gap (~0.006).
4. **We don't know which backbone property causes it.** Depth, width, normalisation, block type and
   feature scale all differ between the two backbones. Two backbones cannot tell them apart. More
   backbones are needed, not more seeds.

**The design half held up.** At matched budget, bottleneck still won accuracy 8/8 and near-OOD 8/8, all
clearly beyond noise.

**Restated claim.** Accuracy and near-OOD ranking follow adapter **design**. Calibration follows the
**backbone**.

**Checks passed.**

- All 48 runs finished with no errors.
- Parameter counts were exactly as intended.
- Only the intended config keys differed between runs.
- The 18 runs that repeated grid settings reproduced the grid **exactly**.

**Why it's new.** No prior work found that separates adapter design from adapter size by outcome.
Deliberately using a reversed size ordering to do this appears to have no precedent. The pre-registration
makes the result hard to argue with.

**Sources.** `docs/RQ3_MATCHED_BUDGET_PLAN.md` (pre-registration), `step_writeups/rq3_matched_budget.txt`,
`results/rq3_matched/verdict.json`.

---

## 3.6 RQ4 — Fixing evidential calibration after training

**Question.** Can we improve the evidential head's calibration by refitting only its 2 numbers
(`scale`, `bias`) on validation episodes? And does the OOD ranking survive?

**Why the answer isn't obvious.** Vacuity depends on all 5 scores together, so changing scale and bias can
reorder which images look most "unknown".

**Answer.**

- ECE improved in **48 of 48** evidential cells. The average fell by 0.137 (from 0.327 to 0.190).
- Refitted scales were mostly about 7–14 (full range 3.6–14.5, median 8.3). The values the head had
  **learned during training** were much lower: scale 1.5–4.5, median 2.9. Training the affine jointly with
  the adapter, under the evidential loss, left it far from the NLL-optimal operating point. *(Older
  documents compared the refit to a "frozen default of 2". The affine was never frozen; see
  `01_what_we_built.md` §1.2.)*
- OOD AUROC stayed within 0.005 in **150 of 192** comparisons (78%). The average change was +0.004.
- Reordering does happen, but the worst case still had rank correlation 0.921.
- In the other 22% of comparisons AUROC dropped by more than 0.005. The worst drops were about −0.03.

⚠️ **Still worse than softmax — say this too.** We checked `results/rq_summary.json`. After refitting,
evidential ECE is still **worse than plain softmax in 48 of 48 cells** (on average 2.18× higher) and
**worse than temperature-scaled softmax in 48 of 48** (on average 11.9× higher). The refit shrinks the
gap. It does not close it. `RQ_SUPERVISOR_REPORT.md` §6 does not state this; it should.

**Correction of our own earlier work.** Step 4.5 swept the *loss* (KL weight × variance term) and found a
flat calibration surface. That search tuned the wrong knob: the evidence affine was never swept, and the
refit shows it is the part that controls calibration.

**Not new.** Post-hoc calibration in general. An improvement of this size (Guo et al. 2017 and BEL report
similar or bigger drops). That evidential defaults are poorly tuned.

**New.** A two-number refit of an evidential *prototype* head in frozen-backbone few-shot learning, with
OOD-ranking preservation **measured, not assumed**.

**Name collision to address.** Kull et al. 2019's "Dirichlet calibration" is unrelated to evidential
Dirichlet heads. Say so in the thesis.

---

## 3.7 A hypothesis that did not survive (former RQ5)

**Idea.** Calibration is best at a *middle* parameter budget, and that budget differs from the best one
for accuracy.

**Early evidence (confounded).** Across Linear Probe → LoRA → bottleneck → Full FT, ECE fell then rose.
But those points differ in adapter type too, not just size.

**Controlled test.** Same adapter, backbone and dataset; only the rank changed (1, 2, 4, 8, 16, 32, 64) ×
3 seeds = 21 runs.

- **No middle optimum.**
- Evidential ECE is lowest at rank 1 (0.291) and drifts up to rank 64 (0.309).
- Softmax ECE moves the other way (0.093 → 0.081).
- The best rank for accuracy is 64; the best for evidential ECE is 1.

**Safe wording.** "No interior optimum was observed in the tested range." Three seeds may be too few to
rule out a subtle one.

**A paper that partly disagrees.** *LoRA-Ensemble* (arXiv:2405.14438) sees calibration worsen at rank 32 on
CIFAR-100. Our response (`RQ_SUPERVISOR_REPORT.md` §7.1): they use a ViT with attention LoRA and a trained
linear head; we use a CNN, 1×1 conv adapters and a parameter-free head. Do **not** use the old argument
comparing to Full FT; it relies on the confounded table.

---

## 3.8 The original proposal questions (still answered)

| Original question | Answer | Why it is not the thesis's main claim |
|---|---|---|
| **Orig-RQ1** Adapter placement | Serial and parallel **tie** on accuracy (Step 6, one setting); parallel is better on OOD. In the grid, parallel bottleneck beats LoRA **16/16** (+2.1 to +8.3 points). On MobileNetV3-Small it is also cheaper. A 31.7k-parameter adapter beats Full FT at 5-shot on ResNet-18 (91.44% vs 90.47%). | TSA (CVPR 2022) already showed parallel wins, with the same backbone and protocol |
| **Orig-RQ2** Evidential calibrates better? | **No — 0/20.** Worse than softmax by 1.4–9.1×, worse than TS-softmax by 5.3–51×. | Bayesian calibration effects are well studied; our negative is a boundary case |
| **Orig-RQ3** Bayesian prior helps near-OOD? | Vacuity beats MSP / TS-MSP in 37–38 of 40 settings, more so at 1-shot. **But energy beats vacuity in ~70%** of comparisons. | Standard evidential and energy results |
| **Orig-RQ4** Speed vs uncertainty trade-off | Backbone drives latency (5.12×); adapter barely does (3.9–5.8%). Recommended: MobileNetV3-Small + parallel + evidential, 11.86 ms, 6,930 params (CIFAR-FS); ResNet-18 + parallel + evidential, 62.38 ms (MiniImageNet). | A Pareto plot is a presentation, not a finding |

⚠️ **Wording check.** `RQ_SUPERVISOR_REPORT.md` Appendix A labels Orig-RQ1 as "serial versus parallel"
and then gives "Parallel wins 16/16". That 16/16 is **parallel bottleneck vs LoRA**, not serial vs
parallel. Serial was never in the grid.

### How the three numbering schemes map

| Final (use this) | Five-RQ draft (retired, Aug 23) | Closest original question (our reading, not an official mapping) | Data used |
|---|---|---|---|
| RQ1 — what drives what | RQ4 | — (new question) | Step 10 grid |
| RQ2 — objective vs score | RQ1 | related to Orig-RQ3 | Phase A factorial |
| RQ3 — design vs budget vs backbone | RQ3 | related to Orig-RQ1 | Grid + matched-budget experiment |
| RQ4 — post-hoc refit | RQ2 | related to Orig-RQ2 | Phase A refit |
| (dropped) interior optimum | RQ5 | — | Phase B rank sweep |
| — | — | Orig-RQ4 speed | Step 11 (kept as supporting result) |

---

## 3.9 How we compare to published few-shot methods

From `docs/RESULTS_MASTER.md` §4 and `docs/DEFENCE_BRIEF.md`. Same 5-way protocol as P>M>F
(arXiv:2204.07305).

- Our ResNet-18 parallel adapter (31,744 trainable params) is **1.06 points behind** a ViT-S model on
  CIFAR-FS 5-shot that trains **662× more** parameters.
- It is **3.56 points ahead** of the backbone-matched ResNet-50 on MiniImageNet 5-shot, with 788× fewer
  trainable parameters.
- It costs more at 1-shot and with MobileNetV3-Small on MiniImageNet (−7.90 points at 5-shot, −18.18 at
  1-shot vs ViT-S).
- **Caveat.** Our backbones are ImageNet-pretrained, so absolute accuracies can't be compared with methods
  trained from scratch.

The "CNNs are outdated" objection is answered in `docs/DEFENCE_BRIEF.md`.

---

## 3.10 Safe wording — say this, not that

| Don't say | Say |
|---|---|
| "Calibration follows the parameter budget." | "Calibration follows the backbone. Budget changes the size of the effect; the backbone decides its direction." |
| "Evidential uncertainty is on par with energy." | "Vacuity is a much better OOD score than softmax-probability scores, but a good logit-based score like energy still beats it." |
| "Evidential training improves OOD detection." | "The OOD gain comes from the scoring rule, not the training objective." |
| "The score matters 163× more than the objective." | "The score explains 43.7% (far) / 13.0% (near) of the variation; the objective under 1%." |
| "Refitting fixes evidential calibration." | "Refitting improves it in 48/48 cells, but it stays worse than plain softmax in all 48." |
| "Refitting always preserves the OOD ranking." | "It survives in 78% of comparisons, with a measured minority exception." |
| "Our 6,928-parameter adapter beats full fine-tuning." | "It matches full fine-tuning. The margin is inside its own seed spread. The ResNet-18 +0.98 points is the margin that clears noise." |
| "Calibration error has an interior optimum." | "No interior optimum was observed in the tested range." |
| "We discovered that calibration and accuracy have different drivers." | "We formally decompose it in this setting; the general pattern goes back to Guo et al. 2017." |
| "We're the first to do a factorial ANOVA on OOD." | "Concurrent WACV 2026 work does this at larger scale. What has no precedent is applying energy to Dirichlet models." |
| "We know why the backbone controls calibration." | "The backbone matters; which property of it is responsible is not identified." |
| "All four research questions were confirmed." | "All four were answered. Several not in the direction we expected." |

---

## 3.11 Full grid tables (all 40 cells)

The per-cell numbers behind every RQ, generated from `results/mvt_results.json`. Each value is the mean
over 3 training seeds; each seed is scored on the same 600 test episodes.

**How to read the columns.**

- **Accuracy**: mean over episodes, ± the SD across the 3 seeds. The 95% CI over 600 episodes is a separate,
  larger quantity, stored as `accuracy_ci95` (about ±0.8 points at 1-shot).
- **ECE**: pooled over all 45,000 query predictions, 15 bins.
- **TS-ECE**: ECE after temperature scaling (softmax rows only).
- **Native** OOD score: vacuity for evidential rows, MSP for softmax rows. Energy is only stored for softmax
  rows; energy on evidential models is in RQ2 (§3.4), not here.
- Full FT and Linear Probe have seed SD 0.00 because their seed axis does nothing (§4.3).

<!-- Generated 2026-09-14 from results/mvt_results.json; regenerate rather than hand-edit. -->

### CIFAR-FS

| Shot | Backbone | Adapter | Head | Params | Accuracy % (± seed SD) | ECE | TS-ECE | SVHN AUROC (native) | TIN AUROC (native) | TIN AUROC (energy) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ResNet-18 | Bottleneck-par. | softmax | 31,744 | 78.57 ± 0.48 | 0.056 | 0.030 | 0.784 | 0.738 | 0.821 |
| 1 | ResNet-18 | Bottleneck-par. | evidential | 31,746 | 79.19 ± 0.43 | 0.277 | — | 0.876 | 0.831 | — |
| 1 | ResNet-18 | LoRA | softmax | 12,288 | 75.44 ± 0.37 | 0.097 | 0.025 | 0.747 | 0.710 | 0.785 |
| 1 | ResNet-18 | LoRA | evidential | 12,290 | 73.76 ± 1.09 | 0.297 | — | 0.833 | 0.771 | — |
| 1 | ResNet-18 | Full FT | softmax | 11,176,512 | 81.14 ± 0.00 | 0.085 | 0.022 | 0.835 | 0.793 | 0.837 |
| 1 | ResNet-18 | Full FT | evidential | 11,176,514 | 80.36 ± 0.00 | 0.321 | — | 0.880 | 0.831 | — |
| 1 | ResNet-18 | Linear Probe | softmax | 0 | 70.25 ± 0.00 | 0.282 | 0.064 | 0.672 | 0.728 | 0.893 |
| 1 | ResNet-18 | Linear Probe | evidential | 2 | 70.25 ± 0.00 | 0.440 | — | 0.682 | 0.868 | — |
| 1 | MNv3-S | Bottleneck-par. | softmax | 6,928 | 78.80 ± 0.32 | 0.050 | 0.032 | 0.750 | 0.787 | 0.863 |
| 1 | MNv3-S | Bottleneck-par. | evidential | 6,930 | 78.88 ± 0.07 | 0.254 | — | 0.832 | 0.870 | — |
| 1 | MNv3-S | LoRA | softmax | 10,752 | 75.43 ± 0.31 | 0.029 | 0.040 | 0.756 | 0.698 | 0.779 |
| 1 | MNv3-S | LoRA | evidential | 10,754 | 74.03 ± 0.52 | 0.216 | — | 0.818 | 0.747 | — |
| 5 | ResNet-18 | Bottleneck-par. | softmax | 31,744 | 91.44 ± 0.14 | 0.067 | 0.015 | 0.885 | 0.852 | 0.929 |
| 5 | ResNet-18 | Bottleneck-par. | evidential | 31,746 | 91.58 ± 0.17 | 0.301 | — | 0.917 | 0.926 | — |
| 5 | ResNet-18 | LoRA | softmax | 12,288 | 86.25 ± 0.17 | 0.102 | 0.016 | 0.820 | 0.782 | 0.876 |
| 5 | ResNet-18 | LoRA | evidential | 12,290 | 83.33 ± 1.07 | 0.329 | — | 0.890 | 0.849 | — |
| 5 | ResNet-18 | Full FT | softmax | 11,176,512 | 90.47 ± 0.00 | 0.073 | 0.014 | 0.894 | 0.867 | 0.914 |
| 5 | ResNet-18 | Full FT | evidential | 11,176,514 | 88.82 ± 0.00 | 0.338 | — | 0.909 | 0.897 | — |
| 5 | ResNet-18 | Linear Probe | softmax | 0 | 87.41 ± 0.00 | 0.448 | 0.030 | 0.855 | 0.843 | 0.926 |
| 5 | ResNet-18 | Linear Probe | evidential | 2 | 87.41 ± 0.00 | 0.623 | — | 0.708 | 0.886 | — |
| 5 | MNv3-S | Bottleneck-par. | softmax | 6,928 | 90.74 ± 0.17 | 0.070 | 0.007 | 0.845 | 0.886 | 0.951 |
| 5 | MNv3-S | Bottleneck-par. | evidential | 6,930 | 90.24 ± 0.18 | 0.304 | — | 0.944 | 0.919 | — |
| 5 | MNv3-S | LoRA | softmax | 10,752 | 88.05 ± 0.39 | 0.062 | 0.006 | 0.829 | 0.810 | 0.837 |
| 5 | MNv3-S | LoRA | evidential | 10,754 | 86.97 ± 0.74 | 0.288 | — | 0.930 | 0.880 | — |

### MiniImageNet

| Shot | Backbone | Adapter | Head | Params | Accuracy % (± seed SD) | ECE | TS-ECE | SVHN AUROC (native) | TIN AUROC (native) | TIN AUROC (energy) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ResNet-18 | Bottleneck-par. | softmax | 31,744 | 85.03 ± 0.52 | 0.101 | 0.007 | 0.816 | 0.810 | 0.856 |
| 1 | ResNet-18 | Bottleneck-par. | evidential | 31,746 | 84.81 ± 0.08 | 0.307 | — | 0.914 | 0.870 | — |
| 1 | ResNet-18 | LoRA | softmax | 12,288 | 80.29 ± 0.35 | 0.175 | 0.008 | 0.809 | 0.774 | 0.789 |
| 1 | ResNet-18 | LoRA | evidential | 12,290 | 79.16 ± 0.37 | 0.371 | — | 0.960 | 0.774 | — |
| 1 | MNv3-S | Bottleneck-par. | softmax | 6,928 | 74.92 ± 0.12 | 0.065 | 0.006 | 0.733 | 0.743 | 0.861 |
| 1 | MNv3-S | Bottleneck-par. | evidential | 6,930 | 75.61 ± 0.34 | 0.253 | — | 0.881 | 0.856 | — |
| 1 | MNv3-S | LoRA | softmax | 10,752 | 72.45 ± 0.43 | 0.024 | 0.012 | 0.638 | 0.689 | 0.776 |
| 1 | MNv3-S | LoRA | evidential | 10,754 | 72.48 ± 0.16 | 0.219 | — | 0.832 | 0.790 | — |
| 5 | ResNet-18 | Bottleneck-par. | softmax | 31,744 | 95.56 ± 0.05 | 0.085 | 0.006 | 0.911 | 0.912 | 0.962 |
| 5 | ResNet-18 | Bottleneck-par. | evidential | 31,746 | 95.88 ± 0.12 | 0.294 | — | 0.973 | 0.958 | — |
| 5 | ResNet-18 | LoRA | softmax | 12,288 | 91.56 ± 0.49 | 0.193 | 0.019 | 0.918 | 0.849 | 0.884 |
| 5 | ResNet-18 | LoRA | evidential | 12,290 | 88.32 ± 0.35 | 0.405 | — | 0.984 | 0.869 | — |
| 5 | MNv3-S | Bottleneck-par. | softmax | 6,928 | 90.10 ± 0.34 | 0.111 | 0.021 | 0.795 | 0.844 | 0.915 |
| 5 | MNv3-S | Bottleneck-par. | evidential | 6,930 | 90.64 ± 0.14 | 0.318 | — | 0.910 | 0.911 | — |
| 5 | MNv3-S | LoRA | softmax | 10,752 | 87.96 ± 0.14 | 0.095 | 0.022 | 0.751 | 0.792 | 0.863 |
| 5 | MNv3-S | LoRA | evidential | 10,754 | 87.96 ± 0.27 | 0.305 | — | 0.912 | 0.890 | — |

**Three things to see in these tables before reading any RQ.**

1. **The evidential ECE column never drops below 0.21, and the softmax column never rises above 0.20**
   except for Linear Probe. This is RQ1's 83%, visible by eye.
2. **Temperature scaling brings softmax ECE to 0.006–0.064.** Temperature-scaled softmax is the calibration
   bar every evidential number is compared against.
3. **Bottleneck-par. beats LoRA on accuracy in all 16 matched rows**, even on MobileNetV3-Small, where it
   has fewer parameters (6,928 vs 10,752). This is the observation RQ3 starts from.

Macro-F1, Brier, FPR@95 and the CIFAR-100/MiniImageNet-held-out pools are in `docs/RESULTS_MASTER.md`.
