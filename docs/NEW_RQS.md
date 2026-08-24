# Proposed Research Questions for Verification

**Project:** B-PEFT — parameter-efficient fine-tuning of frozen lightweight CNN backbones for few-shot vision, evaluated for accuracy, calibration, and OOD detection.

**Purpose of this document:** These five research questions are proposed replacements for the project's original four, which a literature check found were largely pre-empted by prior work. This document is written to be **self-contained for an independent reviewer with no access to the repository**. It states the experimental setup, the raw numbers, the novelty claims, and — deliberately — the strongest counter-explanations we are aware of.

**Numbering.** RQ1–RQ5 below are ranked strongest-to-weakest by an independent novelty cross-check performed 2026-08-23 (10-15+ literature searches per question). Each RQ carries a short **Novelty check** callout with that verdict. This numbering replaces the earlier internal RQ-A…RQ-E labels used while drafting; the *original* four proposal RQs referenced in §3 and §5 are labeled **Orig-RQ1…Orig-RQ4** throughout this document to avoid collision with the new numbering.

**What we are asking the reviewer to do** is listed in §7. In short: attack the novelty claims and the causal interpretations, not the arithmetic.

---

## 1. Experimental setup

Every number in this document comes from one completed experiment grid.

### 1.1 Model

```
frozen ImageNet-pretrained CNN backbone
        ↓
small trainable adapter          ← the only trained parameters
        ↓
parameter-free prototype head    ← classifies by similarity to support-set class means
```

The backbone is **never updated** (except in the Full-FT baseline). The head has **no trainable weights**: classification logits are the similarity between a query embedding and the mean embedding of each class's support images. So the trainable-parameter count of a configuration is essentially the adapter's size.

**Head "interpretation"** is a separate axis from head type. The prototype head always emits raw similarity logits. Those logits are then read in one of two ways:

- **Softmax** — logits → softmax → predictive probabilities. Confidence = max softmax probability.
- **Evidential (Dirichlet)** — `evidence = softplus(logits × scale + bias)`, `α = evidence + 1`, `S = Σα`. Predictive probability = `α/S`. Uncertainty = *vacuity* = `K/S` where K = 5.

The evidential mapping adds exactly **2 trainable parameters** (`scale`, `bias`), which is why evidential cells show counts like 31,746 against the softmax cell's 31,744. These two parameters are frozen at (2, −6) across the whole grid, tuned once on a validation split in an earlier step.

### 1.2 Protocol

| | |
|---|---|
| Task | 5-way {1, 5}-shot episodic classification |
| Test episodes | 600 fixed episodes, seeds frozen in a version-controlled file |
| Training seeds | 3 per configuration (42, 43, 44) |
| Datasets | CIFAR-FS (Bertinetto split), MiniImageNet (Ravi & Larochelle split) |
| Backbones | ResNet-18 (11.7M params), MobileNetV3-Small (2.5M params) — both ImageNet-pretrained, frozen |
| Hyperparameters | **One frozen recipe across all 40 configurations.** Not re-tuned per cell. |

Test classes are disjoint from the classes the adapter meta-trains on.

### 1.3 The grid

40 unique configurations × 3 seeds = **120 runs**, all completed, no missing cells.

- 2 datasets × 2 shot regimes × 2 backbones × 2 adapters × 2 head interpretations = 32 cells (**balanced full factorial**)
- \+ 8 baseline cells (Full-FT, Linear-Probe) on CIFAR-FS × ResNet-18 only (unbalanced, so excluded from the variance decomposition)

**Adapters compared:**
- **Bottleneck-parallel** — a down-project → ReLU → up-project residual branch alongside a conv block.
- **LoRA** — low-rank update applied to a 1×1 convolution.
- **Full-FT** (baseline) — all backbone parameters trainable.
- **Linear-Probe** (baseline) — nothing trainable except the 2 evidence-affine parameters in the evidential variant.

### 1.4 Metrics

- **Accuracy**, macro-F1 — mean over 600 episodes.
- **ECE (pooled)** — expected calibration error; the average gap between claimed confidence and observed correctness. **Lower is better.** This is the "is the model honest about its confidence" metric.
- **Brier score**.
- **OOD AUROC** — how well the model's uncertainty score separates in-distribution query images from OOD images. **Higher is better.** OOD pools: SVHN and Gaussian noise (far-OOD); CIFAR-100-heldout / MiniImageNet-heldout and TinyImageNet (near-OOD).

**Uncertainty scores.** Evidential cells are scored with **vacuity** (`K/S`). Softmax cells are scored with **MSP** (max softmax probability), **TS-MSP** (after temperature scaling), and **energy** (`logsumexp` of logits). Note this asymmetry — it is the subject of RQ1.

---

## 2. The structural fact that makes RQ3 and RQ5 possible

Trainable parameter counts (softmax variant; evidential adds 2):

| Backbone | Bottleneck-parallel | LoRA | Which adapter is larger? |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck, by 2.6× |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA**, by 1.6× |

**The ordering reverses between backbones.** This was not designed — it falls out of the two backbones' channel widths. It means adapter *architecture* and adapter *parameter budget* are partially deconfounded within the existing grid, which is normally impossible in a PEFT study because the "better" adapter is usually also the bigger one everywhere.

This document's central claims rest on that reversal. §7 asks the reviewer to attack it.

---

## 3. Status of the original research questions

Included so the reviewer can judge whether the replacements are an improvement.

| Original RQ | Finding obtained | Prior work that pre-empts it |
|---|---|---|
| **Orig-RQ1** Adapter placement, serial vs parallel | Parallel wins 16/16 | [TSA, Li/Liu/Bilen CVPR 2022](https://arxiv.org/abs/2107.00358) — frozen ResNet-18, 600 episodic tasks, parameter-free nearest-centroid head, finds parallel/residual wins "in almost all cases". Same backbone, same protocol, same answer. |
| **Orig-RQ2** Does evidential calibrate better than softmax under a tiny budget | No — evidential worse in 20/20 matched pairs, by 1.4×–9.1× | EDL calibration is well studied. [BEL](https://ar5iv.labs.arxiv.org/html/2207.13137) and [BayesAdapter](https://arxiv.org/abs/2412.09718) both report Bayesian calibration *improving*. Our negative reads as a boundary case of a known effect. |
| **Orig-RQ3** Does a Bayesian prior improve near-OOD detection | Vacuity beats MSP in 37–38/40, but loses to energy in ~70% of comparisons | Vacuity > MSP is the standard EDL claim; energy > MSP is [Liu et al. 2020](https://arxiv.org/pdf/2010.03759). Confirmation at scale, not discovery. |
| **Orig-RQ4** Latency vs uncertainty-quality Pareto frontier | Backbone drives latency (5.1×); adapter choice does not (3.9%) | Pareto reporting is a standard device, not a research finding. |

**Diagnosis.** All four are *comparison* questions ("does A beat B"). The replacements below are *attribution* and *structure* questions ("which factor controls which outcome, and is there an optimum"), which the balanced factorial design supports and which are far less crowded in the literature.

---

## 4. The proposed research questions

Ranked strongest-to-weakest by the 2026-08-23 novelty cross-check.

### RQ1 — Is the uncertainty benefit produced by the training objective (evidential vs. softmax) or by the OOD scoring rule (vacuity, MSP, energy) — and can the two be separated?

**Question.** Is the observed uncertainty-detection behaviour attributable to the *training objective* (evidential/Dirichlet vs softmax cross-entropy) or to the *readout score* (vacuity vs MSP vs energy) — and can the two be separated?

**The confound in the current results.** The grid scores evidential runs with vacuity only, and softmax runs with MSP/TS-MSP/energy only. Every reported comparison therefore varies objective and score *together*. When the current results say "energy beats vacuity in ~70% of cells", it is not possible to tell whether energy is a better score or whether softmax training produces better-separated features. The fix is to compute all four scores on all runs, giving a clean **2 objectives × 4 scores** factorial.

**Novelty anchor.** This gap is explicitly named as an open limitation by a recent systematic study: [A Systematic Comparison of Training Objectives for OOD Detection in Image Classification (2026)](https://arxiv.org/html/2603.07571v2) states that in its own evaluation the training objective and OOD scoring rule "are not fully factorized, since each objective is evaluated with the confidence measure most natural to its output space." That study uses ResNet-18 under standardized OpenOOD protocols, compares Cross-Entropy/Triplet/Prototype/Average-Precision losses — **no evidential/Dirichlet objective at all** — and does **not** cover frozen backbones, few-shot episodic evaluation, or parameter-efficient adaptation.

**Status.** One diagonal of the 2×4 table exists. The cross terms do not.

**Extra compute required.** Evaluation-only *if* the grid's per-run checkpoints still exist (the training script writes them; they are not present in the local working copy and may still be on the cloud training host). Otherwise this requires re-running the grid, ≈ 36 GPU-hours.

**Novelty check (2026-08-23).** ✅ Confirmed novel, no risk flags. The arXiv:2603.07571 quote was verified directly against the paper's raw text and is accurately paraphrased — but note precisely: that paper flags this gap for a *different* objective family (metric/ranking losses, not evidential), so it is a strong motivating analogy, not existing coverage. No paper found computing the evidential×softmax cross with vacuity/MSP/TS-MSP/energy. Closest adjacent work (`arXiv:2605.06382`, vacuity cardinality-artefact critique; OpenOOD/OpenOOD-1.5) does not cross scores onto Dirichlet-trained models either.

---

### RQ2 — Can an evidential head be recalibrated after training by refitting only its two evidence-affine parameters, and does its OOD ranking survive that recalibration?

**Question.** Softmax outputs can be recalibrated post-hoc by temperature scaling; the current results have no equivalent for evidential outputs, which makes every reported calibration comparison structurally unfavourable to the evidential head. The evidence mapping `softplus(logits × scale + bias)` already contains exactly two free parameters. If they are fitted on a validation split rather than frozen, does ECE improve — and does the OOD ranking survive?

**Why the answer is not obvious.** The affine is monotone in each individual logit, so one might expect ranking preservation for free. But vacuity is `K/Σα`, a function of *all* K logits jointly, so a per-logit monotone transform does **not** guarantee that the ordering of vacuity across samples is preserved. Whether ranking survives is therefore an empirical question, not a derivation.

**Both outcomes are publishable.** If ECE drops and AUROC survives, this is a two-parameter post-hoc recalibration for evidential heads — a Dirichlet analogue of temperature scaling, cheap enough for edge deployment. If AUROC collapses, it quantifies a calibration/ranking trade in evidential uncertainty that we have not seen documented.

**Novelty positioning — this is the narrowest claim of the five and must be scoped carefully.** Post-hoc calibration is a mature field: Dirichlet calibration (Kull et al. 2019, NeurIPS) and [accuracy-preserving post-hoc calibration via invertible logit transforms](https://arxiv.org/html/2608.10372) both exist. The claim is **not** "first post-hoc calibration". It is specifically: a two-parameter recalibration of an *evidential prototype head* in a frozen-backbone few-shot regime, with OOD-ranking preservation measured rather than assumed.

**Extra compute required.** Validation fit plus re-evaluation. Small. The mechanism is already implemented in the codebase.

**Novelty check (2026-08-23).** ✅ Confirmed novel, no risk flags. Verified Kull et al.'s "Dirichlet calibration" is a *different* use of the word — a generic calibration map applicable to any softmax classifier, unrelated to evidential-deep-learning Dirichlet evidence — so it does not pre-empt this claim. `arXiv:2608.10372` confirmed to apply only to standard softmax classifiers, no evidential/vacuity content. No paper found proving or disproving ranking preservation for a sum-of-affine-transformed-logits construction like vacuity. Closest-but-non-matching prior work (density-informed EDL recalibration, `arXiv:2602.01477`; Invascal, `arXiv:2606.00069`) use architectural or training-time fixes, not a post-hoc two-parameter refit.

---

### RQ3 — Which property of a parameter-efficient adapter governs accuracy, and which governs calibration — its architecture, or its trainable-parameter budget?

**Question.** In parameter-efficient adaptation of a frozen backbone, accuracy and calibration are usually assumed to improve together with a "better" adapter. Do they? Or are they controlled by different properties of the adapter — its architecture versus its parameter count?

**Hypothesis (supported by the evidence below).** They are dissociated: accuracy and OOD ranking follow the adapter's architecture; calibration follows its parameter budget.

**Evidence.** 16 matched bottleneck-vs-LoRA comparisons (2 datasets × 2 shot regimes × 2 backbones × 2 head interpretations). Head interpretation is held fixed within each pair.

| Dataset | Shots | Backbone | Head | Btl params | LoRA params | Larger arm | ECE btl | ECE LoRA | Better-calibrated arm |
|---|---|---|---|---:|---:|---|---:|---:|---|
| CIFAR-FS | 1 | MobileNetV3-S | evid. | 6,930 | 10,754 | LoRA | 0.2540 | **0.2156** | LoRA |
| CIFAR-FS | 1 | MobileNetV3-S | softmax | 6,928 | 10,752 | LoRA | 0.0499 | **0.0287** | LoRA |
| CIFAR-FS | 1 | ResNet-18 | evid. | 31,746 | 12,290 | bottleneck | **0.2765** | 0.2970 | bottleneck |
| CIFAR-FS | 1 | ResNet-18 | softmax | 31,744 | 12,288 | bottleneck | **0.0560** | 0.0969 | bottleneck |
| CIFAR-FS | 5 | MobileNetV3-S | evid. | 6,930 | 10,754 | LoRA | 0.3043 | **0.2879** | LoRA |
| CIFAR-FS | 5 | MobileNetV3-S | softmax | 6,928 | 10,752 | LoRA | 0.0703 | **0.0616** | LoRA |
| CIFAR-FS | 5 | ResNet-18 | evid. | 31,746 | 12,290 | bottleneck | **0.3010** | 0.3294 | bottleneck |
| CIFAR-FS | 5 | ResNet-18 | softmax | 31,744 | 12,288 | bottleneck | **0.0670** | 0.1016 | bottleneck |
| MiniImageNet | 1 | MobileNetV3-S | evid. | 6,930 | 10,754 | LoRA | 0.2534 | **0.2194** | LoRA |
| MiniImageNet | 1 | MobileNetV3-S | softmax | 6,928 | 10,752 | LoRA | 0.0650 | **0.0242** | LoRA |
| MiniImageNet | 1 | ResNet-18 | evid. | 31,746 | 12,290 | bottleneck | **0.3073** | 0.3708 | bottleneck |
| MiniImageNet | 1 | ResNet-18 | softmax | 31,744 | 12,288 | bottleneck | **0.1012** | 0.1748 | bottleneck |
| MiniImageNet | 5 | MobileNetV3-S | evid. | 6,930 | 10,754 | LoRA | 0.3177 | **0.3053** | LoRA |
| MiniImageNet | 5 | MobileNetV3-S | softmax | 6,928 | 10,752 | LoRA | 0.1107 | **0.0955** | LoRA |
| MiniImageNet | 5 | ResNet-18 | evid. | 31,746 | 12,290 | bottleneck | **0.2938** | 0.4049 | bottleneck |
| MiniImageNet | 5 | ResNet-18 | softmax | 31,744 | 12,288 | bottleneck | **0.0850** | 0.1930 | bottleneck |

**Summary across the same 16 pairs:**

| Outcome | Winner is the bottleneck architecture | Winner is the larger-budget arm |
|---|---:|---:|
| Accuracy | **16 / 16** | 8 / 16 |
| OOD AUROC (near-OOD, native score) | **16 / 16** | 8 / 16 |
| OOD AUROC (TinyImageNet near) | **16 / 16** | 8 / 16 |
| OOD AUROC (SVHN far) | 11 / 16 | 7 / 16 |
| ECE (pooled) | 8 / 16 | **16 / 16** |

The SVHN far-OOD row is reported for completeness and is **not** consistent with either hypothesis — it is close to chance on both. Far-OOD separation appears to be driven by something outside the adapter axis entirely (the η² table in RQ4 attributes 40.8% of far-OOD variance to the head interpretation and only 1.8% to the adapter). Only the near-OOD rows support the architecture claim.

**The precise claim, stated so it cannot be overread.** The "8/16" entries are *arithmetically implied*, not independent evidence: because the budget ordering reverses between the two backbones, any outcome that is perfectly consistent with architecture is necessarily 8/16 on budget, and vice versa. The real content is the contrast between two directly observed facts:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both backbones despite having 2.6× *more* parameters on one and 1.6× *fewer* on the other.
2. **The ECE winner does change, and changes exactly in step with the budget ordering.** LoRA is better calibrated in all 8 MobileNetV3-Small pairs (where LoRA is larger); bottleneck is better calibrated in all 8 ResNet-18 pairs (where bottleneck is larger).

**Statistical strength.** Each direction is a 16/16 sign consistency, two-sided p ≈ 3.05×10⁻⁵ under a null of random direction. However, the *magnitudes* are weaker: |ΔECE| exceeds 2× the pooled across-seed standard deviation in only **10 / 16** pairs — all 8 MiniImageNet pairs, but only 2 of 8 CIFAR-FS pairs. The direction is robust; the per-pair effect sizes on CIFAR-FS are not.

**Novelty claim.** We could not find prior work dissociating adapter architecture from adapter budget by outcome metric. The closest work found ([CP tensor adapters, 2026](https://arxiv.org/html/2606.00428v1)) studies the accuracy–budget curve's resolution but does not measure calibration.

**Extra compute required.** None. The result is entirely within existing data.

**Novelty check (2026-08-23).** ✅ Mostly confirmed novel, one open item. Verified the CP tensor adapters characterization is accurate (accuracy-only, single backbone, no reversal design). No paper found combining a cross-backbone parameter-ordering reversal with calibration as the outcome — closest partial hits (LoRA-Ensemble, `arXiv:2405.14438`, budget-vs-ECE within one architecture; "On Fairness of Low-Rank Adaptation", `arXiv:2405.17512`, multi-backbone calibration but LoRA-vs-full-FT not architecture-vs-architecture) don't combine all three ingredients. **Open item:** a paper titled "Robust Calibration of Large Vision-Language Adapters" (ECCV 2024) could not be fully read (PDF extraction failed twice) — manually check this one before presenting, its title is close enough to warrant it.

---

### RQ4 — Across the design axes of a PEFT few-shot system — dataset, shot count, backbone, adapter, uncertainty head — how is the variance in accuracy, calibration, and OOD detection distributed?

**Question.** Across the design space of a PEFT few-shot system — dataset, shot count, backbone, adapter, uncertainty head — how is the variance in each outcome metric (accuracy, calibration, OOD detection) distributed across those axes?

**Why this is answerable here and rarely elsewhere.** The 32-cell subgrid is a *balanced full factorial*, which permits an effect-size decomposition (η², the share of total variance explained by each main effect). Most PEFT papers vary one axis at a time and cannot compute this.

| Metric | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.8% | **76.3%** | 3.9% | 9.2% | 0.2% | 8.6% |
| ECE | 2.0% | 2.2% | 4.7% | 0.6% | **84.0%** | 6.5% |
| OOD AUROC (far) | 1.2% | 23.7% | 12.1% | 1.8% | **40.8%** | 20.3% |
| OOD AUROC (TinyIN near) | 1.6% | **43.7%** | 0.6% | 21.9% | 21.6% | 10.6% |

**The head interpretation explains 84.0% of calibration variance and 0.2% of accuracy variance** — close to a clean separation of concerns.

**Supporting result on the calibration/OOD relationship.** Across all 40 cells, the rank correlation between ECE and OOD AUROC is *positive* (Spearman ρ = +0.433 for SVHN-far, +0.477 for TinyImageNet-near) — i.e. worse-calibrated configurations detect OOD better. But this is a head effect: controlling for head interpretation collapses it to ρ = +0.150/+0.195 (evidential) and +0.263/+0.242 (softmax). **The defensible statement is orthogonality, not tension** — once the head is fixed, ECE carries almost no information about OOD AUROC. That calibration and OOD detection are distinct objectives is already known in general; what is absent from the literature is anyone measuring the relationship on the same runs in a PEFT setting.

**Supporting measurement — inference cost (not part of the core research question; reported here as a secondary result carried over from the original RQ4/Pareto work).** Measured on a single-thread edge-proxy CPU: the evidential head's mean absolute latency difference from softmax at matched (backbone, adapter) is **1.29%**, below the session's own 5.91% measurement-noise floor. The uncertainty-scoring stage for a 75-query episode costs 1.09% of one image's backbone forward pass. Combined statement: uncertainty is effectively free at inference; its cost is paid at design time, in calibration, and only if the wrong head is chosen.

**Extra compute required.** None.

**Novelty check (2026-08-23).** ⚠️ Split verdict — present carefully. The exact *method* (balanced 5-axis factorial + formal η² decomposition of ECE/accuracy/AUROC in a PEFT/few-shot setting) appears genuinely novel — no matching methodology found. But the *qualitative pattern* ("calibration and accuracy are governed by different factors") is established since [Guo et al. 2017](https://arxiv.org/abs/1706.04599) and should be cited as prior grounding, not presented as a new discovery — only the rigorous decomposition and the PEFT/few-shot setting are new. One near-miss flagged for a manual methods check: "A Benchmark Study on Calibration" (`arXiv:2308.11838`, ~117K networks) — could not confirm from its abstract alone whether it already uses ANOVA/η²-style decomposition. Separately, **the inference-cost claim is not novel** — "evidential/Dirichlet heads add negligible inference overhead vs. softmax, because it's a single deterministic forward pass" is essentially established since [Sensoy et al. 2018](https://arxiv.org/abs/1806.01768), the original EDL paper, which frames this as EDL's explicit selling point over MC-dropout/ensembles. Present this as *rigorous confirmation in this specific setting*, not as a discovery.

---

### RQ5 — Does calibration error reach an optimum at an intermediate trainable-parameter budget, and does that budget differ from the one that maximises accuracy?

**Question.** Does calibration keep improving as trainable-parameter budget increases, or is there a budget beyond which it degrades again? And if there is such an optimum, does it coincide with the budget that maximises accuracy?

**Why this matters.** If the calibration-optimal budget differs from the accuracy-optimal budget, then selecting an adapter size by validation accuracy — which is what everyone does — silently costs reliability, and PEFT budget selection needs its own criterion.

**Evidence.** CIFAR-FS × ResNet-18 is the only panel with all four budget levels. ECE falls then rises in **4 / 4** curves:

| Head, shots | 0–2 params (Linear-Probe) | 12.3k (LoRA) | 31.7k (bottleneck) | 11.18M (Full-FT) |
|---|---:|---:|---:|---:|
| Softmax, 1-shot | 0.2818 | 0.0969 | **0.0560** | 0.0854 |
| Softmax, 5-shot | 0.4476 | 0.1016 | **0.0670** | 0.0728 |
| Evidential, 1-shot | 0.4397 | 0.2970 | **0.2765** | 0.3207 |
| Evidential, 5-shot | 0.6234 | 0.3294 | **0.3010** | 0.3383 |

Corresponding accuracies (softmax): 1-shot 70.25 → 75.44 → 78.57 → **81.14**; 5-shot 87.41 → 86.25 → **91.44** → 90.47.

**So at 1-shot the two optima genuinely differ**: Full-FT maximises accuracy (81.14%) while the 31.7k adapter minimises ECE (0.0560).

**What the literature already covers, and what it doesn't.** [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) established that calibration error grows with *total* model capacity (depth/width) — a different axis from *trainable* parameters under a frozen backbone. A closer precedent is [LoRA-Ensemble](https://arxiv.org/abs/2405.14438), which sweeps LoRA rank under a frozen ViT backbone and reports ECE *degrading* at high rank ("unnecessarily high ranks may degrade calibration") — this is a genuine partial precedent for the **high-budget half** of a U-shape in a similarly frozen-backbone PEFT setting, though it never sweeps down to a near-zero-parameter probe or up to full fine-tuning, so it does not show the full curve or the accuracy/calibration budget mismatch.

**⚠️ Citation needs re-verification before presenting.** An earlier draft of this section cited "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (`arXiv:2410.21228`) for the numbers ECE 0.018 (Full-FT) vs 0.149–0.152 (LoRA), as an example of the literature reporting full-FT as better calibrated. Two independent direct fetches of that paper found **no calibration/ECE content in it at all** — its actual subject is SVD "intruder dimensions" and forgetting. These numbers could not be traced to this source and have been removed pending the student locating the correct citation (or dropping the claim).

**Status — this is the weakest-evidenced of the five and is proposed as an experiment, not a finding.** Four budget points only; budget is confounded with adapter type (LoRA vs bottleneck) and with *which* weights are trained (adapter vs whole backbone); Full-FT and Linear-Probe have inert seed axes by construction, so they are single measurements with no variance estimate.

**Extra compute required.** One clean rank sweep: fix dataset, backbone and adapter family; vary bottleneck rank across ~7 values × 3 seeds ≈ 21 runs. At the grid's measured mean of 1,054 s/run this is ≈ 6 GPU-hours.

**Novelty check (2026-08-23).** ❌ Weakest of the five — real gaps found. LoRA-Ensemble already shows the high-budget half of this U-shape in a similar frozen-backbone setting, so "no source reports this turning point" overclaims; the honest framing is "the full curve, plus the accuracy/calibration budget mismatch, is new — the high-budget trend direction is not." The `arXiv:2410.21228` citation is unverified and should not be presented until its source is confirmed. This RQ is already self-flagged above as the weakest-evidenced (n=1 panel) — the novelty check reinforces treating it as a proposed follow-up experiment, not a finished claim.

---

## 5. Proposed reframing

No experiment is discarded. What changes is the question the existing work is presented as answering.

- **From:** "Bayesian PEFT for reliable few-shot vision" — a method contribution whose method mostly loses.
- **To:** "What governs reliability under parameter-efficient adaptation?" — an attribution study over a balanced factorial grid, whose central result is that accuracy and calibration are controlled by different, non-competing design levers.

Under this framing the negative results become load-bearing: Orig-RQ2's failure is the measurement establishing that the head axis owns 84% of calibration variance (RQ4), and Orig-RQ3's energy correction becomes the motivation for RQ1.

---

## 6. Known limitations applying to all of the above

1. **ImageNet-pretraining overlap.** The backbones are ImageNet-pretrained while the standard CIFAR-FS/MiniImageNet protocol trains from scratch on base classes. MiniImageNet classes *are* ImageNet classes, so "novel" test classes were seen in pretraining. Absolute accuracies are therefore not comparable to from-scratch few-shot literature. The proposed RQs are less exposed than the original ones because they concern relative differences between design choices, but the caveat is not eliminated.
2. **One frozen hyperparameter recipe** across all 40 configurations, tuned once on a single configuration. The grid answers "how do these axes compare under one fixed recipe", not "what is each cell's best achievable number".
3. **Three seeds**, and effectively one for the Full-FT and Linear-Probe baselines whose seed axis is inert by construction.
4. **Vacuity's known evaluation artefact does not apply here, and this should be stated in the thesis rather than left implicit.** [Rethinking Vacuity for OOD Detection in EDL (2026)](https://arxiv.org/html/2605.06382v1) shows vacuity-based AUROC can be inflated when class cardinality differs between ID and OOD evaluation. In this protocol every OOD sample is scored under the same 5-way episode head, so K_ID = K_OOD = 5 always.
5. **Novelty claims rest on a non-exhaustive search.** Twelve targeted web searches and five paper fetches, August 2026 (initial pass, 2026-08-21), plus an independent five-agent, 60+ search re-verification pass, 2026-08-23. Absence of found prior art is not proof of absence.

---

## 7. What we are asking the reviewer to check

Ranked by how much damage a negative answer would do.

1. **RQ3's causal interpretation — the most important check.** The budget ordering reverses *with the backbone*, so "ECE follows the budget" and "ECE follows something else about MobileNetV3-Small vs ResNet-18" predict the same 16/16 pattern in this data. The budget explanation is more parsimonious and is the only one that also explains the Linear-Probe → LoRA → bottleneck ordering within a fixed backbone (RQ5's table), but the grid alone cannot fully separate them. **Is there a backbone-intrinsic explanation we have missed?** Note this is precisely what RQ5's rank sweep would settle, since it varies budget with backbone and adapter family fixed.

2. **Novelty of RQ3.** Has anyone dissociated adapter architecture from adapter parameter budget, by outcome metric, in any modality? Suggested search terms: *adapter capacity calibration*, *rank versus architecture calibration*, *PEFT reliability attribution*, *low-rank adaptation confidence capacity*. Also specifically check "Robust Calibration of Large Vision-Language Adapters" (ECCV 2024) — flagged as unread/unverified in the 2026-08-23 pass.

3. **Novelty of RQ5.** Has anyone reported a non-monotonic (U-shaped) relationship between *trainable* parameter count and calibration error, with a frozen backbone, across the *full* range from near-zero to full fine-tuning? Distinguish this carefully from the well-established relationship between *total* model capacity and calibration, and from LoRA-Ensemble's partial (high-budget-only) precedent. Suggested terms: *calibration sweet spot trainable parameters*, *non-monotonic calibration capacity*, *adapter rank calibration optimum*. Also resolve the unverified `arXiv:2410.21228` citation before this RQ is presented.

4. **Whether RQ1's gap is really open.** Confirm the quoted limitation in arXiv 2603.07571 and check whether any OOD benchmark (OpenOOD or otherwise) already reports a fully factorised objective × score matrix, in any regime.

5. **Whether RQ2 is too narrow to count as a contribution** given existing post-hoc calibration literature, and whether the ranking-preservation question has already been answered analytically for Dirichlet vacuity somewhere.

6. **Statistical framing.** Is a sign test over 16 pairs the right test given that pairs sharing a backbone are not fully independent? Is the η² decomposition appropriate for a factorial with n = 1 aggregated observation per cell (seeds averaged before decomposition), or should it be recomputed on the per-seed observations?

7. **Anything in §6 we have understated.**

---

*Numbers in this document are computed from the aggregated grid results file (`results/mvt_results.json`, 120 runs, generated 2026-08-06). Literature verification performed 2026-08-21; independent novelty cross-check and renumbering performed 2026-08-23.*
