# B-PEFT — Research Questions, Results, Findings, and Novelty Verification Package

**Purpose.** Two audiences. (a) An independent researcher/agent with **no repository access**, verifying
the novelty and causal claims below — every citation and prior-art check found across three internal passes
is included so nothing has to be re-discovered. (b) A **teammate implementing the matched-budget
experiment that closes RQ3's open identification problem** — see
[RQ3_MATCHED_BUDGET_PLAN.md](RQ3_MATCHED_BUDGET_PLAN.md), not this document.

**Status:** current as of 2026-08-26. Target framing: **masters defence** (not conference submission) —
this affects which questions are load-bearing; see §2.

**Numbering warning.** This document uses a **four-RQ** structure that supersedes the five-RQ draft in
`docs/NEW_RQS.md`. The mapping is given in §2. Original proposal questions are labelled **Orig-RQ1…4**
throughout and appear only in the Appendix.

---

## 0. What to do with this document

**For novelty review:** attack the novelty claims and causal interpretations, not the arithmetic. Each RQ
carries a recorded verdict (✅ novel / ⚠️ needs reframing / ❌ weak). Re-derive them independently using the
anchors and open items listed. §9 is the reviewer checklist.

**For implementation:** the matched-budget experiment for RQ3 lives in its own document,
[RQ3_MATCHED_BUDGET_PLAN.md](RQ3_MATCHED_BUDGET_PLAN.md) — a complete, self-contained build spec. Every
parameter count in it was validated against committed run values, and every config key against the actual
source. Read its §2 before writing code — one non-obvious constraint there will silently invalidate the
experiment if missed. §8 below is only a one-paragraph pointer to it.

**Search provenance already spent** (a starting point, not proof of absence): 12 targeted searches + 5 paper
fetches (2026-08-21); an independent 5-agent / 60+ search re-verification (2026-08-23); a full-PDF
literature stress-test (2026-08-26); and an independent deep-research agent pass (post-2026-08-26, §9.1).

---

## 1. Experimental setup

### 1.1 Model

```
frozen ImageNet-pretrained CNN backbone
        ↓
small trainable adapter          ← the only trained parameters
        ↓
parameter-free prototype head    ← classifies by similarity to support-set class means
```

The backbone is **never updated** (except in the Full-FT baseline). The head has **no trainable weights**:
logits are the similarity between a query embedding and each class's mean support embedding. A
configuration's trainable-parameter count is therefore essentially the adapter's size.

**Head "interpretation"** is a separate axis from head type. The prototype head always emits raw similarity
logits, read one of two ways:

- **Softmax** — logits → softmax → probabilities. Confidence = max softmax probability.
- **Evidential (Dirichlet)** — `evidence = softplus(logits × scale + bias)`, `α = evidence + 1`, `S = Σα`.
  Probability = `α/S`. Uncertainty = *vacuity* = `K/S`, K = 5.

The evidential mapping adds exactly **2 trainable parameters** (`scale`, `bias`) — hence counts like 31,746
against softmax's 31,744. Frozen at (2, −6) across the base grid; RQ4 refits them.

### 1.2 Protocol

| | |
|---|---|
| Task | 5-way {1, 5}-shot episodic classification |
| Test episodes | 600 fixed episodes, seeds frozen in a version-controlled file |
| Training seeds | 3 per configuration (42, 43, 44) |
| Datasets | CIFAR-FS (Bertinetto split), MiniImageNet (Ravi & Larochelle split) |
| Backbones | ResNet-18 (11.7M params), MobileNetV3-Small (2.5M params) — ImageNet-pretrained, frozen |
| Hyperparameters | **One frozen recipe across all 40 configurations.** Not re-tuned per cell. |

Test classes are disjoint from the classes the adapter meta-trains on.

### 1.3 The grid

40 unique configurations × 3 seeds = **120 runs**, all completed.

- 2 datasets × 2 shot regimes × 2 backbones × 2 adapters × 2 head interpretations = 32 cells
  (**balanced full factorial**)
- \+ 8 baseline cells (Full-FT, Linear-Probe) on CIFAR-FS × ResNet-18 only (unbalanced; excluded from the
  variance decomposition)

**Adapters:** *Bottleneck-parallel* (1×1 down → ReLU → 1×1 up, run on the block input and summed at its
output, at the final block of each of 4 stages); *LoRA* (low-rank update injected into one 1×1 convolution);
*Full-FT* and *Linear-Probe* as baselines.

### 1.4 Metrics

Accuracy and macro-F1 (mean over 600 episodes); **ECE (pooled)** — lower is better; Brier; **OOD AUROC** —
higher is better, over SVHN and Gaussian noise (far-OOD) and CIFAR-100-heldout / MiniImageNet-heldout and
TinyImageNet (near-OOD).

**Uncertainty scores.** Evidential cells scored with **vacuity**; softmax cells with **MSP**, **TS-MSP**, and
**energy**. This asymmetry is the subject of RQ2.

### 1.5 The structural fact that makes RQ3 possible

Trainable parameter counts (softmax; evidential adds 2):

| Backbone | Bottleneck-parallel | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck, by 2.58× |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA**, by 1.55× |

**The ordering reverses between backbones** — not by design, but as a consequence of channel widths. This
partially deconfounds adapter *architecture* from adapter *budget*, which is normally impossible because the
"better" adapter is usually also the bigger one everywhere. RQ3 rests on this reversal; §5 states its limits
and §8 is the experiment that closes them.

### 1.6 Limitations applying to every RQ

1. **ImageNet-pretraining overlap.** Backbones are ImageNet-pretrained while the standard protocol trains
   from scratch on base classes. MiniImageNet classes *are* ImageNet classes, so "novel" test classes were
   seen in pretraining. Absolute accuracies are not comparable to from-scratch few-shot literature. The RQs
   concern relative differences, which mitigates but does not eliminate this.
2. **One frozen hyperparameter recipe** across all configurations, tuned once. The grid answers "how do
   these axes compare under one recipe," not "what is each cell's best achievable number."
3. **Three seeds**, and effectively one for the Full-FT/Linear-Probe baselines whose seed axis is inert.
4. **Vacuity's known evaluation artefact does not apply here** — [Rethinking Vacuity for OOD Detection in
   EDL (2026)](https://arxiv.org/html/2605.06382v1) shows vacuity AUROC inflates when ID/OOD class
   cardinality differs. Here every OOD sample is scored under the same 5-way episode head, so
   K_ID = K_OOD = 5 always. State this explicitly rather than leaving it implicit.
5. **Novelty claims rest on a non-exhaustive search** (§0).

---

## 2. The four research questions

**Overarching question**

> **What governs reliability — predictive accuracy, confidence calibration, and out-of-distribution
> detection — in parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image
> classification, and can the reliability deficits so identified be remediated post-hoc?**

| New | Question | Role | Was |
|---|---|---|---|
| **RQ1** | How is variance in accuracy, calibration and OOD detection distributed across the design axes? | Frame | old RQ4 |
| **RQ2** | Is OOD performance attributable to the training objective or the scoring rule? | Resolves the OOD outcome | old RQ1 |
| **RQ3** | Does accuracy follow adapter architecture while calibration follows parameter budget? | Resolves the adapter axis; **novelty claim** | old RQ3 |
| **RQ4** | Can the evidential head be recalibrated post-hoc without destroying its OOD ranking? | Remediation | old RQ2 |

The old **RQ5** (interior-optimum budget) is **demoted** to §7 — a tested hypothesis that did not survive,
reported honestly rather than defended as a contribution.

**Why this set, for a defence.** RQ2 is the most bulletproof result in the project (a 163× effect) and is
retained despite thin methodological novelty, because a defence rewards soundness over unprecedentedness.
RQ3 is the most novel but has the softest causal identification, stated up front in §5 rather than
footnoted. §8 exists to close that gap.

---

## 3. RQ1 — Attribution of outcome variance across design axes

> **How is variance in accuracy, calibration, and out-of-distribution detection distributed across the
> principal design axes of a parameter-efficient few-shot classification system — dataset, shot count,
> backbone, adapter type, and uncertainty-head interpretation?**

**Sub-questions.** RQ1a: which axis dominates for each outcome? RQ1b: how associated are calibration and
OOD detection once head interpretation is held constant?

**Hypotheses.** H1.1 accuracy variance is dominated by shot count, not adaptation choices. H1.2 calibration
variance is dominated by head interpretation. H1.3 ECE and OOD AUROC are approximately conditionally
independent given head interpretation.

**Operationalisation.** Main-effects η² over the balanced 2⁵ factorial, computed on **per-seed** (96
observations), not cell-averaged, so a genuine residual is retained. Spearman ρ marginally and stratified.

### Result (per-seed, computed from `results/mvt_results.json`)

| Metric | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.75% | **76.05%** | 3.92% | 9.14% | 0.19% | 8.95% |
| ECE | 2.02% | 2.13% | 4.63% | 0.63% | **82.89%** | 7.70% |
| OOD AUROC (far) | 1.18% | 22.68% | 11.54% | 1.73% | **39.01%** | 23.87% |
| OOD AUROC (TinyIN near) | 1.64% | **42.60%** | 0.61% | 21.98% | 20.98% | 11.42% |

**Head interpretation explains ~83% of calibration variance and ~0.2% of accuracy variance** — close to a
clean separation of concerns.

**Per-seed recomputation — resolved.** The seed-averaged version of this table gave 84.0%/0.2% with
residuals of 6.5–20.3%. The concern (raised by a WACV 2026 study carrying a genuine 10.6% per-observation
residual) was that averaging seeds before decomposing suppresses the error term and manufactures apparent
cleanliness. Recomputing on all 96 raw per-seed observations moves the residual by only **0.3–3.6
percentage points** — nowhere near enough to explain the split as an averaging artefact. **The claim holds
on per-seed data.**

**Data-quality note found during that recomputation.** One of 96 observations is missing:
`cifar_fs / 5-shot / mobilenetv3_small / lora / evidential`, seed 42's TinyImageNet-near AUROC (seeds 43/44
present; 95/96 for that metric only, all others complete). Does not change any conclusion, but
`mvt_results.json`'s `missing_cells: []` field tracks only entirely-absent cells, not partially-absent
per-seed metrics — worth a log check before calling the raw grid 100% complete.

**Supporting result — calibration/OOD relationship.** Across all 40 cells the ECE↔AUROC rank correlation is
*positive* (ρ = +0.433 SVHN-far, +0.477 TinyIN-near) — worse-calibrated configs appear to detect OOD better.
This is a head effect: stratifying collapses it to ρ = +0.150/+0.195 (evidential) and +0.263/+0.242
(softmax). **The defensible statement is orthogonality, not tension.**

**Supporting measurement — inference cost.** Evidential vs. softmax mean absolute latency difference at
matched (backbone, adapter), single-thread edge-proxy CPU: **1.29%**, below the session's own 5.91%
noise floor. Uncertainty scoring for a 75-query episode costs 1.09% of one image's backbone forward pass.

### Novelty check (2026-08-23)
⚠️ **Split verdict.** The *method* (balanced 5-axis factorial + formal η² of ECE/accuracy/AUROC in a
PEFT/few-shot setting) appears novel. The *qualitative pattern* ("calibration and accuracy are governed by
different factors") is established since [Guo et al. 2017](https://arxiv.org/abs/1706.04599) — cite as prior
grounding, not discovery. **The inference-cost claim is not novel** — negligible EDL inference overhead is
established since [Sensoy et al. 2018](https://arxiv.org/abs/1806.01768), which frames it as EDL's selling
point over MC-dropout/ensembles. Present as rigorous confirmation in this setting.

### Literature stress-test (2026-08-26)
⚠️ **Mixed.**
- ["A Benchmark Study on Calibration" (ICLR 2024, arXiv:2308.11838)](https://arxiv.org/abs/2308.11838),
  fetched and read in full: despite being the largest calibration study extant (117,702 architectures), it
  runs **no** ANOVA/η²/variance decomposition — box plots and rank correlations only. This thesis's
  decomposition is more statistically precise than the largest existing calibration benchmark.
- ["One Model, Many Behaviors" (WACV 2026, arXiv:2601.10836)](https://arxiv.org/abs/2601.10836) runs a real
  three-way ANOVA on OOD-AUROC with a genuine per-observation residual (10.6%). This motivated the per-seed
  recomputation above, **now done and passing**.
- Qualitative pattern confirmed old news via Guo et al. 2017 and
  [Minderer et al. 2021](https://arxiv.org/abs/2106.07998) — present as re-confirmation.
- The **correlation-collapse-under-control** result (ECE↔AUROC orthogonality once head is fixed) was **not
  found anywhere else** — genuinely new; keep as the headline addition.

**Stated limits.** Main effects only; interactions not modelled. Conclusions are relative contributions
under one recipe.

---

## 4. RQ2 — Training objective versus scoring rule in OOD detection

> **Is out-of-distribution detection performance attributable to the training objective under which the
> model was adapted, or to the scoring rule by which its outputs are read — and can the two be separated
> experimentally?**

**Sub-questions.** RQ2a: what share of OOD variance does each factor explain when varied independently?
RQ2b: does a score developed for softmax outputs retain effectiveness on Dirichlet-parameterised outputs?

**Hypotheses.** H2.1 the scoring rule accounts for substantially more OOD variance than the objective.
H2.2 a logit-space score yields comparable AUROC irrespective of which objective produced the logits.

**The confound being dissolved.** The base grid scores evidential runs with vacuity only and softmax runs
with MSP/TS-MSP/energy only — every comparison varies objective and score *together*. The fix computes all
four scores on all runs: a clean **2 objectives × 4 scores** factorial.

### Result (`results/rq_factorial/`, `results/rq_summary.json`)

**ANSWERED — the score dominates the objective, decisively.** η² over the full 2×4 factorial (n = 792
comparisons per pool):

| Pool | Scoring rule | Training objective | Ratio |
|---|---:|---:|---:|
| Far-OOD | **43.7%** | 0.27% | 163× |
| Near-OOD | **13.0%** | 0.60% | 22× |

Energy scores 0.911 AUROC on evidential-trained logits and 0.929 on softmax-trained logits (far-OOD) —
near-identical from the *same score* regardless of the objective that produced the logits — while MSP stays
around 0.79 under either. Both pools: `dominant = "score"`.

**Coverage.** Computed on 99/120 recoverable checkpoints; all included cells are matched evidential-vs-
softmax pairs. A regression guard confirms the refactor that added the cross-terms did not perturb the
original diagonal (99/99 cells exact match against committed metrics).

### Novelty anchor
[A Systematic Comparison of Training Objectives for OOD Detection (2026,
arXiv:2603.07571)](https://arxiv.org/html/2603.07571v2) names this exact gap as an open limitation:
objective and scoring rule "are not fully factorized, since each objective is evaluated with the confidence
measure most natural to its output space." That paper uses ResNet-18 under OpenOOD protocols and compares
Cross-Entropy/Triplet/Prototype/Average-Precision losses — **no evidential/Dirichlet objective at all** —
and does not cover frozen backbones, few-shot episodic evaluation, or PEFT.

### Novelty check (2026-08-23)
✅ **Confirmed novel at the time, no risk flags.** The arXiv:2603.07571 quote was verified against the
paper's raw text. Note precisely: it flags the gap for a *different* objective family (metric/ranking
losses), so it is a motivating analogy, not existing coverage. Closest adjacent work checked and ruled out:
arXiv:2605.06382 (vacuity cardinality critique) and OpenOOD/OpenOOD-1.5 — neither crosses scores onto
Dirichlet-trained models.

### Literature stress-test (2026-08-26)
⚠️ **Methodology less novel than it looked; the finding is unaffected.**
- [arXiv:2601.10836 (WACV 2026)](https://arxiv.org/abs/2601.10836) already runs a comparable
  training-method × scoring-method ANOVA at much larger scale (56 models × 21 detectors × 8 OOD sets), the
  same year — **cite it; do not let it surface as a surprise.**
- The qualitative pattern ("post-hoc score beats training method") is old news via OpenOOD, Bitterwolf et
  al. 2022, and Guo et al. 2017's calibration analogue.
- [A theory paper (arXiv:2605.22746)](https://arxiv.org/abs/2605.22746) proves softmax is a mathematical
  special case of an evidential classifier — this **pre-explains** why the objective barely matters. Cite
  proactively; do not leave it for a committee member to raise.
- **What still stands:** cross-applying energy to Dirichlet logits and finding near-equivalence to
  softmax-trained energy has no precedent found anywhere. That evidential-specific angle — not the ANOVA
  technique — is the remaining contribution.

**Defence framing.** Present as *confirmation and precise quantification in a new regime*, with the
concurrent WACV work and the theory result cited up front. Under a defence bar this is a strength: the
result is unambiguous and the prior art is acknowledged rather than missed.

---

## 5. RQ3 — Architecture versus parameter budget as the governing adapter property

> **Within the adapter axis, are accuracy and calibration governed by the same property of the adapter, or
> does accuracy follow adapter architecture while calibration follows trainable-parameter budget?**

**Relation to RQ1 — state this first, or it looks like a contradiction.** RQ1 finds the adapter axis
explains only **0.63%** of calibration variance. RQ3 asks a different question: *conditional on comparing
two adapters directly, which property predicts the winner*. An axis can contribute little total variance
while exhibiting a highly consistent direction. **RQ3 is a claim about direction, not magnitude.**

**Sub-questions.** RQ3a: does the accuracy ordering persist when the budget ordering reverses? RQ3b: does
the calibration ordering track budget rather than architecture identity?

**Hypotheses.**
- **H3.1 (architecture).** Accuracy and near-OOD ranking follow adapter architecture, invariant to which
  arm holds the larger budget.
- **H3.2 (budget).** Calibration follows the larger trainable-parameter budget, irrespective of
  architecture.
- **H3.2-alt (rival).** Calibration differences are attributable to a backbone-intrinsic property rather
  than to budget.

### Evidence — 16 matched bottleneck-vs-LoRA comparisons

Head interpretation held fixed within each pair.

| Dataset | Shots | Backbone | Head | Btl params | LoRA params | Larger | ECE btl | ECE LoRA | Better calibrated |
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

| Outcome | Winner is bottleneck architecture | Winner is larger-budget arm |
|---|---:|---:|
| Accuracy | **16 / 16** | 8 / 16 |
| OOD AUROC (near, native score) | **16 / 16** | 8 / 16 |
| OOD AUROC (TinyImageNet near) | **16 / 16** | 8 / 16 |
| OOD AUROC (SVHN far) | 11 / 16 | 7 / 16 |
| ECE (pooled) | 8 / 16 | **16 / 16** |

**The claim, stated so it cannot be overread.** The "8/16" entries are *arithmetically implied*, not
independent evidence: because the budget ordering reverses between backbones, any outcome perfectly
consistent with architecture is necessarily 8/16 on budget, and vice versa. The real content is two
directly observed facts:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both
   backbones despite holding 2.58× *more* parameters on one and 1.55× *fewer* on the other.
2. **The ECE winner does change, exactly in step with the budget ordering.** LoRA is better calibrated in
   all 8 MobileNetV3-Small pairs (where LoRA is larger); bottleneck in all 8 ResNet-18 pairs (where
   bottleneck is larger).

**Statistical strength.** Each direction is 16/16 sign consistency, two-sided p ≈ 3.05×10⁻⁵ under a null of
random direction. **Magnitudes are weaker:** |ΔECE| exceeds 2× the pooled across-seed SD in only **10/16**
pairs — all 8 MiniImageNet pairs, but only 2 of 8 CIFAR-FS pairs. Direction is robust; per-pair effect
sizes on CIFAR-FS are not.

**Pre-stated scope on OOD pools.** Hypotheses concern near-OOD only. Far-OOD separation is near-chance under
both accounts (11/16 architecture, 7/16 budget) and RQ1 attributes it predominantly to head interpretation
(39.0% head vs 1.7% adapter). Reported as a non-finding, not excluded post hoc.

### ⚠️ Identification limits — stated in the main text, not footnoted

**Because the budget ordering reverses *with* the backbone, H3.2 and H3.2-alt predict identical patterns in
this data.** The budget account is preferred on parsimony, **not established by discrimination.** A
controlled rank sweep intended to adjudicate the two (§7) returned a direction-dependent result and does not
settle it. **H3.2 is advanced as the better-supported of two live explanations, not as a demonstrated causal
claim.**

**§8 is the experiment that closes this gap** by comparing both architectures *at matched parameter budget
within each backbone*, under which the three hypotheses make divergent predictions for the first time.

### Novelty check (2026-08-23)
✅ **Mostly confirmed novel.** [CP tensor adapters (2026)](https://arxiv.org/html/2606.00428v1) studies the
accuracy–budget curve but does not measure calibration (verified: accuracy-only, single backbone, no
reversal design). Closest partial hits — [LoRA-Ensemble
(arXiv:2405.14438)](https://arxiv.org/abs/2405.14438) (budget-vs-ECE within one architecture) and ["On
Fairness of Low-Rank Adaptation" (arXiv:2405.17512)](https://arxiv.org/abs/2405.17512) (multi-backbone
calibration, but LoRA-vs-full-FT rather than architecture-vs-architecture) — combine at most two of the
three required ingredients.

### Literature stress-test (2026-08-26)
✅ **Holds up best of all the RQs under adversarial re-checking.** No comparable architecture-vs-budget
dissociation found in vision or NLP PEFT. The standard NLP-PEFT convention is to *equalize* budgets for fair
comparison, never to deliberately exploit a reversed ordering — **the reversal trick itself appears to have
no precedent.** No direct contradiction found. **Required addition:** ["Be Confident in What You Know:
Bayesian Parameter Efficient Fine-Tuning of Vision Foundation
Models" (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f1fbd5ab8d58d0ecf33c95fd46b900e-Abstract-Conference.html)
is genuinely distinct (ViT foundation models, base-rate-adjustment + evidential ensembling, no
architecture-vs-budget dissociation) but close enough in name ("Bayesian-PEFT") to require an explicit
citation and differentiation paragraph.

### Independent agent verification (post-2026-08-26)
✅ **"FULLY NOVEL — strongest contribution."** Independently confirmed distinct from both near-miss papers:
"Robust Calibration of Large Vision-Language Adapters" (ECCV 2024) — previously unread, now checked — and
the NeurIPS 2024 Bayesian-PEFT paper above. Both prior open items closed.

---

## 6. RQ4 — Post-hoc remediation of the calibration deficit

> **Can the calibration of an evidential prototype head be improved post-hoc by refitting only the two
> parameters of its evidence-affine transform on held-out validation episodes, and is the head's
> out-of-distribution ranking preserved under that refitting?**

**Relation to RQ1.** RQ1 identifies head interpretation as the dominant source of calibration variance and
the evidential head as the poorly-calibrated arm. RQ4 tests whether that deficit is *intrinsic to the head*
or an artefact of a fixed default parameterisation.

**Sub-questions.** RQ4a: magnitude and consistency of the ECE change? RQ4b: is vacuity's sample ordering
preserved?

**Hypotheses.** H4.1 refitting reduces ECE relative to the fixed default. H4.2 OOD ranking is preserved
within a pre-specified tolerance.

**Theoretical motivation for RQ4b.** The transform is monotone in each individual logit, which might suggest
ranking preservation follows automatically. But vacuity is `K/Σα`, a function of *all K logits jointly*, so
a per-logit monotone transform does **not** guarantee preservation of the induced sample ordering.
Preservation is an empirical question, not an analytic consequence.

**Operationalisation.** Parameters fitted **exclusively on the validation episode split**, evaluated on the
frozen 600-episode test split. Reported as per-cell ΔECE, per-(cell, pool) ΔAUROC against a pre-specified
tolerance of Δ ≥ −0.005, and Spearman ρ between pre- and post-refit vacuity orderings.

### Result (`results/rq_summary.json` → `rq2_rows`/`rq2_verdict`)

**ANSWERED — mostly the good outcome.**

- ECE improved in **48/48 evidential cells (100%)**, mean **−0.137 absolute** (many cells drop from
  0.25–0.44 to 0.10–0.31). Refit values cluster around scale 7–14 vs. the frozen (2, −6), implying the
  default sat well off the optimum throughout — and a much larger gain than the flat ~0.285–0.296 ECE
  surface an earlier single-configuration sweep had found.
- OOD-AUROC preserved (Δ ≥ −0.005) in **150/192 comparisons (78%)**; mean ΔAUROC ≈ +0.004 (flat).
- `reordering_ever_observed = true` — reordering does occur, confirming the affine is not automatically
  rank-preserving for vacuity — but the **worst-case** cell across every pool still had Spearman ρ = 0.921.

**Honest caveat.** AUROC dropped by more than 0.5pp in the remaining 22% of comparisons (worst single-pool
drops ≈ −0.03). State as *"survives in the large majority of cases, with a measured minority exception,"*
never *"always."*

### Novelty positioning — narrowest claim of the four, scope it carefully
Post-hoc calibration is mature: Dirichlet calibration (Kull et al. 2019, NeurIPS) and
[accuracy-preserving post-hoc calibration via invertible logit
transforms](https://arxiv.org/html/2608.10372) both exist. **The claim is NOT "first post-hoc
calibration."** It is: a two-parameter recalibration of an *evidential prototype head* in a frozen-backbone
few-shot regime, with OOD-ranking preservation **measured rather than assumed**.

### Novelty check (2026-08-23)
✅ **Confirmed novel, no risk flags.** Kull et al.'s "Dirichlet calibration" is a *different* use of the
word — a generic calibration map for any softmax classifier, unrelated to evidential Dirichlet evidence — so
it does not pre-empt this. arXiv:2608.10372 confirmed to apply only to standard softmax classifiers. No
paper found proving or disproving ranking preservation for a sum-of-affine-transformed-logits construction
like vacuity. Closest non-matching prior work: density-informed EDL recalibration
([arXiv:2602.01477](https://arxiv.org/abs/2602.01477)) and Invascal
([arXiv:2606.00069](https://arxiv.org/abs/2606.00069)) — both architectural or training-time, not post-hoc
two-parameter refits.

### Literature stress-test (2026-08-26)
⚠️ **Magnitude unremarkable; reframe as confirmation.**
- [Guo et al. 2017](https://arxiv.org/abs/1706.04599) reports comparable or larger temperature-scaling drops
  (CIFAR-100/ResNet-110: 16.53%→1.26%).
- Bayesian Evidential Learning ([arXiv:2207.13137](https://ar5iv.labs.arxiv.org/html/2207.13137)), a
  directly analogous few-shot evidential paper, already reports −11 to −15pp — the same ballpark as −13.7pp.
- "Ranking isn't perfectly preserved" is what calibration theory predicts for any non-scalar transform —
  textbook, not discovery.
- "EDL default hyperparameters are poorly tuned" is a repeated documented complaint
  ([arXiv:2510.08938](https://arxiv.org/abs/2510.08938),
  [arXiv:2410.00393](https://arxiv.org/abs/2410.00393)).
- **Required reframe:** *"we confirm and precisely quantify a known class of calibration/ranking tradeoff
  for a new mechanism (the evidence-affine),"* not *"we discovered that refitting helps."* The mechanism and
  the 48-cell / 192-comparison systematic quantification are the genuinely new parts.

---

## 7. Demoted — the interior-optimum hypothesis (former RQ5)

Retained as a **tested hypothesis that did not survive**, reported for completeness and because the
correction is itself informative. **Not a load-bearing contribution.** Presenting it as a full RQ would
require defending an underpowered design against a published contradiction, for little gain.

**Original hypothesis.** Calibration error reaches an optimum at an intermediate trainable-parameter budget,
and that budget differs from the accuracy-optimal one.

**Superseded four-point evidence** (CIFAR-FS × ResNet-18 — ECE fell then rose in 4/4 curves):

| Head, shots | 0–2 params (Linear-Probe) | 12.3k (LoRA) | 31.7k (bottleneck) | 11.18M (Full-FT) |
|---|---:|---:|---:|---:|
| Softmax, 1-shot | 0.2818 | 0.0969 | **0.0560** | 0.0854 |
| Softmax, 5-shot | 0.4476 | 0.1016 | **0.0670** | 0.0728 |
| Evidential, 1-shot | 0.4397 | 0.2970 | **0.2765** | 0.3207 |
| Evidential, 5-shot | 0.6234 | 0.3294 | **0.3010** | 0.3383 |

This confounded budget with adapter type *and* with which weights are trainable.

**Controlled result (`results/rq5/`, `rq5_rank_sweep.png`).** Rank sweep with architecture, backbone,
dataset and shots all fixed and only `adapter.rank` moving — rank ∈ {1,2,4,8,16,32,64} × 3 seeds, 21/21 runs
completed. **`ece_optimum_is_interior: false`.** No U-shape: evidential ECE is lowest at rank 1 (0.291) and
drifts noisily upward to rank 64 (0.309); softmax ECE moves the *opposite* way, 0.093 → 0.081. The
accuracy/calibration mismatch survives (`best_ece_rank = 1`, `best_accuracy_rank = 64`) but via two
roughly monotonic opposing trends, not an interior optimum.

**Required wording.** 3 seeds × 7 ranks is plausibly underpowered to exclude a subtle U-shape. Always write
**"no interior optimum observed in the tested range."**

### The LoRA-Ensemble contradiction — response, with one correction

🚩 [LoRA-Ensemble (arXiv:2405.14438)](https://arxiv.org/abs/2405.14438), verified directly from its PDF
(Appendix C, Fig. 11), reports an interior-optimum-like reversal in single-network ECE on **CIFAR-100** —
the same dataset family: *"at rank 32, the calibration of a single network augmented with LoRA begins to
deteriorate."* Their range was {1,…,32}; this thesis's softmax curve shows the **opposite** pattern in the
overlapping region. **This must be addressed in text — a supervisor who knows the paper will ask.**

Three legitimate distinguishing factors (usable as drafted):

1. **Adaptation target.** LoRA-Ensemble injects rank into Multi-Head Self-Attention projections in a ViT;
   this sweep uses bottleneck-parallel adapters on 1×1 convolutional channels in a CNN.
2. **Head design.** They use a trainable linear head over 100 classes; this thesis uses a parameter-free
   nearest-centroid prototype head, whose similarity-based logits are intrinsically bounded in a way a free
   linear layer's are not — a plausible reason high-rank overconfidence would appear later here.
3. **Relative capacity.** Rank 64 here is ~63k parameters against an 11.7M frozen backbone (0.56%); the
   comparable point in their setup injects far more of the ViT's representational capacity.

**⚠️ One sub-argument must be cut.** A drafted version also argued the turning point "exists, just further
out," citing Full-FT's ECE (0.0854) against the 31.7k bottleneck's (0.0560). **That reuses the superseded
four-point table above** — the exact confounded comparison this sweep was built to eliminate. Full-FT is not
"more rank on the same architecture"; it is a different architecture entirely. Citing it here silently
reintroduces the confound. **Delete rather than soften.** The three factors stand without it, and the honest
position is that no reversal was observed in the tested range and the thesis takes no position beyond it.

**Two supporting findings.** (1) A real "calibration double descent" is documented
([arXiv:2302.09369](https://arxiv.org/abs/2302.09369), ICLR 2023) over a numerically overlapping range, but
via a mechanism that does not transfer (whole-network sparsification at an interpolation threshold vs. a
tiny adapter on an already-expressive frozen backbone) — cite and explain why it does not apply.
(2) EDL/KL-annealing literature confirms evidential training is non-monotonically sensitive to its
regularisation coefficient, supporting reading the noisy rank-8 spike as a training-dynamics artefact.

**Why this still earns its place.** "The interior-optimum story does not survive when architecture is held
fixed" is itself a finding, and it retroactively sharpens RQ3: budget's effect on calibration is
direction-dependent on head interpretation even within one fixed architecture.

---

## 8. Implementation plan — the matched-budget experiment that closes RQ3

**Moved to its own document:** [docs/RQ3_MATCHED_BUDGET_PLAN.md](RQ3_MATCHED_BUDGET_PLAN.md) — a
self-contained build spec for whoever implements this, so they aren't scrolling past novelty-review
material to find it.

**One-paragraph summary.** RQ3's evidence (§5) rests on a coincidence: bottleneck is the larger adapter on
ResNet-18, LoRA is the larger adapter on MobileNetV3-Small. Because that reversal is welded to the backbone,
"budget drives calibration" (H3.2) and "something about the backbone drives calibration" (H3.2-alt) predict
identical patterns and cannot be told apart by the existing grid. The linked plan builds both architectures
at matched parameter budgets within each backbone (residual mismatch ≤ 3%, vs. 55–158% today) — 30 new
runs, ≈ 8.8 GPU-hours — with a pre-registered decision rule that lets the three live hypotheses (H3.1 /
H3.2 / H3.2-alt) diverge for the first time.

**If it doesn't get run before submission:** report RQ3 exactly as §5 states it — H3.2 as the
better-supported of two live explanations, H3.2-alt named explicitly, and the linked experiment described
as the specific test that would discriminate them. That is a legitimate, defensible position at a masters
defence; see [RQ3_MATCHED_BUDGET_PLAN.md](RQ3_MATCHED_BUDGET_PLAN.md) §9.

---

## 9. Reviewer checklist

Ranked by how much damage a negative answer would do.

1. **RQ3's causal interpretation — the most important check.** The budget ordering reverses *with the
   backbone*, so H3.2 and H3.2-alt predict the same 16/16 pattern. **Is there a backbone-intrinsic
   explanation we have missed?** §8 is the experiment designed to settle it; critique that design too.
2. **Novelty of RQ3.** Has anyone dissociated adapter architecture from adapter budget, by outcome metric,
   in any modality? Terms: *adapter capacity calibration*, *rank versus architecture calibration*, *PEFT
   reliability attribution*, *low-rank adaptation confidence capacity*.
3. **Whether RQ2's gap is really open.** Confirm the arXiv:2603.07571 limitation quote, and check whether
   any OOD benchmark already reports a fully factorised objective × score matrix in any regime. Verify
   precisely what arXiv:2601.10836 does and does not cover.
4. **Whether RQ4 is too narrow to count** given existing post-hoc calibration literature, and whether
   ranking preservation for Dirichlet vacuity has been settled analytically somewhere.
5. **Statistical framing.** Is a sign test over 16 pairs right given that pairs sharing a backbone are not
   fully independent? (The η² per-seed question is now **closed** — see §3.)
6. **Anything in §1.6 understated?**

### 9.1 Outstanding action items

- ✅ **CLOSED** — RQ3: "Robust Calibration of Large Vision-Language Adapters" (ECCV 2024) read; distinct.
- ✅ **CLOSED** — RQ3: NeurIPS 2024 "Bayesian-PEFT" paper confirmed distinct; differentiation paragraph
  still to be *written into the thesis text*.
- ✅ **CLOSED** — RQ1: per-seed η² recomputation done (§3); split confirmed.
- ✅ **CLOSED** — §7: LoRA-Ensemble contradiction response drafted, with one flawed sub-argument identified
  and removed.
- ⬜ **OPEN** — §8: the matched-budget experiment.
- ⬜ **OPEN** — Cite arXiv:2601.10836, arXiv:2605.22746, arXiv:2608.10372, Sensoy et al. 2018, Guo et al.
  2017, Minderer et al. 2021 as prior grounding wherever the corresponding claim appears in the thesis text.
- ⬜ **OPEN** — Check the training/eval logs for the one missing per-seed near-OOD value identified in §3.

### 9.2 Independent agent verification (post-2026-08-26)

An independent deep-research pass **confirmed the internal verdicts** rather than overturning them:
RQ3 "fully novel, strongest contribution"; RQ2 "novel, scoped to the Dirichlet-energy cross-application";
RQ1 "methodologically novel" with the per-seed recomputation required (now done); RQ4 "conditionally novel —
a systematic re-quantification"; former RQ5 "original claim disproven, reframe as a negative result." No new
prior-art contradictions surfaced.

---

## Appendix — Original proposal RQs (`proposal.txt` §4), superseded

The proposal posed four *comparison* questions ("does A beat B"). All four were answered and the results are
retained and reported. A literature review conducted after the grid completed found each had close
precedent. The four questions above reformulate the same completed experiments as *attribution* questions,
which the balanced factorial supports and for which precedent is substantially thinner. **No experiment was
discarded, and two original findings were corrected against earlier single-configuration results — both
corrections are reported.**

| # | Question | Result | Prior work that pre-empts it |
|---|---|---|---|
| Orig-RQ1 | Adapter placement: serial vs. parallel | Parallel wins **16/16** (+2.1 to +8.3 pp). Strict Pareto win on MobileNetV3-Small. At 5-shot a 31.7k-param adapter beats Full-FT (91.44% vs 90.47%) at 0.28% of the parameter cost. | [TSA, CVPR 2022, arXiv:2107.00358](https://arxiv.org/abs/2107.00358) — frozen ResNet-18, 600 episodic tasks, parameter-free nearest-centroid head, finds parallel wins "in almost all cases." Same backbone, protocol, and answer. |
| Orig-RQ2 | Evidential vs. softmax calibration under a tiny budget | **No — 0/20.** Evidential ECE worse than softmax by 1.4–9.1×, worse than TS-softmax by 5.3–51×. Accuracy does not compensate (7/20). | Well-studied. [BEL](https://ar5iv.labs.arxiv.org/html/2207.13137) and [BayesAdapter](https://arxiv.org/abs/2412.09718) both report Bayesian calibration *improving*; this negative reads as a boundary case. |
| Orig-RQ3 | Bayesian prior vs. near-OOD detection in low-data regimes | Vacuity beats MSP/TS-MSP in ~37–38/40 cells (mean +0.05 to +0.13 AUROC), advantage grows as shots shrink — but training-free **energy** beats vacuity in ~70% of comparisons, reversing an earlier single-configuration finding. | Vacuity > MSP is the standard EDL claim; energy > MSP is [Liu et al. 2020](https://arxiv.org/pdf/2010.03759). Confirmation at scale. |
| Orig-RQ4 | Latency vs. uncertainty-quality Pareto frontier | Backbone drives latency (5.12×); adapter choice does not (3.9%). Evidential heads ~free at inference (1.29%, below the 5.91% noise floor). Recommended edge point: MobileNetV3-Small + parallel bottleneck + evidential (11.86 ms, 6,930 params). | Pareto reporting is a standard presentational device, not a research finding. |

---

## Sources

- [docs/NEW_RQS.md](NEW_RQS.md) — the earlier five-RQ draft this document supersedes; retains the full
  original novelty-check narrative.
- [docs/RESULTS_MASTER.md](RESULTS_MASTER.md) — full grid tables (accuracy, ECE, Brier, OOD AUROC/FPR@95,
  parameter efficiency, Pareto) and §4 "Positioning against the state of the art."
- `results/mvt_results.json` (120 runs, 2026-08-06) — the aggregated grid every number here traces to.
- `results/rq_factorial/`, `results/rq5/`, `results/rq5_rank_sweep.png`, `results/rq_summary.json` — raw
  outputs backing RQ2, RQ4 and §7.
- [docs/RQ3_MATCHED_BUDGET_PLAN.md](RQ3_MATCHED_BUDGET_PLAN.md) — the implementation plan for §8, including
  `scripts/rq5_sweep.py` as its build template and the `src/adapters/` formula derivations.
- `progress.txt` — canonical status tracker and decisions log.
