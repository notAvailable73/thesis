# B-PEFT — Defending the Four Research Questions

**Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**

> Every number below is traceable to `docs/RESULTS_MASTER.md`, `progress.txt`, and the Step 6/10/11
> writeups. This deck only covers the four RQs: what we did, what the closest prior paper already
> answered, and the new answer we're giving that theirs doesn't cover.

**7 slides.**

---

## SLIDE 1 — Title

**B-PEFT: four research questions, four grounded answers**

- 120 training runs, 40 unique (dataset × shot × backbone × adapter × head) configurations, 3 seeds
  each, 600 frozen test episodes per run — 0 errors.
- Scorecard up front, so nothing is hidden until the end: **RQ1 passed, RQ2 failed, RQ3 split, RQ4
  passed.**

---

## SLIDE 2 — The four RQs

| # | Question | In plain terms |
|---|---|---|
| **RQ1** | Does adapter placement (serial vs. parallel) in a frozen CNN change the accuracy/parameter trade-off? | Where do you bolt the adapter on, and is it worth the params? |
| **RQ2** | Does an Evidential Dirichlet head calibrate better than softmax under a near-parameter-free budget? | Are the Bayesian head's confidence numbers more honest? |
| **RQ3** | Does a Bayesian loss prior improve near-OOD detection, especially in low-data regimes? | Does it spot unfamiliar inputs better, especially with less data? |
| **RQ4** | What is the latency-vs-uncertainty-quality Pareto frontier on edge hardware? | What should you actually deploy? |

**Every RQ has been asked, in some form, by someone before us.** The question this deck answers for
each one is narrower and more useful: *does our specific setting — frozen CNN, near-parameter-free
adapter, true disjoint-class episodic few-shot — give the same answer as the closest prior paper, or
a different one?*

---

## SLIDE 3 — RQ1: adapter placement — ✅ PASSED, but the placement half replicates prior work

**What we did:** Step 6 ran serial vs. parallel vs. post-pool head-to-head (frozen ResNet-18, 600 test
episodes, one frozen recipe). Step 10's grid then took the winner and ran it against LoRA across 2
datasets × 2 shot regimes × 2 backbones — 16 matched comparisons.

**What prior work already answered:** **TSA** (arXiv:2107.00358, CVPR 2022) runs this *exact*
comparison — frozen ResNet-18, serial vs. parallel adapter connection, 600 sampled episodic tasks,
parameter-free head — and already concludes parallel/residual connections "perform better than the
serial one in almost all cases." **We replicate TSA's answer; we do not claim it as new.**

**Our new answer:** TSA and Conv-Adapter never compare against LoRA, and never test a second
backbone. We do both:
- Parallel bottleneck beats LoRA in **16/16** matched comparisons, **+2.13 to +8.26 pp** — direct
  evidence LoRA (a transformer-native reparameterisation) is not the right default adapter for
  convolutional backbones.
- On MobileNetV3-Small it's a **strict Pareto win**: 6,928 params vs. LoRA's 10,752 — cheaper *and*
  more accurate.
- A 31,744-param adapter **beats full fine-tuning of an 11.18M-param net by +0.98 pp** at 5-shot.

⚠️ **Caveat:** at 1-shot the ordering flips — Full-FT wins by 2.57 pp. The MobileNetV3-S-vs-Full-FT
comparison is cross-backbone (Full-FT has never been run on MobileNetV3-S). Serial vs. parallel
itself is an accuracy *tie* (0.9145 vs 0.9146) — parallel only wins on the OOD tiebreak.

---

## SLIDE 4 — RQ2: evidential calibration — ❌ FAILED, and it's the interesting kind of failure

**What we did:** Step 4/4.5 found the negative result on one config; Step 10's grid tested it at
scale — 20 matched (dataset, shot, backbone, adapter) comparisons of evidential vs. plain softmax vs.
temperature-scaled softmax ECE.

**What prior work already answered — and all three say "it works":**

| Paper | What's different from us | Their result |
|---|---|---|
| **BEL** (arXiv:2207.13137) | Backbone is **trainable** | Calibration improves (3.59% vs 14.69% ECE) |
| **BayesAdapter** (arXiv:2412.09718, IJCV) | Frozen, but same-class (not disjoint) + CLIP-semantic init | Calibration improves (~2.5 pp ECE gain) |
| **MetaQDA** (arXiv:2101.02833, ICCV 2021) | Frozen, true disjoint episodic (**same protocol family as us**) — but a large ~100K+-param meta-learned prior | Calibration improves *a lot* |

**Our new answer:** in the one setting all three deny at once — frozen backbone **and**
near-parameter-free classifier (2–31,744 total params) **and** true disjoint-class episodic — evidential
calibration **fails decisively, 20/20 matched pairs**, 1.4×–9.1× worse than plain softmax and
5.3×–51.2× worse than temperature-scaled softmax. A real VAL-only hyperparameter sweep found the ECE
surface **flat** (~0.285–0.296) — not a tuning artefact. Theory backs this: Bengs et al. (NeurIPS
2022) prove no loss function gives an evidential model incentive to make its uncertainty *magnitude*
match reality. **We're the first to measure how badly that bites on real vision benchmarks, in the
one deployment corner (edge CNN, tiny budget) that denies the backbone/init/prior capacity that made
BEL, BayesAdapter and MetaQDA succeed.**

⚠️ **Caveat:** temperature scaling is a free post-hoc fix available to softmax and not implemented
for evidential here — the TS comparison is structurally favourable to softmax (still standard and
fair to report, but stated). RQ2's proposal wording ("<500 params") literally fits only the
Linear-Probe cell (2 params); our headline answer widens to the full 2–31,744 range.

---

## SLIDE 5 — RQ3: near-OOD detection — ⚠️ SPLIT, and we correct our own earlier claim

**What we did:** Step 7 established the pattern on the placement winner; Step 10's grid tested vacuity
vs. MSP, TS-MSP, and energy across far/near OOD and 1-shot/5-shot, specifically to check whether the
advantage grows in low data — the actual RQ3 prediction.

**What prior work already answered:** the general literature (Malinin & Gales; Sensoy et al. 2018;
the OpenOOD benchmark) predicts Dirichlet/evidential scores should beat softmax confidence on OOD, and
that near-OOD is where softmax-based scores degrade most. Liu et al. (2020) established **energy** as
a strong logit-space OOD score that needs no Bayesian machinery at all. Nobody had run this at our
scale, in our regime, with both comparisons on the same runs.

**Our new answer, in two parts:**

| Comparison | Far-OOD win rate | Near-OOD win rate |
|---|---:|---:|
| vacuity vs. **MSP** | 38/40 (Δ+0.111) | 37/40 (Δ+0.053) |
| vacuity vs. **TS-MSP** | 38/40 (Δ+0.127) | 38/40 (Δ+0.067) |
| vacuity vs. **energy** | 10/40 (Δ−0.022) | 14/40 (Δ−0.007) |

1. **The low-data hypothesis holds, numerically:** near-OOD advantage over MSP is **+0.064 at
   1-shot vs. +0.043 at 5-shot** — bigger where there's less data, exactly RQ3's prediction.
2. **We overturn our own earlier claim.** Step 4.5 (a single configuration) said evidential was
   roughly on par with energy. At grid scale, energy wins **~70%** of comparisons. We report this
   correction ourselves rather than let it be found.

**The defensible claim:** vacuity is a substantially better OOD ranker than softmax-probability
scores, and the advantage grows in the lowest-data regime — but a well-chosen logit-space score
(energy) still beats it more often than not.

⚠️ **Caveat:** for OOD detection alone, the honest recommendation to a practitioner is *softmax +
energy*, not evidential. Evidential is the better *native probabilistic* option, not the best option
overall.

---

## SLIDE 6 — RQ4: latency vs. uncertainty Pareto — ✅ PASSED, measured not assumed

**What we did:** Step 11 ran the canonical measurement on real hardware (Kaggle T4 GPU + CPU) across
12 (backbone, adapter, head) configurations. Cost and quality axes, and the tolerance band, were
pre-registered before any latency number existed.

**What prior work already answered:** the TinyML/efficiency literature measures latency, params, and
quantisation extensively but never touches calibration, uncertainty, or OOD. The Bayesian-PEFT
literature measures uncertainty but not edge latency. No paper we found puts both axes on the same
plot for a small edge-deployable CNN.

**Our new answer:** a deployable recommended point, measured on real hardware — MobileNetV3-Small +
parallel bottleneck + evidential: **11.86 ms/image, 6,930 params, near-OOD AUROC 0.870 (1-shot) /
0.919 (5-shot)**. The most novel finding: **backbone choice drives latency (5.12×), adapter choice
barely does (3.9–5.8% despite up to 2.6× parameter differences)** — a practical deployment rule
nobody else states this way, because nobody else measures both axes together. "Evidential uncertainty
is free at inference" is now **measured**, not assumed: mean latency delta 1.29%, below this
session's own 5.91% measurement-noise floor.

⚠️ **Caveat:** the Pareto claim is native-score-conditional — if softmax is allowed to use its best
score (energy) instead of MSP, evidential's frontier presence on CIFAR-FS 5-shot collapses to zero.
Two real bugs were found and fixed during closeout, including a **silent** one (no crash, no test
failure) that mis-selected latency numbers by up to 47% on individual cells until a manual cross-check
caught it — flagged as needing a regression test, which does not exist yet.

---

## SLIDE 7 — Scorecard

| RQ | Verdict | What's actually new vs. the closest prior paper |
|---|---|---|
| **RQ1** | ✅ PASSED | Placement result *replicates* TSA; the LoRA comparison and MobileNetV3-S strict-Pareto result do not exist elsewhere |
| **RQ2** | ❌ FAILED (0/20) | Contradicts BEL/BayesAdapter/MetaQDA's "Bayesian calibration works" — because we're the first to deny all three of their capacity sources at once, and theory (Bengs et al.) explains why |
| **RQ3** | ⚠️ SPLIT | Confirms the low-data-amplification prediction numerically; corrects our *own* earlier claim once tested at grid scale instead of one config |
| **RQ4** | ✅ PASSED | First measured (not literature-cited) joint latency/uncertainty Pareto frontier for this regime |

**Say this out loud:** a thesis where all four hypotheses confirmed would be *more* suspicious, not
less. The RQ2 negative result is well-powered (20/20, zero exceptions) and theoretically explained —
it is the finding that most changes what the next researcher should do, and it is the one we can
defend the hardest precisely because it disagrees with the literature for a stated, checked reason.
