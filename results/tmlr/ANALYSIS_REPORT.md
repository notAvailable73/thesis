# TMLR follow-up — analysis report

**Date:** 2026-11-08 · **Runs:** 658 / 658 (652 TEST-mode + 6 VAL-only), 0 errors, 0 missing
**Produced by:** `uv run --frozen python scripts/tmlr_analysis.py` (byte-deterministic: two runs give identical JSON)
**Verdicts:** `results/tmlr/verdicts.json`, computed only by the pre-specified `scripts/tmlr_adjudicate.py` (pushed at commit `787aa25` before any run)
**Supporting numbers:** `results/tmlr/analysis/*.json`
**Numbering:** new four-RQ framing (RQ1 variance shares · RQ2 objective vs score · RQ3 architecture vs budget · RQ4 post-hoc refit). Rules R-A … R-F are defined in `docs/TMLR_PRESPEC.md`.

Every verdict below is reported exactly as computed. "Inconclusive" means the rule ran on a complete design and no outcome reached its threshold. It never means data was missing: all 11 rule evaluations had `complete: true`.

---

## 0. Reproduction gate (pre-spec §6)

| Check | Result |
|---|---|
| Regression anchors (E4 + E1 seed-42 twins + E1 rq3_matched arms) | 129 `exact`, 1 `MISMATCH`, 0 `within_tol` |
| Gate status | **pass_with_explained_reference_defect**, so adjudication proceeds |

**The one MISMATCH** is `e4_cifar_5s_mnv3_lora_ss_grid_r16_evid_s42`.
- **The reference file is defective.** Its committed reference (`results/grid/…mbnet_lora_seed42…evidential_metrics.json`) has `num_episodes: 20`. It is the Step 10a smoke run, thesis problem B5, and the only non-600-episode file among the 120 grid JSONs.
- **The training reproduced exactly.** The new run equals the 600-episode re-score of the original thesis checkpoint (`results/rq_factorial/…`) bit for bit on all 87 shared summary keys.
- **The handling is checked in code.** `tmlr_analysis.reproduction_gate` decides it, and the tolerance is unchanged. Logged in the pre-spec changelog, 2026-11-08.

**E4 reproduces the thesis analyses exactly.** It re-trains the 120-run grid, using the E1 twins for MiniImageNet 5-shot seed 42.

| Thesis analysis | E4 vs thesis |
|---|---|
| RQ1 variance shares, 4 outcomes, main + two-way | max \|diff\| 1.8e-15 (pinned by `test_e4_reproduces_thesis_rq1`, tol 1e-9) |
| RQ2 objective × score, η², far + near | max \|diff\| 8.6e-16 |
| RQ4 refit (60 evidential models) | identical: ECE better in 60/60, AUROC preserved in 192/240, min ρ 0.866 |

"Thesis" here means the 600-episode-corrected numbers (`results/rq_completion/rq1_interactions_smoke_obs_replaced.json`). The *published* RQ1 table still contains the 20-episode observation. Versus that table, main effects move by at most 0.70 pp: near-OOD `k_shot` goes from 42.60% to 43.30% and `head` from 20.98% to 21.44%. No conclusion changes.

---

## 1. Verdict summary

| Rule | Question | Verdict |
|---|---|---|
| R-A Mini, accuracy | sites or module? | **inconclusive** |
| R-A Mini, near-OOD | sites or module? | **inconclusive** |
| R-A Aircraft, accuracy | sites or module? | **inconclusive** |
| R-A Aircraft, near-OOD | sites or module? | **inconclusive** |
| R-B Mini, R18 + MNv3 (thesis replication) | budget / architecture / backbone? | **inconclusive** (thesis: backbone_intrinsic) |
| R-B Mini, all 4 backbones | same | **inconclusive** |
| R-B Aircraft, R18 + MNv3 | same | **inconclusive** |
| R-C Mini | does LoRA's site role explain the calibration flip? | **inconclusive** |
| R-D Mini | is the calibration sign a family property? | **not_family_consistent** |
| R-E Aircraft | do the grid findings replicate off-ImageNet? | **replicated** |
| R-F | is "head explains calibration" one EDL recipe's artefact? | **head_dominance_robust** |

In short:
- **The reliability findings replicate strongly.** These are RQ1's head → calibration and shots → accuracy, RQ2's score ≫ objective and RQ4's refit. They hold on a dataset with no ImageNet class overlap (R-E) and under a VAL-selected alternative EDL recipe (R-F).
- **The RQ3 calibration claim only partly replicates.** With 5 seeds and a varied training stream, it holds on ResNets and does not hold on MobileNets (R-B, R-D).
- **The adapter-architecture question has a clear but backbone-dependent answer** that no pre-specified R-A verdict category covers.

---

## 2. R-A — insertion sites vs module type

S = site effect (multi − single site), M = module effect (bottleneck − LoRA), I = interaction. `*` = |effect| > 2σ. Verdict rule: site ≥ 75% positive and M ≤ 25% firing, module the mirror image, both ≥ 75%, otherwise inconclusive.

### MiniImageNet accuracy (16 cells)

Counts:
- S fires positive in **13 of 16 (81%)**.
- M fires positive in **8 of 16 (50%)**.
- I fires in 5.

"site" fails because M fires in 50% > 25%. "both" fails because M < 75%.

| Backbone | S fires (+) | M fires (+) | Typical S / M |
|---|---|---|---|
| ResNet-18 | 4/4 | 4/4 | S +0.028…+0.057, M +0.014…+0.038 |
| ResNet-34 | 4/4 | 4/4 | S +0.022…+0.039, M +0.009…+0.021 (I also fires 4/4) |
| MobileNetV3-Small | 4/4 | 0/4 | S +0.012…+0.016, M ≤ +0.007 |
| MobileNetV2 | 1/4 | 0/4 | S −0.008…+0.021 |

**Reading (descriptive, not a verdict):**
- **The split is clean.** M fires in exactly the 8 ResNet cells and in none of the 8 MobileNet cells.
- **On MobileNets the bottleneck's accuracy edge is a site effect.** Multi-site beats single-site, and the module adds nothing measurable.
- **On ResNets both effects contribute, plus a positive interaction:** the bottleneck module gains more from multiple sites than LoRA does.

### MiniImageNet near-OOD (16 cells)

Counts: S fires positive in 10/16 (63%), M in 3/16. Inconclusive.
- **ResNets:** S fires in 8 of 8 cells (+0.060…+0.102). M fires in 3, all evidential.
- **MobileNets:** S fires in 2 of 8 (MNv3 softmax). M fires in none.

### Aircraft accuracy (8 cells)

Counts: S fires positive in 5/8, M in 2/8, I in 3/8. Inconclusive.
- **MNv3:** S fires in 4 of 4 (+0.037…+0.064). M never fires.
- **R18:** mixed. S fires in 1, M in 2 (both softmax), I in 3. The R18 evidential cells are noisy (2σ up to 0.064).

### Aircraft near-OOD (8 cells)

Counts: S fires positive in 3, M in 3, I in 2. Inconclusive.

**R18 softmax is the surprise.** S is *negative* and fires at H (−0.042), while M (+0.042) and I (+0.065) fire. So on Aircraft near-OOD, the multi-site LoRA arm is the weak one there, not single-site.

---

## 3. R-B — budget, architecture or backbone (ECE, original pair)

Δ = ECE(lora_ss) − ECE(btl_ms), averaged over matched levels L and H. CR = collapse ratio. Positive Δ = LoRA worse calibrated.

### MiniImageNet, R18 + MNv3, 5 seeds, varied stream (threshold 3/4) → inconclusive

| Cell | Δ matched | 2σ | Significant | Δ unmatched | CR | Fires |
|---|---|---|---|---|---|---|
| R18 evidential | +0.087 | 0.070 | yes | +0.101 | 0.86 | backbone_intrinsic |
| R18 softmax | +0.104 | 0.023 | yes | +0.108 | 0.96 | backbone_intrinsic |
| MNv3 evidential | +0.012 | 0.058 | no | +0.024 | 0.49 | budget |
| MNv3 softmax | −0.006 | 0.015 | no | −0.004 | 1.72 | — |

Counts: backbone_intrinsic 2, budget 1, architecture 0. No rule reaches 3.

### MiniImageNet, all four backbones (threshold 6/8) → inconclusive

- **ResNet-34 behaves like ResNet-18.** Δ = +0.203 (evidential) and +0.129 (softmax), both significant, CR 0.95–0.99. backbone_intrinsic fires in all 4 ResNet cells.
- **MobileNetV2 behaves like MobileNetV3.** No cell is significant. Δ = +0.016 (evidential) and +0.006 (softmax). budget fires on MNv2 softmax (CR 0.48).
- **Counts:** backbone_intrinsic 4, budget 2, architecture 0.

### Aircraft, R18 + MNv3 (threshold 3/4) → inconclusive

| Cell | Δ matched | 2σ | Significant | Fires |
|---|---|---|---|---|
| R18 evidential | −0.029 | 0.021 | yes | architecture, backbone_intrinsic |
| R18 softmax | +0.000 | 0.027 | no | — |
| MNv3 evidential | −0.030 | 0.027 | yes | architecture, backbone_intrinsic |
| MNv3 softmax | −0.008 | 0.024 | no | — |

**Surprise:** on Aircraft, R18 evidential flips sign relative to MiniImageNet (−0.029 vs +0.087). LoRA is the *better*-calibrated arm on both backbones there.

### Per-backbone exact sign tests (supplementary, never pooled)

Pairs are matched ECE deltas by (level, head, seed).

| Dataset | Backbone | LoRA worse / better | p (two-sided) |
|---|---|---|---|
| Mini | ResNet-18 | 20 / 0 | 1.9e-6 |
| Mini | ResNet-34 | 20 / 0 | 1.9e-6 |
| Mini | MobileNetV3-Small | 7 / 13 | 0.26 |
| Mini | MobileNetV2 | 11 / 9 | 0.82 |
| Aircraft | ResNet-18 | 4 / 8 | 0.39 |
| Aircraft | MobileNetV3-Small | 2 / 10 | 0.039 |

---

## 4. R-C and R-D

### R-C (lora_ms vs btl_ms, ECE, 8 Mini cells) → inconclusive

- **Counts:** significant in 3 cells (R18 softmax +0.060, R34 evidential +0.076, R34 softmax +0.034), all positive. Within 2σ in 5 cells.
- **Why it's inconclusive:** "site_role_explains" needs ≥ 6 significant cells sharing one sign, or ≥ 6 quiet cells. "flip_persists" needs ≥ 6 with the family sign. Neither is reached.
- **Descriptive:** all 8 Δ_ms are positive, including the MobileNets. The family sign predicted *negative* there, but none of the MobileNet cells is significant.

### R-D (family consistency, original pair) → not_family_consistent (2 of 4 checks pass)

| Check | Result |
|---|---|
| ResNet-34 vs ResNet-18, evidential | pass (both significant, same sign) |
| ResNet-34 vs ResNet-18, softmax | pass |
| MobileNetV2 vs MobileNetV3, evidential | fail |
| MobileNetV2 vs MobileNetV3, softmax | fail |

**Precise reading:** the MobileNet checks fail because *no* MobileNet cell is significant, not because the two MobileNets disagree in sign. Within the ResNet family the sign is consistent. Within the MobileNet family there is no resolvable matched-budget calibration gap at all.

---

## 5. Stream robustness (thesis design vs new design)

Same rule (R-B), R18 + MNv3, MiniImageNet:

| Design | Verdict | backbone_intrinsic / budget |
|---|---|---|
| Thesis: fixed stream, seeds 42–44 (`rq3_matched` + grid r16 pair) | **backbone_intrinsic** | 3 / 0 (reproduces `results/rq3_matched/verdict.json`) |
| New: varied stream, seeds 42–44 only | inconclusive | 2 / 0 |
| New: varied stream, seeds 42–46 | inconclusive | 2 / 1 |

| Cell | Δ old → new | σ old → new | Significant old → new |
|---|---|---|---|
| R18 evidential | +0.112 → +0.087 | 0.018 → 0.035 (×2.0) | yes → yes |
| R18 softmax | +0.099 → +0.104 | 0.006 → 0.011 (×1.9) | yes → yes |
| **MNv3 evidential** | **−0.006 → +0.012** | **0.0008 → 0.029 (×37)** | **yes → no** |
| MNv3 softmax | −0.006 → −0.006 | 0.007 → 0.008 | no → no |

**This is the most important change versus the thesis.**
- **The thesis's third backbone_intrinsic cell rested on an implausibly small σ.** That cell is MNv3 evidential, with σ = 0.0008. Under the fixed stream, the seed changes only the adapter initialisation (progress.txt, 2026-08-06), so the seeds were nearly identical.
- **Varying the stream removes the effect.** σ grows 37× and the Δ changes sign. The effect is caused by the stream, not by the seed count: seeds 42–44 alone already lose it.
- **The ResNet effect survives** the doubled σ with a large margin.

---

## 6. R-E — Aircraft replication (E3a, 48 runs) → replicated

| Check | Result |
|---|---|
| 1. Head is the largest ECE main effect | yes: head η² = 93.6% [92.4, 94.8] (shots 2.4%) |
| 2. Score η² > objective η², objective < 2% | yes. Far: score 53.2% [48.9, 57.6] vs objective 0.06%. Near: 46.4% [45.2, 47.6] vs 0.18% |
| 3. VAL refit lowers evidential ECE in ≥ 90% | yes, 24/24 |
| 4. btl_ms beats lora_ss accuracy in ≥ 6/8 | yes, 8/8 |

Other numbers:
- **Accuracy:** shots η² = 80.5% [77.6, 83.2] and adapter 11.9%, the same ordering as the thesis.
- **RQ4 on Aircraft:**
  - **ECE:** improves in 24/24 models, mean Δ −0.128.
  - **OOD AUROC:** preserved (≥ −0.005) in 79/96 comparisons (82%), against the thesis's 80%.
  - **Ranking:** the minimum Spearman ρ is 0.39, against 0.87 in the thesis grid. The refit reorders the OOD ranking more on Aircraft. Worth a sentence.

---

## 7. R-F — alternative EDL recipe (E5, 48 runs) → head_dominance_robust

- **The selected recipe** was chosen on VAL only: `kl_weight_max = 0.05`, `prior_per_class = 0.5`.
- **With that recipe substituted,** the head is still the largest ECE main effect: η² = 68.5% [66.5, 70.4], against 83.0% [82.0, 84.0] with the original recipe.
- **Evidential is still worse calibrated than softmax in 16/16 pairs.**
- **The recipe does help.** It lowers evidential ECE in 16/16 pairs, with a mean of 0.296 → 0.238 (softmax: 0.086), and accuracy moves 0.833 → 0.838.

So the head share falls by 14.5 pp, but the ordering and the sign are robust.

---

## 8. RQ2 with six scores (MSP, TS-MSP, energy, vacuity-VAL-refit, MaxLogit, Mahalanobis)

| Design | Pools | Score η² [95% CI] | Objective η² [95% CI] |
|---|---|---|---|
| Grid (E4, 120 models) | far | 26.8% [25.7, 27.9] | 0.42% [0.26, 0.60] |
| Grid (E4, 120 models) | near | 33.0% [32.3, 33.6] | 0.20% [0.15, 0.24] |
| Matched Mini (E1 + E2, 350 models) | far | 32.3% [31.3, 33.4] | 1.67% [1.34, 2.02] |
| Matched Mini (E1 + E2, 350 models) | near | 53.9% [53.1, 54.6] | 0.00% [0.00, 0.01] |
| Aircraft (E3a, 48 models) | far | 40.2% | 0.59% |
| Aircraft (E3a, 48 models) | near | 31.8% | 0.06% |

**The RQ2 conclusion holds with six scores everywhere.** The scoring rule dominates, and the objective stays under 2%.

The far-OOD score share on the grid is lower than with the thesis's four scores (42.7%). A likely reason, not tested here: the two new scores sit at opposite ends (MaxLogit near energy, Mahalanobis well below), which changes the between-score spread the share is measured against.

Mean AUROC on the grid:
- **Best scores:** MaxLogit and energy (far ≈ 0.93 softmax, near ≈ 0.84–0.85).
- **Vacuity-VAL-refit** is close behind (far 0.913 vs 0.931 softmax).
- **MSP / TS-MSP** are about 0.79–0.80.

**Surprise: Mahalanobis is the weakest score on near-OOD everywhere.** On the grid it scores 0.68, below MSP. On far-OOD it is also low (0.78–0.83).
- **Caveat:** this is our flagged variant: tied covariance from TRAIN-class embeddings, prototypes as means, no input preprocessing (plan A7).
- **Paper wording:** report it as this variant's result, not as "Mahalanobis fails".

---

## 9. Other checks

- **All evidential models are underconfident.** 350/350 natively (median signed gap −0.318, max −0.158). After the VAL refit the gap shrinks but stays negative in 350/350 (median −0.172, max −0.043). The refit reduces underconfidence and never overshoots into overconfidence.
- **Paired episode bootstrap.** These are seed-by-seed btl_ms − lora_ss accuracy at matched budget, over 600 shared TEST episodes. The 95% CI excludes zero:
  - ResNet-18, ResNet-34 and MNv3: in every seed of every cell, and always positive.
  - MobileNetV2: in 4–5 of 5 seeds per cell, but 1–2 seeds per cell are negative, some of them significantly.
  - Aircraft: in every seed, and always positive.

  Episode noise is therefore small next to across-seed variation. The σ that drives R-A/R-B is seed (training) variance, not test-episode noise.

---

## 10. What changed versus the thesis

1. **RQ3 calibration claim: narrower.**
   - **The thesis said** calibration follows the backbone (backbone_intrinsic, 3 of 4 cells).
   - **With a varied training stream and 5 seeds,** that holds for ResNets (4 of 4 ResNet cells, R18 and R34, p = 1.9e-6 per backbone). MobileNets show no resolvable matched-budget calibration gap (0 of 4 cells significant).
   - **The thesis's MobileNet cell was a fixed-stream artefact** (σ = 0.0008; §5).
   - **The R-B verdict is inconclusive at every threshold.**
   - **Defensible wording:** "on ResNets, LoRA at matched budget is consistently and substantially worse calibrated than the bottleneck adapter, and this does not depend on the budget; on MobileNets no matched-budget calibration difference is detectable."
   - **On Aircraft the ResNet sign does not carry over** (evidential reverses).
2. **RQ3 accuracy claim: refined.**
   - The bottleneck still wins accuracy at matched budget.
   - On MobileNets the win is a site effect: more insertion points, not the module. On ResNets both effects contribute and they interact.
   - R-A is inconclusive by rule because the two families behave differently.
3. **RQ1, RQ2 and RQ4 hold** on a re-trained grid (bit-identical), on a non-ImageNet dataset (R-E replicated), under an alternative EDL recipe (R-F robust) and with two more OOD scores.
4. **The B5 data-quality defect is now closed for the paper.** The E4 numbers are the 600-episode numbers. Published-table shifts are ≤ 0.70 pp.
5. **Standing wording rules still apply.** Never "calibration follows the parameter budget". No "163×"/"250×" ratios. The rules are "pre-specified", not "pre-registered".

---

## 11. Limits of this analysis

- **R-A/R-B thresholds pool backbone families.** Several "inconclusive" verdicts come from the two families behaving differently. The per-backbone tables above are descriptive and were not pre-specified. Present them as exploratory.
- **The matched-budget experiments are MiniImageNet and Aircraft, 5-shot only.**
- **ω² is reported next to η²,** using the standard fixed-effects formula (general knowledge; no paper summary covers it). It is close to η² for the large main effects (for example R-F head: 68.41% vs 68.49%). The largest gap anywhere is 3.8 pp, on a two-way term (Aircraft near-OOD design:score, 48 models). Quote ω² for small-sample interaction terms.
- **η² CIs are percentile bootstraps** over seeds within each configuration (2000 resamples, seed 20261108). They hold the design fixed and reflect only seed variance.
