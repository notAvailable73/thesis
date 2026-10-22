# TMLR follow-up — pre-specified hypotheses and decision rules

**Status:** pre-specified. Committed and pushed **before any result of the runs below existed.**
**Commit:** `<filled in by the follow-up commit>`. **Pushed to GitHub:** `<date/time UTC>`.
**Executable form:** [`scripts/tmlr_adjudicate.py`](../scripts/tmlr_adjudicate.py), which is authoritative if this prose and the code ever disagree. Tests: `tests/test_tmlr_adjudicate.py`.
**Design:** [`scripts/tmlr_design.py`](../scripts/tmlr_design.py) and `configs/tmlr/manifest.json` (658 runs).
**Plan:** [`docs/superpowers/plans/2026-10-22-tmlr-publication.md`](superpowers/plans/2026-10-22-tmlr-publication.md), Part A.

This document is called *pre-specified*, not *pre-registered*: it is not deposited with an external registry. The evidence that it predates the results is its public Git history. Optional extra: OSF deposit (plan D6).

Any change after the push is appended to the **Changelog** at the bottom, never edited in place, and every change is reported in the paper.

---

## 1. What is being tested

The thesis found two things this follow-up tests further:
- Accuracy follows adapter **architecture**: the parallel bottleneck beats LoRA 16/16 in the grid and 8/8 at matched budget.
- Calibration follows the **backbone**, not the parameter budget: the budget rule fired in 0 of 4 cells, the backbone rule in 3 of 4.

Reviewers will raise three objections:

1. **"Architecture" is confounded with insertion sites.** The bottleneck sits in every stage; LoRA sits in one conv. On ResNets that conv is on the *shortcut* path (`layer4.0.downsample.0`), while on MobileNets it is on the main path.
2. **Two backbones** cannot show whether the calibration pattern is a property of the backbone *family*.
3. **MiniImageNet's test classes are ImageNet classes.**

## 2. Arms, budgets, seeds

| Arm | Module | Sites |
|---|---|---|
| `btl_ms` | parallel 1×1 bottleneck (ReLU, biases) | the stage-final block of every stage |
| `lora_ss` | LoRA (A keeps the base kernel, B is 1×1, α = rank) | one conv: `default_lora_targets` |
| `lora_ms` | LoRA, per-site ranks | ResNet: `<stage>.conv2`. Inverted residual: the projection conv |
| `btl_ss` | parallel 1×1 bottleneck around one conv | the `lora_ss` site |

**Matched budgets.** Trainable parameters with the softmax head; the evidential head adds 2. They are instantiated and asserted before every session.

| Backbone | Level | btl_ms | lora_ss | lora_ms | btl_ss | spread |
|---|---|---|---|---|---|---|
| ResNet-18 / ResNet-34 | L | r6 12,504 | r16 12,288 | [1,1,2,1] 12,160 | r15 12,047 | 3.79% |
| ResNet-18 / ResNet-34 | H | r16 31,744 | r41 31,488 | [3,3,4,3] 31,360 | r40 31,272 | 1.51% |
| MobileNetV3-Small | L | r16 6,928 | r10 6,720 | [6,6,6,5] 6,864 | r10 6,826 | 3.10% |
| MobileNetV3-Small | H | r22 9,448 | r14 9,408 | [7,7,7,8] 9,464 | r14 9,518 | 1.17% |
| MobileNetV2 | L | r15 11,731 | r9 11,520 | [4,4,4,4,5] 11,648 | r9 11,849 | 2.86% |
| MobileNetV2 | H | r22 17,030 | r13 16,640 | [7,6,6,6,7] 17,080 | r13 16,973 | 2.64% |

**Unmatched (default) pair**, needed by rule B: `btl_ms` r16 vs `lora_ss` r16 on every backbone.
- ResNets: they sit in levels H and L respectively.
- MobileNetV3-Small: `btl_ms` r16 is level L, and `lora_ss` r16 is run as level U.
- MobileNetV2: both are run as level U.

**Seeds.**
- MiniImageNet matched experiments (E1, E2): seeds 42–46.
- Aircraft matched experiment (E3b): seeds 42–44.
- In all three, the training-episode stream varies with the seed (`trainer.vary_train_stream_with_seed`). Seed 42 is identical to the old fixed stream.
- The grid-style runs (E3a, E4, E5) keep the original fixed stream and seeds 42–44.

**Known asymmetries, stated now and reported in the paper:**
- **The single LoRA site differs in kind across backbones.**
  - ResNet-18/34: the shortcut 1×1 conv `layer4.0.downsample.0`, not among the multi-site targets.
  - MobileNetV3-Small: the deepest block's projection conv, which is also the fourth `lora_ms` site.
  - MobileNetV2: the projection conv of `features.17` (960→320). That block is the non-residual block after the last `lora_ms` site (`features.16`).
- **ResNet `lora_ms` uses rank 1 at three of four sites at level L.** That is the cost of matching about 12.5K parameters on 3×3 convs.
- **MobileNetV2 has five residual width groups,** so its multi-site arms insert at five sites.
- **One E5-val cell (kl 0.1 / prior 1.0) is trained twice:** its configuration equals E4's `cifar_5shot_r18_parallel_evidential_seed42`. It is kept as a free determinism check (the two must agree) rather than reusing a checkpoint that does not exist yet when E5-val runs.

## 3. Outcomes (exact keys in each run's `metrics.json → summary`)

- **Accuracy:** `accuracy_mean__<pset>`.
- **Calibration:** `ece_pooled__<pset>`, where `<pset>` is `evidential_native` (evidential head) or `softmax` (softmax head).
- **Near-OOD AUROC:** the mean of `ood_auroc__<near>__<score>` and `ood_auroc__tin_near__<score>`.
  - `<near>` is `mini_near` (MiniImageNet) or `aircraft_near` (Aircraft).
  - `<score>` is `vacuity_native` (evidential) or `msp` (softmax).
- **Far-OOD AUROC** (reported, no rule): the same, over `svhn_far` and `gaussian_far`.

## 4. Statistical conventions

- **σ** is the pooled across-seed SD of the arms involved: √(mean of the arms' variances), sample variance, ddof = 1.
- An effect **fires** when |effect| > 2σ.
- Arm differences are always LoRA-type minus bottleneck-type.
- A **cell** is (dataset, backbone, level, head). Rules B–D average levels L and H inside a (backbone, head) cell, as the thesis did.
- **Completeness:** a rule is evaluated only when every run it needs exists. Otherwise its verdict is *inconclusive*, and nothing is adjudicated from a partial design.
- **Exclusions:** a run is excluded only if it errors or fails the parameter re-check. Such runs are re-run, never dropped. There is no outcome-based exclusion.

## 5. Rules

### R-A — insertion sites vs module type (accuracy; near-OOD AUROC)

Per cell, with arm means m:
- S (site effect) = ½[(m_btl_ms − m_btl_ss) + (m_lora_ms − m_lora_ss)]
- M (module effect) = ½[(m_btl_ms − m_lora_ms) + (m_btl_ss − m_lora_ss)]

Positive S means multi-site is better; positive M means the bottleneck module is better.

Evaluated over the 16 MiniImageNet cells (4 backbones × 2 levels × 2 heads), and separately over the 8 Aircraft cells. With *n* the number of cells:

| Verdict | Condition |
|---|---|
| **site** | S fires positive in ≥ 75% of cells **and** M fires in ≤ 25% |
| **module** | M fires positive in ≥ 75% **and** S fires in ≤ 25% |
| **both** | S and M each fire positive in ≥ 75% |
| **inconclusive** | otherwise |

### R-B — budget, architecture or backbone (ECE; original pair btl_ms vs lora_ss)

This is the thesis's rule, unchanged in definition. Per (backbone, head) cell: Δ = mean over L,H of [ECE(lora_ss) − ECE(btl_ms)]; σ = mean of the level σ's; collapse ratio = mean|Δ_level| / |Δ_unmatched|.

| Rule | Fires in a cell when |
|---|---|
| **budget** | collapse ratio ≤ 0.50 and \|Δ\| ≤ 2σ |
| **architecture** | \|Δ\| > 2σ, and every significant cell shares one sign and the significant cells cover every backbone in the set |
| **backbone_intrinsic** | \|Δ\| > 2σ and sign(Δ) = sign(Δ_unmatched) |

The verdict is the single rule reaching the threshold. If none does, or more than one does, the verdict is inconclusive. Evaluated three times:
1. MiniImageNet, ResNet-18 + MobileNetV3-Small, 5 seeds (threshold 3 of 4). This replicates the thesis with more seeds and a varied stream.
2. MiniImageNet, all four backbones (threshold 6 of 8).
3. Aircraft, ResNet-18 + MobileNetV3-Small (threshold 3 of 4).

### R-C — does LoRA's site role explain the flip? (ECE; lora_ms vs btl_ms)

Per (backbone, head) cell over all four MiniImageNet backbones (8 cells): Δ_ms = mean over L,H of [ECE(lora_ms) − ECE(btl_ms)], with σ as in R-B.

| Verdict | Condition |
|---|---|
| **site_role_explains** | ≥ 6 cells have \|Δ_ms\| > 2σ and all those share one sign; **or** ≥ 6 cells have \|Δ_ms\| ≤ 2σ |
| **flip_persists** | ≥ 6 cells have \|Δ_ms\| > 2σ with the family sign (+ on ResNets, − on MobileNets) |
| **inconclusive** | otherwise |

### R-D — is the calibration sign a family property? (original pair)

Using R-B's MiniImageNet cells, there are four checks: (ResNet-34 vs ResNet-18) and (MobileNetV2 vs MobileNetV3-Small), for each head. A check passes when both cells are significant and share a sign.

- **family_consistent:** 4 of 4 pass.
- **partial:** 3 of 4 pass.
- **not_family_consistent:** ≤ 2 pass.

### R-E — Aircraft replication of the grid findings (E3a: 2 shots × 2 backbones × 2 adapters × 2 heads × 3 seeds)

Four checks, each reported separately:
1. The head is the largest main-effect η² for ECE.
2. On both far-OOD (`svhn_far`, `gaussian_far`) and near-OOD (`aircraft_near`, `tin_near`), the scoring rule's η² exceeds the training objective's η², and the objective's η² is < 2%. The score axis is MSP, TS-MSP, energy and vacuity (VAL-refit).
3. The two-parameter VAL refit lowers evidential ECE (`ece_pooled__evidential_valfit` < `__evidential_native`) in ≥ 90% of evidential models.
4. `btl_ms` beats `lora_ss` on mean accuracy in ≥ 6 of 8 (shots × backbone × head) pairs.

Verdict: **replicated** if all four hold, otherwise **not_replicated**.

### R-F — is "head explains calibration" one EDL recipe's artefact? (E5)

The recipe is selected on VAL only (plan A8): lowest VAL pooled ECE, with ties within 0.002 going to higher VAL accuracy, over `kl_weight_max ∈ {0, 0.05, 0.1}` × `prior_per_class ∈ {1.0, 0.5}`.
- If the selected recipe is the current one (0.1, 1.0), E5 is not run, and this is reported as "the current recipe is VAL-optimal within the sweep".
- Otherwise the evidential half of the 2⁵ grid is re-run with it, and R-F substitutes it for the original evidential runs.
  - **head_dominance_robust:** the head is still the largest ECE main effect **and** evidential ECE > softmax ECE in ≥ 12 of the 16 matched (dataset × shots × backbone × adapter) pairs.
  - **not_robust:** otherwise.

## 6. Reproduction gates (not hypotheses — they gate whether anything above is trusted)

- **E4 and its E1 twins:** all 114 grid cells must reproduce `results/grid/*_metrics.json` as `exact` or `within_tol` (|diff| ≤ 1e-6).
- **E1 seed-42 existing arms:** all 16 must reproduce `results/rq3_matched/*.json` the same way.
- If either gate fails, adjudication stops until the cause is found. Tolerances are never widened.

## Changelog

*(empty — any post-push change is appended here with date, reason and the commit)*
