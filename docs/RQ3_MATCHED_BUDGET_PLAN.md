# RQ3 Matched-Budget Experiment — Pre-Registration

> ## ✅ RUN — 2026-08-27. Verdict: `backbone_intrinsic` (H3.2-alt).
>
> This document is now the **pre-registration record**: the design, the parameter formulas and the §5
> decision rule below were fixed *before* any of the numbers existed, which is what makes the verdict
> non-post-hoc. It is preserved as written — nothing in §1–§6 was edited after the run — except this
> banner and the §7 checkboxes.
>
> - **Result and interpretation:** [RQ_RESULTS_SUMMARY.md](RQ_RESULTS_SUMMARY.md) §5.1.
> - **Raw verdict:** `results/rq3_matched/verdict.json`; per-cell JSONs in `results/rq3_matched/`.
> - **Executed:** 48 runs, not the 30 planned — the three existing grid arms were **re-trained** rather
>   than reused from Step 10 checkpoints (see §6 Step 4). Strictly stronger; it turned the §7 reuse check
>   into a full end-to-end reproduction, which passed exactly.
> - **H3.2 (budget) fired in 0 of 4 cells; H3.2-alt in 3 of 4.** Equalising the budget left ResNet-18's
>   calibration gap intact (collapse ratios 1.01 / 0.92) and roughly halved MobileNetV3-Small's
>   (0.50 / 0.61). Secondary: bottleneck still wins accuracy and near-OOD **8/8**, all beyond 2σ.

**Owner:** implemented 2026-08-27. **Goal:** resolve an open causal-identification gap in RQ3 (below)
that the existing 120-run grid cannot separate on its own. **Cost:** 30 runs ≈ **8.8 GPU-hours** at the
grid's measured mean of 1,054 s/run (actual: 48 runs, ≈ 6.5 h wall on one Kaggle T4).

Every parameter count in this document was **derived from source and validated against committed run
values**; every config key was checked against the actual files in this repo. If you find a discrepancy,
that is a bug in this plan — report it rather than silently working around it.

---

## 0. Context — why this experiment exists

The thesis's RQ3 asks: *within the adapter axis, does accuracy follow adapter architecture while
calibration follows trainable-parameter budget?*

**The evidence so far** (16 matched bottleneck-vs-LoRA comparisons across 2 datasets × 2 shot regimes × 2
backbones × 2 head interpretations): the accuracy winner (bottleneck) never changes even when the parameter
budget ordering reverses between backbones, but the calibration winner changes exactly in step with which
adapter happens to hold the larger budget. Full details, tables, and citations are in
[docs/RQ_RESULTS_SUMMARY.md](RQ_RESULTS_SUMMARY.md) §5 — read that first if you want the complete picture;
this document only needs §0.1 below to be actionable.

### 0.1 The specific gap this experiment closes

The parameter ordering between the two adapters reverses between backbones purely as a side effect of
channel widths:

| Backbone | Bottleneck-parallel | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | 31,744 | 12,288 | bottleneck, by 2.58× (158% more) |
| MobileNetV3-Small | 6,928 | 10,752 | LoRA, by 1.55× (55% more) |

Because that reversal is welded to the backbone, two explanations predict the exact same 16/16 pattern seen
in the data and **cannot be told apart by the existing grid**:

- **H3.2 (budget)** — whichever adapter has more trainable parameters wins calibration.
- **H3.2-alt (backbone-intrinsic)** — something about ResNet-18 vs. MobileNetV3-Small itself (not budget)
  drives the calibration difference.

A prior attempt to settle this with a rank sweep (varying only bottleneck size, one backbone, one
architecture) came back direction-dependent and did not resolve it — see
[docs/RQ_RESULTS_SUMMARY.md](RQ_RESULTS_SUMMARY.md) §7.

**The fix:** build both adapter architectures at the *same* parameter budget, within each backbone. With
budget equalised, H3.1 (architecture), H3.2 (budget), and H3.2-alt (backbone-intrinsic) make **divergent**
predictions for the first time — see §5 below.

---

## 1. The idea in one paragraph

RQ3's evidence rests on a coincidence: bottleneck is the *larger* adapter on ResNet-18 and the *smaller* one
on MobileNetV3-Small. Because that reversal is welded to the backbone, "budget drives calibration" and
"something about the backbone drives calibration" are indistinguishable in the current data. **Building
both architectures at the same parameter budget within each backbone breaks the weld.** With budget
equalised, the three hypotheses predict different things for the first time.

---

## 2. ⚠️ Constraints that will silently invalidate the experiment if missed

1. **Both arms must inherit the same training recipe.** Verified true today: on each (dataset, backbone),
   the bottleneck and LoRA configs already extend a *common parent*, so LR, epochs, patience, KL schedule
   and evidence affine are identical by inheritance. **Never hand-copy hyperparameters into the new
   configs** — override only `adapter.rank` (plus bookkeeping). A guard enforcing this is specified in §6.
2. **`rank` must not exceed the shallowest stage's channel count for bottleneck adapters.**
   `Conv1x1Bottleneck(channels, rank)` builds `Conv2d(channels, rank, 1)`; if `rank > channels` this is an
   *over-complete projection*, not a bottleneck. MobileNetV3-Small's shallowest stage has **24 channels**,
   so bottleneck ranks must stay **≤ 23**. This rules out the otherwise-attractive rank 25 — see §4.
   ResNet-18's shallowest stage is 64 channels, so its ranks are unconstrained in the range used here.
3. **Run this on MiniImageNet 5-shot, not CIFAR-FS.** RQ3's ECE effects clear 2× the across-seed SD in
   **8/8** MiniImageNet pairs but only **2/8** CIFAR-FS pairs. A null result on CIFAR-FS would be
   uninterpretable — indistinguishable from insufficient power. MiniImageNet 5-shot also carries the
   largest gaps to close (ResNet-18: ΔECE ≈ 0.111). **`k_shot: 5` is already the default in
   `configs/base.yaml` — do not override it.**
4. **Train both head interpretations separately.** A prior sweep trained evidential parents only and read
   softmax off the same checkpoints. RQ3's original 16 pairs used *separately trained* evidential and
   softmax cells, so a faithful matched-budget replication must train both objectives. This is why the run
   count is ×2.

---

## 3. Verified parameter formulas

Derived from `src/adapters/placement.py` and `src/adapters/lora.py`; all four reproduce the committed
`n_params` at rank 16 exactly.

```
bottleneck_parallel(r) = Σ_stages (2·C·r + r + C)          # + 2 if evidential
    ResNet-18          stages C = (64, 128, 256, 512)  →  1924·r + 960
    MobileNetV3-Small  stages C = (24, 40, 48, 96)     →   420·r + 208

lora(r)                = r · (C_in + C_out)                # + 2 if evidential
    ResNet-18          layer4.0.downsample.0    (256→512)  →  768·r
    MobileNetV3-Small  features.11.block.3.0    (576→96)   →  672·r
```

Validation at r = 16 against `results/mvt_results.json` (softmax cells): ResNet-18 bottleneck 31,744 ✓,
ResNet-18 LoRA 12,288 ✓, MobileNetV3 bottleneck 6,928 ✓, MobileNetV3 LoRA 10,752 ✓.

---

## 4. The design — 2 backbones × 2 matched budget levels

All cells: **MiniImageNet, 5-shot, 3 seeds (42, 43, 44), both head interpretations.**

**ResNet-18**

| Level | Arm | rank | params (softmax) | params (evid.) | Status | Mismatch |
|---|---|---:|---:|---:|---|---:|
| **L ≈ 12.4k** | LoRA | 16 | 12,288 | 12,290 | **existing grid arm** | — |
| | bottleneck-parallel | **6** | 12,504 | 12,506 | **NEW** | 1.76% |
| **H ≈ 31.6k** | bottleneck-parallel | 16 | 31,744 | 31,746 | **existing grid arm** | — |
| | LoRA | **41** | 31,488 | 31,490 | **NEW** | 0.81% |

**MobileNetV3-Small**

| Level | Arm | rank | params (softmax) | params (evid.) | Status | Mismatch |
|---|---|---:|---:|---:|---|---:|
| **L ≈ 6.8k** | bottleneck-parallel | 16 | 6,928 | 6,930 | **existing grid arm** | — |
| | LoRA | **10** | 6,720 | 6,722 | **NEW** | 3.00% |
| **H ≈ 9.4k** | bottleneck-parallel | **22** | 9,448 | 9,450 | **NEW** | — |
| | LoRA | **14** | 9,408 | 9,410 | **NEW** | 0.42% |

**Note on MobileNetV3 Level H.** The tightest match to the existing LoRA-16 arm (10,752) would be
bottleneck rank 25 (10,708, 0.41%) — but rank 25 violates constraint §2.2 (> 24 channels at the shallowest
stage). Level H therefore uses **rank 22 / rank 14**, a fresh pair at ~9.4k with an even better 0.42% match
and a legitimate bottleneck at every stage. This costs one extra new arm. *(If GPU budget is tight, the
rank-25 vs. existing-LoRA-16 variant saves 6 runs but must be reported with the over-completeness caveat;
prefer the clean version.)*

**Residual budget mismatch is ≤ 3.0%, against the 55–158% mismatch in the original comparison** (on
ResNet-18 the larger arm exceeds the smaller by 158.3%; on MobileNetV3-Small by 55.2%). Report the exact
residual; do not describe the arms as "identical."

**New arms: 5** — `r18/btl/r6`, `r18/lora/r41`, `mnv3/lora/r10`, `mnv3/btl/r22`, `mnv3/lora/r14`.
**Runs: 5 arms × 2 heads × 3 seeds = 30.**

---

## 5. Pre-registered decision rule — write this down before looking at results

Primary outcome per (backbone, level, head):
`ΔECE = mean_seeds ECE(LoRA arm) − mean_seeds ECE(bottleneck arm)`

Unmatched baselines from the existing grid (MiniImageNet 5-shot) — **note the sign flip between backbones,
which is the entire phenomenon under test**:

| Backbone | Head | ΔECE unmatched | Sign |
|---|---|---:|---|
| ResNet-18 | evidential | +0.1111 | LoRA worse (bottleneck is larger) |
| ResNet-18 | softmax | +0.1080 | LoRA worse |
| MobileNetV3-Small | evidential | −0.0124 | LoRA better (LoRA is larger) |
| MobileNetV3-Small | softmax | −0.0152 | LoRA better |

Divergent predictions at matched budget:

| Hypothesis | Prediction | Declared supported when |
|---|---|---|
| **H3.2 — budget** | the sign flip disappears; gaps collapse toward zero on **both** backbones | mean \|ΔECE_matched\| ≤ 50% of \|ΔECE_unmatched\| **and** within 2× pooled across-seed SD of zero, in ≥ 3 of 4 (backbone × head) cells |
| **H3.1 — architecture** | one architecture is better calibrated regardless of budget | ΔECE_matched keeps a **consistent sign across both backbones** with \|ΔECE\| > 2σ, in ≥ 3 of 4 cells |
| **H3.2-alt — backbone-intrinsic** | the original per-backbone signs persist even at matched budget | ΔECE_matched retains its **unmatched sign per backbone** (positive on ResNet-18, negative on MobileNetV3) with \|ΔECE\| > 2σ, in ≥ 3 of 4 cells |

If no rule fires, record **inconclusive** — do not narrate a preferred story from a null.

**Secondary outcomes (control, and a second live test).** Also compute ΔAccuracy and Δ near-OOD AUROC at
matched budget. **H3.1 predicts bottleneck still wins accuracy at matched budget on both backbones** — this
is the architecture half of RQ3 and it is genuinely at risk here. If the accuracy gap *also* collapses when
budgets are equalised, then accuracy was tracking budget too, and RQ3's headline dissociation weakens
substantially. Report this outcome whichever way it falls.

---

## 6. Build steps

**Step 1 — `scripts/rq3_matched.py`.** Mirror `scripts/rq5_sweep.py` closely; it is the proven template for
a controlled sweep in this repo.

Parent configs (verified present; both arms of each backbone already share a common ancestor):

| Backbone | Bottleneck parent | LoRA parent | Common ancestor |
|---|---|---|---|
| ResNet-18 | `exp_phase5_mini_parallel_{head}.yaml` | `exp_phase5_mini_lora_{head}.yaml` | `exp_phase5_mini_base_{head}.yaml` |
| MobileNetV3-Small | `exp_phase5_mini_mbnet_parallel_{head}.yaml` | `exp_phase5_mini_mbnet_lora_{head}.yaml` | `exp_phase5_mini_mbnet_postpool_{head}.yaml` |

where `{head}` ∈ {`evidential`, `softmax`}.

Each generated config is a **minimal override** — never a copied hyperparameter block:

```yaml
extends: ../exp_phase5_mini_parallel_evidential.yaml
seed: 42
adapter:
  rank: 6
output:
  run_tag: rq3m_r18_btl_r6
  results_dir: results/rq3_matched
wandb:
  group: rq3-matched-budget
  tags: [rq3, matched_budget, evidential]
```

Write to `configs/rq3_matched/`, cell id
`rq3m_{backbone}_{arm}_{head}_r{rank}_seed{seed}`, plus `_index.json` in the same shape `rq5_sweep.py`
emits (`cell`, `config`, `run_tag`, `results_suffix`, `checkpoint`, `results_json`, and the metadata fields
`backbone`, `arm`, `level`, `rank`, `head`, `seed`). Derive `checkpoint` and `results_json` with
`_checkpoint_tag` and `_head_descriptor` imported from `scripts/train.py` — do **not** hand-format those
paths.

**Step 2 — parameter assertion (run BEFORE any training).** Build each model and assert the *instantiated*
trainable count equals the intended value from §4. This is the only real safety net: the formulas were
validated at rank 16 but the new ranks have never been instantiated.

```
expected_params(backbone, arm, rank, head) ==  count_trainable_params(build_model(cfg))
```

Abort the whole run on any mismatch. Also assert, per (backbone, level), that the two arms' counts differ by
**≤ 3.1%** — if that fails, the level is not matched and the experiment is void.

**Step 3 — the control guard.** Adapt `assert_only_rank_and_seed_differ` from `scripts/rq5_sweep.py`.
**The allowed-difference set must be widened** relative to that sweep, because two *different adapter
types* are being compared here:

```
ALLOWED = {seed,
           adapter.type, adapter.rank, adapter.placement, adapter.block_ids,
           adapter.lora_targets, adapter.alpha, adapter.dropout,
           output.run_tag, output.results_dir,
           wandb.group, wandb.tags, wandb.mode, wandb.disabled}
```

Assert every *other* merged key is identical across all 30 configs — LR, epochs, patience, KL schedule,
evidence affine, dataset, episode files, n_way, k_shot. Check the **merged** configs, not the YAML source,
so a difference inherited from a parent cannot hide. Fail loudly.

**Step 4 — execute.** Reuse `run_phase_b`-style driving from `scripts/rq_drivers.py`: skip cells whose
output JSON exists, train via `train_cell` (in-process — subprocess output can silently vanish on hosted
notebooks), then evaluate via `factorial_run_one`. Reuse existing grid checkpoints for the three
already-existing arms where recoverable (`_find_grid_twin` is the existing pattern; it will need
generalising beyond its hardcoded CIFAR-FS/5-shot/bottleneck_parallel filter). Honour a `max_minutes`
budget so a Kaggle session ends cleanly mid-sweep and resumes next session.

**Step 5 — aggregate and adjudicate.** Emit `results/rq3_matched/verdict.json` containing: per-cell ECE /
accuracy / near-OOD AUROC with across-seed SD; ΔECE matched and unmatched side by side; which of the three
decision rules in §5 fired; and an explicit `verdict` field ∈ {`budget`, `architecture`, `backbone_intrinsic`,
`inconclusive`}. Extend `scripts/rq_aggregate.py` rather than writing a parallel aggregator.

---

## 7. Acceptance criteria — all met (2026-08-27)

- [x] 30/30 runs complete, no errors in `_run_log.jsonl`. → **48/48**, every row `status: ok`.
- [x] Parameter assertion passed for all 5 new arms (instantiated == intended). → **16/16 arms exact**,
      0 failures; the 5 new ranks instantiated to 12,504/12,506 · 31,488/31,490 · 6,720/6,722 ·
      9,448/9,450 · 9,408/9,410 exactly as §4 predicted.
- [x] Matched-budget mismatch ≤ 3.1% at every level, recorded in the verdict file. → max **3.095%**
      (MobileNetV3 Level L), min 0.425%.
- [x] Control guard passed: only the allowed keys differ across all 30 merged configs. → **48 merged
      configs, 0 unaccounted keys**; within each (backbone, head) group, 0 offending keys.
- [x] Existing-arm numbers reproduce the committed grid values for the three reused arms. → the arms were
      re-trained rather than reused, so this became a full reproduction: **18/18 cells `exact`**, max abs
      diff **0.0** across 12–13 metric keys each.
- [x] `verdict.json` names exactly one of the four outcomes, by the pre-registered rule, without post-hoc
      adjustment. → **`backbone_intrinsic`**, 3/4 cells against a threshold of 3; `budget` 0/4,
      `architecture` 0/4.
- [x] Secondary accuracy outcome reported whichever way it falls. → bottleneck wins **8/8**, all beyond 2σ;
      no cell's accuracy gap collapsed. Near-OOD AUROC likewise 8/8. Reported in §5.1's secondary table.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Bottleneck rank > shallowest stage channels silently becomes over-complete | Constraint §2.2; assert `rank ≤ 23` on MobileNetV3 in the generator |
| New ranks change trainable count in an unanticipated way | Step 2 parameter assertion before any GPU time is spent |
| A parent-config difference silently confounds the comparison | Step 3 guard on **merged** configs |
| Reused grid checkpoints came from a different recipe | Verify reused arms reproduce committed metrics (§7) |
| Null result mistaken for evidence of H3.2 | Decision rule requires a *positive* collapse criterion plus a power check (within 2σ), not merely a non-significant difference |
| Run selected on CIFAR-FS where effects are within noise | Constraint §2.3 |

---

## 9. If this does not get run — *moot; it was run 2026-08-27. Retained as written.*

Report RQ3 exactly as [docs/RQ_RESULTS_SUMMARY.md](RQ_RESULTS_SUMMARY.md) §5 states it: H3.2 as the
better-supported of two live explanations, H3.2-alt named explicitly, and **this experiment described as
the specific test that would discriminate them.** A masters committee does not require every thread closed —
it requires that you can identify what would close it. Naming the experiment you did not have time to run
is a substantially stronger position than an unexamined causal claim.

---

## Sources

- [docs/RQ_RESULTS_SUMMARY.md](RQ_RESULTS_SUMMARY.md) §5 — the full RQ3 writeup, evidence table, novelty
  checks, and identification-limits discussion this plan resolves.
- `scripts/rq5_sweep.py` — the proven controlled-sweep template this plan mirrors.
- `scripts/rq_drivers.py`, `scripts/rq_core.py`, `scripts/rq_aggregate.py` — run drivers, evaluation
  primitives, and aggregation to extend.
- `src/adapters/placement.py`, `src/adapters/lora.py` — the adapter sources the §3 formulas derive from.
- `results/mvt_results.json` — source of the committed parameter counts and unmatched ΔECE baselines in §5.
