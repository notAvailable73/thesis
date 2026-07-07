# Step 4.5 — "Settle the Science" Design Spec

Date: 2026-07-06
Branch base: `step4-best-of-both`
Status: approved design (pre-plan)
Author note: reasoning follows `thesis_implementation_instructions.txt` — every
component names its source paper(s), states pros/cons/fit, and flags deviations.

---

## 1. Problem & motivation

On the real CIFAR-FS Bertinetto split (Step 4, 5-way 5-shot, 600 episodes) the
thesis's headline claim — *evidential PEFT is more reliable (calibration + OOD)
than softmax* — did not hold:

| metric | evidential | softmax | verdict |
|---|---|---|---|
| accuracy | 0.870 | 0.875 | tie |
| OOD AUROC (SVHN, far) | 0.843 | 0.838 | +0.006 (< 0.05 bar) |
| ECE | 0.344 | 0.119 | evidential worse |
| Brier | 0.353 | 0.190 | evidential worse |

Source: `results/phase2_bottleneck_prototype-{evidential,softmax}_metrics.json`.

Three diagnosed causes (grounded in the paper summaries):

1. **Near-tautological comparison.** Evidential OOD score is vacuity `u=K/S`;
   softmax is `1 − max_p`. Both are monotone functions of the *same*
   cosine-to-prototype logits. The parameter-free prototype head gives the
   evidential model only **2 extra trainable scalars** (`n_params` 16914 vs
   16912), so it has almost no capacity to learn an evidence function distinct
   from "similarity to nearest prototype." → structurally hard to beat MSP.
2. **Wrong OOD test.** Far-OOD SVHN is easy for MSP (Hendrycks 2017 baseline);
   evidential/Dirichlet methods differentiate on **near-OOD**, where MSP
   degrades (Malinin & Gales Prior Networks; OpenOOD near/far split).
3. **Self-inflicted calibration loss.** The aggressive evidence-affine recentre
   + KL-to-uniform anneal (`kl_weight_max: 0.5`) drives ID under-confidence →
   high ECE. The R-EDL relaxation (Survey EDL) shows the rigid "+1" prior and
   variance/KL regularizer can *induce* miscalibration and are droppable.

**Goal of Step 4.5:** produce a *defensible verdict* on whether evidential PEFT
has a real reliability edge, by (a) testing on the right axis (near-OOD), (b)
comparing against the baselines that actually threaten the thesis
(temperature-scaled softmax + energy), and (c) giving the evidential head its
best fair shot (loss retune, not affine-twiddling) — then deciding the thesis
narrative from the result.

## 2. Decision rule (the bar)

Tiered, applied after the experiments:

- **Tier 1 — headline stands:** evidential beats **both** softmax-MSP **and**
  temperature-scaled softmax on **near-OOD** AUROC by ≥ 0.03, and is not
  meaningfully worse-calibrated than TS-softmax on ID.
- **Tier 2 — softened headline:** evidential ties the best baseline on near-OOD
  but wins ID calibration (fair, post-TS) → "competitive uncertainty, better
  calibrated, param-cheap."
- **Tier 3 — honest reframe:** neither holds → "parity at a calibration cost;
  reliability edge is setup-dependent" (a valid Master's finding).

Rationale: a committee will ask "did you compare against temperature scaling?"
(Guo et al. — named in the calibration summary as *the* baseline to beat). All
selection (loss retune, temperature T) is on the frozen **val** episodes; the
600 test episodes are read-only.

## 3. Non-goals (YAGNI)

- No Mahalanobis OOD score (defer to Step 7).
- No CIFAR-10-C corruption robustness (Step 12).
- No added capacity to the evidential head yet — the "2-scalar bottleneck" is
  *documented as a finding/limitation*; only revisit if we land in Tier 3.
- No LoRA / BitFit / full grid work (Step 5+). No changes to backbones.
- No online W&B requirement — metrics are written to disk regardless.

## 4. Workstreams

### W0 — Housekeeping (unblocks reproducibility; cheap)
- Commit the **real Bertinetto** CIFAR-FS split JSON (repo currently holds the
  synthetic fallback → R-SPLIT-DRIFT). Assert disjoint splits, union = 0..99.
- Fix the pre-existing buggy test
  `tests/test_prototype_head.py::test_prototype_head_raises_on_missing_class`
  (it drops the class-2 label so `n_way` infers to 2 and nothing is actually
  missing — the test asserts a raise that legitimately never happens).

### W1 — Fair baselines
**Temperature scaling** (Guo et al. 2017).
- *Fit:* one **global** scalar `T` fit on the frozen val-episode (seeds
  10000–10099) pooled query logits by minimizing NLL; `T` frozen thereafter.
  This is the episodic analog of Guo's "held-out val, model frozen" protocol —
  documented as a methodology choice in the writeup.
- *Apply:* at test time, softmax(logits / T) → recompute ECE/Brier and an
  MSP-based OOD score (TS-MSP). Accuracy is unchanged (T > 0 preserves argmax).
- *Component:* new `src/evaluators/temperature.py` with
  `fit_temperature(logits, targets) -> float` (1-D optimization) and
  `apply_temperature(logits, T) -> probs`.
- *Pros:* trivial, strong, mandatory baseline. *Cons:* needs a val logit dump
  pass. *Fit:* directly answers the thesis's biggest threat.

**Energy score** (Liu et al. 2020, EBO).
- ID-ness score = `logsumexp(logits)` (higher = more in-distribution; energy is
  its negative). Parameter-free.
- *Component:* add `energy_score(logits) -> Tensor` to `src/evaluators/ood.py`.
- *Pros:* nearly free, strongest cheap logit OOD baseline; conceptual parallel
  to Dirichlet strength `S`. *Cons:* none material. *Fit:* contextualizes
  whether evidential OOD is competitive at all.

### W2 — Evidential loss retune (best fair shot, via the loss not the affine)
- Extend `src/losses/evidential.py` so the loss is configurable:
  (a) `kl_weight_max` sweepable including low / 0;
  (b) an **R-EDL** variant (Survey EDL): tunable prior weight `W` replacing the
  rigid `alpha = evidence + 1` with `alpha = evidence + a*W`, and an option to
  drop the variance term. Backward compatible: `W=1`, variance-on reproduces the
  current Sensoy MSE+KL loss exactly.
- Select the operating point (`kl_weight_max`, evidence-affine init, and/or
  R-EDL `W`) by a **val-only** sweep that trades ID-ECE against val-OOD; never
  touch test. This replaces the manual affine-twiddling on
  `configs/exp_phase2_evidential.yaml`.
- *Pros:* principled, is a legitimate RQ2 ablation regardless of verdict.
  *Cons:* EDL losses can be unstable at few classes (EDL summary flags this) —
  keep the collapse guard active. *Fit:* gives evidential its honest best case.

### W3 — Near-OOD evaluation (the decisive test)
- **Refactor `evaluate_episodic`** from single-score/single-OOD-pool to a
  matrix: for each episode, compute ID scores for the query and OOD scores for
  each OOD pool under each applicable scoring method, and accumulate AUROC +
  FPR@95 per (method, OOD-set). Scoring methods:
  - evidential checkpoint → **vacuity** (`1 − u`);
  - softmax checkpoint → **MSP**, **TS-MSP**, **energy**.
- **Near-OOD source:** `src/datasets/tinyimagenet_ood.py` — download (reuse
  `_robust_download.py` pattern), resize to backbone input, sample a fixed
  seeded near-OOD pool. *Secondary, zero-download near-OOD:* CIFAR-100 held-out
  (val-split) classes, disjoint from every episode's support classes — same
  visual domain, novel classes (adapter never received gradient from them).
  TinyImageNet is primary (matches Step 7 plan + OpenOOD); CIFAR-100-heldout is
  a cheap corroborating second near-OOD.
- **Run** best-evidential + softmax over 600 frozen test episodes against:
  far-OOD SVHN (existing) + near-OOD TinyImageNet (+ optional CIFAR-100-heldout).

### W4 — Verdict + writeup
- Aggregate one master comparison table: {score/head} × {acc, F1, ECE (raw &
  TS), Brier, far-OOD AUROC/FPR, near-OOD AUROC/FPR}.
- Apply the §2 decision rule; write `step_writeups/step4_5.txt`
  (supervisor-ready, same honest structure as step4.txt) stating the tier
  reached and the recommended thesis narrative.
- Update `progress.txt` (Step 4 exit-criteria + this sub-step) and the
  `step4-evidential-collapse-fix` memory with the final verdict.

## 5. Architecture & interfaces

New files:
- `src/evaluators/temperature.py` — `fit_temperature`, `apply_temperature`.
- `src/datasets/tinyimagenet_ood.py` — `get_tinyimagenet_ood(...) -> Tensor`
  images (same call shape as `get_svhn_ood`).
- `configs/exp_phase2_evidential_retuned.yaml` — val-selected evidential config.
- `notebooks/step4_5_settle.ipynb` — Colab runner (mirrors step4 notebook).

Extended files:
- `src/losses/evidential.py` — R-EDL `W` + variance-drop options (back-compat).
- `src/evaluators/ood.py` — `energy_score`; keep existing fns.
- `src/evaluators/episodic.py` — multi-score × multi-OOD-pool matrix output;
  optional temperature `T` argument for the TS-MSP score path.
- `scripts/evaluate.py` — build val-logit dump + fit `T`; pass OOD pools + `T`
  into the episodic evaluator; widen the metrics JSON schema (new keys are
  additive; existing keys unchanged for back-compat).

Interfaces stay small and testable: temperature fit is pure
`(logits, targets) -> T`; energy is pure `logits -> score`; the near-OOD loader
matches the SVHN loader signature so the evaluator treats OOD pools uniformly.

## 6. Testing (local, CPU)

- `test_temperature.py`: `fit_temperature` returns ~1.0 on already-calibrated
  synthetic logits; lowers NLL on over-confident logits; `apply_temperature`
  preserves argmax.
- `test_ood.py` (extend): `energy_score` monotonic in logit scale; higher for
  confident-ID than flat-OOD synthetic logits.
- `test_losses.py` (extend): R-EDL loss with `W=1`, variance-on equals current
  `evidential_mse_loss` bit-for-bit; `W`/variance knobs change it as expected.
- `test_tinyimagenet_ood.py`: loader returns correct shape/normalization/dtype
  for a tiny stubbed sample (no network in CI — guard/skip if download absent).
- Fix + keep `test_prototype_head_raises_on_missing_class`.
- Byte-identical metrics JSON on rerun still holds (sort_keys dump unchanged).

## 7. Compute / execution model

- Build + unit-test everything **locally on CPU** (per proposal §6: local Fedora
  CPU dev, Colab GPU training).
- The 600-episode evals + val temperature fit run on **Colab GPU** via
  `notebooks/step4_5_settle.ipynb`. All behavior is config-driven so a Colab run
  is `train.py` / `evaluate.py` + a config.

## 8. Risks

- **R-NEAR-OOD-DATA:** TinyImageNet download flaky on Colab → reuse robust
  downloader; provide the CIFAR-100-heldout near-OOD as a zero-download
  fallback so the verdict is never blocked on a download.
- **R-EDL-INSTABILITY:** evidential loss unstable at 5-way with low KL → keep
  the two-sided collapse guard; sweep on val, abort early.
- **R-TS-LEAKAGE:** temperature/operating-point must be fit on val only — assert
  the val seed list is disjoint from test seeds (10000+ vs 0..599; already true).
- **R-SCHEMA-DRIFT:** widening the metrics JSON must be additive; downstream
  Step 5+ readers must not break. Keep all existing keys.

## 9. Success criteria (done = all of)

- W0 committed: real Bertinetto split frozen; the buggy test fixed; `pytest`
  green.
- TS + energy baselines implemented, unit-tested, and reported on far-OOD SVHN.
- Retuned evidential config selected on val only; documented.
- Near-OOD (TinyImageNet, ≥1 near-OOD set) evaluated for evidential + softmax at
  600 episodes; master table produced.
- `step_writeups/step4_5.txt` states the tier reached + recommended narrative;
  `progress.txt` + memory updated.

## 10. Open questions (resolve during planning, defaults chosen)

- Near-OOD primary = TinyImageNet (default) vs CIFAR-100-heldout. Default:
  TinyImageNet primary, CIFAR-100-heldout secondary/fallback.
- R-EDL depth: implement full tunable-`W` + variance-drop (default) vs only a
  `kl_weight_max` sweep (lighter). Default: implement both knobs but only sweep
  what the val result motivates.
