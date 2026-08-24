# Task Plan — New Research Questions (RQ1–RQ5)

**Created:** 2026-08-23
**Source of questions:** `docs/NEW_RQS.md` (RQ1–RQ5, ranked strongest-to-weakest by the 2026-08-23 novelty cross-check)
**Status of this file:** planning document. Nothing here is started. Update the checkboxes as work lands, mirroring the `progress.txt` convention.

---

## 0. Executive summary — what actually has to happen

Five RQs, but only **two of them need new experiments**. The split is:

| RQ | Needs new compute? | Blocked by? | Est. effort |
|---|---|---|---|
| RQ1 — objective × score factorial | **Yes** — eval-only *if* checkpoints recoverable, else ~36 GPU-h retrain | **T0 (checkpoints)** | 1–2 weeks |
| RQ2 — post-hoc evidence-affine refit | **Yes** — small, but needs trained adapters | **T0 (checkpoints)** | 3–5 days |
| RQ3 — architecture vs. budget | No — existing data | nothing | 2–3 days |
| RQ4 — variance attribution | No — existing data | nothing | 2–3 days |
| RQ5 — interior calibration optimum | **Yes** — ~21 runs, ~6 GPU-h | nothing | 1 week |

**The critical path is T0.** RQ1 and RQ2 are the two highest-ranked (most novel) questions and *both* are gated on whether the 120 grid checkpoints can be recovered. Everything else can proceed in parallel.

**Do T3 and T4 first.** They need zero compute and zero checkpoints, they strengthen claims already in the thesis, and they can run while T0's recovery attempt is in flight.

---

## 1. Ground truth established 2026-08-23 (verified, not assumed)

These were checked against the working copy before this plan was written. Re-verify if time passes.

- **`checkpoints present: 0/120`** — enumerated every `checkpoint` path in `configs/grid/_index.json`; none exist locally. Only two stale Phase-2 files survive (`checkpoints/model_bottleneck_{softmax,evidential}_seed42.pt`), which are **not** grid cells.
- **Why they are gone:** `scripts/run_mvt_grid.py` accepts `--keep-checkpoints seedNN`, which *deletes* every other seed's checkpoint after a successful cell (see `scripts/run_mvt_grid.py:168-173`). The Step 10 grid session very likely ran with this flag to survive Colab/Kaggle disk limits.
- **The RQ1 confound has a single code site:** `_id_score_set()` in [src/evaluators/episodic.py:51](src/evaluators/episodic.py#L51) branches on `interpretation` and returns `{"vacuity"}` for evidential, `{"msp", "energy", "ts_msp"}` for softmax. This one function *is* the missing-cross-terms problem.
- **The evidence map is already centralised:** `head.to_evidence(logits)` (`src/heads/prototype_head.py`) is the single source of truth shared by trainer loss, eval probs, and eval OOD score. RQ2 refits *its* two parameters. Do not reimplement softplus anywhere (this caused the Step 4 evidential collapse).
- **A VAL-only fitting precedent already exists:** `_fit_val_temperature()` in [scripts/evaluate.py:278](scripts/evaluate.py#L278) fits one global temperature on the frozen val seeds `[10000..10099]`. RQ2's affine refit must mirror this pattern exactly.
- **Frozen files (never regenerate):** `configs/test_episodes.yaml` (seeds 0–599), `configs/val_episodes.yaml` (seeds 10000–10099), `data/cifar_fs_split.json`.
- **Existing relevant tests:** `tests/test_ood_scores.py`, `tests/test_temperature.py`, `tests/test_prototype_head.py`, `tests/test_evaluators.py`, `tests/test_episodic_matrix.py`.

---

## T0 — Checkpoint recovery audit  ⚠️ BLOCKS RQ1 + RQ2

**Why first:** the two most novel RQs both need trained adapters. The difference between "recoverable" and "gone" is the difference between ~2 days of evaluation and ~36 GPU-hours of retraining. Decide this before committing to any schedule.

- [ ] **T0.1** Search the Colab/Drive training host for the 120 grid checkpoints. Expected names follow `_index.json`'s `checkpoint` field, e.g. `checkpoints/model_phase2_grid_cifar_1shot_r18_parallel_prototype-evidential_seed42.pt`. Check `MyDrive/` (the repo copy the Step 10 grid ran from) and any Kaggle output/dataset attachments from the Step 10 session.
- [ ] **T0.2** Determine the retention reality: did Step 10 run with `--keep-checkpoints`? Check `results/step11_session.log` and any Step 10 session log for the exact invocation. If it ran with `--keep-checkpoints seed42`, then **40 of 120** (seed-42 cells only) may survive — enough for a single-seed version of RQ1/RQ2, not enough for seed variance.
- [ ] **T0.3** Record the verdict in this file and in `progress.txt`, one of:
  - **(a) All 120 recovered** → RQ1 and RQ2 are evaluation-only. Best case.
  - **(b) Seed-42 subset (40 cells) recovered** → RQ1/RQ2 run at n=1 seed; report without seed error bars and state that limitation explicitly. Acceptable.
  - **(c) None recovered** → RQ1 requires the ~36 GPU-h grid retrain; RQ2 can piggyback on that retrain (fit the affine on the same runs) rather than costing extra.
- [ ] **T0.4** Regardless of outcome: **add checkpoint retention to the run protocol** so this cannot recur. Either drop `--keep-checkpoints` for any run whose outputs feed a future RQ, or write the per-episode logits to disk at eval time (see T1.1 — cheaper than keeping full checkpoints and sufficient for *all* post-hoc scoring work).

**Exit criteria:** a written verdict (a/b/c) with evidence, plus a decision recorded on whether RQ1 proceeds as eval-only or as a retrain.

---

## T1 — RQ1: fully factorised objective × score  *(highest novelty; blocked by T0)*

> *Is the uncertainty benefit produced by the training objective (evidential vs. softmax) or by the OOD scoring rule (vacuity, MSP, energy) — and can the two be separated?*

**The goal:** fill in the missing cross-terms of a 2 (objective) × 4 (score) table. Currently only the diagonal exists.

### The design decision that must be made before any code

Two cross-terms are not mechanically obvious and need a *documented, defensible* convention — get this wrong and the result is meaningless:

- [ ] **T1.0a** **Energy on an evidential-trained model.** Energy is `logsumexp(logits)`. Decide (and justify in writing) whether it is computed on the **raw prototype logits** (pre-`to_evidence`) or on the evidence/alpha. Recommendation: **raw prototype logits** — this keeps energy the same function of the same quantity across both objectives, which is precisely what makes the comparison a clean score-axis contrast. Anything else re-confounds the axes.
- [ ] **T1.0b** **Vacuity on a softmax-trained model.** A softmax-trained cell has no *trained* evidence affine (`scale`, `bias`). Options: (i) apply the frozen grid constants (2, −6); (ii) fit the affine on VAL episodes per cell, exactly as T2 does. Recommendation: **(ii) fit on VAL** — option (i) would hand the softmax arm an untuned mapping and stack the comparison in evidential's favour, which is the mirror image of the bias RQ2 is meant to remove. Whichever is chosen, apply the *same* rule to both arms and state it.
- [ ] **T1.0c** Confirm TS-MSP is well-defined on an evidential-trained model (temperature fit on val logits — the existing `_fit_val_temperature` path is interpretation-agnostic at the logit level; verify this by reading it, don't assume).

### Implementation

- [ ] **T1.1** Refactor `_id_score_set()` ([src/evaluators/episodic.py:51](src/evaluators/episodic.py#L51)) to return **all four scores for both interpretations**, driven by an explicit score list rather than an `if interpretation ==` branch. Keep the `_native_score()` legacy keys populated exactly as now so existing result JSONs and downstream tables do not change shape.
- [ ] **T1.2** **Persist per-episode logits at eval time** (new, small): dump the raw prototype logits + targets + OOD-pool logits per episode to a compressed file alongside `metrics.json`. This makes every *future* post-hoc scoring question (RQ1, RQ2, and anything after) a pure re-analysis with no retraining — the structural fix for the T0 problem.
- [ ] **T1.3** Re-run evaluation across the grid (eval-only if T0 = a/b; as part of the retrain if T0 = c). Write to a **new** results namespace (e.g. `results/rq1_factorial/`) so Step 10's frozen artifacts stay byte-identical.
- [ ] **T1.4** Aggregate into a 2×4 table per (dataset, shot, backbone, adapter) cell, plus a marginal table answering the actual question: **variance attributable to objective vs. attributable to score**.

### Tests (new file: `tests/test_factorial_scores.py`)

- [ ] **T1.5** Every score is produced for **both** interpretations (no silent `KeyError`, no empty dict).
- [ ] **T1.6** **Regression guard:** on a softmax cell, the newly-computed `msp`/`energy`/`ts_msp` values are *bit-identical* to what the old branch produced. Same for `vacuity` on an evidential cell. This proves the refactor added cross-terms without perturbing the diagonal — without it, RQ1 could silently invalidate Step 10.
- [ ] **T1.7** Energy computed on evidential logits equals `logsumexp` of the same raw logits a softmax cell would use (i.e. the T1.0a convention is actually implemented).
- [ ] **T1.8** Determinism: two runs on one fixed config produce byte-identical output (the repo-wide reproducibility invariant).

**Exit criteria:** a populated 2×4 matrix on ≥1 full (dataset × shot × backbone × adapter) sweep; a stated verdict on whether objective or score dominates; T1.6 passing so Step 10's numbers are provably untouched.

---

## T2 — RQ2: post-hoc evidence-affine recalibration  *(blocked by T0)*

> *Can an evidential head be recalibrated after training by refitting only its two evidence-affine parameters, and does its OOD ranking survive that recalibration?*

- [ ] **T2.1** Implement `fit_evidence_affine(model, cfg, ...)` mirroring `_fit_val_temperature()` ([scripts/evaluate.py:278](scripts/evaluate.py#L278)): optimise `(scale, bias)` on the **frozen VAL seeds `[10000..10099]` only**, minimising NLL (matching Guo et al.'s temperature-scaling protocol). **Never touch the 600 test seeds** — this is the repo's hard convention.
- [ ] **T2.2** Route the fitted parameters through `head.to_evidence()` — do **not** add a second evidence path. The Step 4 collapse happened exactly because train and eval evidence maps drifted apart.
- [ ] **T2.3** Report, per cell, four numbers: ECE before/after and OOD-AUROC before/after (all OOD pools). The headline is the **joint** outcome — ECE improvement is only interesting if AUROC survives.
- [ ] **T2.4** Record the fitted `(scale, bias)` per cell and compare against the frozen grid constants `(2, −6)`. If refitting barely moves them, that is itself the answer (and explains the flat ECE surface found in the Step 4.5 sweep).

### Tests (new file: `tests/test_evidence_affine_fit.py`)

- [ ] **T2.5** The fit **only ever reads val seeds** — assert the test-seed loader is never constructed during fitting. This guards the project's single most important scientific convention.
- [ ] **T2.6** Fitting is deterministic under a fixed seed.
- [ ] **T2.7** A degenerate case behaves: fitting on already-optimal logits leaves `(scale, bias)` ~unchanged rather than diverging.
- [ ] **T2.8** **The ranking question, as a test:** construct logits where a per-logit affine *does* reorder vacuity across samples, and assert the code detects/reports it. This turns the paper's central empirical question into an executable check rather than a claim.

**Exit criteria:** ECE-before/after and AUROC-before/after reported for every evidential cell available; an explicit written verdict on whether OOD ranking is preserved, with the counterexample test (T2.8) documenting whether reordering is even possible in practice.

---

## T3 — RQ3: strengthen architecture-vs-budget from existing data  *(no compute, no blockers — START NOW)*

> *Which property of a parameter-efficient adapter governs accuracy, and which governs calibration — its architecture, or its trainable-parameter budget?*

The result already exists. What is missing is statistical rigour and one unclosed literature check.

- [ ] **T3.1** **Fix the statistics** (`docs/NEW_RQS.md` §7 item 6). The current claim is a 16/16 sign test at p≈3.05×10⁻⁵, but **pairs sharing a backbone are not independent** — the budget ordering is constant within each backbone, so the effective n is closer to 2 than 16. Recompute with a test that respects the nesting (e.g. a per-backbone test, or a mixed-effects model with backbone as a random effect). **Expect the p-value to weaken substantially; report the honest one.** A supervisor will find this.
- [ ] **T3.2** Re-derive the ECE comparison on **per-seed** observations rather than seed-averaged, and report how many of the 16 pairs survive at 2× the pooled across-seed SD (the doc currently says 10/16, with only 2/8 CIFAR-FS pairs — confirm this from the raw grid file, don't re-quote it).
- [ ] **T3.3** **Close the open literature item:** manually obtain and read *"Robust Calibration of Large Vision-Language Adapters"* (ECCV 2024) — the 2026-08-23 novelty check could not extract its PDF (failed twice) and flagged it as the single closest-sounding unverified title. Record whether it pre-empts the claim.
- [ ] **T3.4** Address §7 item 1 in writing: is there a **backbone-intrinsic** explanation (not budget) that predicts the same 16/16 ECE pattern? State the counter-hypothesis and say plainly that the grid alone cannot fully separate it — and that T5's rank sweep is what would settle it.

**Exit criteria:** a corrected significance statement that accounts for within-backbone dependence; the ECCV paper checked; the counter-hypothesis stated in the doc rather than left for a reviewer to raise.

---

## T4 — RQ4: tighten variance attribution + reframe the latency claim  *(no compute, no blockers — START NOW)*

> *Across the design axes — dataset, shot count, backbone, adapter, uncertainty head — how is the variance in accuracy, calibration, and OOD detection distributed?*

- [ ] **T4.1** **Recompute η² on per-seed observations.** The current decomposition averages seeds before decomposing, giving n=1 per cell and no residual/error term to test against (`docs/NEW_RQS.md` §7 item 6). Redo it on all 3 seeds per cell so the residual is a genuine error term. Report both, and whether the headline (head-type owns 84.0% of ECE variance) survives.
- [ ] **T4.2** **Reframe, do not re-claim.** The novelty check found the *qualitative* pattern ("calibration and accuracy are driven by different factors") is established since [Guo et al. 2017](https://arxiv.org/abs/1706.04599). Cite it as prior grounding; claim only the rigorous decomposition and the PEFT/few-shot setting as new.
- [ ] **T4.3** **Close the near-miss:** read the methods section of *"A Benchmark Study on Calibration"* ([arXiv:2308.11838](https://arxiv.org/abs/2308.11838), ~117K networks) and determine whether it already performs an ANOVA/η²-style decomposition. If it does, RQ4's methodological novelty narrows sharply and the framing must change again.
- [ ] **T4.4** **Demote the latency claim everywhere it appears.** "Evidential uncertainty is free at inference" is not new — it follows from EDL being a single deterministic forward pass, the explicit selling point of [Sensoy et al. 2018](https://arxiv.org/abs/1806.01768). Already demoted in `docs/NEW_RQS.md`; propagate the same reframing to `docs/RESULTS_MASTER.md`, `step_writeups/step11.txt`, and the presentation decks (`docs/RQ_DEFENSE_SLIDES.md`, `docs/SUPERVISOR_PRESENTATION_*.md`) so no artifact still presents it as a discovery.
- [ ] **T4.5** Verify the ECE↔AUROC correlation-collapse numbers (ρ = +0.433/+0.477 overall, collapsing to +0.150/+0.195 and +0.263/+0.242 within head type) recompute correctly from `results/mvt_results.json`.

**Exit criteria:** η² recomputed per-seed with a real error term; arXiv:2308.11838 checked; the latency claim reframed as confirmation in **every** artifact that repeats it, not just `NEW_RQS.md`.

---

## T5 — RQ5: rank sweep + fix the broken citation  *(independent compute; not blocked)*

> *Does calibration error reach an optimum at an intermediate trainable-parameter budget, and does that budget differ from the one that maximises accuracy?*

This is the weakest-evidenced RQ and the one carrying a citation defect. Both must be fixed.

- [ ] **T5.1** ⚠️ **Resolve the unverified citation — do this before showing RQ5 to anyone.** `docs/NEW_RQS.md` previously cited *"LoRA vs Full Fine-tuning: An Illusion of Equivalence"* ([arXiv:2410.21228](https://arxiv.org/abs/2410.21228)) for ECE 0.018 (Full-FT) vs 0.149–0.152 (LoRA). Two independent fetches found **no calibration/ECE content in that paper at all** — its subject is SVD "intruder dimensions" and forgetting. Either locate the true source of those numbers, or drop the comparison permanently. The numbers are already removed from the doc pending this.
- [ ] **T5.2** **Soften the novelty framing.** [LoRA-Ensemble](https://arxiv.org/abs/2405.14438) already reports ECE degrading at high LoRA rank under a frozen ViT backbone — the **high-budget half** of the U-shape. The defensible claim is the *full* curve (near-zero → full-FT) plus the accuracy/calibration budget mismatch, not the turning point per se. Already noted in the doc; keep it consistent everywhere.
- [ ] **T5.3** **Run the clean rank sweep** that removes the current confound. Today's four budget points confound *budget* with *adapter type* (LoRA vs bottleneck) and with *which weights train* (adapter vs whole backbone). Fix dataset (CIFAR-FS), backbone (ResNet-18), and adapter family (bottleneck), then vary **only** bottleneck rank across ~7 values × 3 seeds ≈ **21 runs ≈ 6 GPU-h** at the measured 1,054 s/run.
- [ ] **T5.4** Generate the configs via `scripts/build_grid_configs.py` (or a sibling) rather than hand-writing 21 YAMLs, so the sweep is reproducible and the seeds are recorded.
- [ ] **T5.5** Plot ECE-vs-rank and accuracy-vs-rank on one axis; report whether the interior optimum survives when adapter type is held constant. **This is also the experiment that settles RQ3's causal ambiguity** (T3.4) — it varies budget with backbone *and* architecture fixed, which the current grid cannot do.

### Tests

- [ ] **T5.6** The generated sweep configs differ **only** in rank and seed — assert every other key is identical across the 21 configs. A single stray hyperparameter difference would silently reintroduce the confound the sweep exists to remove.

**Exit criteria:** citation resolved or claim dropped; 21-run sweep complete; a stated verdict on whether the U-shape holds at fixed architecture — which simultaneously strengthens or breaks RQ3.

---

## T6 — Propagate the new RQs through the repo  *(do last — after T1/T2 verdicts land)*

The repo still describes the **original** four RQs in several load-bearing places. Leaving these inconsistent is how a supervisor ends up reading two different sets of research questions.

- [ ] **T6.1** `CLAUDE.md` — the "four research questions (proposal.txt §4)" block still lists Orig-RQ1–4. Update to the new RQ1–5, keeping the Orig-RQ labels for historical entries.
- [ ] **T6.2** `progress.txt` — add a new phase/step section for the RQ work with checkboxes and exit criteria, matching the existing 13-step convention. Add a decisions-log entry recording *why* the RQs changed (prior work pre-empted the originals).
- [ ] **T6.3** `proposal.txt` §4 — the canonical statement of the RQs. Decide with the supervisor whether this is edited in place or superseded by `docs/NEW_RQS.md`; do not silently fork the source of truth.
- [ ] **T6.4** `docs/RESULTS_MASTER.md` — retarget its RQ sections onto the new numbering.
- [ ] **T6.5** `docs/RQ_DEFENSE_SLIDES.md`, `docs/SUPERVISOR_PRESENTATION_SLIDES.md`, `docs/SUPERVISOR_PRESENTATION_SPEECH.md`, `docs/DEFENCE_BRIEF.md` — all four currently present the *old* RQ1–4 with pass/fail verdicts. Rebuild against the new questions once T1/T2 have answers.
- [ ] **T6.6** Keep `step_writeups/` historical — do **not** rewrite closed step writeups to use new RQ numbers; add a pointer note instead.

**Exit criteria:** no file in the repo states a set of research questions that contradicts `docs/NEW_RQS.md`.

---

## T7 — Standing test debt (carried from Step 11, still open)

- [ ] **T7.1** **Regression test for the Step 11 silent latency-selection bug.** Per `progress.txt`, three independently-implemented functions (`scripts/pareto_plots.py:_primary_cost_profile`, `scripts/make_master_tables.py:_eff_primary_latency`, `scripts/make_results_master.py:_eff_lat`) all picked a dev-laptop profile over the canonical Kaggle CPU profile by dict-insertion-order accident — up to 47% error, no crash, no test failure, caught only by manual cross-check. It is fixed but **still has no regression test**, and it is flagged as the top backlog item precisely because a bug that produces no error is what a test suite exists to catch.
- [ ] **T7.2** Confirm the full suite still passes before and after each of T1/T2/T5 (`python -m pytest`). The repo's reproducibility invariant is that a rerun on the same config yields a byte-identical `metrics.json`.

---

## 2. Suggested execution order

```
NOW, in parallel:
  T0 (checkpoint hunt — unblocks the two best RQs)
  T3 (RQ3 stats fix + ECCV paper)        ← zero compute
  T4 (RQ4 per-seed η² + reframing)       ← zero compute
  T5.1 (kill the bad citation)           ← 10 minutes, highest embarrassment-avoidance per minute

THEN, once T0 returns a verdict:
  T1 (RQ1 factorial) ── the flagship result
  T2 (RQ2 affine refit) ── shares T0's recovered checkpoints
  T5.3–T5.5 (rank sweep) ── independent GPU time, settles RQ3 + RQ5 together

LAST:
  T6 (propagate new RQs through repo + decks)
  T7 (test debt, anytime)
```

## 3. Risks worth stating plainly

1. **T0 comes back "none recovered."** Then RQ1 — the single most novel question — costs ~36 GPU-hours before it produces a first number. Mitigation: T1.2 (persist per-episode logits) makes this a one-time cost rather than a recurring one.
2. **T3.1 weakens RQ3's headline.** Correcting for within-backbone dependence will very likely move p≈3×10⁻⁵ to something far less impressive. This is the right thing to do anyway — better to report it yourself than have it found.
3. **T4.3 could narrow RQ4's novelty.** If arXiv:2308.11838 already does an η²-style decomposition, RQ4's contribution shrinks to "in a PEFT/few-shot setting."
4. **T5.3 could break RQ3.** If the U-shape vanishes when architecture is held fixed, the "calibration follows budget" story loses its cleanest support. Worth knowing before the thesis is written, not after.
5. **Novelty checks are non-exhaustive.** The 2026-08-23 pass was 5 agents × 10–15 searches each. Absence of found prior art is not proof of absence.
