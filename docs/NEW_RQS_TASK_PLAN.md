# Task Plan — New Research Questions (RQ1–RQ5)

**Created:** 2026-08-23
**Source of questions:** `docs/NEW_RQS.md` (RQ1–RQ5, ranked strongest-to-weakest by the 2026-08-23 novelty cross-check)
**Status of this file:** T0, T1, T2, T5 executed 2026-08-26 (two Kaggle notebook sessions, "Phase A" = T0+T1+T2, "Phase B" = T5). Results landed in `results/rq_factorial/`, `results/rq5/`, `results/rq_summary.json`, and summarized as **Result** callouts in `docs/NEW_RQS.md`. T3, T4, T6, T7 are still unstarted — update their checkboxes as work lands, mirroring the `progress.txt` convention.

---

## 0. Executive summary — what actually has to happen

Five RQs, but only **two of them need new experiments**. The split is:

| RQ | Needs new compute? | Blocked by? | Est. effort | **Status (2026-08-26)** |
|---|---|---|---|---|
| RQ1 — objective × score factorial | **Yes** — eval-only *if* checkpoints recoverable, else ~36 GPU-h retrain | **T0 (checkpoints)** | 1–2 weeks | ✅ **Done** — score dominates objective 163× (far) / 22× (near) |
| RQ2 — post-hoc evidence-affine refit | **Yes** — small, but needs trained adapters | **T0 (checkpoints)** | 3–5 days | ✅ **Done** — ECE improved 48/48 cells, AUROC preserved in 78% |
| RQ3 — architecture vs. budget | No — existing data | nothing | 2–3 days | ⬜ Not started (T3) |
| RQ4 — variance attribution | No — existing data | nothing | 2–3 days | ⬜ Not started (T4) |
| RQ5 — interior calibration optimum | **Yes** — ~21 runs, ~6 GPU-h | nothing | 1 week | ✅ **Done — hypothesis REJECTED.** No U-shape once architecture is held fixed |

**The critical path was T0.** It came back **partial** (99/120 checkpoints recovered, verdict `b-partial` — see updated §T0 below), which was enough to run RQ1 and RQ2 as evaluation-only on the recoverable subset without a retrain. See `docs/NEW_RQS.md` for the full per-RQ **Result** callouts with numbers.

**Do T3 and T4 next.** They need zero compute and zero checkpoints, they strengthen claims already in the thesis, and nothing blocks them now that T0/T1/T2/T5 are clear.

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

- [x] **T0.1** Search the Colab/Drive training host for the 120 grid checkpoints. Expected names follow `_index.json`'s `checkpoint` field, e.g. `checkpoints/model_phase2_grid_cifar_1shot_r18_parallel_prototype-evidential_seed42.pt`. Check `MyDrive/` (the repo copy the Step 10 grid ran from) and any Kaggle output/dataset attachments from the Step 10 session.
- [x] **T0.2** Determine the retention reality — resolved implicitly: recovery found 99/120, not the clean 40-cell seed-42-only pattern `--keep-checkpoints seed42` would produce, so retention was mixed/partial rather than a single clean flag. See `results/rq_checkpoint_audit.json`.
- [x] **T0.3** Verdict recorded: **(b) partial — 99/120 recovered**, not (a) or (c). Missing 21 cells are concentrated in CIFAR-FS 5-shot (`bottleneck_parallel` + `lora`, both backbones, both head interpretations, all 3 seeds) — see `missing_cells` in `results/rq_checkpoint_audit.json` (`verdict: "b-partial"`). Coverage by slice: `cifar_fs/1shot` 36/36, `cifar_fs/5shot` 15/36, `mini_imagenet/1shot` 24/24, `mini_imagenet/5shot` 24/24. **Decision:** proceeded as evaluation-only on the 99 recovered cells (no retrain) — see T1/T2 results. RQ1/RQ2 findings should state this coverage explicitly wherever quoted (already done in `docs/NEW_RQS.md`'s Result callouts).
- [ ] **T0.4** Still open. The per-episode logits dump capability was **written into the code** (`rq_core.py`'s `logits_out`/`logits_dir` parameters, invoked conditionally in `rq_drivers.py`) but **not activated** in either Kaggle session — no `.npz` files were produced. Turn it on for the next training/eval session so this recovery problem cannot recur; the capability already exists, it just needs a `logits_dir` set in the notebook control panel.

**Exit criteria — MET.** Verdict: **(b) partial, 99/120**, recorded above with evidence. RQ1 and RQ2 proceeded as evaluation-only on the recovered subset.

---

## T1 — RQ1: fully factorised objective × score  ✅ DONE 2026-08-26  *(highest novelty)*

> *Is the uncertainty benefit produced by the training objective (evidential vs. softmax) or by the OOD scoring rule (vacuity, MSP, energy) — and can the two be separated?*

**Result: score dominates objective by 163× (far-OOD) / 22× (near-OOD) in variance explained.** Full numbers in `docs/NEW_RQS.md`'s RQ1 **Result (2026-08-26)** callout; raw data in `results/rq_factorial/` (99 cells) and `results/rq_summary.json` → `rq1`/`rq1_tables`/`rq1_verdict`. Implementation in `scripts/rq_core.py` + `scripts/rq_drivers.py`.

### The design decisions — resolved as follows (confirmed from the actual implementation, not assumed)

- [x] **T1.0a** **Energy on an evidential-trained model.** Implemented as recommended: energy is computed on the **raw prototype logits**, same function for both objectives — confirmed by the near-identical energy AUROC across objectives in the results (0.911 evidential vs. 0.929 softmax, far-OOD), which is only possible if it's the same computation on comparable inputs.
- [x] **T1.0b** **Vacuity on a softmax-trained model.** Implemented as recommended: **fit on VAL** per cell — visible directly in the data as the `vacuity_valfit` score name and the `affine_refit` field (distinct from `affine_trained`, which is `(1.0, 0.0)`/identity for softmax cells since no evidence affine was ever trained for them).
- [x] **T1.0c** TS-MSP confirmed interpretation-agnostic — computed and reported for both objectives in `rq1_tables`.

### Implementation — all done

- [x] **T1.1** `_id_score_set()`-equivalent logic refactored in `scripts/rq_core.py` to return all four scores for both interpretations, with `native_score_name()` preserving the legacy diagonal keys.
- [~] **T1.2** Per-episode logits dump **coded but not activated this session** — see T0.4. Turn on for the next session.
- [x] **T1.3** Ran on all 99 recoverable cells → `results/rq_factorial/*.json` (state coverage: 99/120, concentrated gaps in CIFAR-FS 5-shot — see T0.3).
- [x] **T1.4** Aggregated into the 2×4 table + η² marginal — `results/rq_summary.json` → `rq1_verdict`.

### Tests — regression guard ran inline and passed; standalone pytest file not yet extracted

- [ ] **T1.5** Not yet split into a standalone `tests/test_factorial_scores.py` — the check exists inline in `scripts/rq_core.py` (`regression_guard()`) and ran for real, but isn't in the pytest suite yet. Worth promoting so it runs on every future change to this code.
- [x] **T1.6** **Regression guard ran for real and passed: 99/99 cells `"exact"`** (bit-identical to the committed Step 10 metrics — see the three-tier `exact`/`within_tol`/`MISMATCH` grading in `regression_guard()`, `scripts/rq_core.py`). Step 10's numbers are provably untouched.
- [x] **T1.7** Confirmed by the near-equal cross-objective energy AUROC noted under T1.0a above.
- [ ] **T1.8** Determinism across two independent runs not separately re-verified (the 99/99 exact-match against Step 10's *original* run is strong indirect evidence, but a same-session repeat wasn't done).

**Exit criteria — MET.** 2×4 matrix populated for the full recovered grid (not just one sweep); verdict is unambiguous (`dominant: "score"` in both OOD pools); T1.6 passed at 99/99.

---

## T2 — RQ2: post-hoc evidence-affine recalibration  ✅ DONE 2026-08-26

> *Can an evidential head be recalibrated after training by refitting only its two evidence-affine parameters, and does its OOD ranking survive that recalibration?*

**Result: yes on calibration (48/48 cells improved, mean ΔECE −0.137), mostly yes on ranking (AUROC preserved in 150/192 comparisons, worst-case Spearman ρ = 0.921).** Full numbers in `docs/NEW_RQS.md`'s RQ2 **Result (2026-08-26)** callout; raw per-cell data in `results/rq_summary.json` → `rq2_rows`/`rq2_verdict`.

- [x] **T2.1** Implemented, fitting `(scale, bias)` on VAL seeds only — confirmed in `scripts/rq_core.py` (`assert it is [10000..10099] and disjoint from the 600 test seeds`, ~line 282).
- [x] **T2.2** Routed through the existing `to_evidence()` path — no second evidence path introduced (confirmed by reading `scripts/rq_core.py`).
- [x] **T2.3** Reported per cell: ECE before/after, AUROC before/after across all 4 OOD pools (svhn_far, gaussian_far, cifar100_near/mini_near, tin_near) — see `rq2_rows`.
- [x] **T2.4** Recorded. **Refitting does NOT barely move the affine** — refit values (scale ~7–14) are far from both the frozen constant `(2, −6)` and the trained-evidential affine (~2–4.5), meaning the originally-used operating point was substantially suboptimal. This is itself a finding: it explains why the Step 4.5 ECE surface looked flat — the search never left a bad region.

### Tests — the exact T2.5–T2.8 checks ran inline; not yet extracted to pytest

- [x] **T2.5** Val-only fitting verified by direct code inspection (see T2.1) — ran for real across all 48 cells with no test-seed access.
- [ ] **T2.6** Determinism under fixed seed not separately re-verified this session.
- [ ] **T2.7** Degenerate-case behavior not explicitly tested (would be a good pytest addition).
- [x] **T2.8** **Ran as designed, and the answer is real, not hypothetical.** `ranking_shift()` in `scripts/rq_core.py` measures exactly this — `reordering_ever_observed: true` confirms vacuity's sum-of-logits construction DOES let a per-logit-monotone affine reorder samples in practice, matching the theoretical concern, but the *magnitude* is small (min ρ = 0.921 across every cell × pool).
- [ ] Not yet split into a standalone `tests/test_evidence_affine_fit.py` — same promotion note as T1.5.

**Exit criteria — MET.** ECE and AUROC before/after reported for all 48 recoverable evidential cells; explicit verdict: ranking is preserved in the large majority of cases (not universally — 22% of comparisons dropped by >0.5pp AUROC), and T2.8's counterexample-style measurement confirms reordering is real but empirically minor.

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

## T5 — RQ5: rank sweep  ✅ DONE 2026-08-26 — **hypothesis REJECTED**

> *Does calibration error reach an optimum at an intermediate trainable-parameter budget, and does that budget differ from the one that maximises accuracy?*

**Result: NO interior optimum once architecture is held fixed — `ece_optimum_is_interior: false`.** This is the single most important finding across all three executed phases: it overturns RQ5 as originally framed. Full detail in `docs/NEW_RQS.md`'s RQ5 **Result (2026-08-26)** callout; raw data in `results/rq5/` (21/21 runs) and `results/rq5_rank_sweep.png`.

- [x] **T5.1** **Resolved by supersession, not by finding the source.** The controlled sweep below falsifies the interior-optimum claim the arXiv:2410.21228 citation was supporting, so hunting further for those numbers is no longer worthwhile — the claim they'd support doesn't stand either way. **Decision: drop the citation permanently, already done in `docs/NEW_RQS.md`.**
- [x] **T5.2** Moot for the same reason — RQ5 no longer claims a turning point exists at all, so there's nothing left to soften.
- [x] **T5.3** **Ran exactly as specified:** CIFAR-FS × ResNet-18 × bottleneck-parallel (`parallel`) × evidential, rank ∈ {1,2,4,8,16,32,64} × 3 seeds = 21/21 runs, 0 errors (`[B] done: {'ok': 21, ... 'error': 0}`). Configs in `configs/rq5/`.
- [x] **T5.4** Configs generated via `scripts/rq5_sweep.py`, indexed in `configs/rq5/_index.json`.
- [x] **T5.5** Plotted in `results/rq5_rank_sweep.png`. **Verdict: the interior optimum does NOT survive** — evidential ECE is lowest at rank 1 and drifts upward (noisily) through rank 64; softmax ECE moves the opposite direction, decreasing from rank 1 to rank 64. Neither is U-shaped. `best_ece_rank=1`, `best_accuracy_rank=64` — the accuracy/calibration mismatch itself is *confirmed* to persist, just not via a U-shape. **This also answers RQ3's §7-item-1 causal-ambiguity check (T3.4):** with backbone and architecture both held fixed, budget's effect on calibration is real but head-interpretation-dependent in direction — a more complex, and more defensible, story than either a clean monotonic or a clean U-shaped relationship.

### Tests

- [ ] **T5.6** Not separately verified that the 21 generated configs differ only in rank/seed — worth a quick pytest check on `configs/rq5/_index.json` before citing the sweep as clean, though the tight, sensible ECE/accuracy trends observed are themselves indirect evidence nothing else drifted.

**Exit criteria — MET, with a reframe required.** 21-run sweep complete, verdict stated unambiguously: the U-shape does **not** hold at fixed architecture. This strengthens RQ3 (architecture is the cleaner lever) while requiring RQ5's own framing in `docs/NEW_RQS.md` to shift from "we find an interior optimum" to "the previously-apparent interior optimum was an architecture-change artifact" — already rewritten in the doc.

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
3. **T4.3 could narrow RQ4's novelty.** If arXiv:2308.11838 already does an η²-style decomposition, RQ4's contribution shrinks to "in a PEFT/few-shot setting." **Resolved 2026-08-26 (see T8.2): it does NOT — but a different paper does, on a different metric, which is arguably worse. See T8.2.**
4. **T5.3 could break RQ3 — this risk materialized.** The U-shape vanished when architecture was held fixed (`ece_optimum_is_interior: false`, 2026-08-26). RQ3's core claim (architecture governs accuracy, budget governs calibration *direction*) survives — it was never a claim about a U-shape — but RQ5 needed a real reframe, done in `docs/NEW_RQS.md`. Better this surfaced now, from a controlled experiment, than from a supervisor's question.
5. **Novelty checks are non-exhaustive.** The 2026-08-23 pass was 5 agents × 10–15 searches each. Absence of found prior art is not proof of absence.
6. **A second, deeper pass (2026-08-26) found a genuine partial contradiction for RQ5.** See T8.5 — this is the single highest-priority open item in this document now.

---

## T8 — Literature stress-test of the actual RESULTS (2026-08-26, 5 agents, results-level not question-level)

After T0/T1/T2/T5 produced real numbers, a second round of parallel deep research checked each RESULT (not just whether the question had been asked) against the literature — comparable magnitudes, stronger versions elsewhere, or outright contradictions. Full detail in the **Literature stress-test (2026-08-26)** callouts now under every RQ in `docs/NEW_RQS.md`. Summary and required actions below.

- [ ] **T8.1 (RQ1).** Cite ["One Model, Many Behaviors" (WACV 2026, arXiv:2601.10836)](https://arxiv.org/abs/2601.10836) — a larger-scale training-method × scoring-method ANOVA published the same year — before a reviewer finds it. Also cite [arXiv:2605.22746](https://arxiv.org/abs/2605.22746) (proves softmax is a special case of an evidential classifier — pre-explains why objective barely matters) proactively rather than let it surface as a rebuttal. Verify the protocol controls for the vacuity class-cardinality confound in [arXiv:2605.06382](https://arxiv.org/abs/2605.06382). Reframe RQ1's contribution around the still-open evidential-specific angle (cross-applying energy to Dirichlet logits), not the ANOVA methodology itself.
- [ ] **T8.2 (RQ4, upgrades T4.3).** `arXiv:2308.11838` ("A Benchmark Study on Calibration") confirmed to run **no** variance decomposition at all, despite being the largest calibration study that exists (117,702 architectures) — so RQ4's methodological novelty is *not* narrowed by it. However `arXiv:2601.10836` (same paper as T8.1) runs a real ANOVA on OOD-AUROC **with a genuine per-observation residual term**, which the current RQ4 decomposition lacks (it's seed-averaged). **This makes T4.1 (recompute η² per-seed) mandatory before citing 84.0%/0.2% anywhere** — those numbers may currently be partly a mechanical artifact of missing a real error term. Cite Guo et al. 2017 and [Minderer et al. 2021](https://arxiv.org/abs/2106.07998) as the qualitative prior grounding (already partly done, reconfirm).
- [ ] **T8.3 (RQ3).** Add an explicit citation + differentiation paragraph for ["Be Confident in What You Know: Bayesian PEFT" (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f1fbd5ab8d58d0ecf33c95fd46b900e-Abstract-Conference.html) — a real, distinct paper close enough in name to this thesis's own "B-PEFT" that omitting it would look bad. No other action needed; RQ3 held up as the strongest RQ under adversarial re-checking.
- [ ] **T8.4 (RQ2).** Reframe the ECE-improvement magnitude (−0.137) as *consistent with* known temperature-scaling and few-shot-evidential-calibration magnitudes (Guo et al. 2017; arXiv:2207.13137), not as an unusually large effect. Reframe the ranking-preservation finding as confirming known calibration theory (multi-parameter transforms aren't rank-preserving by construction), not as a discovery. Cite the EDL hyperparameter-sensitivity literature (`arXiv:2510.08938`, `arXiv:2410.00393`) rather than presenting "the frozen default was poorly tuned" as a new finding.
- [ ] **T8.5 (RQ5) — HIGHEST PRIORITY, do before anything else in this list.** [LoRA-Ensemble (arXiv:2405.14438)](https://arxiv.org/abs/2405.14438) reports a genuine ECE reversal at rank 32 on CIFAR-100 (same dataset family as CIFAR-FS) — directly overlapping this thesis's tested range, and in the opposite direction from this thesis's softmax-read curve (which improves monotonically through rank 64). **Write an explicit discussion paragraph** addressing why the two results differ (different architecture — LoRA-on-attention/ViT vs. bottleneck-parallel/CNN; different task — full classification vs. few-shot prototype-based; their reversal sits at the edge of their tested range, so this thesis's architecture may reverse above rank 64, untested). Cite the real "calibration double descent" phenomenon (`arXiv:2302.09369`) and explain why its mechanism (whole-network sparsification hitting an interpolation threshold) doesn't transfer to a frozen-backbone-plus-tiny-adapter setup. **Change every instance of "no interior optimum" to "no interior optimum observed in the tested range"** — the unscoped claim is not defensible given the LoRA-Ensemble contradiction and the noted statistical-power limitation (3 seeds × 7 ranks may be underpowered to detect a subtle U-shape).

**Exit criteria:** every citation in T8.1–T8.4 added where the corresponding RQ is discussed in the thesis; T8.5's discussion paragraph written and the claim's wording scoped everywhere it appears (not just in `docs/NEW_RQS.md`).
