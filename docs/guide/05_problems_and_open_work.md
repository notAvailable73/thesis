# 5. Problems, Limits, and Open Work

This page lists everything that is wrong, weak, unfinished or undecided, so nothing surprises you at the
defence. It is split into:

- **A.** problems in our own documents
- **B.** problems in code and tests
- **C.** limits of the study to state in the thesis
- **D.** writing tasks
- **E.** decisions that are still open
- **F.** optional experiments never run

Checked against the repo on 2026-09-14.

---

## A. Problems in our own documents — fix before they reach the thesis

### A1. The "163×" figure is not stable *(high priority — now demonstrated, not just argued)*

RQ2's headline said the scoring rule matters "163×" more than the training objective (far-OOD).

- **Why it's unstable.** The objective's share is tiny, so dividing by it swings a lot.
- **Now proven by the completion run (2026-09-15).** The 21 missing checkpoints were trained and scored,
  taking coverage to **120/120, fully crossed**. The two shares barely moved — far 43.68% → **42.69%**
  for the score, 0.267% → **0.170%** for the objective — but the quotient jumped **163× → 250.5×**.
  Near-OOD: 12.96%/0.601% (21.6×) → **14.72%/0.634% (23.2×)**. Same conclusion, wildly different
  multiplier. That is exactly the instability this item warned about.
- **Fix.** Quote the η² shares — **42.7% vs 0.17% far; 14.7% vs 0.63% near** — or say "two to three
  orders of magnitude". Never the ratio.
- **Where it still needs fixing:**
  - `docs/RQ_SUPERVISOR_REPORT.md` §1 table and §4.2 (the `.pdf` too);
  - `docs/RQ_RESULTS_SUMMARY.md` lines ~150 and ~258;
  - `docs/DEFENCE_SLIDE_PLAN.md`: slide map, slide 2 table, the **slide 24 title**, the slide 24 table,
    and the contributions slide.
  - ✅ `docs/DEFENCE_DECK_25_SLIDES.md` — done (Slides 2, 20, 25, crib, language table).
- **Source.** `results/rq_completion/REPORT.md`, `results/rq_completion/rq2_rq4_summary.json`.

### A2. RQ4 leaves out that evidential is still worse than softmax *(high priority)*

`RQ_SUPERVISOR_REPORT.md` §6 reports the refit improving ECE, but not that the refitted evidential head
is still worse-calibrated than softmax. **Updated to full 120/120 coverage (2026-09-15):**

| | 48 cells (old) | **60 cells (current)** |
|---|---|---|
| ECE improved | 48/48 | **60/60** |
| Still worse than plain softmax | 48/48 (mean 2.18×, best 1.07×) | **60/60** (mean **2.18×**, best 1.07×, worst 5.8×) |
| Still worse than TS-softmax | 48/48 (mean 11.9×, best 2.27×) | **60/60** (mean **13.6×**, best 2.27×) |
| AUROC preserved (Δ ≥ −0.005) | 150/192 (78%) | **192/240 (80%)** |
| Mean ΔECE | −0.1373 | **−0.1387** (0.323 → 0.185) |
| Worst Spearman ρ | 0.9212 | **0.8658** |

Recomputed from `results/rq_completion/rq2_rq4_summary.json` → `rq4_rows` (60 rows). Without this, a
reader may think the refit closes the gap. **Note the direction:** completing coverage made the
ranking-preservation result slightly *weaker* (the 12 new cells preserved 42/48 = 87.5% but had mean
ΔAUROC −0.0016 and contain the new worst-case ρ). Report it that way.

**Status (2026-09-15):** the 48-cell version is in `RQ_SUPERVISOR_REPORT.md` §6.1 and is now itself
outdated; both it and the `.pdf` need the 60-cell numbers. ✅ `docs/DEFENCE_DECK_25_SLIDES.md` Slide 22 is
current.

### A6. The evidence scale/bias were described as "frozen at (2, −6)" *(high priority — fixed in the .md files)*

They are **learnable** (`evidence_affine: true`), initialised at (2, −6). Over the 48 evidential checkpoints
training moved them to scale 1.51–4.52 and bias −9.27 to −5.58 (`results/rq_summary.json` →
`affine_trained`). The error changed the RQ4 story from "the trained operating point was far from optimal"
to "a default was far from optimal".

- **Fixed on 2026-09-14 in:** `01_what_we_built.md`, `03_results.md`, `RQ_SUPERVISOR_REPORT.md` §2.1/§6,
  `RQ_RESULTS_SUMMARY.md` §1.1/§6, and `DEFENCE_SLIDE_PLAN.md` (RQ4 slide table).
- **Still stale:** `RQ_SUPERVISOR_REPORT.pdf`. Regenerate it.

### A7. The grid's `kl_weight_max` is not the value the VAL sweep chose

The Step 4.5 VAL-only sweep ranked `kl_weight_max` = 0.05 first (VAL ECE 0.252 vs 0.260 for 0.1;
`step_writeups/step4_5.txt` §4). `configs/exp_phase2_evidential_retuned.yaml` kept 0.1, and all 120 grid runs
inherit it. No decision-log entry explains why.

- **Impact:** small; the surface is flat.
- **Wording:** do not describe 0.1 as "VAL-selected". Disclose it (`09_methodology.md` §9.5, §9.12 #6).
- **Open:** if someone knows the reason, add it to `progress.txt`.

### A8. The prototype similarity metric is not stated in the reports

Every grid run uses **cosine similarity × 10**. The reports say only "similarity". `configs/base.yaml`
defaults to `l2`, and the ProtoNet summary recommends squared Euclidean distance, so a reader would likely
assume L2. The rationale is documented in `09_methodology.md` §9.3.3.

### A3. Orig-RQ1 row is mislabelled

`RQ_SUPERVISOR_REPORT.md` Appendix A (and the matching row in `RQ_RESULTS_SUMMARY.md`) lists Orig-RQ1 as
"serial versus parallel", then gives "Parallel wins 16/16".

- The 16/16 is **parallel bottleneck vs LoRA**.
- Serial vs parallel was only tested once (Step 6), where they tied on accuracy.

### A4. `progress.txt` is out of date

- The header says "Updated 2026-08-27", and "Next: Step 12 or Phase 6". The supervisor report says no
  experiments are outstanding.
- The Step 4 section still shows "NOT STARTED", although Step 4 ran.
- There is no entry for Phase A or Phase B (RQ2, RQ4, rank sweep). `04_experiments.md` §4.5–4.6 is now the
  only written record of those runs.
- It still uses the original RQ numbers throughout.

### A5. Other stale references

- **`results/slides_outline.md`** is the April pre-defence slide outline, left over in `results/`. It
  describes the old story.
- **Pre-defence / Step 1 files** still sit in `results/` next to current results: `metrics.json`,
  `reliability_plot.png`, `ood_histogram.png`, `training_curve.png`, and `smoketest_*`. They are easy to
  mistake for current results.
- **Retired numbering in the results summary.** `results/rq_summary.json` uses the retired draft
  numbering in its key names: `rq1_verdict` is the **RQ2** answer, and `rq2_rows` / `rq2_verdict` are the
  **RQ4** answer.
- **Citation audit.** `docs/CITATION_AUDIT.md` names some documents deleted on 2026-09-14 as places where
  papers were mentioned. Its paper list is still valid.
- **Code comments** in `scripts/`, `tests/` and `configs/` mention deleted planning files (`plan.md`,
  `instructions.txt`, `implementation.txt`, `docs/superpowers/`). They are harmless but point nowhere. The
  files can be seen with `git show 8d706ea:<path>`.
- **Old step write-ups.** `step_writeups/step4_5.txt` to `step9.txt` contain claims later overturned
  (e.g. "on par with energy"). They are kept as history; don't copy claims from them without checking
  `03_results.md`.

---

## B. Problems in code and tests

| # | Problem | Risk | Fix |
|---|---|---|---|
| B1 | `eta_squared()` in `scripts/rq_aggregate.py` marks a design "balanced" even when a whole cell is missing (it only compares counts of cells that exist) | Silent wrong variance numbers | **Mitigated 2026-09-15**, not fixed: `scripts/rq_completion.py` adds an explicit completeness check (20/20 designs, 0 absent, 0 one-arm-missing, 0 empty objective×score cells) and falls back to OLS Type II when rows are unbalanced. The `rq_aggregate.py` flag itself is still naive — fix it there too |
| B2 | No test for the Step 11 silent latency bug (laptop CPU chosen over Kaggle CPU in 3 scripts) | The same wrong-number bug could come back unnoticed | Add a test with both profiles present (`progress.txt` item 12.K) |
| B3 | Phase A checks run inline only; no `tests/test_factorial_scores.py` or `tests/test_evidence_affine_fit.py` | Regressions not caught by pytest | Move the checks into pytest |
| B4 | TinyImageNet overlap filter for MiniImageNet runs is correct by code reading, but no logged count confirms it | Near-OOD numbers on MiniImageNet could include overlapping classes | Log the removed-class count once |
| B5 | **Root cause found 2026-09-15 — it is worse than "a missing value".** The committed metrics file for `cifar_fs/5shot/mobilenetv3_small/lora/evidential` seed 42 was written by a **20-episode smoke run**, not the 600-episode protocol: 8 metric keys instead of 12, which is why TinyImageNet AUROC was absent and RQ1's near-OOD row had 95/96 | That cell's accuracy/ECE in `results/mvt_results.json` and `RESULTS_MASTER.md` come from 20 episodes. Visible symptom: its ±1.82 seed spread vs ±0.50–0.63 everywhere else at 5-shot | **Impact measured:** re-scoring over 600 episodes and recomputing every η² moves main effects ≤0.70 pp (near-OOD `k_shot` 42.60→43.30%); `backbone:adapter` on ECE 3.28→3.23%. No conclusion changes. Decide: regenerate the grid tables, or disclose in text. See deck Slide 24 self-correction 5 |
| B6 | No repeated full run to confirm Phase A determinism | Low (the 99/99 exact match is strong indirect evidence) | Optional |

---

## C. Limits of the study — state these in the thesis

1. **ImageNet pretraining overlap.** Our backbones saw ImageNet. MiniImageNet test classes *are* ImageNet
   classes. Absolute accuracies can't be compared with few-shot methods trained from scratch. Our
   questions are about *relative* differences, which reduces but doesn't remove this.
2. **One fixed training recipe**, tuned once on ResNet-18 + CIFAR-FS + 5-shot. The grid shows how choices
   compare under one recipe, not each setting's best possible number.
3. **Three seeds only.**
   - The seed changes only the adapter's starting weights, never the order of training episodes.
   - Full FT and Linear Probe have **no** seed spread (effectively 1 run), so claims of beating them by
     under 1 point need care.
4. ~~**RQ2 and RQ4 use 99 of 120 models.**~~ **Closed 2026-09-15.** All 21 were trained and scored;
   coverage is **120/120, fully crossed** (20/20 designs complete, 0 empty objective×score cells). All 21
   reproduced the committed Step 10 grid exactly (max abs diff 0.0, best-epoch 21/21), and the unchanged
   aggregator still reproduces the committed summary on the original 99 records. See
   `results/rq_completion/REPORT.md`.
5. **Two backbones only.** RQ3 shows the backbone decides calibration but cannot say *which property* of it
   does.
6. ~~**RQ1 is main effects only.**~~ **Closed 2026-09-15.** All ten two-way interactions are now computed
   with 2000-resample bootstrap intervals and F-test p-values (`results/rq_completion/rq1_interactions.json`).
   Headline: adding them cuts the unexplained residual from 8.95%→1.31% (accuracy), 7.70%→2.05% (ECE),
   23.87%→9.60% (far-OOD), 11.42%→6.23% (near-OOD). The `backbone:adapter` term is significant **only on
   calibration** (3.28%, p<1e-6 — an independent corroboration of RQ3) and ≈0 elsewhere; the largest
   interaction on the other three outcomes is `dataset:backbone`, which no research question addresses.
   **Still unmodelled:** three-way and higher terms.
7. **Baselines exist only for ResNet-18 + CIFAR-FS.** So "MobileNetV3-Small adapter matches full
   fine-tuning" compares across backbones.
8. **No real edge device.** Latency is from a single CPU thread standing in for one, and is not
   byte-reproducible.
9. **Novelty searches were thorough but not exhaustive.** Not finding prior work does not prove there is
   none.
10. **Far-OOD shows no clear adapter effect in RQ3** (bottleneck wins 11/16 unmatched, 5/8 matched). Report
    it as a non-finding.
11. **One known vacuity problem does not apply here:** every OOD image is scored in the same 5-way episode,
    so class counts match (K = 5). Say this explicitly.
12. **The held-out-class near-OOD pools are the validation classes.** `cifar100_near` / `mini_near` sample
    the 16 VAL classes. The same classes drive early stopping, temperature fitting and the RQ4 refit. They
    are disjoint from the test classes, but model selection has seen them as in-distribution. TinyImageNet
    has no such overlap.
13. **Cosine prototype head, not ProtoNet's L2.** The results describe cosine heads (A8).
14. **No data augmentation.** *A Closer Look* warns this underestimates baselines. It applies to every cell
    equally.
15. **The evidential head is underconfident** in the plotted reliability diagrams (ResNet-18 + bottleneck,
    seed 42): accuracy is above confidence in every bin. Check the other cells before generalising
    (`10_discussion_and_conclusion.md` §10.2.4).

---

## D. Thesis writing tasks

`progress.txt` Step 13 (thesis writing) is **not started**. The thesis text is not in this repo.

**Chapter drafts now exist as source material (2026-09-14):**

| Chapter | Draft |
|---|---|
| Abstract, Ch1 | `07_introduction.md` |
| Ch2 | `08_literature_review.md` |
| Ch3–4 | `09_methodology.md` |
| Ch5 | `03_results.md` (incl. §3.11 full grid tables) |
| Ch6–7 | `10_discussion_and_conclusion.md` |

They still need to be written up in the thesis template, with figures, and reviewed by the supervisor. The
"must write" paragraphs below are drafted in them:

- Bayesian-PEFT differentiation: 08 §8.6.2, 10 §10.2.4.
- LoRA-Ensemble response: 10 §10.2.5.
- Kull name collision and quantization false friend: 08 §8.5.
- RQ3 mechanism stated as a limit: 10 §10.2.3.

Citation clean-up items are collected in 08 §8.10.

### Chapters (Step 13 plan)
- Ch1 Introduction (2–3 pages)
- Ch2 Literature review (5–8 pages)
- Ch3 Methodology (8–12 pages). Document the three changes from the proposal:
  - linear → prototype head;
  - ReLU → softplus;
  - BatchNorm statistics frozen.
- Ch4 Experimental setup (3–5 pages), with a reproducibility statement citing `configs/test_episodes.yaml`
- Ch5 Results (10–15 pages)
- Ch6 Discussion (3–5 pages), one subsection per RQ with a number and a figure or table
- Ch7 Conclusion and future work (2–3 pages)
- Abstract (250 words), an alphabetised BibTeX file, supervisor review, submission

### Must write
- [ ] A paragraph separating this thesis from **"Be Confident in What You Know: Bayesian PEFT of Vision
  Foundation Models" (NeurIPS 2024)**. The names are too close to leave out.
- [ ] A response to **LoRA-Ensemble** (arXiv:2405.14438). Draft: `RQ_SUPERVISOR_REPORT.md` §7.1. Drop the
  Full FT sub-argument.
- [ ] A note that **Kull et al. 2019 "Dirichlet calibration"** is a different thing from evidential
  Dirichlet heads.
- [ ] A note on the "ResNet calibrates better than MobileNet" results found online: they use
  *quantization* calibration, a different meaning.
- [ ] The RQ3 mechanism stated as a limit. Do not invent an explanation.
- [ ] Fixes A1–A3 above, applied to the report and the slide plan.

### Must cite
- **Prior grounding:** arXiv:2601.10836, arXiv:2605.22746, arXiv:2608.10372, Sensoy et al. 2018, Guo
  et al. 2017, Minderer et al. 2021.
- **Near-misses a reviewer might raise:** arXiv:2603.07571, 2605.06382, 2602.01477, 2606.00069,
  2606.10777, 2606.00428, 2308.11838, 2302.09369.
- **The sources of our splits.** Bertinetto et al. (CIFAR-FS) and Ravi & Larochelle (MiniImageNet) are
  missing from the reading list, even though the frozen splits come from them.
- `docs/CITATION_AUDIT.md` lists about 60 works used in the repo but absent from the paper list; 8 listed
  papers are never used; `docs/refs.bib` holds the bibliography. One citation (*LoRA vs Full Fine-tuning:
  An Illusion of Equivalence*, arXiv:2410.21228) was flagged as needing re-checking before use.

### Defence preparation (`docs/DEFENCE_SLIDE_PLAN.md`)
- [ ] Draw the two missing figures: the **system pipeline** (slide 15) and the **factorial design**
  (slide 18). Optional: the two adapter designs side by side (slide 16).
- [ ] Re-check the run counts (slides 2 and 18) and the recommended Pareto point (slide 27) against the
  repo.
- [ ] Apply fixes A1–A3 to the slide text.

---

## E. Decisions still open

| # | Decision | Options |
|---|---|---|
| E1 | Are the optional Step 12 experiments dropped? | Formally drop them in `progress.txt`, or schedule some. The supervisor asked earlier that "every combination should be tested" (item 12.F) |
| E2 | ~~Retrain the 21 missing CIFAR-FS 5-shot models for RQ2/RQ4?~~ | **Done 2026-09-15** — `scripts/rq_completion.py`, outputs in `results/rq_completion/`. C4 closed |
| E3 | Add backbones to find RQ3's mechanism? | Needs a third and fourth backbone; otherwise state it as a limit |
| E4 | Is `proposal.txt` §4 edited to the new RQs, or kept and marked superseded? | Decide with the supervisor; don't keep two "official" versions |
| E5 | Where do the large raw files live? | About 4.7 GB of zips in `results/` are not in git (saved models and per-episode scores). They may be the only copies. Back them up or accept losing the ability to re-analyse without retraining |
| E6 | Keep the GPU relay code? | `src/vgpu/`, `mainul-doc/`, `thesis-gpu/`, `integrations/`, `third_party/gpu_pool`, `docs/gpu-relay-guide.md`, `notebooks/step9-mini-v2.ipynb`, `requirements-vgpu.txt`. Not used by any thesis result |
| E7 | Fix `progress.txt` (A4)? | Update it, or treat this guide as the current status |

---

## F. Optional experiments that were never run

From `progress.txt` Step 12. None has been started.

| Item | What | Estimate | Note |
|---|---|---|---|
| 12.F | Full FT and Linear Probe on every backbone × dataset (24 configs × 3 seeds = 72 runs) | ~18–19 GPU-h | Would make every full-FT comparison same-backbone |
| 12.I | Train a ViT-Tiny / DeiT-Small arm in the same grid | Not estimated | Most direct answer to "why not transformers?" |
| 12.J | Train from a random (not ImageNet) backbone | Not estimated | Removes the pretraining-overlap objection |
| 12.G | Longer KL warm-up / patience, VAL-only, MobileNetV3 parallel evidential | ~9 runs | Motivating symptom didn't reproduce; low priority |
| 12.H | Energy score on evidential models | — | **Effectively done by Phase A** (RQ2) for 99 of 120 models |
| 12.K | Regression tests for the Step 11 bugs | No GPU | Same as B2 |
| 12.A | ConvNeXt-Nano backbone | Not estimated | From the proposal |
| 12.B | BitFit in the grid | Not estimated | Built in Step 5 |
| 12.C / 12.D / 12.E | CUB-200 / ISIC / CIFAR-10-C | Not estimated | From the proposal |
