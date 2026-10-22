# TMLR Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the B.Sc. thesis *Reliability in Few-Shot Classification with Frozen CNN Adapters: A Controlled Study* into a TMLR submission by running the experiments a reviewer will demand (658 runs across 4 Kaggle accounts in parallel), re-analysing everything with stronger statistics, and writing and releasing the paper.

**Architecture:** New adapter arms, backbones, a dataset and OOD scores are added to the existing factory-based codebase without changing any old config's behaviour. A single design module turns every experiment into a manifest of `RunSpec`s. A shard packer turns the manifest into Kaggle-session-sized shards with two GPU lanes each. One parametrised Kaggle notebook runs any shard. A collector merges shard outputs back into `results/tmlr/`. Pre-specified decision rules (committed and pushed before the first real wave) produce the verdicts.

**Tech Stack:** Python 3.12, PyTorch + torchvision (ImageNet weights), scikit-learn (Ledoit–Wolf, AUROC), PyYAML, pytest; Kaggle notebooks (GPU T4 x2); optional Hugging Face Hub dataset repo for central result storage; LaTeX with the TMLR style file.

**Spec:** Part A of this document. It is the decision record from the 2026-10-22 publishability review (venue = TMLR; checkpoints lost; 4 Kaggle accounts available).

## Global Constraints

- Read `thesis_implementation_instructions.txt` before any implementation task. For every new component, write the short "papers / pros-cons-fit / deviations / limitations" note it requires (§5) in the task's PR description or commit message draft.
- New methods must be grounded in a `PAPER SUMMARIES/*.txt` entry. If none exists, Task 0.2 adds one from the PDF **before** the code task starts. Never invent hyperparameters.
- **Never commit, add or push.** Every "🔒 HUMAN GATE" step means: stop, show the diff summary and test output, and wait for the human to commit and push.
- Never regenerate or edit any file marked "DO NOT REGENERATE — frozen". This includes `configs/test_episodes.yaml` (seeds 0–599), `configs/val_episodes.yaml` (seeds 10000–10099), the CIFAR-FS and MiniImageNet splits, and, once created, `AIRCRAFT_SPLIT`.
- Hyperparameter selection uses the VAL seeds 10000–10099 only. The 600 TEST seeds are touched only for final reporting.
- Every evidence computation goes through `PrototypeHead.to_evidence()`. Never reimplement softplus-of-logits.
- Existing results are read-only: `results/grid/`, `results/rq3_matched/`, `results/rq_completion/`, `results/rq_factorial/`, `results/mvt_results.json`. All new outputs go under `results/tmlr/`.
- Existing configs (`configs/grid/`, `configs/rq3_matched/`, `configs/rq5/`, `configs/exp_*.yaml`) stay byte-identical in behaviour. All new knobs default to the old behaviour.
- Do not touch, read or propose the GPU relay path (`src/vgpu/`, `mainul-doc/`, `thesis-gpu/`, `*-v2.ipynb`, `requirements-vgpu.txt`).
- Kaggle runs use the **T4** accelerator only (never P100), and code is cloned at a pinned `GIT_SHA`.
- Determinism: same config + same `GIT_SHA` + T4 ⇒ identical metrics. Every run that repeats a committed run carries a `regression_ref` and must come back `exact` or `within_tol` (≤ 1e-6).
- Use the thesis's own wording rules. Never write "calibration follows the parameter budget", never quote the "163×" or "250×" ratio, and do not call the new decision rules "pre-registered" (say "pre-specified, committed and pushed before the runs").

## Review Focus

1. **A Kaggle session dies mid-shard** (12 h cap, preemption, OOM). Expected: finished runs survive in the output, and re-running the same shard with the old output attached skips them. Pinned by `test_skips_done_runs` (Task 8) and the notebook's `try/finally` pack (Task 9).
2. **One lane process crashes silently.** Expected: the notebook reports the missing run IDs and the version ends in error *after* packing. Pinned by `test_shard_report_lists_missing` (Task 8).
3. **Kaggle's PyTorch version differs from August's**, so old runs no longer reproduce. Expected: Wave 0's regression runs fail loudly before any real wave. Pinned by the `regression_ref` guard (Task 7, `test_regression_refs_assigned`) and the Wave 0 gate.
4. **A rank vector is mistyped**, so the arms are no longer matched. Expected: the preflight refuses to spend GPU time. Pinned by `test_expected_params_match_instantiated` and `test_level_spread_within_tolerance` (Task 7), and re-checked per run in the runner (`test_param_drift_marks_error`, Task 8).
5. **Two accounts produce the same run ID with different contents** (wrong `GIT_SHA`, edited config). Expected: the collector refuses the conflict instead of silently picking one. Pinned by `test_conflicting_duplicate_raises` and `test_stale_config_flagged` (Task 10).

---

# PART A — Decisions (the spec)

## A1. Venue and framing

- Target: **TMLR** (OpenReview, double-blind, rolling). Its acceptance bar is "claims supported by accurate, convincing evidence" plus "of interest to some of the audience". Novelty is not required.
- Lead with RQ2 (objective vs. score) and RQ3 (architecture vs. budget, now decomposed into insertion sites vs. module type). The evidential calibration result plus RQ4 come third. RQ1 is a framing table.
- Drop from the paper: adapter-beats-full-FT (one effective run, one backbone), the latency section (not new), and the SOTA table (ImageNet overlap makes it a liability).

## A2. Experiments (658 runs — as implemented in `scripts/tmlr_design.py`; see the note under the table)

| ID | Purpose | Dataset / shots | Backbones | Arms | Heads | Seeds | Runs |
|---|---|---|---|---|---|---|---|
| **E1** | Decompose "architecture" into **sites × module** at matched budget; replicate RQ3 with 5 seeds and a varied training stream | MiniImageNet 5-shot | R18, MNv3 | 4 arms × 2 levels | both | 42–46, stream varies | 170 (incl. MNv3 lora_ss r16 as level U) |
| **E2** | Same as E1 on two **new backbones**: is the calibration flip a backbone-*family* property? | MiniImageNet 5-shot | R34, MNv2 | 4 arms × 2 levels (+2 unmatched arms on MNv2) | both | 42–46, stream varies | 180 |
| **E3a** | Replicate the core grid on a dataset with **no ImageNet class overlap** (RQ1/RQ2/RQ4) | FGVC-Aircraft 1+5-shot | R18, MNv3 | btl_ms r16, lora_ss r16 | both | 42–44, fixed stream (grid recipe) | 48 |
| **E3b** | Matched-budget test on Aircraft | FGVC-Aircraft 5-shot | R18, MNv3 | 4 arms × 2 levels | both | 42–44, stream varies | 94 (incl. MNv3 lora_ss r16 level U; minus seed-42 twins of E3a) |
| **E4** | Re-run the original 120-run grid. Checkpoints are lost; this adds MaxLogit + Mahalanobis, per-episode logits and released adapters, and every run is a reproduction check | CIFAR-FS + MiniImageNet, 1+5-shot | R18, MNv3 (+ baselines) | original | both | 42–44, fixed | 112 (the 8 MiniImageNet 5-shot seed-42 cells identical to E1 runs are trained once, in E1, and keep their grid regression reference) |
| **E5-val** | VAL-only sweep to pick an alternative EDL recipe | CIFAR-FS 5-shot | R18 | btl_ms r16 | evidential | 42 | 6 |
| **E5** | Re-run the evidential half of the 2⁵ grid with the selected recipe: is "head explains 83% of ECE" one recipe's artefact? Skipped if the selection equals the current recipe | CIFAR-FS + Mini, 1+5-shot | R18, MNv3 | btl_ms r16, lora_ss r16 | evidential | 42–44, fixed | 48 |

**Implementation note (2026-10-22, Task 7).**
- **Duplicates trained once.** Runs that would train an identical model exist only once, matched by `tmlr_design.training_identity`:
  - 8 E4 grid cells are E1 seed-42 runs;
  - 6 E3b runs are E3a seed-42 runs.
- **Unmatched default pair added for every backbone.** It is run under the same seeds and stream as the matched levels, so rule R-B never mixes training regimes. This added MNv3 `lora_ss` r16 as level U in E1 (+10 runs) and E3b (+4).
- **Analysis matches runs by attributes, never by experiment.**

## A3. Adapter arms

`ms` = multi-site (one insertion per stage), `ss` = single-site (the current LoRA site).

| Arm | Module | Sites | Status |
|---|---|---|---|
| `btl_ms` | parallel 1×1 bottleneck (down → ReLU → up, biases) around the last block of each stage | every stage (`resolve_stage_paths`) | existing (`placement: parallel`) |
| `lora_ss` | LoRA on one conv | `default_lora_targets(backbone)` (R18/R34: `layer4.0.downsample.0`, shortcut path; MNv3: `features.11.block.3.0`; MNv2: `features.17.conv.2`) | existing |
| `lora_ms` | LoRA (`LoRAConv2d`, A keeps the base kernel) with per-site ranks | ResNet: `<stage>.conv2` (3×3). Inverted residual: the stage block's last dense 1×1 (projection) conv | **new** |
| `btl_ss` | parallel 1×1 bottleneck wrapped around one conv | the `lora_ss` site | **new** |

The 2×2 {sites} × {module} design answers "is the bottleneck's win due to more insertion sites or due to the module?". `btl_ss` vs `lora_ss` differ only by the ReLU and the two biases (Task 3 proves the linear equivalence). `lora_ms` also tests whether the RQ3 calibration flip is an artefact of LoRA sitting on the *shortcut* in ResNets but on the *main path* in MobileNets.

**Matched budgets.** These are trainable parameters with the softmax head; the evidential head adds 2. They were verified by instantiating the torchvision modules on 2026-10-22, and Task 7 re-verifies them. Tolerance: the spread within a level is ≤ 4.0%, against 55–158% unmatched.

| Backbone | Level | btl_ms | lora_ss | lora_ms (per-site ranks) | btl_ss | spread |
|---|---|---|---|---|---|---|
| R18 | L | r6 → 12,504 | r16 → 12,288 | [1,1,2,1] → 12,160 | r15 → 12,047 | 3.79% |
| R18 | H | r16 → 31,744 | r41 → 31,488 | [3,3,4,3] → 31,360 | r40 → 31,272 | 1.51% |
| R34 | L | r6 → 12,504 | r16 → 12,288 | [1,1,2,1] → 12,160 | r15 → 12,047 | 3.79% |
| R34 | H | r16 → 31,744 | r41 → 31,488 | [3,3,4,3] → 31,360 | r40 → 31,272 | 1.51% |
| MNv3 | L | r16 → 6,928 | r10 → 6,720 | [6,6,6,5] → 6,864 | r10 → 6,826 | 3.10% |
| MNv3 | H | r22 → 9,448 | r14 → 9,408 | [7,7,7,8] → 9,464 | r14 → 9,518 | 1.17% |
| MNv2 | L | r15 → 11,731 | r9 → 11,520 | [4,4,4,4,5] → 11,648 | r9 → 11,849 | 2.86% |
| MNv2 | H | r22 → 17,030 | r13 → 16,640 | [7,6,6,6,7] → 17,080 | r13 → 16,973 | 2.64% |
| MNv2 | unmatched pair | r16 → 12,488 | r16 → 20,480 | — | — | (the default pair, for the collapse ratio) |

Shape facts behind the table:
- Stage widths: R18/R34 [64,128,256,512]; MNv3 [24,40,48,96]; MNv2 [24,32,64,96,160].
- Bottleneck rank cap: R18/R34 ≤ 63; MNv3/MNv2 ≤ 23.
- `lora_ms` cost per unit of rank per site: R18/R34 [640,1280,2560,5120]; MNv3 [112,280,192,672]; MNv2 [168,224,448,672,1120].
- `btl_ss` = r·(Cin+Cout) + r + Cout: R18/R34 Cin→Cout 256→512 (stride 2); MNv3 576→96; MNv2 960→320.
- R18 `lora_ms` uses rank 1 at three sites. That is the price of matching 12.5K on 3×3 convs; state it in the paper.
- R18/MNv3 existing arms keep their exact old ranks, so seed 42 reproduces `results/rq3_matched/`.

## A4. New backbones

- **ResNet-34** (torchvision `IMAGENET1K_V1`, 512-d): same block family as R18, deeper.
- **MobileNetV2** (torchvision `IMAGENET1K_V1`, 1280-d pooled): same inverted-residual family as MNv3, without SE or h-swish.
- Together they give a 2-families × 2-members design. Existing `resolve_stage_paths` and `default_lora_targets` already work on both (verified).
- Paper summaries for He et al. 2016 and Sandler et al. 2018 do not exist yet; Task 0.2 adds them.

## A5. New dataset — decision D1 (default: FGVC-Aircraft)

- **FGVC-Aircraft** at variant level: 100 classes × ~100 images. ImageNet-1k has only coarse aircraft classes (airliner, warplane, airship, space shuttle), not variants.
- Protocol: crop each image to its bounding box, resize to 84×84 (same source resolution as MiniImageNet; then the usual resize to 224 + ImageNet normalisation), and use a new frozen **64/16/20** class split (seed 2026), so the protocol shape matches CIFAR-FS/MiniImageNet.
- The bounding-box crop and 84 px follow Meta-Dataset's practice. **Verify this against the Meta-Dataset paper or code** and record it in the paper summary.
- OOD pools: `svhn_far`, `gaussian_far`, `aircraft_near` (500 images from the 16 VAL classes), and `tin` (TinyImageNet with every aircraft-related wnid removed, and the removed count logged).
- Alternative: CUB-200 (in the proposal, standard few-shot split). Rejected as default because some of its species are ImageNet-1k classes, which is the very objection this experiment exists to answer.

## A6. Seeds and training stream

- New knob `trainer.vary_train_stream_with_seed` (default `false`). When `true`, the effective train offset is `train_seed_offset + 100_000 × (seed − 42)`.
- Seed 42 is therefore unchanged (20000–22999). Seeds 43–46 use 120000+, 220000+ and so on, which is disjoint from TEST (0–599) and VAL (10000–10099).
- E1/E2/E3b use `true` with seeds 42–46 (E3b 42–44). E3a/E4/E5 use `false`, the original grid recipe.
- Comparing the old fixed-stream 3-seed results with the new varied-stream results is itself a reported robustness check.

## A7. New OOD scores

- **MaxLogit** = max raw prototype logit. For a cosine head this is τ × the nearest-prototype cosine. No summary exists (Hendrycks et al. 2022, "Scaling OOD Detection to Real-World Settings"); Task 0.2 adds it.
- **Mahalanobis** (Lee et al. 2018, summary exists). Deviations, all flagged:
  - Class means are the episode prototypes, computed from L2-normalised adapted support embeddings.
  - The tied covariance is estimated from adapted embeddings of 50 images per TRAIN class (3,200 images) with Ledoit–Wolf shrinkage.
  - No input preprocessing and no layer ensemble.
  - Score = −min_c (q−μ_c)ᵀ P (q−μ_c) on the L2-normalised query embedding.
- kNN-to-support is not added: at 1-shot it equals MaxLogit/τ.

## A8. Alternative EDL recipe (E5)

- E5-val sweeps `loss.kl_weight_max ∈ {0.0, 0.05, 0.1}` × `loss.prior_per_class ∈ {1.0, 0.5}` on R18 + btl_ms r16 + CIFAR-FS 5-shot + evidential + seed 42.
- Selection: lowest VAL pooled ECE (evidential native); ties (|Δ| < 0.002) go to higher VAL accuracy. TEST is never evaluated for E5-val.
- If the winner is the current recipe (0.1, 1.0), E5 is skipped and reported as "the current recipe is VAL-optimal within this sweep".

## A9. Pre-specified hypotheses and rules (full text goes into `docs/TMLR_PRESPEC.md`, Task 11)

Notation:
- A *cell* is a (dataset, backbone, level, head). σ is the pooled across-seed SD of the arms involved (sample SD, ddof=1).
- An effect *fires* if |effect| > 2σ. Arm differences are ECE(LoRA-type) − ECE(bottleneck-type).
- Outcomes:
  - Accuracy.
  - ECE (pooled, native probabilities).
  - Near-OOD AUROC under the native score: Mini/Aircraft near pools = held-out + TIN, averaged.

The rules:

- **R-A (sites vs. module; accuracy and near-OOD).**
  - Per cell: S = ½[(btl_ms−btl_ss)+(lora_ms−lora_ss)] and M = ½[(btl_ms−lora_ms)+(btl_ss−lora_ss)].
  - Verdict over the 16 Mini cells (E1+E2), separately for Aircraft's 8 cells:
    - `site` if S fires with a positive sign in ≥ 75% of cells and M fires in ≤ 25%.
    - `module` if the reverse.
    - `both` if both fire in ≥ 75%.
    - `inconclusive` otherwise.
- **R-B (budget / architecture / backbone; ECE; original pair btl_ms vs lora_ss).** The thesis's three rules, unchanged in definition, applied per (backbone × head) cell with levels averaged.
  - Over R18+MNv3 with 5 seeds: threshold 3 of 4.
  - Over all four backbones: threshold 6 of 8.
- **R-C (does LoRA's site role explain the flip?).** ΔECE_ms = ECE(lora_ms) − ECE(btl_ms) per (backbone × head), levels averaged.
  - `site-role explains flip` if ΔECE_ms has one sign across both families with |Δ| > 2σ in ≥ 6 of 8, **or** |Δ| ≤ 2σ in ≥ 6 of 8.
  - `flip persists` if ΔECE_ms keeps the family pattern (ResNets +, MobileNets −) with |Δ| > 2σ in ≥ 6 of 8.
  - Otherwise `inconclusive`.
- **R-D (family).**
  - `family-consistent` if, for the original pair, R34's ΔECE sign equals R18's and MNv2's equals MNv3's, each with |Δ| > 2σ, in 4 of 4 (backbone × head) checks.
  - `not family-consistent` if ≤ 2 of 4.
  - Otherwise `partial`.
- **R-E (Aircraft replication, E3a).** Replicated if all four hold; each is reported separately anyway:
  - Head is the largest ECE factor.
  - The score's η² exceeds the objective's η² and the objective's η² is < 2% on both OOD types.
  - The refit improves ECE in ≥ 90% of evidential models.
  - btl_ms beats lora_ss on accuracy in ≥ 75% of pairs.
- **R-F (EDL recipe).** `head-dominance robust` if, with E5's recipe substituted for the evidential arm in the 2⁵ design, the head is still the largest ECE factor and evidential ECE > softmax ECE in ≥ 75% of the 20 matched pairs.
- **Exclusions.** A run is excluded only if it errors or fails the parameter check. Such runs are re-run, never dropped. There is no outcome-based exclusion.

## A10. Compute and schedule

**Per-run estimates** (packing defaults, replaced by Wave 0's measured medians):

| Run type | Minutes |
|---|---|
| Mini/Aircraft 5-shot: R18 | 9 |
| Mini/Aircraft 5-shot: MNv3 | 7 |
| Mini/Aircraft 5-shot: R34 | 13 |
| Mini/Aircraft 5-shot: MNv2 | 8 |

- 1-shot runs: ×0.85. CIFAR-FS: ×1.4. `full_ft`: ×3.
- Measured history: the matched-budget session averaged 476 s per run; the grid averaged 1,054 s per run.

**Totals.** About 115–170 lane-hours. That is about 15–21 session-hours per account *if* a T4 x2 session's two lanes count once against the quota and give ≥ 1.7× throughput. Wave 0 measures both. If they don't, it's about 29–43 h per account, and E4 is the first thing cut.

**Lanes.** 4 accounts × 1 session at a time × 2 lanes (T4 x2) = 8 parallel runs. Shards are packed to ≤ 540 estimated minutes per lane (the Kaggle cap is 12 h).

| Wave | Content | Shards | Wall time |
|---|---|---|---|
| W0 | 8 validation runs (Task 13) | 1 shard, 1 account | ~1.5–2 h |
| W1 | E1 + E2 + E5-val (≈340 runs) | ≈5–8 shards, round-robin over 4 accounts | ≈8–10 h |
| W2 | E3a + E3b + E5 + E4 (≈306 runs) | ≈8 shards, 2 per account back-to-back | ≈2 × 5–6 h |
| WR | retries of anything missing or errored | as needed | ≤ 3 h |

Calendar target: about 5 working days of code (Phase 1), then about 3–4 days of compute, analysis overlapping, then 2–3 weeks of writing.

## A11. Decisions the human owns

| ID | Decision | Default |
|---|---|---|
| D1 | Dataset | **DECIDED 2026-10-22: FGVC-Aircraft** |
| D2 | Code licence | MIT |
| D3 | Author list, order, affiliations; supervisor approval to submit. The new work is done by the user alone, but the paper is built on the joint thesis (code, grid runs, analyses; see `docs/MY_CONTRIBUTIONS.md` §9). The former teammates' authorship or written consent must therefore be settled with the supervisor before submission. | settle before Task 17 |
| D4 | Central result store: a free HF account with a private dataset repo `bpeft-tmlr-runs` and a write token added to each Kaggle account's Secrets as `HF_TOKEN` | **DECIDED 2026-10-22: yes.** The AI gives step-by-step setup instructions right before Wave 0 |
| D5 | Exclude GPU-relay code from the public release branch | yes |
| D6 | Also register `docs/TMLR_PRESPEC.md` on OSF | optional, recommended |

---

# PART B — Tasks

File map (new unless marked *mod*):

```
src/backbones/resnet34.py, src/backbones/mobilenetv2.py, src/backbones/__init__.py (mod)
src/adapters/conv_parallel.py, src/adapters/lora.py (mod), src/adapters/__init__.py (mod)
src/utils/seed.py (mod), src/utils/checkpoint_io.py, scripts/train.py (mod), configs/base.yaml (mod: one key)
src/datasets/fgvc_aircraft.py, src/datasets/__init__.py (mod), src/datasets/tinyimagenet_ood.py (mod)
scripts/tmlr_build_aircraft_cache.py
scripts/rq_core.py (mod), scripts/rq_drivers.py (mod), scripts/rq_aggregate.py (mod: B1 fix)
scripts/tmlr_design.py, scripts/tmlr_configs.py, scripts/tmlr_shards.py, scripts/tmlr_run_shard.py
scripts/tmlr_collect.py, scripts/tmlr_pull.sh, scripts/tmlr_adjudicate.py, scripts/tmlr_analysis.py, scripts/tmlr_paper_assets.py
configs/tmlr/parents/*.yaml, configs/tmlr/<exp>/*.yaml (generated), configs/tmlr/manifest.json, configs/tmlr/shards_w*.json
notebooks/tmlr_runner.ipynb
docs/TMLR_PRESPEC.md, docs/KAGGLE_RUNBOOK.md
paper/ (TMLR LaTeX), tests/test_tmlr_*.py and the per-task test files below
```

Run tests with `uv run --frozen python -m pytest -q --ignore-glob='tests/test_vgpu_*'`. Baseline on 2026-10-22: **262 passed, 2 skipped**. That count may only grow.

## Phase 0 — Preparation

### Task 0.1: Human setup (no code)

- [ ] Decide D1–D6 and write the answers at the top of `docs/KAGGLE_RUNBOOK.md` (Task 9 creates the file; until then, write them in this plan's A11 table).
- [ ] On all 4 Kaggle accounts:
  - Complete phone verification (needed for GPU and Internet).
  - Note the username.
  - Note the weekly GPU quota shown in Settings.
- [ ] Data access. From 2026-10-22 the user works alone; all 4 Kaggle accounts are the user's own.
  - If the existing Kaggle dataset `beft-thesis-data` (CIFAR-100, SVHN `test_32x32.mat`, `tiny-imagenet-200`, MiniImageNet pickles) is on one of the user's accounts, add the other three accounts as collaborators: Dataset → Settings → Sharing.
  - If it belongs to a former teammate's account, re-create it as a private dataset `bpeft-data` on the user's main account with the same four sources, then share it with the other three accounts. The data finders search `/kaggle/input` at any depth, so the dataset name does not matter.
- [ ] If D4 = yes:
  - Create an HF account, a **private** dataset repo `<hf-user>/bpeft-tmlr-runs`, and a write token.
  - On each Kaggle account, add the token under Add-ons → Secrets with the label `HF_TOKEN`.
- [ ] Download the FGVC-Aircraft archive (`fgvc-aircraft-2013b.tar.gz`, about 2.7 GB, from the Oxford VGG FGVC-Aircraft page) to a local machine. Task 5 consumes it.

### Task 0.2: Paper summaries for components that have none

**Files:** Modify `PAPER SUMMARIES/CNN_paper_summaries.txt`, `PAPER SUMMARIES/ODD_paper_summaries.txt`, `PAPER SUMMARIES/FEW SHOTpaper_summaries.txt`.

- [ ] Add entries in the existing 8-section format (`==== PAPER: … ====`, sections 1–8), written from the PDFs (the human supplies them, or fetch the arXiv versions):
  - He et al. 2016 *Deep Residual Learning* (ResNet-34 layout).
  - Sandler et al. 2018 *MobileNetV2*.
  - Hendrycks et al. 2022 *Scaling Out-of-Distribution Detection for Real-World Settings* (MaxLogit).
  - Maji et al. 2013 *Fine-Grained Visual Classification of Aircraft*.
  - Triantafillou et al. 2020 *Meta-Dataset* (bounding-box crop and 84 px practice for Aircraft).
- [ ] In each entry's section 8, state the fit and deviations from Part A (A4, A5, A7). Where the PDF does not state something this plan assumes (e.g. the Meta-Dataset crop), write that explicitly.
- [ ] 🔒 HUMAN GATE.

## Phase 1 — Code (TDD, CPU-only tests, `pretrained=False` backbones)

### Task 1: ResNet-34 and MobileNetV2 backbones

**Files:**
- Create: `src/backbones/resnet34.py`, `src/backbones/mobilenetv2.py`
- Modify: `src/backbones/__init__.py`
- Test: `tests/test_backbones_tmlr.py`

**Interfaces:**
- Produces:
  - `build_frozen_resnet34(pretrained: bool = True) -> nn.Module` (fc = Identity, all params frozen, eval mode, `(B,512)`).
  - `build_frozen_mobilenetv2(pretrained: bool = True) -> nn.Module` (classifier = Identity, `(B,1280)`).
  - `MOBILENETV2_FEATURE_DIM = 1280`.
  - Canonical names `"resnet34"` and `"mobilenetv2"`; aliases `resnet_34`, `mobilenet_v2`, `mbnet2`.
  - Entries in `BACKBONE_FEATURE_DIMS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_resnet34_contract():
    bb = build_backbone("resnet34", pretrained=False)
    assert bb(torch.zeros(2, 3, 224, 224)).shape == (2, 512)
    assert not any(p.requires_grad for p in bb.parameters()) and not bb.training

def test_mobilenetv2_contract():
    assert build_backbone("mbnet2", pretrained=False)(torch.zeros(2, 3, 224, 224)).shape == (2, 1280)
    assert backbone_feature_dim("mobilenet_v2") == 1280

def test_stage_paths_new_backbones():
    assert resolve_stage_paths(build_backbone("resnet34", pretrained=False)) == ["layer1.2", "layer2.3", "layer3.5", "layer4.2"]
    assert resolve_stage_paths(build_backbone("mobilenetv2", pretrained=False)) == [
        "features.3", "features.6", "features.10", "features.13", "features.16"]

def test_default_lora_targets_new_backbones():
    assert default_lora_targets(build_backbone("resnet34", pretrained=False)) == ["layer4.0.downsample.0"]
    assert default_lora_targets(build_backbone("mobilenetv2", pretrained=False)) == ["features.17.conv.2"]
```

- [ ] **Step 2:** Run `pytest tests/test_backbones_tmlr.py -v`. Expected: FAIL (`Unknown backbone`).
- [ ] **Step 3:** Implement, mirroring `resnet18.py` and `mobilenetv3.py`. Add one docstring line in `placement.py` noting that `_is_mobilenetv3` / `mobilenetv3_stage_paths` also serve MobileNetV2 because they key on `use_res_connect`. Do **not** rename them (Step 8 reproducibility).
- [ ] **Step 4:** Run the task tests, then the full suite. Expected: all pass, count ≥ 262 + 4.
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 2: Multi-site LoRA with per-site ranks

**Files:**
- Modify: `src/adapters/lora.py`, `src/adapters/__init__.py`
- Test: `tests/test_lora_multisite.py`

**Interfaces:**
- Consumes: `resolve_stage_paths`.
- Produces:
  - `multisite_lora_targets(backbone: nn.Module) -> List[str]`. For each stage path: ResNet → `f"{path}.conv2"`; inverted residual → the block's last dense 1×1 conv (same traversal rule as `_mobilenetv3_default_lora_target`).
  - `LoRAAdapter(backbone, rank, alpha=None, targets=None, dropout=0.0, ranks: Optional[Sequence[int]] = None)`. With `ranks`, target *i* gets rank `ranks[i]`, and alpha defaults to that rank (scaling 1). `self.ranks: List[int]` is exposed.
  - `build_adapter` accepts `lora_targets: "multisite"` and `lora_ranks: [..]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_multisite_targets():
    assert multisite_lora_targets(r18()) == ["layer1.1.conv2", "layer2.1.conv2", "layer3.1.conv2", "layer4.1.conv2"]
    assert multisite_lora_targets(mnv3()) == ["features.3.block.2.0", "features.6.block.3.0",
                                             "features.8.block.3.0", "features.11.block.3.0"]
    assert multisite_lora_targets(mnv2()) == [f"features.{i}.conv.2" for i in (3, 6, 10, 13, 16)]

@pytest.mark.parametrize("bb,ranks,expected", [
    ("resnet18", [1, 1, 2, 1], 12160), ("resnet18", [3, 3, 4, 3], 31360),
    ("mobilenetv3_small", [6, 6, 6, 5], 6864), ("mobilenetv3_small", [7, 7, 7, 8], 9464),
    ("mobilenetv2", [4, 4, 4, 4, 5], 11648), ("mobilenetv2", [7, 6, 6, 6, 7], 17080)])
def test_per_target_ranks_param_count(bb, ranks, expected): ...   # count_trainable_params == expected

def test_identity_at_init(): ...        # adapted backbone output == frozen output, atol=0
def test_rank_length_mismatch_raises(): ...   # ValueError
def test_single_target_unchanged(): ...       # {type: lora, rank: 16, lora_targets: [layer4.0.downsample.0]} -> 12288
```

- [ ] **Step 2:** Run. Expected: FAIL (ImportError on `multisite_lora_targets`).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite (the existing `tests/test_lora.py` must stay green).
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 3: Single-site parallel bottleneck (`btl_ss`)

**Files:**
- Create: `src/adapters/conv_parallel.py`
- Modify: `src/adapters/__init__.py`
- Test: `tests/test_conv_parallel.py`

**Interfaces:**
- Produces:
  - `ConvParallelBottleneck(base_conv: nn.Conv2d, rank: int)`:
    - forward: `base(x) + up(relu(down(x)))`.
    - `down = Conv2d(Cin, r, 1, stride=base.stride, bias=True)`; `up = Conv2d(r, Cout, 1, bias=True)`, zero-initialised (weight and bias); base frozen.
    - Raises `ValueError` unless the base kernel is 1×1 with padding 0.
  - `SingleSiteBottleneckAdapter(backbone, rank, target: Optional[str] = None)`. `backbone_trainable = True`; target defaults to `default_lora_targets(backbone)[0]`; identity forward; exposes `.target`.
  - `build_adapter`: `{type: bottleneck, placement: conv_parallel, rank: r, conv_target: optional}`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("bb,r,expected", [("resnet18", 15, 12047), ("resnet18", 40, 31272),
    ("mobilenetv3_small", 10, 6826), ("mobilenetv3_small", 14, 9518),
    ("mobilenetv2", 9, 11849), ("mobilenetv2", 13, 16973)])
def test_param_counts(bb, r, expected): ...

def test_identity_at_init(): ...
def test_same_site_as_lora(): ...      # adapter.target == default_lora_targets(backbone)[0]
def test_stride_preserved(): ...       # R18 site: wrapper output shape == base output shape (stride 2)
def test_rejects_kxk_base(): ...       # 3x3 base conv -> ValueError

def test_linear_equivalence_with_lora():
    # With act=Identity and both biases zero, and down/up weights copied into a
    # LoRAConv2d(base, rank=r, alpha=r) as lora_A/lora_B, outputs match (atol 1e-6).
    # Documents: btl_ss differs from lora_ss ONLY by the ReLU and the two biases.
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Reuse `_replace_submodule` from `lora.py`; do not copy it.
- [ ] **Step 4:** Run the task tests and the full suite.
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 4: Seed-dependent training stream

**Files:**
- Modify: `src/utils/seed.py`, `scripts/train.py:270` (`train_seed_offset = …`), `configs/base.yaml` (add `trainer.vary_train_stream_with_seed: false` with a comment)
- Test: `tests/test_train_stream.py`

**Interfaces:**
- Produces: `effective_train_seed_offset(cfg) -> int`, using the rule in A6.
  - Raises `ValueError` if the flag is on and `seed < 42`.
  - Asserts the full stream `[offset, offset + num_epochs × episodes_per_epoch)` does not intersect [0, 600) or [10000, 10100).
- `scripts/train.py` calls it instead of `int(cfg.trainer.train_seed_offset)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_unchanged(): ...            # flag absent: seeds 42 and 43 -> 20000
def test_vary(): ...                         # flag on: 42 -> 20000, 43 -> 120000, 46 -> 420000
def test_rejects_seed_below_42(): ...
def test_old_configs_byte_identical_offset():
    for p in glob("configs/grid/*.yaml") + glob("configs/rq3_matched/*.yaml"):
        assert effective_train_seed_offset(load_config(p)) == 20000
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite (`tests/test_episodic_trainer.py` must stay green).
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 5: FGVC-Aircraft dataset

**Files:**
- Create: `scripts/tmlr_build_aircraft_cache.py`, `src/datasets/fgvc_aircraft.py`, `configs/tmlr/parents/aircraft_{r18,mnv3}_{softmax,evidential}.yaml`
- Modify: `src/datasets/__init__.py`, `src/datasets/tinyimagenet_ood.py`
- Test: `tests/test_fgvc_aircraft.py`

**Interfaces:**
- `scripts/tmlr_build_aircraft_cache.py --raw <fgvc-aircraft-2013b dir> --out <dir> [--make-split --seed 2026]`:
  - Reads all 10,000 images (official image splits merged; few-shot uses class splits).
  - Crops each image to `images_box.txt`, resizes to 84×84 with `PIL.Image.BILINEAR`, and writes `aircraft_84.npz` with `images uint8 (N,84,84,3)`, `labels int16`, `variants str[100]`.
  - Writes `aircraft_84.meta.json` (sha256, crop, resize method, source archive sha256).
  - `--make-split` prints a sorted, `RandomState(2026)`-permuted 64/16/20 split as a Python literal.
- `src/datasets/fgvc_aircraft.py`:
  - `AIRCRAFT_SPLIT: Dict[str, List[str]]`: the literal from `--make-split`, pasted once and marked `# DO NOT REGENERATE — frozen`.
  - `AIRCRAFT_SPLIT_SHA256: str`.
  - `get_fgvc_aircraft(data_root="data", image_size=224, split="test")` returns the **same type** `get_mini_imagenet` returns, with the same transform (reuse `mini_imagenet._build_transform`).
  - `get_aircraft_heldout_ood(data_root, image_size, num_samples, seed, heldout_split="val") -> Tensor`.
  - `_find_aircraft_cache(data_root) -> Optional[Path]`: searches `data_root`, then `/kaggle/input`, depth-bounded like `_kaggle_candidate_dirs`.
- `src/datasets/__init__.py`:
  - Dispatch `name == "fgvc_aircraft"` in `build_dataset`, `get_id_split`, `get_heldout_near_ood` (pool `"aircraft_near"`).
  - `tin_exclusions(dataset_name: str) -> Tuple[str, ...]`: Mini returns `MINI_IMAGENET_ALL_WNIDS`. Aircraft returns `AIRCRAFT_TIN_EXCLUDE_WNIDS`, fixed once by grepping TinyImageNet's `words.txt` for airliner / warplane / airship / space shuttle / aircraft / plane and reviewed at the gate.
- `get_tinyimagenet_ood` logs and returns the removed-class count (closes problem B4 for Mini too).
- Parents extend the Mini Step-10 parents and change only `dataset.name: fgvc_aircraft`.

- [ ] **Step 1: Write the failing tests.** Use a synthetic npz fixture in `tmp_path`: 100 classes × 3 images, 84×84. Never use real data.

```python
def test_split_frozen():                     # 64/16/20, disjoint, 100 unique
def test_split_hash():                       # sha256(json.dumps(AIRCRAFT_SPLIT, sort_keys=True)) == AIRCRAFT_SPLIT_SHA256
def test_loader_on_fixture():                # split="val": 48 items, labels in 0..15, x.shape == (3,224,224)
def test_episodes_on_fixture():              # EpisodicIterableDataset(5-way, 1-shot, 2-query) yields valid episodes
def test_dispatch_near_pool():               # get_heldout_near_ood({"name": "fgvc_aircraft", ...}) -> ("aircraft_near", tensor)
def test_tin_exclusion_counts(tmp_path):     # fake TIN dir with n02690373 -> removed count == 1
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the loader and dispatch (TDD on the fixture).
- [ ] **Step 4:** Run the cache builder on the real archive locally: `uv run --frozen python scripts/tmlr_build_aircraft_cache.py --raw <dir> --out <dir> --make-split --seed 2026`. Paste the split into `AIRCRAFT_SPLIT` and set `AIRCRAFT_SPLIT_SHA256`. Expected: N == 10000, 100 variants, every class ≥ 66 images.
- [ ] **Step 5:** Run the task tests and the full suite.
- [ ] **Step 6:** 🔒 HUMAN GATE. Then the human:
  - uploads `aircraft_84.npz` and `aircraft_84.meta.json` as a new version of the shared Kaggle dataset (or a new dataset `bpeft-aircraft84` shared the same way);
  - commits the split constant.

### Task 6: New OOD scores, calibration sign, environment record, adapter-only checkpoints

**Files:**
- Create: `src/utils/checkpoint_io.py`
- Modify: `scripts/rq_core.py`, `scripts/rq_drivers.py`
- Test: `tests/test_tmlr_scores.py`, `tests/test_checkpoint_io.py`

**Interfaces:**
- `rq_core.FACTORIAL_SCORES` gains `"maxlogit"` and `"mahalanobis"`. Existing entries and their order are unchanged.
- `all_id_scores` adds `scores["maxlogit"] = logits.max(-1).values`.
- `mahalanobis_score(q_emb: Tensor, protos: Tensor, precision: Tensor) -> Tensor`: inputs are L2-normalised; returns −min_c (q−μ_c)ᵀP(q−μ_c), so higher means more in-distribution.
- `fit_base_precision(model, cfg, device, *, n_per_class: int = 50, seed: int = 31337) -> Tensor`:
  - Adapted embeddings come from `model.adapter_features_from_features(model.backbone(x))`, L2-normalised.
  - Uses TRAIN classes only, with a **private** `numpy.random.RandomState(seed)` (the global RNG is never touched).
  - Class-centred pooled covariance → `sklearn.covariance.LedoitWolf` → precision.
- `factorial_evaluate(..., precision: Tensor | None = None)` computes Mahalanobis for ID and every OOD pool when given. It also adds `signed_calibration_gap__{probset}` = mean max-probability − accuracy over pooled predictions.
- `factorial_run_one(...)`:
  - Fits the precision.
  - Adds `record["env"] = {torch, torchvision, cuda, gpu_name, git_sha (env GIT_SHA), torch_threads}`.
  - Saves `trainable_state_dict(model)` to a path passed by the caller (`adapter_out: Path | None`).
- `rq_drivers.val_only_run_one(repo_root, config_path, checkpoint_path, out_json, *, device, log=print) -> dict`: for E5-val.
  - Uses only `R.load_val_logits` (seeds 10000–10099) and `R.all_prob_sets`.
  - Writes `{val_ece_pooled__evidential_native, val_accuracy__evidential_native, val_nll__evidential_native, env, n_params}`.
  - Never calls `_load_test_seeds`.
- `checkpoint_io.trainable_state_dict(model) -> Dict[str, Tensor]`: every `requires_grad` parameter plus the head's evidence scale/bias tensors.
- `checkpoint_io.load_trainable_state(model, state) -> None`: raises on unknown keys.

- [ ] **Step 1: Write the failing tests**

```python
def test_maxlogit_is_max_logit(): ...
def test_mahalanobis_identity_precision():       # P = I: score == -min squared distance to a prototype
def test_mahalanobis_prefers_near_prototype(): ...
def test_fit_base_precision_leaves_global_rng():  # torch.get_rng_state() and np.random.get_state() unchanged
def test_existing_keys_unchanged():
    # tiny CPU model (resnet18 pretrained=False, synthetic split, 3 episodes):
    # every key of summary(precision=None) has an identical value in summary(precision=P)
def test_signed_gap_sign(): ...                   # overconfident synthetic probabilities -> positive
def test_val_only_never_touches_test(): ...       # monkeypatch _load_test_seeds to raise; val_only_run_one succeeds
def test_roundtrip_reproduces_logits(): ...       # perturb lora_ms params, save trainable state, fresh model, load -> identical outputs
def test_trainable_state_size(): ...              # element count == count_trainable_params(model) + evidence tensors
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests, then the full suite. `tests/test_ood_scores.py` must stay green; the T1.6 regression guard compares only old keys.
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 7: Experiment design, config generation, manifest

**Files:**
- Create: `scripts/tmlr_design.py`, `scripts/tmlr_configs.py`, `configs/tmlr/parents/mini_{r34,mnv2}_{softmax,evidential}.yaml` (extend the Mini R18 / MNv3 parents; change only `backbone.name` and `backbone.feature_dim`)
- Generated: `configs/tmlr/<exp>/*.yaml`, `configs/tmlr/manifest.json`
- Test: `tests/test_tmlr_design.py`

**Interfaces:**
- `tmlr_design.LEVELS`: the A3 table, keyed `(backbone, level) -> {arm: rank_spec}`, plus `UNMATCHED = {"mobilenetv2": {"btl_ms": 16, "lora_ss": 16}}`.
- `tmlr_design.SHAPES`: hardcoded per-backbone shape tables, independent of the code under test (stage widths, `lora_ss` Cin/Cout/kernel, `lora_ms` per-rank units).
- `expected_params(backbone: str, arm: str, rank_spec: int | list[int], head: str) -> int`: closed forms from `SHAPES` (+2 for evidential).
- `@dataclass(frozen=True) RunSpec`: `run_id, experiment, dataset, k_shot, backbone, arm, level, rank_spec, head, seed, vary_stream, config, parent, expected_n_params, est_minutes, regression_ref, wave, mode` (`"test"` | `"val_only"`).
- `run_id` format: `{exp}_{ds}_{k}s_{bb}_{arm}_{level|grid}_r{ranks joined by -}_{evid|soft}_s{seed}`, e.g. `e1_mini_5s_r18_lora_ms_H_r3-3-4-3_evid_s43`.
- Arm vocabulary: `btl_ms`, `lora_ss`, `lora_ms`, `btl_ss`. E4 additionally uses `full_ft` and `linear_probe` (the original grid's baseline arms). `level` is `"L"`, `"H"`, `"U"` (the MNv2 unmatched pair) or `None` (grid-style runs).
- `WAVE0_RUN_IDS: tuple[str, ...]`: the 9 runs listed in Task 12, which get `wave = 0`. Every other run gets `wave = 1` (E1, E2, E5-val) or `wave = 2` (E3a, E3b, E4, E5).
- `build_design() -> list[RunSpec]` covers E1–E5 per A2.
  - E4 specs wrap `configs/grid/<cell>.yaml` via `extends`, overriding only `output.*` and `wandb.*`. Their `regression_ref` = the committed `results/grid/*_metrics.json`.
  - E1 seed-42 runs of existing arms get `regression_ref` = `results/rq3_matched/<cell>.json`.
  - E5 specs depend on `configs/tmlr/e5_selected.yaml`; while it is absent they are emitted with `wave = -1` (blocked).
- `tmlr_configs.py`:
  - `write_configs(specs) -> None`: minimal-override YAMLs with seed, the full `adapter` block, `trainer.vary_train_stream_with_seed`, `output.run_tag = run_id`, `output.results_dir = results/tmlr/<exp>`, and `wandb.disabled: true`. Plus `manifest.json` (sorted; each entry includes `merged_config_sha256`). Refuses to change an existing config once `configs/tmlr/FROZEN` exists.
  - `assert_budgets(specs) -> dict`: instantiates each unique arm on CPU and enforces the ≤ 4.0% spread.
  - `control_guard(specs) -> dict`: within a comparison group, the merged configs differ only in `adapter.*`, `seed`, `output.*`, `wandb.*`.
  - `select_e5(val_records) -> dict`: applies A8 and writes `e5_selected.yaml`.
  - CLI: `--write | --check | --select-e5 <dir> | --freeze`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_counts():
    d = Counter(s.experiment for s in build_design())
    assert (d["e1"], d["e2"], d["e3a"], d["e3b"], d["e4"], d["e5val"], d["e5"]) == (170, 180, 48, 94, 112, 6, 48)
def test_run_ids_unique(): ...
def test_level_spread_within_tolerance(): ...          # every LEVELS row: max/min - 1 <= 0.040
@pytest.mark.parametrize("arm_key", unique_arm_keys())
def test_expected_params_match_instantiated(arm_key): ...   # closed form == count_trainable_params(build_model(cfg))
def test_e4_wraps_original_grid(): ...                 # merged E4 cfg == merged grid cfg except output/wandb
def test_regression_refs_assigned(): ...               # 114 E4 (grid) + 16 E1 seed-42 existing-arm runs (factorial), each ref file exists
def test_seed42_vary_matches_rq3_matched(): ...        # merged recipe identical; effective offset 20000
def test_frozen_files_untouched(): ...                 # sha256 of test/val episode files unchanged after write_configs
def test_e5_blocked_until_selected(): ...
def test_wave0_runs(): ...                             # 9 ids, all in the manifest, all wave 0; 3 carry a regression_ref; no Aircraft
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Reuse `rq3_matched` helpers (`_differing_keys`, `_flatten`, `_checkpoint_tag`, `_head_descriptor`) by import, not by copy.
- [ ] **Step 4:** Run `uv run --frozen python scripts/tmlr_configs.py --write --check`. Expected: it prints 658 runs (48 blocked), the budget table matching A3 exactly, and `control_guard ok`.
- [ ] **Step 5:** Run the full suite.
- [ ] **Step 6:** 🔒 HUMAN GATE.

### Task 8: Shard packer and lane runner

**Files:**
- Create: `scripts/tmlr_shards.py`, `scripts/tmlr_run_shard.py`
- Test: `tests/test_tmlr_shards.py`, `tests/test_tmlr_run_shard.py`

**Interfaces:**
- `pack_shards(specs, *, wave: int, lanes_per_shard=2, lane_budget_min=540, timings: dict | None = None) -> list[dict]`. Each shard is `{"shard_id": "W1-01", "wave": 1, "lanes": [[run_id, ...], [run_id, ...]], "est_minutes": [m0, m1]}`.
  - Packing unit = one comparison group (all arms of a (dataset, backbone, level, head, seed); one grid cell × seed for E4).
  - Longest-processing-time-first over lanes; deterministic given sorted input.
  - Within a lane, ResNet-18 groups come first, then by (level, head, seed).
- `assign_accounts(shards, accounts) -> dict[str, list[str]]`: balances estimated minutes; prints a schedule table.
- `make_retry_shards(coverage: dict, specs) -> list[dict]`: IDs `R{n}-xx`, covering missing and errored runs only.
- CLI: `tmlr_shards.py --wave N [--timings results/tmlr/_timings.json] [--retry results/tmlr/_coverage.json]` writes `configs/tmlr/shards_w{N}.json`.
- `tmlr_run_shard.py --shard-file F --shard-id ID --lane i --out-dir D --max-minutes M --num-episodes 600 [--no-logits]`. Per run, in-process:
  - Skip if `D/<exp>/<run_id>/metrics.json` exists.
  - Otherwise `rq_drivers.train_cell(config)`, then `factorial_run_one(..., committed_metrics=regression_ref, logits_out=D/<exp>/<run_id>/logits.npz, adapter_out=D/<exp>/<run_id>/adapter.pt)`. For `mode == "val_only"`: VAL metrics only; TEST is never loaded.
  - Re-check `n_params == expected_n_params`; a mismatch is an error.
  - Delete the full checkpoint after the adapter is saved, except for `full_ft`.
  - Append to `D/_lane{i}_runlog.jsonl`.
  - Catch exceptions per run.
  - Stop starting new runs once `max_minutes` has elapsed.
  - Write `D/_lane{i}_summary.json`.
- `shard_report(shard, out_dir) -> dict`: `{expected, done, error, missing, guards: {exact, within_tol, fail}}`.

- [ ] **Step 1: Write the failing tests.** The runner tests monkeypatch `train_cell` and `factorial_run_one` with fakes.

```python
def test_every_run_in_exactly_one_shard(): ...
def test_lane_budget_respected(): ...        # each lane <= 540 est. min unless it holds one oversize group alone
def test_groups_not_split(): ...
def test_deterministic(): ...                # identical JSON on two calls
def test_retry_covers_missing_only(): ...
def test_skips_done_runs(): ...
def test_error_in_one_run_does_not_stop_lane(): ...
def test_budget_stops_new_runs(): ...        # fake monotonic clock
def test_param_drift_marks_error(): ...
def test_val_only_never_loads_test(): ...    # fake _load_test_seeds raises if called
def test_shard_report_lists_missing(): ...
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite.
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 9: Kaggle runner notebook and runbook

**Files:**
- Create: `notebooks/tmlr_runner.ipynb`, `docs/KAGGLE_RUNBOOK.md`

**Notebook cells, in order.** The control cell is the only cell a human edits.

1. **Control:**
   - `SHARD_FILE = "configs/tmlr/shards_w1.json"`, `SHARD_ID = "W1-01"`, `GIT_SHA = "<40-hex>"`.
   - `LANES = 2`, `MAX_MINUTES = 600`, `NUM_EPISODES = 600`, `PERSIST_LOGITS = True`.
   - `UPLOAD_TO_HF = True`, `HF_REPO = "<hf-user>/bpeft-tmlr-runs"`.
2. **Environment:**
   - Assert `torch.cuda.device_count() >= LANES`.
   - Assert every device name contains `"T4"`.
   - Print the torch, torchvision and CUDA versions.
3. **Code:**
   - `git clone` the public repo, `git checkout $GIT_SHA`, and assert `git rev-parse HEAD == GIT_SHA`.
   - `pip install -q -r requirements.txt`.
   - Set `sys.path` exactly as `rq3_matched_budget.ipynb` does.
4. **Data:** symlink staged datasets via the existing finders (`_find_staged_cifar100_root`, `_find_staged_svhn_root`, `_find_extracted_tin_root`, `_find_zenodo_pkls`) plus `_find_aircraft_cache`. Print what was found and what is missing.
5. **Preflight (CPU):** `assert_budgets` + `control_guard` on this shard's runs only. Raise **before** any GPU work on failure.
6. **Restore:** copy any `<exp>/<run_id>/{metrics.json,adapter.pt,logits.npz}` found under `/kaggle/input/**` (an attached earlier output) into `OUT = /kaggle/working/tmlr_out/<SHARD_ID>`.
7. **Run, in a single `try/finally`:**
   - Launch `LANES` subprocesses of `scripts/tmlr_run_shard.py`, each with `CUDA_VISIBLE_DEVICES=i`, `GIT_SHA`, and `OMP_NUM_THREADS=2`. Each lane's stdout and stderr go to `OUT/_lane{i}.log`.
   - Every 120 s, print the last line of each log and the done count.
   - `finally:`
     - write `shard_report.json`;
     - zip `OUT` (with a sha256 `MANIFEST.txt`) to `/kaggle/working/<SHARD_ID>.zip`;
     - upload it to `HF_REPO` at `shards/<SHARD_ID>.zip` if enabled, catching and printing any upload error.
8. **Verdict:** print the report; raise if `missing` or `error` is non-empty, so the Kaggle version shows "Error" while the outputs are already saved.

**Runbook (`docs/KAGGLE_RUNBOOK.md`)** contents:
- The per-account checklist:
  - File → Import Notebook (GitHub URL or upload).
  - Session options: Accelerator **GPU T4 x2**, Internet **On**.
  - Add Input: the shared dataset(s), plus the previous output when resuming.
  - Add-ons → Secrets: `HF_TOKEN`.
  - Edit the control cell.
  - **Save Version → Save & Run All (Commit)**. Close the tab; the run continues in the background.
  - Check Versions → Logs.
  - Download the output, or rely on HF.
- Rules:
  - One GPU session per account at a time.
  - Check the quota before and after Wave 0 to learn how T4 x2 is billed.
  - Never edit code inside Kaggle; change `GIT_SHA` instead.
- The wave schedule table, filled in by `assign_accounts`.

- [ ] **Step 1:** Write the notebook and runbook.
- [ ] **Step 2:** Validate the notebook JSON: `uv run --frozen python -c "import json;json.load(open('notebooks/tmlr_runner.ipynb'))"`. Run cells 1–6 locally on CPU with `LANES = 0`, `NUM_EPISODES = 3`, and a 1-run shard file. Expected: preflight passes and the restore step finds nothing.
- [ ] **Step 3:** 🔒 HUMAN GATE.

### Task 10: Collect, pull, coverage, retries

**Files:**
- Create: `scripts/tmlr_collect.py`, `scripts/tmlr_pull.sh`
- Test: `tests/test_tmlr_collect.py`

**Interfaces:**
- `collect(incoming: Path, dest: Path = Path("results/tmlr"), manifest: Path = Path("configs/tmlr/manifest.json")) -> dict`:
  - Accepts shard zips or folders and verifies each `MANIFEST.txt` sha256.
  - Per run, checks that the record's merged-config sha256 equals the manifest's; if not, the run is marked `stale_code`.
  - An identical duplicate is ignored. A differing duplicate raises `ConflictError` listing both sources.
  - Copies run folders to `dest/<exp>/<run_id>/`.
  - Writes:
    - `dest/_coverage.json`: per experiment `done`/`missing`/`error`/`stale_code`.
    - `dest/_timings.json`: median wall seconds per (dataset, backbone, arm kind, shots).
    - `dest/_regression.json`: guard status per `regression_ref` run.
- `tmlr_pull.sh`:
  - `hf` mode: `huggingface-cli download <repo> --repo-type dataset --local-dir incoming/`.
  - `kaggle` mode: loop over `~/.kaggle/acct{1..4}/kaggle.json` with `KAGGLE_CONFIG_DIR` and run `kaggle kernels output <user>/<slug> -p incoming/<user>`. Download each account's output before its next session overwrites "latest".

- [ ] **Step 1: Write the failing tests**

```python
def test_identical_duplicate_ok(): ...
def test_conflicting_duplicate_raises(): ...
def test_stale_config_flagged(): ...
def test_bad_zip_hash_rejected(): ...
def test_coverage_counts(): ...
def test_timings_written(): ...
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite.
- [ ] **Step 5:** 🔒 HUMAN GATE.

### Task 11: Pre-specification and adjudication code (must be pushed before Wave 1)

**Files:**
- Create: `docs/TMLR_PRESPEC.md`, `scripts/tmlr_adjudicate.py`
- Test: `tests/test_tmlr_adjudicate.py`

**Interfaces:**
- `cell_stats(records, outcome: str) -> dict[tuple, dict]`: (dataset, backbone, level, head, arm) → `{mean, sd, n}`.
- `sites_modules(cell: dict) -> dict`: returns S, M, interaction, σ, `site_fires`, `module_fires`.
- `verdict_RA(records, outcome, dataset) -> dict`.
- `verdict_RB(records, backbones, threshold) -> dict`: generalise `rq_aggregate.rq3m_adjudicate` to n seeds and any backbone list; the original 4-cell behaviour must be reproduced on `results/rq3_matched/`.
- `verdict_RC(records) -> dict`, `verdict_RD(records) -> dict`, `verdict_RE(records) -> dict`, `verdict_RF(records) -> dict`.
- `write_verdicts(dest=Path("results/tmlr/verdicts.json")) -> dict`.

`docs/TMLR_PRESPEC.md` contents:
- A9 verbatim, with exact outcome keys (e.g. `ece_pooled__evidential_native` / `__softmax`, `accuracy_mean__*`, `ood_auroc__{mini_near,tin}__vacuity_native` / `__msp`, and so on).
- The A3 table and seeds.
- The exclusion rule.
- "Verdicts are computed by `scripts/tmlr_adjudicate.py` at commit <filled at gate>."
- A changelog section. Any later edit is appended there, never silently rewritten.

- [ ] **Step 1: Write the failing tests.** Use synthetic records engineered for each outcome.

```python
def test_RA_site_only(): ...   ; def test_RA_module_only(): ...   ; def test_RA_both(): ...   ; def test_RA_inconclusive(): ...
def test_RB_reproduces_thesis_verdict():   # on committed results/rq3_matched/*.json: verdict == "backbone_intrinsic", fired 3/4
def test_RC_explains_vs_persists(): ...
def test_RD_family_consistent_vs_not(): ...
def test_sigma_uses_ddof1(): ...
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite.
- [ ] **Step 5:** 🔒 HUMAN GATE. The human commits **and pushes** the prespec, the adjudicator and all Phase 1 code, then:
  - writes the commit SHA and the push time into the prespec header in a follow-up commit;
  - optionally registers the doc on OSF (D6);
  - runs `python scripts/tmlr_configs.py --freeze`, then commits and pushes again. This final SHA is the `GIT_SHA` for every wave.

## Phase 2 — Runs on Kaggle (the human operates; AI prepares shards and reads results)

### Task 12: Wave 0 — end-to-end validation (1 account, about 2 h)

- [ ] Run `uv run --frozen python scripts/tmlr_shards.py --wave 0`. This builds the Wave 0 shards from these 9 manifest runs (`tmlr_design.WAVE0_RUN_IDS`), one per new code path, plus regression anchors. No Aircraft run is included, because Aircraft is built during Wave 1:
  1. E1 R18 btl_ms H r16 evid s42 (`regression_ref` rq3_matched)
  2. E1 MNv3 lora_ss L r10 soft s42 (`regression_ref` rq3_matched)
  3. E1 R18 lora_ms H [3,3,4,3] evid s42
  4. E1 R18 btl_ss H r40 soft s42
  5. E2 R34 btl_ms H r16 soft s42
  6. E2 MNv2 lora_ss unmatched r16 evid s42
  7. E2 MNv2 btl_ss H r13 evid s42
  8. E4 `cifar_1shot_r18_parallel_softmax_seed42` (`regression_ref` results/grid)
  9. E5-val kl 0.0 / prior 1.0 (the VAL-only path)
- [ ] The human runs W0-01 on account 1 and notes the GPU quota before and after.
- [ ] Run `scripts/tmlr_pull.sh`, then `uv run --frozen python scripts/tmlr_collect.py --incoming incoming/`.
- [ ] **Gate (all must hold):**
  - 9/9 done.
  - Runs 1, 2 and 8 have guard `exact` or `within_tol`.
  - n_params match.
  - `env.gpu_name` contains T4.
  - Mahalanobis and MaxLogit keys are present.
  - The Aircraft TIN-removed count is logged.
- [ ] If a regression guard fails, use superpowers:systematic-debugging. The first suspect is the torch version: compare `env` and pin `torch==<version>` in the notebook. Do not proceed until the guard passes.
- [ ] Record the measured lane throughput and quota billing in the runbook. If two lanes give < 1.3× one lane, set `LANES = 1` and re-pack.

### Task 13: Wave 1 — E1 + E2 + E5-val (4 accounts in parallel)

- [ ] Run `scripts/tmlr_shards.py --wave 1 --timings results/tmlr/_timings.json`, then `assign_accounts` for 4 accounts. Paste the schedule into the runbook.
- [ ] The human launches one shard per account at once and then each account's next shard as soon as its session ends. Pull and collect after every session.
- [ ] Run `tmlr_shards.py --retry` until `_coverage.json` shows E1, E2 and E5-val complete with 0 errors.
- [ ] Run `uv run --frozen python scripts/tmlr_configs.py --select-e5 results/tmlr/e5val`. This writes `e5_selected.yaml` (or records "current recipe selected; E5 skipped").
- [ ] 🔒 HUMAN GATE: commit and push the selection. The new SHA is used for Wave 2.

### Task 14: Wave 2 — E3a + E3b + E5 + E4

- [ ] Run `scripts/tmlr_shards.py --wave 2`. Expect about 8 shards, 2 per account, run back to back.
- [ ] Collect and retry until coverage is complete.
- [ ] Gate: all 114 E4 runs show `exact`/`within_tol` against `results/grid`, and all 16 E1 seed-42 anchors against `results/rq3_matched`. A failure means the August grid and today's Kaggle disagree. Stop and debug before trusting any new number.
- [ ] Back up immediately: HF repo (D4) **plus** one more copy (Google Drive or Zenodo draft). Raw results must never again exist in only one place.

## Phase 3 — Analysis

### Task 15: Statistics upgrade, verdicts, robustness

**Files:**
- Create: `scripts/tmlr_analysis.py`
- Modify: `scripts/rq_aggregate.py` (`eta_squared` balance flag, problem B1)
- Test: `tests/test_tmlr_analysis.py`

**Interfaces:**
- `eta_squared_with_ci(rows, outcome, factors, *, two_way=True, n_boot=2000, seed=0) -> dict`: main effects **and** two-way η² with 95% bootstrap CIs (resampling seeds within configuration, reusing `rq_completion.seed_bootstrap` / `ols_type2`), plus ω².
- `episode_bootstrap_delta(logits_a, logits_b, targets, metric: str, *, n_boot=2000, seed=0) -> dict`: paired over the 600 shared test episodes.
- `per_backbone_sign_test(pairs) -> dict`: an exact binomial per backbone. Never pool across backbones.
- `underconfidence_table(records) -> list[dict]`: `signed_calibration_gap__evidential_native` for every evidential model.
- `stream_robustness(old_records, new_records) -> dict`: σ and verdicts for fixed-stream 3-seed vs. varied-stream 5-seed runs.
- `run_all() -> None`: writes `results/tmlr/analysis/*.json` covering:
  - the RQ1/RQ2/RQ4 recomputation on E4 (which must equal the thesis numbers);
  - Aircraft (R-E);
  - EDL-recipe substitution (R-F);
  - RQ2 with six scores;
  - all adjudications via `tmlr_adjudicate.write_verdicts`.

- [ ] **Step 1: Write the failing tests**

```python
def test_eta_shares_and_residual_sum_to_one(): ...
def test_ci_contains_point_estimate(): ...
def test_episode_bootstrap_identical_inputs_zero_width(): ...
def test_eta_squared_flags_missing_cell():       # B1: drop one cell -> balanced is False
def test_e4_reproduces_thesis_rq1(): ...          # on fixtures copied from results/rq_completion: shares match to 1e-9
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the task tests and the full suite.
- [ ] **Step 5:** Run `uv run --frozen python scripts/tmlr_analysis.py`. Read `results/tmlr/verdicts.json` and write `results/tmlr/ANALYSIS_REPORT.md`. It must cover:
  - every verdict, with the rule, counts and a pass/fail per cell;
  - every surprise;
  - what changed versus the thesis.

  Report negative or inconclusive verdicts exactly as computed.
- [ ] **Step 6:** Update `progress.txt` with a "TMLR phase" entry: what ran, coverage, verdicts, and the decisions log. Use the new RQ numbering and say so.
- [ ] **Step 7:** 🔒 HUMAN GATE.

## Phase 4 — Writing the paper

### Task 16: Paper assets

- [ ] Create `scripts/tmlr_paper_assets.py`. It reads only `results/tmlr/analysis/*.json`, `verdicts.json` and the committed thesis results, and writes vector PDFs to `paper/figures/` and booktabs `.tex` tables to `paper/tables/`.
- [ ] Load the `dataviz` skill before writing plot code; use one palette across all figures.
- [ ] Figures:
  1. The findings map (updated F16).
  2. The objective × score AUROC grid with six scores (RQ2).
  3. The sites × module decomposition per backbone (R-A).
  4. Matched vs. unmatched ΔECE across four backbones (R-B/R-C/R-D).
  5. Evidential ECE before and after the refit (RQ4).
  6. The Aircraft replication panel.
- [ ] Test `tests/test_tmlr_paper_assets.py::test_every_number_traceable`: each `\newcommand` value in `paper/tables/numbers.tex` is produced from a named JSON key, never typed by hand.

### Task 17: Manuscript (`paper/`)

- [ ] Set up the official TMLR LaTeX style (the `tmlr.sty` repository linked from the TMLR author guide), in anonymous mode.
- [ ] Write `paper/CLAIMS.md` before the prose: one row per claim → result file → key → rule ID. Every sentence with a number must map to a row.
- [ ] Sections:
  1. Introduction: the reliability-attribution question and contributions.
  2. Related work: refreshed (next step).
  3. Setup: protocol, frozen episodes, VAL-only selection, the 4 arms, matched budgets, datasets including Aircraft.
  4. Objective vs. score (RQ2, six scores, Aircraft replication).
  5. Architecture vs. budget vs. backbone (RQ3, E1/E2/E3b, R-A…R-D).
  6. Evidential calibration and the post-hoc refit (RQ1 head effect, R-F, RQ4, underconfidence).
  7. Robustness: stream/seeds, E4 reproduction, budget-spread sensitivity.
  8. Limitations: ImageNet pretraining, the one-recipe policy, the evidential head studied as one family, latency not measured on device, non-exhaustive literature search.
  9. Conclusion.
  - Appendix: full tables, prespec, compute, reproducibility.

  Keep the main text near the length TMLR's guide recommends for normal review time; check the current author guide.
- [ ] Refresh the related work with the `anthropic-skills:deep-research` skill:
  - Search for anything after 2026-08 on: objective × score crossings for OOD; evidential + energy; adapter architecture vs. budget; PEFT calibration on CNNs; few-shot reliability on frozen backbones.
  - Re-verify Krumpl (WACV 2026), Genç (ECCV-W 2026), Hayta (arXiv:2605.22746), BayesAdapter (IJCV 2026), CP-tensor adapters, LoRA-Ensemble, and "Be Confident in What You Know" (NeurIPS 2024).
  - Every bib entry gets the "checked against publisher or arXiv page" comment that `docs/thesis_report/refs.bib` already uses.
- [ ] Wording audit: grep the manuscript for "pre-registered", "163", "250×", "follows the parameter budget", "state-of-the-art". Each hit must be removed or justified.
- [ ] Anonymisation: no names, no IUT, no GitHub URL. Code is linked through an anonymous mirror (Task 19). Remove self-identifying thesis citations, or cite the thesis in the third person per TMLR policy.
- [ ] Supervisor review round; integrate the comments.
- [ ] 🔒 HUMAN GATE.

## Phase 5 — Release and submission

### Task 18: Release branch cleanup

- [ ] Create branch `tmlr-release` (human).
- [ ] Remove the relay paths listed in Global Constraints (D5) and the stale results files (`results/slides_outline.md`, `results/smoketest_*`, `results/metrics.json`, `results/reliability_plot.png`, `results/ood_histogram.png`, `results/training_curve.png`).
- [ ] Add `LICENSE` (D2), `CITATION.cff`, and a paper-focused `README.md` covering how to reproduce a run, how to reproduce every table from `results/tmlr/`, and where the artifacts live.
- [ ] Fix stale `CLAUDE.md` references (docs deleted in commit 44d4857) and the "Master's" vs "B.Sc." inconsistency in `docs/MY_CONTRIBUTIONS.md`.
- [ ] Full test suite green on the branch.
- [ ] 🔒 HUMAN GATE.

### Task 19: Artifacts and submission

- [ ] Upload to the HF dataset (kept private until acceptance, then public), or to Zenodo with a DOI at camera-ready: per-run `metrics.json`, `adapter.pt` (KB-sized, which makes every model reproducible from the public torchvision weights), `logits.npz`, the manifest, the shard reports and the verdicts.
- [ ] Create an anonymous mirror of `tmlr-release` (e.g. anonymous.4open.science) and put its link in the paper.
- [ ] Final checks:
  - every number is traceable (`paper/CLAIMS.md`);
  - `pytest` is green;
  - the PDF compiles from clean;
  - an author-identity grep of the PDF and supplementary zip comes back empty.
- [ ] The human submits on OpenReview (TMLR), with the supplementary material: code zip and the prespec.

---

## Execution order (chosen 2026-10-22: Native, maximum parallelism)

Task numbers are documentation labels; this is the order of work.

1. **Block A — AI codes, human does setup in parallel.**
   - AI: Task 0.2 (ResNet, MobileNetV2, MaxLogit summaries only), Tasks 4 → 1 → 2 → 3 → 6 → 7.
   - **A fresh review after Task 7**, before any further code.
   - AI: Tasks 8 → 9 → 10 → 11.
   - Human, meanwhile: Task 0.1 (Kaggle data sharing, HF token, quota check, Aircraft download).
2. **Gate.** The human commits and pushes; this gives the `GIT_SHA`.
3. **Wave 0 split across all 4 accounts:** 2 runs each, about 30–45 min. This also measures T4 x2 throughput and quota billing on every account at once.
4. **Wave 1 on all 4 accounts:** E1 + E2 + E5-val. While it runs, the AI does Task 0.2 (Aircraft and Meta-Dataset summaries), Task 5 (Aircraft), Task 15 (analysis) and Task 16 (paper assets), testing them against Wave 1 results as they arrive.
5. **Gate.** The human pushes the Aircraft code and the E5 selection.
6. **Wave 2 on all 4 accounts:** E3a + E3b + E5 + E4, 2 sessions per account back to back. The AI drafts the paper sections meanwhile.
7. **Retries, then analysis, then writing and release** (Tasks 15 → 19).

Aircraft is deferred because only Wave 2 needs it. That takes Task 5 off the critical path to the first GPU run.

## Execution notes for the AI

- The AI cannot click through Kaggle. It prepares shards and schedules, reads the collected results, and decides on retries; the human launches sessions.
- When something fails on Kaggle, read `_lane{i}.log` and `_lane{i}_runlog.jsonl` from the collected zip before guessing.
- Never "fix" a failing regression guard by widening tolerances.
- If a verdict comes out negative or inconclusive, it goes in the paper exactly as computed. The prespec changelog is the only place a rule may change, and every change is reported in the paper.
