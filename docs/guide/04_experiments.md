# 4. Experiment Log

Every experiment the project ran, in order: what it did, how big it was, which files hold the code,
configs and results, which checks passed, and which checks were **not** done.

Use this page to find the evidence behind a number, or to rerun something.

---

## 4.1 Overview

| # | Experiment | Date (2026) | Runs | Where it ran | Still used in the thesis? |
|---|---|---|---|---|---|
| 1 | Step 1 — calibration fix | Apr 21 | 20 episodes, stand-in data | Colab | History only |
| 2 | Step 2 — clean repo | May 13 | Reproduction of Step 1 | Colab | Code base |
| 3 | Step 3 — W&B + reproducibility | May 16 | 2 configs × 600 episodes | Colab | 600 frozen test episodes |
| 4 | Step 4 — real CIFAR-FS + episodic training | May–Jun | 2 configs | Colab | History (negative result) |
| 5 | Step 4.5 — fair baselines | Jul 6–7 | 2 configs + VAL sweep | Colab | Recipe used by everything after |
| 6 | Step 5 — LoRA, BitFit, Full FT, Linear Probe | Jul 21 | 8 configs | Colab | Superseded by grid |
| 7 | Step 6 — adapter placement | Jul 21 | 4 configs | Kaggle T4 | Only source for serial vs parallel |
| 8 | Step 7 — Gaussian OOD | Jul 22 | 2 configs | Kaggle T4 | Superseded by grid |
| 9 | Step 8 — MobileNetV3-Small | Jul 26 | 4 configs, ~44 min | Kaggle T4 | Superseded by grid |
| 10 | Step 9 — MiniImageNet | Jul 30 | 10 configs, ~142 min | Kaggle T4 | Superseded by grid |
| 11 | **Step 10 — full grid** | Aug 2–6 | **120 runs, 36.3 h** | Kaggle T4 (3 notebooks) | **Yes — main data** |
| 12 | **Step 11 — efficiency / Pareto** | Aug 7–9 | 12 measured settings | Kaggle T4 GPU + CPU | **Yes** |
| 13 | **Phase A — factorial + refit** (RQ2, RQ4) | Aug 25–26 | 99 recovered models, 5.51 GPU-h | Kaggle | **Yes** |
| 14 | **Phase B — rank sweep** (former RQ5) | Aug 26 | 21 runs | Kaggle | **Yes** (negative result) |
| 15 | **RQ3 matched-budget** | Aug 26–27 | **48 runs, 6.2 h** | Kaggle T4 | **Yes — strongest result** |

"Superseded by grid" means the same kind of setting was re-run inside the Step 10 grid with 3 seeds. The
single-seed numbers from those steps are history, not thesis results.

---

## 4.2 Early steps (1–9)

Each closed step has a full write-up in `step_writeups/stepN.txt`. Those files are **historical**: some
headline claims in them were later corrected (see `02_research_story.md` §2.3).

### Step 1 — Calibration fix (Apr 21)
- **Did:** fixed two evidential-loss bugs; switched ReLU → softplus; added KL annealing.
- **Result:** ECE 0.526 → 0.167.
- **Files:**
  - notebook `notebooks/step1_calibration_fix.ipynb`
  - results `results/metrics.json`, `reliability_plot.png`, `ood_histogram.png`, `training_curve.png`
- **Note:** used a stand-in setup (CIFAR-100 test images, 20 episodes), not the real benchmark.

### Step 2 — Clean repo (May 13)
- **Did:** split `src/` into packages; added YAML configs with `extends:`; made `scripts/train.py` and
  `scripts/evaluate.py` the entry points; 28 tests.
- **Files:** `notebooks/step2_clean_repo.ipynb`, `configs/exp_step1*.yaml`, `results/step2_eval_*.json`.
- **Check passed:** reproduced Step 1 within tolerance.

### Step 3 — W&B + reproducibility (May 16)
- **Did:** W&B wrapper; full seeding; `configs/test_episodes.yaml` (600 seeds, frozen).
- **Files:** `notebooks/step3_wandb_repro.ipynb`, `results/step3_*`.
- **Check passed:** same config twice gives a byte-identical results file.
- **Note:** W&B ran offline, because the API key was invalid.

### Step 4 — Real CIFAR-FS + episodic training (May–Jun)
- **Did:** Bertinetto split; episodic meta-training; prototype head; macro-F1 and FPR@95 metrics.
- **Result:** OOD advantage almost gone (+0.006); evidential worse calibrated.
- **Files:**
  - notebook `notebooks/step4_episodic.ipynb`
  - configs `configs/exp_phase2_{evidential,softmax}.yaml`
  - results `results/phase2_*_metrics.json`
- ⚠️ **Tracker bug:** in `progress.txt` the Step 4 section still shows "NOT STARTED" with unticked boxes,
  even though it ran. Its headline numbers are at the top of the same file.

### Step 4.5 — Settle the science (Jul 6–7)
- **Did:**
  - temperature scaling and energy baselines;
  - better evidential loss options;
  - TinyImageNet and CIFAR-100-held-out near-OOD sets;
  - a VAL-only tuning sweep (`kl_weight_max` ∈ {0.05, 0.1, 0.25}).
- **Files:**
  - notebook `notebooks/step4_5_settle.ipynb`
  - config `configs/exp_phase2_evidential_retuned.yaml`
  - scripts `scripts/step45_val_sweep.py`, `step45_verdict.py`
  - results `results/step45_*_metrics.json`
- **Lesson:** TinyImageNet hung for over an hour when extracted onto Google Drive. The loader now reads
  straight from the zip.

### Step 5 — More adapters and baselines (Jul 21)
- **Did:** LoRA (on `layer4.0.downsample.0`, 12,288 params), BitFit (4,800), Full FT (11,176,512),
  Linear Probe (0/2).
- **Files:**
  - notebook `notebooks/step5_phase3.ipynb`
  - configs `configs/exp_phase3_{lora,bitfit,full_ft,linear_probe}_{evidential,softmax}.yaml`
  - results `results/phase3_{lora,bitfit,full_ft,linear_probe}_*`
- **Decisions:**
  - BatchNorm statistics stay frozen for every adapter.
  - Full FT uses LR 1e-5 and weight decay 1e-4.
  - The LoRA parameter formula in the old spec was wrong by 2×; the tests check the correct one.

### Step 6 — Adapter placement (Jul 21)
- **Did:** post-pool vs serial vs parallel bottleneck, both readings.
- **Files:**
  - notebook `notebooks/step6_placement.ipynb`
  - configs `configs/exp_phase3_placement_{serial,parallel}_*.yaml`
  - results `results/phase3_placement_*`, `results/step6_placement_comparison.png`
- **Note:** this is the **only** serial-vs-parallel comparison in the project (one seed, ResNet-18,
  CIFAR-FS 5-shot).

### Step 7 — Gaussian far-OOD (Jul 22)
- **Did:** added a Gaussian-noise OOD set; re-ran the parallel winner.
- **Files:** `notebooks/step7_ood.ipynb`, `results/phase4_*`, `results/step7_ood_comparison.png`.
- **Note:** TinyImageNet was not filtered for class overlap here.

### Step 8 — MobileNetV3-Small (Jul 26)
- **Did:** second backbone; 4 configs (post-pool and parallel × 2 readings).
- **Files:**
  - notebook `notebooks/step8_mbnet.ipynb`
  - configs `configs/exp_phase5_mbnet_*.yaml`
  - results `results/phase5_mbnet_*`, `phase5_backbone_table.json`, `step8_*.png`, `step8_mbnet_MANIFEST.txt`
  - checkpoints in local `checkpoints/` (not in git)

### Step 9 — MiniImageNet (Jul 30)
- **Did:** second dataset; 10 configs; removed the 25 TinyImageNet classes that overlap MiniImageNet.
- **Files:**
  - notebook `notebooks/step9-mini(1).ipynb` (the completed run)
  - configs `configs/exp_phase5_mini_*.yaml`
  - results in `results/step9_mini_artifacts/`
- **Not confirmed:** the TinyImageNet overlap filter is correct by code reading only; no logged count
  confirms it.
- **Tests:** 208 passed, 6 failed. All 6 failures came from the Kaggle environment (an attached dataset
  confused tests that expect none), not from logic bugs.
- **Note:** `notebooks/step9-mini-v2.ipynb` is an alternative run through the GPU relay system. It is not
  the canonical run.

---

## 4.3 Step 10 — The full grid (Aug 2–6) — main data

**Design.** 2 datasets × 2 shot counts × 2 backbones × 2 adapters (parallel bottleneck, LoRA) × 2 readings
= 32 settings. Add 8 baseline settings (Full FT and Linear Probe, ResNet-18 + CIFAR-FS only) to get
**40 settings × 3 seeds = 120 runs**. One frozen recipe throughout.

| Item | Location |
|---|---|
| Notebooks | `notebooks/step10a_cifar_5shot.ipynb` (36 runs), `step10b_cifar_1shot.ipynb` (36), `step10c_mini.ipynb` (48) |
| Config generator | `scripts/build_grid_configs.py` → `configs/grid/*.yaml` (120) + `configs/grid/_index.json` |
| Runner | `scripts/run_mvt_grid.py` (+ `run_mvt_grid.sh`) |
| Aggregation | `scripts/aggregate_grid.py` → `results/mvt_results.json` |
| Tables | `scripts/make_master_tables.py` → `results/mvt_table_*.{tex,png}` |
| Plots | `scripts/grid_plots.py` → `results/grid_plots/` (16 plots + manifest) |
| Results document | `scripts/make_results_master.py` + `docs/RESULTS_MASTER_template.md` → `docs/RESULTS_MASTER.md` |
| Per-run results | `results/grid/` (metrics JSON + plots per run, and `_run_log.jsonl`) |
| Saved models | `results/step10{a,b,c}_*_artifacts.zip` (**not in git**, about 2 GB total) |
| Write-up | `step_writeups/step10.txt` |

**Checks passed.**

- 120/120 settings present; `missing_cells: []`.
- Run log: 123 "ok", 1 "skipped_done", 0 errors.
- Seed-42 grid runs reproduce Step 6's committed numbers **exactly** (29/29 and 55/55 keys).
- 233 tests passed.

**Checks not done / caveats.**

- The W&B dashboard grouping was checked from the configs only, never in W&B itself.
- `step10a_cifar_5shot.ipynb` is saved **without outputs**; the executed copy was not kept.
- Full FT and Linear Probe give identical numbers for all 3 seeds (no random initialisation), so their
  error bars are not real.
- One per-seed value is missing: `cifar_fs/5shot/mobilenetv3_small/lora/evidential`, seed 42,
  TinyImageNet near-OOD AUROC. `missing_cells` does not catch partial gaps like this.
- Planned side study 10.9 (longer KL warm-up for MobileNetV3 parallel evidential) was **deferred, never
  run**.

---

## 4.4 Step 11 — Efficiency and Pareto (Aug 7–9)

**Design.** Measure trainable/total params, FLOPs, GPU and CPU latency, and peak memory for the 12
(backbone, adapter, reading) combinations. Build Pareto frontiers. No training.

| Item | Location |
|---|---|
| Notebook | `notebooks/step11_efficiency.ipynb` |
| Measurement code | `src/utils/efficiency.py`, `scripts/efficiency_table.py` |
| Pareto code | `src/utils/pareto.py`, `scripts/pareto_plots.py` |
| Results | `results/efficiency_table.json`, `efficiency_table_rerun.json`, `pareto_frontier.json`, `pareto_*.png`, `pareto_audit/`, `mvt_table_efficiency.*`, `step11_session.log` |
| Write-up | `step_writeups/step11.txt` (§8 has both bug reports) |

**Checks passed.** The measured FLOPs match published figures for both backbones. 294 tests passed. An
extra architecture-only measurement of ViT-B/16 and DeiT-Tiny was made (size and speed only; no training).

**Bugs found.**

1. **Crash bug.** A bonus feature crashed and threw away a correct measurement 3 times. Fixed at the root,
   with a GPU-only regression test.
2. **Silent data bug.** Three scripts (`pareto_plots.py`, `make_master_tables.py`,
   `make_results_master.py`) picked the laptop CPU latency instead of the Kaggle CPU latency. Errors were
   up to 47%. Fixed and all outputs regenerated.
   - The recommended points did not change.
   - Two MobileNetV3-Small LoRA settings joined the CIFAR-FS frontier.
   - ⚠️ **No test covers this bug yet.**

**Limit.** No Jetson Nano. A single CPU thread stands in for an edge device. Latency is hardware-dependent
and **not** byte-reproducible.

---

## 4.5 Phase A — Objective × score factorial and evidence refit (Aug 25–26) — RQ2 and RQ4

This phase had **no write-up and no `progress.txt` entry** until this guide. This section is its record.

**Design.**

1. Recover the saved Step 10 models.
2. For each model, compute all four OOD scores (MSP, TS-MSP, energy, vacuity) under both readings → RQ2.
3. For each evidential model, refit `(scale, bias)` on the VAL episodes and re-evaluate on the 600 test
   episodes → RQ4.

No new training.

| Item | Location |
|---|---|
| Notebook (executed) | `notebooks/New_1,2,5A.ipynb` (`RUN_PHASE_A = True`) |
| Code | `scripts/rq_core.py` (scoring, refit, regression guard), `scripts/rq_drivers.py` (drivers), `scripts/rq_aggregate.py` (η² and verdicts) |
| Per-model results | `results/rq_factorial/*.json` (99) + `_run_log.jsonl` |
| Summary | `results/rq_summary.json` (`rq1_verdict` = RQ2 answer, `rq2_rows` / `rq2_verdict` = RQ4 answer — the key names use the retired draft numbering) |
| Model recovery record | `results/rq_checkpoint_audit.json` |
| Session log | `results/new_rqs_session_phaseA.log` |
| Raw archives | `results/new_rqs_results.zip`, `results/new_rqs_logits.zip` (1.04 GB of saved per-episode scores) — **not in git** |
| Independent audit | `docs/NEW_RQS_PROGRESS_REPORT.md` (Aug 25, **not in git**) |
| Old task plan | `docs/NEW_RQS_TASK_PLAN.md` — deleted Sep 14; recover with `git show 8d706ea:docs/NEW_RQS_TASK_PLAN.md` |

**Coverage.** 99 of 120 models were recovered. All 21 missing ones are CIFAR-FS 5-shot adapter models
(parallel and LoRA, both backbones); that slice has 15 of 36.

**Result.** 99 models OK, 0 errors, 21 skipped (no model file). 5.51 GPU-hours.

**Checks passed.**

- Regression guard: all 99 models reproduce the committed Step 10 numbers **exactly**.
- VAL discipline: all 99 used only VAL seeds 10000–10099 for fitting. No test seed was touched.
- Self-test 5/5 passed, including energy on evidential models = `logsumexp` of the same scores, and two
  runs byte-identical.
- Energy is computed on raw prototype scores with the same function for both readings.

**Not done.**

- The inline checks were never turned into pytest tests (`tests/test_factorial_scores.py`,
  `tests/test_evidence_affine_fit.py` don't exist).
- No repeated full run to confirm determinism.
- No test of degenerate cases for the refit.
- The 21 missing CIFAR-FS 5-shot models were never retrained.
- **Bug:** the `balanced` check in `scripts/rq_aggregate.py` reports `true` even though one setting is
  missing a whole reading. This is why "163×" is unstable (see `05_problems_and_open_work.md`).

---

## 4.6 Phase B — Rank sweep (Aug 26) — former RQ5

**Design.** CIFAR-FS × ResNet-18 × parallel bottleneck × evidential training (softmax and TS-softmax ECE
read from the same runs). Ranks {1, 2, 4, 8, 16, 32, 64} × seeds {42, 43, 44} = 21 new training runs.
Parameter counts run from 2,886 (rank 1) to 124,098 (rank 64).

| Item | Location |
|---|---|
| Notebook (executed) | `notebooks/NewRQ_1,2,5_B.ipynb` (`RUN_PHASE_B = True`) |
| Config generator | `scripts/rq5_sweep.py` → `configs/rq5/*.yaml` (21) + `_index.json` |
| Results | `results/rq5/*.json` (21) + `_run_log.jsonl`, `results/rq_summary.json` → `rq5`, `results/rq5_rank_sweep.png` |
| Session log | `results/new_rqs_session_phaseB.log` |

**Result.** 21/21 OK, 0 errors. No interior optimum (see `03_results.md` §3.7).

**Checks.** The parameter formula was verified for all 7 ranks. A config check that only rank and seed
differ was reported as passing (47 keys) in the Aug 25 audit. There is no pytest test for it.

---

## 4.7 RQ3 matched-budget experiment (Aug 26–27)

**Design (pre-registered before running).** MiniImageNet, 5-shot. Both adapters at the **same** budget
within each backbone:

| Backbone | Level | Bottleneck (rank → params) | LoRA (rank → params) | Mismatch |
|---|---|---|---|---:|
| ResNet-18 | Low | 6 → 12,504 | 16 → 12,288 | 1.76% |
| ResNet-18 | High | 16 → 31,744 | 41 → 31,488 | 0.81% |
| MobileNetV3-Small | Low | 16 → 6,928 | 10 → 6,720 | 3.10% |
| MobileNetV3-Small | High | 22 → 9,448 | 14 → 9,408 | 0.43% |

(Softmax counts shown; evidential adds 2.) × 2 readings, trained separately × 3 seeds = **48 runs**.

| Item | Location |
|---|---|
| Pre-registration | `docs/RQ3_MATCHED_BUDGET_PLAN.md` (§1–6 unchanged since before the run) |
| Notebooks | `notebooks/rq3_matched_budget.ipynb` (source), `notebooks/rq3_budget.ipynb` (executed run record) |
| Code | `scripts/rq3_matched.py` (design, guards, runner, verdict); aggregation in `scripts/rq_aggregate.py` ("RQ3 — matched budget" section) |
| Configs | `configs/rq3_matched/*.yaml` (48) + `_index.json` |
| Results | `results/rq3_matched/*.json` (48), `verdict.json`, `_run_log.jsonl`; `results/rq3_matched_delta_ece.png`; `results/rq3_matched_session.log` |
| Raw archives | `results/rq3_matched_results.zip`, `results/rq3_matched_checkpoints.zip` (1.08 GB) — **not in git** |
| Write-up | `step_writeups/rq3_matched_budget.txt` |

**How it ran.** One Kaggle T4 session, Aug 26 21:12 → Aug 27 03:24 UTC.

- 6.21 h in total; 192 s of training per run on average.
- 30 new runs, plus 18 re-runs of grid settings.
- **Deviation from plan:** the plan said 30 runs and reuse old models. All 48 were trained fresh instead,
  which is stronger.

**All 7 acceptance criteria passed.**

- 48/48 OK.
- Parameter counts exact for 16/16 arms.
- Mismatch ≤ 3.1%.
- Only allowed config keys differ (0 unexplained keys).
- 18/18 re-runs exactly match the grid.
- The verdict follows the pre-set rule.
- The secondary accuracy result is reported.

**Observation, not a cause.** ResNet-18 LoRA stops training 3–4× earlier than the other arms (best
epoch 1–4). This goes with its worse accuracy and calibration. It is offered only as something to look
at next.

**The verdict can be rebuilt offline** from `results/rq3_matched/`, without a GPU.

---

## 4.8 Rules every experiment followed

- **Never tune on test episodes.** Tuning and fitting use VAL seeds 10000–10099 only. Final numbers use
  test seeds 0–599 only.
- **Never regenerate frozen files:** `configs/test_episodes.yaml`, `configs/val_episodes.yaml`,
  `data/cifar_fs_split.json`, `data/mini_imagenet_split.json`.
- **Keep old results reproducible.** New code keeps old defaults. Since Step 8 the metrics JSON format has
  been kept fixed, so earlier results can still be re-run byte-identically.
- **Transcribe numbers from result files, never from memory or notebook printouts.**
