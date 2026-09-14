# 6. Repo Map

Where everything lives, which files are current and which are history, how to run things, and the rules
of the repo.

---

## 6.1 Top level

| Path | What it is | Status |
|---|---|---|
| `README.md` | Short entry page | Current |
| `CLAUDE.md` | Instructions for AI coding assistants working in this repo | Current |
| `proposal.txt` | The original research proposal (original RQs, planned datasets and backbones) | History; still cited for "proposed vs delivered" |
| `progress.txt` | Step-by-step status log + decisions log | Current but partly out of date (see `05_problems_and_open_work.md` A4) |
| `thesis_implementation_instructions.txt` | Rule: justify implementation choices against the paper summaries | Current |
| `PAPER SUMMARIES/` | Our summaries of the 32 papers in the reading list | Current |
| `docs/` | Reports, results documents, defence material, this guide | See 6.2 |
| `step_writeups/` | Detailed write-up per step | History (see 6.2) |
| `src/` | Model, data, training and evaluation code | Current |
| `scripts/` | Command-line entry points and analysis scripts | See 6.4 |
| `configs/` | YAML experiment configs | See 6.5 |
| `tests/` | pytest tests (36 test files) | Current |
| `notebooks/` | Kaggle/Colab notebooks that ran each experiment | See 6.6 |
| `results/` | Result files, tables and plots | See 6.7 |
| `app/` | Sentinel demo web app | Demo only; not connected to the trained adapters |
| `requirements.txt`, `pyproject.toml`, `uv.lock` | Python dependencies | Current |
| `mainul-doc/`, `thesis-gpu/`, `integrations/`, `third_party/`, `src/vgpu/`, `requirements-vgpu.txt`, `docs/gpu-relay-guide.md`, `docs/verified_gpu_pool_probe.py` | GPU relay system (an optional way to run on a shared GPU) | Not used by any thesis result; decision open (E6) |

**Not in git** (local only): `data/` (datasets and split files), `checkpoints/`, `.venv/`, `.env`, and the
large zips in `results/`.

---

## 6.2 Documents

### Current — use these

| File | Purpose |
|---|---|
| `docs/guide/` | **This guide** — plain-language overview of everything |
| `docs/RQ_SUPERVISOR_REPORT.md` (+ `.pdf`) | Formal report: four RQs, answers, novelty, limits, traceability |
| `docs/RQ_RESULTS_SUMMARY.md` | Full evidence for each RQ, including novelty checks and the reviewer checklist |
| `docs/RESULTS_MASTER.md` | All grid tables (accuracy, F1, ECE, Brier, AUROC, FPR@95, efficiency, Pareto) and positioning against published methods. **Generated** — edit `docs/RESULTS_MASTER_template.md`, then run `scripts/make_results_master.py` |
| `docs/RQ3_MATCHED_BUDGET_PLAN.md` | RQ3 pre-registration. Do not edit §1–6 |
| `docs/DEFENCE_SLIDE_PLAN.md` | Slide-by-slide defence plan, Q&A practice, wording rules, asset checklist |
| `docs/DEFENCE_BRIEF.md` | Answer to "CNNs are outdated". Uses the **original** RQ numbers |
| `docs/CITATION_AUDIT.md` | Which papers are used, missing or unused |
| `docs/refs.bib` | Bibliography |
| `step_writeups/rq3_matched_budget.txt` | Execution write-up for the RQ3 experiment |

### History — useful background, but check claims against `docs/guide/03_results.md`

| File | Purpose |
|---|---|
| `step_writeups/step1.txt` … `step11.txt` (+ `step4_5.txt`) | Why each step's results came out as they did. Use original RQ numbering. Steps 4.5–9 contain claims later overturned |
| `docs/NEW_RQS_PROGRESS_REPORT.md` | Aug 25 audit of Phase A (**not in git**). The only other record of Phase A besides `04_experiments.md` |

### Deleted on 2026-09-14 (recoverable)

Old plans, handoff notes, old presentation scripts, the five-RQ draft and its task plan, plain-language
step explainers, and duplicate notebooks. The full list and reasons are in `progress.txt`'s last
decisions-log entry. To view one: `git show 8d706ea:<path>`, e.g.
`git show 8d706ea:docs/NEW_RQS_TASK_PLAN.md`.

---

## 6.3 Source code (`src/`)

The model is assembled by `build_model(cfg)` in `src/models/bpeft_model.py`. Each part is picked by a
config key through a small factory function in its package's `__init__.py`.

| Package | Contents |
|---|---|
| `src/backbones/` | `resnet18.py`, `mobilenetv3.py` (frozen, pretrained) |
| `src/adapters/` | `bottleneck.py`, `placement.py` (serial/parallel sites), `lora.py`, `bitfit.py`, `full_ft.py`, `linear_probe.py` |
| `src/heads/` | `prototype_head.py` (main; holds `to_evidence()`), `linear_head.py` (legacy) |
| `src/losses/` | `evidential.py` (Sensoy loss + options), `cross_entropy.py` |
| `src/trainers/` | `episodic_trainer.py` (main), `fewshot_trainer.py` (legacy single-episode) |
| `src/evaluators/` | `episodic.py` (score × OOD-set matrix), `calibration.py`, `ood.py` (incl. energy), `temperature.py`, `accuracy.py` |
| `src/datasets/` | `cifar_fs.py`, `mini_imagenet.py`, `episode_sampler.py`, `svhn_ood.py`, `tinyimagenet_ood.py`, `gaussian_noise_ood.py` |
| `src/utils/` | `config.py` (YAML with `extends:`), `seed.py`, `wandb_utils.py`, `plots.py`, `efficiency.py`, `pareto.py` |

Two training/evaluation paths exist, chosen by `cfg.trainer.type`:

- `single_episode` — Steps 1–3; kept unchanged so old tests still pass.
- `episodic` — Step 4 onward; used for everything in the thesis.

---

## 6.4 Scripts (`scripts/`)

### Core
| Script | Does |
|---|---|
| `train.py` | Train one config |
| `evaluate.py` | Evaluate one config on the 600 test episodes |
| `build_cifar_fs_split.py`, `build_mini_imagenet_split.py` | Create the frozen split files in `data/` |

### Step 10 grid
| Script | Does |
|---|---|
| `build_grid_configs.py` | Generate the 120 grid configs |
| `run_mvt_grid.py` (+ `run_mvt_grid.sh`) | Run the grid, resumable |
| `aggregate_grid.py` | Build `results/mvt_results.json`; also re-checks the Step 6 reproduction |
| `make_master_tables.py` | LaTeX/PNG tables |
| `grid_plots.py` | Reliability and OOD plots |
| `make_results_master.py` | Build `docs/RESULTS_MASTER.md` from its template |

### Step 11 efficiency
| Script | Does |
|---|---|
| `efficiency_table.py` | Measure params, FLOPs, latency, memory (`--device cuda/cpu/both`, `--include-reference-backbones`) |
| `pareto_plots.py` | Pareto frontiers and plots |

### New RQs
| Script | Does |
|---|---|
| `rq_core.py` | All-scores evaluation (RQ2), evidence refit (RQ4), regression guard |
| `rq_drivers.py` | Runs Phase A over recovered models |
| `rq_aggregate.py` | η² and verdicts for RQ2/RQ4/rank sweep, plus the RQ3 matched-budget rule. **Has the `balanced` bug (B1)** |
| `rq5_sweep.py` | Rank-sweep configs (former RQ5) |
| `rq3_matched.py` | RQ3 matched-budget configs, guards, runner, `build_verdict()`. No command-line interface; called from the notebook |

### Historical (earlier steps)
`step45_val_sweep.py` (still the pattern for VAL-only tuning), `step45_verdict.py`, `step6_placement_plot.py`,
`step7_ood_consolidate.py`, `step8_backbone_compare.py`, `step9_dataset_compare.py`, `run_step5.sh`,
`run_grid.sh`.

---

## 6.5 Configs (`configs/`)

| Path | Contents |
|---|---|
| `base.yaml` | Defaults that every config extends |
| `test_episodes.yaml` | **Frozen** 600 test seeds (0–599) |
| `val_episodes.yaml` | **Frozen** 100 validation seeds (10000–10099) |
| `exp_step1*.yaml` | Steps 1–3 |
| `exp_phase2_*.yaml` | Steps 4–4.5 (`exp_phase2_evidential_retuned.yaml` is the evidential recipe base) |
| `exp_phase3_*.yaml` | Steps 5–6 |
| `exp_phase5_mbnet_*.yaml`, `exp_phase5_mini_*.yaml` | Steps 8–9; also the parents of the grid configs |
| `grid/` | 120 Step 10 configs + `_index.json` |
| `rq5/` | 21 rank-sweep configs + `_index.json` |
| `rq3_matched/` | 48 matched-budget configs + `_index.json` |

Configs use `extends:` to inherit from a parent and change only what differs. Don't delete old
`exp_*.yaml` files: the grid configs inherit from them.

---

## 6.6 Notebooks (`notebooks/`)

| Notebook | Experiment | Has outputs? |
|---|---|---|
| `step1_calibration_fix.ipynb` | Step 1 | Yes |
| `step2_clean_repo.ipynb` | Step 2 | Yes |
| `step3_wandb_repro.ipynb` | Step 3 | Yes |
| `step4_episodic.ipynb` | Step 4 | Yes |
| `step4_5_settle.ipynb` | Step 4.5 | Yes |
| `step5_phase3.ipynb` | Step 5 | Yes |
| `step6_placement.ipynb` | Step 6 | Yes |
| `step7_ood.ipynb` | Step 7 | Yes |
| `step8_mbnet.ipynb` | Step 8 | Yes |
| `step9-mini(1).ipynb` | Step 9 (completed run) | Yes |
| `step9-mini-v2.ipynb` | Step 9 via GPU relay (alternative) | Partly |
| `step10a_cifar_5shot.ipynb` | Step 10, CIFAR-FS 5-shot (36 runs) | **No** — executed copy not kept |
| `step10b_cifar_1shot.ipynb` | Step 10, CIFAR-FS 1-shot (36 runs) | Yes |
| `step10c_mini.ipynb` | Step 10, MiniImageNet (48 runs) | Yes |
| `step11_efficiency.ipynb` | Step 11 | Yes |
| `New_1,2,5A.ipynb` | Phase A (RQ2 + RQ4) | Yes |
| `NewRQ_1,2,5_B.ipynb` | Phase B (rank sweep) | Yes |
| `rq3_matched_budget.ipynb` | RQ3 matched budget — source | No |
| `rq3_budget.ipynb` | RQ3 matched budget — executed run record | Yes |

Notebook names `New_1,2,5A` / `NewRQ_1,2,5_B` use the **retired** five-RQ numbering.

New experiment notebooks for Kaggle should be **self-contained** (code written into cells), not
dependent on a fresh `git pull`.

---

## 6.7 Results (`results/`)

### Current thesis evidence
| Path | Backs |
|---|---|
| `mvt_results.json` | Step 10 grid; RQ1; RQ3's 16 pairs; Orig-RQ1–3 |
| `grid/` | Per-run grid metrics + `_run_log.jsonl` |
| `mvt_table_*.{tex,png}`, `grid_plots/` | Grid tables and plots |
| `efficiency_table.json`, `efficiency_table_rerun.json`, `pareto_frontier.json`, `pareto_*.png`, `pareto_audit/`, `step11_session.log` | Step 11; Orig-RQ4 |
| `rq_factorial/`, `rq_summary.json`, `rq_checkpoint_audit.json`, `new_rqs_session_phaseA.log` | Phase A; RQ2 and RQ4 |
| `rq5/`, `rq5_rank_sweep.png`, `new_rqs_session_phaseB.log` | Phase B; former RQ5 |
| `rq3_matched/` (incl. `verdict.json`), `rq3_matched_delta_ece.png`, `rq3_matched_session.log` | RQ3 matched budget |

### History (earlier steps)
`phase2_*` (Step 4), `step45_*` (Step 4.5), `phase3_*` (Steps 5–6), `phase4_*` (Step 7), `phase5_*` and
`step8_*` (Step 8), `step9_mini_artifacts/` (Step 9), `step2_eval_*`, `step3_*`, `step6_*`, `step7_*`.
Keep them: `aggregate_grid.py` compares the grid against the Step 6 results.

### Leftovers that look current but aren't
`slides_outline.md` (April pre-defence slides), `metrics.json`, `reliability_plot.png`,
`ood_histogram.png`, `training_curve.png` (pre-defence / Step 1), `smoketest_*`.

### Large files not in git (~4.7 GB)
`step10a_cifar_5shot_artifacts.zip`, `step10b_cifar_1shot_artifacts.zip`, `step10c_mini_artifacts.zip`
(grid models), `new_rqs_results.zip`, `new_rqs_logits.zip` (Phase A per-episode scores),
`rq3_matched_results.zip`, `rq3_matched_checkpoints.zip`. They may be the only copies (decision E5).

---

## 6.8 How to run

```bash
# setup
pip install -r requirements.txt
python scripts/build_cifar_fs_split.py
python scripts/build_mini_imagenet_split.py

# tests
python -m pytest -q

# train + evaluate one config
python scripts/train.py    --config configs/<name>.yaml
python scripts/evaluate.py --config configs/<name>.yaml --num-episodes 600 --wandb-mode disabled

# rebuild grid summary and the results document (no GPU needed)
python scripts/aggregate_grid.py
python scripts/make_results_master.py
```

For real experiments, use the matching notebook on Kaggle (T4 GPU, internet on). Large datasets
should be attached as Kaggle datasets.

**Lessons learned about hosted notebooks.**

- Don't extract archives with many files onto Google Drive; it can hang for hours. Read straight from the
  zip.
- Run long jobs in-process, or tee their output to a log file. Output from subprocesses can disappear
  from the notebook.

---

## 6.9 Rules of the repo

1. **Tune only on VAL episodes** (seeds 10000–10099). Report only on TEST episodes (seeds 0–599).
   Nothing in the code enforces this; it's on you.
2. **Never regenerate frozen files:** `configs/test_episodes.yaml`, `configs/val_episodes.yaml`, and the
   split files in `data/`.
3. **Same config twice must give byte-identical results.** If not, find the non-determinism before
   trusting the result.
4. **Keep old results reproducible.** New code must keep old defaults; don't add keys to the metrics JSON.
5. **One evidence formula.** Use `PrototypeHead.to_evidence()`; never re-implement
   `softplus(score × scale + bias)` elsewhere. That mistake caused the Step 4 collapse.
6. **Numbers come from result files**, not from memory or notebook printouts.
7. **Don't commit without human review.**
8. **Use the final RQ numbering** in new writing, and label original questions as Orig-RQ.
