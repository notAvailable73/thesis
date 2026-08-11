# Step 10 — MVT Grid Execution: Implementation Plan

Status: planning only, nothing implemented yet.
Depends on: Steps 1–9 (all CLOSED per `progress.txt`).
References: `plan.txt` §Phase 5, `implementation.txt` §10, `progress.txt`
("STEP 10 — MVT GRID EXECUTION"), `instructions.txt`, `step_writeups/step9.txt`.

---

## 1. What Step 10 is

120 runs, one big results table, all four RQs answered at once.

| Axis | Values | Count |
|---|---|---|
| dataset | cifar_fs, mini_imagenet | 2 |
| shots | 1-shot, 5-shot | 2 |
| backbone | resnet18, mobilenetv3_small | 2 |
| adapter | bottleneck (placement `parallel` — the Step 6 winner), lora | 2 |
| head | prototype-softmax, prototype-evidential | 2 |
| seed | 42, 43, 44 | 3 |

= **96 PEFT runs**.

Plus the proposal's mandatory baselines (`proposal.txt` §6): full_ft, linear_probe,
on ResNet-18 × CIFAR-FS only × 2 shots × 2 heads × 3 seeds = **24 runs**.

**Total: 120 runs.**

Nothing here is scientifically new — every adapter/head/backbone/dataset
combination has already run individually in Steps 4–9. Step 10 is a
**plumbing + compute** step: generate configs, run them resumably, aggregate,
tabulate.

---

## 2. Three real risks (test these before burning GPU quota)

Everything else in the grid is a repeat of Steps 5–9. Only these are genuinely new:

1. **1-shot has never run anywhere in this thesis.** Steps 4–9 are all
   5-shot. The prototype head at k=1 means "the prototype *is* the single
   support image"; the KL schedule and evidence affine
   (`scale 2 / bias -6`, see `configs/base.yaml`) were tuned on 5-shot only.
2. **LoRA on MobileNetV3-Small has never run.** `src/adapters/lora.py`'s
   `_mobilenetv3_default_lora_target()` branch exists but has never been
   exercised against a real model in a real run.
3. **LoRA on MiniImageNet has never run.** Step 9 only ran Bottleneck +
   Linear-Probe on MiniImageNet (`step_writeups/step9.txt` §5).

**Action 10.0 (pre-flight smoke test):** run 3 configs covering these three
cases with `--num-episodes 20` and `trainer.num_epochs: 2` (~10 min total).
Do this first, inside notebook 10a (see §7).

---

## 3. Carry-overs from Steps 8/9 — decision: freeze the recipe

`progress.txt`'s "Next" note hands Step 10 three open items:
per-backbone/per-dataset VAL tuning, the mbnet/parallel/evidential
KL-warm-up-truncation bug (2-for-2 across CIFAR-FS and MiniImageNet per
`step_writeups/step9.txt` §5), and the MobileNetV3-vs-ResNet-18 gap that
ballooned on MiniImageNet.

**Decision: keep the grid recipe FROZEN across all 120 runs** (LR 5e-3,
rank 16, `kl_weight_max` 0.1, evidence affine scale 2 / bias -6 — same as
`configs/exp_phase2_evidential_retuned.yaml`). This keeps every cell a
controlled comparison that differs only on the grid's own axes, and — just
as importantly — keeps the three notebooks in §7 independently runnable
without one notebook's tuning results depending on another's.

Per-cell VAL tuning is **not** folded into the grid. Instead:

- **Side-study 10.9** (separate, small, VAL-seeds-only, ~9 runs,
  ~2 GPU-hours): sweep `kl_anneal_steps` (longer warm-up) ×
  `early_stop_patience` ∈ {5, 10} for the mbnet/parallel/evidential cell
  specifically, using the `scripts/step45_val_sweep.py` in-process pattern
  (never the 600 frozen TEST seeds in `configs/test_episodes.yaml` — only
  `configs/val_episodes.yaml`, seeds 10000–10099, per
  `thesis_implementation_instructions.txt`'s VAL/TEST rule).
- If it finds a clear VAL win, re-run only the affected cells and report
  both numbers side by side in `step_writeups/step10.txt` — never silently
  swap them into the main table.
- Report the untuned-recipe limitation honestly in the writeup, exactly as
  `step_writeups/step9.txt` §5 already does.

---

## 4. Dataset setup — exact upload structure

### 4.1 One Kaggle dataset

**Slug:** `beft-thesis-data` (owner `notavailable73`) — note this is NOT the
`bpeft-step10-data` slug this section originally planned. The dataset was
actually created as `beft-thesis-data` (the display title reads
"bpeft-thesis-data", but the slug lost the "p" — a pre-existing typo in how
Kaggle derived it; slugs are frozen once a dataset has data). Verified
present and populated via the Kaggle API on 2026-08-03 (see 4.2 for its
real structure, confirmed to differ from what this section originally
specified). There is also a leftover duplicate, `bpeft-thesiss-data` (extra
"s"), same byte size — likely an accidental double-create; not used by any
notebook, flagged for the user to clean up or ignore.

Attach the same dataset to all three notebooks / all three accounts — no
need to split it per notebook.

### 4.2 Actual uploaded structure (verified via the Kaggle API, 2026-08-03)

```
bpeft-data/                          <- top-level folder inside the dataset
├── cifar-100-python/                <- meta, train, test (torchvision layout)
├── svhn/
│   └── test_32x32.mat
├── tinyimagenet/
│   └── tiny-imagenet-200/
│       └── tiny-imagenet-200/       <- Kaggle's auto-unzip double-wrapper;
│                                       has wnids.txt, train/<wnid>/images/,
│                                       val/, test/images/ (standard layout,
│                                       train+test images spot-checked, val/
│                                       and wnids.txt inferred from the
│                                       dataset's total byte size matching
│                                       the full ~120k-image archive)
├── miniimagenet/
│   ├── mini-imagenet-cache-train.pkl        (1,145,461,190 B)
│   ├── mini-imagenet-cache-validation.pkl   (292,661,258 B)
│   └── mini-imagenet-cache-test.pkl         (353,647,539 B)
└── splits/                          <- unused by the notebooks; cifar_fs_split.json
    ├── cifar_fs_split.json             + mini_imagenet_split.json are regenerated
    └── mini_imagenet_split.json        fresh by scripts/build_*_split.py instead
```

This differs from the original plan in two ways, both fine as-is (no
re-upload needed):
- No `cifar100/` wrapper directory — `cifar-100-python/` sits directly
  under the dataset's top-level `bpeft-data/` folder.
- MiniImageNet shipped as the 3 Zenodo pkl caches (learn2learn record
  7978538), not the `.npy`/`.json` 84×84 format this section originally
  assumed. `src/datasets/mini_imagenet.py`'s `_find_zenodo_pkls` already
  supports this layout natively (it is in fact the SAME cache format the
  module downloads from Zenodo itself if nothing is staged) — the file
  sizes above match `_ZENODO_FILES` in that module exactly, byte for byte.

The `tinyimagenet/tiny-imagenet-200/tiny-imagenet-200/` double-wrapper
confirms the note below still applies — Kaggle's auto-unzip really does add
an extra layer here, same as observed in Step 9.

**Two things to watch (learned in Step 9, still true):**
- Kaggle auto-unzips `.zip` uploads and sometimes adds an extra wrapper
  folder (confirmed above for TinyImageNet). The notebooks' staging cell
  (Section 4.4) now handles this by reusing the pipeline's own
  any-depth staged-path finders instead of hardcoding a fixed depth, so no
  manual path-pasting is needed even if Kaggle's mount nesting changes
  between sessions.
- Pkl / mat / json files are **not** zips — they were uploaded as-is.

### 4.3 Producing the 6 MiniImageNet cache files — NOT NEEDED

Originally planned as a manual pre-processing step, but moot: the dataset
already ships the 3 Zenodo pkl caches directly (see 4.2), which
`src/datasets/mini_imagenet.py` reads natively. Skip this section.

### 4.4 What each notebook does with these paths

Symlinks them into the repo's `data/` folder (not copies — instant, saves
~2 GB of the 20 GB `/kaggle/working` budget), by calling the exact same
staged-path finder functions `scripts/train.py`/`evaluate.py` use at
runtime (`_find_staged_cifar100_root`, `_find_staged_svhn_root`,
`_find_extracted_tin_root`, `_find_zenodo_pkls`) rather than a hardcoded
`ROOT`/`DATA_PATHS` dict — this way the symlink cell always matches
whatever those modules would discover themselves, regardless of the exact
Kaggle mount depth or unzip wrapping in a given session:

```
data/cifar-100-python                       -> <discovered>/cifar-100-python
data/svhn/test_32x32.mat                    -> <discovered>/test_32x32.mat
data/tiny-imagenet-200                      -> <discovered tiny-imagenet-200 root>
data/mini-imagenet-cache-{train,validation,test}.pkl -> <discovered>/...
```

This matters for speed: `src/datasets/mini_imagenet.py`,
`src/datasets/svhn_ood.py`, `src/datasets/cifar_fs.py`, and
`src/datasets/tinyimagenet_ood.py` all check `data_root` first and only
fall back to walking `/kaggle/input`. Without symlinks, all 120 eval runs
would each recursively glob TinyImageNet's ~120,000 files. If a symlink
can't be created, the cell falls back to a real copy.

---

## 5. Three notebooks instead of one

Splitting by (dataset, shots) makes each notebook independently debuggable
and runnable on a separate Kaggle account. The grid factors into five
24-run buckets, grouped as follows:

| Notebook | Buckets | Runs | Est. GPU-h |
|---|---|---|---|
| **10a — CIFAR-FS, 5-shot** | cifar_5shot_peft (24) + baselines_5shot (12) | 36 | ~9h |
| **10b — CIFAR-FS, 1-shot** | cifar_1shot_peft (24) + baselines_1shot (12) | 36 | ~9h |
| **10c — MiniImageNet, both shots** | mini_5shot_peft (24) + mini_1shot_peft (24) | 48 | ~12h |

Total: 120, matches exactly.

**Why this split:**

- **10a and 10b are CIFAR-FS-only** — infrastructure risk already retired
  by Steps 4–8. Safe to run in parallel on two accounts; if one session
  dies you know it's a quota/timeout issue, not a new bug.
- **10c is entirely MiniImageNet** — this is where the real new risk lives
  (LoRA on MiniImageNet never run; Zenodo-pkl cache path only lightly
  exercised). Isolating it means a MiniImageNet-specific bug only breaks
  10c.
- Matches `implementation.txt` §10's own risk note ("if compute runs out,
  missing cells must be on MiniImageNet") — 10c is the one to drop or run
  last if short on time, and the only one that doesn't touch the
  proposal's non-negotiable baselines.
- **10a must run first no matter what** — it alone satisfies the
  proposal's mandatory minimum (CIFAR-FS + Full-FT + Linear-Probe
  baselines, 5-shot).
- Inside 10c, run `mini_5shot_peft` before `mini_1shot_peft` — 5-shot is
  the more important claim, so a timeout still leaves the better half done.

Since the three buckets are disjoint on (dataset, shots), their
`results/grid/*.json` and `checkpoints/*.pt` never collide — all three can
run simultaneously with zero coordination.

---

## 6. Files to build

### 10.1 — Six new "parent" configs (`configs/`)

Four grid combinations have no parent config yet. New files:

```
configs/exp_phase5_mbnet_lora_{evidential,softmax}.yaml       (mbnet × LoRA × CIFAR-FS)
configs/exp_phase5_mini_lora_{evidential,softmax}.yaml        (r18 × LoRA × MiniImageNet)
configs/exp_phase5_mini_mbnet_lora_{evidential,softmax}.yaml  (mbnet × LoRA × MiniImageNet)
```

Full parent map (every grid cell must `extends:` one of these, never
retype a recipe):

| dataset | backbone | adapter | extends |
|---|---|---|---|
| cifar_fs | r18 | bottleneck-parallel | `configs/exp_phase3_placement_parallel_{evidential,softmax}.yaml` |
| cifar_fs | r18 | lora | `configs/exp_phase3_lora_{evidential,softmax}.yaml` |
| cifar_fs | mbnet | bottleneck-parallel | `configs/exp_phase5_mbnet_parallel_{evidential,softmax}.yaml` |
| cifar_fs | mbnet | lora | **NEW** |
| mini | r18 | bottleneck-parallel | `configs/exp_phase5_mini_parallel_{evidential,softmax}.yaml` |
| mini | r18 | lora | **NEW** |
| mini | mbnet | bottleneck-parallel | `configs/exp_phase5_mini_mbnet_parallel_{evidential,softmax}.yaml` |
| mini | mbnet | lora | **NEW** |
| cifar_fs | r18 | full_ft / linear_probe | `configs/exp_phase3_{full_ft,linear_probe}_{evidential,softmax}.yaml` |

### 10.2 — `scripts/build_grid_configs.py` + `configs/grid/*.yaml`

Generates all 120 tiny YAMLs from the parent map above (one generator, not
three — the notebooks filter at run time instead, see 10.3). Each grid
config overrides only 5 keys:

```yaml
extends: ../exp_phase5_mini_parallel_evidential.yaml
seed: 43
dataset:
  k_shot: 1
output:
  run_tag: grid_mini_1shot_r18_parallel
  results_dir: results/grid
wandb:
  group: mini-1shot
  tags: [phase5, step10, grid]
```

Naming: `configs/grid/{dataset}_{k}shot_{backbone}_{adapter}_{head}_seed{seed}.yaml`

Also writes `configs/grid/_index.json` — a machine-readable map of every
cell:

```json
{"cells": [{"dataset": "mini_imagenet", "k_shot": 1, "backbone": "resnet18",
            "adapter": "bottleneck_parallel", "head": "evidential", "seed": 43,
            "config": "configs/grid/...yaml",
            "results_suffix": "grid_mini_1shot_r18_parallel_seed43",
            "results_json": "results/grid/grid_mini_1shot_r18_parallel_seed43_bottleneck_prototype-evidential_metrics.json",
            "checkpoint": "checkpoints/model_phase2_grid_mini_1shot_r18_parallel_prototype-evidential_seed43.pt",
            "priority": 4}]}
```

**Why the index matters:** per `scripts/evaluate.py`'s `_checkpoint_tag` /
result-filename logic, the output filename is
`{results_suffix}_{adapter.type}_{head_descriptor}` — it does **not**
encode dataset, shots, backbone, or seed. And the metrics-JSON schema is
**frozen** (hard constraint in `instructions.txt` — adding a key breaks
every earlier config's byte-identical rerun). So the grid axes must live in
this sidecar index; the aggregator (10.4) reads the index, never the
filename or the JSON schema.

### 10.3 — `scripts/run_mvt_grid.py` (+ thin `.sh` wrapper for the named deliverable)

`implementation.txt` §10.1 names a `.sh`, but Step 9 found subprocess
output can silently vanish on hosted notebooks — build the Python driver,
keep a 3-line `.sh` wrapper on top.

Flags:
- `--resume` — skip any cell whose results JSON already exists
- `--priority` — run in priority order (see table below)
- `--only dataset=cifar_fs,shots=5` — filter for a specific notebook/session
- `--max-minutes 660` — stop cleanly before Kaggle's 12h wall
- `--dry-run` — print the plan, run nothing
- `--keep-checkpoints seed42` — delete non-seed-42 checkpoints after their
  eval completes, to manage disk (see §8)

Logs one JSON line per run to `results/grid/_run_log.jsonl` (cell id, wall
time, `best_val_epoch`, collapse-guard status, exit code).

One driver script, three invocations (no per-notebook code needed):

```bash
# 10a
python scripts/run_mvt_grid.py --resume --only "dataset=cifar_fs,shots=5"
# 10b
python scripts/run_mvt_grid.py --resume --only "dataset=cifar_fs,shots=1"
# 10c
python scripts/run_mvt_grid.py --resume --only "dataset=mini_imagenet" --priority
```

Priority order overall (so a compute shortfall drops the least important
cells first, per `implementation.txt` §10 "RISKS"):

| P | What | Runs | Est. |
|---|---|---|---|
| 1 | CIFAR-FS 5-shot, PEFT + baselines | 36 | ~9h |
| 2 | MiniImageNet 5-shot, PEFT | 24 | ~6h |
| 3 | CIFAR-FS 1-shot, PEFT + baselines | 36 | ~9h |
| 4 | MiniImageNet 1-shot, PEFT | 24 | ~6h |

### 10.4 — Aggregation (run ONCE, after 10a+10b+10c all finish; not a notebook)

Once all three artifact zips exist, merge them into one local `results/grid/`
folder and run:

```bash
python scripts/aggregate_grid.py      # -> results/mvt_results.json
python scripts/make_master_tables.py  # -> 3 tables, LaTeX + 300dpi PNG
python scripts/grid_plots.py          # -> 16 plots @ 300dpi
```

This needs no GPU, so it can run on a plain CPU runtime (Colab CPU or
locally) rather than Kaggle.

`scripts/aggregate_grid.py` reads `configs/grid/_index.json`, loads each
metrics JSON, aggregates the 3 seeds into mean / std / ci95.

Schema: `{dataset}.{kshot}.{backbone}.{adapter}.{head}.{metric} = {mean, std, ci95, per_seed, n_seeds}`

Metrics carried (reusing the exact key names `scripts/step9_dataset_compare.py`
already reads — no new key names invented): `accuracy_mean`,
`accuracy_ci95`, `f1_macro_mean`, `ece_pooled`, `ece_ts`, `brier_mean`,
`n_params`, `best_val_epoch`, and `ood_auroc__{pool}__{score}` /
`fpr_at_95_tpr__{pool}__{score}` for pools
`{svhn, cifar100_near / mini_val_near, tin_near, gaussian}` and scores
`{vacuity, msp, ts_msp, energy}`.

Also emits a `"missing_cells"` list — "no missing cells" needs to be a
*measurement*, not an assumption.

`n_params` and `best_val_epoch` are carried deliberately: Step 11 (Pareto)
needs the former; the KL-warm-up diagnosis (§3) needs the latter.

**Free reproducibility win:** the grid's seed-42 cells for
(cifar_fs, 5-shot, r18, bottleneck-parallel, {evidential, softmax}) use the
exact same recipe as the already-committed Step 6 `phase4_parallel_*`
results. If the generator correctly extends the parent config, those two
grid JSONs should be numerically identical to the committed Step 6 files.
Add that diff as an assertion in `aggregate_grid.py` — it closes the
byte-identical-rerun item that has been open and carried over since Step 8
(`step_writeups/step9.txt` §5), for free.

### 10.5 — `scripts/make_master_tables.py`

Three tables from `mvt_results.json`, each as LaTeX booktabs + 300 dpi PNG:
1. **Accuracy** — with a visually separate "Baselines" block (Full-FT,
   Linear-Probe) at the bottom.
2. **Calibration** — ECE, ECE-TS, Brier.
3. **OOD AUROC** — SVHN (far) + TinyImageNet (near), evidential-vacuity vs
   softmax-MSP vs TS-MSP vs energy.

### 10.6 — `scripts/grid_plots.py` ("16 plots")

Note for the writeup: the exit criterion phrase "reliability + OOD
histogram per dataset × head" is only 4 combinations; the actual 16 =
dataset (2) × shots (2) × head (2) × plot type (2). Flag this reading
explicitly in `step_writeups/step10.txt` so it isn't read as a silently
changed criterion.

`scripts/evaluate.py` already writes a reliability + OOD histogram PNG per
run; this script picks the representative cell per (dataset, shot, head)
— best adapter, seed 42 — and re-renders at 300 dpi.

### 10.7 — `tests/test_grid_configs.py`

Offline, no GPU. Asserts:
- exactly 120 configs generated, matching `_index.json`
- every `run_tag`, `results_suffix`, checkpoint path, and results path is unique
- zero collisions with any existing file under `results/` (hard constraint:
  nothing Step 10 writes may overwrite a Steps 1–9 result)
- every config loads and `build_model(cfg)` succeeds; param counts match
  expected values per (backbone, adapter)
- the seed-42 grid cells resolve to a config identical to their Step
  5/6/8/9 parent apart from the 5 overridden keys

### 10.8 — `step_writeups/step10.txt`

Same shape as `step_writeups/step9.txt`: §0 paper-grounded reasoning, §1
what was built, §2 param counts, §3 how to reproduce, §4 results
(transcribed verbatim from `mvt_results.json`, never retyped), §5 honest
caveats, §6 exit criteria.

---

## 7. The three notebooks

Lean by design — Step 9's notebook is 39 cells; each of these is **~8 code
cells**. Explanation lives in `step_writeups/step10.txt` and this plan, not
in the notebook.

`notebooks/step10a_cifar_5shot.ipynb`, `notebooks/step10b_cifar_1shot.ipynb`,
`notebooks/step10c_mini.ipynb` — identical skeleton, differing only in the
`--only` filter passed to `run_mvt_grid.py` and (10c only) the extra
MiniImageNet symlink/cache-check.

| # | Type | Content |
|---|---|---|
| 0 | markdown | Settings: GPU T4, Internet ON, attach `beft-thesis-data`. Link to `step_writeups/step10.txt`. |
| 1 | code | GPU check + clone repo + `pip install -r requirements.txt` |
| 2 | code | Reuse `_find_staged_*`/`_find_zenodo_pkls` (§4.4) → symlink into `data/` → print OK/MISSING per item |
| 3 | code | `!python scripts/build_cifar_fs_split.py && python scripts/build_mini_imagenet_split.py` |
| 4 | code | `!python scripts/build_grid_configs.py && python -m pytest -q tests/test_grid_configs.py` |
| 5 | code | **Smoke test** (§2) — 10a only, first run; skippable after |
| 6 | code | `!python scripts/run_mvt_grid.py --resume --only "<notebook's filter>" --priority --max-minutes 660` |
| 7 | code | Pack + push this notebook's slice only (`step10a_cifar_5shot_artifacts.zip`, etc.) — same pattern as Step 9's Section 9/9b |

Cell 6 is the same command every session per notebook; re-running top to
bottom is safe because `--resume` skips everything already done.

Run order: **10a first** (satisfies the proposal's mandatory minimum on
its own) → 10b and 10c can run in parallel on separate accounts after.

---

## 8. Compute budget and schedule

Step 9 measured 10 runs in 142 min on a Kaggle T4 ≈ 13.4 min/run (excluding
cache build). Full-FT is heavier; 1-shot is slightly lighter.

- Estimate: 24–36 GPU-hours total, call it ~30h.
- Kaggle free tier: 30 GPU-h/week, 12h max session.
- → 4–6 sessions across ~2 weeks if run on one account; fewer calendar days
  if 10b/10c run in parallel on separate accounts.

| Session | Target | Runs |
|---|---|---|
| 1 (10a) | Setup + smoke test + CIFAR-FS 5-shot (PEFT + baselines) | 36 |
| 2 (10b) | CIFAR-FS 1-shot (PEFT + baselines) | 36 |
| 3 (10c, part 1) | MiniImageNet 5-shot | 24 |
| 4 (10c, part 2) | MiniImageNet 1-shot | 24 |
| 5 | Side-study 10.9 + aggregation (§10.4) + tables + writeup | — |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Disk: 120 checkpoints × ~45 MB (state_dict includes frozen backbone) ≈ 5–6 GB, plus heavier Full-FT checkpoints. `/kaggle/working` is 20 GB. | `--keep-checkpoints seed42`: keep only seed-42 checkpoints (needed for the byte-identical rerun check); delete seeds 43/44 after their eval completes. |
| Filename collisions overwriting Steps 1–9 results | `tests/test_grid_configs.py` asserts zero collisions against the existing `results/` listing before any run starts. |
| Metrics-JSON schema is frozen | Aggregation reads `_index.json`, never adds a key to any metrics file. |
| Kaggle session timeout mid-grid | `--resume` + `--max-minutes 660`; push artifacts every session. |
| W&B exit criterion (dashboard grouped by dataset+shots, baselines tagged) | Needs `WANDB_API_KEY` in Kaggle Secrets and `wandb.disabled: false` in generated grid configs. Fall back to `mode: offline` + `wandb sync` later if the key is unavailable. |
| 1-shot / LoRA-on-mbnet / LoRA-on-mini untested | Pre-flight smoke test (§2), run inside 10a before the full grid starts. |
| Splitting across 3 notebooks / accounts could double-run or skip cells | Buckets are disjoint on (dataset, shots) by construction — no coordination needed between notebooks; `_index.json` + `--only` guarantee no overlap. |

---

## 10. Deliverables checklist

```
[ ] 10.0   Pre-flight smoke test (1-shot, LoRA×mbnet, LoRA×mini) — WRITTEN into
           notebook 10a §4; NOT YET RUN (needs a real Kaggle GPU session)
[x] 10.1   6 new parent configs (LoRA × {mbnet, mini, mini+mbnet}) — DONE
[x] 10.2   scripts/build_grid_configs.py + configs/grid/*.yaml (120) + _index.json — DONE
[x] 10.3   scripts/run_mvt_grid.py (+ .sh wrapper) — resumable, priority-ordered, time-capped — DONE
[x] 10.4   scripts/aggregate_grid.py → results/mvt_results.json (+ missing_cells) — CODE DONE,
           smoke-tested against synthetic fixtures; run once for real after 10a+10b+10c finish
[x] 10.5   scripts/make_master_tables.py → 3 tables, LaTeX + 300 dpi PNG — CODE DONE, smoke-tested
[x] 10.6   scripts/grid_plots.py → 16 plots @ 300 dpi — CODE DONE, smoke-tested
[x] 10.7   tests/test_grid_configs.py — DONE, 19/19 passing (233/233 full suite)
[ ] 10.8a  notebooks/step10a_cifar_5shot.ipynb  (36 runs, ~9h, RUN FIRST) — WRITTEN, NOT YET RUN
[ ] 10.8b  notebooks/step10b_cifar_1shot.ipynb  (36 runs, ~9h) — WRITTEN, NOT YET RUN
[ ] 10.8c  notebooks/step10c_mini.ipynb         (48 runs, ~12h, drop first if short on time) — WRITTEN, NOT YET RUN
[ ] 10.9   VAL-only side-study: KL warm-up / patience for mbnet-parallel-evidential — NOT STARTED
[~] 10.10  step_writeups/step10.txt — §0-3/5-6 done, §4 Results an explicit unfilled template
[~] 10.11  progress.txt updated (Step 10 boxes + CURRENT STATUS block) — done for the
           plumbing-complete state; needs a final update once the 120 runs finish
```

Status as of 2026-08-02: **plumbing complete, execution pending** — no GPU/Kaggle
was available this session, so every item above was built and offline/smoke-tested
(against synthetic fixtures, never real data) but none of the 120 grid cells have
actually run. Full detail: `step_writeups/step10.txt`'s STATUS header.

---

## Open decision, confirmed

**Grid recipe: FROZEN across all 120 runs** (not per-cell VAL-tuned). Tuning
is handled as a bounded, separate side-study (§3, item 10.9) so it doesn't
contaminate the headline table and so 10a/10b/10c stay independently
runnable.
