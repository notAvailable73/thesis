# Task: Complete RQ2/RQ4 checkpoint coverage + compute RQ1's two-way interactions

**Audience:** whoever on the team has a Kaggle/Colab GPU session available. This assumes familiarity
with the repo's existing Phase A / RQ pipeline (`scripts/rq_core.py`, `scripts/rq_drivers.py`,
`scripts/rq_aggregate.py`) but not prior knowledge of these two specific gaps — everything needed is
below.

**Read `CLAUDE.md`, `instructions.txt` and `progress.txt` first if you haven't touched this repo
before.** The rules that matter most for this task specifically:

- Never regenerate a file marked "DO NOT REGENERATE — frozen" (`configs/test_episodes.yaml`,
  `configs/val_episodes.yaml`, `data/cifar_fs_split.json`).
- Hyperparameter/selection decisions use `configs/val_episodes.yaml` (seeds 10000–10099) only, never
  the 600 frozen test seeds (0–599).
- Never commit to git unless explicitly asked — a human reviews and commits.
- `results/grid/*.json` and `results/mvt_results.json` are the **regression baseline**. Nothing in this
  task should overwrite them; the whole point of the "regression guard" described below is to prove new
  runs reproduce them, not replace them.

There are two independent workstreams here. **They do not depend on each other** — do them in either
order, or split them between two people.

---

## Part A — Retrain/recover the 21 missing checkpoints (closes RQ2/RQ4 coverage gap)

### A.0 — What's actually missing, and why it matters

The Step 10 grid trained all **120/120** cells successfully — accuracy, ECE, and native-score AUROC for
every cell are complete in `results/mvt_results.json` (`missing_cells: []`, `n_cells_present: 120`).
**This task does not touch that file and does not need to.**

What's incomplete is a *separate* later phase (Phase A, `results/rq_factorial/`) that re-scores each
grid model under **all four** OOD scores (MSP, TS-MSP, energy, vacuity) regardless of which score it was
originally trained/evaluated with, and refits the evidential head's two-parameter evidence affine on
VAL. Phase A needs the **saved model weights** (`.pt` files), not just the metrics — and only 99 of 120
were found when Phase A ran. The other 21 were never located.

I pulled the exact list from `results/rq_checkpoint_audit.json` (`missing_cells`, `n_present: 99`,
`n_total: 120`, `verdict: "b-partial"`). All 21 are CIFAR-FS 5-shot adapter models:

```
configs/grid/cifar_5shot_r18_parallel_evidential_seed{42,43,44}.yaml
configs/grid/cifar_5shot_r18_parallel_softmax_seed{42,43,44}.yaml
configs/grid/cifar_5shot_r18_lora_evidential_seed{42,43,44}.yaml
configs/grid/cifar_5shot_r18_lora_softmax_seed{42,43,44}.yaml
configs/grid/cifar_5shot_mbnet_parallel_evidential_seed{42,43,44}.yaml
configs/grid/cifar_5shot_mbnet_parallel_softmax_seed{42,43,44}.yaml
configs/grid/cifar_5shot_mbnet_lora_evidential_seed{42,43,44}.yaml
```

That's 7 (backbone × adapter × head) combinations × 3 seeds = 21. The 8th combination in that same
slice — `cifar_5shot_mbnet_lora_softmax` (all 3 seeds) — **was** recovered; it's the one arm of the
"cifar_fs/5shot/mobilenetv3_small/lora" cell that the RQ2 write-up flags as having softmax but not
evidential. Closing this gap fixes that specific instability at its root (see A.5).

All 21 `.yaml` configs still exist as committed files — verified, `ls` confirms both endpoints of the
list are present. **No config regeneration needed**, only training + re-scoring.

**Why this matters for the deck/thesis:**
- Removes the "163×/394× unstable ratio" caveat on RQ2 (Slide 20) — the instability is caused by exactly
  this gap (one wholly-missing objective×score cell), and closing it makes the η² ratio well-defined
  again rather than something to hedge about.
- Removes limitation 8 ("RQ2 and RQ4 cover 99 of 120 models, non-randomly, concentrated in the
  most-quoted slice").
- Extends RQ4's refit coverage into cells it currently has no data for.

### A.1 — Step 0: try recovery before retraining

**Do this first — it's minutes, not GPU-hours, and it might make Part A unnecessary.**

`scripts/rq_drivers.py::recover_checkpoints(repo_root, search_roots=("/kaggle/input",))` scans given
paths for either the original Step 10 artifact zips (`results/step10{a,b,c}_*_artifacts.zip` — these
exist per the guide but are **not in git**, ~2 GB total, so they only exist wherever they were originally
saved: a Kaggle dataset, Google Drive, or a local machine) or loose `model_phase2_*.pt` files, and copies
whatever it finds into `checkpoints/` without touching committed `results/`.

```python
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
import rq_drivers as D

# If you have the step10 zips attached as a Kaggle dataset, or mounted from Drive,
# point search_roots at wherever they land, e.g. ("/kaggle/input", "/content/drive/MyDrive/bpeft_data")
result = D.recover_checkpoints(Path("."), search_roots=("/kaggle/input",))
print(result)  # {"zips_seen": [...], "copied": [...], "already_present": [...]}
```

Then re-run the audit to see what's still missing:

```python
missing_after = D.audit_checkpoints(Path("."))
print(missing_after["n_present"], "/", missing_after["n_total"])
```

**If this finds all 21** — skip straight to A.4 (re-run Phase A on the newly-present cells; no training
needed, `factorial_run_one` will just score them). **If it finds some but not all** — proceed to A.2 for
the remainder. **If it finds none** — the zips genuinely aren't in whatever storage you have attached;
check with whoever ran the original Step 10 notebooks (`step10a_cifar_5shot.ipynb`,
`step10b_cifar_1shot.ipynb`, `step10c_mini.ipynb`) for where those zips live before assuming a full
retrain is the only option.

### A.2 — If retraining is needed

The existing Phase A driver already does "train if missing, then score" as one step —
`run_phase_a(..., allow_retrain=True)` calls `train_cell()` internally whenever a cell's checkpoint file
doesn't exist, then immediately runs `factorial_run_one()` on the freshly-trained model. **You do not
need to write a separate training loop.**

Build the 21-cell list by filtering `configs/grid/_index.json`'s `cells` array against the config paths
above:

```python
import json
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
import rq_drivers as D

repo_root = Path(".")
grid_index = json.load(open(repo_root / "configs/grid/_index.json"))["cells"]

MISSING_CONFIGS = {
    "configs/grid/cifar_5shot_r18_parallel_evidential_seed42.yaml",
    "configs/grid/cifar_5shot_r18_parallel_evidential_seed43.yaml",
    "configs/grid/cifar_5shot_r18_parallel_evidential_seed44.yaml",
    "configs/grid/cifar_5shot_r18_parallel_softmax_seed42.yaml",
    "configs/grid/cifar_5shot_r18_parallel_softmax_seed43.yaml",
    "configs/grid/cifar_5shot_r18_parallel_softmax_seed44.yaml",
    "configs/grid/cifar_5shot_r18_lora_evidential_seed42.yaml",
    "configs/grid/cifar_5shot_r18_lora_evidential_seed43.yaml",
    "configs/grid/cifar_5shot_r18_lora_evidential_seed44.yaml",
    "configs/grid/cifar_5shot_r18_lora_softmax_seed42.yaml",
    "configs/grid/cifar_5shot_r18_lora_softmax_seed43.yaml",
    "configs/grid/cifar_5shot_r18_lora_softmax_seed44.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_evidential_seed42.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_evidential_seed43.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_evidential_seed44.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_softmax_seed42.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_softmax_seed43.yaml",
    "configs/grid/cifar_5shot_mbnet_parallel_softmax_seed44.yaml",
    "configs/grid/cifar_5shot_mbnet_lora_evidential_seed42.yaml",
    "configs/grid/cifar_5shot_mbnet_lora_evidential_seed43.yaml",
    "configs/grid/cifar_5shot_mbnet_lora_evidential_seed44.yaml",
}

cells_to_run = [c for c in grid_index if c["config"] in MISSING_CONFIGS]
assert len(cells_to_run) == 21, f"expected 21, got {len(cells_to_run)} — re-check the audit list"
```

Then run Phase A on just this subset (reusing whatever `device`, `out_dir`, `logits_dir` conventions the
original Phase A notebook (`notebooks/New_1,2,5A.ipynb`) used — open that notebook and copy its exact
`run_phase_a(...)` call, just swap `cells=cells_to_run`):

```python
counts = D.run_phase_a(
    repo_root, cells_to_run,
    device="cuda",                       # match the original notebook's device setup
    out_dir=repo_root / "results" / "rq_factorial",
    logits_dir=repo_root / "results" / "rq_factorial_logits",   # or None if the original didn't save these
    num_episodes=600,
    allow_retrain=True,                  # <-- the key flag: trains the cell first if no checkpoint exists
    wandb_mode="disabled",
    max_minutes=None,                    # or set a budget if your session has a time limit
)
print(counts)  # {"ok": n, "skipped_done": n, "no_checkpoint": n, "error": n, "trained": n}
```

**Expect `counts["trained"] == 21` and `counts["ok"] == 21`** (or fewer `trained` if A.1's recovery
found some already). Anything in `counts["error"]` needs individual investigation — check
`results/rq_factorial/_run_log.jsonl` for the specific exception per cell.

### A.3 — Time/cost estimate

Per the guide, the original Step 10 grid trained 120 cells in 36.3 GPU-hours ≈ 18 min/cell average, but
CIFAR-FS 5-shot cells specifically ran faster (fewer episodes to converge than 1-shot in some
configurations — check `step_writeups/step10.txt` if you want per-slice timing). A safe estimate for 21
cells: **~5–7 GPU-hours** on a T4, consistent with the number already quoted in the deck's limitation 8.
Scoring each freshly-trained model under all 4 metrics (`factorial_run_one`) is fast (seconds per cell,
it's inference not training) — the training is the cost.

### A.4 — Verify against the regression baseline

`factorial_run_one` (inside `run_phase_a`) automatically compares each newly-trained/scored cell against
the **already-committed** `results/grid/*.json` metrics for that exact config+seed (the
`committed_metrics` argument) and logs a `regression_guard` block:

```
guard: <status> (<n_exact>/<n_keys> exact, max|diff|=<value>)
```

**This is the check that matters most.** The whole project's reproducibility claim rests on "same config
+ same seed → same numbers." If any of the 21 newly-trained cells show a nonzero `max_abs_diff` against
the original grid's committed accuracy/ECE, that's a real finding to report, not something to paper
over — it would mean the retrained cell used a different environment/library version than the original
Aug 2–6 session, and the deck's "byte-identical reproducibility" claim (which currently rests on the
RQ3 matched-budget experiment's 18/18 exact re-runs) would need to be re-examined for this slice too.
Expect exact matches (this project's track record on this check is 18/18 and 99/99 in the two prior
instances), but **check, don't assume.**

### A.5 — Regenerate RQ2's aggregation and update the deck

Once `results/rq_factorial/` has all 120 (or however many you recovered) cells:

1. Re-run the aggregation cell(s) in `notebooks/New_1,2,5A.ipynb` (or call
   `scripts/rq_aggregate.py`'s functions directly — `load_records`, `factorial_observations`,
   `rq1_verdict` — note the retired-numbering confusion: the function named `rq1_verdict` computes the
   **objective-vs-score** decomposition, i.e. what the deck calls RQ2. See `docs/guide/03_results.md`
   §3.8 "How the three numbering schemes map" if this is confusing.)
2. **Before trusting the new ratio**, fix the aggregator bug that caused the original instability (see
   Part B's note on this same function — `eta_squared()`'s `balanced` flag only compares *counts* of
   present cells, so it won't notice if the full 8-arm objective×score grid is now complete unless you
   also check that no cell is silently still at `n=0`). With all 21 recovered, this should no longer be
   an issue, but confirm `balanced: True` appears in the new `rq1_verdict` (=RQ2) output before quoting
   a new ratio.
3. Recompute the far-OOD and near-OOD score/objective η² shares and the ratio. Expect them to move
   slightly from 43.7%/0.27% (far) and 13.0%/0.60% (near) — report the **new** numbers, don't just note
   they moved.
4. Update `docs/DEFENCE_DECK_25_SLIDES.md` Slide 20: the "Coverage" caveat box currently says "computed
   on 99 of 120 recoverable checkpoints... 21 missing checkpoints are not a random sample." Once this is
   closed, that caveat becomes historical — replace it with the new coverage (120/120, or however many
   you actually recovered) and the new stable ratio, and remove the now-resolved "163×/394×" instability
   note since the root cause is fixed. Also update `docs/DEFENCE_DECK_25_SLIDES.md` limitation 8 on
   Slide 24, `docs/RQ_SUPERVISOR_REPORT.md` (flagged stale in its own `.pdf` per A1/A2 in
   `docs/guide/05_problems_and_open_work.md`), and `docs/RQ_RESULTS_SUMMARY.md`.
5. Check whether RQ4's refit coverage (`results/rq_summary.json` → `rq2_rows`/`rq2_verdict`, again
   retired-numbering — this is the deck's RQ4) also gained new evidential cells from this recovery, and
   whether the 48/48-improved, 150/192-preserved headline numbers move. If they do, update Slide 22 the
   same way.

### A.6 — Do not skip

- **Never touch `configs/test_episodes.yaml` or `configs/val_episodes.yaml`.** They're frozen; this task
  doesn't need to regenerate them.
- **Every newly-trained cell uses the exact same recipe as the original grid** — same seed, same
  hyperparameters (the config file is unchanged, so this is automatic as long as you use `train_cell()`
  on the existing `.yaml`, not a hand-edited copy).
- **Don't commit the new checkpoints or raw archives to git** — per the repo's own note, the original
  Step 10/Phase A raw archives (`results/new_rqs_results.zip`, `results/rq3_matched_checkpoints.zip`,
  etc.) are deliberately not in git (multi-GB). Follow the same pattern: keep new `.pt` files in
  `checkpoints/` (already gitignored per the existing pattern) and only commit the small per-cell JSON
  outputs in `results/rq_factorial/` if/when a human reviews and asks for a commit.

---

## Part B — Compute RQ1's two-way interaction terms

### B.0 — Why this exists as a task at all

RQ1's headline five-factor decomposition (Slide 19: shots → 76.05% of accuracy variance, head → 82.89%
of ECE variance, etc.) is a **main-effects-only** decomposition. It cannot see effects that only show up
when two factors combine — and RQ3's entire finding (the adapter-budget ordering flips depending on
which backbone you're on) is exactly that kind of effect. The current deck states this as a limitation
rather than a computed number (Slide 19: "the two-way interactions are a named item for future work, not
a number claimed here").

**Important finding from inspecting the code:** this is *not* a new statistics implementation task. The
function that computes the main-effects table, `eta_squared()` in `scripts/rq_aggregate.py`, **already
computes every pairwise two-way interaction term by default** (`interactions: bool = True` is the
function's default, and the body already loops `for a, b in combinations(factors, 2)` and computes an
interaction sum-of-squares for every pair — see lines 85–133). The main-effects table shown on Slide 19
was almost certainly produced by calling this function and then only printing the single-factor terms
from its output, discarding the `"adapter:backbone"`, `"adapter:head"`, etc. keys it already returns.

**So this task is: locate/rebuild the 96-row per-seed table, call the existing function correctly, and
report what it already computes** — not write new analysis code.

### B.1 — Data source

`results/mvt_results.json` has the complete, already-verified grid (120/120 cells, `missing_cells: []`).
The 32-cell × 3-seed = 96-row subset RQ1 uses **excludes** the 8 Full-FT/Linear-Probe baseline cells
(those aren't part of the balanced 2×2×2×2×2 design — they don't have a comparable "adapter type"
level, and are ResNet-18+CIFAR-FS only, so including them would break the balance the SS-decomposition
needs to be exact).

The nested structure (verified directly, this session) is:

```
results[dataset]["1shot"|"5shot"][backbone][adapter][head] → {
    "accuracy_mean": {"per_seed": {"42": v, "43": v, "44": v}, "mean": ..., "std": ..., ...},
    "ece_per_episode_mean": {"per_seed": {...}, ...},
    ... (similar per-metric sub-dicts, each with its own "per_seed")
}
```

Flatten it into the row format `eta_squared()` expects — one dict per (cell, seed) with the factor
columns plus the outcome value:

```python
import json
from pathlib import Path

d = json.load(open("results/mvt_results.json"))["results"]

BASELINE_ADAPTERS = {"full_ft", "linear_probe"}  # exclude these — not part of the balanced design

def flatten(metric_key: str) -> list[dict]:
    rows = []
    for dataset, by_shot in d.items():
        for shot_key, by_backbone in by_shot.items():
            k_shot = int(shot_key.replace("shot", ""))
            for backbone, by_adapter in by_backbone.items():
                for adapter, by_head in by_adapter.items():
                    if adapter in BASELINE_ADAPTERS:
                        continue
                    for head, leaf in by_head.items():
                        per_seed = leaf[metric_key]["per_seed"]
                        for seed, value in per_seed.items():
                            rows.append({
                                "dataset": dataset, "k_shot": k_shot,
                                "backbone": backbone, "adapter": adapter, "head": head,
                                "seed": int(seed), "value": value,
                            })
    return rows

accuracy_rows = flatten("accuracy_mean")
print(len(accuracy_rows))   # expect 96 (32 cells x 3 seeds)
```

**Check the row count is exactly 96 for accuracy/ECE/far-OOD, and 95 for near-OOD-TinyImageNet** before
trusting anything downstream — the known partial gap (`cifar_fs/5shot/mobilenetv3_small/lora/evidential`
seed 42, TinyImageNet AUROC missing) should show up here as 95, not 96, and `eta_squared()`'s `balanced`
flag should correctly report `False` for that one outcome (it detects reduced-count cells correctly —
the aggregator bug flagged elsewhere in this project is specifically about *wholly absent* cells with
zero observations, not partially-reduced ones, which is a different and narrower failure mode; see the
note in Part A.5).

Confirm which exact metric keys back each of the four deck numbers before running — check
`docs/RESULTS_MASTER.md` §3.11's column definitions (ECE = pooled per-episode, AUROC = native score per
head) if the key names above (`accuracy_mean`, `ece_per_episode_mean`) don't obviously match; the far/near
AUROC values live under different keys per OOD pool (e.g. an `svhn_far`/`tin_near` split) — inspect one
leaf dict's full key list (`list(leaf.keys())`) before writing the four `flatten()` calls.

### B.2 — Run the existing function, don't reimplement it

```python
import sys
sys.path.insert(0, "scripts")
from rq_aggregate import eta_squared

FACTORS = ["dataset", "k_shot", "backbone", "adapter", "head"]

result = eta_squared(accuracy_rows, FACTORS, value="value", interactions=True)
print("balanced:", result["balanced"])
print("n:", result["n"])
for term, share in sorted(result["eta_squared"].items(), key=lambda kv: -kv[1]):
    print(f"  {term:<20} {share:.4f}")
```

The output's `eta_squared` dict will contain **both** the five main-effect keys you already have
(`dataset`, `k_shot`, `backbone`, `adapter`, `head`) **and** all ten pairwise interaction keys
(`"dataset:k_shot"`, `"dataset:backbone"`, ..., `"adapter:head"`, etc. — `combinations(5, 2) = 10`
terms), plus `"residual"`. Repeat for the ECE and both OOD-AUROC row sets.

**The term to look at first is `"adapter:backbone"`.** If a meaningful share of the previously-unexplained
residual (23.87% on far-OOD, 8.95% on accuracy, etc.) moves into that specific term, that is a direct,
quantitative confirmation of RQ3's qualitative finding — the adapter's effect genuinely depends on which
backbone it's paired with, and now there's a number for it instead of just the sign-test evidence RQ3
already has.

### B.3 — A caveat to build into the write-up, not discover in Q&A

The SS-decomposition is **exact for a balanced design** (the function's own docstring says this) — and a
2-level-per-factor, 96-observation design has comparatively few degrees of freedom once you start
carving off ten interaction terms plus five main effects plus a grand mean (16 parameters from 96
observations, before any residual). The main-effect shares are already fairly noisy at this scale (the
project's own robustness check found the residual moves 0.3–3.6 percentage points depending on whether
seeds are averaged first); interaction terms, being smaller and split ten ways instead of five, will be
noisier still. **Do not report an interaction share as more precise than the main-effect shares next to
it** — if anything, flag it as the *more* uncertain of the two, and consider whether a bootstrap
(resample the 3 seeds with replacement within each of the 32 cells, recompute the decomposition, look at
the spread of `"adapter:backbone"` across resamples) is worth doing before quoting a specific percentage
in the thesis text, even though the point estimate itself takes seconds to compute.

### B.4 — What to report, and how to phrase it

Once you have the numbers:

- Update Slide 19's "Stated limitation" box in `docs/DEFENCE_DECK_25_SLIDES.md` — if the
  `adapter:backbone` interaction share is non-trivial, this stops being a stated limitation and becomes
  a **fifth finding**: RQ1's decomposition, extended to two-way terms, directly corroborates RQ3's
  matched-budget result rather than merely being silent about it.
- Use the same language discipline as the rest of the deck: report the **share**, not an inflated
  causal claim. "The adapter:backbone interaction explains X% of far-OOD variance, consistent with RQ3's
  finding that calibration depends on backbone identity" is defensible. "This proves RQ3" is not — RQ3's
  own pre-registered matched-budget experiment is still the stronger, causally-cleaner evidence; this
  interaction term is corroborating, not primary.
- If the interaction share turns out to be **small** (most of the residual stays unexplained even after
  adding all ten two-way terms), that is also a reportable, honest result — it would mean RQ3's effect,
  while real (established by the matched-budget experiment's own much stronger design), doesn't show up
  as a large term in this particular decomposition, possibly because it's a higher-order or
  non-additive effect the linear SS-decomposition isn't well-suited to catch. Say that plainly rather
  than searching for a way to make the number look bigger.

### B.5 — Time estimate

No GPU needed — this is pure post-hoc analysis on already-committed JSON. Once the row-flattening
function is correct (the main place to spend care, given the nested key structure), running
`eta_squared()` on four outcome variables takes seconds. Budget **half a day** total, most of it spent
double-checking the metric-key mapping against `RESULTS_MASTER.md` and deciding whether to bootstrap
per B.3, not on computation time.

---

## Appendix — file/script map for this task

| Thing | Where |
|---|---|
| List of 21 missing configs, ground truth | `results/rq_checkpoint_audit.json` (`missing_cells`) |
| Checkpoint recovery (try before retraining) | `scripts/rq_drivers.py::recover_checkpoints()` |
| Checkpoint audit (re-check coverage after recovery/training) | `scripts/rq_drivers.py::audit_checkpoints()` |
| Train-if-missing + score, one call | `scripts/rq_drivers.py::run_phase_a(..., allow_retrain=True)` |
| Single-cell training only (if you want it separate) | `scripts/rq_drivers.py::train_cell()` |
| Per-cell scoring + regression guard | `scripts/rq_drivers.py::factorial_run_one()` |
| Grid cell index (config/checkpoint/results_json paths) | `configs/grid/_index.json` |
| Original Phase A notebook (copy its exact call conventions) | `notebooks/New_1,2,5A.ipynb` |
| SS-decomposition / η² (main effects **and** interactions) | `scripts/rq_aggregate.py::eta_squared()` |
| RQ2's (confusingly named `rq1_verdict`) aggregation | `scripts/rq_aggregate.py::rq1_verdict()` |
| Full grid's committed per-cell metrics | `results/mvt_results.json` |
| Per-cell Phase A outputs | `results/rq_factorial/*.json` |
| RQ2/RQ4 summary (retired-numbering keys, see note below) | `results/rq_summary.json` |
| Numbering-scheme decoder ring | `docs/guide/03_results.md` §3.8 |
| Pre-registration convention this project follows | `docs/RQ3_MATCHED_BUDGET_PLAN.md` (as a template) |

**Retired-numbering warning, worth repeating:** `results/rq_summary.json`'s keys (`rq1_verdict`,
`rq2_rows`, `rq2_verdict`) use an earlier five-question draft's numbering, not the final four-RQ scheme
the deck and thesis use. `rq1_verdict` in the code = **RQ2** in the deck (objective vs. score).
`rq2_rows`/`rq2_verdict` in the code = **RQ4** in the deck (post-hoc refit). Don't let this cause a
mismatched claim in the write-up — `docs/guide/03_results.md` §3.8 has the full mapping table.

**When you're done with either part, ping back with:** the new coverage count (Part A), the new RQ2
far/near shares and whether the ratio is now stable without the "one cell missing an arm" caveat (Part
A), and the four interaction-term tables plus whether `adapter:backbone` moved meaningfully off the
residual (Part B). Those are the four numbers this task exists to produce.
