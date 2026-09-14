# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A masters thesis codebase: **B-PEFT** (Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with
Lightweight CNN Backbones). It trains a frozen ImageNet-pretrained CNN backbone (ResNet-18 or MobileNetV3-Small) + a
small trainable adapter (parallel Bottleneck / LoRA; Full-FT and Linear-Probe as baselines; BitFit built but not in
the grid) + a parameter-free Prototype head read as softmax or Evidential Dirichlet, evaluated on 5-way {1,5}-shot
episodes (CIFAR-FS, MiniImageNet) for accuracy, calibration (ECE, Brier), and OOD detection (SVHN / Gaussian far-OOD,
CIFAR-100- or MiniImageNet-heldout / TinyImageNet near-OOD).

**Current state (2026-09-14): all experiments are done; the remaining work is thesis writing and the defence.** The
thesis uses a **new four-RQ framing** that replaced the proposal's original questions after a literature check
found those had close precedent. Canonical statement: `docs/RQ_SUPERVISOR_REPORT.md`.

- **RQ1** — how is variance in accuracy / calibration / OOD distributed across design axes? Shot count drives
  accuracy; head type drives calibration.
- **RQ2** — is OOD detection caused by the training objective or the scoring rule? The scoring rule, decisively.
- **RQ3** — does accuracy follow adapter architecture and calibration follow parameter budget? Accuracy follows
  architecture; calibration follows the *backbone* (pre-registered matched-budget experiment). Strongest claim.
- **RQ4** — can the evidential head's calibration be fixed post-hoc without breaking OOD ranking? Yes, 48/48 cells.

The proposal's original questions (placement, evidential calibration, Bayesian near-OOD, latency Pareto) are still
answered and reported as **Orig-RQ1…4**. `progress.txt` and `step_writeups/step1-11.txt` use that *original*
numbering — never mix the two.

Git Repo link - https://github.com/notAvailable73/thesis.git

## Required reading before doing any work here

Read these before touching code or thesis text, and re-read `progress.txt` any time you're unsure what's done:

0. **`docs/guide/`** — plain-language map of the whole project (2026-09-14): research story, results with
   safe wording, every experiment's files and checks (the only write-up of the Phase A/B new-RQ runs), and
   known problems in the other docs (`05_problems_and_open_work.md` — e.g. do not quote "163×").
1. **`docs/RQ_SUPERVISOR_REPORT.md`** — the four RQs, their answers, novelty assessment, limitations, and a
   traceability table mapping every number to a result file.
2. **`thesis_implementation_instructions.txt`** — the _process_ rule for this repo: implementation choices must be
   justified against the paper summaries in `PAPER SUMMARIES/*.txt` (pros/cons/fit reasoning), not implemented from
   general training-data knowledge. If a paper summary and general knowledge conflict, defer to the summary and flag
   it. Do not invent hyperparameters/results not stated in a summary.
3. **`progress.txt`** — status tracker (13 original steps + the RQ3 matched-budget experiment) with a running
   decisions log at the bottom. Single source of truth for "what's done". Uses Orig-RQ numbering.
4. **`docs/RQ_RESULTS_SUMMARY.md`** (full evidence per RQ), **`docs/RESULTS_MASTER.md`** (all 120-run grid
   tables — generated; edit `docs/RESULTS_MASTER_template.md` and run `scripts/make_results_master.py`),
   **`docs/RQ3_MATCHED_BUDGET_PLAN.md`** (RQ3 pre-registration — do not edit §1–6).
5. **Defence material**: `docs/DEFENCE_SLIDE_PLAN.md` (slide-by-slide plan), `docs/DEFENCE_BRIEF.md` ("CNNs are
   outdated" objection; uses Orig-RQ numbering), `docs/CITATION_AUDIT.md` + `docs/refs.bib`.
6. **`step_writeups/`** — dated, historical per-step write-ups explaining _why_ results came out as they did.
   Headline claims in the Step 4.5–9 write-ups were later superseded by the grid (see corrections below); the
   write-ups are kept as a record, not rewritten. `proposal.txt` is the original proposal.

Real entry points are `scripts/train.py` / `scripts/evaluate.py`; the RQ analyses are `scripts/rq_core.py`,
`scripts/rq_drivers.py`, `scripts/rq_aggregate.py`, `scripts/rq3_matched.py`, `scripts/rq5_sweep.py`.

Do not treat this as a normal library-consumer codebase: correctness here means "matches the frozen protocol and is
honestly reported," not just "code runs." Two conventions enforce that:

- **Never regenerate a file marked "DO NOT REGENERATE — frozen"** (e.g. `configs/test_episodes.yaml`,
  `configs/val_episodes.yaml`, `data/cifar_fs_split.json`) — these fix the episode seeds / class splits so results
  stay comparable across runs and people.
- **Hyperparameter search must select on the VAL split only** (`configs/val_episodes.yaml`, seeds 10000-10099),
  never on the 600 frozen TEST seeds (`configs/test_episodes.yaml`, seeds 0-599). This is a convention, not an
  enforced code path — see `scripts/step45_val_sweep.py` for the pattern (in-process, not subprocess, so output
  can't silently vanish on Colab).
- **Never commit to git unless explicitly asked.** This project's convention is that a human reviews and commits.

## How do we run this project

We use google colab to run this project with Jupiter notebook. You will see the notebooks for every step inside /notebooks folder.

**Never bring the GPU relay / vGPU system into a task unless the user names it.** The relay path
(`src/vgpu/`, `docs/gpu-relay-guide.md`, `notebooks/*-v2.ipynb`, `requirements-vgpu.txt`, `GPU_POOL_URL`) is an
opt-in alternative, not the default. Default to the plain hosted-notebook path (Colab/Kaggle: clone repo → install
requirements → `scripts/train.py` / `scripts/evaluate.py`). When a step has both a `stepN.ipynb` and a
`stepN-v2.ipynb`, "stepN" means the plain v1 notebook. Do not propose, edit, or read relay code as part of an
unrelated request.

## Architecture

**Pipeline**: `frozen backbone -> trainable adapter -> head`, assembled by `build_model(cfg)` in
[src/models/bpeft_model.py](src/models/bpeft_model.py). Every piece is chosen by a config key and built via a small
factory in each subpackage's `__init__.py` (`build_backbone`, `build_adapter`, `build_head`, `build_loss`,
`build_dataset`) — adding a new variant means adding a class + a branch in that one factory function, not touching
callers.

**Two parallel trainer/evaluator protocols, dispatched on `cfg.trainer.type`** — both live side by side in
`scripts/train.py` and `scripts/evaluate.py`, each with its own private helper function:

- `single_episode` (Step 1-3, legacy): one fixed episode, 200 inner gradient steps directly on a fresh model each
  time, LinearHead/EvidentialHead. Kept byte-identical on purpose so old reproduction tests still pass — don't
  "clean up" this path.
- `episodic` (Step 4+, current): true episodic meta-training via
  [src/trainers/episodic_trainer.py](src/trainers/episodic_trainer.py) — many sampled episodes/epoch update a
  shared adapter, validated each epoch on a fixed val-episode stream, early-stopped on val-accuracy plateau. Uses
  the parameter-free `PrototypeHead` ([src/heads/prototype_head.py](src/heads/prototype_head.py)); classification
  logits come from support-set prototype similarity (L2 or cosine), not a trained linear layer, because a linear
  head trained on the 64 train classes cannot transfer to the 20 disjoint test classes (documented deviation from
  proposal §5B — see `progress.txt`'s 2026-05-19 decisions log).

**Head "type" vs. "interpretation"**: `PrototypeHead` always emits raw similarity logits; whether those logits are
read as softmax logits or mapped to Dirichlet evidence is decided _outside_ the head, by `cfg.head.interpretation`
(`softmax` | `evidential`) and consumed in the loss/evaluator. The evidence mapping itself
(`evidence = softplus(logits * scale + bias)`) lives in `PrototypeHead.to_evidence()` as the single source of truth
shared by the trainer's loss and the evaluator's OOD score — never reimplement softplus-of-logits elsewhere, or
train/eval can silently drift apart (this exact bug caused the Step 4 "evidential collapse": raw L2 logits are
large-negative for ResNet-18 features, so `softplus(logit)~0` everywhere → uniform Dirichlet → dead gradients).

**Config system** ([src/utils/config.py](src/utils/config.py)): YAML files support `extends: <relative-path.yaml>`
(string or list), recursively deep-merged with the child's keys winning. `configs/base.yaml` is the default; every
experiment config extends it and overrides only what differs. `ConfigDict` gives attribute access
(`cfg.train.lr == cfg["train"]["lr"]`). When adding a new experiment config, extend `base.yaml` (or the closest
existing sibling) rather than duplicating the whole tree.

**Reproducibility invariants** (enforced by convention/tests, not runtime asserts everywhere — respect them when
changing code): `src/utils/seed.py:set_seed()` seeds python/numpy/torch CPU+CUDA and forces
`cudnn.deterministic`; every eval script run on the same config must produce a byte-identical `metrics.json`
(`json.dump(..., sort_keys=True)` + fixed episode seeds) — if a rerun differs, something non-deterministic was
introduced and must be fixed before trusting results from that config.

**Result JSON naming**: `results/<results-suffix>_<adapter.type>_<head-descriptor>_metrics.json`, where
`head-descriptor` is `cfg.head.type` except for `prototype`, which becomes `prototype-<interpretation>` (otherwise
the two Phase-2 configs collide on disk). The same descriptor function (`_head_descriptor`) is duplicated
identically in `scripts/train.py` and `scripts/evaluate.py` — keep them in sync if you touch either.

**W&B**: `src/utils/wandb_utils.py:WandbRun` wraps online/offline/disabled modes uniformly; `wandb.disabled: true`
in a config makes it a true no-op (no init, no files, no console noise) — this is the default, not opt-in.

## Known state of the science (don't re-litigate without new evidence)

Answers to all four RQs are in `docs/RQ_SUPERVISOR_REPORT.md` §1. At grid scale (Step 10, 120 runs): evidential
vacuity beats softmax-probability scores (MSP, TS-MSP) in 37–38/40 cells on every OOD pool, but the evidential head
is worse calibrated than plain softmax in 20/20 matched pairs (Orig-RQ2 is a clean negative). Two earlier claims
were overturned by the project's own follow-up experiments:

**Correction (2026-08-06) — see `progress.txt`'s Step 10 entry and `docs/RESULTS_MASTER.md` Table 5 / RQ3:** the
Step 4.5 finding that evidential vacuity is roughly on par with
the non-probabilistic energy score (winning far-OOD and CIFAR-100-near, losing only TinyImageNet-near) was a
single-configuration result and does **not** generalise. The Step 10 MVT grid (120 runs, 40 aggregated
`(dataset, shot, backbone, adapter, head)` cells) found vacuity beats energy in only 10/40 far-OOD and 14/40
near-OOD matched comparisons — energy is the better default OOD score in **most** cells (~70% of comparisons),
reversing the earlier read. The probabilistic-score claim (vacuity vs. MSP/TS-MSP) is unaffected and strengthens at
grid scale: 37–38/40 wins on every pool. Do not cite "evidential is on par with energy" anywhere in the thesis text
without this correction; the defensible claim is narrower — vacuity is a substantially better OOD ranker than
softmax-probability scores, but a well-chosen logit-space score (energy) still beats it.

**Correction (2026-08-27) — RQ3's causal claim changed; see `progress.txt`'s "RQ3 matched-budget headline"
and `docs/RQ_RESULTS_SUMMARY.md` §5.1.** Note the numbering: this is the *new* four-RQ RQ3 — adapter
architecture vs. trainable-parameter budget — not Orig-RQ3 (Bayesian prior vs. near-OOD). The 16-pair grid evidence showed
the accuracy winner (bottleneck) never changing when the parameter-budget ordering reverses between
backbones, while the calibration winner changed exactly in step with it; that was originally read as
"calibration follows the larger budget" (H3.2). It was untestable on the grid, because the budget ordering
is welded to the backbone. A pre-registered 48-run matched-budget experiment — both architectures at the
same budget *within* each backbone, MiniImageNet 5-shot — returned verdict **`backbone_intrinsic`**: H3.2
fired in **0 of 4** cells, H3.2-alt in 3 of 4. Equalising the budget leaves ResNet-18's calibration gap
entirely intact and only halves MobileNetV3-Small's. **Do not write "calibration follows the parameter
budget" anywhere in the thesis text.** The defensible claim is: accuracy and near-OOD ranking follow adapter
*architecture* (bottleneck wins 8/8 at matched budget, all beyond 2σ); calibration follows the *backbone*,
by a mechanism this two-backbone design cannot identify. Treat this as settled — re-opening it would take
more backbones, not more seeds.

Always Read `thesis_implementation_instructions.txt` before any implementation step.
