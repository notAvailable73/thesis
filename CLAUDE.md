# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A masters thesis codebase: **B-PEFT** (Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with
Lightweight CNN Backbones). It trains a frozen CNN backbone (ResNet-18) + a small trainable adapter (Bottleneck /
LoRA / BitFit / Full-FT / Linear-Probe) + a classification head (Softmax, Evidential Dirichlet, or a parameter-free
Prototype head), evaluated on few-shot episodes (CIFAR-FS, 5-way k-shot) for accuracy, calibration (ECE, Brier), and
OOD detection (SVHN far-OOD, CIFAR-100-heldout / TinyImageNet near-OOD).

The four research questions (proposal.txt §4) that every experiment ultimately serves:
- **RQ1**: adapter placement (serial vs parallel) — accuracy vs. parameter count tradeoff.
- **RQ2**: does an Evidential Dirichlet head calibrate better than softmax under <500 trainable params?
- **RQ3**: does a Bayesian loss prior improve near-OOD detection in low-data regimes?
- **RQ4**: latency vs. uncertainty-quality Pareto frontier on edge hardware.

Git Repo link - https://github.com/notAvailable73/thesis.git

## Required reading before doing any work here

This project tracks its own state in plain-text files at the repo root — read them in this order before touching
code, and re-read `progress.txt` any time you're unsure what's already done:

1. **`instructions.txt`** (untracked, meant for whoever/whatever picks up the repo next) — current task, gotchas,
   quick commands. Read this first, always.
2. **`thesis_implementation_instructions.txt`** — the *process* rule for this repo: implementation choices must be
   justified against the paper summaries in `PAPER SUMMARIES/*.txt` (pros/cons/fit reasoning), not implemented from
   general training-data knowledge. If a paper summary and general knowledge conflict, defer to the summary and flag
   it. Do not invent hyperparameters/results not stated in a summary.
3. **`progress.txt`** — canonical status tracker: one section per step (13 steps across 6 phases), checkboxes, exit
   criteria, and a running decisions log at the bottom. This is the single source of truth for "what's done" —
   trust it over README.md, which is stale (describes a `src.train`/`src.evaluate` module layout that no longer
   exists; the real entry points are `scripts/train.py` / `scripts/evaluate.py`).
4. **`plan.txt`** / **`proposal.txt`** / **`implementation.txt`** — proposal → phased plan → step-by-step build spec,
   in that order of increasing detail. `implementation.txt` has the exact spec (file list, config knobs, exit
   criteria) for whichever step is next.
5. **`step_writeups/stepN.txt`** — the write-up for the most recently closed step; explains *why* results came out
   the way they did, which matters for interpreting the next step's results.

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
read as softmax logits or mapped to Dirichlet evidence is decided *outside* the head, by `cfg.head.interpretation`
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

Per `progress.txt`: on real CIFAR-FS (Bertinetto split, 600 test episodes, 5-way 5-shot), the evidential head's
uncertainty (vacuity) decisively beats every softmax-based confidence score (plain max-prob, temperature-scaled
max-prob) on far-OOD (SVHN) and both near-OOD sets (CIFAR-100-heldout, TinyImageNet) tested, by +0.076 to +0.141
AUROC — but is still ~7x worse calibrated (ECE) than temperature-scaled softmax, and a real VAL-only hyperparameter
sweep confirmed that calibration gap doesn't close easily (the ECE surface is flat ~0.285-0.296). Against the
non-probabilistic "energy" OOD score specifically, evidential wins on far-OOD and CIFAR-100-near but loses on
TinyImageNet-near by -0.013. This is the Tier-3 verdict from `scripts/step45_verdict.py` / `step_writeups/step4_5.txt`.
Treat this as the current baseline any new PEFT method (LoRA, BitFit, Full-FT, Linear-Probe — Step 5) is compared
against, not something to reprove from scratch.
