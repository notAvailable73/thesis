# The ML Model — How Training Works & How the App Connects

This explains where the machine learning actually lives, how to run training on
your device, and how the trained thesis model would connect to the Sentinel app.
Read this if you're unsure "how to start the ML stuff."

> **TL;DR** — Your device is already ML-ready (torch + cached weights). The app
> runs its *own* self-contained ML today and is **not** connected to the trained
> thesis model in `src/`. Connecting them is an optional extra step that needs a
> saved checkpoint. See [Two tracks](#two-ml-tracks-in-this-repo) and
> [Connecting](#how-to-connect-the-trained-model-to-the-app).

---

## Your device is already set up

You do **not** need to install or start anything ML-specific separately:

- `torch 2.12.1+cpu` and `torchvision 0.27.1+cpu` are installed in `.venv`.
- ImageNet ResNet18 weights are already cached at
  `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`.

When you run `./app/run.sh`, the ML runs on CPU **inside the same process** — there
is no separate model server to launch.

---

## Two ML tracks in this repo

The repo contains two different ML paths, and **they are currently not connected.**
That is the main source of confusion.

### Track A — the app's ML (self-contained, runs today)

The app ships its own simplified copy of the architecture:

| Piece | File | What it does |
|-------|------|--------------|
| Frozen backbone | `app/backend/ml/backbone.py` | ImageNet ResNet18 → 512-d embedding |
| Prototype head  | `app/backend/services/prototype_store.py` | product = **mean** of its reference embeddings (parameter-free, no training) |
| Evidential math | `app/backend/ml/evidential.py` | similarities → belief + vacuity (OOD) |

This is a *training-free* version of the thesis method, so the demo works standalone.

### Track B — the thesis model you're building (in `src/`)

This is the real research code:

| Piece | File(s) |
|-------|---------|
| Trainable adapters (the PEFT study) | `src/adapters/` — `lora.py`, `bottleneck.py`, `bitfit.py`, `full_ft.py`, `linear_probe.py` |
| Heads | `src/heads/` — `linear_head.py`, `prototype_head.py` |
| Losses | `src/losses/` — `cross_entropy.py`, `evidential.py` |
| Model assembly | `src/models/bpeft_model.py` |
| Training entry point | `scripts/train.py` |
| Evaluation entry point | `scripts/evaluate.py` |

**The gap:** the app (Track A) never loads anything from `src/` or any trained
checkpoint. The `backbone.py` docstring says so directly: *"The adapter lives in
the research code (`src/adapters/`); swapping it in here would mean loading a
trained checkpoint into this module."* As of now, no `.pt` checkpoint exists on
disk yet, so there is nothing trained to load anyway.

---

## How training works (Track B)

### The one command

```bash
source .venv/bin/activate
python scripts/train.py --config configs/exp_phase2_evidential.yaml
```

Everything is **config-driven** — you pick a YAML from `configs/`; it inherits
defaults from `configs/base.yaml` via `extends:`. You don't edit Python to run an
experiment.

### What a config controls

Each run is defined by four choices (see `configs/base.yaml`):

| Block | Options | Meaning |
|-------|---------|---------|
| `backbone` | resnet18 (512-d) | Frozen feature extractor (same as the app) |
| `adapter`  | `bottleneck` \| `lora` \| bitfit \| full_ft \| linear_probe | The **trainable** PEFT module — the thesis subject |
| `head`     | `softmax` \| `evidential` \| `prototype` | softmax = overconfident baseline; evidential/prototype = the "honest AI" side |
| `loss`     | auto-matched to head | cross-entropy vs. evidential (Dirichlet) |

Config filenames encode the combo, e.g. `exp_phase3_lora_evidential.yaml` = LoRA
adapter + evidential head.

### Two training modes (`trainer.type`)

- **`single_episode`** (Steps 1–3) — trains on *one* few-shot episode, 200 inner
  steps. Quick; good for a first end-to-end sanity run.
- **`episodic`** (Phase 2 / Step 4) — proper episodic meta-training: many episodes
  per epoch from the CIFAR-FS train split (64 classes), validate on 16, early-stop.
  The "real" training; slower on CPU but works.

### Data — auto-downloaded

- Dataset: **CIFAR-FS** (built on CIFAR-100), `data_root: data`. The repo `data/`
  starts empty, so the **first run downloads it automatically** (one time).
- OOD test set: **SVHN**, also auto-downloaded.
- You fetch nothing manually.

### Where it runs & what you get

- **Runs on your CPU** — no GPU needed.
- **Weights & Biases is OFF by default** (`disabled: true` in `base.yaml`) — no
  account or login required. Ignore all wandb output.
- **Outputs** (`scripts/train.py`):
  - `checkpoints/model_<tag>.pt` — trained weights (**the artifact the app would
    load**), including `state_dict`, `adapter_type`, `head_type`.
  - `results/*.json` — metrics (accuracy, ECE calibration, OOD).

### Typical research flow

1. `python scripts/train.py --config configs/exp_phase2_evidential.yaml` → checkpoint
2. `python scripts/evaluate.py ...` → accuracy / calibration / OOD numbers
3. Repeat across adapter/head combos — `scripts/run_grid.sh` sweeps them.

---

## How to connect the trained model to the app

To make the app run your *trained* B-PEFT model instead of the plain ImageNet
backbone:

1. **Train and save a checkpoint** — run `scripts/train.py` with a config; it
   writes `checkpoints/model_<tag>.pt`.
2. **Load that checkpoint in the app** — modify `app/backend/ml/backbone.py` /
   `app/backend/services/model_service.py` to build the same backbone + adapter as
   `src/models/bpeft_model.py` and load the trained weights, instead of the bare
   `resnet18`.
3. **(Optional) align the head** — if the trained model uses a learned evidential
   head rather than the parameter-free prototype head, mirror that in the app so
   its decisions match the thesis numbers.

Until those steps are done, the app runs Track A (self-contained) and its accuracy
reflects the plain ImageNet backbone, **not** your trained thesis model.

---

## Related

- [evaluation-plan.md](evaluation-plan.md) — measuring the app's accuracy on
  real-life data (uses whichever ML track the app is currently running).
