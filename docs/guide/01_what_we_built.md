# 1. What We Built

This page explains the model, the data, and how we test it. It uses simple words. The last section is a
glossary of the terms used everywhere in the project.

---

## 1.1 The goal

A normal image model needs many examples to learn a new class. We want a **small, cheap** model that can
learn a new class from **1 or 5 examples**. It should also know **when it is unsure** and **when it sees
something it was never taught**.

The thesis title is: *Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with
Lightweight CNN Backbones.*

---

## 1.2 The model, step by step

```
input image
     |
     v
frozen backbone        <- already trained on ImageNet; we never change it
     |
     v
small adapter          <- the ONLY part we train (a few thousand parameters)
     |
     v
prototype head         <- has no trainable weights; compares the image to class averages
     |
     v
raw scores (logits)
     |
     +--> read as SOFTMAX     -> probabilities, confidence
     +--> read as EVIDENTIAL  -> probabilities + an uncertainty number (vacuity)
```

### Backbones (the frozen part)

| Backbone | Size | Output features | Notes |
|---|---|---|---|
| ResNet-18 | 11.7M parameters (11,176,512 without its final layer) | 512 | The main backbone |
| MobileNetV3-Small | 2.5M parameters (frozen trunk 927,008) | 576 | The small "edge device" backbone |

Both come pretrained on ImageNet and stay frozen. The only exception is the Full Fine-Tuning baseline.

### Adapters (the trained part)

| Adapter | What it does | Trainable params (ResNet-18 / MobileNetV3-Small) | Used in |
|---|---|---|---|
| **Bottleneck, parallel** | A small side branch (1×1 conv shrink → ReLU → 1×1 conv grow) next to the last block of each of the 4 stages. Its output is added to the block output. | 31,744 / 6,928 | Main grid |
| **LoRA** | A low-rank update added inside one existing 1×1 convolution | 12,288 / 10,752 | Main grid |
| **Full Fine-Tuning** (baseline) | Unfreezes the whole backbone | 11,176,512 | Grid, ResNet-18 + CIFAR-FS only |
| **Linear Probe** (baseline) | Trains nothing except the 2 evidential numbers | 0 (softmax) / 2 (evidential) | Grid, ResNet-18 + CIFAR-FS only |
| Bottleneck, serial | Same branch, placed in the main path instead of beside it | 31,744 (ResNet-18) | Step 6 only |
| Bottleneck, post-pool | Applied once, after the backbone's final pooling | 16,912 / 19,024 | Steps 4–9 only |
| BitFit | Trains only bias terms | 4,800 (ResNet-18) | Step 5 only |

All adapters use rank 16 in the main grid.

**A useful accident.** On ResNet-18 the bottleneck adapter is *bigger* than LoRA. On MobileNetV3-Small it
is *smaller*. That happens only because the two backbones have different channel widths. RQ3 is built on
this.

### The head

The **prototype head** has no trainable weights. For each class, it averages the features of that class's
example images; that average is the "prototype". A new image is scored by how close it is to each
prototype. In every grid run "closeness" is **cosine similarity × 10**, not Euclidean distance (`metric:
cosine`, `cosine_scale: 10` in every grid config; `configs/base.yaml` still says `l2`, but the grid never
uses that default). See `09_methodology.md` §3.3 for why this matters.

We use this instead of a normal trained classifier layer. A layer trained on the 64 training classes cannot
transfer to the 20 different test classes. This is a documented change from the proposal.

### Two ways to read the output

The head always produces the same raw scores. We then read them in one of two ways:

- **Softmax.** Turn the scores into probabilities. Confidence is the highest probability.
- **Evidential (Dirichlet).** Turn the scores into "evidence":
  `evidence = softplus(score × scale + bias)`, then `alpha = evidence + 1`, `S = sum of alpha`.
  The probability of class k is `alpha_k / S`. Uncertainty ("vacuity") is `K / S`, where K = 5 classes.
  Little evidence gives high uncertainty.

`scale` and `bias` are the only 2 extra numbers. They are **learnable**: every evidential grid config sets
`evidence_affine: true`, so they start at `(2, −6)` and Adam trains them together with the adapter. Over
the 48 evidential models Phase A recovered, training moved them to scale 1.51–4.52 (median 2.93) and bias
−9.27 to −5.58 (median −6.30). RQ4 is the experiment that **refits** them after training, on validation
episodes. *(Earlier documents said they were "fixed" or "frozen" at (2, −6). That was wrong and has been
corrected.)*

The evidence formula lives in one place in the code: `PrototypeHead.to_evidence()` in
`src/heads/prototype_head.py`. Training and evaluation must both use it.

---

## 1.3 How training works

We use **episodic meta-training**. The model practises on many small tasks, called episodes, drawn
from the 64 training classes.

- One **episode** = pick 5 classes, give 1 or 5 example images per class (the "support set"), then test
  on 15 more images per class (the "query set", 75 images).
- Each epoch trains on 100 episodes, then checks accuracy on **100 fixed validation episodes**.
- Training stops early when validation accuracy stops improving (patience 5, at most 30 epochs).
- The backbone stays in evaluation mode, so its BatchNorm statistics never change.

**Losses.** Softmax runs use cross-entropy. Evidential runs use the evidential loss from Sensoy et al.
2018: squared error plus a KL term. The KL weight rises linearly from 0 to 0.1 over the first 1,000 training
episodes. The loss drops Sensoy's variance term (`use_variance: false`, an R-EDL relaxation).

**One fixed recipe for everything.** Adam, learning rate 5e-3, no weight decay, adapter rank 16,
`kl_weight_max` 0.1, evidence affine initialised at `(scale, bias) = (2, −6)`. Full FT is the exception: LR
1e-5, weight decay 1e-4. The recipe was set on ResNet-18 + CIFAR-FS + 5-shot in Step 4.5 and then used
unchanged everywhere. ⚠️ The Step 4.5 VAL sweep actually ranked `kl_weight_max` = **0.05** first (VAL ECE
0.252 vs 0.260 for 0.1), but the config the grid inherits kept **0.1**. No decision-log entry explains this;
see `05_problems_and_open_work.md` A7.

**Images.** All images (CIFAR-FS 32×32, MiniImageNet 84×84, and every OOD set) are resized to 224×224 and
normalised with ImageNet mean/std. There is **no data augmentation**.

**Seeds.** Each setting is trained 3 times, with seeds 42, 43 and 44. The seed only changes the random
starting weights of the adapter. It does **not** change the order of training episodes.

---

## 1.4 How testing works

- **600 fixed test episodes** (seeds 0–599, stored in `configs/test_episodes.yaml`). Everyone uses the
  exact same 600.
- 5-way tasks, with 1-shot or 5-shot support sets.
- Test classes never appear in training.
- Running the same setting twice gives a **byte-identical** results file. This was checked several times.

### Datasets

| Dataset | What it is | Split (train / val / test classes) |
|---|---|---|
| CIFAR-FS | 100 classes from CIFAR-100, small 32×32 images | 64 / 16 / 20 (Bertinetto split) |
| MiniImageNet | 100 classes from ImageNet, 84×84 images (resized to 224) | 64 / 16 / 20 (Ravi & Larochelle split) |

### OOD test sets (images the model should flag as "unknown")

| Type | Set | Meaning |
|---|---|---|
| Far-OOD | SVHN (house-number photos) | Very different from the training images |
| Far-OOD | Gaussian noise | Random noise; a sanity check |
| Near-OOD | CIFAR-100 held-out classes (for CIFAR-FS) / MiniImageNet held-out classes (for MiniImageNet) | Similar-looking images from classes not in the task. "Held-out" means the dataset's 16 **validation** classes — disjoint from the 20 test classes, but the same classes the validation episodes use |
| Near-OOD | TinyImageNet | Similar natural images; for MiniImageNet runs, the 25 classes it shares with MiniImageNet are removed |

### OOD scores (ways to measure "how unknown is this image")

| Score | Comes from | Idea |
|---|---|---|
| MSP | Softmax | Low top probability means probably unknown |
| TS-MSP | Softmax | Same, after temperature scaling (fitted on validation episodes) |
| Energy | Raw scores | A smooth summary of all raw scores (Liu et al. 2020) |
| Vacuity | Evidential | High uncertainty means probably unknown |

In the main grid, evidential runs were scored only with vacuity. Softmax runs were scored only with MSP,
TS-MSP and energy. RQ2 fixed this by computing all four scores on all runs.

### Metrics

| Metric | Measures | Better is |
|---|---|---|
| Accuracy, Macro-F1 | Correct answers | Higher |
| ECE (pooled) | Gap between confidence and real accuracy | **Lower** |
| Brier score | Probability error | Lower |
| AUROC | How well a score separates known from unknown images | Higher (1.0 is perfect, 0.5 is guessing) |
| FPR@95 | False alarms when 95% of known images are kept | Lower |
| Params, FLOPs, latency, memory | Cost | Lower |

**Hardware.** Training ran on Kaggle / Google Colab GPUs (mostly an NVIDIA T4). Edge speed was measured
on a single CPU thread as a stand-in for a small device. No Jetson Nano was available.

---

## 1.5 The demo app

`app/` contains **Sentinel**, a small web app for industrial product inspection. You register a product
with a few photos. The app then identifies items and flags unknown ones for manual checking.

It uses a frozen ResNet-18, a prototype store and an evidential uncertainty rule. It is a
**demonstration only**:

- it runs its own simple model;
- it is **not connected to the trained thesis adapters**;
- it has **no accuracy test**.

See `app/README.md` and `app/doc/`.

---

## 1.6 Glossary

| Term | Plain meaning |
|---|---|
| **Few-shot** | Learning a class from very few examples (here 1 or 5) |
| **N-way K-shot** | A task with N classes and K examples of each |
| **Backbone** | The big pretrained network that turns an image into features |
| **Frozen** | Not changed during training |
| **Adapter / PEFT** | A small trainable add-on. PEFT = parameter-efficient fine-tuning |
| **Rank** | The size of the adapter's narrow middle layer. Bigger rank means more parameters |
| **Episode** | One small practice or test task |
| **Support / query set** | The example images / the images to classify in an episode |
| **Prototype** | The average feature of a class's example images |
| **Logits** | The raw scores before any probability step |
| **Softmax** | Standard way to turn scores into probabilities |
| **Evidential / Dirichlet** | A reading that turns scores into "evidence" and gives an uncertainty value |
| **Vacuity** | The evidential uncertainty number `K / S`; high means "I don't know" |
| **Calibration** | Whether 80% confidence really means 80% correct |
| **ECE** | Expected Calibration Error; the main calibration number |
| **Temperature scaling (TS)** | A standard fix for softmax calibration: divide scores by one fitted number |
| **OOD** | Out-of-distribution: an image from outside what the model was taught |
| **Near / far OOD** | Similar-looking unknown images / very different unknown images |
| **AUROC** | How well a score separates known from unknown images |
| **η² (eta-squared)** | The share of the variation in a result explained by one factor |
| **Grid / factorial** | Running every combination of the chosen factors |
| **Cell** | One combination in the grid (e.g. CIFAR-FS, 5-shot, ResNet-18, LoRA, softmax) |
| **VAL / TEST episodes** | Validation episodes (seeds 10000–10099) for tuning; test episodes (seeds 0–599) for final numbers only |
| **Pre-registration** | Writing down the decision rule *before* running the experiment |
| **Pareto frontier** | The settings where you cannot get better on one axis without getting worse on another |
