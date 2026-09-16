# B-PEFT Thesis — Full Progress Notes & ML/DL Concepts Explained

> **Who this is for:** you, if you want to actually understand *what has been built, why
> it was built, and what the results mean* — without needing a machine-learning
> background going in. Every concept is explained once, in plain language, with a
> real-world example, before it's used in the step-by-step story.
>
> **Source of truth:** everything below is transcribed/summarized from
> [`progress.txt`](../progress.txt) (the canonical tracker) and the per-step writeups in
> [`step_writeups/`](../step_writeups/). If numbers ever look off, `progress.txt` wins —
> this file is a teaching companion, not a replacement.

---

## Part 0 — The One-Paragraph Version

This thesis asks: *can we take a small, already-trained image-recognition network,
teach it a brand-new task from just a handful of example photos, using barely any extra
computation — and can we make it honestly say "I'm not sure" when it's shown something
it's never seen before, instead of confidently guessing wrong?* The project has spent 11
(of 13 planned) steps building the pipeline, proving out each technique against
real benchmarks, and running an increasingly large grid of experiments. The headline
result so far: the "where you attach the small trainable piece" question (RQ1) is
answered decisively, the "does a fancier uncertainty-aware loss calibrate better" question
(RQ2) is answered decisively **no**, the "does it detect unfamiliar inputs better" question
(RQ3) is answered **yes, partially** (it beats the obvious baselines but not the
strongest alternative), and the "what does this cost on real hardware" question (RQ4) is
the one currently blocking — the code is finished and tested, but the actual
measurement run keeps crashing on Kaggle and hasn't produced final numbers yet.

---

## Part 1 — The Big Picture: What Problem Is Being Solved?

Three real problems motivate this work (see [`proposal.txt`](../proposal.txt) §3):

1. **You rarely have much labeled data.** Training a large neural network from scratch
   needs thousands of examples per category. In medicine, rare wildlife, or manufacturing
   defect detection, you might have 5 or 10 examples of the thing you care about. Training
   (or even fully *re*-training) a huge network on 5 images causes catastrophic
   overfitting — the model just memorizes those 5 photos instead of learning the concept.
2. **Models lie about how sure they are.** A standard classifier outputs something that
   *looks* like a probability ("95% confident this is a cat") but that number is often not
   trustworthy — the model can be 95% "confident" about something it has never actually
   seen anything like, e.g. an X-ray of an organ it wasn't trained on. That's dangerous in
   safety-critical settings (self-driving cars, medical imaging, industrial inspection).
3. **Most of this research is done on giant Transformer models (ViT), not the small CNNs
   that actually run on cheap edge hardware** (a phone, a Raspberry Pi, a security camera).
   How these techniques behave inside a *convolutional* network specifically is
   under-studied.

The thesis's answer, in one line: **freeze a small pretrained CNN, attach a tiny
trainable "adapter," and swap the usual softmax output for an "evidential" one that can
express genuine uncertainty** — then measure, rigorously, whether that combination is
actually better across accuracy, calibration, out-of-distribution detection, and
real-hardware efficiency.

---

## Part 2 — The ML/DL Concepts You Need

Read this part once; the step-by-step story in Part 4 will keep referring back to it.

### 2.1 Backbone, pretraining, and "freezing"

A **backbone** is a neural network that has already been trained (usually on ImageNet, a
huge dataset of 1.2 million labeled photos across 1000 categories) to turn a raw image
into a compact numerical summary — a **feature vector** — that captures useful visual
information (edges, textures, shapes, object parts). This project uses two backbones:

- **ResNet-18** — a classic, medium-sized CNN (11.7M parameters).
- **MobileNetV3-Small** — a much smaller CNN (0.9–2.5M parameters) specifically designed
  to run fast on phones and other low-power ("edge") hardware.

**"Frozen"** means: during training, none of the backbone's own numbers (weights) are
allowed to change. The network's general visual knowledge is locked in place; only a
small extra piece attached to it gets trained.

> **Analogy:** imagine a chef who spent 10 years mastering French cuisine (that's
> pretraining on ImageNet). You don't want to send them through culinary school again to
> teach them Thai cooking (that would be slow, expensive, and might make them forget
> French cuisine too — this is called *catastrophic forgetting*). Instead, you keep their
> core knife skills and palate exactly as they are ("freeze" them) and just give them a
> short workshop on Thai ingredients and techniques (a small trainable "adapter").
>
> **Real-world use case:** a hospital wants an X-ray triage tool but has only 200 labeled
> scans for a rare condition. Instead of training a CNN from scratch (which would need
> tens of thousands of images), they start from an ImageNet-pretrained backbone and
> fine-tune only a small piece on their 200 images.

### 2.2 Few-shot learning ("N-way K-shot")

**Few-shot learning** means training/evaluating a model's ability to learn a new
category from only a handful of examples. This project uses **5-way K-shot** episodes:

- **"5-way"** = the model must pick the correct answer among 5 candidate classes it has
  never explicitly been trained to recognize by name.
- **"K-shot"** = the model is shown K example photos of each of those 5 classes right
  before being tested (K=1 or K=5 in this project — "1-shot" and "5-shot").

Each such test — "here are 5 classes, here are K examples of each (the **support set**),
now classify these new photos (the **query set**)" — is called an **episode**.

> **Analogy:** show a child exactly one photo each of five bird species they've never
> seen before (1-shot, 5-way), then show them a sixth, new photo of one of those species
> and ask "which of the five is this?" No prior study time on these specific species —
> just quick pattern-matching from a single glimpse.
>
> **Real-world use case:** identifying a newly discovered manufacturing defect type from
> just the 3 photos an inspector happened to take, without waiting to collect thousands
> more examples first.

### 2.3 Episodic (meta-)training and Prototype heads

If few-shot learning is the *test*, **episodic training** (a.k.a. **meta-training**) is
the matching *training* procedure: instead of training on one big fixed dataset, the
model is trained on thousands of *simulated* few-shot episodes (sampled from a large
pool of "training classes," disjoint from the classes used for testing). Each episode:
show a few support examples, make a prediction on new query examples, check how wrong it
was, and adjust. Repeating this thousands of times teaches the model *how to quickly
adapt to any new small set of classes* — not to memorize specific classes.

A **Prototype head** is the specific way this project turns "a few example photos" into
a decision: it averages the feature vectors of the K support photos in each class to get
one summary vector — a **prototype** — per class, then classifies a new photo by which
prototype it's numerically closest to (like a "center of mass" for each category).

> **Analogy:** to decide whether a new fruit is an "apple" or an "orange," compute the
> average color/shape of the 5 apples and 5 oranges someone showed you, then check which
> average the new fruit's color/shape is closer to.
>
> This project originally planned a trained linear classifier head instead, but that
> doesn't work for few-shot testing: a linear layer trained to recognize 64 specific
> training classes has no way to output a class it's never seen at test time (20 new,
> disjoint test classes). The prototype approach sidesteps this entirely — it's a
> documented, deliberate deviation from the original proposal (see the 2026-05-19 entry
> in `progress.txt`'s decisions log).

### 2.4 PEFT — Parameter-Efficient Fine-Tuning

**PEFT** is the umbrella idea: instead of updating all of a large pretrained network's
weights (which is slow, memory-hungry, and prone to overfitting on small data), freeze
almost everything and train only a *small* number of new or existing parameters. This
project compares five PEFT-style methods head-to-head:

| Method | What actually gets trained | Rough size (ResNet-18, this project) |
|---|---|---|
| **Bottleneck Adapter** | A tiny new block: shrink the feature down to a narrow "bottleneck" width, then expand back up | ~17K–32K params |
| **LoRA** (Low-Rank Adaptation) | Two small matrices `A`, `B` added on top of one frozen conv layer, so the effective change is `A × B` (a *low-rank* update) instead of a full new matrix | ~12K params |
| **BitFit** | Only the *bias* terms (small per-channel offsets) already inside the frozen backbone | ~4.8K params |
| **Full Fine-Tuning** (baseline) | The *entire* backbone, unfrozen | ~11.2M params |
| **Linear Probe** (baseline) | Nothing in the backbone — literally 0 new trainable weights when paired with the prototype head | 0 params |

> **Analogy (Bottleneck Adapter):** a small converter box spliced into a pipe — it
> squeezes the flow down to a narrow point and widens it back out, and only *that box* is
> new/adjustable; the pipe itself (the frozen backbone) is untouched.
>
> **Analogy (LoRA):** rather than repainting an entire wall (the full weight matrix), you
> overlay a small stencilled pattern on top of it — cheap to make, cheap to store, and
> the underlying wall paint never changes.
>
> **Analogy (BitFit):** the backbone is a fully-built machine; BitFit only lets you turn
> a handful of small calibration dials (biases), not touch any of the internal gears
> (weights).
>
> **Real-world use case:** this is exactly how companies today fine-tune huge language
> models (GPT-style, billions of parameters) for a specific customer's chatbot — via LoRA
> — without ever touching (or having to re-store a whole copy of) the billions of frozen
> weights. This project asks the same question but for small CNNs and images.

**Why compare against Full Fine-Tuning and Linear Probe at all?** They're the two
extremes — "train everything" and "train (almost) nothing" — and act as sanity-check
baselines. If a PEFT method can't come close to Full-FT's accuracy, or does no better
than Linear Probe, it isn't earning its keep.

### 2.5 Adapter placement: serial vs. parallel

Beyond *what* the adapter looks like, *where* it's inserted into the frozen backbone
matters — this is RQ1.

- **post_pool** — bolted on at the very end, after the backbone has already finished
  processing the image into its final summary vector.
- **serial** — inserted *in the middle* of the backbone, directly in the data's path;
  every image *must* pass through the adapter on its way through the network.
- **parallel** — inserted *alongside* a stage of the backbone; the adapter processes a
  copy of the data off to the side, and its output is simply *added back in* without ever
  blocking the main path.

> **Analogy:** post_pool is a **shed built onto the driveway** after the house is
> otherwise finished — you walk out the front door and across the yard to reach it.
> Serial is a **checkpoint booth built into the hallway** — everyone walking through the
> house must pass through it. Parallel is a **bypass lane** that runs alongside the
> hallway and merges its own contribution back in without blocking the main corridor.

### 2.6 Softmax and the overconfidence problem

A standard classifier's last step is usually **softmax**: it turns a set of raw scores
into numbers between 0 and 1 that sum to 1, which *look* like probabilities ("92% cat, 5%
dog, 3% other"). The well-documented problem: softmax scores are frequently
**overconfident**, especially on inputs unlike anything the model was trained on — the
model can output "99% confident" on a picture that isn't even one of its known
categories, because softmax is mathematically forced to sum to 1 no matter what.

> **Real-world danger:** an object-detection system on a self-driving car reporting
> "99% confident: not an obstacle" about something it's never encountered — high stated
> confidence with no real basis for it.

### 2.7 Evidential Deep Learning & the Dirichlet distribution (RQ2/RQ3's core idea)

Instead of directly outputting a probability, an **evidential** classifier outputs
**evidence** for each class — a non-negative number representing "how much support did I
find for this class in the image." These evidence values become the parameters of a
**Dirichlet distribution** (a probability distribution *over* probability distributions —
i.e., instead of a single guess like "70% cat", it represents a whole spread of plausible
guesses, wide if evidence is thin, narrow if evidence is strong).

Key derived quantity: **vacuity** (epistemic uncertainty) = `K / S`, where `K` is the
number of classes and `S` is the total evidence collected. Lots of evidence → low
vacuity → confident. Little evidence → high vacuity → "I genuinely don't know."

> **Analogy:** think of a jury weighing evidence for several possible verdicts. If the
> jury has seen very little evidence overall, the *whole panel* is uncertain — not because
> any one verdict looks unlikely, but because there simply isn't enough to go on yet. A
> softmax model, by contrast, is like a jury forced to always announce percentages that
> add up to 100%, even when they've barely heard the case — it can't say "we don't have
> enough evidence," only "X% guilty."
>
> **Real-world use case:** a defect-detection system on a factory line that can flag "this
> part looks unlike anything I was trained on — send to a human" instead of confidently
> (and wrongly) sorting it into "good" or "defective."

A crucial early bug (Step 1) and a subtle implementation detail worth knowing: the
evidence must go through a **softplus** activation (a smooth version of ReLU that never
fully zeroes out), not ReLU — ReLU caused "dead neurons" (a negative score → zero
evidence → zero gradient → the model can never recover). The **KL-divergence** term in
the loss also needs to be *annealed* — its weight starts at 0 and ramps up gradually
during training — because turning it on at full strength immediately overwhelms the
learning signal on tiny 25-image support sets and the model gets stuck outputting "totally
uncertain" for everything.

### 2.8 Calibration: ECE and Brier score

**Calibration** asks: when the model says "I'm 90% confident," is it actually right about
90% of the time? A model can be *accurate* (usually right) yet badly *calibrated*
(its confidence numbers don't mean what they claim).

- **ECE (Expected Calibration Error)** — bucket predictions by their stated confidence,
  and for each bucket compare stated confidence to actual accuracy; ECE is the (weighted)
  average gap. Lower is better; 0 is perfect.
- **Brier score** — the average squared difference between the predicted probability and
  the actual outcome (1 if correct, 0 if not). Lower is better.
- **Temperature scaling (TS)** — a simple, cheap post-hoc fix: divide the raw scores by a
  single learned "temperature" number before softmax, which stretches or squashes the
  confidence distribution to better match reality, without retraining anything else. It
  turns out to be a very strong, hard-to-beat calibration baseline in this project.

> **Analogy:** a well-calibrated weather forecaster who says "70% chance of rain" should,
> across all the days they said that, be right about 70% of the time — not 95% and not
> 40%. ECE measures exactly this kind of track-record discrepancy.

### 2.9 OOD (out-of-distribution) detection: far vs. near, AUROC, FPR@95, energy score

**OOD detection** asks: can the model recognize when an input doesn't belong to *any* of
its trained categories at all? Two flavors, by how different the "foreign" input is:

- **Far-OOD** — wildly different domain, e.g. a photo of house numbers (SVHN dataset)
  shown to a model trained on animals/objects. Easy case.
- **Near-OOD** — genuinely different classes that still *look* visually similar to the
  training domain, e.g. new object categories from TinyImageNet or held-out CIFAR-100
  classes. Hard case — this is where RQ3 lives, and where the thesis argues an
  uncertainty-aware model should have the bigger edge.

Metrics used to score "how well does the model's uncertainty separate in-distribution
from OOD inputs":

- **AUROC** — Area Under the ROC Curve; 0.5 = no better than random guessing, 1.0 =
  perfect separation. This project's evidential vacuity typically scores 0.85–0.93.
- **FPR@95** — "if the detector is tuned to correctly flag 95% of true OOD inputs, what
  fraction of normal, in-distribution inputs does it *wrongly* flag too?" Lower is
  better.
- **Energy score** — a strong non-probabilistic OOD-detection baseline (Liu et al.) that
  doesn't need a Bayesian/evidential model at all — just a different way of reading the
  raw scores from an ordinary softmax classifier. This project treats it as the toughest
  competitor for the evidential approach, and it repeatedly turns out to be a genuinely
  strong one (see Part 4).

> **Real-world use case:** a fraud-detection model flags a transaction that looks
> unlike *anything* in its training data (far-OOD, easy) — but the harder, more valuable
> case is catching a *new kind* of fraud that superficially resembles normal transactions
> (near-OOD).

### 2.10 Backbone efficiency: ResNet-18 vs. MobileNetV3-Small, and edge hardware

Not all backbones cost the same to *run*. **MobileNetV3-Small** was specifically
engineered (using depthwise-separable convolutions) to be small and fast enough for
phones and other "edge" devices, at some accuracy cost on the original ImageNet
benchmark. RQ4 asks whether that ImageNet accuracy cost actually carries over to this
project's few-shot setting — and Step 8's finding is that it mostly *doesn't* (see Part
4).

### 2.11 Params, FLOPs, latency, and the Pareto frontier (RQ4)

Efficiency isn't one number — it's several, and they trade off against each other:

- **Trainable parameters** — how many numbers actually get updated during training (this
  is the "PEFT" axis: smaller is cheaper to train and store per new task).
- **FLOPs / MACs** — floating-point operations (or multiply-accumulates) needed for one
  forward pass; a proxy for raw compute cost, independent of any specific chip.
- **Latency** — actual measured wall-clock time for one prediction, on real hardware
  (this project measures Kaggle's T4 GPU and a CPU as an edge-device stand-in, since no
  Jetson Nano was available).
- **Peak memory** — how much RAM/VRAM one forward (or training) pass needs.

A **Pareto frontier** is the set of options where you can't improve one metric (say,
accuracy) without making another one (say, latency) worse. A point that's beaten on
*every* axis by some other point is "dominated" and gets dropped from the frontier; a
point that's better than everything else on at least one axis, while not being clearly
worse on the others, survives.

> **Analogy:** choosing a phone camera mode. "Pro" mode gets the best photo quality but
> is slow and drains battery; "Auto" mode is fast and battery-friendly but the photo isn't
> quite as good. Neither is objectively "better" — they sit on the *frontier* of the
> quality-vs-speed tradeoff. A mode that's slower AND takes worse photos than another
> mode would be *dominated* and pointless to keep.

---

## Part 3 — The Four Research Questions (RQs)

Every experiment in this project ultimately serves one of these ([`proposal.txt`](../proposal.txt) §4):

| # | Question in plain English | Status as of Step 10 |
|---|---|---|
| **RQ1** | Does *where* you attach the small adapter (serial vs. parallel) matter for accuracy-per-parameter? | **Answered.** Parallel wins — see Part 4, Steps 6, 8, 10. |
| **RQ2** | Does the evidential/Dirichlet head produce *better-calibrated* confidence than plain softmax, when trained with under 500 parameters? | **Answered: NO.** A well-powered negative result (Step 10) — evidential calibration is consistently *worse*. |
| **RQ3** | Does the evidential/Bayesian approach detect near-OOD (subtle, "sneaky") examples better in a low-data regime? | **Answered: partially yes.** Clearly beats plain-softmax scores; roughly ties (does not clearly beat) the strongest non-Bayesian alternative (energy score). |
| **RQ4** | What's the best tradeoff between inference speed and uncertainty quality on real (edge-like) hardware? | **In progress — Step 11**, currently blocked on a crashing measurement run. |

---

## Part 4 — The Step-by-Step Journey

Six phases, thirteen planned steps. Phases 1–4 and most of 5 are closed; Step 11 (still
in Phase 5) is the current front line.

### Phase 1 — Foundation (Steps 1–3, all CLOSED)

**Step 1 — Fixing the evidential loss (2026-04-21).**
The pre-existing demo had catastrophically bad calibration (ECE 0.526 — essentially
random). Diagnosis found *two* real bugs: the loss was applying ordinary cross-entropy on
top of the softplus-activated evidence (mathematically wrong for the evidential
framework — see §2.7) and the KL-divergence regularizer was accidentally disabled
entirely. Fixed both, switched ReLU→softplus, added KL annealing (0→0.5 over 200 steps).
Result: ECE dropped 0.526→0.167 (an order of magnitude), OOD AUROC improved by +14
percentage points over softmax, accuracy stayed within 5pp. This established the
project's first real evidence for the "evidential detects OOD better" idea (RQ2/RQ3),
though on a toy/simplified setup — not yet the real benchmark.

**Step 2 — Turning it into a proper, config-driven codebase (2026-05-13).**
Pure infrastructure: reorganized the code into clean subpackages (`adapters/`,
`backbones/`, `heads/`, `losses/`, `trainers/`, …), built a YAML config system where
every experiment config just *extends* a base config and overrides only what differs
(critical for later, when the project needed 120+ near-identical configs), and added 28
automated tests. No new science — this made every later step possible to run
reproducibly from the command line instead of hand-editing a notebook.

**Step 3 — Experiment tracking + reproducibility guarantees (2026-05-16).**
Added Weights & Biases (W&B) logging, deterministic random-seeding (so results don't
silently change between reruns), and — importantly — **froze** the exact 600 random
seeds used for every future test-set evaluation (`configs/test_episodes.yaml`, marked "DO
NOT REGENERATE") so every step's numbers stay comparable to every other step's. This
frozen-seed discipline is why the project can later say things like "these numbers
reproduced byte-for-byte."

### Phase 2 — Real Benchmark + Episodic Protocol (Step 4, Step 4.5)

**Step 4 — Moving to the real CIFAR-FS benchmark.**
Everything before this used a simplified stand-in. Step 4 switched to the real,
standard **CIFAR-FS** few-shot benchmark (100 classes carved out of CIFAR-100, split
64 training / 16 validation / 20 test classes, frozen so results are comparable across
runs) and to true **episodic meta-training** (§2.3) with the **Prototype head** (§2.3).
Result, on 600 real test episodes: evidential accuracy 0.870 vs. softmax 0.875 (parity,
as expected) — but the OOD AUROC gap was only **+0.006**, far short of the 0.05 bar the
project had set itself, and evidential's calibration (ECE 0.344) was *worse* than
softmax's (0.119). **The original Step 1 finding did not survive contact with the real
benchmark.** This triggered an "escalation" — rather than quietly move on, the project
paused to figure out *why*, which became Step 4.5.

**Step 4.5 — "Settle the science" (the fair rematch).**
The diagnosis: Step 4 had compared against a *weak* softmax baseline and only tested
far-OOD (the easy case). Step 4.5 added the missing pieces to make it a fair fight:
**temperature scaling** (§2.8, a strong calibration baseline) and the **energy score**
(§2.9, a strong OOD-detection baseline) for the softmax side; real **near-OOD** test sets
(TinyImageNet + held-out CIFAR-100 classes) in addition to far-OOD (SVHN); and a genuine
**validation-only hyperparameter sweep** (never touching the frozen test seeds) to retune
the evidential loss properly instead of using a guessed configuration.

Final result: evidential **decisively beats both plain softmax and temperature-scaled
softmax on every single OOD pool tested** (+0.076 to +0.141 AUROC — a real, repeated
win). Against the tougher **energy** baseline specifically, it's closer: evidential wins
far-OOD and one near-OOD set, but *loses* on TinyImageNet near-OOD by a small margin
(-0.013). Calibration improved somewhat (ECE 0.285, down from 0.344) but the VAL sweep
confirmed this is close to a hard ceiling — it stays roughly 7x worse than
temperature-scaled softmax (0.041) no matter what's tuned. This became the project's
settled, careful baseline story for every step after it.

### Phase 3 — All PEFT Methods + Placement (Steps 5–6)

**Step 5 — LoRA, BitFit, and the Full-FT/Linear-Probe baselines (2026-07-21).**
All five PEFT-style methods from §2.4 were implemented and run head-to-head for the
first time. Two real findings:
1. The evidential OOD-detection edge over softmax **generalizes** — it holds for every
   PEFT method tested (LoRA, BitFit, Full-FT, Bottleneck), not just the one adapter type
   used so far.
2. **Full Fine-Tuning did not collapse/overfit**, contrary to what the original proposal
   expected on 25-image support sets — it turned out to be the single *most* accurate
   configuration (0.905). The reason isn't a data leak; it's a protocol mismatch: episodic
   meta-training freezes a *shared* backbone at test time, so there's no per-episode
   fine-tuning step left for it to overfit *during*. The honest reframe: PEFT methods
   reach **~98% of Full-FT's accuracy while training 660×–2330× fewer parameters** — the
   real story is efficiency, not "Full-FT is broken."

**Step 6 — Where should the adapter go? (RQ1, 2026-07-21.)**
This is the step that directly answers RQ1. The same Bottleneck adapter was tested in
three placements (§2.5): post_pool, serial, parallel. **Parallel is the clear Pareto
pick**: 0.915 accuracy at 31,744 trainable parameters — 3–4 percentage points better than
post_pool, and even edging out Step 5's Full-FT baseline (0.905) while using only 0.3% as
many trainable parameters. Serial ties parallel on raw accuracy, but parallel wins the
tiebreak on OOD detection (better vacuity scores on every far/near-OOD pool). From here
on, "parallel" becomes the default configuration future steps build on.

### Phase 4 — OOD Breadth (Step 7)

**Step 7 — Near-OOD + Gaussian far-OOD, consolidating RQ3 (2026-07-22).**
Added a synthetic Gaussian-noise far-OOD pool (a sanity check — if a model can't
distinguish real photos from pure random static, something is badly wrong; both heads
pass near-ceiling here) and consolidated the RQ3 story specifically on the Step 6
"parallel" winner. The key finding: **evidential's advantage over plain softmax is
larger on near-OOD (+0.08 to +0.10 AUROC) than on far-OOD (+0.045 to +0.048)** — exactly
the pattern RQ3 hypothesized (an uncertainty-aware model should shine most on the *hard*,
subtle OOD cases, not the easy obvious ones).

### Phase 5 — Full Grid + Backbones + Edge Efficiency (Steps 8–11, mostly closed)

**Step 8 — Adding the second, smaller backbone (2026-07-26).**
Swapped in **MobileNetV3-Small** (§2.10), 12× smaller than ResNet-18 (927K vs.
11.2M frozen parameters), and re-ran the parallel-winner configuration on it. Two
surprises:
1. The accuracy cost of the smaller backbone was tiny — **0.3 to 1.6 percentage points**,
   not the ~7–8pp that MobileNetV3's known ImageNet accuracy deficit would predict. For
   this project's 5-way few-shot task, most of what ImageNet-top-1 measures apparently
   isn't needed.
2. RQ1's finding got *stronger* here: on MobileNetV3, parallel placement beats post_pool
   AND uses **2.75× fewer** parameters (a strict win on both axes — no tradeoff to argue
   at all), whereas on ResNet-18 the same accuracy win had cost *more* parameters. One
   regression was also flagged honestly: on this smaller backbone, the "energy" OOD
   baseline started beating evidential vacuity on most pools — the earlier "evidential is
   roughly on par with energy" claim turned out to be a ResNet-18-specific statement, not
   a general one.

**Step 9 — A second in-distribution dataset, MiniImageNet (2026-07-30).**
Swapping CIFAR-FS for **MiniImageNet** served as a check against a specific confound: is
this project's accuracy partly just measuring how much a class already resembles
something ImageNet was pretrained on (a "pretraining overlap" effect), rather than
genuine few-shot learning ability? Using the Linear-Probe baseline (0 trainable
parameters — pure frozen-feature transfer) as a direct measuring stick, the answer is
**yes, partially**: the accuracy boost from adding the adapter shrinks by roughly
half-to-a-third on MiniImageNet compared to CIFAR-FS, meaning part of CIFAR-FS's earlier
numbers really were inflated by pretraining overlap. Also: the evidential-vs-plain-softmax
OOD win still holds everywhere, but the evidential-vs-energy comparison gets *worse* here
(evidential now wins only 7 of 16 comparisons, down from "roughly even" on CIFAR-FS) —
and the Step 8 "backbone swap is nearly free" finding does **not** replicate on this
dataset (the gap grows to 4.7–5.7pp here, well outside noise).

**Step 10 — The full grid, and the RQ verdicts (CLOSED 2026-08-06).**
This is the project's largest single experiment: **120 total runs**, systematically
combining 2 backbones × multiple adapters/placements × 2 datasets × 2 shot-counts ×
3 random seeds — everything built in Steps 4–9, run together for the first time so the
comparisons are apples-to-apples and statistically solid (not just single-seed
anecdotes). All 120 cells completed with zero errors, ~36 hours of total training time.
This step is where three of the four RQs got their final, well-powered verdicts:

- **RQ1 (placement) — CONFIRMED.** Parallel bottleneck beats LoRA in all 16 of 16 matched
  comparisons (+2.1 to +8.3 percentage points), and on MobileNetV3 does so with *fewer*
  parameters too (a strict Pareto win, replicating Step 8's finding at scale).
- **RQ2 (calibration) — NEGATIVE, settled.** Evidential's calibration (ECE) was worse
  than plain softmax in **20 of 20** matched comparisons, and worse than
  temperature-scaled softmax by 5–51×. This is no longer a single-config finding
  (Step 4.5) — it's now a decisively powered negative result. The honest thesis
  conclusion: **evidential heads do not calibrate better here, and shouldn't be sold as
  if they do.**
- **RQ3 (near-OOD detection) — SPLIT, as expected from earlier steps.** Vs. plain
  softmax scores: a clean, wide win (37–38 of 40 comparisons on both far and near-OOD).
  Vs. the tougher energy baseline: a much weaker showing (only 10/40 far-OOD, 14/40
  near-OOD wins). One nuance worth remembering: the near-OOD *advantage over softmax* is
  actually *larger* at 1-shot than 5-shot — exactly the "helps most in the lowest-data
  regime" pattern the proposal hypothesized.
- **RQ4 — deferred to Step 11**, since it needs real-hardware latency measurement, not
  just accuracy/calibration numbers.

### Phase 5, continued — Step 11 (currently IN PROGRESS)

**Step 11 — Efficiency + Pareto Frontier (RQ4). Code complete 2026-08-07, measurement
still pending.**
This step needs to actually *measure*, on real hardware, the four efficiency axes from
§2.11 (parameters, FLOPs, latency, peak memory) for every one of the grid's
configurations, then compute the Pareto frontier (§2.11) between "how good is the
uncertainty estimate" and "how fast/cheap is it to run." All of the supporting code is
built and unit-tested (56 new tests, zero regressions to the existing 233-test suite):
the measurement script, the Pareto-frontier computation, the plotting scripts, and the
result-table generators. A `--check-params-only` mode independently re-derived the
parameter counts for all 40 configurations from scratch and matched Step 10's numbers
exactly, which is a good sign the measurement pipeline itself is trustworthy.

What's still missing is the **canonical measurement run itself** — it needs a real
Kaggle GPU session (T4) plus a CPU session (standing in for an edge device, since no
Jetson Nano hardware is available). This has been attempted multiple times and has
actually **crashed three times in a row**, always at the same spot: an optional bonus
feature (`--include-reference-backbones`, which adds ViT/DeiT architecture-only
comparison rows) had a bug where it tried to measure FLOPs on a model that had already
been moved onto the GPU using a CPU-only "trace" tensor — a device-mismatch error.
Frustratingly, because the script only writes its results file at the very end, each
crash discarded the entire ~45–50 minutes of otherwise-successful core measurement. This
bug has now been fixed twice (2026-08-07 and a more thorough root-cause fix on
2026-08-09 that made the whole *class* of bug impossible, not just patched the one call
site that crashed) — but as of the last update, the fix has **not yet been verified on
an actual GPU**, since the development environment has no GPU or even `torch` installed
in some sessions. **This is the single next action the project needs**: run
`notebooks/step11_efficiency.ipynb` on Kaggle, confirm it now gets past the point it
previously crashed, and transcribe the resulting numbers into the writeup and
`docs/RESULTS_MASTER.md`.

### What's left: Step 12 (optional extensions) and Step 13 (thesis writing)

**Step 12 — not started, deliberately deferred until Step 11 fully closes**, since RQ4
is the last unanswered research question. If picked up, its highest-priority items are:
extending the Full-FT/Linear-Probe baselines to *every* (backbone, dataset) combination
for a fully fair comparison table (item 12.F); a cheap follow-up computing the "energy"
OOD score on the evidential head too, closing a scoring asymmetry (12.H); and a trained
ViT-Tiny/DeiT-Small arm to directly address the "why not just use a Transformer"
objection (12.I). Lower-priority items (ConvNeXt-Nano backbone, CUB-200 and ISIC
datasets, CIFAR-10-C corruption robustness) are explicitly in a "drop first if time is
short" order.

**Step 13 — thesis writing**, not started: turning all of the above into the actual
seven-chapter document (Introduction → Literature Review → Methodology → Experimental
Setup → Results → Discussion → Conclusion), with one subsection per RQ giving an explicit
numerical answer and figure/table reference.

---

## Part 5 — Current Status Snapshot (as of 2026-08-11)

| Step | Status | One-line verdict |
|---|---|---|
| 1 — Calibration fix | ✅ Closed | Fixed two real bugs; evidential loss now works as intended (on a toy setup). |
| 2 — Config-driven repo | ✅ Closed | Infrastructure only; enabled everything after it. |
| 3 — W&B + reproducibility | ✅ Closed | Frozen seeds + deterministic runs — the backbone of every later "trust these numbers" claim. |
| 4 — Real CIFAR-FS benchmark | ✅ Closed (negative result) | Step 1's finding did *not* survive real data — triggered Step 4.5. |
| 4.5 — Settle the science | ✅ Closed (Tier 3) | Evidential beats softmax decisively; ties/near-loses to "energy" baseline. |
| 5 — LoRA/BitFit/baselines | ✅ Closed | Full-FT doesn't collapse; PEFT gets ~98% of its accuracy at 1/1000th the params. |
| 6 — Adapter placement (RQ1) | ✅ Closed | **Parallel wins.** |
| 7 — OOD breadth (RQ3) | ✅ Closed | Evidential's edge is bigger on near-OOD than far-OOD, as hypothesized. |
| 8 — MobileNetV3 backbone | ✅ Closed | Tiny backbone ≈ free accuracy-wise; energy baseline starts winning here. |
| 9 — MiniImageNet dataset | ✅ Closed | Confirms a real pretraining-overlap confound; some Step 8 findings don't replicate. |
| 10 — Full 120-cell grid | ✅ Closed | **RQ1 confirmed, RQ2 settled negative, RQ3 split** (wins vs. softmax, weak vs. energy). |
| 11 — Efficiency/Pareto (RQ4) | 🟡 In progress | Code + tests done; canonical GPU measurement run still needs to succeed. |
| 12 — Optional extensions | ⬜ Not started | Deliberately waiting on Step 11. |
| 13 — Thesis writing | ⬜ Not started | Waiting on 11 and 12. |

**The one blocking task right now:** get `notebooks/step11_efficiency.ipynb` to run
successfully on a real Kaggle GPU session and produce `results/efficiency_table.json`,
then feed that into the Pareto plots and the final RQ4 writeup.

---

## Part 6 — Quick Glossary (cheat sheet)

| Term | One-line meaning |
|---|---|
| Backbone | A pretrained network kept frozen, used to turn images into feature vectors. |
| Adapter | A small trainable piece attached to a frozen backbone. |
| PEFT | Parameter-Efficient Fine-Tuning — train a tiny fraction of parameters instead of the whole network. |
| N-way K-shot | A few-shot task: pick among N classes, given K examples of each. |
| Episode | One simulated few-shot task (support examples + query examples). |
| Episodic training | Training on thousands of simulated episodes instead of one fixed dataset. |
| Prototype head | Classify by nearest average ("prototype") of the support examples' features. |
| Softmax | Standard output layer that forces probabilities to sum to 1; can be overconfident. |
| Evidential / Dirichlet head | Outputs "evidence," letting the model express genuine uncertainty. |
| Vacuity | The evidential model's uncertainty score — high = "not enough evidence." |
| Calibration / ECE | Whether stated confidence matches actual correctness. |
| Temperature scaling | A simple post-hoc fix that often makes softmax calibration hard to beat. |
| OOD (out-of-distribution) | An input that doesn't belong to any trained class. |
| Far-OOD / Near-OOD | Obviously different domain vs. subtly, deceptively similar-looking different classes. |
| AUROC | 0.5 = random, 1.0 = perfect at separating in-distribution from OOD. |
| Energy score | A strong, non-Bayesian OOD-detection baseline computed from ordinary softmax scores. |
| FLOPs / MACs | A hardware-independent proxy for how much compute one prediction costs. |
| Latency | Actual measured real-hardware prediction time. |
| Pareto frontier | The set of options where improving one metric necessarily worsens another. |

---

*For the numeric details behind any headline above, see the corresponding
`step_writeups/stepN.txt` and the running decisions log at the bottom of
[`progress.txt`](../progress.txt). For a reader-facing, one-step-at-a-time version of
Steps 5–10 specifically (with more analogies), see [`docs/explainers/`](explainers/).*
