# Step 10 — The Full Inspection: Every Combination, Side by Side

> Continues from [Step 9](step9.md); see [Step 5](step5.md) for the core vocabulary
> (backbone, adapter, few-shot learning, softmax vs. evidential head).
>
> **Companion documents:** the complete numeric tables live in
> [RESULTS_MASTER.md](../RESULTS_MASTER.md); the technical write-up is
> `step_writeups/step10.txt`.

## 1. What we did

Steps 5 through 9 each changed **one thing** and measured what happened: which adapter
method is best (Step 5), where to attach it (Step 6), how well it spots unfamiliar
images (Step 7), what a smaller backbone costs (Step 8), and whether any of it survives
a second dataset (Step 9). Each of those steps answered its own question well, but each
one also held everything *else* fixed — so the findings sat in separate pockets, tested
under slightly different conditions, and never directly against each other.

Step 10 ran **all of it at once**: a full grid of every meaningful combination of

- **2 datasets** — CIFAR-FS and MiniImageNet
- **2 difficulty levels** — 5 example images per category ("5-shot"), and the much
  harder 1 example per category ("1-shot")
- **2 backbones** — ResNet-18 and the smaller MobileNetV3-Small
- **4 adapter methods** — the parallel bottleneck adapter, LoRA, plus two mandatory
  reference points: full fine-tuning (retrain everything) and linear probe (retrain
  nothing)
- **2 head types** — softmax and evidential

That is **40 distinct configurations**, each trained **3 times** with different random
starting points to make sure the results aren't luck: **120 training runs**, every one
evaluated on the same frozen set of 600 test episodes. It took **36.3 hours** of GPU
time across three Kaggle notebook sessions.

Crucially, every run used **exactly the same training recipe** — same learning rate,
same settings, no per-combination tuning. That was a deliberate choice, and Section 5
explains both why it was the right one and what it costs.

**Result: all 120 runs completed.** No missing cells, no errors, 40 out of 40
combinations with all 3 repeats present.

## 2. Why we did it

Two reasons, one obvious and one that matters more than it sounds.

**The obvious one:** a thesis needs a single results table that a reader can look at and
see every claim supported in one place. Findings scattered across five steps, each
measured under slightly different conditions, are not that table. Step 10 produces it.

**The one that matters more:** when you only test one thing at a time, you can't tell a
real finding from a lucky one. Step 4.5 found the evidential head detecting unfamiliar
images better than the alternatives — in *one* configuration. Is that a property of the
evidential head, or a property of that particular backbone, dataset, and difficulty
level? There is no way to know from one measurement. Running 40 configurations turns
"we saw this once" into "we saw this in 37 of 40 comparisons" — or, just as valuably,
into "actually, that didn't hold up." Both happened here. See Section 5.

## 3. The analogy

Every step so far sent **one renovation crew** to **one house** and had **one inspector**
check **one thing** — the plumbing this time, the wiring next time, the insulation the
time after. Useful reports, but you could never lay them side by side, because each
inspection happened in a different house under different weather.

Step 10 is the **full property survey**. Every crew, every house type, both
neighborhoods, both budget levels — 40 combinations — with the *same* inspector using
the *same* checklist on the *same* day, and each job done three separate times to make
sure a good result wasn't just a good crew having a good morning.

The deliberate constraint is that every crew got **identical instructions and identical
tools**. Nobody was allowed to adapt their method to the particular house they were
assigned. That makes it a genuinely fair race — the differences you see are differences
between the *methods*, not between how much attention each one happened to get. The
price is that no single crew is shown at its personal best. That trade was made on
purpose, and it's the honest limitation of the whole survey.

One more thing the full survey caught that no single inspection could: two of the
"crews" — the retrain-everything crew and the retrain-nothing crew — turned in **byte-
for-byte identical work all three times**. Not suspicious, as it turns out, but genuinely
informative; Section 5 explains why, and why it slightly weakens one of the claims.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| The full property survey, every crew and house at once | The **MVT grid** — 40 configurations × 3 seeds = 120 training runs |
| Two neighborhoods | CIFAR-FS and MiniImageNet datasets |
| Two budget levels | 5-shot (5 example images per category) and 1-shot (just 1) |
| Two house types | ResNet-18 and MobileNetV3-Small backbones |
| Four crews | Parallel bottleneck adapter, LoRA, full fine-tuning, linear probe |
| Same inspector, same checklist, same day | The same 600 frozen test episodes for every run |
| Each job done three times | 3 random seeds (42, 43, 44) per configuration |
| Identical instructions for every crew | One frozen training recipe across all 120 runs — no per-combination tuning |
| Two crews turning in identical work three times | Full fine-tuning and linear probe have no randomly-initialised parts, so the random seed has nothing to change |

## 5. Benefit / what it found

Every number below comes from `results/mvt_results.json`; the full tables are in
[RESULTS_MASTER.md](../RESULTS_MASTER.md).

### The parallel adapter beat LoRA every single time — 16 out of 16

Across both datasets, both difficulty levels, both backbones and both head types, the
parallel bottleneck adapter won by **2.1 to 8.3 percentage points**. There is no
configuration in the entire grid where LoRA came out ahead.

On the small MobileNetV3-Small backbone this is a clean sweep in both directions at
once: the parallel adapter is both **cheaper** (6,930 adjustable settings vs. LoRA's
10,754) *and* **more accurate**. There's no trade-off to weigh — LoRA is simply the
worse choice there. On ResNet-18 there is a real trade-off: the parallel adapter costs
about 2.6× more settings and buys 3–8 percentage points of accuracy.

This makes sense in hindsight. LoRA was invented for transformers — a different kind of
network architecture. This project uses convolutional networks, and an adapter shaped
like the network it's attached to fits better. The grid is what turns that plausible
story into evidence.

### A 7,000-setting adapter matched retraining an 11-million-setting network

On CIFAR-FS with 5 examples per category:

| Approach | Accuracy | Adjustable settings |
|---|---|---|
| Parallel adapter on ResNet-18 | **91.44 %** | 31,744 |
| Parallel adapter on MobileNetV3-Small | 90.74 % | **6,928** |
| Full fine-tuning (retrain everything) | 90.47 % | 11,176,512 |
| Linear probe (retrain nothing) | 87.41 % | 0 |

The ResNet-18 adapter **beats** full retraining by about 1 percentage point while
adjusting **0.28 %** as many settings. The MobileNetV3-Small adapter **matches** it with
**0.06 %** — about 1,600× fewer. For a thesis about running on small devices, this is
the headline.

Two honest qualifications. First, at the harder 1-shot difficulty the ordering flips and
full retraining wins by 2.6 points — with only one example per category, the extra
capacity genuinely helps. Second, the MobileNet result is a *match*, not a win: its
0.27-point margin is smaller than its own run-to-run variation, so "matches full
retraining at 0.06 % of the cost" is the claim that holds. The ResNet-18 result is the
one with margin to spare.

### The evidential head's calibration is worse. Every time. Without exception.

"Calibration" means: when the model says it's 90 % sure, is it right about 90 % of the
time? This was **RQ2**, one of the thesis's four core questions, and the honest answer is
a clear **no**.

Across all 20 head-to-head comparisons, the evidential head's calibration error was
worse than plain softmax **20 times out of 20** — and worse than *corrected* softmax by
factors ranging from **5× to 51×**. It didn't make up for it in accuracy either
(0.6 points worse on average).

A negative result across 20 comparisons, two datasets, two difficulty levels, two
backbones and four adapters is not a disappointment — it's a finding. It is far stronger
evidence than the single-configuration hint from Step 4.5, and it's exactly the kind of
result that stops the next research group spending six months rediscovering it.

### But that same head is *better* at spotting unfamiliar images

Here's the genuinely interesting part, and it's only visible because both things were
measured on the same 120 runs.

The evidential head — the badly-calibrated one — is a **better detector of images it
shouldn't recognise** than the standard confidence score, winning **93.8 %** of
comparisons against it and **95.0 %** against its temperature-corrected version.

So "how well-calibrated is it?" and "how well does it flag unfamiliar input?" turn out
to be **different questions with different answers**. A paper reporting only calibration
would have concluded the evidential head is useless. A paper reporting only unfamiliar-
image detection would have concluded it's great. Both would be wrong.

### The Bayesian advantage grows exactly where the thesis predicted

**RQ3** asked whether the Bayesian approach helps more when data is scarce. It does: the
evidential head's advantage at spotting near-miss unfamiliar images is **+0.064** with
one example per category versus **+0.043** with five. Less data, bigger advantage —
the predicted direction.

### And a finding that goes against the project's own earlier conclusion

There's a third way to flag unfamiliar images, called the **energy score**, which
requires no Bayesian machinery, no special training and no extra settings — it's
essentially free. Step 4.5 had concluded, from one configuration, that the evidential
approach beat it in most cases.

**Across 40 configurations, it doesn't.** The free energy score wins about 70 % of the
comparisons, including on the far-OOD cases where Step 4.5 had the evidential head
ahead.

The earlier finding wasn't wrong about its own configuration — it just didn't generalise,
and 40 configurations outrank one. This is recorded as a correction to the project's
official "known state of the science" note rather than quietly dropped, and the thesis's
claim has been narrowed to the version that survives: *among scores derived from the
model's own predicted probabilities, the evidential one is clearly best, and its
advantage grows as data shrinks — but a well-chosen alternative that doesn't use
probabilities at all still beats it.*

### A reproducibility check that had been stuck open since Step 8

This project promises that re-running the same configuration produces byte-identical
results. Steps 8 and 9 both left that promise **unverified** — the check needed
checkpoint files that were never saved.

The grid closed it for free. Two of its runs happen to repeat Step 6's exact
configuration, months later on different hardware — and they reproduced Step 6's saved
results **exactly**, on all 29 and all 55 recorded numbers respectively. The promise is
now demonstrated rather than asserted.

### Why two of the crews returned identical work three times

Full fine-tuning and linear probe produced **bit-for-bit identical results** for all
three random seeds, right down to which training epoch was selected.

This is expected, not broken. The random seed only affects parts of the model that start
from *random* values. Full fine-tuning starts from pretrained weights and the linear
probe has no adjustable parts at all — so neither has anything for the seed to change.
The adapters do, which is precisely why only they show variation.

It has one real consequence, recorded rather than glossed over: for those two baselines,
"3 repeats" is effectively **1 measurement with no error bar**, so any close-run claim
against them is weaker than three repeats would suggest.

## 6. How this fits the overall thesis

Step 10 is the thesis's central results chapter. It answers three of the four research
questions with evidence spanning 120 runs rather than one:

- **RQ1** (which adapter, at what parameter cost): answered — parallel bottleneck, 16/16,
  and at 0.28 % of full retraining's cost it still wins.
- **RQ2** (does the evidential head calibrate better): answered, **negative**, 0/20.
- **RQ3** (does the Bayesian prior help spot unfamiliar images when data is scarce):
  answered with a **split verdict** — clearly yes against probability-based scores and
  more so with less data, clearly no against the free energy score.
- **RQ4** (speed vs. uncertainty quality on real hardware): **not this step.** The grid
  measured the "settings count" half of that trade-off; actual timing on edge hardware
  is Step 11.

It also produced the two documents a supervisor or reviewer actually needs: the complete
40-configuration results table including macro-F1, and an explicit comparison against
published state-of-the-art work — including a frank account of why this project's
accuracy numbers **cannot** be placed next to standard few-shot benchmarks (the
backbones were pretrained on ImageNet, and MiniImageNet's categories *come from*
ImageNet). See [RESULTS_MASTER.md §4](../RESULTS_MASTER.md) — the case for
publishability rests on calibration, unfamiliar-image detection and parameter cost,
where the comparison is fair, not on raw accuracy, where it isn't.

## 7. What's next, and why

**Step 11 — RQ4: timing on real edge hardware.** The grid supplies one axis of the
speed-vs-reliability trade-off already (how many settings each configuration adjusts);
Step 11 needs the wall-clock measurements to complete the picture.

**Three items carried forward, explicitly rather than silently:**

1. **The deferred tuning study (10.9).** [Step 9's explainer](step9.md) predicted Step 10
   would re-tune settings per backbone/dataset combination. It deliberately didn't — a
   fair race requires identical instructions, and per-combination tuning would have made
   the 40 results incomparable. That study still exists as a separate, smaller job. Its
   original motivation was a checkpoint-timing problem seen twice in Steps 8 and 9; that
   problem **did not reappear anywhere in the grid**, so it's now less urgent — but "did
   not reappear in 4 cases" is weaker than actually testing it, so it stays on the list.
2. **Updating the project's "known state of the science" note** to reflect the energy-
   score correction above, before any later step relies on the outdated version.
3. **A from-scratch backbone control run.** This is the single highest-value addition to
   the publication case: it would let at least one number in this thesis be compared
   directly against published few-shot results, and remove the pretraining objection a
   reviewer will raise first.
