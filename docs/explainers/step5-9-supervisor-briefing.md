# Steps 5–9 — Supervisor Briefing

> This is a meeting-prep summary, not a replacement for the full write-ups. Each step
> has its own detailed explainer in this folder ([step5.md](step5.md), [step6.md](step6.md),
> [step7.md](step7.md), [step8.md](step8.md), [step9.md](step9.md)) with the full analogy,
> the complete numbers, and every caveat. This file pulls the five steps into one
> narrative — what we found, how it compares to what came before, why we made each call,
> and what's still open — in the order you'd want to walk a supervisor through it.

## The setup, in one paragraph

The thesis (**B-PEFT**) starts from a pretrained, frozen image-recognition network (the
**backbone** — ResNet-18, sometimes MobileNetV3-Small) that already knows a lot about
images in general but nothing about our specific task. We attach a small, trainable
**adapter** so it can learn brand-new categories from just 5 example photos each
(**few-shot learning**), and we test two different final decision-makers (**heads**): a
plain **softmax** head (outputs a probability) and an **evidential** head (also reports
how much genuine *uncertainty* it has — closer to "how much evidence do I actually have"
than "how confident do I feel"). Every result below comes from the same fixed test: 600
held-out episodes, 5-way 5-shot, so numbers are directly comparable across steps.

**The four questions the whole thesis is answering:**

| RQ | Question | Status after Step 8 |
|---|---|---|
| RQ1 | Does *where* you attach the adapter matter? | **Answered** — yes, decisively (Step 6) |
| RQ2 | Does the evidential head calibrate better than softmax under a tiny parameter budget? | **No** — it's worse-calibrated everywhere tested; this is an honest negative finding, not a gap |
| RQ3 | Does a Bayesian-style loss improve near-OOD (hard, "unfamiliar but plausible") detection? | **Answered** — yes, and the edge is *bigger* on the hard cases (Step 7) |
| RQ4 | Latency vs. uncertainty-quality trade-off on edge hardware? | **Prep only so far** (Step 8); the actual latency measurement is Step 11 |

---

## Step 5 — Comparing four ways to nudge the model (LoRA, BitFit, Full-FT, Linear-Probe)

**What we tested:** Up to this point the thesis had only proven that *one* lightweight
adapter (a small "Bottleneck" add-on) works. Step 5 added three more methods — **LoRA**
(a tiny patch on one weight matrix inside the backbone), **BitFit** (touches only the
smallest possible pieces — bias values), and the two extremes the proposal specifically
asked for: **Full Fine-Tuning** (retrain the whole 11.18M-parameter backbone) and
**Linear-Probing** (retrain nothing at all). Each was tested with both heads — 8 result
files.

**Result:**

| Method | Params (trainable) | Accuracy (softmax) | Accuracy (evidential) |
|---|---|---|---|
| Full-FT | 11,176,512 | **0.905** (best overall) | 0.888 |
| Bottleneck | 16,912 | 0.875 | 0.884 |
| BitFit | 4,800 | 0.881 | 0.869 |
| Linear-Probe | 0 / 2 | 0.874 | — (degenerate, see caveat) |
| LoRA | 12,288 | 0.860 | 0.828 (weakest) |

**Compared to before:** this is the first step with more than one adapter method on the
table, so there's no direct "previous step" comparison — but it is the first real test of
whether the evidential-vs-softmax OOD advantage (established for Bottleneck alone,
Step 4.5) was a fluke of that one method. It wasn't: evidential vacuity beat softmax on
every OOD test for all four methods, by +0.010 to +0.134 AUROC.

**Why this decision mattered — a real surprise:** the original proposal predicted
Full-FT would *overfit and collapse* on 5-shot data (only 25 support images), and that
this would be the proof PEFT is necessary. It didn't collapse — it scored *highest*.
Root cause: episodic meta-training trains the shared backbone across hundreds of
episodes, not one; there's nothing to overfit *to* the way the proposal assumed. **We
reframed the thesis argument** from "Full-FT fails, PEFT rescues it" to "PEFT reaches
~98% of Full-FT's accuracy at 660–2,330× fewer trained parameters" — an efficiency story
instead of a viability story. This is a more honest and, frankly, more publishable claim.

**Open problem carried forward:** calibration. Evidential's ECE (~0.29–0.33) stayed
4–6× worse than softmax's after standard correction (0.005–0.041). This does not close
in any step that follows.

---

## Step 6 — Does adapter *placement* matter? (RQ1)

**What we tested:** kept the Bottleneck adapter's design identical and only moved
*where* it sits inside the backbone: **post_pool** (bolted on at the very end — the
Step 5 default), **serial** (inserted directly in the backbone's main data path — data
must pass through it), and **parallel** (runs alongside the main path, its output just
added back in). Serial and parallel were given exactly the same parameter count
(31,744) so any difference is attributable to placement alone, not capacity.

**Result:**

| Placement | Params | Accuracy | Best far-OOD (SVHN) |
|---|---|---|---|
| post_pool | 16,912 | 0.875–0.884 | 0.914 (reused from Step 5) |
| serial | 31,744 | 0.9145 (evid) / 0.9130 (softmax) | 0.889 |
| **parallel** | 31,744 | 0.9146 (evid) / 0.9125 (softmax) | **0.933** |

**Compared to Step 5:** moving the adapter *inside* the backbone (serial or parallel)
beat leaving it at the end by **+3 to +4 percentage points** — for roughly double the
parameters (16.9K → 31.7K). That gain also puts it within striking distance of Step 5's
Full-FT number (0.905) at 0.3% of the parameters.

**Why we picked parallel specifically:** accuracy alone was a dead heat between serial
and parallel (0.9145 vs. 0.9146). The tiebreaker was OOD detection — parallel beat
serial on *every* OOD test, most clearly on far-OOD (0.933 vs. 0.889 AUROC on SVHN).
This matches prior "Conv-Adapter" research (parallel > serial for CNN adapters), and we
had an open question whether that would replicate in a frozen-backbone, few-shot,
evidential setting — it did. **Parallel placement became the fixed configuration for
every step after this one.**

**Also resolved:** a predicted downside of the adapter's simple 1×1 shape ("losing
locality" — losing track of *where* in the image something is) never showed up; in-block
placement beat post_pool on accuracy regardless.

---

## Step 7 — Stress-testing the winner: near vs. far unfamiliar images (RQ3)

**What we tested:** took the Step 6 winner (parallel) and asked a sharper question than
"can it detect *anything* unfamiliar" — specifically, does the evidential edge hold up
on the *hard* cases? OOD images split into **far-OOD** (obviously, drastically
different — added a new pure-noise sanity check here) and **near-OOD** (still natural
photos, just of categories never trained on — CIFAR-100-heldout, TinyImageNet). All four
OOD sets were run together against the Step 6 winner for the first time.

**Result (AUROC, higher = better; evidential vs. softmax MSP):**

| | SVHN (far) | Gaussian noise (far) | CIFAR-100 (near) | TinyImageNet (near) |
|---|---|---|---|---|
| Evidential | 0.933 | 0.980 | 0.912 | 0.929 |
| Softmax MSP | 0.888 | 0.932 | 0.814 | 0.848 |
| **Gap** | +0.045 | +0.048 | **+0.098** | **+0.080** |

**Compared to Step 6:** the SVHN/CIFAR-100/TinyImageNet numbers are the *exact same*
parallel-placement numbers as Step 6 (verified byte-identical — a good reproducibility
check), just now placed alongside the new noise set in one consolidated table.

**Why this is the thesis's central RQ3 finding:** the gap isn't just present everywhere
— it's *bigger on the hard cases* (near-OOD, +0.08 to +0.10) than the easy ones
(far-OOD, +0.045 to +0.048). That's exactly the pattern the theory predicts: a plain
softmax "confidence" score stays falsely high on near-OOD images because they *look*
similar to known categories, while the evidential head keeps noticing it doesn't
actually have solid evidence. On the hardest test (CIFAR-100-near), this isn't abstract:
the evidential head's false-alarm rate was 0.393 vs. softmax's 0.667 — a large,
practical gap. The new noise sanity check also passed cleanly (both heads near-ceiling),
confirming the pipeline itself isn't broken before trusting the harder numbers.

**Caveat carried forward:** TinyImageNet is still "uncurated" (some class overlap with
training categories not filtered out) — the near-OOD claim leans on the clean
CIFAR-100-heldout set as the primary evidence, with TinyImageNet as corroboration.

---

## Step 8 — Does it all survive a much smaller backbone? (RQ4 prep)

**What we tested:** swapped ResNet-18 (11.18M frozen parameters) for **MobileNetV3-Small**
(927K — about 12× smaller, closer to what real edge/mobile hardware would run), and
re-ran the Step 6 winning recipe (parallel) plus post_pool as an internal control — 4
configs, same 600-episode protocol.

**Result:**

| | ResNet-18 (parallel) | MobileNetV3-Small (parallel) | Cost of the swap |
|---|---|---|---|
| Accuracy (softmax) | 0.9125 | 0.9090 | −0.35 pp (inside the ±0.5pp margin of error) |
| Accuracy (evidential) | 0.9146 | 0.9048 | −0.98 pp |
| Trunk size | 11,176,512 | 927,008 | 12.1× smaller |

**Compared to Step 6/7:** the accuracy hit was **0.3–1.6 percentage points** — nowhere
near the ~7–8pp gap MobileNetV3's general ImageNet performance deficit would predict.
Few-shot 5-way classification apparently doesn't need most of the capacity that
ImageNet-scale classification does. On top of that, **RQ1's finding got stronger, not
weaker**: within MobileNetV3, parallel beat post_pool by +3.7pp while using **2.75×
fewer** parameters (6,930 vs. 19,026) — on ResNet-18 that same win had *cost* more
parameters. Here it's a pure win, no trade-off to argue.

**Why this matters for the thesis's headline number:** the small-backbone/parallel
combination (6,928 trainable parameters, 927K trunk) reached 0.9090 accuracy —
*above* Step 5's Full-Fine-Tuning result (0.9047, 11.18M params) — using **1,613× fewer**
trained parameters, and within 0.35pp of the best result in the whole thesis so far
(ResNet-18/parallel, 0.9125). That's the strongest efficiency story the project has
produced.

**What did *not* fully carry over (reported honestly, not hidden):**
- Against a non-probabilistic comparison score called "energy," evidential had been
  roughly tied on ResNet-18; on MobileNetV3 it now **loses on 3 of 4 OOD pools**. "Vacuity
  is on par with energy" is a ResNet-18-only statement — it does not generalize as-is.
- The calibration gap got *wider* on the small backbone (35× worse than softmax vs.
  16.6× on ResNet-18) — but the write-up flags the leading suspect: this particular
  evidential run's checkpoint was selected at epoch 3, *during* the KL warm-up, unlike
  every other evidential run in the thesis. That's untested — it's a Step 10 item, not a
  settled conclusion.

No collapse anywhere; all four runs early-stopped normally. Latency itself — the actual
point of RQ4 — was **not measured yet**; Step 8 only established that accuracy survives
the smaller backbone. The real speed numbers are Step 11.

---

## Step 9 — A second dataset (MiniImageNet) — IN PROGRESS, NO RESULTS YET

**What we're testing:** everything through Step 8 used one dataset, CIFAR-FS. Step 9
asks whether the findings above are specific to that one benchmark, or general. It
re-runs the same 10 core comparisons (both backbones × both placements × both heads,
plus a linear-probe reference) on **MiniImageNet** instead.

**Why MiniImageNet specifically, and why the extra linear-probe reference run:**
MiniImageNet's 100 categories are drawn from ImageNet — the same enormous collection
both backbones were *pretrained* on. That means the backbone may have effectively
already "seen" these categories before the few-shot task even starts, unlike CIFAR-FS's
categories. Prior research (Chen et al. 2019, cited in the write-up) found that under
this exact condition, even a do-nothing baseline (frozen backbone, no adapter at all)
can become surprisingly competitive. Rather than assume that risk away, we added a
linear-probe run specifically to *measure* it directly on this dataset.

**Honest status — this is the one piece of news that isn't a finished result:**
Code is complete (the loader, dispatch layer, 10 configs, 35 new tests) and two real
bugs found while trying to run it were fixed:
1. A setup step assumed a dataset file would already be attached to the Kaggle
   session; it now falls back to downloading it, like every other dataset already does.
2. MiniImageNet's file host (Zenodo) was silently blocking our download's browser
   identity string — the opposite problem from an earlier fix that *required* a
   specific identity string for a different host. Fixed so each download picks its own.

**Two runs have been attempted, and neither has landed results in this repo yet:**
- The plain, single-session notebook's saved output shows all **10 configurations
  completed successfully on Kaggle** with a real results table — but the 266MB zip of
  those results never made it back out of Kaggle into this repo. Right now there are
  **zero** `phase5_mini*` result files locally.
- A second attempt used a more robust delivery system (a "GPU relay" that writes
  results straight into this repo as each piece finishes, instead of zipping everything
  up at the end) specifically to prevent that exact problem from recurring. It passed
  every safety check (213 local tests, a live GPU verification probe) and started
  transferring data, but was interrupted partway through uploading the large
  MiniImageNet cache files — before any training actually ran.

**Bottom line for the meeting:** Step 9 has **no numbers to report yet**, and nothing
above should be cited as a finding. The most likely fastest path to real numbers is
recovering the already-completed Kaggle run's zip file rather than re-running the whole
matrix a third time; failing that, resuming the interrupted relay run is the fallback.

---

## Scoreboard: how accuracy moved across steps 5, 6, and 8

*(Step 7 didn't change the model — it added OOD tests. Step 9 has no numbers yet.)*

| Step | Best configuration | Accuracy | Trainable params | Note |
|---|---|---|---|---|
| 5 | Full-FT (softmax) | 0.905 | 11,176,512 | Highest raw accuracy; reframed as inefficient, not "necessary" |
| 5 | Bottleneck (evidential) | 0.884 | 16,912 | The PEFT reference point going in |
| 6 | **Parallel Bottleneck (evidential)** | **0.9146** | 31,744 | Selected as the standing configuration |
| 8 | Parallel Bottleneck, MobileNetV3 (softmax) | 0.9090 | 6,928 | Beats Full-FT's accuracy at 1,613× fewer params |

## What's next

1. **Close out Step 9.** Recover the stranded Kaggle results zip (fastest path), or
   resume the interrupted relay notebook; either way, transcribe real numbers into
   `step_writeups/step9.txt`, confirm the test suite passes, and tick the boxes in
   `progress.txt`.
2. **Step 10 — the full grid (~120 runs).** Every step from 5 through 9 deliberately
   reused one training recipe that was only ever tuned for ResNet-18 on CIFAR-FS, to
   isolate whichever axis was being tested (adapter, placement, backbone, or dataset).
   Step 10 is where that gets fixed properly: a real per-backbone/per-dataset
   hyperparameter sweep (on the VAL split only, never the frozen test seeds), plus
   revisiting the one open Step 8 caveat — the MobileNetV3/parallel evidential run whose
   checkpoint was picked mid-KL-warm-up.
3. **Step 11 — the actual RQ4 measurement.** Step 8 only showed accuracy survives a
   smaller backbone; Step 11 is where we measure real latency/FLOPs/memory and build the
   Pareto plots that turn "the small backbone is almost free" into a hard latency number.
