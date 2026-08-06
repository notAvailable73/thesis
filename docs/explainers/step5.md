# Step 5 — Comparing Four Ways to Adjust the Model

## 1. What we did

The project uses a neural network — a computer program that has learned to recognize
patterns in images by looking at millions of example photos. Specifically, it uses
**ResNet-18**, a network that was **pretrained** (already trained by someone else) on
**ImageNet**, a huge, general-purpose photo dataset. This pretrained network is called
the **backbone**, because its job is to turn a raw photo into a set of numbers
("features") that describe what's in it.

The backbone already knows a lot about images in general. What it *doesn't* know is our
specific task: recognizing new categories of objects after seeing only a handful of
example photos of each. This is called **few-shot learning**. One "round" of
practice — show the model 5 example photos of each new category, then test it — is
called an **episode**. In Step 5, everything was tested on 600 of these episodes (a
standard, fixed number used throughout the project) to make sure the results are
reliable and not a fluke.

**Fine-tuning** means taking a pretrained model and nudging it toward a new task.
Retraining *all* ~11 million numbers ("parameters" or "weights") inside the backbone
from scratch, using only a handful of images per category, would normally cause the
model to memorize those few images instead of learning anything general — a failure
called **overfitting**. **Parameter-Efficient Fine-Tuning (PEFT)** is the idea of
nudging the model by changing only a *small* number of its parameters, leaving the rest
frozen (untouched).

In Step 5, we built and tested **four different ways** of doing this nudging, all on the
same task (CIFAR-FS, a standard few-shot image dataset) and run through the same
training loop and the same parameter-free decision-making method (comparing a new
photo's features to the average, or "prototype," of each category):

- **LoRA** — adds one small, simplified patch that adjusts a single weight matrix inside
  the backbone (12,288 trainable numbers).
- **BitFit** — touches almost nothing: only the smallest possible pieces, the "bias"
  shift values inside the backbone (4,800 trainable numbers).
- **Full Fine-Tuning (Full-FT)** — unfreezes and retrains the *entire* backbone
  (11,176,512 trainable numbers).
- **Linear-Probing** — freezes the backbone completely; nothing inside it changes at
  all (0 or 2 trainable numbers, depending on setup).

Each of the four was also tested with two different "heads" — the final
decision-making layer that turns features into a classification:

- **Softmax head** — outputs a standard probability for each category.
- **Evidential head** — outputs something closer to "how much evidence do I have for
  each category," which lets it express genuine uncertainty ("I don't know") rather
  than just a probability.

4 methods × 2 heads = **8 result files**, each reporting accuracy and other metrics over
600 test episodes.

## 2. Why we did it

An earlier step (Step 4/4.5) had already built and tested one PEFT method — a small
add-on piece called a "Bottleneck" adapter — and it worked well. But the thesis's
research plan always called for comparing *multiple* PEFT methods against each other,
and against two extreme reference points:

- **Full Fine-Tuning**, the maximum-effort approach (change everything), and
- **Linear-Probing**, the zero-effort approach (change nothing).

Without those two extremes on the table, there's no way to know whether the PEFT
methods are actually landing in a *good* spot on the effort-vs-accuracy trade-off — you
need the full range, from "touch nothing" to "touch everything," as a ruler to measure
against.

There was also a specific prediction to test: the original project proposal expected
Full Fine-Tuning to overfit badly and perform *worse* than the lightweight PEFT methods,
because updating 11 million numbers from a tiny handful of images per category is
normally a recipe for memorization, not learning. Step 5 existed partly to check whether
that prediction actually held up under the project's specific training setup — it did
not (see §5).

## 3. The analogy

Picture the pretrained backbone as **an old, sturdy house that a previous owner already
built and furnished**. The house is structurally sound and full of good, general-purpose
craftsmanship (it "knows" a lot about images in general, from ImageNet), but it wasn't
built with *your* specific new tenants in mind — recognizing brand-new categories from
just a few photographs each.

You, the new owner, want to adapt the house for these new tenants. But you don't want
to demolish and rebuild it — that's expensive, slow, and risky, especially since you
only have a few "reference photos" of what the new tenants need. So instead you call in
a series of different contractors, each with a different renovation philosophy, and
compare the results:

- One contractor makes a single small, clever structural tweak — one reinforced beam in
  one wall — instead of touching the whole house.
- Another contractor touches nothing structural at all; they just repaint the trim and
  adjust a few small fixtures.
- A third contractor guts and rebuilds the entire house from the studs.
- A fourth contractor changes nothing whatsoever — you just move in and use the house
  exactly as it was left.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| The old, pre-built house | The pretrained ResNet-18 backbone (trained on ImageNet) |
| Renovating the house for new tenants | Fine-tuning the backbone for the new, few-shot task |
| Only having a few reference photos of what the tenants need | Few-shot learning — only 5 example images per new category |
| Contractor #1: one reinforced beam in one wall | **LoRA** — a small low-rank patch on one weight inside the backbone |
| Contractor #2: repaint trim, adjust fixtures only | **BitFit** — only the bias values change (4,800 numbers) |
| Contractor #3: gut and rebuild the whole house | **Full Fine-Tuning** — the entire backbone is retrained (11.18M numbers) |
| Contractor #4: change nothing, move in as-is | **Linear-Probing** — the backbone stays fully frozen |
| Two different final walkthroughs/appraisals of the house | The two heads: **softmax** (a plain probability estimate) vs. **evidential** (an estimate that also reports how much evidence/uncertainty it has) |

## 5. Benefit / what it improved

All numbers below are copied directly from the project's results (600 test episodes,
5-way 5-shot, CIFAR-FS):

- **Accuracy ranking (best to worst):** Full-FT/softmax (0.905) > Full-FT/evidential
  (0.888) > Bottleneck/evidential (0.884) > BitFit/softmax (0.881) >
  Bottleneck/softmax (0.875) > Linear-Probe (0.874) > BitFit/evidential (0.869) >
  LoRA/softmax (0.860) > LoRA/evidential (0.828).
- **The "Full-FT will overfit and collapse" prediction did not hold.** Full-FT/softmax
  scored the *highest* accuracy in the whole study (0.905) rather than the lowest. The
  honest explanation isn't a bug or data leak — it's that training happens across
  hundreds of episodes covering many images (not one single 25-image episode), so
  Full-FT ends up learning genuinely useful features rather than memorizing.
- **This reframes the thesis argument.** Instead of "Full Fine-Tuning fails, so PEFT is
  necessary," the honest finding is: PEFT methods reach roughly 98% of Full-FT's
  accuracy (e.g., Bottleneck 0.884 vs. Full-FT 0.905) while training 660–2,330 times
  *fewer* parameters (16,912 or 4,800 vs. 11,176,512) — an efficiency argument rather
  than a viability argument.
- **LoRA turned out to be the weakest PEFT method** (0.828 evidential / 0.860 softmax),
  most likely because it was deliberately given a small "patch" (only one 1×1 weight
  matrix, 12,288 parameters) — a capacity limit, not a flaw in the method itself.
- **The evidential head's uncertainty measure beat every softmax-based confidence score
  on telling in-category images apart from out-of-category ("OOD") images**, across all
  four trained methods, by +0.010 to +0.134 AUROC (a 0–1 score measuring how well two
  groups are separated; higher is better). This showed the project's central claim about
  evidential uncertainty wasn't just a fluke of the one Bottleneck method tested
  earlier — it holds up across different renovation strategies.
- **Calibration remained an open problem.** ("Calibration" means: if the model says
  "90% confident," is it actually right 90% of the time?) The evidential head's error on
  this measure (ECE ≈ 0.29–0.33) stayed about 4–6× worse than the softmax head's after a
  standard correction ("temperature scaling," ECE 0.005–0.041).

## 6. How this fits the overall thesis

The thesis (project name: **B-PEFT**, Bayesian Parameter-Efficient Fine-Tuning) is
built around comparing different lightweight ways to adapt a frozen backbone, and around
testing whether an evidential ("Bayesian-flavored") head gives more trustworthy
uncertainty estimates than a standard softmax head — especially under very tight
parameter budgets. Step 5 is the step that broadens the comparison from one PEFT method
to four, plus the two non-PEFT extremes the original proposal asked for. This matters
because it turns "the evidential head beats softmax" from a one-method observation into
a pattern that holds across genuinely different renovation strategies, which is exactly
the kind of generalization a thesis needs before making a claim in its conclusions.

## 7. What's next, and why

The next step (Step 6) took the one PEFT method that stayed constant in form across
Step 5 — the small Bottleneck adapter — and asked a different question: not "how big
should the renovation be," but "*where* in the house should it go." Step 5 only ever
tested the Bottleneck adapter bolted on at one fixed spot (after all the backbone's own
processing was done). Step 6 tests inserting that same-sized adapter at different points
*inside* the backbone itself, to see whether placement — not just size — changes the
accuracy/parameter trade-off.
