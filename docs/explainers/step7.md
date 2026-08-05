# Step 7 — Testing the Model Against "Strangers" (Out-of-Distribution Detection)

> This step continues directly from [Step 6](step6.md), which selected **parallel
> placement** (see that file for what this means) as the best adapter setup. If you
> haven't read [Step 5](step5.md) or [Step 6](step6.md) yet, the short version: we have
> a pretrained, frozen image-recognition network (the **backbone**) with a small trained
> add-on (the **adapter**) that lets it learn new categories from just a few example
> images, and two different styles of final decision-maker (**head**): a plain
> **softmax** head (outputs a probability) and an **evidential** head (also reports how
> much *uncertainty* it has).

## 1. What we did

Every step so far tested the model on images it was *supposed* to recognize. Step 7
instead asks: **can the model tell when it's looking at something it was never trained
to recognize at all?** This is called **out-of-distribution (OOD) detection** — "the
distribution" is just the set of image categories the model learned from, and
"out-of-distribution" means an image from *outside* that set entirely, not just a new
example of a known category.

OOD test images come in two flavors, which the field calls **near** and **far**:

- **Far-OOD** — images that are obviously, drastically different from anything the
  model trained on. Easy, in principle, for any reasonable model to flag as unfamiliar.
- **Near-OOD** — images that are still natural photos, plausible-looking, but of
  categories the model never saw. Much harder to flag correctly, because the model's
  usual confidence signals can stay high even though it's wrong.

Before Step 7, the project already tested two OOD sets: SVHN (street-number photos,
used as a far-OOD set) and CIFAR-100 / TinyImageNet (used as near-OOD sets, since
they're still natural photos of unseen categories). In Step 7 we added a brand-new,
extreme far-OOD test: **pure random noise images** (seeded Gaussian/"static" noise,
clamped to a fixed range) — the simplest, most unambiguous "this is definitely not a
real photo" test possible. We then re-ran the Step 6 winner (parallel placement, both
heads) against **all four** OOD sets together (SVHN, the new noise set, CIFAR-100,
TinyImageNet) and consolidated everything into one results table and one comparison
plot, so the near-vs-far pattern could be read at a glance.

## 2. Why we did it

This step answers the thesis's third research question (RQ3): **does the evidential
head's uncertainty hold up better than the softmax head's confidence, specifically on
the *hard*, near-OOD cases — not just the easy, far-OOD ones?** Prior research on this
kind of uncertainty modeling predicts that a plain softmax "confidence" score tends to
stay falsely high on near-OOD images (because they look similar to known categories),
while a genuine uncertainty estimate like the evidential head's should keep noticing
that it doesn't actually have solid evidence for any category.

The new random-noise test set serves a different purpose: it's a **sanity check**, not
a real test of skill. If the model can't even tell pure static apart from a real photo,
something in the pipeline would be broken. Passing this easy check first makes the
harder near-OOD results trustworthy.

## 3. The analogy

Picture the renovated house from [Step 5](step5.md) and [Step 6](step6.md) — the one
with the small side-alcove addition built in parallel to the hallway. Now imagine it's
been fitted with a **security system**, and we're testing whether it can tell residents
apart from strangers at the door.

Two different security guards are on duty, and we're comparing how well each one does:

- The **softmax guard** just gives a gut-feeling percentage: "I'm 90% sure this is a
  resident." That's it — one number, no explanation of how much evidence backs it up.
- The **evidential guard** instead reasons about the *evidence itself*: "here's how much
  I actually know about this person" — and is willing to say "I genuinely have very
  little evidence either way" when that's true, rather than forcing out a confident
  number.

We test both guards against two kinds of strangers:

- **Far strangers** — someone wearing an obvious costume, or static noise on the porch
  camera. Blatantly, unmistakably not a resident. Any reasonable guard should catch
  these easily.
- **Near strangers** — a look-alike who resembles a resident closely enough that a
  guard going purely on gut-feeling confidence might wave them through. This is the real
  test of a *good* guard.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| Testing the security system against strangers at the door | Out-of-distribution (OOD) detection |
| An obvious costume / static on the porch camera | Far-OOD test sets: SVHN (street numbers) and the new **Gaussian noise** set |
| A close look-alike who isn't actually a resident | Near-OOD test sets: **CIFAR-100** (held-out categories) and **TinyImageNet** |
| The softmax guard's gut-feeling percentage | Softmax head's confidence score (MSP — "max softmax probability") |
| The evidential guard's "how much evidence do I actually have" reasoning | Evidential head's uncertainty score (**vacuity** — literally "how empty of evidence") |
| The guard's overall track record, tested across many strangers | **AUROC** — a 0–1 score for how well a method separates residents from strangers over many trials; higher is better |
| How often a stranger fools the guard when the guard is instructed to be 95% sure | **FPR@95** — the false-alarm rate when the system is tuned to catch 95% of real strangers; lower is better |
| A quick, obvious test to make sure the alarm isn't broken at all | The new Gaussian-noise sanity check |

## 5. Benefit / what it improved

All numbers below are copied directly from the project's results table
(`phase4_ood_table.json`; parallel placement, 600 test episodes, 500 images per OOD set):

**AUROC (higher = better separation of familiar vs. unfamiliar images):**

| Head | SVHN (far) | Gaussian noise (far) | CIFAR-100 (near) | TinyImageNet (near) |
|---|---|---|---|---|
| Evidential (vacuity) | 0.933 | 0.980 | 0.912 | 0.929 |
| Softmax (MSP) | 0.888 | 0.932 | 0.814 | 0.848 |
| Evidential's edge | +0.045 | +0.048 | **+0.098** | **+0.080** |

- **The evidential head beat the softmax head on every single OOD test set** — but the
  key finding is *where* the gap is biggest: it's small on the easy far-OOD sets
  (+0.045, +0.048) and **noticeably larger on the harder near-OOD sets** (+0.098,
  +0.080). This is exactly the pattern RQ3 predicted.
- **FPR@95 tells the same story, more sharply.** On the hardest test (CIFAR-100 near-OOD),
  the evidential head's false-alarm rate was 0.393 versus the softmax head's 0.667 — a
  large practical gap in a setting where the model has to make real accept/reject calls.
- **The random-noise sanity check passed.** Both heads scored near the top of the scale
  on pure noise (evidential AUROC 0.980; softmax MSP 0.932; a third, non-probability-based
  "energy" score reached 0.984) — confirming the whole pipeline is working correctly at
  the easy end before trusting the harder near-OOD numbers.
- Four new automated tests (`test_gaussian_ood.py`) were added to check the new noise
  generator behaves correctly (right shape, right value range, reproducible with a fixed
  seed), bringing the project's total automated test count to 107 passing.

## 6. How this fits the overall thesis

This step delivers the thesis's central RQ3 result: it's not enough to show the
evidential head is *sometimes* better at spotting unfamiliar images — the thesis's real
claim is that it's better *specifically where it counts most*, on the hard near-OOD
cases that a real deployed system is most likely to get wrong quietly and confidently.
By consolidating far and near OOD results into one table on the single best-performing
configuration (parallel placement, from [Step 6](step6.md)), this step turns three
separate steps of adapter experiments into one clean, defensible headline finding:
under a tight parameter budget, an evidential/Bayesian-flavored head buys meaningfully
better uncertainty exactly on the hardest, most realistic cases of unfamiliar input —
at the cost of the calibration gap noted back in [Step 5](step5.md).

## 7. What's next, and why

According to the project's own tracking file (`progress.txt`), the next planned step is
**Step 8: swapping in a different, smaller backbone (MobileNetV3-Small)**, in
preparation for the thesis's fourth research question (RQ4), which is about the
trade-off between speed and uncertainty quality on lightweight/edge hardware. Everything
through Step 7 used the same ResNet-18 backbone throughout, so the next logical move —
now that a winning adapter placement and a validated uncertainty advantage are both
established — is to check whether those same findings still hold on a backbone built for
lightweight, real-world deployment rather than a research benchmark. This wasn't stated
inside the Step 7 write-up itself, so treat it as the documented next step from the
project tracker rather than a claim made by Step 7's own results.
