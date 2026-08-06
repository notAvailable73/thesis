# Step 6 — Where Should the Adapter Go? (Adapter Placement Study)

> This step continues directly from [Step 5](step5.md). If you haven't read that one
> yet, the short version: a **backbone** is a pretrained image-recognition network we
> keep frozen (unchanged); an **adapter** is a small extra piece we add so the model can
> learn a new task from just a few example images (**few-shot learning**) without
> retraining the whole backbone.

## 1. What we did

In Step 5, several different *kinds* of small adapters were compared, but the one
called "Bottleneck" (a small piece that shrinks the data down to a bottleneck and back
up — think of it as a narrow-waisted hourglass shape) was always attached at the exact
same spot: right at the very end, after the backbone had already finished processing the
image into its final summary (its "pooled feature").

In Step 6, we kept the Bottleneck adapter itself unchanged and instead tested **three
different placements** for it inside the frozen ResNet-18 backbone:

- **post_pool** — the original placement from Step 4/5: bolted on at the very end, after
  all of the backbone's own processing is finished. (This one wasn't re-run; its numbers
  were reused from the earlier step.)
- **serial** — inserted *in the middle* of the backbone's internal processing, directly
  in the path the data must flow through. Every image's data has to pass *through* the
  adapter on its way to the next stage.
- **parallel** — inserted *alongside* the backbone's internal processing. The adapter
  runs on a copy of the data off to the side, and its output is simply added back in,
  without ever blocking or redirecting the main flow.

Both placements (serial and parallel) were tested with both decision-making heads
(softmax and evidential, explained in [Step 5](step5.md)), giving 4 new result files,
plus the 2 reused post_pool results from before — 6 result sets total, each measured
over 600 test episodes on the same CIFAR-FS few-shot task.

## 2. Why we did it

This step answers one of the thesis's core research questions (RQ1): **does *where* you
place an adapter matter, separately from *how big* it is?** Prior research on adapters
inside CNNs (the "Conv-Adapter" paper) found that placing an adapter running in parallel
tends to beat placing it in the direct, serial path, at least for classification tasks.
That earlier finding was made in a different setting, so it was an open question whether
it would hold true here — for a frozen backbone, tiny few-shot episodes, and both a
softmax and an evidential head.

To make the comparison fair, the *only* thing that changed across serial, parallel, and
post_pool was the location. The internal design of the adapter (the same 1×1 "bottleneck"
shape) stayed identical, and serial and parallel were deliberately given the exact same
number of trainable numbers (31,744), so any accuracy difference between them could only
be explained by placement — not by one having more capacity than the other.

## 3. The analogy

Picture the house from [Step 5](step5.md) again — the old, pretrained house we're
renovating for new tenants. In Step 5, every contractor bolted their renovation onto the
house at the very *end*, like adding a **shed attached to the driveway** after
everything else about the house was already finished — you walk out the front door,
across the yard, and into the shed separately.

In Step 6, we try two new placements for that same-sized addition, built right *into*
the house itself:

- **Serial placement**: we knock a hole in an existing hallway wall and insert the new
  room directly in the hallway's path. Now, to get from the living room to the kitchen,
  everyone in the house has to walk *through* the new room — it's mandatory, right in
  the main flow of foot traffic.
- **Parallel placement**: we build the new room as a **side alcove off the hallway**,
  with its own door back into the hallway a few steps further down. People can glance
  into the alcove and let its contents inform them, but the main hallway traffic keeps
  flowing straight through, undisturbed. Whatever happens in the alcove just gets added
  onto the hallway experience, rather than sitting directly in the path.

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| Shed attached to the driveway, added after the house is finished | **post_pool** placement — adapter added after the backbone's own processing is done |
| Hole knocked in the hallway wall, room inserted directly in the walking path | **serial** placement — adapter inserted directly into the backbone's data flow; data must pass through it |
| Side alcove off the hallway, with output simply folded back in | **parallel** placement — adapter runs alongside the backbone's flow; its output is added in without interrupting the main path |
| The renovation design/blueprint itself stays identical each time | The Bottleneck adapter's internal shape is unchanged — only its position varies |
| Two different final walkthroughs of the finished house | The two heads: softmax and evidential (see [Step 5](step5.md)) |

## 5. Benefit / what it improved

All numbers below are copied directly from the project's results (600 test episodes,
5-way 5-shot, CIFAR-FS):

- **Moving the adapter inside the backbone (serial or parallel) clearly beat leaving it
  at the end (post_pool).** Accuracy jumped from 0.875–0.884 (post_pool) to 0.913–0.915
  (serial/parallel) — a **+3 to +4 percentage-point gain** — for roughly double the
  trainable parameters (16,912 → 31,744).
- **That gain is now within striking distance of Step 5's Full Fine-Tuning result**
  (0.905 accuracy, 11,176,512 parameters) — but using only **0.3% of the parameters**
  Full-FT needed.
- **Serial and parallel tied almost exactly on plain accuracy** (0.9145 vs. 0.9146
  evidential; 0.9130 vs. 0.9125 softmax) — placement inside the backbone helped a lot,
  but serial-vs-parallel specifically didn't show up in the accuracy numbers.
- **Parallel pulled ahead on the harder measure: telling familiar images apart from
  unfamiliar ("out-of-distribution" or OOD) ones.** On the evidential head's uncertainty
  score, parallel beat serial on every OOD test set — most notably on the "far" OOD set
  (0.933 vs. 0.889 AUROC on SVHN, a set of street-number photos very different from the
  training images). This matches the earlier Conv-Adapter research finding that parallel
  placement tends to edge out serial placement.
- **The predicted downside of a simple 1×1-shaped adapter (that it "loses locality," or
  loses track of *where* in the image things are) did not show up here** — the in-block
  placements still outperformed post_pool on accuracy despite using that simple shape.
- Based on this, **parallel placement was selected as the winning configuration**
  going forward — the best balance of accuracy and OOD detection for a given parameter
  budget.

## 6. How this fits the overall thesis

This step directly answers Research Question 1 of the thesis: adapter placement is not
a minor implementation detail — it materially changes the accuracy/parameter trade-off,
delivering a 3–4 percentage-point accuracy gain for only about double the parameters
compared to the earlier post-pool placement. It also strengthens the thesis's broader
claim about the evidential head: even as the underlying adapter placement changed, the
evidential head's uncertainty score kept outperforming the plain softmax confidence
score on every out-of-distribution test — meaning that advantage is tied to the
*head design*, not to a lucky adapter setup. Settling on "parallel" as the best placement
also gives every later step in the thesis a single, justified configuration to build on,
rather than an arbitrary choice.

## 7. What's next, and why

Step 6 identified **parallel placement as the winning setup**, but only checked it
against three out-of-distribution test sets that were already in use from earlier steps
(SVHN as a "far" unfamiliar set, and CIFAR-100 / TinyImageNet as "near" unfamiliar
sets — see [Step 7](step7.md) for what "near" vs. "far" means). Step 7 takes that
winning parallel configuration and stress-tests it further: it adds a brand-new,
extremely easy "sanity check" unfamiliar-image test (pure random noise), and pulls all
of the out-of-distribution results together into one consolidated comparison — the
logical next move once you've picked your best configuration and want to characterize
it thoroughly before treating it as the thesis's headline result.
