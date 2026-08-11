# Step 8 — Trying a Smaller House (MobileNetV3-Small Backbone)

> Continues from [Step 7](step7.md). Recap of the vocabulary you'll need: a
> **backbone** is a pretrained, frozen image-recognition network; an **adapter** is a
> small trained add-on that lets it learn new categories from a few example images
> (**few-shot learning**); **parallel placement** (from [Step 6](step6.md)) is the
> winning way of attaching that adapter, alongside the backbone's internal flow rather
> than blocking it; and the **evidential** vs. **softmax** heads are two different
> styles of final decision-maker, where evidential also reports how much uncertainty it
> has (see [Step 5](step5.md)).

## 1. What we did

Every step so far used the same backbone: ResNet-18, a network with about 11.2 million
frozen numbers ("parameters"). Step 8 swapped in a **second, much smaller backbone**:
MobileNetV3-Small, a network designed specifically to be lightweight — its own frozen
trunk has about 927,000 parameters here (roughly 12× smaller than ResNet-18).

We then re-ran the Step 6 winning recipe — the small Bottleneck adapter in **parallel
placement** — on this new, smaller backbone, plus a **post_pool** version (the older,
end-of-the-line placement from Step 5) so the placement comparison could be checked
again *within* this new backbone too. That's 4 configurations (2 placements × 2 heads),
each tested over 600 episodes on the same CIFAR-FS few-shot task used throughout the
project.

## 2. Why we did it

The thesis's fourth research question (RQ4) is about the trade-off between speed
(useful for lightweight/edge hardware) and the quality of the model's uncertainty
estimates. That question is meaningless with only one backbone to look at — you need at
least two, one big and one small, to see whether a trade-off even exists. Step 8 exists
to put a genuinely small backbone into the comparison, as prep work for that later
speed-focused analysis.

It also served as a check on everything found so far: does "parallel placement beats
post_pool" (Step 6) and "the evidential head's uncertainty beats softmax confidence on
unfamiliar images" (Steps 5–7) still hold true if you change the underlying house
entirely — not just where you renovate it? If those findings only worked on one specific
backbone, they'd be a much weaker thesis claim.

## 3. The analogy

So far, every renovation in this story (see [Step 5](step5.md), [Step 6](step6.md),
[Step 7](step7.md)) happened to the same old, sturdy, spacious house. Step 8 asks a
different question: **what if, instead of a big house, we tried the exact same
renovation plan on a much smaller, more modest house?**

Picture a small studio apartment instead of the big house — built by a different
contractor, with far fewer rooms and far less floor space overall, but still
pre-furnished with decent, general-purpose fixtures. We hire the *same* renovation crew,
with the *same* blueprint (the parallel-placement Bottleneck adapter), and have them do
the identical side-alcove renovation in this smaller building instead. Then we compare:
does the small apartment work almost as well as the big house for the new tenants, once
it's been renovated the same way? And does the same security system (from
[Step 7](step7.md)) still do a good job telling residents from strangers in this smaller
building?

## 4. Mapping analogy to reality

| Analogy element | Real technical thing |
|---|---|
| The original big, sturdy house | ResNet-18 backbone (~11.2M frozen parameters) |
| The new, smaller studio apartment | MobileNetV3-Small backbone (~927,000 frozen parameters) |
| Same renovation crew, same blueprint, new building | The Step 6-winning parallel-placement Bottleneck adapter, applied unchanged to the new backbone |
| Re-testing the old "shed on the driveway" renovation for comparison | The post_pool placement, re-tested on the new backbone as a control |
| Checking whether the smaller apartment holds up nearly as well as the big house | Comparing MobileNetV3-Small's accuracy against ResNet-18's, at matched placement |
| Re-testing the security system in the smaller building | Re-running the OOD ("stranger") detection tests on the new backbone |

## 5. Benefit / what it improved

All numbers below are copied directly from the project's results
(`phase5_backbone_table.json`; 600 test episodes, parallel = the Step 6 winner):

- **The small backbone cost far less accuracy than expected.** MobileNetV3-Small landed
  only **0.3 to 1.6 percentage points below ResNet-18** at matched placement (e.g.
  parallel/softmax: 0.9090 vs. ResNet-18's 0.9125), not the roughly 7–8 percentage
  points its general ImageNet performance gap might have suggested. Two of those four
  gaps (the softmax ones, −0.3pp) are actually smaller than the study's own statistical
  margin of error (±0.5pp over 600 episodes), so they can't even be called a real
  difference at this sample size.
- **Parallel placement beat post_pool on the new backbone too — and by more than it
  cost.** On MobileNetV3-Small, parallel placement won by about +3.7 percentage points
  in accuracy while using **2.75× fewer** trainable parameters than post_pool (6,928 vs.
  19,024). On ResNet-18 the same win had cost *more* parameters, not fewer. So on this
  smaller backbone, parallel placement is a win with no trade-off to argue about at all.
- **The small-backbone-plus-parallel-adapter combination essentially matched Step 5's
  Full Fine-Tuning result** (0.9090 vs. Full-FT's 0.9047 softmax accuracy) while training
  **1,613× fewer parameters** (6,928 vs. 11,176,512) on a trunk that is itself over 12×
  smaller.
- **The evidential head's advantage over plain softmax confidence survived the backbone
  swap**, on every one of the 8 test-set/placement combinations checked.
- **Two findings did NOT carry over cleanly, and are flagged honestly rather than
  glossed over:** (1) against a third, non-probability-based comparison score called
  "energy," the evidential head had been roughly tied on ResNet-18, but on
  MobileNetV3-Small it now *lost* on 3 of 4 test sets. (2) The gap between the
  evidential head's calibration (how well its stated confidence matches reality) and the
  softmax head's calibration got *wider* on the small backbone (35× worse vs. 16.6× on
  ResNet-18) — though the write-up notes this is partly because the softmax side got
  unusually good here, not purely because evidential got worse.
- 28 new automated checks were added, bringing the project's total passing test count
  to 135.

## 6. How this fits the overall thesis

Step 8 is direct groundwork for RQ4 — you cannot study a speed-vs-uncertainty trade-off
with only one backbone size on the table, so this step put a genuinely lightweight
second option into the picture. But it did more than just prep the ground: it acted as
a stress test on everything claimed in Steps 5–7. The headline finding — that a small,
efficient backbone loses almost none of its accuracy, and that the "evidential
uncertainty beats plain softmax confidence" claim survives the swap — makes the thesis's
core argument considerably stronger, because it's no longer tied to one specific choice
of network. The two things that *didn't* fully carry over (the energy-score comparison,
and the wider calibration gap) are reported as honestly as the wins, which is exactly
the kind of finding a thesis needs to state precisely rather than overclaim.

## 7. What's next, and why

Step 9 kept the *architecture* side of the study fixed (same two backbones, same
parallel-placement recipe) and instead varied a different axis: the **dataset** itself.
Having shown the adapter and placement choices generalize across two different-sized
backbones, the natural next question is whether they also generalize across a different
set of images entirely — which is what Step 9 set out to test using a second dataset
called MiniImageNet.
