# B-PEFT — Supervisor Presentation (Slide Contents)

**Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**

> **Honesty contract for this deck.** Every number below is traceable to `results/mvt_results.json`,
> `results/efficiency_table.json`, `progress.txt` or `docs/RESULTS_MASTER.md`. Nothing is estimated,
> rounded up, or projected. Where a claim has a caveat, the caveat is **on the same slide as the
> claim**, not hidden in a backup slide. Two of our four hypotheses did not pass, and those slides
> are written as plainly as the ones that did.

**Deck length:** 24 slides. Target ~28 min + questions.

---

## SLIDE 1 — Title

**B-PEFT: Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**

Status as of 2026-08-20:
- **11 of 13 implementation steps closed**
- **All four research questions have numerical answers** (RQ4 closed 2026-08-09)
- **120 training runs completed**, 0 errors, 36.3 GPU-hours
- Remaining: Step 12 (optional breadth) and Phase 6 (thesis writing)

---

## SLIDE 2 — The problem, in one slide

**Setting:** an edge device (phone / MCU-class hardware) must learn to recognise 5 new classes from
5 example images each — and it cannot retrain a large model, because it does not have the memory.

**What everybody measures:** accuracy.

**What actually matters on a device that no human is watching:**

| Question | Metric | Who reports it |
|---|---|---|
| Did it get the answer right? | Accuracy / macro-F1 | Everyone |
| When it says "90% sure", is it right 90% of the time? | ECE (calibration) | Almost nobody in this space |
| Does it notice when the input is something it has never seen? | OOD AUROC | Almost nobody in this space |
| Does it fit and run on the device? | Params / MACs / latency | The efficiency literature only |

**The thesis question:** if you adapt a frozen lightweight CNN with only a few thousand trainable
parameters, does the uncertainty machinery still work — or does it quietly break?

---

## SLIDE 3 — What we built

**Pipeline:** `frozen CNN backbone → tiny trainable adapter → head`

| Component | Choices tested | Trainable? |
|---|---|---|
| Backbone | ResNet-18, MobileNetV3-Small (both ImageNet-pretrained) | **Frozen** — 0 params updated |
| Adapter | Bottleneck-parallel, LoRA, Full-FT\*, Linear-Probe\* | **Yes** — 0 to 11.18M params |
| Head | Parameter-free prototype head | No trained weights |
| Head *interpretation* | Softmax vs. Evidential Dirichlet | 0 or 2 extra params |

\* mandatory baselines, not PEFT methods.

**Trainable budget actually used by the PEFT methods: 6,928 – 31,746 parameters** — between 0.06%
and 0.28% of the backbone.

**The Bayesian part in one line:** softmax is *forced* to output a confident distribution even for
a nonsense input; the Evidential Dirichlet head can represent "I have no evidence for anything."
That is the mechanism we set out to test.

---

## SLIDE 4 — The four research questions (as written in the proposal)

| RQ | Proposal wording | What it means in practice |
|---|---|---|
| **RQ1** | How does adapter placement (serial vs. parallel) within a CNN bottleneck affect few-shot accuracy vs. parameter count? | Where do we bolt the adapter on, and is it worth the parameters? |
| **RQ2** | Can Evidential Dirichlet Networks provide superior calibration vs. standard softmax when fine-tuning with <500 trainable parameters? | Are the Bayesian head's confidence numbers more honest? |
| **RQ3** | Does a Bayesian loss prior improve detection of **near**-OOD samples in low-data regimes? | Does it spot unfamiliar inputs better, especially with less data? |
| **RQ4** | What is the Pareto frontier between inference latency and uncertainty quality on edge hardware? | What should you actually deploy? |

⚠️ **Two honest scope notes to state now, not later:**
1. **RQ2 says "<500 trainable parameters."** Only the Linear-Probe cell (**2 params**) literally
   meets that. Our PEFT adapters sit at 6.9k–31.7k. We report the sub-500 case, but the headline
   RQ2 answer covers the whole range — this is a widening of the proposal's wording, and we say so.
2. **RQ1 was answered in two parts.** The literal serial-vs-parallel question was settled in Step 6;
   the main grid then compared the *winner* (parallel) against LoRA. Both results are on Slide 7.

---

## SLIDE 5 — The evidence base (how much we actually ran)

**This is the slide that makes everything else credible.**

| Axis | Values | Count |
|---|---|---|
| Dataset | CIFAR-FS, MiniImageNet | 2 |
| Shots | 1-shot, 5-shot (5-way throughout) | 2 |
| Backbone | ResNet-18, MobileNetV3-Small | 2 |
| Adapter | Bottleneck-parallel, LoRA, Full-FT\*, Linear-Probe\* | 4 |
| Head | Evidential, Softmax | 2 |
| **Unique configurations** | | **40** |
| Training seeds per config | 42, 43, 44 | 3 |
| **Total training runs** | | **120** |
| Test episodes per run | frozen seeds 0–599 | **600** |

**Run integrity:** `results/grid/_run_log.jsonl` = 123 `ok` + 1 `skipped_done`, **0 errors**.
`results/mvt_results.json` reports 120/120 cells, 40/40 groups with all 3 seeds, `missing_cells: []`.
Total wall time **36.3 hours**.

**Plus 5 OOD evaluation pools per run:** SVHN (far), Gaussian noise (far), CIFAR-100-heldout (near),
MiniImageNet-heldout (near), TinyImageNet (near).

---

## SLIDE 6 — Why the numbers can be trusted

| Guarantee | How it is enforced |
|---|---|
| **Frozen episodes** | 600 test episode seeds fixed in `configs/test_episodes.yaml` — never regenerated |
| **No test-set tuning** | All hyperparameter search ran on VAL seeds 10000–10099 only |
| **Byte-identical reruns** | Same config → byte-identical `metrics.json`; grid seed-42 cells reproduced Step 6's earlier committed metrics **exactly** (29/29 and 55/55 numeric keys) |
| **One frozen recipe** | All 120 cells use the same LR / rank / KL weight — deliberate, so the grid is a controlled comparison rather than 40 separately-tuned numbers |
| **Pre-registration (RQ4)** | Cost and quality axes and the tolerance band were written down *before* any latency number existed |

**Seed noise:** seed std exceeds 1 pp in only **2 of 40** configurations; everywhere else ≤0.74 pp.
→ Differences >2 pp between cells are real, not noise.

⚠️ **The one exception, stated up front:** `Full-FT*` and `Linear-Probe*` have **zero** seed
variance *by construction* — Full-FT starts from fixed pretrained weights and Linear-Probe has no
trainable tensor at all, so there is nothing for the seed to perturb. For those two rows, n=3 is
effectively n=1.

---

## SLIDE 7 — RQ1 result: adapter placement ✅ PASSED

**Part A — the literal proposal question (Step 6, serial vs. parallel):**

| Placement | Accuracy | Best OOD (evidential vacuity) |
|---|---|---|
| Serial (in-block) | 0.9145 | SVHN 0.889 / C100 0.904 / TIN 0.929 |
| **Parallel (in-block)** | 0.9146 | **SVHN 0.933 / C100 0.912 / TIN 0.929** |
| Post-pool | 0.875–0.884 | — |

→ Serial vs. parallel is an **accuracy tie**. Parallel wins the tiebreak on OOD. In-block beats
post-pool by 3–4 pp.

⚠️ **This replicates prior work, and the closest match is closer than a first check found.** TSA
(Task-Specific Adapters, arXiv:2107.00358, CVPR 2022) already runs this exact comparison — frozen
**ResNet-18**, serial vs. parallel adapter connection, **600 sampled episodic tasks** on held-out
domains, **parameter-free nearest-centroid head** — same backbone, same episode count, same
question, and finds *"residual [parallel] connections perform better than the serial one in almost
all cases."* Same answer as this slide. Conv-Adapter is a secondary precedent for a separate point
(locality preservation, Slide 19/20) — TSA, not Conv-Adapter, is the primary one for this specific
result.

**Part B — the winner vs. LoRA, across the full grid: 16/16 matched comparisons, +2.13 to +8.26 pp.
Not one configuration where LoRA wins.**

| Dataset | Shots | Backbone | Parallel bottleneck | LoRA | Δ |
|---|---|---|---|---|---:|
| CIFAR-FS | 5-shot | ResNet-18 | 91.44 (31,744 p) | 86.25 (12,288 p) | **+5.19** |
| CIFAR-FS | 5-shot | MobileNetV3-S | 90.74 (**6,928 p**) | 88.05 (10,752 p) | **+2.70** |
| MiniImageNet | 5-shot | ResNet-18 | 95.56 (31,744 p) | 91.56 (12,288 p) | **+4.00** |
| MiniImageNet | 1-shot | ResNet-18 | 85.03 (31,744 p) | 80.29 (12,288 p) | **+4.75** |

**On MobileNetV3-Small it is a strict Pareto win** — parallel bottleneck is *both cheaper*
(6,930 vs 10,754 params) *and more accurate*. LoRA is simply dominated there.

**Interpretation worth stating:** LoRA is a transformer-native reparameterisation. This is direct
evidence it should not be the default for convolutional backbones — this LoRA-specific comparison
is not run by TSA or Conv-Adapter, and is the part of this slide that's actually new.

---

## SLIDE 8 — RQ1 continued: the parameter-efficiency headline

**CIFAR-FS 5-shot, all four adapters:**

| Configuration | Trainable params | Accuracy | % of Full-FT budget |
|---|---:|---:|---:|
| Parallel bottleneck, ResNet-18 | 31,744 | **91.44 %** | 0.28 % |
| Parallel bottleneck, MobileNetV3-S | **6,928** | 90.74 % | **0.06 %** |
| Full fine-tuning, ResNet-18 | 11,176,512 | 90.47 % | 100 % |
| Linear probe (no adaptation) | 0 | 87.41 % | 0 % |

**The claim:** a **31,744-parameter adapter beats full fine-tuning of an 11,176,512-parameter
network by +0.98 pp** while training 0.28 % of the parameters.

⚠️ **Three caveats on the same slide as the claim:**
1. The MobileNetV3-Small row **matches** Full-FT (+0.27 pp) — it does **not** beat it. That margin
   is smaller than the cell's own seed spread. We phrase it as *matches*, deliberately.
2. This is a **cross-backbone** comparison: `full_ft` has never been run on MobileNetV3-Small, so
   the only Full-FT number that exists is ResNet-18's. See Slide 22 (item 12.F).
3. **At 1-shot the ordering flips** — Full-FT wins by 2.57 pp (81.14 vs 78.57). With one image per
   class, the extra capacity buys something the adapter cannot recover.

---

## SLIDE 9 — RQ2 result: evidential calibration ❌ FAILED (0/20)

**This was the central hypothesis of the thesis. It did not pass.**

| | Evidential ECE | Softmax ECE | TS-Softmax ECE |
|---|---|---|---|
| Range across all 20 matched configs | 0.2156 – 0.6234 | 0.0242 – 0.4476 | 0.0057 – 0.0642 |

- Evidential is worse than **plain** softmax in **20/20** matched pairs — by **1.4× to 9.1×**
- Evidential is worse than **temperature-scaled** softmax by **5.3× to 51.2×**
- Accuracy does not compensate: evidential is **0.61 pp worse** on average, winning only 7/20

**Two worked examples:**

| Config | Evidential | Softmax | TS-Softmax | Gap |
|---|---:|---:|---:|---:|
| CIFAR-FS 5-shot, R18, parallel | 0.3010 | 0.0670 | 0.0152 | 20× vs TS |
| MiniImageNet 5-shot, R18, parallel | 0.2938 | 0.0850 | **0.0057** | **51× vs TS** |

**It is not a tuning artefact.** A VAL-only sweep in Step 4.5 found the ECE surface **flat at
≈0.285–0.296** across the hyperparameter range tested — the gap does not close by tuning.

⚠️ **Fairness caveat we volunteer:** temperature scaling is a cheap post-hoc fix available to
softmax and **not implemented for evidential in this codebase**, so the TS column is structurally
favourable to softmax. It is still the right comparison — TS is standard and free — but the
asymmetry is real and stated.

---

## SLIDE 10 — Why RQ2 failed: the boundary, checked from three independent directions

**Why it failed (mechanism):** the evidential head must *learn* a mapping from "how much evidence"
to "how confident to sound." We froze the backbone and gave it as few as **2 trainable parameters**
and 5 images per class. There is not enough capacity or signal to fit that mapping, so its stated
confidence stops tracking its actual hit-rate.

**It is well-powered, not a single-config fluke:** 2 datasets × 2 shot regimes × 2 backbones × 4
adapters = 20 matched pairs, **zero exceptions**.

**Three independent papers, three different Bayesian mechanisms — all on the other side of the boundary:**

| Work | What's different from our setup | Result |
|---|---|---|
| **BEL** (arXiv:2207.13137) | Backbone is **trainable** (meta-trained end-to-end) | Calibration *improves* — 3.59% vs 14.69% ECE |
| **BayesAdapter** (arXiv:2412.09718, IJCV 2025–26) | Frozen, but same-class protocol (not disjoint) + adapter initialised from CLIP's text-encoder prototypes (a strong semantic head-start) | Calibration *improves* — ~2.5% ECE gain |
| **MetaQDA** (arXiv:2101.02833, ICCV 2021) | **Frozen backbone, true 600-episode disjoint-class few-shot — same protocol family as ours** — but its classifier is a large, meta-learned prior (~100K+ params, trained across thousands of episodes), not near-parameter-free | Calibration *improves a lot* — beats even a trained, non-Bayesian linear classifier on the same features |
| **Ours** | Frozen **and** near-parameter-free classifier (2–31,744 total params) **and** true disjoint-class episodic | Calibration **fails, 20/20** |

⚠️ **This corrects a framing we held until we checked MetaQDA.** We first read this as "Bayesian
calibration breaks when the backbone is frozen." MetaQDA disproves that version directly — its
backbone is frozen too, and its calibration is excellent (better than a trained linear classifier
on the same frozen features). **The real boundary is capacity in the probabilistic component
itself, not whether the backbone is frozen.** Calibration works when the classifier gets real room
to learn — a trainable backbone (BEL), a strong semantic head-start (BayesAdapter), or a large
meta-learned prior (MetaQDA). Ours is the one setting that denies it all three at once.

**The contribution framing:** we did not fail to reproduce BEL, BayesAdapter, or MetaQDA. We
located where the boundary sits, from three independent directions, with three different Bayesian
mechanisms — and it is precisely the edge-deployment corner, because a 256 kB device cannot afford
a trainable backbone, a foundation-model-scale adapter, or thousands of meta-training episodes.

---

## SLIDE 11 — The boundary is backed by theory, not just three papers — and it explains RQ3 too

**Bengs, Hüllermeier & Waegeman (NeurIPS 2022, arXiv:2203.06102)** prove, specifically for
Dirichlet/evidential classifiers (their Theorem 3.2 covers this exact case), that **no loss
function gives an evidential model an incentive to make its uncertainty magnitude match reality.**
They test this at multiple sample sizes (N = 100, 500, 1000) — **the failure does not shrink as
data grows**, so this is a structural property, not a small-data artefact we happened to land in.

**The one exception they name is the one that matters most to us:** relative/ranking uncertainty
("is A more suspicious than B") is fine on their own account — only the *magnitude* is untrustworthy.
That maps exactly onto calibration (RQ2, needs magnitude) vs. OOD detection (RQ3, needs ranking
only) — **the theory predicts our RQ2/RQ3 split before we ever measured it.**

**Shen et al. ("Is EDL a Mirage?", NeurIPS 2024, arXiv:2402.06160)** independently confirm the
non-vanishing-uncertainty result, and go further: **Section 5.2 formally derives that the evidential
evidence score belongs to the same mathematical family as the energy-based OOD score** (Liu et al.
2020 — already our Slide 13/14 baseline). Quote: *"EDL methods can be better understood as an
EBM-based OOD detector... rather than a statistically meaningful mechanism."*

**Why this matters for the RQ3 counterpoint (Slide 14):** it gives a theoretical reason — not just
an empirical grid result — for why a plain energy score does at least as well as our Bayesian
vacuity score. Vacuity may simply be a noisier version of the same signal energy computes directly.

⚠️ **What this theory does NOT do:** neither paper tests real images, a frozen CNN, few-shot
episodes, or PEFT adapters — and neither measures ECE directly. They prove the failure is *possible
in general*; they do not measure *how badly it bites, or where*, in a deployment-realistic setting.
**That measurement is what Slides 9–10 supply.** Theory said this kind of failure was possible;
nobody had shown it actually happens, this decisively, on real vision benchmarks anyone might
deploy.

---

## SLIDE 12 — Have we checked whether RQ2 is already answered? Yes — exhaustively

**What we searched:** every direct citation of BEL, PEFT-at-the-edge literature, adapter/LoRA/BitFit
+ uncertainty combinations, recent (2025–2026) evidential-deep-learning papers, Dirichlet-vs-softmax
ablation studies, and few-shot-uncertainty surveys.

**The one close call, checked and ruled out:** a preprint (arXiv:2602.15283) trains a tiny
(0.7K-parameter) evidential head on a frozen CIFAR-10 backbone and gets near-random accuracy (9.6%)
with a deceptively low ECE (0.004). **Its own authors disclaim it** — trained with plain
cross-entropy, not the correct evidential loss, and explicitly excluded from their comparative
claims. Not counted as evidence either way, but consistent with how fragile this exact regime is.

**One item we could not fully verify — flagged, not buried:** *"Be Confident in What You Know:
Bayesian-PEFT"* (Pandey, Pyakurel, Yu, **NeurIPS 2024**) is the single closest-*named* paper found —
same research group behind the theory work on Slide 11. Confirmed: it targets **ViT-scale
foundation models** (not a small edge CNN), fixes **under**confidence (the opposite miscalibration
direction from ours), and uses an **ensemble** of evidential components (real added capacity, not a
single near-parameter-free head) plus a "base-rate" mechanism leaning on pretraining priors. All
three point toward "another capacity-available case" — consistent with the boundary story — but PDF
and OpenReview access blocked six separate extraction attempts, so the exact protocol and numbers
are **not independently confirmed. Recommended: read this one yourself before the meeting.**

**The verdict:** nobody has run the exact controlled ablation RQ2 answers — same frozen tiny CNN,
same near-parameter-free head, evidential-vs-softmax interpretation only, true disjoint-class
episodic few-shot. Every related paper changes at least one of those axes at the same time it goes
Bayesian. **RQ2 is unscooped.**

---

## SLIDE 13 — RQ3 result: OOD detection ⚠️ SPLIT

**Against every softmax-probability score — a decisive win, exactly as the theory on Slide 11 predicts for ranking tasks:**

| Comparison | Far-OOD | Near-OOD | Overall win rate |
|---|---|---|---:|
| vacuity vs. **MSP** | **38/40**, mean Δ +0.111 | **37/40**, mean Δ +0.053 | **93.8 %** |
| vacuity vs. **TS-MSP** | **38/40**, mean Δ +0.127 | **38/40**, mean Δ +0.067 | **95.0 %** |
| vacuity vs. **energy** | 10/40, mean Δ −0.022 | 14/40, mean Δ −0.007 | **30.0 %** |

**The low-data hypothesis held — this is the specific RQ3 prediction:**

| Near-OOD advantage of vacuity | 1-shot | 5-shot |
|---|---:|---:|
| over MSP | **+0.0637** | +0.0431 |
| over TS-MSP | **+0.0725** | +0.0614 |

→ The Bayesian prior helps **more** exactly where there is **less** data. That is what RQ3 predicted,
and it survived contact with the grid.

**Concrete example** (CIFAR-FS 5-shot, R18, parallel, near-OOD C100): vacuity 0.9031 vs MSP 0.8179.

---

## SLIDE 14 — RQ3's honest counterpoint (volunteer this, do not let it be found)

**A training-free "energy" score beats our Bayesian score in ~70 % of comparisons.**

- Energy needs **no evidential training, no Bayesian prior, no extra parameters** — it is computed
  from the same raw logits a plain softmax model already produces.
- Across the grid: vacuity wins only **10/40** far-OOD and **14/40** near-OOD against it.

**This is no longer just an empirical surprise — Slide 11 gives it a mechanism.** Shen et al.'s
formal result that EDL's evidence score is mathematically in the same family as the energy score
means vacuity losing to energy is close to what theory would predict, not a fluke of our grid.

⚠️ **This corrects our own earlier recorded finding.** Step 4.5 (a *single* configuration) concluded
evidential was roughly on par with energy — winning far-OOD and CIFAR-100-near, losing only
TinyImageNet-near. **The 120-run grid shows that does not generalise.** The correction is recorded
in `CLAUDE.md` and `progress.txt`'s decisions log (2026-08-06).

**The defensible claim, narrowed to what the data supports:**

> *Among scores derived from the model's own predictive distribution, Dirichlet vacuity is a
> substantially better OOD ranker than max-softmax-probability with or without temperature scaling,
> and its advantage grows in the lowest-data regime — but it does not beat a well-chosen
> logit-space score.*

**What we would tell a practitioner today:** for OOD detection alone, use softmax + energy. Evidential
is the better *native probabilistic* option, not the best option overall.

---

## SLIDE 15 — The cross-cutting finding: calibration and OOD are decoupled

**The same head that is 5–51× worse calibrated is simultaneously the better OOD detector 94 % of the time.**

| | Calibration (RQ2) | OOD detection (RQ3) |
|---|---|---|
| What it demands | The confidence **number** must be honest (magnitude) | Unfamiliar must score **lower** than familiar (ranking) |
| Evidential's result | ❌ 0/20 | ✅ 93.8 % / 95.0 % vs softmax scores |
| Why the difference | Requires a correctly-fitted evidence→confidence mapping — which our parameter budget cannot fit | Only requires directional consistency, which survives a badly-scaled mapping |

**Why this matters as a result:** these are routinely treated as one property ("uncertainty quality").
They are not — and Slide 11's theory says this split should be expected in general, not just here.
**Reporting ECE alone would have hidden this entirely; so would reporting AUROC alone.**
We only see it because both were measured on the same 120 runs.

---

## SLIDE 16 — RQ4 result: efficiency and Pareto frontier ✅ PASSED (closed 2026-08-09)

**Measured on Kaggle T4 (GPU) + Kaggle CPU. Axes pre-registered before any number existed.**

| Backbone | Adapter | Head | Trainable | Total params | GMACs | CPU ms/img (1 thr) | GPU ms/img | On frontier? |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| ResNet-18 | Bottleneck-par | Evid. | 31,746 | 11,208,258 | 1.8306 | 62.38 | 3.36 | yes |
| ResNet-18 | Bottleneck-par | Softmax | 31,744 | 11,208,256 | 1.8306 | 62.18 | 3.35 | yes |
| ResNet-18 | LoRA | Softmax | 12,288 | 11,188,800 | 1.8192 | 59.76 | 2.87 | yes |
| ResNet-18 | Full-FT\* | Softmax | 11,176,512 | 11,176,512 | 1.8186 | 61.52 | 2.82 | yes |
| **MobileNetV3-S** | **Bottleneck-par** | **Evid.** | **6,930** | **933,938** | **0.0593** | **11.86** | 6.53 | **yes** |
| MobileNetV3-S | LoRA | Softmax | 10,752 | 937,760 | 0.0586 | 11.44 | 5.88 | yes |

**Recommended operating point:** MobileNetV3-Small + parallel bottleneck + evidential on CIFAR-FS —
**11.86 ms/image, 6,930 params, near-OOD AUROC 0.870 (1-shot) / 0.919 (5-shot)**.
On MiniImageNet it shifts to ResNet-18 (62.38 ms, AUROC 0.870 / 0.958) because MobileNetV3-Small
falls outside the pre-registered accuracy tolerance there.

*Note: 933,938 is our **measured** deployed parameter count. The literature's "2.5 M" for
MobileNetV3-Small includes the ImageNet classifier head, which our stack does not deploy.*

---

## SLIDE 17 — RQ4's most novel finding

**Backbone choice drives latency. Adapter choice does not.**

| Comparison | Parameter difference | Latency difference |
|---|---:|---:|
| ResNet-18 vs. MobileNetV3-Small (matched adapter) | 12× total params | **5.12×** (62.18 → 12.14 ms) |
| Parallel bottleneck vs. LoRA on ResNet-18 | **2.58×** trainable params | **3.9 %** |
| Parallel bottleneck vs. LoRA on MobileNetV3-Small | LoRA has **more** params | LoRA is **5.8 % faster** |

**Why:** both adapters' parameter deltas are swamped by the frozen trunk's forward-pass cost.

**The practical rule this gives a deployer:**
> The **adapter** decision is an **accuracy** decision (RQ1, up to +8.3 pp).
> The **backbone** decision is the **latency** decision (RQ4, 5.12×).

**"Evidential uncertainty is free at inference" — now measured, not assumed:**
- Mean |latency delta| evidential vs softmax at matched (backbone, adapter): **1.29 %**
- This session's own measurement-noise floor: **5.91 %** → the head cost is **below noise**
- Full uncertainty-scoring stage for a 75-query episode: **1.09 %** of one image's backbone forward pass

⚠️ **Conditional we state:** under the primary native-score reading, evidential heads anchor every
strict frontier. Under a "softmax gets its best score (energy)" reading, evidential's frontier
presence on CIFAR-FS 5-shot **collapses to zero**. The Pareto claim is native-score-conditional.

---

## SLIDE 18 — Scorecard

| RQ | Question | Verdict | Strength of evidence |
|---|---|---|---|
| **RQ1** | Adapter placement / parameter efficiency | ✅ **PASSED** | 16/16 matched, +2.13–8.26 pp, zero exceptions |
| **RQ2** | Evidential calibrates better? | ❌ **FAILED** | 0/20 matched, 1.4×–51× worse, VAL sweep confirms flat, boundary checked from 3 independent papers + 2 theory papers |
| **RQ3** | Bayesian prior helps near-OOD? | ⚠️ **SPLIT** | 93.8–95.0 % vs softmax scores ✅; 30 % vs energy ❌ |
| **RQ4** | Latency vs. uncertainty Pareto | ✅ **PASSED** | Measured on real hardware, axes pre-registered |

**Two passes, one clean negative, one split.**

**Say this out loud:** a thesis where all four hypotheses confirmed would be *more* suspicious, not
less — it would suggest the questions were chosen safely rather than tested honestly. The negative
result on RQ2 is the one that most changes what the next researcher should do.

---

## SLIDE 19 — What is novel, and what is NOT

**NOT novel — we invented none of these, and we say so:**

| Component | Prior work |
|---|---|
| Evidential Dirichlet heads | Sensoy et al., 2018 |
| Bottleneck adapters / LoRA / BitFit | Existing PEFT methods, borrowed |
| Episodic few-shot training, prototype networks | Standard since ~2017 |
| Freezing a pretrained backbone | Standard transfer learning |
| Energy score, temperature scaling, ECE | Liu et al. 2020; Guo et al. 2017 |

**Novel — ranked by defensibility, strongest first:**

1. **A well-powered negative that contradicts three independent prior papers, and explains why —
   now with theoretical backing.** BEL (trainable backbone), BayesAdapter (variational Bayes, strong
   semantic init, arXiv:2412.09718) and MetaQDA (large meta-learned classifier, arXiv:2101.02833)
   all found calibration *improving*; we find it degrading 20/20 — the difference is capacity in the
   probabilistic component, not whether the backbone is frozen (MetaQDA is frozen too). Bengs et al.
   (NeurIPS 2022) and Shen et al. (NeurIPS 2024) independently prove this kind of failure is
   theoretically possible; we are the first to measure how badly it bites in a real,
   deployment-realistic setting. We locate the boundary from three directions, not two. **The
   single strongest claim in the deck.**
2. **Calibration and OOD-ranking quality are empirically decoupled** — demonstrated on the same 120
   runs, and predicted in advance by the same theory (Slide 11). Measuring only one would have
   hidden it.
3. **We overturned our own earlier claim** (evidential ≈ energy) when we scaled from 1 configuration
   to 40 — energy wins ~70%. Self-correcting at scale, in the open, is itself evidence the grid
   results can be trusted over the earlier pilot.
4. **Backbone drives latency, adapter does not** (5.12× vs 3.9%) — and "evidential is free at
   inference" is measured at 1.29%, below the noise floor, rather than asserted.

**The "unstudied regime" claim (weakest of the five — checked, not just asserted, and the closest
match turned out closer than we first found):** frozen lightweight CNN, parameter-free prototype
head, trainable budget down to **2**, disjoint-class episodic few-shot, calibration **and** OOD
**and** params **and** latency together. **TSA** (arXiv:2107.00358, CVPR 2022) is the primary
analogue, not Conv-Adapter: frozen ResNet-18, serial-vs-parallel adapters, 600 sampled episodic
tasks, parameter-free head — same backbone, protocol, and head design as ours, same placement
answer (Slide 7). **FiT** (arXiv:2206.08671) is a second: frozen CNN + FiLM adapters as small as
11,648 parameters. Neither compares LoRA specifically or reports ECE/OOD. **Tip-Adapter/CLIP-Adapter**
were also read in full: they train and test on the *same* classes (their own paper contrasts this
with disjoint-category meta-learning splits), report one fixed-test-set accuracy rather than an
episodic average, report no ECE/OOD, and use adapters 17× to orders-of-magnitude larger than our
6,928–31,746 range. Across all four: the frozen-backbone-plus-adapter *idea* is shared; a budget
down to 2 parameters, a second backbone, the LoRA-specific comparison, and the calibration/OOD
measurement are not — say "checked, not discovered" if this comes up, not "nobody has done this."

---

## SLIDE 20 — The gap in the literature (why this intersection was empty)

| Literature | Representative work | Accuracy | Macro-F1 | Calibration | OOD | Param budget | Few-shot | Edge backbone |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical few-shot | ProtoNet, MAML, MetaOptNet | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot | P>M>F, CoOp, Tip-Adapter, CLIP-Adapter, DINOv3 | ✅ | ❌ | ❌ | ❌ | partial | partial | ❌ |
| PEFT for ViTs | VPT, AdaptFormer, SSF | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge | Conv-Adapter, LoRA-C, LoRA-Edge, CoLoRA | ✅ | ❌ | ❌ | ❌ | ✅ | partial | ✅ |
| Frozen-CNN episodic few-shot adapters | **TSA** (primary RQ1 precedent), FiT | ✅ | ❌ | ❌ | ❌ | ✅ | partial | partial |
| Bayesian PEFT | Laplace-LoRA, BLoB, BaRA, **BayesAdapter**, Bayesian-PEFT (Pandey et al., unverified — Slide 12) | ✅ | ❌ | ✅ | partial | ✅ | ❌ | ❌ |
| Evidential few-shot | BEL, **MetaQDA** (Bayesian, not evidential, but same protocol family) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | partial |
| TinyML / TinyDL | TinyDL survey | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **B-PEFT (ours)** | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Two of those ❌ are verified absences, not our inference:**
- The **ViT-PEFT survey** (arXiv:2402.02242), covering the entire field, **never discusses**
  calibration, uncertainty quantification, or OOD detection.
- The **TinyML→TinyDL survey** (arXiv:2506.18927, 2025) covers quantisation extensively and makes
  **no mention** of uncertainty, calibration, or OOD.

**The three "partial" few-shot cells were checked by reading the papers in full, not assumed from
category — because the closest ones are close enough that assuming would be sloppy:**
- **TSA** (arXiv:2107.00358, **RQ1's primary closest precedent**, Slide 7) runs the serial-vs-parallel
  comparison on a frozen ResNet-18 with a parameter-free head over 600 sampled episodic tasks — genuinely
  episodic, but its adapters (175K–1.22M params) are far above our range and it never compares LoRA.
  **Conv-Adapter** (secondary precedent) evaluates 1/2/4/8-shot *fine-tuning* on FGVC with a trainable
  per-task head — not disjoint-class episodic meta-testing.
- **Tip-Adapter / CLIP-Adapter** train and test on the *same* classes (Tip-Adapter's own text
  contrasts itself with meta-learning protocols that split into disjoint category subsets), report
  one fixed-test-set accuracy rather than an episodic average, report no ECE/OOD, and use adapters
  17× to orders-of-magnitude larger than our 6,928–31,746 range.
- **MetaQDA** (arXiv:2101.02833, Slide 10) is frozen-backbone, true 600-episode disjoint-class
  few-shot — the same protocol family as ours — but it is a classifier-design paper (a meta-learned
  Bayesian QDA prior), not an adapter-placement paper, and never compares PEFT adapter types.

Two independent communities, the same blind spot.

⚠️ **Honesty note:** the PEFT-for-edge row was checked against abstracts and summaries, **not** by
exhaustively reading every table. Flagged in `RESULTS_MASTER.md` §4.6 as needing verification before
it goes in the written thesis.

**Is the question dated?** No — ≥6 Bayesian-PEFT papers in the last 12 months, plus a dedicated
*benchmark* paper in 2026 (Bayesian Adaptation Gym), plus a NeurIPS 2024 paper literally named
"Bayesian-PEFT" (Slide 12) on foundation-model vision. Fields do not build benchmarks or name papers
after dead questions. **Almost all are on language models or foundation-scale vision backbones. None
is few-shot vision on a small edge-deployable CNN.**

---

## SLIDE 21 — Positioning: what parameter efficiency actually costs

**Same protocol (5-way few-shot episodic), so accuracy IS comparable. P>M>F meta-trains the whole
backbone; we freeze it.**

| Method | Backbone | Trainable | CIFAR-FS 5-shot | MiniIN 5-shot | CIFAR-FS 1-shot | MiniIN 1-shot |
|---|---|---:|---:|---:|---:|---:|
| Sup-21k > ProtoNet | ViT-B/16 | ~85.8 M | **96.7** | **99.2** | 92.3 | 97.2 |
| DINO > ProtoNet | ViT-S/16 | ~21 M | 92.5 | 98.0 | 81.1 | 93.1 |
| DINO > ProtoNet | ResNet-50 | ~25 M | — | 92.0 | — | 79.2 |
| BEL (evidential few-shot) | ResNet-12 | not reported | 86.92 | 79.60 | 73.96 | 63.10 |
| **Ours, parallel bottleneck** | **ResNet-18, frozen** | **31,744** | **91.44** | **95.56** | 78.57 | 85.03 |
| **Ours, parallel bottleneck** | **MobileNetV3-S, frozen** | **6,928** | **90.74** | 90.10 | 78.80 | 74.92 |

**The trade, stated plainly — three separate conclusions:**

| Ours vs. | Param saving | CIFAR-FS 5-shot | MiniIN 1-shot |
|---|---:|---:|---:|
| ResNet-18 adapter vs DINO>PN **ViT-S** | **662× fewer** | **−1.06 pp** | −8.07 pp |
| ResNet-18 adapter vs DINO>PN **ResNet-50** | 788× fewer | — | **+5.83 pp** |
| MobileNetV3-S adapter vs DINO>PN ViT-S | **3,031× fewer** | −1.76 pp | **−18.18 pp** |

1. **At 5-shot on CIFAR-FS the saving is nearly free** — 1.06 pp behind ViT-S at 662× fewer params.
2. **Backbone-family-matched, we win outright** — +5.83 pp over DINO>ProtoNet ResNet-50. The
   remaining gap is a **ViT gap, not a parameter-efficiency gap**.
3. ⚠️ **The trade is genuinely bad at 1-shot on MiniImageNet with the small backbone: −18.2 pp.**
   We volunteer this. The 6,928-param cell is competitive on CIFAR-FS and **is not** on MiniImageNet.

⚠️ **Comparability caveat that applies to every accuracy number we report:** our backbones are
ImageNet-pretrained, and **MiniImageNet's classes are ImageNet classes** — so every "novel" test
class was seen in pretraining. Our MiniImageNet numbers are not few-shot results in the sense the
benchmark was designed to measure. P>M>F's authors say the same about their own supervised-ImageNet
row. Fix = a from-scratch control run (Slide 22).

---

## SLIDE 22 — Limitations and outstanding work (we raise these, not you)

**Ordered by how badly a reviewer would want them fixed:**

| # | Limitation | Impact | Status |
|---|---|---|---|
| **1** | **Baseline coverage is incomplete — YOUR instruction of 2026-08-06 ("every combination should be tested") is NOT yet done.** `full_ft` and `linear_probe` have real 3-seed coverage on **ResNet-18 × CIFAR-FS only**. `full_ft` has never run on MiniImageNet on any backbone; **neither baseline has ever run on MobileNetV3-Small**. | The "MobileNetV3-S matches full fine-tuning" claim is a **cross-backbone** comparison | **Scoped as Step 12.F: 24 configs / 72 runs / ~18–19 GPU-h.** Deliberately deferred until Step 11 closed — that condition is now met |
| **2** | **ImageNet-pretraining confound** | MiniImageNet accuracies are not from-scratch few-shot numbers | Documented; fix = from-scratch control run |
| **3** | **No transformer arm in our own grid** | We argue from published ViT numbers, not one we ran | Step 11 gives a real *architecture-only* DeiT-Tiny/ViT-B measurement, but no trained accuracy |
| **4** | **3 seeds, effectively 1 for the two baselines** | Any claim on a <1 pp margin is thin | Stated on every affected claim |
| **5** | **One frozen recipe across all 120 cells** | Grid answers "how do these axes compare under one recipe", not "best per cell" | Deliberate design choice (controlled comparison) |
| **6** | **Evidential has no post-hoc calibration fix** | TS column is structurally favourable to softmax | Stated as an asymmetry |
| **7** | **TinyImageNet OOD exclusion check** traced by code inspection, never confirmed by a logged count from a real run | All TinyImageNet near-OOD numbers carry the caveat | `step_writeups/step9.txt` §5 |
| **8** | **Proposal scope trimmed** — ConvNeXt-Nano, CUB-200, ISIC, CIFAR-10-C, BitFit-in-grid all deferred | Less breadth than proposed | Step 12.A–12.E backlog, explicitly listed, not dropped silently |
| **9** | **One paper we could not fully verify** — "Bayesian-PEFT" (Pandey, Pyakurel, Yu, NeurIPS 2024, Slide 12) — repeated PDF/OpenReview extraction failures mean its exact protocol and numbers are unconfirmed | Small risk the RQ2 boundary framing needs a footnote if this paper turns out to share our exact regime | Flagged for personal reading before the meeting, not resolved |

---

## SLIDE 23 — Engineering honesty: two real bugs found during Step 11 closeout

**We report these because the process caught them, and because a supervisor who finds them
unprompted would weigh them more heavily.**

| Bug | Nature | Detection | Resolution |
|---|---|---|---|
| **1** | Crash in an optional bonus feature that discarded a correct measurement before it could save | Crashed loudly | Fixed at the root 08-07/08-09 |
| **2** | **Silent** — three independent latency-selection functions (`pareto_plots.py`, `make_master_tables.py`, `make_results_master.py`) all picked the dev laptop's noisy latency over the canonical Kaggle CPU number, by dict-insertion-order accident. **Errors up to 47 % on individual cells.** | **No crash, no test failure** — caught only by a manual cross-check | Fixed and verified; Table 8, `pareto_frontier.json` and all six Pareto PNGs regenerated from corrected data |

**Impact assessment:** every panel's *recommended* Pareto point was unaffected. *Frontier membership*
was affected — with the corrected Kaggle-only data, both MobileNetV3-Small LoRA variants are
genuinely on the CIFAR-FS strict frontier.

⚠️ **Open action item:** bug 2 has **no regression test yet**. It is the top item on the Step-12
backlog, because the bug class that produces no error is exactly what a test suite exists to catch.

---

## SLIDE 24 — Where we are, and what we need from you

**Done:** Steps 1–11 of 13. All four RQs answered. 120 runs, 0 errors, full results reproducible.

**The decision this meeting is for:**

| Option | What it costs | What it buys |
|---|---|---|
| **A. Start Phase 6 (thesis writing) now** | — | All four RQs already have defensible answers |
| **B. Run 12.F first** (your 2026-08-06 instruction) | ~18–19 GPU-h, 72 runs | Removes the cross-backbone caveat from the headline RQ1 claim |
| **C. Run the from-scratch control** | Not yet scoped | Removes the ImageNet-pretraining objection entirely — highest scientific value |
| **D. Add a trained ViT-Tiny arm** | Not yet scoped | Converts the "CNNs are right for edge" argument into our own measurement |
| **E. Write the regression test for the silent-data bug** | Small | Protects every future number |

**Our recommendation, for your challenge:** E (cheap, protects everything) → B (your instruction,
and it hardens the headline claim) → run Phase 6 writing **in parallel**, since no RQ is blocked.
C and D are the highest-value additions if the timeline allows, in that order.

**Questions we expect and have answers for:** Is the question dated? Why CNNs and not transformers?
Is a negative result publishable? Why is energy beating your Bayesian score? Why did you not run
CUB-200/ISIC? Are three seeds enough? Has anyone already answered RQ2? (Slide 12 — checked
exhaustively, one paper flagged as unverified, everything else ruled out or folded into the
boundary argument.)
