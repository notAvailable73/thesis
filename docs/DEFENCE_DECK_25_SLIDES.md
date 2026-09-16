# B-PEFT — Thesis Defence Deck (23 slides + references)

**Official thesis title (registered — do not alter):** *Bayesian Parameter-Efficient Fine-Tuning
(B-PEFT) for Reliable Few-Shot Vision with Lightweight CNN Backbones*

**Presentation subtitle (framing, Slide 1 only):** *A controlled experimental attribution study — which
design choices govern which aspects of reliability?*

> The official title stays. The subtitle exists because the title names the **method family** while the
> contribution is the **attribution study**. If you are asked why "Bayesian" is in the title when energy
> outperforms vacuity as an OOD ranker, the answer is Slide 18 and Slide 22 — the evidential
> formulation is what gives the model an explicit uncertainty representation to *measure*, and RQ2 is
> the experiment that separates the objective from the score. See the "Bayesian" briefing box below.

**How to use this file.** One `## Slide N` block per slide. **ON THE SLIDE** is paste-ready content —
put that on the slide and nothing more. **SAY** is your spoken line; it is not on the slide.
**VISUAL** names a figure that already exists in this repo. Total speaking time ≈ 25 minutes,
leaving room for questions.

**Timing:** stated per-slide times sum to **~30 min** as scripted (Slide 19 is delivered as
**19A + 19B**, one numbered slide's worth of story across two physical slides). Treat that as a floor:
several slides carry disclosure content added after a fact-check pass, and dense boxes read slower than
their word count suggests. **Budget 32–35 min for a live run-through and adjust from there** — do not
walk in assuming 25–28. References (2 slides) are **not** counted.

**This is a single self-contained deck — no backup slides.** Every number needed to defend a claim is
either on the slide that makes it, in that slide's SAY line, or in the Q&A crib sheet below (presenter
notes, not projected slides). Nothing lives only in a slide the audience never sees.

**Fact-checked 2026-09-14 against `docs/guide/`** (the post-experiment project guide, files 01–10) plus
`RESULTS_MASTER.md`, `DEFENCE_BRIEF.md`, `RQ_SUPERVISOR_REPORT.md` and `refs.bib`. Every figure on every
slide traces to one of those. Where the guide's corrections list (A1–A8, C1–C15) contradicted an earlier
version of this deck, **the guide wins** and the correction is stated on the slide rather than dropped.

**Updated 2026-09-15 — coverage completed and interactions computed.** The 21 Phase A checkpoints that
were missing have been trained and scored: coverage is now **120/120**, and all 21 reproduce the
committed Step 10 grid **exactly** (max absolute difference 0.0, best-epoch match 21/21). RQ1's two-way
interactions were computed at the same time, with 2000-resample bootstrap intervals. Three things change
through this deck as a result: RQ2's coverage caveat is **closed**, RQ1's "main effects only" limitation
is now a **computed result** that corroborates RQ3, and a **new self-correction** surfaced (Slide 22,
item 5). Source: `results/rq_completion/REPORT.md`; code `scripts/rq_completion.py`.

**If your slot is 20–25 minutes, trim in this order** — never cut Slides 2, 13, 19A/19B or 22:

| Cut | Saves | How |
|---|---:|---|
| Slide 6 + 7 → merge into one prior-work slide | ~1.0 min | Keep TSA, BEL and energy; drop Conv-Adapter and BayesAdapter rows |
| Slide 15 → drop the metrics table, keep protocol rigour | ~0.5 min | The metrics are self-evident from the results slides |
| Slide 21 → keep only the three-row table | ~0.25 min | Drop the Pareto figure references from the trimmed version |
| Slide 5 → fold objectives into Slide 4 | ~0.5 min | Last resort — most departments expect objectives |

That saves ~2.25 min off the scripted total — trim for margin, not to hit a specific number, since live
delivery already runs longer than the script.

---

## The one sentence the audience must leave with

> **Different aspects of reliability in parameter-efficient few-shot vision are governed by different
> design choices.**

Everything in this deck is in service of that sentence, decomposed into four findings:

| Outcome | Governed primarily by |
|---|---|
| **Accuracy** | shot count, then adapter **architecture** |
| **Calibration** | **head** interpretation, then **backbone** |
| **OOD detection** | the **scoring rule** |
| **Calibration repair** | post-hoc recalibration — ECE improves in **60/60** cells (still short of softmax in 60/60); OOD ranking is preserved in **80%** of comparisons |

---

**Four rules this deck is built on** (they are what separates a deck that defends *this* thesis from
one that defends a weaker one):

1. **The centrepiece is Slide 13 (the experimental design), not a results table.** This thesis is a
   controlled attribution study, not an architecture proposal. Its accuracy was never trying to be
   state of the art. If your headline is an accuracy table, the first question is "so you're behind a
   ViT?" and you have handed them the wrong frame.
2. **Slide titles are conclusions, not labels.** "RQ3 results" is a label. "Matched budgets point to
   backbone dependence, not parameter count" is a conclusion. Committees read titles even after they
   stop listening.
3. **Volunteer every weakness before it is found.** Slides 2, 15, 22 exist for this. A negative result
   you present yourself is credibility; the same result discovered in Q&A is a wound.
4. **Claim exactly what the design supports.** The word "proves" appears nowhere in this deck; neither
   does "always", "never", "universally" or "SOTA". Where the design supports only association, the
   deck says association. See the language-discipline table at the end — it is checked against the record.

---

## Briefing box — "If energy beats vacuity, why is this a Bayesian thesis?"

Expect this question. Answer in four moves, in this order, in under forty seconds:

1. **The evidential/Dirichlet formulation is what supplies an explicit uncertainty representation.**
   Vacuity = K/S is a quantity softmax does not have. Having it is what makes the reliability axes
   measurable on the same runs at all.
2. **The thesis does not claim evidential training universally produces the best OOD detector.** It
   claims — and RQ2 shows — that in this regime the *scoring rule* explains far more of the variation
   in OOD performance than the *training objective* does.
3. **RQ2 exists precisely to separate those two things**, which the published literature does not do
   because it scores evidential models with vacuity and softmax models with MSP.
4. **The evidential head's calibration deficit is identified and then repaired post-hoc** (RQ4),
   without retraining. Characterisation and controlled attribution are the contribution — not a
   blanket superiority claim for the Bayesian arm.

**Every number below is traceable** to `docs/RQ_SUPERVISOR_REPORT.md`, `docs/RQ_RESULTS_SUMMARY.md`,
`docs/RESULTS_MASTER.md`, `docs/DEFENCE_BRIEF.md`, `progress.txt`, or `step_writeups/*`. Sources are
named per slide.

---

## Slide 1 — Title

**Time:** 0.5 min

**ON THE SLIDE**

- **Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**
- *A controlled experimental attribution study of reliability under lightweight CNN constraints*
- Your name · Student ID
- Supervisor / Co-supervisor names and titles
- Department of Computer Science and Engineering
- Islamic University of Technology
- Date of defence

**SAY:** your name and the title. Nothing else. Move on in fifteen seconds.

---

## Slide 2 — Four questions about reliability, four answers

**Time:** 1.5 min · **This slide buys you the next twenty minutes.**

**ON THE SLIDE**

Overarching question, in a quote box at the top:

> What governs reliability — accuracy, confidence calibration, and out-of-distribution detection — in
> parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image
> classification, and can the deficits so identified be remediated after training?

Then the scorecard:

| | Question | What we found |
|---|---|---|
| **RQ1** | Which design axis drives which outcome? | Shot count dominates accuracy (**76%** of variance); head type dominates calibration (**83%**); the two axes are close to independent |
| **RQ2** | Is OOD detection attributable to the training objective or the scoring rule? | The scoring rule — it explains **42.7%** of far-OOD variance against the objective's **0.17%** |
| **RQ3** | Does accuracy follow adapter architecture and calibration follow parameter budget? | Accuracy — yes. Calibration was **not** explained by parameter budget; a matched-budget experiment supports **backbone** dependence |
| **RQ4** | Can the evidential head's calibration be repaired after training without breaking its OOD ranking? | Partly: ECE improved in **60/60** cells and OOD ranking held in **80%** — but it stays worse-calibrated than softmax in **60/60** |

Footer: **189 training runs · 600 frozen test episodes each · 3 seeds**

**SAY:** "Four research questions, four answers — including two where our initial expectation was not
supported, and I will show you both. One of them was overturned by an experiment we pre-registered
against ourselves."

**NEVER SAY:** "all four hypotheses were confirmed." They were not, and the deck is stronger for it.
Say instead: "the four research questions were answered, including cases where the initial expectation
was not supported."

**Source:** `RQ_SUPERVISOR_REPORT.md` §1; `results/rq_completion/REPORT.md` (2026-09-15 coverage
completion). Run count = 120 main grid + 48 matched-budget + 21 rank sweep. A further 21 grid cells were
**re-trained** in September to recover lost checkpoints; they reproduce the originals exactly and are not
counted again.

---

## Slide 3 — A model that is wrong and confident is worse than one that says "I don't know" — which is why this is a CNN thesis

**Time:** 1.75 min · **Section: A/B. Problem and deployment constraint** · **Merged from two slides,
and Slide 10 (old Research Questions) was cut outright, to buy back time — Slides 17, 18, 20 and 22 all
grew a fact-check pass this round.** Also pre-empts the known pre-defence attack: "CNNs are
outdated in 2026." Deliver that half as a design justification, before anyone asks.

**ON THE SLIDE**

- **Few-shot adaptation at the edge is a real deployment shape.** A device is handed 5 examples of a
  new class and must work immediately — on hardware ranging from microcontrollers with kilobytes of
  SRAM up through phone-class silicon, always tighter on memory and compute than a training-time GPU.
- **Full fine-tuning is not available there** (25 support images against 11.7M parameters overfits,
  and the compute does not exist on the device anyway), so parameter-efficient adaptation solves the
  capacity problem — **and inherits a reliability problem.** Adapters preserve accuracy. What happens
  to the *confidence numbers* is largely unreported.

Boxed consequence:

> A system that reports 95% confidence while being 60% correct is dangerous in exactly the
> applications that motivate edge deployment: inspection, triage, screening.
> **Accuracy alone does not characterise whether such a system is fit to deploy.**

**Why a lightweight CNN family, not a ViT:**

> **The target regime is edge/mobile-class compute — kilobytes to low megabytes of memory, no server
> GPU — not any single deployment tier.** That regime is what rules the large-ViT options out; it is
> not a claim that CNNs beat transformers in general, and it is not a claim about where *this thesis's*
> models were measured.

| Deployed at inference | Parameters | × ours |
|---|---:|---:|
| DINOv3 ViT-7B | 7,000,000,000 | 2,800× |
| CLIP-ViT-B/16 | 86,500,000 | 34.6× |
| DINO-ViT-S/16 | 21,000,000 | 8.4× |
| ResNet-18 (ours) | 11,700,000 | 4.7× |
| **MobileNetV3-Small (ours)** | **2,500,000** | — |

> **The disqualifier is compute, not a specific memory threshold.** At the MCU end of the edge
> spectrum, a 2025 survey measures a transformer attention block at **~180 ms** versus **~8-12 ms** for
> CNN inference on the same device — roughly **15–20×** *(TinyDL survey, arXiv:2506.18927)*. **It is
> their measurement, not ours: this thesis measured on a Kaggle T4 GPU and one CPU thread (Slide 21),
> not on MCU hardware.** Our own backbone (about 2.5 MB at int8) also sits above the MCU tier's own
> memory numbers — we target the edge/mobile-class family the MCU argument motivates, not the MCU tier
> itself, and we say so rather than let the two get conflated.

> In low-data regimes CNNs have been reported to **match** ViTs on other vision tasks
> *(arXiv:2510.04794, a few-shot geometric-estimation study, cited for direction rather than as
> classification-specific evidence)*, so the usual accuracy argument for a transformer is weaker in
> exactly this regime. **This is not a claim that CNNs are superior to transformers in general.**

**VISUAL:** two reliability diagrams side by side —
`results/grid_plots/cifar_fs_5shot_softmax_reliability.png` and `..._evidential_reliability.png`.
Do **not** explain them yet; you explain them on Slide 17.

**SAY:** "This thesis is about the sentence in that box: accuracy alone does not characterise whether a
system is fit to deploy. Not about making few-shot models more accurate — about what happens to their
*honesty* when you shrink them, and which design choice is responsible for which part of that. And the
backbone choice follows from that same deployment regime, not from a claim about CNN superiority: a
transformer attention block costs roughly fifteen to twenty times more compute than CNN inference on
comparable edge hardware — that order of magnitude is what rules large ViTs out, not a specific memory
number. I want to be precise, though: we did not measure on MCU hardware ourselves — our own numbers,
on Slide 21, are GPU and single-CPU-thread proxies for edge silicon more broadly. And in the low-data
regime, the accuracy argument for the transformer is at its weakest anyway."

**Source:** `DEFENCE_BRIEF.md` §1, §2, §3.4.

---

## Slide 4 — Problem statement, challenges and objectives

**Time:** 1.25 min

**ON THE SLIDE**

**Problem statement**, formally:

> Given a frozen backbone **f_θ** pretrained on a source distribution, and an episode **E = (S, Q)**
> with a 5-class support set **S** of *k* labelled images per class drawn from classes **disjoint**
> from anything f_θ was adapted on, learn an adapter **g_φ** with **|φ| ≪ |θ|** such that the
> resulting classifier is **accurate** on Q, **calibrated** (its confidence matches its correctness),
> and **able to reject** inputs drawn from outside the episode's label space.

**Research challenges** (the four this thesis attacks):

1. **Parameter inefficiency** — updating millions of parameters on 25 images overfits, and does not
   fit on the device. ← attacked
2. **The reliability gap** — PEFT methods inherit the base model's overconfidence, and it is rarely
   measured. ← attacked
3. **Architectural bias** — PEFT research is overwhelmingly about transformers; adapter behaviour
   *inside convolutional blocks* is poorly characterised. ← attacked
4. **Unmeasured edge cost** — uncertainty methods are benchmarked on server hardware; the edge cost is
   asserted more often than measured. ← attacked

**Objectives** — five, each verifiable:

1. Build a **frozen-backbone + adapter + parameter-free-head** framework that adapts to a new 5-way
   task from 1–5 examples per class under **~32,000 trainable parameters**.
2. Measure accuracy, calibration and OOD detection **on the same runs**, so their relationship can be
   *observed* rather than assumed.
3. Determine **which design axis governs which reliability outcome**, under a controlled factorial
   rather than one comparison at a time.
4. Establish whether observed differences are associated with the adapter's **architecture**, its
   **parameter budget**, or the **backbone** — by controlled experiment, not by argument.
5. Test whether an identified reliability deficit can be **remediated post-hoc**, without retraining
   and without destroying what the model does well.

**SAY:** "Objective 2 does most of the work. Reporting accuracy, calibration and OOD detection on the
same runs is uncommon — and several of our findings are only visible because we do."

**Source:** `proposal.txt` §3.

---

## Slide 5 — Background: four ideas the rest of the talk assumes

**Time:** 1.0 min · **Section: Background** · **This slide exists because the committee may want to
test the fundamentals before they engage with the findings. Deliver it briskly — it is groundwork,
not a contribution.**

**ON THE SLIDE** — four boxes, one concept each:

**1 · Episodic few-shot learning.** A task is an *episode*: **5 classes (5-way)**, **k labelled
support images** per class (k = 1 or 5), and unlabelled **query** images to classify. Test-episode
classes are **disjoint** from any class seen during adaptation — the model must handle categories it
has never been trained on. Reported as the mean over many sampled episodes (here, 600 fixed ones).

**2 · Parameter-efficient fine-tuning (PEFT).** Freeze the pretrained backbone; train a small module
instead. *Bottleneck adapter*: a narrow down-project → non-linearity → up-project residual branch.
*LoRA*: a low-rank update **W + (α/r)·A·B** folded into an existing weight matrix. Both buy
adaptation for ≪1% of the parameters.

**3 · Evidential deep learning (Dirichlet).** Instead of a point probability, predict **evidence**
per class: **α = evidence + 1**, **S = Σα**, probability **p = α/S**. The spare mass
**vacuity = K/S** is an explicit "I don't know" signal that softmax has no equivalent of.

**4 · The two reliability measurements.**
- **Calibration / ECE** — bin predictions by confidence and compare each bin's confidence to its
  actual accuracy. A perfectly calibrated model that says 90% is right 90% of the time. ↓ better.
- **OOD detection / AUROC** — score every input by uncertainty; AUROC is the probability a random
  out-of-distribution input is scored more uncertain than a random in-distribution one. 0.5 = chance,
  1.0 = perfect. ↑ better.

**SAY:** "Four things the rest of the talk leans on. The one to hold onto is vacuity — it is the
quantity that makes the evidential head different from softmax, and it is what most of the OOD
results turn on."



---

## Slide 6 — Prior work I: adapters on frozen CNNs — our closest precedent, and we replicate it

**Time:** 1.0 min · **Section: C. Research gap (literature)**

**ON THE SLIDE**

- **TSA — Task-Specific Adapters** (Li, Liu & Bilen, CVPR 2022, arXiv:2107.00358): frozen ResNet-18,
  serial vs parallel adapter placement, 600 sampled episodic tasks, parameter-free nearest-centroid
  head. **Finding:** residual/parallel connections beat serial "in almost all cases." Its adapters are
  **~175k–1.22M parameters** — one to two orders of magnitude above our 6,928–31,746.
- **Conv-Adapter** (arXiv:2208.07463): locality-preserving CNN adapters beat 1×1/linear-style ones.
- **LoRA** (Hu et al., 2021) and **bottleneck adapters** (Houlsby et al., 2019): the two adapter
  families we compare.
- **PEFT for ViTs** — VPT, SSF, AdaptFormer: the dominant line of work, and it is transformer-only.
- **What none of them do:** neither TSA nor Conv-Adapter compares against LoRA; neither tests a second
  backbone; **neither reports calibration or OOD detection.**

Boxed, in your own voice — **state our placement evidence precisely; it is weaker than TSA's:**

> We tested serial vs parallel **once** (Step 6: one seed, ResNet-18, CIFAR-FS 5-shot). They came out
> **tied on accuracy — 0.9145 vs 0.9146** — with parallel ahead on OOD, which is why parallel became the
> default. So we **do not** claim to replicate TSA's placement finding: TSA has the stronger evidence and
> we defer to it. Placement is not an axis of this thesis's grid. What is ours is the LoRA comparison,
> the second backbone, and the uncertainty layer on top.

**SAY:** "This is the paper a committee member is most likely to know, and it is the closest precedent
we have. I want to be careful about one thing: we tested serial against parallel only once, and on
accuracy they tied — we chose parallel on its OOD result. TSA's placement evidence is stronger than
ours, so we cite them for it rather than claiming it. Placement is not what this thesis contributes."

---

## Slide 7 — Three papers report Bayesian calibration working. We locate where it stops working.

**Time:** 1.0 min

**ON THE SLIDE**

| Paper | Regime | Their result |
|---|---|---|
| **Sensoy et al. 2018** (arXiv:1806.01768) | Full training; EDL introduced | Dirichlet evidence gives usable uncertainty at negligible inference cost |
| **BEL** (arXiv:2207.13137) | Few-shot, **backbone meta-trained** | Calibration **improves** (3.59% vs 14.69% ECE) |
| **BayesAdapter** (IJCV, arXiv:2412.09718) | Frozen CLIP, **large** linear adapter, up to 32 shots | Calibration **improves** (~2.5 pp ECE) |

Positioning line:

> All three succeed with capacity we deliberately remove — a trainable backbone, a large adapter, or
> more shots. We test the corner where all three are absent — frozen backbone, ≤31,744 trainable
> parameters, true disjoint-class episodes — and calibration **degrades, in 20 of 20 matched pairs**
> (1.4×–9.1× worse than plain softmax; 5.3×–51× worse than temperature-scaled softmax).
> Locating that boundary is the contribution; it is not a failure to reproduce them.

**The honest competitor** — say this before anyone else does:

- **Energy score** (Liu et al. 2020, arXiv:2010.03759): a logit-space OOD score needing **no** Bayesian
  machinery, no training change and no extra parameters. It outperforms max-softmax-probability.
- **OpenOOD / OpenOOD-1.5:** post-hoc scores routinely outperform training-method changes.
- Most evidential papers compare vacuity against **max-softmax-probability only**. That comparison is
  too easy. We run energy on every configuration — and it changes our answer.

**SAY:** "These papers are not contradictions; all three are true in their own regime. The difference
is capacity and data, and the interesting question is where the boundary sits. And if you only compare
Bayesian uncertainty against max-softmax probability, you will conclude that it wins. We compared
against the strongest cheap alternative — which is the reason RQ2 exists at all."

---

## Slide 8 — What existing work covers, and the combination we did not find

**Time:** 0.75 min · **Scope the literature claim carefully — this slide is where over-claiming is
easiest and costs the most.**

**ON THE SLIDE** — the four closest comparisons, not a survey matrix:

| Closest prior work | Backbone / regime | Reports accuracy | Reports **ECE** | Reports **OOD** | Tiny adapter, frozen CNN |
|---|---|:-:|:-:|:-:|:-:|
| **TSA** (CVPR 2022) — frozen-CNN episodic adapters | ResNet-18, episodic | ✅ | ❌ | ❌ | ✅ |
| **BEL** — evidential few-shot | ResNet-12, **backbone meta-trained** | ✅ | ✅ | ❌ | ❌ |
| **BayesAdapter** (IJCV) — Bayesian adapter, frozen backbone | **CLIP ViT**, large adapter | ✅ | ✅ | ❌ | ❌ |
| **Conv-Adapter / LoRA-Edge** — PEFT for CNNs on edge | CNN, non-episodic | ✅ | ❌ | ❌ | ✅ |
| **This thesis** | Frozen ResNet-18 / MobileNetV3-S, episodic | ✅ | ✅ | ✅ | ✅ |

Note the **OOD column**: none of the four closest comparisons reports OOD detection at all. That is the
thinnest part of the map, and it is where RQ2 sits.

Boxed, scoped exactly this way:

> **We did not find prior work combining these factors under this experimental regime** — a frozen
> edge/mobile-class CNN, a ≤31,744-parameter adapter, disjoint-class episodic few-shot, with accuracy,
> calibration **and** OOD detection measured on the same runs.

**And the three gaps that are confounds, not missing experiments** — this is the actual argument:

1. PEFT is usually measured on accuracy alone; its effect on calibration and OOD is rarely reported.
2. Where uncertainty results *are* reported, the **training objective and the scoring rule are varied
   together**, so the published comparison cannot separate "the Bayesian training helped" from "the
   score helped." → **RQ2**
3. Adapter comparisons **confound architecture with parameter budget**, because the better adapter is
   usually also the bigger one. → **RQ3**

**SAY:** "Individually, every one of these rows has checkmarks — the point is not that any single
element is new. Two of the three gaps are *confounds*: the published designs cannot separate the
factors even in principle. That makes this a design problem rather than a compute problem, which is
why the core of this thesis is its experimental design. And the closest misses either sit on a CLIP or
vision-transformer backbone, or meta-train the backbone, or drop the OOD axis entirely. We checked that
five separate times, the last pass specifically trying to break it — which is not the same as proving
there is nothing there."

**Careful — required wording.** Say **"we did not find prior work combining these factors under this
experimental regime."** Do **not** say "no precedent," "nobody has done this," or "the first."
The claim is about what these papers foreground and about a non-exhaustive search (limitation 7,
Slide 22). `RESULTS_MASTER.md` §4.6 states it that way; say it that way.

**Distinguishing detail, if pressed** — the novelty check is **five independent passes**, logged in
`RQ_SUPERVISOR_REPORT.md` Appendix C (2026-08-21 initial review; 2026-08-23 independent re-verification,
5 agents, 60+ searches; 2026-08-26 adversarial full-PDF stress test; an independent deep-research pass;
2026-08-27 post-experiment re-check, 27 searches against RQ3's revised claim). **No pass surfaced a
prior-art contradiction.** Say "five passes, around a hundred searches" — that is what the record
supports.

- **TSA** (arXiv:2107.00358) — closest on the adapter axis, but ~175k–1.22M-parameter adapters, and no
  ECE, no OOD.
- **BayesAdapter** (arXiv:2412.09718) — the closest calibration precedent on a frozen backbone, but
  **CLIP** rather than an edge CNN, a large adapter, up to 32 shots, and no OOD-AUROC.
- **BEL** (arXiv:2207.13137) — closest on evidential few-shot, but the **backbone is meta-trained**, so
  the frozen-backbone constraint this thesis studies is absent, and it reports no OOD-AUROC.
- Two prior open items were **closed by direct reading**: *Robust Calibration of Large Vision-Language
  Adapters* (ECCV 2024) and the NeurIPS 2024 **"Bayesian-PEFT"** paper — both confirmed distinct. The
  name similarity of the latter still needs an explicit differentiation paragraph in the thesis text.

**A wider eight-literature coverage matrix exists in the underlying reports** (`RESULTS_MASTER.md`
§4.6) if a deeper written map is wanted after the defence; the four-row comparison above is what to
defend live.

---

## Slide 9 — The experimental program stayed intact; the questions were refined to match what it can establish

**Time:** 0.75 min · **Do not skip this. Deliver it before anyone asks.** A committee member holding
the proposal *will* notice — and the chronology is what makes the answer safe.

**ON THE SLIDE** — the chronology, left to right, in three labelled columns:

**① Original proposal (written first)** → **② Experiments completed as proposed** → **③ Refined
attribution questions (asked of the same completed data)**

| ① Proposal asked (a comparison) | ② Result obtained — all experiments run and retained | ③ Now presented as (an attribution question) |
|---|---|---|
| Orig-RQ1: adapter placement / choice | Serial vs parallel **tied** on accuracy in the one test of it (Step 6). In the grid, parallel bottleneck beats **LoRA 16/16**, +2.1 to +8.3 pp | Part of RQ3's architecture axis |
| Orig-RQ2: does evidential calibrate better than softmax? | **No — 0/20 matched pairs** | RQ1 (head dominates calibration) + RQ4 (can it be repaired?) |
| Orig-RQ3: does a Bayesian prior improve near-OOD detection? | Yes vs softmax scores (37–38/40); **no vs energy** (10–14/40) | RQ2 (objective vs score) |
| Orig-RQ4: latency vs uncertainty Pareto frontier | Backbone drives latency **5.12×**; adapter 3.9–5.8%; head 1.29% | Supporting evidence under RQ1 |

Boxed — the four facts that make the refinement defensible:

> 1. **No experiment was discarded.** Every proposed experiment was run to completion and is reported.
> 2. **No data were removed.** All 120 grid runs, all 600 frozen test episodes per run, are in the record.
> 3. **Every original question is answered on this slide** — including the two whose answer was "no."
> 4. **The refinement was in the question, not the evidence.** A literature review conducted *after*
>    the grid completed found close precedent for each original *comparison*. The same completed
>    experiments support *attribution* questions, which the balanced factorial was already designed for.

Footer line, in your own voice:

> **The experimental program remained intact; the questions were refined to make the claims
> scientifically testable.**

**SAY:** "The order matters, so let me be explicit about it. The proposal asked four 'does A beat B'
questions. We ran every one of those experiments and answered all four — two of them negatively, and
both negatives are on this slide. The literature review that came afterwards showed each comparison
had close precedent. So we asked a harder set of questions *of the same data* — what is attributable
to what — because a balanced factorial can answer that and a pairwise comparison cannot. Nothing was
dropped to make that work."

**If pressed — "did you change the questions because the results were inconvenient?"** "No, and the
two negative results are the evidence: they are still here, reported, on this slide and on Slide 22.
If the goal had been convenience, the 0-of-20 calibration result is the first thing that would have
disappeared."

**Correction to carry into the thesis text:** `RQ_SUPERVISOR_REPORT.md` Appendix A labels Orig-RQ1
"serial versus parallel" and then reports "parallel wins 16/16". Those are two different comparisons —
the 16/16 is **parallel bottleneck vs LoRA**, and serial was never in the grid. The row above states it
correctly; fix the report before submission.

**Source:** `RQ_RESULTS_SUMMARY.md` Appendix; `RQ_SUPERVISOR_REPORT.md` Appendix A;
`docs/guide/05_problems_and_open_work.md` A3.

---

## Slide 10 — The system

**Time:** 1.0 min · **Section: E. Method** · **Draw this figure yourself — the repo has no version of it.**

**ON THE SLIDE** — a clean left-to-right pipeline diagram:

```
support images ─┐
                ├─→ [ FROZEN backbone ] ─→ [ ADAPTER ] ─→ embeddings ─→ class prototypes ┐
query images  ──┘      ResNet-18 /          the ONLY                                      ├─→ logits
                       MobileNetV3-S        trained part      query embedding ────────────┘
                       (never updated)      (≤ 31,746 params)                              │
                                                                                           ▼
                                                          ┌────────────────────────────────┴───┐
                                                          │ softmax     → probabilities        │
                                                          │ evidential  → α, S → vacuity       │
                                                          └────────────────────────────────────┘
```

Annotate three things **directly on the diagram**:

- **FROZEN** on the backbone — "11.7 M / 2.5 M parameters, never updated."
- **TRAINABLE** on the adapter — "6,928 – 31,746 parameters — the entire trainable budget."
- **PARAMETER-FREE** on the head — "a class = the mean embedding of its support images."

**SAY:** "The head has no weights. A class is literally the average of its five support embeddings,
and you classify by similarity to those averages. That matters for a reason easy to miss: a trained
linear head over the 64 base classes cannot transfer to 20 *disjoint* test classes — so a
parameter-free head is not a shortcut here, it's a requirement of the protocol."

**If asked "isn't that just ProtoNet?":** yes, the head is a prototype head, and the novelty is
nowhere near the head. Say that plainly and point at the adapter and the evaluation.

---

## Slide 11 — Two adapters — and a parameter-count reversal we did not design

**Time:** 1.0 min · **Worth drawing the two architectures side by side.**

**ON THE SLIDE** — left half, the two architectures:

- **Parallel bottleneck:** 1×1 conv down (d→r) → ReLU → 1×1 conv up (r→d), computed on the block
  *input* and summed at the block *output*. Placed at the final block of each of 4 stages.
- **LoRA:** a low-rank update injected **into the backbone's own 1×1 convolution** —
  **W_eff = W_frozen + (α/r)·A·B**. Changes what the backbone *computes*, not what happens after it.

Right half — the table that makes RQ3 askable:

| Backbone | Bottleneck | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck, by **2.58×** |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA**, by **1.55×** |

Boxed:

> **The ordering reverses between backbones** — a consequence of channel widths, not of design. This
> *partially* separates adapter **architecture** from adapter **budget**, which is normally impossible,
> because the better adapter is usually also the bigger one everywhere.

**SAY:** "We did not build this reversal. We *noticed* it. And it is the reason RQ3 can be asked at
all — though, as I will show in a moment, it only gets you part of the way, and we had to run a second
experiment to finish the job."

---

## Slide 12 — The head is one thing; how you read it is another

**Time:** 1.0 min

**ON THE SLIDE**

The prototype head always emits raw similarity logits **z** — **cosine similarity × 10** between the
query embedding and each class prototype (not the L2 distance `configs/base.yaml` defaults to; every grid
config sets `metric: cosine`, `cosine_scale: 10`). Two readings:

| | Softmax | Evidential (Dirichlet) |
|---|---|---|
| Probability | p = softmax(z) | α = softplus(z·s + b) + 1, S = Σα, p = α/S |
| Confidence | max softmax probability | — |
| Uncertainty | none native | **vacuity = K/S**, K = 5 |
| Extra trainable params | 0 | **2** (s, b) |

Boxed:

> Because both interpretations sit on the **same** logits, a matched evidential/softmax pair differs by
> exactly **two parameters** — 31,746 vs 31,744. That is what makes the head axis a clean experimental
> factor rather than a confound.
>
> Those two scalars are **learnable**: initialised at `(scale, bias) = (2, −6)` and trained by Adam
> alongside the adapter. They are not a fixed default — which is what makes RQ4's refit result
> interesting rather than trivial.

**SAY:** "This is the design decision that makes the whole factorial work. The evidential and softmax
arms are not two different models — they are the same model read two ways, differing by two scalars.
So when RQ1 attributes 83% of calibration variance to the head, it is attributing it to those two
scalars and the loss that trained them, not to a different architecture."

**Keep in reserve for "what was your hardest bug?":** an early **evidential collapse** — raw L2 logits
are large-negative for ResNet-18 features, so softplus(z) ≈ 0 everywhere, giving a uniform Dirichlet
and dead gradients. Fixed with cosine similarity plus the learnable evidence affine, and the mapping
now lives in exactly **one** function shared by training and evaluation
(`PrototypeHead.to_evidence()`), so train and eval cannot silently drift apart.

---

## Slide 13 — A balanced factorial, not a sequence of comparisons

**Time:** 1.5 min · **Section: F. Experimental design** · **THE CENTREPIECE OF THE DECK. Give it the
most visual care; draw it yourself.**

**ON THE SLIDE** — the grid, as five axes:

```
      dataset     ×  shots ×  backbone        ×  adapter     ×  head
   CIFAR-FS          1        ResNet-18          bottleneck     softmax
   MiniImageNet      5        MobileNetV3-S      LoRA           evidential
       2         ×   2    ×      2             ×    2         ×   2     = 32 cells
                                        + 8 baseline cells (Full-FT, Linear-Probe)
                                        = 40 configurations × 3 seeds  = 120 runs
```

The three experiments that make up the evidence base — **keep these visually distinct**:

| Experiment | Runs | What it is for | What it can establish |
|---|---:|---|---|
| **Main factorial grid** (2⁵ + baselines) | **120** | RQ1, RQ2, and RQ3's first evidence | Attribution of variance to axes; detects that the adapter axis is confounded |
| **Matched-budget adjudication** (**pre-registered**) | **48** | Settles RQ3 | Separates architecture from budget *within* each backbone |
| **Controlled rank sweep** | **21** | Tests an interior-optimum hypothesis | Only `adapter.rank` varies — and the hypothesis is rejected in the tested range |

Boxed, the methodological claim:

> Instead of comparing two arbitrary models, we varied the major design axes systematically, so their
> effects on each outcome could be **attributed** rather than merely ranked.

Footer: **36.3 h + 6.2 h GPU wall time · 120/120 and 48/48 completed · zero run errors**

**SAY:** "Every prior comparison in this literature moves one thing at a time and reports the winner.
A balanced factorial lets you do something different: attribute the variance in an outcome to the
axes that produced it. That is what converts 'parallel beat LoRA' into 'accuracy is governed by
adapter architecture' — and it is the methodological core of this thesis. Note the second row: when
the grid turned out to confound two factors, the response was to design an experiment that separates
them, not to argue about which explanation was more plausible."

**If asked early whether one frozen recipe across 40 configurations is fair:** "Deliberate. It makes
the grid a controlled comparison rather than 40 independently-tuned numbers, and it is stated as a
limitation on Slide 22."

---

## Slide 14 — Two in-distribution datasets, five OOD pools

**Time:** 0.75 min

**ON THE SLIDE**

**In-distribution** (5-way episodic; test classes disjoint from training classes):

| Dataset | Split | Classes (train/val/test) | Native resolution |
|---|---|---|---|
| CIFAR-FS | Bertinetto | 64 / 16 / 20 | 32×32, derived from CIFAR-100 |
| MiniImageNet | Ravi & Larochelle | 64 / 16 / 20 | 84×84, ImageNet-derived |

All inputs are **resized to 224×224 and ImageNet-normalised**, because the backbones are
ImageNet-pretrained and frozen.

**Out-of-distribution:**

- **Far-OOD:** SVHN · Gaussian noise
- **Near-OOD:** CIFAR-100-heldout · MiniImageNet-heldout · TinyImageNet
  (with the 25 wnids it shares with MiniImageNet excluded for MiniImageNet runs)

> **Second stated limitation — put it on the slide.** The two "heldout" near-OOD pools are the dataset's
> **16 validation classes**. They are disjoint from the 20 test classes, but they are the same classes
> that drive early stopping, temperature fitting and RQ4's refit — so model selection has seen them as
> in-distribution. **TinyImageNet has no such overlap**, which is why it is the near-OOD pool quoted in
> the headline results.

Caveat — **on the slide, not just spoken**:

> **Stated limitation.** Our backbones are ImageNet-pretrained, while the standard few-shot protocol
> trains from scratch on base classes. MiniImageNet's classes **are** ImageNet classes, so "novel"
> test classes were seen during pretraining. **Absolute accuracies are therefore not comparable to
> from-scratch few-shot literature.** Every research question here concerns **relative** differences
> between design choices under one fixed recipe — which is what makes the results usable despite this.

**SAY the caveat out loud.** It costs 15 seconds and removes an entire line of attack.

---

## Slide 15 — Metrics, protocol and why these numbers can be trusted

**Time:** 1.0 min

**ON THE SLIDE** — left column, metrics:

| Metric | What it answers | Dir. |
|---|---|:-:|
| Accuracy, macro-F1 | Is it right? (F1 exposes per-class recall collapse accuracy hides) | ↑ |
| **ECE** (pooled, 15 bins) | When it says 90%, is it right 90% of the time? | ↓ |
| Brier | Accuracy and calibration in one proper scoring rule | ↓ |
| **OOD AUROC** | Can it rank an unfamiliar input as more uncertain than a familiar one? | ↑ |
| FPR@95%TPR | At a usable operating point, how often does it wave through an outlier? | ↓ |
| Latency, MACs, peak memory, trainable params | Can it actually be deployed? | ↓ |

Four uncertainty scores, **all computed on all runs**: **vacuity** (evidential) · **MSP** ·
**temperature-scaled MSP** · **energy**.

Right column — **protocol rigour**:

- **Frozen test episodes.** 600 episodes, seeds fixed in a version-controlled file, identical across
  every run in the thesis. **One disclosed exception:** one cell (`cifar_fs/5shot/mobilenetv3_small/
  lora/evidential`, seed 42) was originally scored on a 20-episode smoke subset instead of the full
  600 — caught and re-scored on the full protocol on 2026-09-15 (self-correction 5, Slide 22). No test
  set was otherwise ever re-sampled.
- **Selection on VAL only.** A separate 100-episode validation split (seeds 10000–10099). The 600 test
  seeds were never used for any selection decision. *One disclosed exception to "VAL-selected": the grid
  inherits `kl_weight_max` = 0.1, while the Step 4.5 VAL sweep ranked 0.05 first (VAL ECE 0.252 vs
  0.260). The surface is flat and no comparison is confounded, but do not call 0.1 VAL-selected.*
- **Byte-identical reproducibility, on two separate occasions.** The 48-run matched-budget experiment
  (August) re-trained 18 existing grid arms from scratch and reproduced the committed numbers at **max
  absolute difference 0.0**. An independent September session re-trained **21 more** grid cells, on a
  different machine and session, and also reproduced them at **0.0** — including the early-stopping epoch
  in 21/21 cases. **39 cells, two occasions, zero drift.**
- **Pre-registration.** RQ3's decision rule, thresholds and an explicit "inconclusive" outcome were
  written down **before any of the deciding runs existed** (`docs/RQ3_MATCHED_BUDGET_PLAN.md`).
- **Control guards.** Automated checks that only the intended axis differs between compared configs —
  **0 unaccounted config keys** across all 48 matched-budget runs.

Environment footer: **Kaggle / Colab T4 GPU · episodic meta-training, 100 episodes/epoch, ≤30 epochs,
early stop patience 5 · LR 5e-3, adapter rank 16 throughout (Full-FT exception: LR 1e-5, wd 1e-4) ·
5 support + 15 query images/class/episode · no data augmentation · seeds 42/43/44**

**SAY:** "Two independent re-runs reproducing to a maximum absolute difference of *zero* is the number
I'd point at if you only trust one thing on this slide — eighteen cells in August, another twenty-one
in September. Thirty-nine cells, two occasions, three months apart, different notebooks, different
Kaggle sessions, and every one came back byte-identical. That's not something you can assume; it's
something we checked twice."



---

## Slide 16 — A 31,744-parameter adapter matches full fine-tuning of the same backbone

**Time:** 1.25 min · **Section: G. Results**

**ON THE SLIDE** — CIFAR-FS, 5-way 5-shot:

| Configuration | Trainable params | Accuracy | Episode 95% CI |
|---|---:|---:|---:|
| Parallel bottleneck, ResNet-18 | 31,744 | **91.44%** | ±0.51 |
| Full fine-tuning, ResNet-18 | 11,176,512 | 90.47% | ±0.50 |
| Parallel bottleneck, MobileNetV3-Small | **6,928** | 90.74% | ±0.52 |
| Linear probe (no adaptation) | 0 | 87.41% | ±0.56 |

Boxed: **31,744 trainable parameters (0.28% of the backbone) match full fine-tuning of the same
ResNet-18.** Full-FT's accuracy is byte-identical across all three seed files — it has no adapter to
randomly initialise, so it is effectively **one run, not three** (zero seed spread). Against that fixed
value, all **three** bottleneck seeds independently clear it, even the worst one (91.25% vs 90.47%,
+0.78 pp). A formal paired test isn't possible here — per-episode accuracy arrays aren't retained in the
result files, only aggregate mean/std/CI — so this across-seed comparison is the honest margin to quote,
not a quadrature-combined CI (which would assume independence between two arms scored on the same 600
frozen episodes).

> **The second row is weaker — say so before it's checked.** MobileNetV3-Small's 6,928-parameter adapter
> is +0.27 pp over the *same* ResNet-18 full-fine-tuning baseline — a **cross-backbone** comparison
> (full fine-tuning of MobileNetV3-Small was never run), and +0.27 pp sits **inside** the ±0.72 pp
> combined CI. Claim *matches*, never *beats*, for this row. `RESULTS_MASTER.md` §4.7 claim 5 is
> explicit; a careful committee member checking the intervals will catch "beats."

**Caveats to volunteer, in small print:**

> **1-shot reverses this.** Full fine-tuning wins by 2.57 pp at 1-shot — the parameter saving is close
> to free at 25 support images, not at 5. **2. Against published few-shot methods**, the same pattern:
> −1.06 pp behind a fully meta-trained DINO ViT-S at CIFAR-FS 5-shot (662× fewer parameters), but up to
> −18.2 pp behind it at MiniImageNet 1-shot with the small backbone — volunteer that number, do not let
> it be found. Backbone-family-matched, it reverses: +3.56 pp *ahead* of DINO>ProtoNet on a same-size
> ResNet-50, so the remaining gap to the ViT is a ViT gap, not a parameter-efficiency gap.

**SAY:** "Two rows, two different strengths of claim. The ResNet-18 row — same backbone, adapter versus
full fine-tuning — I can make cleanly: full fine-tuning has zero seed spread, so it's one number, and
every one of our three seeds beats it, even the worst by three quarters of a point. That's a result. The
MobileNet row is a cross-backbone comparison sitting inside its own combined confidence interval, so the
honest word there is *matches*, not *beats* — and matching full fine-tuning at 0.28% of the parameters
is the interesting result anyway."

---

## Slide 17 — Shots drive accuracy; the head drives calibration

**Time:** 1.75 min

**ON THE SLIDE** — make these three findings visually dominant, one line each, largest type on the slide:

> ### Accuracy → **shot count** dominates (76% of variance)
> ### Calibration → **head type** dominates (83% of variance)
> ### OOD detection → **no single dominant axis**; head, shots and adapter all contribute

**How to read η², one sentence, on the slide:**

> **η² is the share of the total variation in an outcome that is attributable to one design axis.**
> 76% for shots on accuracy means: of everything that made accuracy differ across the grid, about
> three-quarters of it is explained by whether the episode had 1 or 5 shots.

Then the evidence, kept on the slide in smaller type — η² decomposition, main effects over the
balanced factorial, **96 per-seed observations for every outcome** — the one smoke-run-scored cell
(self-correction 5, Slide 22) was re-scored on the full 600-episode protocol before any number below was
computed, so near-OOD AUROC is n=96 like the rest, not the n=95 an earlier pass of this deck reported:

| Outcome | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.69% | **76.31%** | 3.82% | 8.98% | 0.16% | 9.04% |
| ECE | 1.98% | 2.16% | 4.56% | 0.65% | **82.97%** | 7.68% |
| OOD AUROC (far) | 1.23% | 22.59% | 11.71% | 1.79% | **38.93%** | 23.75% |
| OOD AUROC (near) | 1.54% | **43.30%** | 0.55% | 21.68% | 21.44% | 11.49% |

One-line consequence, boxed:

> **Accuracy and calibration are governed by near-disjoint axes** — the head explains 83% of calibration
> variance and 0.16% of accuracy variance. You can choose the head for calibration without paying for
> it in accuracy.

**Tooling note, disclosed alongside self-correction 5:** the missing near-OOD observation didn't trip an
automatic balance check before a human found it. `eta_squared()`'s completeness test only inspects cells
that have *at least one* seed present — a wholly-missing observation is invisible to it, unlike a
merely under-seeded cell, which it does catch. That's a gap in the tooling, not only in the data.
`scripts/rq_completion.py` now runs an explicit design-completeness check (`rq2_completeness()`) that
enumerates the full intended grid instead of trusting that implicit signal.

**Now extended to two-way interactions** (2026-09-15; 2000-resample seed bootstrap). The table above is
main effects only. Adding the ten pairwise terms absorbs most of what was unexplained:

| Outcome | Residual, main effects only | Residual, + 2-way | Largest interaction | **`backbone:adapter`** |
|---|---:|---:|---|---:|
| Accuracy | 9.04% | **1.28%** | `dataset:backbone` 5.37% | 0.77% [0.58, 0.98] |
| **ECE** | 7.68% | **2.03%** | **`backbone:adapter` 3.23%** | **3.23% [2.49, 4.05]**, p<1e-6 |
| OOD AUROC (far) | 23.75% | **9.53%** | `dataset:backbone` 9.11% | 0.20% [0.00, 0.75], n.s. |
| OOD AUROC (near) | 11.49% | **6.21%** | `dataset:backbone` 3.90% | 0.00% [0.00, 0.05], n.s. |

Boxed — **this is the finding, and it is an independent corroboration of RQ3:**

> **On calibration specifically, the backbone × adapter interaction is real and significant** — 3.23%
> of ECE variance (p < 1e-6), the single largest interaction term for that outcome, and **42% of what
> main effects had left unexplained**. That is RQ3's result — the adapter's calibration effect depends
> on which backbone it sits in — arriving independently, from a different method, on the full grid
> rather than 16 matched pairs.
>
> **It is specific to calibration.** On accuracy and both OOD pools the same term is ≈0 and not
> significant. The interaction that dominates *those* outcomes is **`dataset:backbone`** — which no
> research question in this thesis asks about, and which we flag rather than interpret. One plausible
> (untested) mechanism: MiniImageNet's classes are ImageNet-derived and CIFAR-FS's are not, so the two
> backbones' shared ImageNet pretraining likely transfers unevenly depending which dataset they're
> adapted to — a specific instance of limitation 1 (pretraining overlap, Slide 22), not a new claim.

**SAY this if you show the interaction table:** "The main-effects table leaves eight to twenty-four
percent unexplained depending on the outcome. Adding two-way terms absorbs most of it — and on
calibration, the single biggest term is backbone-by-adapter, three point two percent, p below ten to
the minus six. That is RQ3's finding falling out of a completely different analysis. I want to be
equally clear about the negative half: that same term is statistically indistinguishable from zero on
accuracy and on both OOD pools. The interaction doing the work there is dataset-by-backbone, which none
of my research questions are about, and I am not going to interpret it beyond reporting it — though my
best guess is that it's the pretraining-overlap limitation showing up as a number."

**VISUAL:** the two reliability diagrams from Slide 3 — now explained.

**SAY:** "Read the two bold cells. 83% of calibration variance and 0.2% of accuracy variance sit on the
same axis — that is close to a clean separation of concerns. Note also the OOD rows: no single axis
dominates there, which is exactly what motivates RQ2 on the next slide."

**Volunteer:** "The qualitative pattern — that calibration and accuracy have different drivers — goes
back to Guo et al. 2017. We do not claim to have discovered it; what is ours is the formal
decomposition, in this regime."

**A second finding, worth its own thirty seconds — ECE and OOD AUROC are close to orthogonal once the
head is fixed.** Across all 40 cells the rank correlation between ECE and OOD AUROC is *positive*
(ρ = +0.433 far, +0.477 near), which naively reads as "worse-calibrated configurations detect OOD
better." Stratifying by head collapses it: ρ = +0.15/+0.20 within evidential, +0.26/+0.24 within
softmax. **Once the head is fixed, calibration carries little information about OOD detection** — they
are close to orthogonal outcomes, not a trade-off. *We did not find prior work reporting this under
this experimental regime.* This is only visible because accuracy, calibration and OOD are measured on
the same runs — Objective 2 from Slide 4.

---

## Slide 18 — For OOD detection, the score you read matters far more than the loss you trained with

**Time:** 1.75 min

**ON THE SLIDE** — first the confound, in one line:

> Every published comparison we found scores evidential models with **vacuity** and softmax models with
> **MSP**. The objective and the score always move together. We compute **all four scores on all runs**:
> a clean 2 objectives × 4 scores factorial, **n = 960 comparisons per pool**, on **all 120 models**.

Then the headline, in large type, as two lines — **the two shares, not their ratio**:

> ### Far-OOD variance explained by the **scoring rule**: **42.7%**
> ### Far-OOD variance explained by the **training objective**: **0.17%**

The full result, smaller:

| Pool | Scoring rule (η²) | Training objective (η²) |
|---|---:|---:|
| Far-OOD | **42.7%** | 0.17% |
| Near-OOD | **14.7%** | 0.63% |

> Two to three orders of magnitude, on both pools. Design **fully crossed**: 20/20 complete, zero
> empty objective × score cells.

The concrete version — the same score gives near-identical AUROC whichever objective produced the logits:

| Score (far-OOD) | On evidential-trained logits | On softmax-trained logits |
|---|---:|---:|
| **Energy** | 0.9185 | 0.9342 |
| MSP | 0.7993 | 0.8014 |

What this does and does not say, boxed — **put this on the slide**:

> **Says:** in this experiment, the scoring rule explains substantially more of the variation in OOD
> performance than the training objective does.
> **Does not say:** that evidential training is useless. The evidential head is what supplies an
> explicit uncertainty representation at all, it remains the better *native probabilistic* score
> (37–38 of 40 cells vs MSP, +0.111 far-OOD / +0.053 near-OOD mean advantage), and its inference cost is
> below our noise floor (Slide 21). On near-OOD at full coverage the evidential arm is in fact ahead of
> the softmax arm on every score (e.g. vacuity 0.8347 vs 0.8233) — a small effect, but in its favour.

Small print on the slide — **volunteer both of these:**

> **We quote the two shares, not their ratio — and here is the proof that matters.** Earlier drafts
> reported the ratio as "163×", computed on 99 of 120 models. Completing the missing 21 moved it to
> **250×** — the shares barely moved (43.7→42.7, 0.27→0.17), but the quotient jumped 53%, because
> dividing by a near-zero denominator is unstable by construction. The conclusion never changed. **This
> is why the deck quotes shares.**
>
> **Coverage: now complete.** 120/120 models, **fully crossed**, zero empty cells. The earlier 99/120
> gap — all 21 in the CIFAR-FS 5-shot slice — was closed on 2026-09-15; all 21 re-trained cells
> reproduce the committed grid at max absolute difference **0.0**. The unchanged aggregator also still
> reproduces the committed summary on the original 99 records exactly, so the two analyses are
> measuring the same quantity.
>
> **The 37–38/40 (vs MSP/TS-MSP) and 10/40 far-OOD / 14/40 near-OOD (vs energy) win counts** quoted
> above and on Slides 9 and 22 come from `results/mvt_results.json`, a separately-tracked dataset that
> already had all 120 cells present before the 2026-09-15 session — it was **never affected by the
> 21-cell coverage gap** that hit the RQ1/RQ4 analysis. It **was** touched by the same smoke-run cell,
> though: that cell's Gaussian-far and TinyImageNet-near vacuity AUROC are still averaged over 2 of its
> 3 seeds in the committed file (seed 42's smoke-run output never wrote those two metrics), and — unlike
> the RQ1/RQ4 analysis — this file has **not yet been regenerated** with the corrected cell (see
> "Before the defence" below). We checked the impact directly rather than leaving it asserted: patching
> in the corrected 3-seed values changes **no** win/loss and moves every mean Δ by ≤0.0001. **The
> 40-cell counts are unchanged, verified rather than assumed** — but the underlying file is still on the
> to-fix list.

**SAY:** "Forty-two point seven percent against zero point one seven — two to three orders of
magnitude, and the same gap holds on near-OOD. The same score gives near-identical AUROC regardless of
which objective produced the logits. So in this regime the Bayesian *training* was not what bought the
OOD performance; the *score* was. And there is a 2026 theory result showing softmax is a mathematical
special case of an evidential classifier, which pre-explains why the objective would matter so little.
I would rather cite that myself than have it raised from the floor."

**Volunteer:** "Concurrent WACV 2026 work runs a comparable objective × score ANOVA at larger scale, so
the technique itself is not new. What we did not find prior work doing is cross-applying **energy onto
Dirichlet-parameterised logits** and finding near-equivalence. That is the surviving contribution."

---

## Slide 19A — Architecture and parameter budget were initially confounded

**Time:** 1.0 min · **Slide 19 is the most important in the deck and is delivered as two slides. This
is beat one: the pattern, and why it cannot be read at face value.**

**ON THE SLIDE** — the pattern first, from the 16 matched comparisons in the main grid:

| Outcome | Winner is the **bottleneck architecture** | Winner is the **larger-budget arm** |
|---|---:|---:|
| Accuracy | **16/16** | 8/16 |
| Near-OOD AUROC | **16/16** | 8/16 |
| ECE | 8/16 | **16/16** |

Sign consistency 16/16, two-sided p ≈ 3.05×10⁻⁵ under a null of random direction *(reported with its
caveat, not hidden: pairs sharing a backbone are not fully independent draws, so treat this as
descriptive strength, not a clean p-value — the 16/16 sign consistency itself is the number to lean on)*.

Of those 16 pairs, only **10 clear 2 standard deviations** on ECE specifically — **8/8 on
MiniImageNet, but only 2/8 on CIFAR-FS**. The *direction* is consistent 16/16; the *magnitude* is
dataset-dependent. That asymmetry is exactly what the matched-budget experiment on the next slide was
built to adjudicate.

Two observed facts, stated so they cannot be overread:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both
   backbones — holding 2.58× *more* parameters on one and 1.55× *fewer* on the other.
2. **The ECE winner does change, exactly in step with the budget ordering.**

Then the reversal that causes the problem — repeat the Slide 11 numbers here:

| Backbone | Bottleneck | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA** |

Boxed — **the reasoning that makes this a scientific slide rather than a results slide:**

> The budget ordering reverses **with the backbone**. So "calibration follows the budget" and
> "calibration follows the backbone" predict the **identical** 16/16 pattern. **The 120-run grid cannot
> identify which of the two is responsible.** We preferred the budget account on parsimony — which is
> not the same as having evidence for it.

**SAY:** "The accuracy column is clean: the bottleneck wins sixteen out of sixteen whether it is the
bigger adapter or the smaller one, so architecture is doing that work. The calibration column is not
clean, and I want to say why before you ask. In this grid the budget ordering is welded to the
backbone — it reverses exactly when the backbone changes. Two different explanations therefore predict
the same sixteen-out-of-sixteen table. That is a confound in our own design, and no amount of extra
seeds fixes it. It needs a different experiment."

**Sign convention for the next slide, state it here:** **ΔECE = LoRA − bottleneck.** A *positive* ΔECE
means the bottleneck arm is better calibrated; negative means LoRA is.

---

## Slide 19B — Two things survive matched budgets: architecture wins outright, and budget is ruled out as the calibration cause

**Time:** 1.75 min · **Beat two: the pre-registered experiment that adjudicates it. This is your best
45 seconds in the defence — rehearse it.**

**ON THE SLIDE — lead with the clean positive result, then the adjudication:**

> **At matched budget, the bottleneck architecture wins accuracy 8/8 and near-OOD 8/8, all beyond 2σ.**
> The architecture effect does not shrink when its budget advantage is removed — this is the cleanest,
> most positive result in RQ3, and it is budget-controlled by design.

**Design.** Both architectures rebuilt at the **same budget within each backbone**. MiniImageNet
5-shot, 2 backbones × 2 budget levels × 2 arms × 2 heads × 3 seeds = **48 runs**. Residual budget
mismatch **≤ 3.10%**, against 55–158% before. **Decision rule, thresholds and an explicit
"inconclusive" outcome pre-registered before any deciding run existed.**

**Result.** *(ΔECE = LoRA − bottleneck; "collapse ratio" = matched gap ÷ unmatched gap — a
budget-caused gap should collapse toward 0.)*

| Cell | ΔECE unmatched | ΔECE matched | collapse ratio |
|---|---:|---:|---:|
| ResNet-18 / evidential | +0.1111 | **+0.1123** | 1.01 |
| ResNet-18 / softmax | +0.1080 | **+0.0991** | 0.92 |
| MobileNetV3-S / evidential | −0.0124 | −0.0062 | 0.50 |
| MobileNetV3-S / softmax | −0.0152 | −0.0060 | 0.61 |

**Verdict: `backbone_intrinsic`** — fired in **3 of 4** pre-registered cells.
**The parameter-budget hypothesis fired in 0 of 4.**

> **State the asymmetry plainly — it is not "3 of 4, done."** The effect is concentrated on ResNet-18:
> ΔECE ≈ 0.10–0.11, collapse ratio ≈ 1.0 (the gap does not move at all when the budget is equalised).
> MobileNetV3-Small's gap is roughly **10× smaller** (ΔECE ≈ 0.006–0.012) and it *does* roughly halve.
> With two backbones, this result is substantially a ResNet-18 finding that MobileNetV3-Small does not
> contradict — read it as "budget is ruled out on ResNet-18; MobileNetV3-Small is directionally
> consistent but the effect there is small," not as a symmetric result across both backbones.

> **The non-finding, reported rather than omitted:** on **far**-OOD there is no clear adapter effect —
> bottleneck wins only **11/16** unmatched and **5/8** matched. The architecture claim is scoped to
> accuracy and **near**-OOD. RQ1 attributes far-OOD predominantly to the head, not the adapter.

> **Independent corroboration, added 2026-09-15.** RQ1's variance decomposition, extended to two-way
> interactions on the full 120-cell grid, finds the **`backbone:adapter` interaction explains 3.23% of
> ECE variance (p < 1e-6)** — the largest interaction term for calibration, and 42% of what main
> effects left unexplained. Different method, different data slice, same conclusion: the adapter's
> calibration effect depends on the backbone. On accuracy and both OOD pools that same term is ≈0.

Boxed — the claim, scoped exactly:

> **The controlled matched-budget experiment provides evidence against parameter budget as the primary
> explanation for the calibration gap.** Equalising the budget leaves ResNet-18's gap entirely intact
> and only halves MobileNetV3-Small's. The results are consistent with a backbone-dependent mechanism —
> **but which intrinsic backbone property is responsible remains unidentified.**

**SAY (rehearse this):** "Start with the positive: at matched budget, bottleneck still wins accuracy
and near-OOD eight-for-eight, all beyond two standard deviations — the architecture result is
budget-controlled and it holds. Now the adjudication: equalising the budget leaves ResNet-18's
calibration gap intact — collapse ratios of 1.01 and 0.92, meaning the matched gap is as large as the
unmatched one. A gap caused by the budget difference should have shrunk toward zero with it. It did
not. So the hypothesis this project had been working under for two months is not supported; we
pre-registered the rule that would tell us so before we ran it. I'll be precise about scope, too: that
result is carried almost entirely by ResNet-18 — MobileNetV3-Small's gap is an order of magnitude
smaller and does partially shrink, which is consistent with the verdict but not an equally strong
instance of it."

**State the limit before you are asked:** "'Backbone-intrinsic' is where the evidence points, not an
explanation. Which property of ResNet-18 versus MobileNetV3-Small — depth, width, normalisation, the
inverted-residual block, feature-norm scale at the adapter sites — produces the reversal is untested,
and two backbones cannot separate those candidates. That is the honest remaining limit of RQ3, and
closing it takes more backbones, not more seeds."

**ABSOLUTELY DO NOT SAY:** "calibration follows the parameter budget" (superseded), or "we proved the
backbone causes calibration" (the design does not identify the mechanism).

---

## Slide 20 — Two parameters cut evidential ECE nearly in half — but it stays worse than softmax

**Time:** 1.5 min

**ON THE SLIDE**

**What was refit:** only the **two** parameters of the evidence affine (scale, bias), on the
**validation** episode split, evaluated on the frozen 600-episode test split. **No retraining.**

Three headline numbers, equal weight — **the third is not optional**:

> ### Calibration: ECE improved in **60 / 60 cells (100%)** — mean **−0.139** absolute (0.323 → 0.185)
> ### OOD ranking: preserved in **192 / 240 comparisons (80%)** — **20% were not**
> ### Still short of softmax: **60 / 60** cells remain worse-calibrated than plain softmax

Supporting detail *(all 60 evidential cells, full 120/120 coverage as of 2026-09-15)*:

| Outcome | Result |
|---|---|
| Where training had left the affine | `(scale, bias)` is **learnable**, initialised at (2, −6); training moved it to scale **1.51–4.52** (median 2.90) |
| Where the refit puts it | scale **3.61–14.54** (median 8.99) — the jointly-trained operating point was far from the calibration-optimal one |
| Residual gap after refit | worse than plain softmax in **60/60** (mean **2.18×**, best case 1.07×, worst 5.8×); worse than TS-softmax in **60/60** (mean **13.6×**, best 2.27×) |
| Absolute ECE, same 60 cells | evidential post-refit **0.185** vs softmax **0.112** (means) — the 2.18× headline above is a *mean of per-cell ratios*, not this *ratio of means* (0.185/0.112 ≈ 1.65×); both are correct, they answer different questions, and the mean-of-ratios is the one "worse in every cell" needs |
| Preservation criterion | ΔAUROC ≥ −0.005; mean ΔAUROC across comparisons **+0.003** |
| Worst-case rank correlation | Spearman ρ = **0.866** |

> **What adding the 12 new evidential cells changed.** ECE improvement held at 100% (12/12). Ranking
> preservation on the new cells alone was **42/48 (87.5%)**, but their mean ΔAUROC was slightly
> *negative* (−0.0016) and they contain the new worst-case ρ of 0.866 — so fuller coverage made this
> result marginally **weaker**, not stronger. Reported in that direction deliberately.

Caveat — **on the slide, not just spoken**:

> **Reordering does occur.** Vacuity is K/Σα, a function of *all* logits jointly, so a per-logit
> monotone transform does not guarantee the sample ordering survives. In **20%** of comparisons AUROC
> dropped by more than half a point; worst single-pool drop **−0.032** (Gaussian far-OOD). The claim is
> **"OOD ranking is preserved in 80% of comparisons"** — never "always."
>
> **And the repair is partial.** Post-refit evidential ECE is still worse than plain softmax in **all
> 60** cells. The correct claim is **"refitting improves calibration in 60/60 cells but does not close
> the gap to softmax"** — never "refitting fixes evidential calibration."

**SAY:** "Two parameters, refitted on validation episodes, no retraining — and ECE falls in every
single cell. Note what that says: these two scalars are *trained* jointly with the adapter, so the
evidential loss was leaving them far from the calibration-optimal point on its own. That is a sharper
finding than a badly-chosen default would have been. Two things I want to be exact about, though. The
refit **narrows** the gap to softmax, it does not close it — evidential is still the worse-calibrated
arm in all sixty cells. And we *measured* whether the OOD ranking survived rather than assuming it,
because in twenty percent of comparisons it did not."

**Volunteer:** "Post-hoc calibration is a mature field, and Guo et al. and BEL report comparable or
larger drops. We are not claiming to have discovered that refitting helps. What is ours is the
mechanism — this specific two-parameter evidence affine, which turns out to be the knob that actually
controls calibration here — and the systematic quantification across 60 cells and 240 ranking
comparisons, including the failures."

**The self-correction that belongs with this slide:** Step 4.5 swept the *loss* (KL weight × variance
term) and found a flat calibration surface, and we concluded calibration could not be tuned. That search
was tuning the wrong knob — the evidence affine was never swept. RQ4 is what found it.

---

## Slide 21 — The frozen trunk dominates inference cost; the adapter and head barely move it

**Time:** 0.75 min · **Section: H. Practical implications**

**ON THE SLIDE** — measured on real hardware (Kaggle T4, GPU and single-thread CPU). Three numbers, no more:

| Design choice | Effect on latency |
|---|---:|
| **Backbone** | **5.12×** (ResNet-18 62.18 ms vs MobileNetV3-S 12.14 ms, matched adapter) |
| Adapter | **3.9–5.8%**, despite up to 2.6× parameter differences |
| Evidential head | **1.29%** mean — below the session's own **5.91%** noise floor |

Boxed, the deployment rule:

> **Choose the backbone for latency and the adapter for accuracy** — they barely trade against each
> other. **Recommended point (CIFAR-FS):** MobileNetV3-Small + parallel bottleneck + evidential —
> **11.86 ms/image**, **6,930 trainable parameters**, TinyImageNet near-OOD AUROC **0.870** (1-shot) /
> **0.919** (5-shot). **On MiniImageNet**, where MobileNetV3-Small falls outside the accuracy
> tolerance, the point moves to ResNet-18 + parallel + evidential at **62.38 ms**.
>
> **This recommendation knowingly accepts a calibration cost.** Evidential's post-refit ECE (0.185
> mean, Slide 20) is still worse than softmax's in the same cells (0.112 mean) — the choice trades some
> calibration for vacuity's OOD-ranking edge and its native uncertainty signal, not a free win on
> every axis.

> **Say this unprompted — the recommendation is conditional on the scoring rule.** It holds when each
> head uses its *native* score (evidential → vacuity, softmax → MSP). If softmax is instead allowed its
> best score — energy — evidential's presence on the CIFAR-FS 5-shot latency/AUROC frontier goes to
> **zero**. This is not a contradiction of Slide 18; it is the same finding applied to deployment.

**VISUAL:** `results/pareto_latency_vs_auroc__cifar_fs.png`, `__mini_imagenet.png`.

**SAY:** "The practical rule is the boxed line: backbone for latency, adapter for accuracy, because
they barely trade against each other. And 'evidential uncertainty is free at inference' is now a
*measurement below our own noise floor* rather than a claim — Sensoy's 2018 paper asserts it as EDL's
selling point; we did not find it measured in this regime. One condition I want to state before it's
asked: this recommendation holds for each head's native score. Give softmax its best score, energy, and
evidential drops off the CIFAR-FS frontier entirely — which is the same RQ2 finding, now applied to a
deployment decision rather than an OOD-ranking one."

**Provenance note if asked:** latency is the one family of numbers here that is not byte-reproducible
by design — hardware- and session-dependent. All values come from one canonical Kaggle T4 session (GPU
and single-thread CPU); see self-correction 4 on Slide 22 for the selection bug found and fixed here.

---

## Slide 22 — What the experiments changed our mind about, and what they leave unresolved

**Time:** 1.5 min · **Section: I. Limitations and self-corrections** · **This slide is worth more than
any positive result on it. Committees remember it. Deliver it as boundary-setting, not as apology.**

**ON THE SLIDE** — two clearly separated sections, the left one given more visual weight:

### Section 1 — What the experiments changed our mind about *(five items; the fifth was found while completing coverage on 2026-09-15)*

1. **Energy vs vacuity: a single-configuration finding did not generalise.** An early result said
   evidential vacuity was roughly on par with the energy score. At grid scale, **energy wins ~70% of
   comparisons** (vacuity wins only 10/40 far-OOD, 14/40 near-OOD); on the finer matched
   (design, seed, pool) comparison used for RQ2's η² analysis, energy's win rate is higher still —
   **81.7% far-OOD (196/240), 93.8% near-OOD (225/240)**. *Narrowed claim: vacuity is a substantially
   better OOD ranker than softmax-probability scores — 37–38 of 40 cells — but a well-chosen logit-space
   score was the stronger ranker in this study, more decisively at the finer grain than the coarse one.*
2. **The parameter-budget hypothesis: a pre-registered experiment did not support our own preferred
   reading.** RQ3's "calibration follows the parameter budget" account fired in **0 of 4** cells once
   budget was equalised (Slide 19B).
3. **The interior-optimum hypothesis was tested and demoted.** We predicted calibration error would
   reach an optimum at an intermediate parameter budget. A controlled **21-run rank sweep** with
   everything else fixed found **no interior optimum in the tested range** — evidential ECE is lowest
   at rank 1 and drifts up; softmax moves the opposite way. Reported as a negative result — though
   **3 seeds × 7 ranks is underpowered to rule out a subtle U-shape** between the tested points; the
   honest claim is that no *large* interior optimum was found, not that the surface is proven monotonic.
4. **A silent latency-selection bug, caught by manual cross-check rather than by the test suite.**
   Three independent latency-selection functions were silently preferring this repo's noisy dev-laptop
   measurements over the canonical Kaggle CPU numbers — by dict-insertion-order accident, with
   **errors up to 47% on individual cells**, no crash and no failing test. Every downstream artefact
   (efficiency table, Pareto JSON, all six Pareto figures) was regenerated from corrected code. *A
   regression test for this bug class is the top open follow-up.*
5. **A smoke run had been sitting in the published grid — found 2026-09-15, while closing the coverage
   gap.** The committed metrics file for **one cell** — `cifar_fs/5shot/mobilenetv3_small/lora/
   evidential`, seed 42 — had been written by a **20-episode smoke run**, not the 600-episode protocol.
   It carried 8 metric keys instead of 12, which is why RQ1's near-OOD row was missing one observation
   (95 instead of 96), and why that configuration's seed spread in `RESULTS_MASTER.md` (±1.82) is an
   outlier against every other 5-shot cell (±0.50–0.63). **Fixed, not just measured:** the cell was
   re-scored over the full 600 episodes, and every η² shown on Slides 17, 19B and 23 in this deck
   already reflects that fix — every term in every outcome's table (main effects and interactions)
   moved by **≤ 0.44 pp** (largest: near-OOD `k_shot` 42.86% → 43.30%), and the `backbone:adapter` calibration
   finding moved from 3.28% to **3.23%**, still the largest interaction term for ECE. No conclusion in
   this deck changed. *It is disclosed because the thesis claims byte-identical reproducibility on a
   fixed 600-episode protocol, and for this one cell that claim was briefly not true — it is true now.*
   A related tooling gap is disclosed on Slide 17: the missing observation didn't trip `eta_squared()`'s
   balance check, because that check only inspects cells with at least one seed present.

### Section 2 — What remains unresolved

1. **Pretraining overlap** — ImageNet-pretrained backbones mean absolute accuracies are not comparable
   to from-scratch few-shot work. All RQs concern relative differences under one shared pretraining.
2. **One frozen hyperparameter recipe** across all 40 configurations. Deliberate: it keeps the grid a
   controlled comparison, not 40 independently-tuned numbers.
3. **Three seeds, and a narrow one** — the seed changes only the adapter's initialisation, never the
   order of training episodes. Full-FT and Linear-Probe have **zero** seed spread (effectively one run
   each), and exist only for ResNet-18 + CIFAR-FS. Sub-1-point margins should be read accordingly.
4. **Two backbones** — they establish that backbone identity matters; they cannot establish *which
   property* is responsible.
5. ~~**Main effects only**~~ — **closed 2026-09-15.** The η² decomposition now includes all ten two-way
   interaction terms with bootstrap intervals (Slide 17). Three-way and higher terms remain unmodelled.
6. **Temperature-scaling asymmetry** — evidential is not temperature-scalable in this codebase, so the
   TS-softmax comparison is structurally favourable to softmax. Still the right comparison; the
   asymmetry is stated rather than hidden.
7. **Literature-search limitations** — novelty claims rest on a non-exhaustive search: **five**
   independent passes, ~100 searches, full PDF reads of the closest competitors
   (`RQ_SUPERVISOR_REPORT.md` Appendix C). Absence of found prior art is not evidence of absence.
8. ~~**RQ2 and RQ4 cover 99 of 120 models**~~ — **closed 2026-09-15.** All 21 were re-trained and scored;
   coverage is 120/120, fully crossed. They reproduced the committed grid exactly. Adding them made RQ4's
   ranking-preservation result slightly *weaker* (worst ρ 0.921 → 0.866), which is reported as found.
9. **Protocol choices that apply to every cell equally** — no data augmentation; a cosine×10 prototype
   metric rather than ProtoNet's squared-Euclidean; and `kl_weight_max` fixed at 0.1 although the Step 4.5
   VAL sweep ranked 0.05 first (VAL ECE 0.252 vs 0.260), with no logged reason for the override. None
   confounds a comparison; all three are disclosed rather than found.

Closing line, boxed, at the bottom of the slide — **the single most important open question:**

> **What intrinsic property of the backbone drives the observed calibration behaviour?**
> Two backbones cannot separate depth, width, normalisation, block type or feature-norm scale.
> Closing this takes more backbones, not more seeds.

**SAY:** "A thesis where every hypothesis was confirmed would be more suspicious, not less. The left
column is on the slide because we found those things ourselves, at the point where the evidence got
strong enough to find them. Number four is the one I would most want you to notice: it produced no
error and broke no test, and it was caught because the numbers were cross-checked by hand against the
machine they were supposed to come from. Number five is newer and I found it two days ago: one cell in
the published grid turned out to have been scored by a twenty-episode smoke run instead of the six
hundred. I re-scored it and recomputed every affected number — under a percentage point of movement on
any effect, nothing on the conclusions — and every table in this deck already reflects the correction.
It's on the slide anyway, because the thesis claims a fixed protocol and for that one cell the claim was
briefly not true — it is true now. On the right, the boxed question at the bottom is the one I would
spend the next six months on."

**Prepare for "doesn't the energy result undermine your premise?"** — "It narrows it. The premise was
never that Bayesian uncertainty is the best possible OOD score. RQ2 is exactly the experiment that
separates those two claims, and its answer is that the score explains far more of the variation. What
survives is that the evidential head is the better *native probabilistic* option, that it is free at
inference within our noise floor, and that its calibration deficit is repairable post-hoc. If you only
need an OOD ranker, the honest recommendation is softmax plus energy."

---

## Slide 23 — Four findings: different design choices govern different aspects of reliability

**Time:** 1.5 min · **Section: J. Final contribution**

**ON THE SLIDE** — the four findings, large, as the visual core of the slide:

> ### Accuracy → **shots + adapter architecture**
> ### Calibration → **head + backbone**
> ### OOD detection → **scoring rule**
> ### Repair → **post-hoc recalibration improves ECE in 60/60 cells; OOD ranking survives in 80%**
> ### …and the gap to softmax **narrows without closing** — evidential is still worse in 60/60

Then, smaller — **contributions, ordered by strength, labelled honestly:**

1. **[NOVEL] A dissociation between adapter architecture and adapter parameter budget, by outcome
   metric.** Accuracy and near-OOD ranking follow **architecture**, invariant to budget, tested at
   matched budget. Calibration is not explained by budget; the evidence supports **backbone**
   dependence. Established by a pre-registered experiment that did not support the project's own
   working hypothesis.
2. **[PARTLY NOVEL] Separating the OOD training objective from the OOD scoring rule** — the score
   explains **42.7%** of far-OOD variance against the objective's **0.17%**, on a fully-crossed 120-model
   design. Specifically new: cross-applying the **energy score onto Dirichlet-parameterised logits** and
   finding near-equivalence.
3. **[PARTLY NOVEL] A formal variance decomposition** of accuracy, calibration and OOD detection over a
   balanced five-axis PEFT factorial, **main effects and all ten two-way interactions with bootstrap
   intervals** — including that **ECE and OOD AUROC become close to orthogonal once head interpretation
   is controlled**, and that the **`backbone:adapter` interaction carries 3.23% of ECE variance
   (p < 1e-6)**, independently corroborating RQ3.
4. **[PARTLY NOVEL] A two-parameter post-hoc recalibration of an evidential prototype head**, with
   OOD-ranking preservation **measured rather than assumed** (60/60 improved, 80% preserved, and the
   residual gap to softmax reported rather than dropped).
5. **[CONFIRMATORY, and useful] A reproducible harness and evidence base** — 189 runs, frozen episode
   seeds, VAL-only selection, and byte-identical reruns demonstrated **twice independently** (18 cells in
   August, 21 more in September, both at max absolute difference 0.0).

**Future work** (each one closes a numbered limitation):

1. **More backbones** — the only way to identify *which* backbone property is associated with the
   calibration reversal *(closes limitation 4)*.
2. **A from-scratch control run** — removes the pretraining-overlap objection entirely *(closes 1)*.
3. **A transformer arm under our own protocol** — converts "CNNs are appropriate here" from an argument
   into a measurement.
4. **Per-cell VAL tuning** — answers whether any axis effect is a recipe artefact *(closes 2)*.
5. **An evidential analogue of temperature scaling**, so both heads get a post-hoc correction *(closes 6)*.
6. **Three-way and higher interactions**, and an explanation for the large `dataset:backbone` term the
   two-way decomposition surfaced — which no current research question addresses.

**The last line on the slide, and the last thing you say:**

> **The main contribution is not a universally superior model, but a controlled decomposition of which
> design choices govern which aspects of reliability in this few-shot PEFT regime.**

**SAY:** "Read the four lines at the top and then the line at the bottom. If one thing survives this
defence, I would want it to be that sentence — and the reason I would defend contribution one hardest
is the pre-registration, not the result."

**No new result appears on this slide.** Every number here has already been shown.

---

# References (2 slides — not counted in the 25)

IEEE style, numbered to match in-text citations, grouped:

**Few-shot / adapters.** Snell et al. (ProtoNet); Bertinetto et al. (CIFAR-FS / R2D2); Ravi &
Larochelle (MiniImageNet); Li, Liu & Bilen, *Task-Specific Adapters*, CVPR 2022 (arXiv:2107.00358);
Conv-Adapter (arXiv:2208.07463); Hu et al., *LoRA*; Houlsby et al. (bottleneck adapters); Hu et al.,
*P>M>F* (arXiv:2204.07305).

**Uncertainty / calibration.** Sensoy et al. 2018 (arXiv:1806.01768); Guo et al. 2017
(arXiv:1706.04599); Minderer et al. 2021 (arXiv:2106.07998); Kull et al. 2019 (Dirichlet calibration);
Bengs et al., NeurIPS 2022; BEL (arXiv:2207.13137); BayesAdapter (arXiv:2412.09718); Laplace-LoRA,
ICLR 2024.

**OOD.** Liu et al. 2020, energy score (arXiv:2010.03759); OpenOOD / OpenOOD-1.5; arXiv:2603.07571
(objectives for OOD); arXiv:2605.06382 (vacuity cardinality critique).

**Concurrent work, cited proactively.** arXiv:2601.10836 (WACV 2026, objective × score ANOVA);
arXiv:2605.22746 (softmax as a special case of evidential); arXiv:2608.10372 (accuracy-preserving
post-hoc calibration).

**Backbones / edge.** He et al. (ResNet); Howard et al. (MobileNetV3); TinyDL survey
(arXiv:2506.18927); few-shot geometric-estimation CNN/ViT comparison (arXiv:2510.04794, cited for
direction, not as classification-specific evidence — see Slide 3); DINOv3 (arXiv:2508.10104); PEFT-for-ViTs
survey (arXiv:2402.02242 — documents the calibration/OOD absence discussed on Slide 8).

> **Every citation above resolves to an entry in `docs/refs.bib`.** Two arXiv IDs that appeared in earlier
> drafts of this deck (GP-Adapter, Dual-Adapter) had no entry in `refs.bib`, `CITATION_AUDIT.md` or any
> report, and have been removed rather than cited unverified. Of the citations this deck actually uses,
> **two carry an open flag** in `CITATION_AUDIT.md`: *LoRA vs Full Fine-tuning: An Illusion of
> Equivalence* (arXiv:2410.21228 — still needs re-checking before use anywhere) and arXiv:2510.04794
> (flagged as used outside its own domain — already hedged on Slide 3 as "cited for direction, not
> classification-specific evidence"). **arXiv:2605.22746** — the softmax-as-a-special-case-of-evidential
> result spoken aloud on Slide 18 — carries a complete, unflagged `refs.bib` entry; it is not one of the
> open items.

> Full BibTeX: `docs/refs.bib`. Citation provenance: `docs/CITATION_AUDIT.md`.

---

# Q&A crib sheet

One or two sentences each, plus a slide number. **Long answers lose defences.**

| Question | Answer | Slide |
|---|---|---|
| "If energy beats vacuity, why is this a Bayesian thesis?" | The evidential formulation is what supplies an explicit uncertainty representation to measure at all; we never claimed it universally produces the best OOD detector. RQ2 is the experiment that separates objective from score, and RQ4 shows the calibration deficit it creates is repairable post-hoc. | 18, 20, 22 |
| "Isn't a CNN backbone outdated?" | The deployment regime selects the family: transformer attention costs an order of magnitude more compute than CNN inference on comparable edge hardware (~15–20×, TinyDL survey). In the low-data regime other vision tasks report CNNs matching ViTs. We make no general CNN-superiority claim, and we did not measure on MCU hardware ourselves. | 3, 21 |
| "Your accuracy isn't state of the art." | Correct, and it isn't trying to be. 1.06 points behind a fully meta-trained DINO ViT-S at 662× fewer trainable parameters, and 3.56 *ahead* of the backbone-matched ResNet-50. Every RQ is about relative differences. | 16 |
| "Your backbones saw the test classes during ImageNet pretraining." | Yes — stated as limitation 1. It makes absolute accuracies non-comparable to from-scratch work. It does not touch any RQ, because every RQ compares configurations that all share the same pretraining. | 14, 22 |
| "Why did the research questions change from the proposal?" | They were refined from comparison to attribution after a literature review that came *after* the grid completed. No experiment was discarded, no data removed, and every original question is answered — including the two whose answer was "no." | 9 |
| "Did you change the questions because the results were inconvenient?" | The two negative results are still here, on Slides 9 and 22. If the goal had been convenience, the 0-of-20 calibration result is the first thing that would have disappeared. | 9, 22 |
| "Three seeds is not many." | Agreed for sub-1-point differences, and we flag those. Seed spread exceeds 1 point in only 2 of 40 configurations, and every RQ3 claim at matched budget clears 2σ. | 22 |
| "One hyperparameter recipe across 40 configurations isn't fair." | Deliberate — it makes the grid a controlled comparison rather than 40 independently-tuned numbers. We answer "how do these axes compare under one recipe," not "what is each cell's best number." | 22 |
| "Energy outperforms your Bayesian score — doesn't that sink the thesis?" | It narrows it, and RQ2 is the experiment that made it visible. Vacuity is the better *native probabilistic* score — 37–38 of 40 cells against MSP — and free at inference within our noise floor. If you only need an OOD ranker, softmax plus energy is the honest recommendation. | 18, 22 |
| "Isn't 'backbone-intrinsic' just a name for 'we don't know'?" | Largely yes, and we say so. What the experiment establishes is that it is **not** the budget — the gap survives budget equalisation intact on ResNet-18. Which property is responsible needs more backbones. | 19B, 22 |
| "Two backbones can't support a claim about backbones." | Agreed, and the claim is scoped accordingly: backbone identity matters; the responsible property is unidentified. That's the stated limit and the first item of future work. | 19B, 23 |
| "What exactly does your 42.7% η² mean?" | The share of total variation in far-OOD AUROC attributable to the scoring-rule axis, in a main-effects decomposition over a fully-crossed 2 objectives × 4 scores factorial, n = 960 per pool, all 120 models. The objective axis gets 0.17%. | 18 |
| "Why not just give the ratio?" | Because it isn't stable, and we can prove it. The objective's share is near zero, so the quotient swings: 163× at 99/120 coverage, 250× at full 120/120 — while the two shares barely moved. Same conclusion either way, so we quote the shares. | 18 |
| "Is η² appropriate — pairs sharing a backbone aren't independent?" | A fair objection to the sign test over 16 pairs. The decomposition itself is main-effects η² over a balanced factorial on 96 **per-seed** observations with a retained residual — we recomputed per-seed precisely because averaging seeds first can manufacture cleanliness. | 17 |
| "You ran 32×32 CIFAR images through an ImageNet ResNet?" | Resized to 224×224 and ImageNet-normalised first. The backbone is frozen and ImageNet-pretrained, so matching its input statistics is a requirement, not a choice. | 14 |
| "Why a prototype head instead of a linear classifier?" | A linear head trained on the 64 base classes cannot transfer to 20 disjoint test classes. Protocol requirement, not a design shortcut. | 10 |
| "Does post-hoc recalibration break the OOD detection?" | In 80% of comparisons it does not; in 20% AUROC dropped by more than half a point, worst single-pool drop −0.032. We measured it rather than assuming it, and the exception is on the slide. Note completing coverage made this slightly worse, not better — we report it in that direction. | 20 |
| "What was the hardest bug?" | An evidential collapse: raw L2 logits are large-negative for ResNet-18 features, so softplus of the logit is ≈0 everywhere — uniform Dirichlet, dead gradients. Fixed with cosine similarity plus a learnable evidence affine; the mapping now lives in exactly one function shared by training and evaluation. | 12 |
| "How do you know the latency numbers are right?" | They come from one canonical Kaggle T4 session, GPU and single-thread CPU. During closeout we found three functions silently preferring a dev-laptop measurement instead — up to 47% error, no crash, no failing test — fixed it, and regenerated every downstream artefact. It's reported as the fourth self-correction. | 21, 22 |
| "How do you know the results are reproducible?" | Frozen episode seeds, sorted-key JSON, byte-identical reruns — checked twice: the matched-budget experiment (Aug 26–27) re-trained 18 arms from the Aug 2–6 grid, and an independent September session re-trained 21 more; both reproduced the committed numbers at max absolute difference **0.0**. 39 cells, two occasions. | 15 |
| "Why is your main grid's adapter rank 16, when your rank sweep shows evidential ECE is lowest at rank 1?" | Rank 16 was fixed as part of the one shared hyperparameter recipe (Slide 15), chosen before the rank sweep existed. The sweep was a later, narrower diagnostic testing one hypothesis (an interior optimum) with everything else held fixed — it isn't a per-cell retuning experiment, and it doesn't imply the grid's rank choice was wrong, only that lower rank tends to help evidential calibration in the tested range. Retuning the frozen grid's rank is future work (per-cell VAL tuning), not a retrofit. | 22, 23 |
| "What's driving the dataset:backbone interaction you won't interpret?" | We have a candidate, not a claim: MiniImageNet is ImageNet-derived and CIFAR-FS is not, so the two backbones' shared ImageNet pretraining likely transfers differently depending which dataset they're adapted to — that's limitation 1 (pretraining overlap) surfacing as a measured interaction rather than an assumption. It's plausible, not tested, which is why we report the number and don't build a claim on it. | 17 |
| "In what sense is this actually Bayesian? There's no posterior over weights." | Fair distinction. Evidential deep learning is second-order — a Dirichlet over the simplex from one deterministic forward pass, not a posterior over parameters. We use the field's own term for it and state plainly that it is not weight-space Bayesian inference; the thesis's "Bayesian" claim is about the uncertainty representation, not about posterior weight sampling. | 5, 12 |
| "You ran the 120-run grid (Aug 2-6) before your literature review found the four questions already had precedent (Aug 21-23). Why?" | A fair critique of the process, not just the framing. The proposal's own review covered four papers; the deeper pass came after the grid. What we can say is the experimental design — the balanced factorial — was built for attribution from the outset, which is why the same runs support the refined questions without rerunning anything. We are not claiming the timing was ideal. | 9 |
| "Doesn't the interaction analysis just restate RQ3?" | It corroborates it from a different direction — full 120-cell grid, variance decomposition rather than matched pairs — and it is scoped narrower: the backbone×adapter term is significant on calibration (3.23%, p<1e-6) and ≈0 on accuracy and both OOD pools. RQ3's pre-registered matched-budget experiment is still the stronger, causally cleaner evidence. | 17, 19B |
| "One of your grid cells was scored on 20 episodes instead of 600?" | Yes — one cell, one seed, found on 2026-09-15 and disclosed as self-correction 5. We re-scored it properly and measured the impact: under a percentage point on any η² term, no conclusion affected. It is on the slide because the protocol claim matters even when the number doesn't. | 22 |
| "What would you do with six more months?" | More backbones, to identify the mechanism behind RQ3. Everything else on the future-work list is worth less. | 23 |

---

# Language discipline — phrases that will be checked against the record

**General rules, applied throughout this deck:**

| Avoid | Use instead |
|---|---|
| "proves" | "provides evidence that" |
| "causes" | "is associated with" / "the controlled comparison supports" |
| "no precedent", "we are the first" | "we did not find prior work combining these factors under this experimental regime" |
| "always", "never", "universally", "SOTA" | scoped wording: "in this experimental regime", "our controlled comparison indicates", "we did not find evidence that…" |

**Specific phrases, checked against this thesis's record:**

| Never say | Say instead |
|---|---|
| "Calibration follows the parameter budget." | "Calibration was not explained by parameter budget; the matched-budget experiment supports backbone dependence — and the responsible backbone property is unidentified." |
| "We proved the backbone causes calibration." | "The controlled matched-budget experiment provides evidence against parameter budget as the primary explanation." |
| "Evidential uncertainty is on par with energy." | "Vacuity provides a useful uncertainty score, but energy was the stronger OOD ranker in this study." |
| "Our 6,928-parameter adapter **beats** full fine-tuning." | "It **matches** full fine-tuning — the margin is inside its own seed spread. The ResNet-18 configuration's +0.97 points is the margin that clears noise." |
| "Refitting always preserves the OOD ranking." | "OOD ranking is preserved in 80% of comparisons (192/240), with a measured minority exception." |
| "Refitting fixes evidential calibration." | "Refitting improves it in 60/60 cells, but it stays worse than plain softmax in all 60." |
| "The score matters 163× more than the objective." | "The score explains 42.7% of far-OOD variance (14.7% near); the objective under 1%. Two to three orders of magnitude — the ratio itself is unstable: it moved 163×→250× purely by completing coverage." |
| "The evidence affine was frozen at (2, −6)." | "It is learnable, initialised at (2, −6); training moved it to scale ~1.5–4.5, and the refit to ~3.6–14.5." |
| "Our serial-vs-parallel result replicates TSA." | "We tested placement once and they tied on accuracy; TSA's placement evidence is stronger than ours, so we cite them for it." |
| "Our 6,928-parameter adapter matches full fine-tuning." *(unqualified)* | "…matches full fine-tuning **of ResNet-18** — we never ran full fine-tuning on MobileNetV3-Small, so that row is a cross-backbone comparison." |
| "We checked the literature six times." | "Five independent passes, around a hundred searches (Appendix C)." |
| "Calibration error has an interior optimum." | "No interior optimum was observed **in the tested range**." |
| "We discovered that accuracy and calibration have different drivers." | "We formally decompose these effects in this experimental regime; the qualitative pattern goes back to Guo et al. 2017." |
| "We're the first to do a factorial ANOVA on OOD." | "Concurrent WACV 2026 work does this at larger scale. What we did not find prior work doing is cross-applying energy onto Dirichlet logits." |
| "Our method is state of the art." | "We are 1.06 points behind a ViT that trains 662× more parameters, and ahead of every backbone-matched comparison." |
| "All four hypotheses were confirmed." | "The four research questions were answered, including cases where the initial expectation was not supported." |
| "Bayesian is better." | "The evidential head gives an explicit uncertainty representation and the better native probabilistic score; it does not give the best OOD ranker in this study." |

---

# Asset checklist — figures to export before building slides

| Used on | Asset | Path |
|---|---|---|
| Slide 3, 17 | Reliability diagrams | `results/grid_plots/*_reliability.png` |
| Slide 18 | OOD separation histograms | `results/grid_plots/*_ood_histogram.png` |
| Slide 21 | Pareto, latency vs AUROC | `results/pareto_latency_vs_auroc__cifar_fs.png`, `__mini_imagenet.png` |
| Have ready if asked | Full accuracy table, rendered | `results/mvt_table_accuracy.png` |
| Have ready if asked | Full calibration table, rendered | `results/mvt_table_calibration.png` |
| Have ready if asked | Full OOD AUROC table, rendered | `results/mvt_table_ood_auroc.png` |
| Have ready if asked | Rank sweep | `results/rq5_rank_sweep.png` |

**Four figures you must draw yourself** — the repo has no version of any of them, and they are the
most important visuals in the deck:

1. **Slide 10 — the system pipeline.** Frozen backbone → adapter → prototype head → two
   interpretations, with the frozen / trainable / parameter-free annotations.
2. **Slide 13 — the factorial.** Five axes with two levels each, the 32 + 8 cell count, and the three
   experiments kept visually distinct. This slide carries the thesis's method; a plain bulleted list
   wastes it.
3. **Slide 11 — the two adapter architectures side by side.** Bottleneck-parallel (down / ReLU / up,
   summed at the block output) against LoRA (a low-rank update *inside* the frozen 1×1 convolution).
   The difference between "after the block" and "inside the weights" is the whole of RQ3's
   architecture axis, and it is hard to convey in words.
4. **Slide 9 — the chronology.** Three labelled columns, left to right: original proposal → completed
   experiments → refined attribution questions. The left-to-right arrow is doing the defensive work;
   a table alone does not show that the experiments came before the refinement.

---

# Before the defence — re-check against the repo, not against this file

1. **Run counts** on Slides 2 and 13 (`progress.txt`).
2. **The recommended deployment point** on Slide 21 — the only numbers in the thesis that are
   not byte-reproducible by design, because latency is hardware- and session-dependent.
3. **Your own title-slide details** on Slide 1 — the repo has no record of student IDs or supervisor
   names, so they are placeholders here.
4. **The official title** — Slide 1 carries the registered title unchanged, with the attribution-study
   framing as a subtitle. If your department has registered a different string, the registered string
   wins and only the subtitle may be edited.

**Corrections this deck applies that the source documents still need** (from
`docs/guide/05_problems_and_open_work.md`; fix them before the thesis text is written):

| Item | What | Where it still needs fixing |
|---|---|---|
| A1 | Drop the unstable "163×"; quote the η² shares (now 42.7% / 0.17% at full coverage) | `RQ_SUPERVISOR_REPORT.md` §1/§4.2 + the `.pdf`; `RQ_RESULTS_SUMMARY.md`; `DEFENCE_SLIDE_PLAN.md` |
| A2 | State that the refit leaves evidential worse than softmax — now **60/60**, not 48/48 | added to `RQ_SUPERVISOR_REPORT.md` §6.1 — the `.pdf` is stale, regenerate |
| A3 | Orig-RQ1 is mislabelled "serial vs parallel" | `RQ_SUPERVISOR_REPORT.md` Appendix A; `RQ_RESULTS_SUMMARY.md` |
| A6 | The evidence affine is learnable, not frozen at (2, −6) | fixed in the `.md` files; `RQ_SUPERVISOR_REPORT.pdf` still stale |
| A7 | Do not describe `kl_weight_max` 0.1 as VAL-selected | `09_methodology.md` §9.5 discloses it; add a `progress.txt` entry if the reason is known |
| A8 | State the cosine×10 prototype metric | the reports say only "similarity" |
| — | Two unverifiable citations removed from this deck | confirm they are absent from the thesis draft too |
| **NEW** | **Coverage is now 120/120** — every "99 of 120" statement is superseded | `RQ_SUPERVISOR_REPORT.md`; `RQ_RESULTS_SUMMARY.md`; `results/rq_summary.json` and `results/rq_checkpoint_audit.json` still hold the 99-record snapshot **by design** (kept as the "before" baseline) |
| **NEW** | **RQ4 is now 60 cells / 240 comparisons** (60/60 improved, 192/240 preserved, worst ρ 0.866) | `RQ_SUPERVISOR_REPORT.md` §6; `RQ_RESULTS_SUMMARY.md` |
| **NEW** | **RQ1 two-way interactions exist now** — "main effects only" is no longer a limitation | `RQ_SUPERVISOR_REPORT.md` §3; `docs/guide/03_results.md` §3.3; `09_methodology.md` |
| **NEW** | **One grid cell's committed metrics were a 20-episode smoke run** (`cifar_fs/5shot/mobilenetv3_small/lora/evidential` seed 42) | Fixed in this deck — Slides 17, 19B and 23 already show the corrected 43.30%/3.23% figures. Still open: `results/mvt_results.json`, `results/grid/`, and `docs/RESULTS_MASTER.md` (its ±1.82 seed spread is the visible symptom) have not been regenerated from the corrected cell; do that before the thesis text is finalised |

**Provenance for everything else:** `docs/RQ_SUPERVISOR_REPORT.md` (four-RQ framing, answers, novelty
labels) · `docs/RQ_RESULTS_SUMMARY.md` (RQ tables, 16-pair evidence, §5.1 matched-budget) ·
`docs/RESULTS_MASTER.md` (Tables 1–8, §4 positioning, §4.7 publishable claims) ·
`docs/DEFENCE_BRIEF.md` (CNN-backbone objection, deployment arithmetic) ·
`docs/RQ3_MATCHED_BUDGET_PLAN.md` (pre-registration record) · `progress.txt` · `step_writeups/*`.
