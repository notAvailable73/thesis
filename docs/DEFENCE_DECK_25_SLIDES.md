# B-PEFT — Thesis Defence Deck (25 slides + references)

**Official thesis title (registered — do not alter):** *Bayesian Parameter-Efficient Fine-Tuning
(B-PEFT) for Reliable Few-Shot Vision with Lightweight CNN Backbones*

**Presentation subtitle (framing, Slide 1 only):** *A controlled experimental attribution study — which
design choices govern which aspects of reliability?*

> The official title stays. The subtitle exists because the title names the **method family** while the
> contribution is the **attribution study**. If you are asked why "Bayesian" is in the title when energy
> outperforms vacuity as an OOD ranker, the answer is Slide 20 and Slide 24 — the evidential
> formulation is what gives the model an explicit uncertainty representation to *measure*, and RQ2 is
> the experiment that separates the objective from the score. See the "Bayesian" briefing box below.

**How to use this file.** One `## Slide N` block per slide. **ON THE SLIDE** is paste-ready content —
put that on the slide and nothing more. **SAY** is your spoken line; it is not on the slide.
**VISUAL** names a figure that already exists in this repo. Total speaking time ≈ 25 minutes,
leaving room for questions.

**Timing:** the 25 content slides sum to **~28.5 min** as written (Slide 21 is delivered as **21A + 21B**,
which is one numbered slide's worth of story across two physical slides). References (2 slides) and
backups B1–B17 are **not** counted in the 25 and are never shown unprompted.

**Fact-checked 2026-09-14 against `docs/guide/`** (the post-experiment project guide, files 01–10) plus
`RESULTS_MASTER.md`, `DEFENCE_BRIEF.md`, `RQ_SUPERVISOR_REPORT.md` and `refs.bib`. Every figure on every
slide traces to one of those. Where the guide's corrections list (A1–A8, C1–C15) contradicted an earlier
version of this deck, **the guide wins** and the correction is stated on the slide rather than dropped.

**If your slot is 20–25 minutes, trim in this order** — never cut Slides 2, 15, 21A/21B or 24:

| Cut | Saves | How |
|---|---:|---|
| Slide 7 + 8 → merge into one prior-work slide | ~1.0 min | Keep TSA, BEL and energy; drop Conv-Adapter and BayesAdapter rows |
| Slide 17 → drop the metrics table, keep protocol rigour | ~0.5 min | The metrics are self-evident from the results slides (and B14 holds the definitions) |
| Slide 23 → keep only the three-row table | ~0.25 min | Move the Pareto figure to B13 |
| Slide 6 → fold objectives into Slide 5 | ~0.5 min | Last resort — most departments expect objectives |

That lands at **~26 min** with all four load-bearing slides intact.

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
| **Calibration repair** | post-hoc recalibration — ECE improves in **48/48** cells (still short of softmax in 48/48); OOD ranking is preserved in **78%** of comparisons |

---

**Four rules this deck is built on** (they are what separates a deck that defends *this* thesis from
one that defends a weaker one):

1. **The centrepiece is Slide 15 (the experimental design), not a results table.** This thesis is a
   controlled attribution study, not an architecture proposal. Its accuracy was never trying to be
   state of the art. If your headline is an accuracy table, the first question is "so you're behind a
   ViT?" and you have handed them the wrong frame.
2. **Slide titles are conclusions, not labels.** "RQ3 results" is a label. "Matched budgets point to
   backbone dependence, not parameter count" is a conclusion. Committees read titles even after they
   stop listening.
3. **Volunteer every weakness before it is found.** Slides 2, 17, 24 exist for this. A negative result
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
| **RQ2** | Is OOD detection attributable to the training objective or the scoring rule? | The scoring rule — it explains **43.7%** of far-OOD variance against the objective's **0.27%** |
| **RQ3** | Does accuracy follow adapter architecture and calibration follow parameter budget? | Accuracy — yes. Calibration was **not** explained by parameter budget; a matched-budget experiment supports **backbone** dependence |
| **RQ4** | Can the evidential head's calibration be repaired after training without breaking its OOD ranking? | Partly: ECE improved in **48/48** cells and OOD ranking held in **78%** — but it stays worse-calibrated than softmax in **48/48** |

Footer: **189 training runs · 600 frozen test episodes each · 3 seeds · 0 run failures**

**SAY:** "Four research questions, four answers — including two where our initial expectation was not
supported, and I will show you both. One of them was overturned by an experiment we pre-registered
against ourselves."

**NEVER SAY:** "all four hypotheses were confirmed." They were not, and the deck is stronger for it.
Say instead: "the four research questions were answered, including cases where the initial expectation
was not supported."

**Source:** `RQ_SUPERVISOR_REPORT.md` §1. Run count = 120 main grid + 48 matched-budget + 21 rank sweep.

---

## Slide 3 — A model that is wrong and confident is worse than one that says "I don't know"

**Time:** 1.0 min · **Section: A. Problem**

**ON THE SLIDE**

- **Few-shot adaptation at the edge is a real deployment shape.** A device is handed 5 examples of a
  new class and must work immediately — on hardware with kilobytes of SRAM and often no FPU.
- **Full fine-tuning is not available there.** 25 support images against 11.7 million parameters
  overfits, and the compute does not exist on the device anyway.
- **Parameter-efficient adaptation solves the capacity problem and inherits a reliability problem.**
  Adapters preserve accuracy. What happens to the *confidence numbers* is largely unreported.

Boxed consequence:

> A system that reports 95% confidence while being 60% correct is dangerous in exactly the
> applications that motivate edge deployment: inspection, triage, screening.
> **Accuracy alone does not characterise whether such a system is fit to deploy.**

**VISUAL:** two reliability diagrams side by side —
`results/grid_plots/cifar_fs_5shot_softmax_reliability.png` and `..._evidential_reliability.png`.
Do **not** explain them yet; you explain them on Slide 19.

**SAY:** "This thesis is about the sentence in that box. Not about making few-shot models more
accurate — about finding out what happens to their *honesty* when you shrink them, and which design
choice is responsible for which part of that."

---

## Slide 4 — The deployment target chooses the backbone

**Time:** 0.75 min · **Section: B. Deployment constraint** · **Pre-empts the known pre-defence attack:
"CNNs are outdated in 2026."** Deliver it as a design justification, before anyone asks.

**ON THE SLIDE** — one message, one table:

> **The deployment regime selects the model family.** MCU-class targets run **32–512 kB SRAM**, under
> ~1 MB flash, 20–200 MHz, often with no FPU. That budget is what rules out the large-ViT options —
> before any accuracy argument is made.

| Deployed at inference | Parameters | × ours |
|---|---:|---:|
| DINOv3 ViT-7B | 7,000,000,000 | 2,800× |
| CLIP-ViT-B/16 | 86,500,000 | 34.6× |
| DINO-ViT-S/16 | 21,000,000 | 8.4× |
| ResNet-18 (ours) | 11,700,000 | 4.7× |
| **MobileNetV3-Small (ours)** | **2,500,000** | — |

One supporting line, small:

> In low-data regimes the 2025 literature reports CNNs **matching** ViTs *(arXiv:2510.04794)*, so the
> usual accuracy argument for a transformer is at its weakest in exactly this regime. **This is not a
> claim that CNNs are superior to transformers in general.**

**SAY (rehearse verbatim):** "The backbone choice here follows from the deployment regime, not from a
claim about CNN superiority. An 86-million-parameter ViT does not fit in one megabyte of flash — that
is arithmetic, not preference. And in the low-data regime, the accuracy argument for the transformer
is at its weakest."

**Detail moved to backup (B13)** — the ~180 ms vs ~8–12 ms STM32F746 attention-block comparison
*(TinyDL survey, arXiv:2506.18927, 2025)*, kept for the question "how much slower, exactly?"

**Source:** `DEFENCE_BRIEF.md` §1, §2, §3.4.

---

## Slide 5 — Problem statement, challenges and objectives

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

## Slide 6 — Background: four ideas the rest of the talk assumes

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

**Full definitions of every metric and statistic used in this deck: backup B14.**

---

## Slide 7 — Prior work I: adapters on frozen CNNs — our closest precedent, and we replicate it

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

## Slide 8 — Three papers report Bayesian calibration working. We locate where it stops working.

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
> parameters, true disjoint-class episodes — and calibration **degrades, in 20 of 20 matched pairs**.
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

## Slide 9 — What existing work covers, and the combination we did not find

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
> MCU-class CNN, a ≤31,744-parameter adapter, disjoint-class episodic few-shot, with accuracy,
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
Slide 24). `RESULTS_MASTER.md` §4.6 states it that way; say it that way.

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

**The full eight-literature coverage matrix is backup B17** — pull it up if someone wants the wider map.

---

## Slide 10 — Research questions

**Time:** 1.0 min · **Section: D. Research questions**

**ON THE SLIDE** — the overarching question again at the top (same box as Slide 2), then:

- **RQ1 — Attribution.** How is variance in accuracy, calibration and OOD detection distributed across
  dataset, shot count, backbone, adapter type and head interpretation?
  *Role: frames everything else — tells you which axis to look at for which outcome.*
- **RQ2 — Objective vs score.** Is OOD performance attributable to the **training objective** the model
  was adapted under, or to the **scoring rule** its outputs are read with — and can the two be separated?
  *Role: resolves the OOD outcome.*
- **RQ3 — Architecture vs budget.** Within the adapter axis, does accuracy follow adapter
  **architecture** while calibration follows trainable-**parameter budget**?
  *Role: resolves the adapter axis; the question this thesis pushes hardest.*
- **RQ4 — Remediation.** Can the evidential head be recalibrated **post-hoc** by refitting only two
  parameters — and does its OOD ranking survive that refit?
  *Role: turns a diagnosis into a repair.*

**SAY:** "RQ1 is diagnostic — it says which axes matter. RQ2 and RQ3 are adjudication questions: each
takes two competing explanations that the existing literature cannot separate and builds a design that
can. RQ4 is remedial. They are in that order on purpose."

---

## Slide 11 — The experimental program stayed intact; the questions were refined to match what it can establish

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
two negative results are the evidence: they are still here, reported, on this slide and on Slide 24.
If the goal had been convenience, the 0-of-20 calibration result is the first thing that would have
disappeared."

**Correction to carry into the thesis text:** `RQ_SUPERVISOR_REPORT.md` Appendix A labels Orig-RQ1
"serial versus parallel" and then reports "parallel wins 16/16". Those are two different comparisons —
the 16/16 is **parallel bottleneck vs LoRA**, and serial was never in the grid. The row above states it
correctly; fix the report before submission.

**Source:** `RQ_RESULTS_SUMMARY.md` Appendix; `RQ_SUPERVISOR_REPORT.md` Appendix A;
`docs/guide/05_problems_and_open_work.md` A3.

---

## Slide 12 — The system

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

## Slide 13 — Two adapters — and a parameter-count reversal we did not design

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

## Slide 14 — The head is one thing; how you read it is another

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

## Slide 15 — A balanced factorial, not a sequence of comparisons

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
limitation on Slide 24."

---

## Slide 16 — Two in-distribution datasets, five OOD pools

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

## Slide 17 — Metrics, protocol and why these numbers can be trusted

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
  every run in the thesis. No test set was ever re-sampled.
- **Selection on VAL only.** A separate 100-episode validation split (seeds 10000–10099). The 600 test
  seeds were never used for any selection decision. *One disclosed exception to "VAL-selected": the grid
  inherits `kl_weight_max` = 0.1, while the Step 4.5 VAL sweep ranked 0.05 first (VAL ECE 0.252 vs
  0.260). The surface is flat and no comparison is confounded, but do not call 0.1 VAL-selected.*
- **Byte-identical reproducibility.** The 48-run matched-budget experiment re-ran 18 existing grid arms
  from scratch and reproduced the committed numbers at **max absolute difference 0.0**.
- **Pre-registration.** RQ3's decision rule, thresholds and an explicit "inconclusive" outcome were
  written down **before any of the deciding runs existed** (`docs/RQ3_MATCHED_BUDGET_PLAN.md`).
- **Control guards.** Automated checks that only the intended axis differs between compared configs —
  **0 unaccounted config keys** across all 48 matched-budget runs.

Environment footer: **Kaggle / Colab T4 GPU · episodic meta-training, 100 episodes/epoch, ≤30 epochs,
early stop patience 5 · LR 5e-3 · seeds 42/43/44**

**SAY:** "The 18 re-runs reproducing to a maximum absolute difference of *zero* is the number I'd point
at if you only trust one thing on this slide. It means the second experiment measures the same quantity
the first one did — across three weeks, a different notebook and a different Kaggle session, which is
not something you can assume."

**Full metric and statistic definitions: B14. Full configuration listing: B15.**

---

## Slide 18 — 6,928 trainable parameters match full fine-tuning at 5-shot

**Time:** 1.0 min · **Section: G. Results**

**ON THE SLIDE** — CIFAR-FS, 5-way 5-shot:

| Configuration | Trainable params | Accuracy |
|---|---:|---:|
| Parallel bottleneck, ResNet-18 | 31,744 | **91.44%** |
| Parallel bottleneck, MobileNetV3-Small | **6,928** | 90.74% |
| Full fine-tuning, ResNet-18 | 11,176,512 | 90.47% |
| Linear probe (no adaptation) | 0 | 87.41% |

Boxed: **6,928 trainable parameters match the accuracy of retraining 11.18 million — 0.06% of the budget.**

> **Read the comparison correctly — this is on the slide.** Full-FT and Linear-Probe baselines were run
> **only for ResNet-18 on CIFAR-FS**. Full fine-tuning of MobileNetV3-Small was never run. So row 2 vs
> row 3 is a **cross-backbone** comparison: a 6,928-parameter adapter on the small backbone against full
> fine-tuning of the large one. The same-backbone claim is row 1 vs row 3.

**The caveat that belongs on this slide, in small print — volunteer it, do not wait to be asked:**

> **This is the 5-shot result. At 1-shot, full fine-tuning wins by 2.57 pp.** The parameter saving is
> close to free once there are 25 support images; with 5 it is not. The same asymmetry appears against
> published methods (backup B6): −1.06 pp at CIFAR-FS 5-shot, but up to −18.2 pp at MiniImageNet
> 1-shot with the small backbone.

**SAY:** "Two readings, and I want to be precise about which is which. The ResNet-18 row exceeds full
fine-tuning by 0.98 points on the same backbone, which comfortably clears seed noise. The MobileNet row
is +0.27 points, it is **inside** its own seed spread, and it is measured against full fine-tuning of a
*different, larger* backbone — we never ran full fine-tuning on MobileNetV3-Small. So the claim there is
*matches*, not *beats*, and it is a cross-backbone match. Matching full fine-tuning at 0.06% of the
parameters is the interesting result anyway."

**NEVER SAY "beats"** for the MobileNet row. `RESULTS_MASTER.md` §4.7 claim 5 is explicit, and a
careful committee member checking the intervals will catch it.

---

## Slide 19 — Shots drive accuracy; the head drives calibration

**Time:** 1.5 min

**ON THE SLIDE** — make these three findings visually dominant, one line each, largest type on the slide:

> ### Accuracy → **shot count** dominates (76% of variance)
> ### Calibration → **head type** dominates (83% of variance)
> ### OOD detection → **no single dominant axis**; head, shots and adapter all contribute

**How to read η², one sentence, on the slide:**

> **η² is the share of the total variation in an outcome that is attributable to one design axis.**
> 76% for shots on accuracy means: of everything that made accuracy differ across the grid, about
> three-quarters of it is explained by whether the episode had 1 or 5 shots.

Then the evidence, kept on the slide in smaller type — η² decomposition, main effects over the
balanced factorial, **96 per-seed observations** (**95** for near-OOD AUROC: one per-seed value is
missing — `cifar_fs/5shot/mobilenetv3_small/lora/evidential`, seed 42, TinyImageNet):

| Outcome | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.75% | **76.05%** | 3.92% | 9.14% | 0.19% | 8.95% |
| ECE | 2.02% | 2.13% | 4.63% | 0.63% | **82.89%** | 7.70% |
| OOD AUROC (far) | 1.18% | 22.68% | 11.54% | 1.73% | **39.01%** | 23.87% |
| OOD AUROC (near) | 1.64% | **42.60%** | 0.61% | 21.98% | 20.98% | 11.42% |

One-line consequence, boxed:

> **Accuracy and calibration are governed by near-disjoint axes** — the head explains 83% of calibration
> variance and 0.19% of accuracy variance. You can choose the head for calibration without paying for
> it in accuracy.

**VISUAL:** the two reliability diagrams from Slide 3 — now explained.

**SAY:** "Read the two bold cells. 83% of calibration variance and 0.2% of accuracy variance sit on the
same axis — that is close to a clean separation of concerns. Note also the OOD rows: no single axis
dominates there, which is exactly what motivates RQ2 on the next slide."

**Volunteer:** "The qualitative pattern — that calibration and accuracy have different drivers — goes
back to Guo et al. 2017. We do not claim to have discovered it; what is ours is the formal
decomposition, in this regime."

**Moved to backup B16 — the ECE/OOD orthogonality analysis:** across all 40 cells the rank correlation
between ECE and OOD AUROC is *positive* (ρ = +0.433 far, +0.477 near), which naively reads as
"worse-calibrated configurations detect OOD better." Stratifying by head collapses it to ρ =
+0.15/+0.20 (evidential) and +0.26/+0.24 (softmax) — i.e. **once the head is fixed, calibration carries
little information about OOD detection.** *We did not find prior work reporting this under this
experimental regime.* Pull B16 up if asked whether calibration and OOD trade off against each other.

---

## Slide 20 — For OOD detection, the score you read matters far more than the loss you trained with

**Time:** 1.5 min

**ON THE SLIDE** — first the confound, in one line:

> Every published comparison we found scores evidential models with **vacuity** and softmax models with
> **MSP**. The objective and the score always move together. We compute **all four scores on all runs**:
> a clean 2 objectives × 4 scores factorial, **n = 792 comparisons per pool**.

Then the headline, in large type, as two lines — **the two shares, not their ratio**:

> ### Far-OOD variance explained by the **scoring rule**: **43.7%**
> ### Far-OOD variance explained by the **training objective**: **0.27%**

The full result, smaller:

| Pool | Scoring rule (η²) | Training objective (η²) |
|---|---:|---:|
| Far-OOD | **43.7%** | 0.27% |
| Near-OOD | **13.0%** | 0.60% |

> Two to three orders of magnitude, on both pools.

The concrete version — the same score gives near-identical AUROC whichever objective produced the logits:

| Score | On evidential-trained logits | On softmax-trained logits |
|---|---:|---:|
| **Energy** | 0.911 | 0.929 |
| MSP | ~0.79 | ~0.79 |

What this does and does not say, boxed — **put this on the slide**:

> **Says:** in this experiment, the scoring rule explains substantially more of the variation in OOD
> performance than the training objective does.
> **Does not say:** that evidential training is useless. The evidential head is what supplies an
> explicit uncertainty representation at all, it remains the better *native probabilistic* score
> (37–38 of 40 cells vs MSP, backup B4), and its inference cost is below our noise floor (Slide 23).

Small print on the slide — **volunteer both of these:**

> **We quote the two shares, not their ratio.** Earlier drafts reported the ratio as "163×". Because the
> objective's share is so close to zero, that quotient is unstable: recomputing it without the one cell
> that is missing an arm (`cifar_fs/5shot/mobilenetv3_small/lora`) moves it to 394× far-OOD and 19×
> near-OOD. The shares are stable and the conclusion is unchanged, so the shares are what we report.
>
> **Coverage.** Computed on 99 of 120 recoverable checkpoints; a regression guard confirms 99/99 cells
> unchanged against the committed metrics. **The 21 missing checkpoints are not a random sample** — all
> 21 are CIFAR-FS 5-shot adapter models, leaving that slice with 15 of 36. It is also the slice this
> thesis quotes most often elsewhere. Stated as limitation 8 on Slide 24.

**SAY:** "Forty-three point seven percent against zero point two seven — two to three orders of
magnitude, and the same gap holds on near-OOD. The same score gives near-identical AUROC regardless of
which objective produced the logits. So in this regime the Bayesian *training* was not what bought the
OOD performance; the *score* was. And there is a 2026 theory result showing softmax is a mathematical
special case of an evidential classifier, which pre-explains why the objective would matter so little.
I would rather cite that myself than have it raised from the floor."

**Volunteer:** "Concurrent WACV 2026 work runs a comparable objective × score ANOVA at larger scale, so
the technique itself is not new. What we did not find prior work doing is cross-applying **energy onto
Dirichlet-parameterised logits** and finding near-equivalence. That is the surviving contribution."

---

## Slide 21A — Architecture and parameter budget were initially confounded

**Time:** 1.0 min · **Slide 21 is the most important in the deck and is delivered as two slides. This
is beat one: the pattern, and why it cannot be read at face value.**

**ON THE SLIDE** — the pattern first, from the 16 matched comparisons in the main grid:

| Outcome | Winner is the **bottleneck architecture** | Winner is the **larger-budget arm** |
|---|---:|---:|
| Accuracy | **16/16** | 8/16 |
| Near-OOD AUROC | **16/16** | 8/16 |
| ECE | 8/16 | **16/16** |

Sign consistency 16/16, two-sided p ≈ 3.05×10⁻⁵ under a null of random direction.

Two observed facts, stated so they cannot be overread:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both
   backbones — holding 2.58× *more* parameters on one and 1.55× *fewer* on the other.
2. **The ECE winner does change, exactly in step with the budget ordering.**

Then the reversal that causes the problem — repeat the Slide 13 numbers here:

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

## Slide 21B — Matched budgets reject parameter count as the primary explanation for calibration

**Time:** 1.5 min · **Beat two: the pre-registered experiment that adjudicates it. This is your best
45 seconds in the defence — rehearse it.**

**ON THE SLIDE**

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

And the secondary outcome that was genuinely at risk: at matched budget, **bottleneck still wins
accuracy 8/8 and near-OOD 8/8, all beyond 2σ** — the architecture effect does not collapse when the
budget advantage is removed.

> **The non-finding, reported rather than omitted:** on **far**-OOD there is no clear adapter effect —
> bottleneck wins only **11/16** unmatched and **5/8** matched. The architecture claim is scoped to
> accuracy and **near**-OOD. RQ1 attributes far-OOD predominantly to the head, not the adapter.

Boxed — the claim, scoped exactly:

> **The controlled matched-budget experiment provides evidence against parameter budget as the primary
> explanation for the calibration gap.** Equalising the budget leaves ResNet-18's gap entirely intact
> and only halves MobileNetV3-Small's. The results are consistent with a backbone-dependent mechanism —
> **but which intrinsic backbone property is responsible remains unidentified.**

**SAY (rehearse this):** "Equalising the budget leaves ResNet-18's calibration gap intact — collapse
ratios of 1.01 and 0.92, meaning the matched gap is as large as the unmatched one. A gap caused by the
budget difference should have shrunk toward zero with it. It did not. So the hypothesis this project
had been working under for two months is not supported; we pre-registered the rule that would tell us
so before we ran it; and the answer we are left with is narrower than what we had been ready to claim."

**State the limit before you are asked:** "'Backbone-intrinsic' is where the evidence points, not an
explanation. Which property of ResNet-18 versus MobileNetV3-Small — depth, width, normalisation, the
inverted-residual block, feature-norm scale at the adapter sites — produces the reversal is untested,
and two backbones cannot separate those candidates. That is the honest remaining limit of RQ3, and
closing it takes more backbones, not more seeds."

**ABSOLUTELY DO NOT SAY:** "calibration follows the parameter budget" (superseded), or "we proved the
backbone causes calibration" (the design does not identify the mechanism).

---

## Slide 22 — Post-hoc recalibration repairs calibration without retraining

**Time:** 1.5 min

**ON THE SLIDE**

**What was refit:** only the **two** parameters of the evidence affine (scale, bias), on the
**validation** episode split, evaluated on the frozen 600-episode test split. **No retraining.**

Three headline numbers, equal weight — **the third is not optional**:

> ### Calibration: ECE improved in **48 / 48 cells (100%)** — mean **−0.137** absolute (0.327 → 0.190)
> ### OOD ranking: preserved in **150 / 192 comparisons (78%)** — **22% were not**
> ### Still short of softmax: **48 / 48** cells remain worse-calibrated than plain softmax

Supporting detail:

| Outcome | Result |
|---|---|
| Where training had left the affine | `(scale, bias)` is **learnable**, initialised at (2, −6); training moved it to scale **1.51–4.52** (median 2.93) |
| Where the refit puts it | scale **3.6–14.5** (median 8.3) — the jointly-trained operating point was far from the calibration-optimal one |
| Residual gap after refit | worse than plain softmax in **48/48** (mean **2.18×**, best case 1.07×); worse than TS-softmax in **48/48** (mean **11.9×**, best 2.27×) |
| Preservation criterion | ΔAUROC ≥ −0.005; mean ΔAUROC across comparisons **+0.004** |
| Worst-case rank correlation | Spearman ρ = **0.921** |

Caveat — **on the slide, not just spoken**:

> **Reordering does occur.** Vacuity is K/Σα, a function of *all* logits jointly, so a per-logit
> monotone transform does not guarantee the sample ordering survives. In **22%** of comparisons AUROC
> dropped by more than half a point; worst single-pool drop ≈ **−0.03**. The claim is **"OOD ranking is
> preserved in 78% of comparisons"** — never "always."
>
> **And the repair is partial.** Post-refit evidential ECE is still worse than plain softmax in **all
> 48** cells. The correct claim is **"refitting improves calibration in 48/48 cells but does not close
> the gap to softmax"** — never "refitting fixes evidential calibration."

**SAY:** "Two parameters, refitted on validation episodes, no retraining — and ECE falls in every
single cell. Note what that says: these two scalars are *trained* jointly with the adapter, so the
evidential loss was leaving them far from the calibration-optimal point on its own. That is a sharper
finding than a badly-chosen default would have been. Two things I want to be exact about, though. The
refit **narrows** the gap to softmax, it does not close it — evidential is still the worse-calibrated
arm in all forty-eight cells. And we *measured* whether the OOD ranking survived rather than assuming
it, because in twenty-two percent of comparisons it did not."

**Volunteer:** "Post-hoc calibration is a mature field, and Guo et al. and BEL report comparable or
larger drops. We are not claiming to have discovered that refitting helps. What is ours is the
mechanism — this specific two-parameter evidence affine, which turns out to be the knob that actually
controls calibration here — and the systematic quantification across 48 cells and 192 ranking
comparisons, including the failures."

**The self-correction that belongs with this slide:** Step 4.5 swept the *loss* (KL weight × variance
term) and found a flat calibration surface, and we concluded calibration could not be tuned. That search
was tuning the wrong knob — the evidence affine was never swept. RQ4 is what found it.

---

## Slide 23 — The frozen trunk dominates inference cost; the adapter and head barely move it

**Time:** 0.75 min · **Section: H. Practical implications**

**ON THE SLIDE** — measured on real hardware (Kaggle T4, GPU and single-thread CPU). Three numbers, no more:

| Design choice | Effect on latency |
|---|---:|
| **Backbone** | **5.12×** (ResNet-18 62.18 ms vs MobileNetV3-S 12.14 ms, matched adapter) |
| Adapter | **3.9–5.8%**, despite up to 2.6× parameter differences |
| Evidential head | **1.29%** mean — below the session's own **5.91%** noise floor |

Boxed, the deployment rule:

> **Choose the backbone for latency and the adapter for accuracy** — they barely trade against each
> other. **Recommended point:** MobileNetV3-Small + parallel bottleneck + evidential —
> **11.86 ms/image**, **6,930 trainable parameters**, TinyImageNet near-OOD AUROC **0.870** (1-shot) /
> **0.919** (5-shot).

**SAY:** "The practical rule is the boxed line: backbone for latency, adapter for accuracy, because
they barely trade against each other. And 'evidential uncertainty is free at inference' is now a
*measurement below our own noise floor* rather than a claim — Sensoy's 2018 paper asserts it as EDL's
selling point; we did not find it measured in this regime."

**Volunteer if pressed:** "The Pareto claim is native-score-conditional. If softmax is allowed its best
score — energy — rather than max-softmax-probability, evidential's presence on the CIFAR-FS 5-shot
frontier goes to zero. That is in the write-up, and on backup B13."

**Moved to backup B13** — the MiniImageNet deployment point (ResNet-18 + parallel + evidential at
62.38 ms, where MobileNetV3-Small falls outside the accuracy tolerance), the full efficiency table,
the MCU attention-block comparison from Slide 4, and both Pareto figures
(`results/pareto_latency_vs_auroc__cifar_fs.png`, `__mini_imagenet.png`).

---

## Slide 24 — What the experiments changed our mind about, and what they leave unresolved

**Time:** 1.5 min · **Section: I. Limitations and self-corrections** · **This slide is worth more than
any positive result on it. Committees remember it. Deliver it as boundary-setting, not as apology.**

**ON THE SLIDE** — two clearly separated sections, the left one given more visual weight:

### Section 1 — What the experiments changed our mind about

1. **Energy vs vacuity: a single-configuration finding did not generalise.** An early result said
   evidential vacuity was roughly on par with the energy score. At grid scale, **energy wins ~70% of
   comparisons** (vacuity wins only 10/40 far-OOD, 14/40 near-OOD). *Narrowed claim: vacuity is a
   substantially better OOD ranker than softmax-probability scores — 37–38 of 40 cells — but a
   well-chosen logit-space score was the stronger ranker in this study.*
2. **The parameter-budget hypothesis: a pre-registered experiment did not support our own preferred
   reading.** RQ3's "calibration follows the parameter budget" account fired in **0 of 4** cells once
   budget was equalised (Slide 21B).
3. **The interior-optimum hypothesis was tested and demoted.** We predicted calibration error would
   reach an optimum at an intermediate parameter budget. A controlled **21-run rank sweep** with
   everything else fixed found **no interior optimum in the tested range** — evidential ECE is lowest
   at rank 1 and drifts up; softmax moves the opposite way. Reported as a negative result.
4. **A silent latency-selection bug, caught by manual cross-check rather than by the test suite.**
   Three independent latency-selection functions were silently preferring this repo's noisy dev-laptop
   measurements over the canonical Kaggle CPU numbers — by dict-insertion-order accident, with
   **errors up to 47% on individual cells**, no crash and no failing test. Every downstream artefact
   (efficiency table, Pareto JSON, all six Pareto figures) was regenerated from corrected code. *A
   regression test for this bug class is the top open follow-up.*

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
5. **Main effects only** — the η² decomposition does not model interactions.
6. **Temperature-scaling asymmetry** — evidential is not temperature-scalable in this codebase, so the
   TS-softmax comparison is structurally favourable to softmax. Still the right comparison; the
   asymmetry is stated rather than hidden.
7. **Literature-search limitations** — novelty claims rest on a non-exhaustive search: **five**
   independent passes, ~100 searches, full PDF reads of the closest competitors
   (`RQ_SUPERVISOR_REPORT.md` Appendix C). Absence of found prior art is not evidence of absence.
8. **RQ2 and RQ4 cover 99 of 120 models, non-randomly** — all 21 missing checkpoints are CIFAR-FS 5-shot
   adapter models, leaving that slice with 15 of 36. Retraining them is a costed, unrun follow-up (~7
   GPU-hours).
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
machine they were supposed to come from. On the right, the boxed question at the bottom is the one I
would spend the next six months on."

**Prepare for "doesn't the energy result undermine your premise?"** — "It narrows it. The premise was
never that Bayesian uncertainty is the best possible OOD score. RQ2 is exactly the experiment that
separates those two claims, and its answer is that the score explains far more of the variation. What
survives is that the evidential head is the better *native probabilistic* option, that it is free at
inference within our noise floor, and that its calibration deficit is repairable post-hoc. If you only
need an OOD ranker, the honest recommendation is softmax plus energy."

---

## Slide 25 — Four findings: different design choices govern different aspects of reliability

**Time:** 1.5 min · **Section: J. Final contribution**

**ON THE SLIDE** — the four findings, large, as the visual core of the slide:

> ### Accuracy → **shots + adapter architecture**
> ### Calibration → **head + backbone**
> ### OOD detection → **scoring rule**
> ### Repair → **post-hoc recalibration improves ECE in 48/48 cells; OOD ranking survives in 78%**
> ### …and the gap to softmax **narrows without closing** — evidential is still worse in 48/48

Then, smaller — **contributions, ordered by strength, labelled honestly:**

1. **[NOVEL] A dissociation between adapter architecture and adapter parameter budget, by outcome
   metric.** Accuracy and near-OOD ranking follow **architecture**, invariant to budget, tested at
   matched budget. Calibration is not explained by budget; the evidence supports **backbone**
   dependence. Established by a pre-registered experiment that did not support the project's own
   working hypothesis.
2. **[PARTLY NOVEL] Separating the OOD training objective from the OOD scoring rule** — the score
   explains **43.7%** of far-OOD variance against the objective's **0.27%**. Specifically new:
   cross-applying the **energy score onto Dirichlet-parameterised logits** and finding near-equivalence.
3. **[PARTLY NOVEL] A formal variance decomposition** of accuracy, calibration and OOD detection over a
   balanced five-axis PEFT factorial — including that **ECE and OOD AUROC become close to orthogonal
   once head interpretation is controlled.**
4. **[PARTLY NOVEL] A two-parameter post-hoc recalibration of an evidential prototype head**, with
   OOD-ranking preservation **measured rather than assumed** (48/48 improved, 78% preserved, and the
   residual gap to softmax reported rather than dropped).
5. **[CONFIRMATORY, and useful] A reproducible harness and evidence base** — 189 runs, frozen episode
   seeds, byte-identical reruns, VAL-only selection, and a working demonstrator.

**Future work** (each one closes a numbered limitation):

1. **More backbones** — the only way to identify *which* backbone property is associated with the
   calibration reversal *(closes limitation 4)*.
2. **A from-scratch control run** — removes the pretraining-overlap objection entirely *(closes 1)*.
3. **A transformer arm under our own protocol** — converts "CNNs are appropriate here" from an argument
   into a measurement.
4. **Per-cell VAL tuning** — answers whether any axis effect is a recipe artefact *(closes 2)*.
5. **An evidential analogue of temperature scaling**, so both heads get a post-hoc correction *(closes 6)*.

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
(arXiv:2506.18927); CNN-vs-ViT low-data study (arXiv:2510.04794); DINOv3 (arXiv:2508.10104); PEFT-for-ViTs
survey (arXiv:2402.02242 — the documented calibration/OOD absence cited on B17).

> **Every citation above resolves to an entry in `docs/refs.bib`.** Two arXiv IDs that appeared in earlier
> drafts of this deck (GP-Adapter, Dual-Adapter) had no entry in `refs.bib`, `CITATION_AUDIT.md` or any
> report, and have been removed rather than cited unverified. One flagged entry —
> *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (arXiv:2410.21228) — still needs re-checking
> before it is used anywhere.

> Full BibTeX: `docs/refs.bib`. Citation provenance: `docs/CITATION_AUDIT.md`.

---

# Backup slides (B1–B17) — after references, shown only when asked

Pulling up a prepared slide instead of talking from memory is the single most effective thing you can
do in a defence Q&A. **B13–B15 are new in this revision**, holding detail moved off Slides 4, 9, 19
and 23 — the main deck is lighter, the evidence base is not.

| # | Slide | Triggered by |
|---|---|---|
| **B1** | Accuracy, all 40 configurations — `results/mvt_table_accuracy.png` | "What were the actual numbers?" |
| **B2** | Calibration, all 40 — `results/mvt_table_calibration.png` | "How bad is the calibration really?" |
| **B3** | OOD AUROC + FPR@95 — `results/mvt_table_ood_auroc.png` | "Which OOD set, which score?" |
| **B4** | Vacuity vs every softmax-side score | "How does evidential compare to energy?" |
| **B5** | Positioning vs published few-shot methods | "How do you compare to state of the art?" |
| **B6** | What the parameter saving costs | "What does the saving cost you?" |
| **B7** | PEFT budgets on VTAB-1k (with protocol warning) | "How does this compare to VPT/SSF?" |
| **B8** | The 16 RQ3 pairs in full | "Show me all sixteen comparisons" |
| **B9** | Matched-budget design: ranks, budgets, mismatches | "How exactly did you match budgets?" |
| **B10** | The rank sweep — `results/rq5_rank_sweep.png` | "Did you try varying the rank?" |
| **B11** | Reliability diagrams + OOD histograms — `results/grid_plots/*` | "Show me the calibration visually" |
| **B12** | The demonstrator (`app/` — *Sentinel*) | "Does any of this run on anything real?" |
| **B13** | **Deployment detail: full efficiency table, both Pareto figures, MCU arithmetic** | "Where do the latency numbers come from?" / "What about MiniImageNet?" |
| **B14** | **Metrics and statistical interpretation** | "What exactly does your 43.7% η² mean?" / "Why no ratio?" |
| **B15** | **Experimental configuration at a glance** | "Remind me what you actually ran." |
| **B16** | **ECE / OOD orthogonality analysis** | "Do calibration and OOD detection trade off?" |
| **B17** | Eight-literature coverage matrix | "What does the wider literature cover?" |

---

## B13 — Deployment detail (moved off Slides 4 and 23)

**The MCU arithmetic behind Slide 4.** A memory-optimised transformer attention block costs **~180 ms**
on an STM32F746, against **~8–12 ms** for CNN inference *(TinyDL survey, arXiv:2506.18927, 2025)*.
MCU-class targets: **32–512 kB SRAM**, under ~1 MB flash, 20–200 MHz, often no FPU.

**The second deployment point.** On MiniImageNet, where MobileNetV3-Small falls outside the accuracy
tolerance, the recommended point moves to **ResNet-18 + parallel + evidential at 62.38 ms**.

**Native-score conditionality — volunteer this if the Pareto figure is on screen.** The Pareto claim is
conditional on each head using its native score. If softmax is allowed its best score — **energy** —
rather than max-softmax-probability, evidential's presence on the CIFAR-FS 5-shot frontier goes to
**zero**.

**Provenance warning to state if asked.** Latency is the only family of numbers in this thesis that is
not byte-reproducible by design, because it is hardware- and session-dependent. All reported values
come from **one canonical Kaggle T4 session** (GPU and single-thread CPU). See self-correction 4 on
Slide 24 for the selection bug that was found and fixed here.

**VISUAL:** `results/pareto_latency_vs_auroc__cifar_fs.png`, `results/pareto_latency_vs_auroc__mini_imagenet.png`,
plus the full efficiency table (latency, MACs, peak memory, trainable params per configuration).

---

## B14 — Metrics and statistical interpretation

**Purpose:** a defence-ready reference, not a statistics lecture. Every definition below is the one
used to produce the numbers in this deck.

**Outcome metrics**

| Term | Definition as used here | Direction |
|---|---|:-:|
| **ECE** — Expected Calibration Error | Predictions are binned by confidence (**15 bins, pooled over episodes**); ECE is the weighted mean absolute gap between each bin's mean confidence and its accuracy. A perfectly calibrated model that says 90% is right 90% of the time. | ↓ |
| **Brier** | A proper scoring rule combining accuracy and calibration in one number. | ↓ |
| **AUROC** (OOD) | Score every input by uncertainty; AUROC is the probability that a randomly drawn OOD input is scored more uncertain than a randomly drawn in-distribution input. 0.5 = chance, 1.0 = perfect. | ↑ |
| **FPR@95%TPR** | At the threshold that retains 95% of in-distribution inputs, the fraction of OOD inputs wrongly accepted — the usable-operating-point view of the same ranking. | ↓ |
| Accuracy / macro-F1 | Episode-mean query accuracy; macro-F1 exposes per-class recall collapse that accuracy can hide. | ↑ |

**The four uncertainty scores — all computed on all runs**

| Score | Definition | Needs |
|---|---|---|
| **MSP** | Maximum softmax probability, max softmax(z). The standard baseline confidence score. | nothing |
| **TS-MSP** | MSP after temperature scaling — one scalar T fitted on validation, logits divided by T. | 1 fitted scalar |
| **Energy** | A logit-space score, −logΣexp(z) (Liu et al. 2020). No Bayesian machinery, no training change, no extra parameters. | nothing |
| **Vacuity** | The evidential "I don't know" mass, **K/S** with K = 5 and S = Σα, α = softplus(z·s + b) + 1. | the evidential head's 2 parameters |

**η² — the statistic Slides 19 and 20 rest on**

> **η² (eta-squared) is the share of the total variation in an outcome that is attributable to one
> design axis.** It is computed as a **main-effects** decomposition over the **balanced factorial**, on
> **96 per-seed observations** (not seed-averaged — averaging seeds first can manufacture apparent
> cleanliness), with the **residual reported** so the unexplained share is visible.

- "Shots explain 76.05% of accuracy variance" means: of everything that made accuracy differ across the
  grid, roughly three-quarters is attributable to whether the episode had 1 or 5 shots.
- Slide 20's "43.7% vs 0.27%" are two such shares in a **2 objectives × 4 scores** factorial,
  **n = 792 comparisons per pool**. **Report the shares, never their quotient:** the objective's share is
  near zero, so the ratio is unstable (163× on the full set, 394× with the one half-missing cell removed).
  Say "two to three orders of magnitude" if a single phrase is needed.
- **Stated limitation (Slide 24, item 5):** this is a main-effects decomposition — it does not model
  interactions between axes.
- **Known tooling defect, disclose if asked:** `eta_squared()` in `scripts/rq_aggregate.py` reports a
  design as "balanced" by comparing counts of the cells that exist, so it does not notice a wholly
  missing cell. That is why near-OOD η² rests on 95 of 96 observations, and it is the same gap that makes
  the RQ2 ratio unstable. The shares themselves are unaffected in size.

**The comparison used for RQ3**

- **Sign test over matched pairs.** On the main grid: 16 matched comparisons, sign consistency
  **16/16**, two-sided **p ≈ 3.05×10⁻⁵** under a null of random direction. *Known objection, and a fair
  one: pairs sharing a backbone are not fully independent — see the Q&A crib.*
- **ΔECE = LoRA − bottleneck.** Positive ⇒ the bottleneck arm is better calibrated.
- **Collapse ratio = matched ΔECE ÷ unmatched ΔECE.** A gap caused by the budget difference should
  collapse toward 0 once budgets are equalised. Observed: **1.01, 0.92** (ResNet-18) and **0.50, 0.61**
  (MobileNetV3-Small).
- **2σ screen.** Effects are reported as robust when the margin exceeds twice the pooled across-seed
  standard deviation. All 8/8 matched-budget accuracy and 8/8 near-OOD results clear it.
- **Pre-registration.** Hypotheses, decision thresholds and an explicit "inconclusive" outcome were
  fixed in `docs/RQ3_MATCHED_BUDGET_PLAN.md` before any deciding run existed. Verdict
  `backbone_intrinsic` fired in 3 of 4 cells against a pre-registered threshold of 3; the budget
  hypothesis fired in 0 of 4.

**Calibration-improvement convention (RQ4)**

- **ΔECE improvement** = ECE(frozen default) − ECE(refit). Positive = better. Mean **−0.137 absolute**
  change in ECE across 48/48 cells.
- **OOD-ranking preservation criterion:** ΔAUROC ≥ **−0.005** counts as preserved. **150/192 (78%)**
  met it; mean ΔAUROC **+0.004**; worst-case Spearman ρ **0.921**; worst single-pool drop ≈ **−0.03**.

**Sample sizes**

**3 seeds** (42 / 43 / 44) · **600 frozen test episodes** per run · **100 validation episodes**
(seeds 10000–10099) for every selection decision · **189 runs** total (120 grid + 48 matched-budget +
21 rank sweep).

---

## B15 — Experimental configuration at a glance

| Axis | Levels |
|---|---|
| **Datasets** | CIFAR-FS (Bertinetto split) · MiniImageNet (Ravi & Larochelle split) |
| **Episode** | **5-way**, **1-shot / 5-shot**, test classes disjoint from training classes |
| **Backbones** | ResNet-18 (11.7 M, frozen) · MobileNetV3-Small (2.5 M, frozen) |
| **Adapters** | Parallel bottleneck · LoRA (+ Full-FT and Linear-Probe baselines, **ResNet-18 + CIFAR-FS only**) |
| **Heads** | Prototype head, read as **softmax** or **evidential** (Dirichlet) |
| **Seeds** | **42 / 43 / 44** — the seed changes only the adapter's initialisation, **not** the episode order; Full-FT and Linear-Probe therefore have zero seed spread |
| **Test protocol** | **600 fixed test episodes**, seeds version-controlled, identical across every run |
| **Validation** | 100 episodes, seeds 10000–10099 — **all selection decisions, never the test seeds** |
| **Training** | Episodic meta-training · 100 episodes/epoch · **≤30 epochs** · **early stopping patience 5** · **learning rate 5e-3**, no weight decay (**Full-FT exception: LR 1e-5, wd 1e-4**) · Adam · adapter **rank 16** throughout the grid |
| **Episode shape** | 5 classes × (1 or 5) support images + **15 query images per class** (75/episode) → 45,000 pooled query predictions per run |
| **Evidential loss** | Sensoy et al. 2018 squared-error + KL; KL weight ramps 0 → **0.1** over the first 1,000 episodes; Sensoy's variance term dropped (`use_variance: false`, an R-EDL relaxation) |
| **Prototype metric** | **cosine similarity × 10** (`metric: cosine`, `cosine_scale: 10`) — *not* the `l2` default still sitting in `configs/base.yaml` |
| **Augmentation** | **None**, in any cell |
| **Backbone mode** | Kept in eval mode; BatchNorm statistics never update |
| **Trainable budget** | 6,928 – 31,746 parameters (backbone never updated) |
| **Input** | All images resized to **224×224**, ImageNet-normalised (backbones are ImageNet-pretrained) |
| **OOD pools** | Far: SVHN, Gaussian noise · Near: CIFAR-100-heldout, MiniImageNet-heldout (**both = the 16 VAL classes**), TinyImageNet (**no selection overlap**) |
| **Hardware** | Kaggle / Colab **T4 GPU**; latency also measured single-thread CPU |
| **Scale** | 40 configurations × 3 seeds = **120 grid runs**; + **48** matched-budget; + **21** rank sweep = **189** |

---

## B16 — ECE and OOD AUROC are close to orthogonal once the head is controlled

Across all 40 cells the rank correlation between ECE and OOD AUROC is **positive** — ρ = **+0.433**
(far), **+0.477** (near) — which naively reads as "worse-calibrated configurations detect OOD better."

**That is a head effect.** Stratifying by head interpretation collapses it:

| Stratum | ρ (far-OOD) | ρ (near-OOD) |
|---|---:|---:|
| All 40 cells pooled | +0.433 | +0.477 |
| Evidential only | +0.15 | +0.20 |
| Softmax only | +0.26 | +0.24 |

> **Once the head is fixed, calibration carries little information about OOD detection** — they are
> close to orthogonal outcomes rather than a trade-off. **We did not find prior work reporting this
> under this experimental regime.**

This is only visible because accuracy, calibration and OOD are measured **on the same runs** — which is
Objective 2 from Slide 5.

---

## B17 — The wider literature-coverage map

The eight-literature matrix, kept for the question "what does the broader field cover?" The scoped
claim on Slide 9 is the one to defend; this table is context, and the crosses describe what these
literatures **foreground**, not an audit of every table of every paper.

| Literature | Acc | F1 | ECE | OOD | Params | Episodic | Edge |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Classical few-shot (ProtoNet, MAML, MetaOptNet) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot (P>M>F, CoOp, Tip-Adapter) | ✅ | ❌ | ❌ | ❌ | ~ | ~ | ❌ |
| PEFT for ViTs (VPT, SSF, AdaptFormer) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge (Conv-Adapter, LoRA-Edge) | ✅ | ❌ | ❌ | ❌ | ✅ | ~ | ✅ |
| Frozen-CNN episodic adapters (TSA, FiT) | ✅ | ❌ | ❌ | ❌ | ✅ | ~ | ~ |
| Bayesian PEFT (Laplace-LoRA, BLoB, BayesAdapter) | ✅ | ❌ | ✅ | ~ | ✅ | ❌ | ❌ |
| Evidential few-shot (BEL) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ~ |
| TinyML / TinyDL (TinyDL survey) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **B-PEFT (this thesis)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**SAY, if this is on screen:** "Two of those crosses are documented absences rather than our inference.
The survey covering the entire ViT PEFT field — arXiv 2402.02242 — does not discuss calibration,
uncertainty or OOD anywhere. The 2025 TinyML survey, 2506.18927, covers quantisation in depth and does
not mention uncertainty. Two communities, two surveys, the same blind spot."

**Required scoping — do not drop it:** this is about what these papers foreground, under a
non-exhaustive search (limitation 7). `RESULTS_MASTER.md` §4.6 states it that way.

---

## Numbers to have on the remaining backup slides

**B2** — evidential ECE is worse than plain softmax in **20/20** matched pairs (1.4×–9.1×) and worse
than temperature-scaled softmax by **5.3×–51×**. Pair it immediately with Slide 22 — but pair it
*accurately*: the post-hoc refit **narrows** that gap in 48/48 cells and **does not close it** (still
worse than plain softmax in 48/48, mean 2.18×; worse than TS-softmax in 48/48, mean 11.9×). **Note the
stated asymmetry (limitation 6):** evidential is not temperature-scalable in this codebase, so the
TS-softmax comparison is structurally favourable to softmax.

**B4** —

| Comparison | Far-OOD | Near-OOD |
|---|---:|---:|
| vacuity vs MSP | 38/40 (+0.111) | 37/40 (+0.053) |
| vacuity vs TS-MSP | 38/40 (+0.127) | 38/40 (+0.067) |
| vacuity vs **energy** | 10/40 (−0.022) | 14/40 (−0.007) |

Plus: the near-OOD advantage over MSP is **+0.064 at 1-shot vs +0.043 at 5-shot** — larger where there
is less data, the direction the original hypothesis predicted. Say "held between the two shot levels
tested," not "a trend."

**Label the population when you quote energy-vs-vacuity — there are two, and they differ.**

| Population | Scope | Energy beats vacuity |
|---|---|---:|
| 40 grid cells (native scores) | the table above | far 30/40, near 26/40 → **~70%** |
| 198 matched comparisons (RQ2, 99 models, all 4 scores) | Slide 20's factorial | far **162/198** (82%), near **183/198** (92%) |

Both are correct; they count different things. Slide 24's "~70%" is the grid figure. If an examiner
quotes the higher number back at you, this is why.

**B5** — same 5-way episodic protocol, so accuracy *is* comparable (subject to the pretraining caveat,
which applies to both sides):

| Method | Backbone | Trainable | CIFAR-FS 5-shot | MiniIN 5-shot |
|---|---|---:|---:|---:|
| Sup-21k > ProtoNet | ViT-B/16 | ~85.8 M | **96.7** | **99.2** |
| DINO > ProtoNet | ViT-S/16 | ~21 M | 92.5 | 98.0 |
| DINO > ProtoNet | ResNet-50 | ~25 M | — | 92.0 |
| BEL (evidential few-shot) | ResNet-12 | n/r | 86.92 | 79.60 |
| **Ours**, parallel bottleneck | ResNet-18 (frozen) | **31,744** | 91.44 | 95.56 |
| **Ours**, parallel bottleneck | MobileNetV3-S (frozen) | **6,928** | 90.74 | 90.10 |

**B6 — show this before you are asked.** Three separate conclusions; do not blur them:

| Ours vs | Param saving | CIFAR-FS 5-shot | MiniIN 5-shot | MiniIN 1-shot |
|---|---:|---:|---:|---:|
| ResNet-18 adapter vs DINO>PN ViT-S | **662× fewer** | **−1.06 pp** | −2.44 pp | −8.07 pp |
| ResNet-18 adapter vs DINO>PN ViT-B | 2,703× fewer | −0.76 pp | −2.84 pp | −10.27 pp |
| ResNet-18 adapter vs DINO>PN **ResNet-50** | 788× fewer | — | **+3.56 pp** | **+5.83 pp** |
| MobileNetV3-S adapter vs DINO>PN ViT-S | 3,031× fewer | −1.76 pp | −7.90 pp | **−18.18 pp** |

1. At 5-shot on CIFAR-FS the saving is close to free — 1.06 points at 662× fewer parameters.
2. Backbone-family-matched, we are ahead: **+3.56 points** over DINO>ProtoNet on ResNet-50. **The
   remaining gap is a ViT gap, not a parameter-efficiency gap.**
3. The trade is genuinely bad at 1-shot with the small backbone — up to **−18.2 points**. Volunteer
   this; do not let it be found.

**B8** — |ΔECE| exceeds 2× the pooled across-seed SD in **10/16** pairs (all 8 MiniImageNet, only 2 of
8 CIFAR-FS). Direction robust; per-pair effect sizes on CIFAR-FS are not.

**B10** — only `adapter.rank` varies over {1,2,4,8,16,32,64} × 3 seeds; 21/21 completed.
`ece_optimum_is_interior: false`. Evidential ECE lowest at rank 1 (0.291) drifting to 0.309 at rank 64;
softmax moves the *opposite* way, 0.093 → 0.081. **Required wording: "no interior optimum observed *in
the tested range*."** 3 seeds × 7 ranks is plausibly underpowered to exclude a subtle U-shape — saying
so costs nothing.

**B12** — *Sentinel*: an industrial product inspector built on this thesis's **pattern** — frozen
ResNet-18, prototype store, evidential vacuity, explicit **UNKNOWN** routing to manual inspection, with
the softmax baseline shown side by side so the overconfidence is visible. Register a product from 1–10
photos; no runtime training — enrolment is averaging embeddings, detection is one frozen forward pass
(~16 ms/image on CPU).

> **Say this unprompted if you show it.** Sentinel is a **demonstrator, not evidence**. It runs its own
> simple model: it is **not** wired to the trained thesis adapters, and it has **no accuracy
> evaluation**. It shows that the architecture deploys and that an UNKNOWN route is usable. It is not a
> result, and no number in this deck comes from it.

---

# Q&A crib sheet

One or two sentences each, plus a slide number. **Long answers lose defences.**

| Question | Answer | Slide |
|---|---|---|
| "If energy beats vacuity, why is this a Bayesian thesis?" | The evidential formulation is what supplies an explicit uncertainty representation to measure at all; we never claimed it universally produces the best OOD detector. RQ2 is the experiment that separates objective from score, and RQ4 shows the calibration deficit it creates is repairable post-hoc. | 20, 22, 24 |
| "Isn't a CNN backbone outdated?" | The deployment regime selects the family: an 86 M-parameter ViT does not fit in 1 MB of flash. In the low-data regime the 2025 literature reports CNNs matching ViTs. We make no general CNN-superiority claim. | 4, B13 |
| "Your accuracy isn't state of the art." | Correct, and it isn't trying to be. 1.06 points behind a fully meta-trained DINO ViT-S at 662× fewer trainable parameters, and 3.56 *ahead* of the backbone-matched ResNet-50. Every RQ is about relative differences. | B5, B6 |
| "Your backbones saw the test classes during ImageNet pretraining." | Yes — stated as limitation 1. It makes absolute accuracies non-comparable to from-scratch work. It does not touch any RQ, because every RQ compares configurations that all share the same pretraining. | 16, 24 |
| "Why did the research questions change from the proposal?" | They were refined from comparison to attribution after a literature review that came *after* the grid completed. No experiment was discarded, no data removed, and every original question is answered — including the two whose answer was "no." | 11 |
| "Did you change the questions because the results were inconvenient?" | The two negative results are still here, on Slides 11 and 24. If the goal had been convenience, the 0-of-20 calibration result is the first thing that would have disappeared. | 11, 24 |
| "Three seeds is not many." | Agreed for sub-1-point differences, and we flag those. Seed spread exceeds 1 point in only 2 of 40 configurations, and every RQ3 claim at matched budget clears 2σ. | 24, B14 |
| "One hyperparameter recipe across 40 configurations isn't fair." | Deliberate — it makes the grid a controlled comparison rather than 40 independently-tuned numbers. We answer "how do these axes compare under one recipe," not "what is each cell's best number." | 24 |
| "Energy outperforms your Bayesian score — doesn't that sink the thesis?" | It narrows it, and RQ2 is the experiment that made it visible. Vacuity is the better *native probabilistic* score — 37–38 of 40 cells against MSP — and free at inference within our noise floor. If you only need an OOD ranker, softmax plus energy is the honest recommendation. | 24, B4 |
| "Isn't 'backbone-intrinsic' just a name for 'we don't know'?" | Largely yes, and we say so. What the experiment establishes is that it is **not** the budget — the gap survives budget equalisation intact on ResNet-18. Which property is responsible needs more backbones. | 21B, 24 |
| "Two backbones can't support a claim about backbones." | Agreed, and the claim is scoped accordingly: backbone identity matters; the responsible property is unidentified. That's the stated limit and the first item of future work. | 21B, 25 |
| "What exactly does your 43.7% η² mean?" | The share of total variation in far-OOD AUROC attributable to the scoring-rule axis, in a main-effects decomposition over a balanced 2 objectives × 4 scores factorial, n = 792 per pool. The objective axis gets 0.27%. | 20, B14 |
| "Why not just give the ratio?" | Because it isn't stable. The objective's share is near zero, so the quotient swings — 163× on the full set, 394× once the one cell missing an arm is removed. The shares are stable, so we quote the shares. | 20, B14 |
| "Is η² appropriate — pairs sharing a backbone aren't independent?" | A fair objection to the sign test over 16 pairs. The decomposition itself is main-effects η² over a balanced factorial on 96 **per-seed** observations with a retained residual — we recomputed per-seed precisely because averaging seeds first can manufacture cleanliness. | 19, B14 |
| "You ran 32×32 CIFAR images through an ImageNet ResNet?" | Resized to 224×224 and ImageNet-normalised first. The backbone is frozen and ImageNet-pretrained, so matching its input statistics is a requirement, not a choice. | 16, B15 |
| "Why a prototype head instead of a linear classifier?" | A linear head trained on the 64 base classes cannot transfer to 20 disjoint test classes. Protocol requirement, not a design shortcut. | 12 |
| "Does post-hoc recalibration break the OOD detection?" | In 78% of comparisons it does not; in 22% AUROC dropped by more than half a point, worst single-pool drop about −0.03. We measured it rather than assuming it, and the exception is on the slide. | 22 |
| "What was the hardest bug?" | An evidential collapse: raw L2 logits are large-negative for ResNet-18 features, so softplus of the logit is ≈0 everywhere — uniform Dirichlet, dead gradients. Fixed with cosine similarity plus a learnable evidence affine; the mapping now lives in exactly one function shared by training and evaluation. | 14 |
| "How do you know the latency numbers are right?" | They come from one canonical Kaggle T4 session, GPU and single-thread CPU. During closeout we found three functions silently preferring a dev-laptop measurement instead — up to 47% error, no crash, no failing test — fixed it, and regenerated every downstream artefact. It's reported as the fourth self-correction. | 23, 24, B13 |
| "How do you know the results are reproducible?" | Frozen episode seeds, sorted-key JSON, byte-identical reruns — and the matched-budget experiment (Aug 26–27) re-trained 18 arms from the Aug 2–6 grid from scratch, reproducing them at max absolute difference **0.0**. | 17 |
| "What would you do with six more months?" | More backbones, to identify the mechanism behind RQ3. Everything else on the future-work list is worth less. | 25 |

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
| "Our 6,928-parameter adapter **beats** full fine-tuning." | "It **matches** full fine-tuning — the margin is inside its own seed spread. The ResNet-18 configuration's +0.98 points is the margin that clears noise." |
| "Refitting always preserves the OOD ranking." | "OOD ranking is preserved in 78% of comparisons, with a measured minority exception." |
| "Refitting fixes evidential calibration." | "Refitting improves it in 48/48 cells, but it stays worse than plain softmax in all 48." |
| "The score matters 163× more than the objective." | "The score explains 43.7% of far-OOD variance (13.0% near); the objective under 1%. Two to three orders of magnitude — the ratio itself is unstable." |
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

| Slide | Asset | Path |
|---|---|---|
| 3, 19, B11 | Reliability diagrams | `results/grid_plots/*_reliability.png` |
| 20, B11 | OOD separation histograms | `results/grid_plots/*_ood_histogram.png` |
| B13 | Pareto, latency vs AUROC | `results/pareto_latency_vs_auroc__cifar_fs.png`, `__mini_imagenet.png` |
| B1 | Accuracy table, rendered | `results/mvt_table_accuracy.png` |
| B2 | Calibration table, rendered | `results/mvt_table_calibration.png` |
| B3 | OOD AUROC table, rendered | `results/mvt_table_ood_auroc.png` |
| B10 | Rank sweep | `results/rq5_rank_sweep.png` |

**Four figures you must draw yourself** — the repo has no version of any of them, and they are the
most important visuals in the deck:

1. **Slide 12 — the system pipeline.** Frozen backbone → adapter → prototype head → two
   interpretations, with the frozen / trainable / parameter-free annotations.
2. **Slide 15 — the factorial.** Five axes with two levels each, the 32 + 8 cell count, and the three
   experiments kept visually distinct. This slide carries the thesis's method; a plain bulleted list
   wastes it.
3. **Slide 13 — the two adapter architectures side by side.** Bottleneck-parallel (down / ReLU / up,
   summed at the block output) against LoRA (a low-rank update *inside* the frozen 1×1 convolution).
   The difference between "after the block" and "inside the weights" is the whole of RQ3's
   architecture axis, and it is hard to convey in words.
4. **Slide 11 — the chronology.** Three labelled columns, left to right: original proposal → completed
   experiments → refined attribution questions. The left-to-right arrow is doing the defensive work;
   a table alone does not show that the experiments came before the refinement.

---

# Before the defence — re-check against the repo, not against this file

1. **Run counts** on Slides 2 and 15 (`progress.txt`).
2. **The recommended deployment point** on Slide 23 and B13 — the only numbers in the thesis that are
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
| A1 | Drop the unstable "163×"; quote the η² shares | `RQ_SUPERVISOR_REPORT.md` §1/§4.2 + the `.pdf`; `RQ_RESULTS_SUMMARY.md`; `DEFENCE_SLIDE_PLAN.md` |
| A2 | State that the refit leaves evidential worse than softmax in 48/48 | added to `RQ_SUPERVISOR_REPORT.md` §6.1 — the `.pdf` is stale, regenerate |
| A3 | Orig-RQ1 is mislabelled "serial vs parallel" | `RQ_SUPERVISOR_REPORT.md` Appendix A; `RQ_RESULTS_SUMMARY.md` |
| A6 | The evidence affine is learnable, not frozen at (2, −6) | fixed in the `.md` files; `RQ_SUPERVISOR_REPORT.pdf` still stale |
| A7 | Do not describe `kl_weight_max` 0.1 as VAL-selected | `09_methodology.md` §9.5 discloses it; add a `progress.txt` entry if the reason is known |
| A8 | State the cosine×10 prototype metric | the reports say only "similarity" |
| — | Two unverifiable citations removed from this deck | confirm they are absent from the thesis draft too |

**Provenance for everything else:** `docs/RQ_SUPERVISOR_REPORT.md` (four-RQ framing, answers, novelty
labels) · `docs/RQ_RESULTS_SUMMARY.md` (RQ tables, 16-pair evidence, §5.1 matched-budget) ·
`docs/RESULTS_MASTER.md` (Tables 1–8, §4 positioning, §4.7 publishable claims) ·
`docs/DEFENCE_BRIEF.md` (CNN-backbone objection, deployment arithmetic) ·
`docs/RQ3_MATCHED_BUDGET_PLAN.md` (pre-registration record) · `progress.txt` · `step_writeups/*`.
