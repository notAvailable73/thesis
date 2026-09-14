# B-PEFT — Defence Slide Plan

**What this is.** A slide-by-slide content plan for the masters defence presentation of
*Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones*.
Every number on every slide is traceable; the source file and section is named under each slide.

**Built from:** `docs/RQ_SUPERVISOR_REPORT.md` (2026-08-27, the current canonical RQ framing),
`docs/RQ_RESULTS_SUMMARY.md`, `docs/RESULTS_MASTER.md`, `docs/DEFENCE_BRIEF.md`, `progress.txt`,
`step_writeups/*`, `proposal.txt`.

**Assumed format:** 22–25 minutes of talking, then questions. 28 main slides + 14 backup.
Each slide below carries a minute budget and a priority tag:

- **[CORE]** — do not cut. The defence fails without it.
- **[STRONG]** — cut only if the chair gives you 15 minutes instead of 25.
- **[CUT-FIRST]** — first to go when time is short; keep the content as a backup slide.

---

## 0. Deck strategy — read this before you build a single slide

Six decisions shape everything below. They are the difference between a deck that defends this
thesis and a deck that defends a different, weaker one.

### 0.1 Do not copy the reference deck's structure wholesale

The TransSeg deck you were shown is an **architecture-proposal** deck: problem → prior architectures →
our architecture → Dice-score table → "ours is competitive." Its centre of gravity is a pipeline
diagram and a results table where their row should be bolded.

**This thesis is not that kind of work, and presenting it that way loses the defence.** This thesis's
contribution is a **controlled attribution study**: a balanced 120-run factorial plus a pre-registered
adjudication experiment that answers *which design choice causes which reliability outcome*. Its
accuracy is deliberately not state of the art, and was never trying to be. If your headline slide is
an accuracy table, the first committee question is "so you're 5 points behind a ViT?" and you have
handed them the wrong frame.

**Therefore:** the centrepiece slide is the **experimental design** (Slide 18), and the headline
claims are **attribution claims**, not accuracy claims. Keep the departmental skeleton — title, recap,
objectives, problem statement, literature review, gap, RQs, method, datasets, metrics, results,
limitations, conclusion, references — because that is the format the committee expects. Change what
sits in the middle of it.

### 0.2 Put the scorecard on slide 2, negative results included

A defence rewards knowing exactly where your claims stop. Four RQs, four answers, one of which
overturned the project's own preferred hypothesis, and one large well-powered negative result.
Showing that on slide 2 buys you credibility for the next twenty minutes, and it removes the
possibility that a committee member "discovers" the negative result in the Q&A and treats it as
something you hid.

### 0.3 Pre-empt the two known attacks in the first five minutes

Both are already documented, both have answers, and both are much cheaper to spend 40 seconds on
early than to absorb as an ambush:

1. **"CNN backbones are outdated in 2026."** This was raised at the pre-defence. Answer it on
   Slide 5 as a design justification, not as a defence. Full material: `docs/DEFENCE_BRIEF.md`.
2. **"Your accuracies are contaminated by ImageNet pretraining."** True, documented, and it does not
   touch any RQ, because every RQ is a claim about *relative* differences under one fixed recipe.
   Say it on Slide 20 before anyone asks.

### 0.4 Handle the RQ renumbering head-on, on its own slide

The proposal posed four **comparison** questions ("does A beat B"). The thesis now presents four
**attribution** questions ("what causes X"). Same experiments, no data discarded, reframed after a
literature review showed each comparison question had close precedent. **A committee member holding
the proposal will notice.** Slide 14 exists to say this first, in your own words, with the mapping
table. Never let it look like the goalposts moved quietly.

### 0.5 Lead the contributions with RQ3, not with RQ1

Ranked by strength of contribution: **RQ3 > RQ2 > RQ1 > RQ4** (`RQ_SUPERVISOR_REPORT.md` §1).
RQ3 is the only **[NOVEL]** label, and it is the most defensible result in the thesis for a specific
reason worth saying out loud: the hypothesis, the decision rule, the statistical thresholds and an
explicit "inconclusive" outcome were **all committed to writing before any of the deciding runs
existed** — and the experiment then returned a verdict *against* the hypothesis the project had been
favouring. That is the single strongest methodological moment in the whole thesis. Slide 25.

### 0.6 One claim per slide, and the claim is the title

Write slide titles as findings, not as labels. "RQ3 results" is a label. "Accuracy follows the
adapter; calibration follows the backbone" is a finding. The committee reads titles even when they
stop listening.

---

## 1. Slide map

| # | Title | Min | Priority |
|---|---|---:|---|
| 1 | Title | 0.5 | CORE |
| 2 | The thesis in one slide — four questions, four answers | 1.5 | CORE |
| 3 | Recap — what was proposed, what was delivered | 1.0 | STRONG |
| 4 | Motivation — the reliability gap in few-shot edge vision | 1.0 | CORE |
| 5 | Why a frozen CNN and a tiny adapter | 1.0 | CORE |
| 6 | Problem statement | 0.5 | CORE |
| 7 | Objectives | 0.5 | CORE |
| 8 | Literature map — seven fields, one uncovered intersection | 1.0 | CORE |
| 9 | Prior work I — adapters on frozen CNNs (TSA, Conv-Adapter) | 1.0 | STRONG |
| 10 | Prior work II — evidential and Bayesian uncertainty (Sensoy, BEL, BayesAdapter) | 1.0 | STRONG |
| 11 | Prior work III — the non-Bayesian baseline that matters (energy, OpenOOD) | 0.75 | STRONG |
| 12 | The research gap | 1.0 | CORE |
| 13 | Research questions | 1.0 | CORE |
| 14 | From proposal questions to thesis questions | 0.75 | CORE |
| 15 | System pipeline | 1.0 | CORE |
| 16 | The two adapters, and the accident that makes RQ3 possible | 1.0 | CORE |
| 17 | The head — one logit vector, two interpretations | 1.0 | CORE |
| 18 | Experimental design — the factorial | 1.5 | CORE |
| 19 | Protocol rigour — what makes these numbers trustworthy | 1.0 | STRONG |
| 20 | Datasets and OOD pools | 0.75 | CORE |
| 21 | Metrics | 0.5 | CORE |
| 22 | Baseline results — accuracy and parameter cost | 1.0 | CORE |
| 23 | RQ1 — shots drive accuracy, the head drives calibration | 1.5 | CORE |
| 24 | RQ2 — the scoring rule dominates the training objective, 163× | 1.5 | CORE |
| 25 | RQ3 — the dissociation, and the experiment that adjudicated it | 2.5 | CORE |
| 26 | RQ4 — the calibration deficit is fixable post-hoc | 1.25 | CORE |
| 27 | Efficiency — where the deployment cost actually lives | 1.0 | STRONG |
| 28 | Honest reporting — three self-corrections | 1.0 | CORE |
| 29 | Limitations | 1.0 | CORE |
| 30 | Contributions | 1.0 | CORE |
| 31 | Future work | 0.75 | STRONG |
| 32 | References | — | CORE |

Total: ~31 minutes as written. **Cut to fit** by dropping slides 3, 11, 19, 27, 31 (≈4.5 min) and
compressing 9+10 into one prior-work slide (≈1 min). That lands at ~25 minutes.

---

## 2. Slide-by-slide content

---

### SLIDE 1 — Title · [CORE] · 0.5 min

**On the slide**

- **B-PEFT: Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight
  CNN Backbones**
- Your name, student ID
- Supervisor / co-supervisor names and titles
- Department of Computer Science and Engineering, Islamic University of Technology
- University logos

**Say:** nothing beyond your name and the title. Move on in fifteen seconds.

---

### SLIDE 2 — The thesis in one slide · [CORE] · 1.5 min

**Title it:** *Four questions about reliability, four answers*

**On the slide** — the overarching question in a quote box at the top:

> What governs reliability — accuracy, confidence calibration, and out-of-distribution detection — in
> parameter-efficient adaptation of frozen lightweight CNN backbones for few-shot image
> classification, and can the deficits so identified be remediated after training?

Then the scorecard:

| | Question | Answer |
|---|---|---|
| **RQ1** | Which design axis drives which outcome? | Shot count drives accuracy (76%); head type drives calibration (83%); the two are near-independent |
| **RQ2** | Is OOD detection caused by the training objective or the scoring rule? | The scoring rule, by **163×** on far-OOD |
| **RQ3** | Does accuracy follow adapter architecture and calibration follow parameter budget? | Accuracy yes. Calibration follows **neither** — it follows the backbone |
| **RQ4** | Can the evidential head's calibration be fixed after training without breaking its OOD ranking? | Yes: **48/48** cells improved; ranking survives in **78%** of comparisons |

Footer line: **189 training runs · 600 frozen test episodes each · 3 seeds · 0 run failures.**

**Say:** "Those are the four answers. Two of them are not the answer we expected going in, and I'll
show you both, including the one where a pre-registered experiment overturned our own hypothesis."

**Trap to avoid:** do not say "all four confirmed." They did not, and the deck is stronger for it.

**Source:** `RQ_SUPERVISOR_REPORT.md` §1. Run count = 120 (main grid) + 48 (matched budget) + 21
(rank sweep). Earlier pilot phases are additional and not counted here.

---

### SLIDE 3 — Recap: proposed vs delivered · [STRONG] · 1.0 min

**Title it:** *From the proposal to here*

**On the slide** — two columns, proposal on the left, delivered on the right:

| Proposed | Delivered |
|---|---|
| ResNet-18 + MobileNetV3 + ConvNeXt-Nano | ResNet-18 + MobileNetV3-Small (ConvNeXt dropped, scope) |
| Bottleneck / LoRA / BitFit / Full-FT / Linear-Probe | All five implemented and run |
| Evidential Dirichlet vs softmax head | Both, as two *interpretations* of one prototype head |
| CIFAR-FS | CIFAR-FS **+ MiniImageNet** (second dataset added) |
| 5-way k-shot | 5-way 1-shot **and** 5-shot |
| Accuracy, ECE, Brier, OOD AUROC | All four **+ macro-F1, FPR@95, latency, MACs, peak memory** |
| One comparison per RQ | A **balanced 2⁵ factorial**, 3 seeds, 40 configurations |

**Say:** "The scope grew in the directions that mattered — a second dataset, a second backbone, a
second shot regime — and shrank in one place, ConvNeXt-Nano, which we dropped to keep the factorial
balanced rather than run a lopsided grid."

**Trap:** if asked why ConvNeXt was dropped, the honest answer is design balance and compute, not
that it failed. Do not invent a result for it.

---

### SLIDE 4 — Motivation · [CORE] · 1.0 min

**Title it:** *A model that is wrong and confident is worse than one that says "I don't know"*

**On the slide** — three stacked points, each one sentence:

- **Few-shot at the edge is a real deployment shape.** A device is handed 5 examples of a new class
  and must work immediately, on hardware with kilobytes of SRAM and no floating-point unit.
- **Full fine-tuning is not available there.** 25 support images against 11.7 million parameters
  overfits, and the compute does not exist on the device anyway.
- **Parameter-efficient adaptation solves the capacity problem and inherits a reliability problem.**
  Adapters keep accuracy. Nobody checks whether the confidence numbers survive.

Then the consequence, boxed:

> A system that reports 95% confidence while being 60% correct is dangerous in exactly the
> applications that motivate edge deployment: inspection, triage, screening.

**Visual:** a reliability diagram from `results/grid_plots/cifar_fs_5shot_softmax_reliability.png`
next to `..._evidential_reliability.png`. Two pictures of the same failure mode, one much worse.
Do not explain them yet — you explain them on Slide 23.

**Say:** "This thesis is about the second sentence in that box. Not about making few-shot models more
accurate — about finding out what happens to their honesty when you shrink them."

---

### SLIDE 5 — Why a frozen CNN and a tiny adapter · [CORE] · 1.0 min

**Title it:** *The deployment target chooses the backbone*

This slide exists to pre-empt "CNNs are outdated." Frame it as a design justification, delivered
before anyone asks.

**On the slide** — two facts and one table.

Fact 1: MCU-class targets run **32–512 kB SRAM**, under ~1 MB flash, 20–200 MHz, often no FPU. A
memory-optimised transformer attention block costs **~180 ms** on an STM32F746 against **~8–12 ms**
for CNN inference (TinyDL survey, arXiv:2506.18927, 2025).

Fact 2: **in low-data regimes CNNs match ViTs**, even ViTs pretrained at scale, because the
convolutional inductive bias substitutes for data the transformer does not have (arXiv:2510.04794,
Oct 2025). Few-shot *is* the low-data regime.

| Deployed at inference | Parameters | × ours |
|---|---:|---:|
| DINOv3 ViT-7B | 7,000,000,000 | 2,800× |
| CLIP-ViT-B/16 | 86,500,000 | 34.6× |
| DINO-ViT-S/16 | 21,000,000 | 8.4× |
| ResNet-18 (ours) | 11,700,000 | 4.7× |
| **MobileNetV3-Small (ours)** | **2,500,000** | — |

**Say (verbatim, this is rehearsed):** "We are not claiming CNNs beat transformers at scale — they
don't, and the thesis doesn't dispute it. We're claiming an 86-million-parameter ViT does not fit in
one megabyte of flash, which is arithmetic rather than preference, and that in the low-data regime
the accuracy argument for the transformer is weakest."

**Source:** `DEFENCE_BRIEF.md` §1, §2, §3.4.

---

### SLIDE 6 — Problem statement · [CORE] · 0.5 min

**Title it:** *Problem statement*

**On the slide** — the formal statement, then three named problems:

> Given a frozen backbone `f_θ` pretrained on a source distribution, and an episode
> `E = (S, Q)` with a 5-class support set `S` of k labelled images per class drawn from classes
> **disjoint** from anything `f_θ` was adapted on, learn an adapter `g_φ` with `|φ| ≪ |θ|` such that
> the resulting classifier is **accurate** on `Q`, **calibrated** (its confidence matches its
> correctness), and **able to reject** inputs drawn from outside the episode's label space.

1. **Parameter inefficiency** — updating millions of parameters on 25 images overfits and does not
   fit on the device.
2. **The reliability gap** — PEFT methods inherit the base model's overconfidence, and nobody
   measures it.
3. **Architectural bias** — PEFT research is overwhelmingly about transformers. How adapters behave
   inside convolutional blocks is not well characterised.

**Source:** `proposal.txt` §3.

---

### SLIDE 7 — Objectives · [CORE] · 0.5 min

**On the slide** — five numbered objectives, each verifiable:

1. Build a frozen-backbone + adapter + parameter-free-head framework that adapts to a new 5-way task
   from 1–5 examples per class under ~32,000 trainable parameters.
2. Measure accuracy, calibration and OOD detection **on the same runs**, so their relationship can be
   observed rather than assumed.
3. Determine which design axis governs which reliability outcome, under a controlled factorial rather
   than one comparison at a time.
4. Establish whether observed differences are caused by the adapter's architecture, its parameter
   budget, or the backbone — by experiment, not by argument.
5. Test whether an identified reliability deficit can be remediated after training, without
   retraining and without destroying what the model does well.

**Say:** "Objective 2 is the one that does most of the work. Almost nobody reports accuracy,
calibration and OOD detection on the same runs, and several of our findings are only visible because
we do."

---

### SLIDE 8 — Literature map · [CORE] · 1.0 min

**Title it:** *Seven literatures, and the intersection none of them cover*

**On the slide** — the coverage table. This is the single most persuasive literature slide you have,
because it makes the gap visible instead of asserted.

| Literature | Acc | F1 | ECE | OOD | Params | Episodic | Edge |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Classical few-shot (ProtoNet, MAML, MetaOptNet) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot (P>M>F, CoOp, Tip-Adapter) | ✅ | ❌ | ❌ | ❌ | ~ | ~ | ❌ |
| PEFT for ViTs (VPT, SSF, AdaptFormer) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge (Conv-Adapter, LoRA-Edge) | ✅ | ❌ | ❌ | ❌ | ✅ | ~ | ✅ |
| Frozen-CNN episodic adapters (TSA, FiT) | ✅ | ❌ | ❌ | ❌ | ✅ | ~ | ~ |
| Bayesian PEFT (Laplace-LoRA, BLoB, BayesAdapter) | ✅ | ❌ | ✅ | ~ | ✅ | ❌ | ❌ |
| Evidential few-shot (BEL) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ~ |
| **B-PEFT (this thesis)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Say:** "Two of those crosses are documented absences, not our inference. The survey covering the
entire ViT PEFT field does not discuss calibration, uncertainty or OOD anywhere. The 2025 TinyML
survey covers quantisation in depth and never mentions uncertainty. Two communities, two surveys,
the same blind spot."

**Trap:** do not claim you read every table of every paper in the edge-PEFT row. The claim is about
what those papers *foreground*, and that is how `RESULTS_MASTER.md` §4.6 states it. Say it that way.

---

### SLIDE 9 — Prior work I: adapters on frozen CNNs · [STRONG] · 1.0 min

**Title it:** *Task-Specific Adapters — our closest precedent, and we replicate it*

**On the slide**

- **TSA** (Li, Liu & Bilen, CVPR 2022, arXiv:2107.00358): frozen ResNet-18, serial vs parallel
  adapter placement, 600 sampled episodic tasks, parameter-free nearest-centroid head.
  **Finding:** residual/parallel connections beat serial "in almost all cases."
- **Conv-Adapter** (arXiv:2208.07463): locality-preserving CNN adapters beat 1×1/linear-style ones.
- **What they do not do:** neither compares against LoRA; neither tests a second backbone; neither
  reports calibration or OOD detection at all.

Boxed, in your own voice:

> Our serial-vs-parallel result **replicates TSA**. We say so. What is ours is the LoRA comparison,
> the second backbone, and the uncertainty layer on top.

**Say:** "This is the paper a committee member is most likely to know, and it gets the same answer we
do on placement. Claiming that as a discovery would be the easiest way to lose this defence, so we
report it as a replication and point at what is actually new."

**Why this slide earns its minute:** volunteering your closest precedent, unprompted, is worth more
in a defence than any result on the slide.

---

### SLIDE 10 — Prior work II: evidential and Bayesian uncertainty · [STRONG] · 1.0 min

**Title it:** *Three papers say Bayesian calibration works. We find where it stops working.*

**On the slide**

| Paper | Regime | Their result |
|---|---|---|
| **Sensoy et al. 2018** (arXiv:1806.01768) | Full training, EDL introduced | Dirichlet evidence gives usable uncertainty at negligible inference cost |
| **BEL** (arXiv:2207.13137) | Few-shot, **backbone meta-trained** | Calibration *improves* (3.59% vs 14.69% ECE) |
| **BayesAdapter** (IJCV, arXiv:2412.09718) | Frozen CLIP, full linear adapter, up to 32 shots | Calibration *improves* (~2.5 pp ECE) |

Then the positioning line:

> All three succeed with capacity we deliberately remove: a trainable backbone, a large adapter, or
> more shots. We test the corner where all three are absent — frozen backbone, ≤31,744 trainable
> parameters, true disjoint-class episodes — and calibration **degrades**, in 20 of 20 matched pairs.
> Locating that boundary is the contribution, not a failure to reproduce them.

**Say:** "These are not contradictions. All three are true. The difference is capacity and data, and
the interesting question is where the boundary sits."

---

### SLIDE 11 — Prior work III: the baseline that matters · [STRONG] · 0.75 min

**Title it:** *The honest competitor is not softmax — it is the energy score*

**On the slide**

- **Liu et al. 2020** (arXiv:2010.03759): the **energy score**, a logit-space OOD score that needs no
  Bayesian machinery, no training change, and no extra parameters. It beats max-softmax-probability.
- **OpenOOD / OpenOOD-1.5**: post-hoc scores routinely beat training-method changes.
- **Why this slide exists:** most evidential papers compare vacuity against max-softmax-probability
  only. That comparison is too easy. We run energy on every configuration, and it changes our answer.

**Say:** "If you only compare your Bayesian uncertainty against max-softmax probability, you will
conclude that it wins. We compared against the strongest cheap alternative, and that comparison is
the reason RQ2 exists at all."

**Note:** this slide sets up Slide 24 and Slide 28. If you cut it, fold its first bullet into
Slide 24.

---

### SLIDE 12 — The research gap · [CORE] · 1.0 min

**Title it:** *Research challenge*

**On the slide** — four challenges, with the two this thesis attacks highlighted (mirror the
reference deck's green-highlight convention):

1. Parameter-efficient adaptation is measured on accuracy alone; **its effect on calibration and OOD
   detection is essentially unreported.** ← attacked
2. Where results *are* reported, the training objective and the scoring rule are varied together, so
   **no published result separates "the Bayesian training helped" from "the score we read it with
   helped."** ← attacked
3. Adapter comparisons confound architecture with parameter budget, because the better adapter is
   usually also the bigger one.  ← attacked
4. Uncertainty methods are benchmarked on server hardware; the edge cost is asserted, not measured.
   ← attacked

**Say:** "Three of those four are confounds, not missing experiments. That is what makes this a
design problem rather than a compute problem — and it's why the core of this thesis is the
experimental design, which is the next section."

---

### SLIDE 13 — Research questions · [CORE] · 1.0 min

**Title it:** *Research questions*

**On the slide** — the overarching question at the top (same box as Slide 2), then the four RQs
written in full, each with its one-line role:

- **RQ1 — Attribution.** How is variance in accuracy, calibration and OOD detection distributed
  across dataset, shot count, backbone, adapter type and head interpretation?
  *Role: frames everything else — tells you which axis to look at for which outcome.*
- **RQ2 — Objective vs score.** Is OOD performance attributable to the training objective the model
  was adapted under, or to the scoring rule its outputs are read with, and can the two be separated?
  *Role: resolves the OOD outcome.*
- **RQ3 — Architecture vs budget.** Within the adapter axis, does accuracy follow adapter
  architecture while calibration follows trainable-parameter budget?
  *Role: resolves the adapter axis. This is the novelty claim.*
- **RQ4 — Remediation.** Can the evidential head be recalibrated post-hoc by refitting only two
  parameters, and does its OOD ranking survive that refit?
  *Role: turns a diagnosis into a fix.*

**Say:** "RQ1 is diagnostic, RQ2 and RQ3 are causal, RQ4 is remedial. They are in that order on
purpose — RQ1 tells you which axes matter, and RQ2 and RQ3 go and find out why."

---

### SLIDE 14 — From proposal questions to thesis questions · [CORE] · 0.75 min

**Title it:** *The questions were reframed. The experiments were not.*

Do not skip this slide. Deliver it before anyone asks.

**On the slide**

| Proposal asked (comparison) | Result obtained | Now presented as |
|---|---|---|
| Orig-RQ1: serial vs parallel placement | Parallel wins **16/16**, +2.1 to +8.3 pp | Part of RQ3's architecture axis |
| Orig-RQ2: does evidential calibrate better than softmax? | **No — 0/20 matched pairs** | RQ1 (head dominates calibration) + RQ4 (can it be fixed?) |
| Orig-RQ3: does a Bayesian prior improve near-OOD detection? | Yes vs softmax scores (37–38/40); **no vs energy** (10–14/40) | RQ2 (objective vs score) |
| Orig-RQ4: latency vs uncertainty Pareto frontier | Backbone drives latency 5.12×; adapter 3.9–5.8%; head 1.29% | Supporting evidence under RQ1 |

Boxed:

> **No experiment was discarded and no result was dropped.** A literature review conducted after the
> grid completed found each original *comparison* question had close precedent. The same completed
> experiments answer *attribution* questions — which the balanced factorial supports and for which
> precedent is substantially thinner.

**Say:** "The proposal asked four 'does A beat B' questions. We answered all four, and then found
that each had been asked before. The same data answers a harder set of questions — what causes
what — and that is the version in front of you today."

**Source:** `RQ_RESULTS_SUMMARY.md` Appendix; `RQ_SUPERVISOR_REPORT.md` Appendix A.

---

### SLIDE 15 — System pipeline · [CORE] · 1.0 min

**Title it:** *The system*

**Visual:** a clean left-to-right diagram, drawn by you (this is the slide that most needs a real
figure). Elements, in order:

```
support images ─┐
                ├─→ [ FROZEN backbone ]─→[ ADAPTER ]─→ embeddings ─→ class prototypes ┐
query images ───┘      ResNet-18 /         the ONLY                                    ├→ logits
                       MobileNetV3-S       trained part          query embedding ──────┘
                       (never updated)     (≤31,746 params)                             │
                                                                                        ▼
                                                            ┌───────────────────────────┴────┐
                                                            │ softmax → probabilities        │
                                                            │ evidential → α, S → vacuity    │
                                                            └────────────────────────────────┘
```

Annotate three things directly on the diagram:

- **Frozen** on the backbone, with "11.7M / 2.5M parameters, never updated."
- **Trainable** on the adapter, with "6,928 – 31,746 parameters — the entire trainable budget."
- **Parameter-free** on the head, with "a class = the mean embedding of its support images."

**Say:** "The head has no weights. A class is literally the average of its five support embeddings,
and you classify by similarity to those averages. That matters for a reason that is easy to miss: a
trained linear head over the 64 training classes cannot transfer to 20 disjoint test classes, so a
parameter-free head is not a shortcut here, it's a requirement of the protocol."

**Trap:** if asked "isn't that just ProtoNet?" — yes, the head is a prototype head, and the novelty
is nowhere near the head. Say that plainly and point at the adapter and the evaluation.

---

### SLIDE 16 — The two adapters, and a useful accident · [CORE] · 1.0 min

**Title it:** *Two adapters — and a parameter-count reversal we did not design*

**On the slide** — left half, the two architectures:

- **Parallel bottleneck:** 1×1 conv down (d→r) → ReLU → 1×1 conv up (r→d), computed on the block
  input and summed at the block output. Placed at the final block of each of 4 stages.
- **LoRA:** a low-rank update injected into the backbone's own 1×1 convolution —
  `W_eff = W_frozen + (α/r)·A·B`. Changes what the backbone computes, not what happens after it.

Right half, the table that makes RQ3 possible:

| Backbone | Bottleneck | LoRA | Larger arm |
|---|---:|---:|---|
| ResNet-18 | **31,744** | 12,288 | bottleneck, by 2.58× |
| MobileNetV3-Small | 6,928 | **10,752** | **LoRA**, by 1.55× |

Boxed:

> **The ordering reverses between backbones** — a consequence of channel widths, not of design. This
> partially separates adapter *architecture* from adapter *budget*, which is normally impossible
> because the better adapter is usually also the bigger one everywhere.

**Say:** "We did not build this reversal. We noticed it. And it is the reason RQ3 can be asked at
all — though as I'll show in a moment, it only gets you part of the way, and we had to run a second
experiment to finish the job."

---

### SLIDE 17 — One logit vector, two interpretations · [CORE] · 1.0 min

**Title it:** *The head is one thing; how you read it is another*

**On the slide** — the key design decision, which is also a defensible piece of engineering:

The prototype head always emits raw similarity logits `z`. Two readings:

| | Softmax | Evidential (Dirichlet) |
|---|---|---|
| Probability | `p = softmax(z)` | `α = softplus(z·s + b) + 1`, `S = Σα`, `p = α/S` |
| Confidence | max softmax probability | — |
| Uncertainty | none native | **vacuity** = `K/S`, K = 5 |
| Extra trainable params | 0 | **2** (`s`, `b`) |

Boxed:

> Because both interpretations sit on the *same* logits, a matched evidential/softmax pair differs by
> exactly **two parameters** — 31,746 vs 31,744. That is what makes the head axis a clean experimental
> factor rather than a confound.

**Say:** "This is the design decision that makes the whole factorial work. The evidential and softmax
arms are not two different models — they're the same model read two ways, differing by two scalars.
So when RQ1 attributes 83% of calibration variance to the head, it is attributing it to those two
scalars and the loss that trained them, not to a different architecture."

**Worth mentioning if you have the time:** a real bug in this exact mapping caused an early
"evidential collapse" — raw L2 logits are large-negative for ResNet-18 features, so `softplus(z) ≈ 0`
everywhere, giving a uniform Dirichlet and dead gradients. Fixed by moving to cosine similarity plus
the learnable affine. It is a good answer to "what was the hardest bug," and it shows the
single-source-of-truth discipline in `PrototypeHead.to_evidence()`.

---

### SLIDE 18 — Experimental design · [CORE] · 1.5 min

**This is the centrepiece of the deck. Give it the most visual care.**

**Title it:** *A balanced factorial, not a sequence of comparisons*

**On the slide** — the grid, drawn as five axes:

```
        dataset  ×  shots  ×  backbone       ×  adapter      ×  head
     CIFAR-FS        1        ResNet-18         bottleneck      softmax
     MiniImageNet    5        MobileNetV3-S     LoRA            evidential
        2        ×    2    ×      2            ×     2        ×    2      = 32 cells
                                          + 8 baseline cells (Full-FT, Linear-Probe)
                                          = 40 configurations × 3 seeds = 120 runs
```

Then the three experiments that make up the evidence base:

| Experiment | Runs | Purpose |
|---|---:|---|
| Main factorial grid | **120** | RQ1, RQ2, and RQ3's first evidence |
| Matched-budget adjudication (**pre-registered**) | **48** | Settles RQ3's causal claim |
| Controlled rank sweep | **21** | Tests, and rejects, an interior-optimum hypothesis |

Footer: **36.3 h + 6.5 h GPU wall time · 120/120 and 48/48 completed · zero run errors.**

**Say:** "Every prior comparison in this literature moves one thing at a time and reports the
winner. A balanced factorial lets you do something different: attribute the variance in an outcome
to the axes that produced it. That is what converts 'parallel beat LoRA' into 'accuracy is governed
by adapter architecture', and it is the methodological core of the thesis."

**Trap:** someone will ask whether one frozen recipe across 40 configurations is fair. Answer on
Slide 29; if asked early: "deliberate — it makes the grid a controlled comparison rather than 40
independently-tuned numbers, and it's stated as a limitation."

---

### SLIDE 19 — Protocol rigour · [STRONG] · 1.0 min

**Title it:** *Why these numbers can be trusted*

**On the slide** — five bullets, each a concrete mechanism, not a claim:

- **Frozen test episodes.** 600 episodes, seeds fixed in a version-controlled file, identical across
  every run in the thesis. Nobody ever re-sampled a favourable test set.
- **Hyperparameter selection on VAL only.** A separate 100-episode validation split (seeds
  10000–10099). The 600 test seeds were never used for any selection decision.
- **Byte-identical reproducibility.** Re-running a config reproduces its metrics JSON exactly. The
  48-run matched-budget experiment re-ran 18 existing grid arms from scratch and reproduced the
  committed numbers with **max absolute difference 0.0** across 12–13 metric keys per cell.
- **Pre-registration.** RQ3's decision rule, thresholds and an explicit "inconclusive" outcome were
  written down before any of the deciding runs existed (`docs/RQ3_MATCHED_BUDGET_PLAN.md`).
- **Control guards.** Automated checks that only the intended axis differs between the configs in a
  comparison — 0 unaccounted config keys across all 48 matched-budget runs.

**Say:** "The 18 re-runs reproducing to a maximum absolute difference of zero is the number I'd point
at if you only trust one thing on this slide. It means the second experiment measures the same
quantity the first one did, and that is not something you can assume across a four-month gap."

---

### SLIDE 20 — Datasets and OOD pools · [CORE] · 0.75 min

**Title it:** *Two in-distribution datasets, five OOD pools*

**On the slide**

**In-distribution** (5-way episodic, test classes disjoint from training classes):

| Dataset | Split | Classes (train/val/test) | Native resolution |
|---|---|---|---|
| CIFAR-FS | Bertinetto | 64 / 16 / 20 | 32×32, derived from CIFAR-100 |
| MiniImageNet | Ravi & Larochelle | 64 / 16 / 20 | 84×84, ImageNet-derived |

**All inputs are resized to 224×224** and ImageNet-normalised, because the backbones are
ImageNet-pretrained and frozen. Say this if asked — running native 32×32 through a frozen ImageNet
ResNet would destroy the features the whole method depends on.

**Out-of-distribution:**

- **Far-OOD:** SVHN, Gaussian noise
- **Near-OOD:** CIFAR-100-heldout, MiniImageNet-heldout, TinyImageNet (with the 25 wnids it shares
  with MiniImageNet excluded for MiniImageNet runs)

Then the caveat, volunteered:

> **Stated limitation.** Our backbones are ImageNet-pretrained, while the standard few-shot protocol
> trains from scratch on base classes. MiniImageNet's classes *are* ImageNet classes, so "novel" test
> classes were seen during pretraining. **Absolute accuracies are therefore not comparable to
> from-scratch few-shot literature.** Every research question here concerns *relative* differences
> between design choices under one fixed recipe, which is what makes the results usable despite this.

**Say the caveat out loud.** It costs 15 seconds and removes an entire line of attack.

---

### SLIDE 21 — Metrics · [CORE] · 0.5 min

**Title it:** *What we measure, and why each one is there*

| Metric | What it answers | Direction |
|---|---|---|
| Accuracy, macro-F1 | Is it right? (F1 exposes per-class recall collapse accuracy hides) | ↑ |
| **ECE** (pooled) | When it says 90%, is it right 90% of the time? | ↓ |
| Brier | Accuracy and calibration in one proper scoring rule | ↓ |
| **OOD AUROC** | Can it rank an unfamiliar input as more uncertain than a familiar one? | ↑ |
| FPR@95%TPR | At a usable operating point, how often does it wave through an outlier? | ↓ |
| Latency, MACs, peak memory, trainable params | Can it actually be deployed? | ↓ |

Uncertainty scores compared, all four computed on **all** runs:
**vacuity** (evidential), **MSP**, **temperature-scaled MSP**, **energy**.

**Say:** "The last row of that table is the one nobody else in this literature reports alongside the
others, and the four scores in the last line are the reason RQ2 is answerable."

---

### SLIDE 22 — Baseline results · [CORE] · 1.0 min

**Title it:** *What 6,928 parameters buys*

**On the slide** — CIFAR-FS 5-way 5-shot, the headline efficiency result:

| Configuration | Trainable params | Accuracy |
|---|---:|---:|
| Parallel bottleneck, ResNet-18 | 31,744 | **91.44%** |
| Parallel bottleneck, MobileNetV3-Small | **6,928** | 90.74% |
| Full fine-tuning, ResNet-18 | 11,176,512 | 90.47% |
| Linear probe (no adaptation) | 0 | 87.41% |

Boxed: **6,928 trainable parameters reach the accuracy of retraining 11.18 million — 0.06% of the
budget.**

**Say:** "Two readings, and I want to be precise about which is which. The ResNet-18 row beats full
fine-tuning by 0.98 points, which comfortably clears seed noise. The MobileNet row is +0.27 points,
which is *inside* its own seed spread — so the claim there is 'matches', not 'beats'. Matching full
fine-tuning at 0.06% of the parameters is the interesting result anyway."

**Trap:** do not say "beats" for the MobileNet row. `RESULTS_MASTER.md` §4.7 claim 5 is explicit
about this, and a careful committee member checking the confidence intervals will catch it.

---

### SLIDE 23 — RQ1 · [CORE] · 1.5 min

**Title it:** *Shots drive accuracy. The head drives calibration. They barely interact.*

**On the slide** — the η² decomposition, main effects over the balanced factorial, computed on 96
per-seed observations:

| Outcome | Dataset | Shots | Backbone | Adapter | Head | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.75% | **76.05%** | 3.92% | 9.14% | 0.19% | 8.95% |
| ECE | 2.02% | 2.13% | 4.63% | 0.63% | **82.89%** | 7.70% |
| OOD AUROC (far) | 1.18% | 22.68% | 11.54% | 1.73% | **39.01%** | 23.87% |
| OOD AUROC (near) | 1.64% | **42.60%** | 0.61% | 21.98% | 20.98% | 11.42% |

Then the finding that is genuinely new:

> **ECE and OOD AUROC are orthogonal, not in tension.** Across all 40 cells the rank correlation is
> *positive* (ρ = +0.433 far, +0.477 near), which naively says worse-calibrated configurations detect
> OOD better. That is a head effect: stratifying by head collapses it to ρ = +0.15/+0.20 (evidential)
> and +0.26/+0.24 (softmax). **Once the head is fixed, calibration carries almost no information
> about OOD detection.** This result was not found anywhere in the literature.

**Visual:** if you have room, the two reliability diagrams from Slide 4, now explained.

**Say:** "83% of calibration variance and 0.2% of accuracy variance sit on the same axis. That is
close to a clean separation of concerns, and it means you can choose your head for calibration
without paying for it in accuracy. The orthogonality result underneath is the part with no
precedent — and you can only see it if you measure both on the same runs."

**Volunteer:** "The qualitative pattern — that calibration and accuracy have different drivers — goes
back to Guo et al. 2017. What's new here is the formal decomposition and this regime."

---

### SLIDE 24 — RQ2 · [CORE] · 1.5 min

**Title it:** *It is the score you read, not the loss you trained with — by a factor of 163*

**On the slide** — first the confound, in one line:

> Every published comparison scores evidential models with vacuity and softmax models with MSP. The
> objective and the score always move together. We compute **all four scores on all runs**: a clean
> 2 objectives × 4 scores factorial, n = 792 comparisons per pool.

Then the result:

| Pool | Scoring rule | Training objective | Ratio |
|---|---:|---:|---:|
| Far-OOD | **43.7%** | 0.27% | **163×** |
| Near-OOD | **13.0%** | 0.60% | **22×** |

And the concrete version:

| Score | On evidential-trained logits | On softmax-trained logits |
|---|---:|---:|
| **Energy** | 0.911 | 0.929 |
| MSP | ~0.79 | ~0.79 |

**Say:** "The same score gives near-identical AUROC regardless of which objective produced the
logits. So the Bayesian training is not what was buying the OOD performance — the score was. And
there is a 2026 theory result proving softmax is a mathematical special case of an evidential
classifier, which pre-explains exactly why the objective barely matters. I'd rather cite that myself
than have it raised from the floor."

**Volunteer:** "Concurrent WACV 2026 work runs a comparable objective × score ANOVA at larger scale,
so the technique is not unprecedented. What has no precedent is cross-applying energy onto
Dirichlet-parameterised logits and finding near-equivalence. That's the surviving contribution."

**Coverage caveat on the slide, small print:** computed on 99 of 120 recoverable checkpoints; all
included cells are matched pairs; a regression guard confirms 99/99 cells unchanged against the
committed metrics.

---

### SLIDE 25 — RQ3 · [CORE] · 2.5 min · **the most important slide in the deck**

Give this two slides if your template allows. Structure it as a three-beat story.

**Title it:** *Accuracy follows the adapter. Calibration follows the backbone.*

#### Beat 1 — the pattern (16 matched comparisons)

| Outcome | Winner is the bottleneck architecture | Winner is the larger-budget arm |
|---|---:|---:|
| Accuracy | **16/16** | 8/16 |
| Near-OOD AUROC | **16/16** | 8/16 |
| ECE | 8/16 | **16/16** |

Two observed facts, stated so they cannot be overread:

1. **The accuracy winner does not change when the budget ordering reverses.** Bottleneck wins on both
   backbones, holding 2.58× *more* parameters on one and 1.55× *fewer* on the other.
2. **The ECE winner does change, exactly in step with the budget ordering.**

Sign consistency 16/16, two-sided p ≈ 3.05×10⁻⁵.

#### Beat 2 — the problem with that (say this yourself)

> The budget ordering reverses **with the backbone**. So "calibration follows the budget" and
> "calibration follows the backbone" predict the *identical* 16/16 pattern. The 120-run grid cannot
> tell them apart. We preferred the budget account on parsimony — which is not the same as having
> evidence for it.

#### Beat 3 — the experiment that settled it

**Design:** both architectures rebuilt at the **same budget within each backbone**. MiniImageNet
5-shot, 2 backbones × 2 budget levels × 2 arms × 2 heads × 3 seeds = **48 runs**. Residual budget
mismatch **≤ 3.10%**, against 55–158% before. Decision rule pre-registered.

**Verdict: `backbone_intrinsic`.** Fired in 3 of 4 pre-registered cells. The budget hypothesis fired
in **0 of 4**.

| Cell | ΔECE unmatched | ΔECE matched | collapse ratio |
|---|---:|---:|---:|
| ResNet-18 / evidential | +0.1111 | **+0.1123** | 1.01 |
| ResNet-18 / softmax | +0.1080 | **+0.0991** | 0.92 |
| MobileNetV3-S / evidential | −0.0124 | −0.0062 | 0.50 |
| MobileNetV3-S / softmax | −0.0152 | −0.0060 | 0.61 |

And the secondary outcome that was genuinely at risk: at matched budget, **bottleneck still wins
accuracy 8/8 and near-OOD 8/8, all beyond 2σ**, no gap collapsing.

**Say (rehearse this, it is your best 45 seconds):** "Equalising the budget leaves ResNet-18's
calibration gap completely intact — collapse ratios of 1.01 and 0.92, meaning the matched gap is as
large as the unmatched one. A gap caused by the budget difference should have vanished with it. It
didn't. So the hypothesis this project had been working under for two months is wrong, we
pre-registered the rule that would tell us so before we ran it, and the answer is narrower than what
we had been ready to claim."

**State the limit explicitly, do not wait to be asked:** "'Backbone-intrinsic' is where the evidence
points, not an explanation. Which property of ResNet-18 versus MobileNetV3-Small — depth, width,
normalisation, the inverted-residual block, feature-norm scale at the adapter sites — produces the
reversal is untested, and two backbones cannot separate those candidates. That is the honest
remaining limit of RQ3, and closing it takes more backbones, not more seeds."

**Absolutely do not say** "calibration follows the parameter budget." That reading is superseded.

---

### SLIDE 26 — RQ4 · [CORE] · 1.25 min

**Title it:** *The calibration deficit is a default, not a property of the head*

**On the slide**

**What was refit:** only the two parameters of the evidence affine (`scale`, `bias`), on the
**validation** episode split, evaluated on the frozen 600-episode test split. No retraining.

| Outcome | Result |
|---|---|
| ECE improved | **48 / 48 cells (100%)**, mean **−0.137 absolute** |
| Where the trained head sat | refit values cluster at scale 7–14 vs jointly-trained 1.5–4.5 (init 2, −6) |
| OOD ranking preserved (Δ ≥ −0.005) | **150 / 192 comparisons (78%)**, mean ΔAUROC +0.004 |
| Worst-case rank correlation | Spearman ρ = **0.921** |

Then the caveat, on the slide, not just spoken:

> **Reordering does occur.** Vacuity is `K/Σα`, a function of *all* logits jointly, so a per-logit
> monotone transform does not guarantee the sample ordering survives. In 22% of comparisons AUROC
> dropped by more than half a point, worst single-pool drop ≈ −0.03. **The claim is "survives in the
> large majority of cases with a measured minority exception," never "always."**

**Say:** "Two parameters, fitted on validation episodes, with no retraining, recover most of the
calibration gap in every single cell. That relocates the RQ1 finding: the evidential head's
calibration problem was substantially a badly-chosen default, not something intrinsic. And we
*measured* whether the OOD ranking survived rather than assuming it — which matters, because it
doesn't always."

**Volunteer:** "Post-hoc calibration is a mature field, and Guo et al. report comparable or larger
drops from temperature scaling. We are not claiming to have discovered that refitting helps. The new
parts are the mechanism — this specific two-parameter evidence affine — and the systematic
quantification across 48 cells and 192 ranking comparisons."

---

### SLIDE 27 — Efficiency · [STRONG] · 1.0 min

**Title it:** *The frozen trunk dominates inference cost — the adapter does not*

**On the slide** — measured on real hardware (Kaggle T4, GPU and single-thread CPU):

| Finding | Number |
|---|---:|
| Backbone choice drives latency | **5.12×** (ResNet-18 vs MobileNetV3-Small, matched adapter) |
| Adapter choice barely does | **3.9–5.8%**, despite up to 2.6× parameter differences |
| Evidential head cost at inference | **1.29%** mean — below the session's own 5.91% noise floor |

**Recommended deployment point:** MobileNetV3-Small + parallel bottleneck + evidential —
**11.86 ms/image**, **6,930 trainable parameters**, TinyImageNet near-OOD AUROC 0.870 (1-shot) / 0.919 (5-shot).
On MiniImageNet, where MobileNetV3-Small falls outside the accuracy tolerance, the point moves to
ResNet-18 + parallel + evidential at 62.38 ms.

**Visual:** `results/pareto_latency_vs_auroc__cifar_fs.png`.

**Say:** "The practical rule is: choose your backbone for latency and your adapter for accuracy,
because they barely trade against each other. And 'evidential uncertainty is free at inference' is
now a measurement below our own noise floor rather than a claim — Sensoy's 2018 paper asserts it as
EDL's selling point; as far as we found, nobody had measured it in this regime."

**Volunteer if pressed:** "The Pareto claim is native-score-conditional. If softmax is allowed its
best score — energy — rather than max-softmax-probability, evidential's presence on the CIFAR-FS
5-shot frontier goes to zero. That's in the write-up."

---

### SLIDE 28 — Honest reporting · [CORE] · 1.0 min

**Title it:** *Three times the data contradicted us, and what we did about it*

This slide is worth more than any positive result on it. Committees remember it.

**On the slide**

1. **A single-configuration finding did not generalise.** An early result said evidential vacuity was
   roughly on par with the energy score. At grid scale, **energy wins ~70% of comparisons** (vacuity
   wins only 10/40 far-OOD and 14/40 near-OOD). We report the correction ourselves. *The defensible
   claim narrowed to: vacuity is a substantially better OOD ranker than softmax-probability scores —
   37–38 of 40 cells — but a well-chosen logit-space score still beats it.*
2. **A pre-registered experiment overturned our own preferred hypothesis.** RQ3's "calibration
   follows the parameter budget" reading fired in **0 of 4** cells once budget was equalised. The
   claim in this thesis is the narrower, correct one.
3. **A hypothesis was tested and demoted.** We predicted calibration error would reach an optimum at
   an intermediate parameter budget. A controlled 21-run rank sweep with everything else fixed found
   **no interior optimum in the tested range** — evidential ECE is lowest at rank 1 and drifts up;
   softmax moves the opposite way. Reported as a negative result rather than defended.

**Say:** "A thesis where every hypothesis confirmed would be more suspicious, not less. The
corrections are on this slide because we found them ourselves, at the point where the evidence got
strong enough to find them."

**Prepare for:** "Doesn't the energy result undermine your whole premise?" — Answer: "It narrows
it. The premise was never 'Bayesian uncertainty is the best possible OOD score.' RQ2 is exactly the
experiment that separates those two claims, and its answer is that the score dominates. What survives
is that the evidential head is the better *native probabilistic* option, and that it's free at
inference — and if you only need an OOD ranker, the honest recommendation is softmax plus energy."

---

### SLIDE 29 — Limitations · [CORE] · 1.0 min

**Title it:** *Limitations*

Seven, in this order. Say them at pace; the point is that they are volunteered.

1. **ImageNet-pretraining overlap.** Backbones are ImageNet-pretrained; MiniImageNet classes *are*
   ImageNet classes. Absolute accuracies are not comparable to from-scratch few-shot literature. All
   RQs concern relative differences.
2. **One frozen hyperparameter recipe** across all 40 configurations, tuned once. The grid answers
   "how do these axes compare under one recipe," not "what is each cell's best achievable number."
   This was deliberate — it keeps the grid a controlled comparison.
3. **Three seeds**, and effectively one for the Full-FT and Linear-Probe baselines, whose seed axis
   is inert by construction. Any claim resting on a sub-1-point margin should be read accordingly.
4. **RQ3's mechanism is unidentified.** Two backbones establish that backbone identity matters; they
   cannot establish which property is responsible.
5. **Main effects only.** The η² decomposition does not model interactions.
6. **Evidential is not temperature-scalable** in this codebase, so the TS-softmax comparison is
   structurally favourable to softmax. It is still the right comparison — TS is cheap and standard —
   but the asymmetry is stated.
7. **Novelty claims rest on a non-exhaustive search.** Five independent passes, ~100 searches, full
   PDF reads of the closest competitors. Absence of found prior art is not proof of absence.

**Say:** "Number 4 is the one I'd fix first, and it takes more backbones, not more seeds."

---

### SLIDE 30 — Contributions · [CORE] · 1.0 min

**Title it:** *Contributions*

Order by strength, not by RQ number. Label each honestly — the labels are from the novelty
assessment in `RQ_SUPERVISOR_REPORT.md`.

1. **[NOVEL]** — **A dissociation between adapter architecture and adapter parameter budget, by
   outcome metric.** Accuracy and near-OOD ranking follow architecture, invariant to budget, tested
   at matched budget. Calibration follows neither — it follows the backbone. Established by a
   pre-registered experiment that overturned the project's own working hypothesis. No comparable
   dissociation found in vision or NLP PEFT; the standard convention is to *equalise* budgets, never
   to exploit a reversed ordering.
2. **[PARTLY NOVEL]** — **Separating the OOD training objective from the OOD scoring rule**, and
   finding the score dominates by 163×. The specific new part: cross-applying the energy score onto
   Dirichlet-parameterised logits and finding near-equivalence to softmax-trained energy — no
   precedent found.
3. **[PARTLY NOVEL]** — **A formal variance decomposition of accuracy, calibration and OOD detection
   over a balanced five-axis PEFT factorial**, including the result that ECE and OOD AUROC become
   orthogonal once head interpretation is controlled — which was not found anywhere else.
4. **[PARTLY NOVEL]** — **A two-parameter post-hoc recalibration of an evidential prototype head**,
   with OOD-ranking preservation measured rather than assumed: 48/48 cells improved, 78% of rankings
   preserved.
5. **[CONFIRMATORY, and useful]** — A reproducible harness and evidence base: 189 runs, frozen
   episode seeds, byte-identical reruns, VAL-only selection, and a working demonstrator application
   showing few-shot enrolment with honest uncertainty and an explicit "unknown" decision.

**Say:** "If I had to defend one of those, it's the first, and the reason is the pre-registration
rather than the result."

---

### SLIDE 31 — Future work · [STRONG] · 0.75 min

**Title it:** *Future work*

Ordered by value, and each one connected to a limitation on Slide 29 rather than floating free:

1. **More backbones** — the only way to identify *which* backbone property drives the calibration
   reversal (closes limitation 4). Three or four backbones spanning depth, width and normalisation
   would separate the candidates.
2. **A from-scratch control run** — the same grid with a randomly-initialised backbone trained on
   base classes, which removes the pretraining-overlap objection entirely (closes limitation 1).
3. **A transformer arm under our own protocol** — a ViT-Tiny or DeiT-Small inside this grid converts
   the "CNNs are appropriate here" argument into a measurement.
4. **Per-cell hyperparameter tuning on VAL** — answers whether any axis effect is a recipe artefact
   (closes limitation 2).
5. **An evidential analogue of temperature scaling**, so both heads get a post-hoc correction and the
   comparison stops being structurally asymmetric (closes limitation 6).
6. **Extending RQ4's refit to the deployed setting** — recalibrating on-device from the support set.

---

### SLIDE 32 — References · [CORE]

Two slides, IEEE style, numbered to match in-text citations. Minimum set, grouped:

**Few-shot / adapters:** Snell et al. (ProtoNet); Bertinetto et al. (CIFAR-FS / R2D2); Ravi &
Larochelle (MiniImageNet); Li, Liu & Bilen, *Task-Specific Adapters*, CVPR 2022 (arXiv:2107.00358);
Conv-Adapter (arXiv:2208.07463); Hu et al., LoRA; Houlsby et al. (bottleneck adapters);
Hu et al., P>M>F (arXiv:2204.07305).

**Uncertainty / calibration:** Sensoy et al. 2018 (arXiv:1806.01768); Guo et al. 2017
(arXiv:1706.04599); Minderer et al. 2021 (arXiv:2106.07998); Kull et al. 2019 (Dirichlet
calibration); Bengs et al., NeurIPS 2022; BEL (arXiv:2207.13137); BayesAdapter (arXiv:2412.09718);
Laplace-LoRA, ICLR 2024.

**OOD:** Liu et al. 2020, energy score (arXiv:2010.03759); OpenOOD / OpenOOD-1.5;
arXiv:2603.07571 (objectives for OOD); arXiv:2605.06382 (vacuity cardinality critique).

**Concurrent work you cite proactively:** arXiv:2601.10836 (WACV 2026, objective × score ANOVA);
arXiv:2605.22746 (softmax as a special case of evidential); arXiv:2608.10372 (accuracy-preserving
post-hoc calibration).

**Backbones / edge:** He et al. (ResNet); Howard et al. (MobileNetV3); TinyDL survey
(arXiv:2506.18927); CNN-vs-ViT low-data study (arXiv:2510.04794); DINOv3 (arXiv:2508.10104).


---

## 3. Backup slides

Put these after the references, numbered B1–B14. You will not show them unprompted. Each exists
because a specific question is likely, and pulling up a prepared slide instead of talking from
memory is the single most effective thing you can do in a defence Q&A.

| # | Slide | Triggered by |
|---|---|---|
| **B1** | Full accuracy table, all 40 configurations | "What were the actual numbers for X?" |
| **B2** | Full calibration table (ECE, ECE-after-TS, Brier) | "How bad is the calibration really?" |
| **B3** | Full OOD AUROC table + FPR@95 | "Which OOD set, which score?" |
| **B4** | Vacuity vs every softmax score, head to head | "How does evidential compare to energy?" |
| **B5** | Positioning vs published few-shot methods | "How do you compare to state of the art?" |
| **B6** | The parameter/accuracy trade, stated as a cost | "What does the parameter saving cost you?" |
| **B7** | PEFT method budgets (VTAB-1k), with the protocol warning | "How does this compare to VPT/SSF?" |
| **B8** | The 16-pair RQ3 table in full | "Show me all sixteen comparisons" |
| **B9** | The matched-budget design: ranks, budgets, mismatches | "How exactly did you match the budgets?" |
| **B10** | The rank sweep that found no interior optimum | "Did you try varying the rank?" |
| **B11** | The LoRA-Ensemble partial contradiction, and the response | "Doesn't LoRA-Ensemble find the opposite?" |
| **B12** | Reliability diagrams, all four dataset × shot cells | "Show me the calibration visually" |
| **B13** | Vacuity separation histograms (ID vs OOD) | "What does the OOD score actually look like?" |
| **B14** | The demonstrator application | "Does this run on anything real?" |

### B1 — Accuracy, all 40 configurations
Use `results/mvt_table_accuracy.png` directly, or rebuild from `docs/RESULTS_MASTER.md` Table 1.
**Have one number memorised per dataset:** CIFAR-FS 5-shot best 91.58% (ResNet-18, parallel,
evidential); MiniImageNet 5-shot best 95.88% (same configuration). Note that the ± column is the
600-episode CI and the seed-std column is the one to use when comparing two rows.

### B2 — Calibration, all 40 configurations
`results/mvt_table_calibration.png` / `RESULTS_MASTER.md` Table 2. The number to have ready:
evidential ECE is worse than plain softmax in **20/20** matched pairs (1.4×–9.1×) and worse than
temperature-scaled softmax by **5.3×–51.2×**. Pair it immediately with Slide 26 — that gap is
substantially recoverable post-hoc.

### B3 — OOD AUROC and FPR@95
`results/mvt_table_ood_auroc.png` / `RESULTS_MASTER.md` Tables 3 and 4.

### B4 — Vacuity vs every softmax-side score
`RESULTS_MASTER.md` Table 5. The summary line:

| Comparison | Far-OOD | Near-OOD |
|---|---:|---:|
| vacuity vs MSP | 38/40 (+0.111) | 37/40 (+0.053) |
| vacuity vs TS-MSP | 38/40 (+0.127) | 38/40 (+0.067) |
| vacuity vs **energy** | 10/40 (−0.022) | 14/40 (−0.007) |

Plus the low-data trend: near-OOD advantage over MSP is **+0.064 at 1-shot vs +0.043 at 5-shot** —
larger where there is less data, which is the direction the original hypothesis predicted. Describe
it as "held between the two shot levels tested," not as a trend traced across many.

### B5 — Positioning against published few-shot methods
`RESULTS_MASTER.md` §4.4.1. Same 5-way episodic protocol, so accuracy *is* comparable, subject to the
pretraining caveat which applies to both sides.

| Method | Backbone | Trainable | CIFAR-FS 5-shot | MiniIN 5-shot |
|---|---|---:|---:|---:|
| Sup-21k > ProtoNet | ViT-B/16 | ~85.8 M | **96.7** | **99.2** |
| DINO > ProtoNet | ViT-S/16 | ~21 M | 92.5 | 98.0 |
| DINO > ProtoNet | ResNet-50 | ~25 M | — | 92.0 |
| BEL (evidential few-shot) | ResNet-12 | n/r | 86.92 | 79.60 |
| **Ours**, parallel bottleneck | ResNet-18 (frozen) | **31,744** | 91.44 | 95.56 |
| **Ours**, parallel bottleneck | MobileNetV3-S (frozen) | **6,928** | 90.74 | 90.10 |

### B6 — What the parameter saving costs
`RESULTS_MASTER.md` §4.4.2. **Show this before you are asked to.** Three separate conclusions, and
they must not be blurred:

| Ours vs | Param saving | CIFAR-FS 5-shot | MiniIN 5-shot | MiniIN 1-shot |
|---|---:|---:|---:|---:|
| ResNet-18 adapter vs DINO>PN ViT-S | **662× fewer** | **−1.06 pp** | −2.44 pp | −8.07 pp |
| ResNet-18 adapter vs DINO>PN ViT-B | 2,703× fewer | −0.76 pp | −2.84 pp | −10.27 pp |
| ResNet-18 adapter vs DINO>PN **ResNet-50** | 788× fewer | — | **+3.56 pp** | **+5.83 pp** |
| MobileNetV3-S adapter vs DINO>PN ViT-S | 3,031× fewer | −1.76 pp | −7.90 pp | **−18.18 pp** |

1. At 5-shot on CIFAR-FS the saving is close to free — 1.06 points at 662× fewer parameters.
2. Backbone-family-matched, we win outright: +3.56 points over DINO>ProtoNet on ResNet-50.
   **The remaining gap is a ViT gap, not a parameter-efficiency gap.**
3. The trade is genuinely bad at 1-shot and on MiniImageNet with the small backbone — up to −18.2
   points. Volunteer this; do not let it be found.

### B7 — PEFT budgets on VTAB-1k
`RESULTS_MASTER.md` §4.4.3. **Carries a mandatory warning label on the slide:** VTAB-1k gives 1,000
labelled examples per task against our 25. The accuracies are there so the budgets are not floating
free and **must never be tabulated against B5**.

Full ViT-B/16 fine-tuning 85.8 M; SSF 240 k; VPT-Deep 600 k; Adapter 270 k; linear probe 40 k; CoOp
8,192. Ours: 31,744 (ResNet-18) and 6,928 (MobileNetV3-Small).

**On CoOp, the honest exception:** 8,192 trainable parameters is below our ResNet-18 configuration.
Don't dispute it. Point at deployment: CoOp's context vectors steer a frozen CLIP that must be
resident at inference, so the deployed system is 34.6× larger, and CoOp reports neither calibration
nor OOD detection.

### B8 — The 16 RQ3 pairs in full
`RQ_RESULTS_SUMMARY.md` §5. Each row: dataset, shots, backbone, head, both parameter counts, which
arm is larger, both ECEs, which is better calibrated. **Note on the slide:** |ΔECE| exceeds 2× the
pooled across-seed SD in 10/16 pairs — all 8 MiniImageNet pairs, only 2 of 8 CIFAR-FS pairs.
Direction is robust; per-pair effect sizes on CIFAR-FS are not.

### B9 — Matched-budget design
`RQ_RESULTS_SUMMARY.md` §5.1 and `docs/RQ3_MATCHED_BUDGET_PLAN.md`. Show the rank → parameter mapping
per arm and the residual mismatch column (0.43%–3.10%). Two points to make:

- The plan budgeted 30 runs by reusing three existing grid arms. We trained **48**, re-running those
  arms from scratch. That is strictly stronger — every arm in every comparison comes from one
  identical recipe, and it converted a provenance check into a full end-to-end reproduction, which
  passed with max absolute difference 0.0.
- One planned deviation, recorded: MobileNetV3's Level-H pair was built fresh at ≈9.4k (ranks 22/14)
  because bottleneck rank 25 would have exceeded the 24-channel shallowest stage.

### B10 — The rank sweep
`results/rq5_rank_sweep.png`, `results/rq5/`. Architecture, backbone, dataset and shots all fixed;
only `adapter.rank` varies over {1,2,4,8,16,32,64} × 3 seeds, 21/21 runs completed.
`ece_optimum_is_interior: false`. Evidential ECE lowest at rank 1 (0.291), drifting to 0.309 at rank
64; softmax moves the *opposite* way, 0.093 → 0.081. Accuracy-optimal rank is 64, ECE-optimal rank is
1 — the mismatch survives, but via two opposing monotone trends, not a U-shape.

**Required wording:** "no interior optimum observed **in the tested range**." 3 seeds × 7 ranks is
plausibly underpowered to exclude a subtle U-shape, and saying so costs nothing.

### B11 — The LoRA-Ensemble partial contradiction
`RQ_RESULTS_SUMMARY.md` §7. **A supervisor who knows this paper will ask, so have the slide.**
LoRA-Ensemble (arXiv:2405.14438, Appendix C, Fig. 11) reports that "at rank 32, the calibration of a
single network augmented with LoRA begins to deteriorate" — on CIFAR-100, the same dataset family,
over an overlapping rank range, in the opposite direction to our softmax curve.

Three legitimate distinguishing factors:

1. **Adaptation target** — they inject rank into multi-head self-attention projections in a ViT; we
   use bottleneck-parallel adapters on 1×1 convolutional channels in a CNN.
2. **Head design** — they use a trainable linear head over 100 classes; we use a parameter-free
   nearest-centroid head whose similarity logits are intrinsically bounded in a way a free linear
   layer's are not.
3. **Relative capacity** — rank 64 here is ~63 k parameters against an 11.7 M frozen backbone
   (0.56%); their comparable point injects far more of the ViT's representational capacity.

**Do not add a fourth argument** claiming the turning point "exists, just further out" by citing
Full-FT's ECE against the bottleneck's. That reuses the confounded four-point comparison the rank
sweep was built to eliminate. The honest position is that no reversal was observed in the tested
range and the thesis takes no position beyond it.

### B12 — Reliability diagrams
`results/grid_plots/{cifar_fs,mini_imagenet}_{1,5}shot_{softmax,evidential}_reliability.png`.
Four cells, two heads each. The visual story: softmax hugs the diagonal loosely; evidential sits far
below it — systematically under-confident in a way the vacuity mass explains.

### B13 — OOD separation histograms
`results/grid_plots/*_ood_histogram.png`. In-distribution vs OOD score distributions. This is the
picture behind every AUROC number in the deck, and it answers "what does the score actually look
like" far better than a table.

### B14 — The demonstrator
`app/` — *Sentinel*, an industrial product inspector built on this thesis's exact pipeline: frozen
ResNet-18, prototype store, evidential vacuity, explicit **UNKNOWN** routing to manual inspection,
with the softmax baseline shown side by side so the overconfidence is visible. Register a product
from 1–10 reference photos; no runtime training — enrolment is averaging embeddings, detection is
one frozen forward pass (~16 ms/image on CPU).

**Why keep this in reserve rather than in the main deck:** a demo slide in the main flow invites
"does the demo prove the science," which it doesn't. As a backup it answers "does any of this run on
anything real" in fifteen seconds. If the committee is the kind that likes to see a system, promote
it to a main slide after Slide 27.

---

## 4. Q&A rehearsal

The answer to each is one or two sentences plus a backup slide number. Long answers lose defences.

| Question | Answer | Slide |
|---|---|---|
| "Isn't a CNN backbone outdated?" | We don't dispute transformers are stronger at scale. An 86 M-parameter ViT does not fit in 1 MB of flash, and in the low-data regime the 2025 literature finds CNNs match ViTs. | 5 |
| "Your accuracy isn't state of the art." | Correct, and it isn't trying to be. We're 1.06 points behind a fully meta-trained DINO ViT-S at 662× fewer trainable parameters, and 3.56 ahead of the backbone-matched ResNet-50. Every research question is about relative differences, not absolute accuracy. | B5, B6 |
| "Your backbones saw the test classes during ImageNet pretraining." | Yes, stated as limitation 1. It makes our absolute accuracies non-comparable to from-scratch few-shot work. It does not touch any RQ, because every RQ compares configurations that all share the same pretraining. | 20, 29 |
| "Why did the research questions change from the proposal?" | They were reframed from comparison to attribution after a literature review found each comparison had close precedent. No experiment was discarded; here is the mapping. | 14 |
| "Three seeds is not many." | Agreed for sub-1-point differences, and we flag those explicitly. Seed spread exceeds 1 point in only 2 of 40 configurations, and every RQ3 claim at matched budget clears 2σ. | 29 |
| "One hyperparameter recipe across 40 configurations isn't fair." | Deliberate. It makes the grid a controlled comparison rather than 40 independently-tuned numbers. It's limitation 2, and it means we answer "how do these axes compare under one recipe," not "what is each cell's best number." | 29 |
| "Energy beats your Bayesian score — doesn't that sink the thesis?" | It narrows it, and RQ2 is the experiment that made it visible. Vacuity is the better *native probabilistic* score — 37–38 of 40 cells against MSP — and free at inference. If you only need an OOD ranker, softmax plus energy is the honest recommendation. | 28, B4 |
| "Isn't 'backbone-intrinsic' just a name for 'we don't know'?" | Largely, yes, and we say so. What the experiment establishes is that it is *not* the budget — the gap survives budget equalisation completely intact on ResNet-18. Which backbone property is responsible needs more backbones, not more seeds. | 25, 29 |
| "Two backbones can't support a claim about backbones." | Agreed, and the claim is scoped accordingly: backbone identity matters, the responsible property is unidentified. That's the stated limit, and it's the first item of future work. | 25, 31 |
| "Doesn't LoRA-Ensemble contradict your rank result?" | Partially, on CIFAR-100 over an overlapping range. Three differences: adaptation target, head design, relative capacity. We take no position beyond our tested range. | B11 |
| "Is η² appropriate here — pairs sharing a backbone aren't independent." | A fair objection to the sign test over 16 pairs. The decomposition itself is main-effects η² over a balanced factorial on 96 per-seed observations with a retained residual, and we recomputed per-seed specifically because averaging seeds first can manufacture cleanliness. | 23 |
| "You ran 32×32 CIFAR images through an ImageNet ResNet?" | Resized to 224×224 and ImageNet-normalised first. The backbone is frozen and ImageNet-pretrained, so matching its input statistics is a requirement, not a choice. | 20 |
| "Why a prototype head instead of a linear classifier?" | A linear head trained on the 64 base classes cannot transfer to 20 disjoint test classes. It's a protocol requirement, not a design shortcut. | 15 |
| "What was the hardest bug?" | An evidential collapse: raw L2 logits are large-negative for ResNet-18 features, so softplus of the logit is ~0 everywhere, giving a uniform Dirichlet and dead gradients. Fixed with cosine similarity plus a learnable evidence affine, and the mapping now lives in exactly one function shared by training and evaluation. | 17 |
| "How do you know the results are reproducible?" | Frozen episode seeds, sorted-key JSON, byte-identical reruns — and the matched-budget experiment re-ran 18 existing grid arms from scratch four months later, reproducing them at max absolute difference 0.0. | 19 |
| "What would you do with six more months?" | More backbones, to identify the mechanism behind RQ3. Everything else on the future-work list is worth less. | 31 |

---

## 5. Language discipline

Phrases that will be checked against the record. Get these exactly right.

| Never say | Say instead |
|---|---|
| "Calibration follows the parameter budget." | "Calibration follows the backbone — budget modulates the magnitude, the backbone determines the sign." |
| "Evidential uncertainty is on par with energy." | "Vacuity is a substantially better OOD ranker than softmax-probability scores; a well-chosen logit-space score still beats it." |
| "Our 6,928-parameter adapter beats full fine-tuning." | "It matches full fine-tuning — the margin is inside its own seed spread. The ResNet-18 configuration's +0.98 points is the margin that clears noise." |
| "Refitting always preserves the OOD ranking." | "It survives in the large majority of cases — 78% — with a measured minority exception." |
| "Calibration error has an interior optimum." | "No interior optimum was observed in the tested range." |
| "We discovered that calibration and accuracy have different drivers." | "We formally decompose it in this regime; the qualitative pattern goes back to Guo et al. 2017." |
| "We're the first to do a factorial ANOVA on OOD." | "Concurrent WACV 2026 work does this at larger scale. What has no precedent is cross-applying energy onto Dirichlet logits." |
| "Our method is state of the art." | "Our method is 1.06 points behind a ViT that trains 662× more parameters, and ahead of every backbone-matched comparison." |
| "All four research questions were confirmed." | "Four answered. Two of them not in the direction we expected." |

---

## 6. Asset checklist

Figures to export before building slides. All already exist in the repo.

| Slide | Asset | Path |
|---|---|---|
| 4, 23, B12 | Reliability diagrams | `results/grid_plots/*_reliability.png` |
| 24, B13 | OOD separation histograms | `results/grid_plots/*_ood_histogram.png` |
| 27 | Pareto, latency vs AUROC | `results/pareto_latency_vs_auroc__cifar_fs.png`, `__mini_imagenet.png` |
| 27 | Pareto, params vs accuracy | `results/pareto_params_vs_accuracy__*.png` |
| B1 | Accuracy table, rendered | `results/mvt_table_accuracy.png` |
| B2 | Calibration table, rendered | `results/mvt_table_calibration.png` |
| B3 | OOD AUROC table, rendered | `results/mvt_table_ood_auroc.png` |
| B7 | Efficiency table, rendered | `results/mvt_table_efficiency.png` |
| B10 | Rank sweep | `results/rq5_rank_sweep.png` |
| 27 backup | Pareto quality-axis sensitivity | `results/pareto_audit/quality_axis_variants__*.png` |

**Two figures you must draw yourself** — the repo has no version of either, and they are the two most
important visuals in the deck:

1. **Slide 15, the system pipeline.** Frozen backbone → adapter → prototype head → two
   interpretations, with the frozen/trainable/parameter-free annotations.
2. **Slide 18, the factorial.** Five axes with their two levels each, the 32 + 8 cell count, and the
   three experiments. This is the slide that carries the thesis's method; a plain bulleted list
   wastes it.

**A third worth drawing if you have time:** Slide 16's two adapter architectures side by side —
bottleneck-parallel (down/ReLU/up, summed at the block output) against LoRA (a low-rank update inside
the frozen 1×1 convolution). The difference between "after the block" and "inside the weights" is the
whole of RQ3's architecture axis and is hard to convey in words.

---

## 7. Provenance

Every number in this plan traces to one of:

| Source | What it authorises |
|---|---|
| `docs/RQ_SUPERVISOR_REPORT.md` (2026-08-27) | The four-RQ framing, all four answers, novelty labels, limitations |
| `docs/RQ_RESULTS_SUMMARY.md` | RQ tables, the 16-pair evidence, §5.1 matched-budget adjudication, §7 demoted hypothesis |
| `docs/RESULTS_MASTER.md` | Tables 1–8, §4 positioning against the state of the art, §4.7 publishable claims |
| `docs/DEFENCE_BRIEF.md` | The CNN-backbone objection, deployment arithmetic, the parameter trade |
| `docs/RQ3_MATCHED_BUDGET_PLAN.md` | The pre-registration record |
| `progress.txt` | Run counts, wall times, step closures, the decisions log |
| `step_writeups/*.txt` | Per-step reasoning and honest caveats |
| `app/README.md` | The demonstrator |

**Before the defence, re-check two things against the repo rather than against this file:**
the run counts on Slide 2 and 18, and the recommended Pareto point on Slide 27 — the latter is the
only set of numbers in the thesis that is not byte-reproducible by design, because latency is
hardware- and session-dependent.
