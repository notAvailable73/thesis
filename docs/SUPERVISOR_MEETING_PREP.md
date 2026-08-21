# Supervisor Meeting Prep — B-PEFT Status Conversation

**Purpose of this doc:** a script you can actually talk from tomorrow. Every number in here is
pulled from `progress.txt`, `docs/RESULTS_MASTER.md`, and `docs/DEFENCE_BRIEF.md` — nothing is
invented for this conversation. Where a claim has a caveat, the caveat is included, because your
supervisor will find the gap faster than you'd like if you don't mention it first.

**How to use this:** read top to bottom once tonight. In the meeting, open with the 30-second
pitch, then let your supervisor drive — use the Q&A bank to find the answer to whatever they
actually ask, rather than reciting this in order.

---

## 1. The 30-second opening (say this first)

> "All four research questions now have numerical answers — the 120-run grid closed August 6th,
> and Step 11 (latency) closed August 9th. Two of the four hypotheses passed, one failed cleanly,
> one split. I want to walk you through the four verdicts, what's actually novel about this versus
> prior work, and the weak spots I know are there — then figure out whether we're ready to move to
> writing or whether something on the backlog needs to happen first."

That's it. Don't lead with the good news only — leading with "all four RQs answered" signals the
project is at a decision point (write vs. do more), which is the actual question for this meeting.

---

## 2. What we actually did — one paragraph, in plain terms

We take a small CNN (ResNet-18 or MobileNetV3-Small) already pretrained on ImageNet, **freeze
every one of its 11.7M / 2.5M weights**, and bolt on a tiny trainable adapter — between 7,000 and
32,000 parameters, roughly 0.06%–0.28% of the network. That adapter is the *only* thing that
learns. We give the model 5 example images each of 5 brand-new classes it has never seen
(5-way k-shot episodic few-shot learning) and ask it to classify new images of those classes. Then
instead of measuring only accuracy — which is what almost everyone in this space does — we measure
four things every time: **accuracy**, **calibration** (is a "90% confident" prediction actually
right 90% of the time?), **out-of-distribution detection** (does it say "I don't know" when shown
something unrelated, like house numbers or noise?), and **cost** (parameters, latency on CPU).
The Bayesian piece: instead of a standard softmax output, which always commits to *some* confident
answer, we use an **Evidential Dirichlet head** that can represent "no evidence for any class" —
that's the "I don't know" mechanism. The hypothesis was that this gives better calibration *and*
better OOD detection than softmax. It gave one but not the other, which turned out to be the most
interesting finding in the thesis, not a failure.

**Evidence base:** 120 training runs = 40 configurations (2 datasets × 2 shot regimes ×
2 backbones × 4 adapters × 2 heads) × 3 seeds, each evaluated on the same 600 frozen test
episodes so every run is comparable and reproducible.

---

## 3. The four research questions and verdicts

| RQ | Plain-English question | Verdict |
|---|---|---|
| RQ1 | Where should the adapter go, and does it compete with retraining the whole network? | ✅ **Passed** — strongest result in the thesis |
| RQ2 | Does the evidential head give better-calibrated confidence than softmax? | ❌ **Failed** — cleanly, 0/20 |
| RQ3 | Does the Bayesian prior improve OOD detection, especially with less data? | ⚠️ **Split** — wins vs. softmax scores, loses vs. a free baseline |
| RQ4 | What's the latency vs. uncertainty-quality tradeoff on edge hardware? | ✅ **Passed** — closed 2026-08-09 |

### RQ1 — Adapter placement / parameter efficiency

- Parallel bottleneck beat LoRA in **16/16** matched comparisons, by +2.1 to +8.3 pp. Zero
  exceptions. On MobileNetV3-Small it wins *while using fewer parameters than LoRA* — a strict
  Pareto win, not a tradeoff.
- Headline number: on CIFAR-FS 5-shot, a **31,744-parameter adapter beats full fine-tuning of an
  11,176,512-parameter network** by +0.98 pp. A 6,928-parameter adapter on MobileNetV3-Small
  *matches* full fine-tuning — 0.06% of the parameter budget.
- Honest limit: at **1-shot**, full fine-tuning wins by 2.6 pp. One example per class is where the
  tiny adapter runs out of capacity — say this before you're asked.
- **Honest limit #2, and it's bigger than we first found: the placement principle itself (parallel
  > sequential) is a replication, and the closest match is closer than Conv-Adapter.** Task-Specific
  Adapters (TSA, arXiv:2107.00358, CVPR 2022) already runs this exact comparison — **frozen
  ResNet-18**, serial vs. parallel adapter connection, **600 sampled episodic tasks** on held-out
  domains, **parameter-free nearest-centroid head** — same backbone, same episode count, same
  question, same answer ("residual [parallel] performs better... in almost all cases"). Say this
  before you're asked too. What's new: budgets down to **2** parameters (TSA runs 175K–1.22M),
  a second backbone (MobileNetV3-Small), the LoRA-vs-bottleneck comparison specifically (TSA
  doesn't run it), and calibration/OOD numbers TSA never reports. See §4 below.

### RQ2 — Evidential calibration

- **Scope note, say this before you're asked:** the original proposal phrases RQ2 as testing
  calibration "with <500 trainable parameters." Only the Linear-Probe cells (2 params) literally
  hit that — the PEFT adapters carrying most of the grid sit at 6,928–31,746. The sub-500 case is
  reported and fails the same way (it's the worst ECE in the whole grid), but the headline verdict
  below covers the full 2–31,746 range actually tested, which widens the proposal's stated scope.
- Evidential ECE is worse than plain softmax in **20/20** matched pairs (1.4×–9.1× worse), and
  worse than temperature-scaled softmax by 5.3×–51.2×. Worst case: 0.2938 vs. 0.0057.
- Accuracy doesn't compensate — evidential is 0.61 pp worse on accuracy on average too.
- This is not a tuning artefact: an earlier VAL-only sweep (Step 4.5) already showed the ECE
  surface is flat (~0.285–0.296) across the KL-weight range tested.
- **Why this is a good result and not a wasted year:** it's a well-powered negative (2 datasets ×
  2 shots × 2 backbones × 4 adapters, zero exceptions), it's independently supported by a 2026
  theory paper showing standard evidential objectives keep non-vanishing epistemic uncertainty
  even at infinite data, and it directly contradicts **two** independent prior papers, not one:
  BEL (evidential Dirichlet, meta-trained backbone) and BayesAdapter (arXiv:2412.09718 — a
  different Bayesian mechanism, variational Bayes over a linear CLIP adapter, up to 32 shots),
  both of which found calibration *improving*. Two papers, two different Bayesian mechanisms,
  both improving where there's more capacity or data than our grid provides — that makes our
  degradation result a sharper boundary case, not a contradicted outlier. We froze the backbone
  and starved the budget down to 2 parameters in one cell; **where the boundary sits** is itself
  a finding, and now it's triangulated from two directions, not one.

### RQ3 — OOD detection

- vs. **max-softmax probability**: 38/40 far-OOD wins, 37/40 near-OOD wins (93.8% overall).
- vs. **temperature-scaled MSP**: 38/40 and 38/40 (95.0%).
- The low-data prediction held **between the two shot regimes tested**: the near-OOD advantage is
  bigger at 1-shot (+0.064) than 5-shot (+0.043) — more help exactly where there's less data, which
  is the RQ3 hypothesis. Say "held between the two points measured," not "trend" — only two shot
  values were tested, so it's not a continuous curve, and a supervisor who does stats will ask.
- **But** vs. the training-free **energy score** (no Bayesian machinery, no extra training at
  all): evidential wins only 10/40 far-OOD and 14/40 near-OOD. Energy wins ~70% of the time.
- ⚠️ **This reverses an earlier finding in our own repo.** Step 4.5 (a single configuration)
  suggested evidential was roughly on par with energy. The full 120-run grid overturned that. This
  correction is recorded in `CLAUDE.md` and `progress.txt` — volunteer it, don't let it be found.
- The defensible sentence: *vacuity is a much better OOD ranker than any softmax-probability
  score, and the advantage grows with less data — but a well-chosen logit-space score still beats
  it.*
- **The finding worth its own slide:** the head that is 5–51× worse-calibrated is simultaneously
  the better OOD detector 94% of the time. Calibration quality and OOD-ranking quality are
  empirically decoupled — and this is only visible because both were measured together.

### RQ4 — Latency / edge deployment

- Recommended point: **MobileNetV3-Small + parallel bottleneck + evidential head** — 11.86 ms/image
  on 1-thread CPU, 6,930 trainable parameters. (Shifts to ResNet-18 on MiniImageNet, where
  MobileNet's accuracy falls outside the pre-registered tolerance.)
- **Most novel finding:** *backbone* choice drives latency (5.12× difference, ResNet-18 vs.
  MobileNetV3-Small); *adapter* choice barely does (3.9–5.8% despite up to 2.6× parameter
  differences). The frozen trunk's forward pass swamps everything else.
- **"Evidential uncertainty is free at inference" is measured, not assumed**: 1.29% mean latency
  delta between evidential and softmax heads, *below* the session's own 5.91% measurement-noise
  floor.
- Axes (cost = CPU latency, quality = TinyImageNet near-OOD AUROC) were pre-registered before any
  number existed, so they couldn't be tuned after the fact.
- Two real bugs were found and fixed during closeout — see §6 below. Volunteer these; a supervisor
  who finds them unprompted in the code will weigh them more heavily than if you disclose first.

**One-line thesis-defensible headline:**
> A ~7,000-parameter adapter on a frozen MobileNetV3-Small reaches the accuracy of retraining an
> 11.18M-parameter network, and adding evidential uncertainty costs nothing at inference and makes
> it a much better OOD detector — but it does not make it better calibrated, and a free energy
> score is still the better OOD default.

---

## 4. "Isn't this all done before?" — the novelty conversation

Expect this question, or a version of it, and answer it the same way you'd answer it here — don't
oversell.

**What is NOT novel — say this upfront, it costs nothing:**
- Evidential Dirichlet heads (Sensoy et al., 2018)
- Bottleneck adapters, LoRA, BitFit — all existing PEFT methods, not invented here
- Episodic few-shot training, prototype networks (standard since ~2017)
- Freezing a CNN backbone for transfer learning

**What nobody had tested — the actual gap, checked against real papers, not assumed:**

| Literature | Reports | Doesn't report |
|---|---|---|
| Few-shot meta-learning (ProtoNet, MAML...) | Accuracy | Calibration, OOD, parameter budget |
| PEFT for ViTs (VPT, SSF, Adapter...) | Accuracy, param budget | Calibration, OOD, **not few-shot at all** |
| PEFT for CNNs on edge (Conv-Adapter — secondary RQ1 precedent — LoRA-C, LoRA-Edge) | Accuracy, param budget | Calibration, OOD |
| Frozen-CNN episodic few-shot adapters (**TSA — primary RQ1 precedent, same backbone/protocol/head** — FiT) | Accuracy, param budget | Calibration, OOD |
| Frozen-backbone + adapter few-shot (**Tip-Adapter, CLIP-Adapter** — closest for the general recipe) | Accuracy | Calibration, OOD, **disjoint-class episodic testing** (same classes train/test) |
| Bayesian PEFT (Laplace-LoRA, BLoB, BaRA... **BayesAdapter** — 2nd confirming case for RQ2 boundary) | Calibration | **All LLMs except BayesAdapter (CLIP, non-edge) — none edge** |
| Evidential few-shot (BEL — closest prior work) | Accuracy, calibration | OOD, param budget; **meta-trains the backbone** (we freeze it) |
| TinyML/edge surveys | Efficiency | **No mention of uncertainty at all** |

Two of those "doesn't report" cells are verified, not guessed: the ViT-PEFT survey covering the
entire field (arXiv:2402.02242) never mentions calibration/UQ/OOD; the TinyML survey
(arXiv:2506.18927) covers quantization extensively and never mentions uncertainty. Two unrelated
research communities, same blind spot — and that's the cell this thesis sits in.

**The Tip-Adapter/CLIP-Adapter row was checked by reading both papers in full, specifically
because the general recipe (frozen backbone + small adapter + few-shot) is close enough that it
needed verifying, not assuming.** Confirmed differences, not guesses: both train and test on the
*same* classes (Tip-Adapter's own text contrasts itself with meta-learning protocols that split
into disjoint category subsets); both report one fixed-test-set accuracy, not an average over
sampled episodes; neither reports ECE/Brier/OOD-AUROC; and their adapters are 17× to
multiple-orders-of-magnitude larger than this thesis's 6,928–31,746 range. The frozen+adapter
*idea* is shared — the disjoint-class episodic protocol, the parameter scale, and the
calibration/OOD measurement are not. See `docs/DEFENCE_BRIEF.md` §5.3 for the full writeup.

**TSA (Task-Specific Adapters, arXiv:2107.00358, CVPR 2022) is a closer match than the earlier
Conv-Adapter finding and now leads §5.2 in `docs/DEFENCE_BRIEF.md`.** Verified from the full
text: frozen ResNet-18, serial-vs-parallel adapter comparison, 600 sampled episodic tasks on
held-out domains, parameter-free nearest-centroid head — same backbone, same episode count, same
question, same winning answer as Step 6. Its adapters (175K–1.22M params) are far above this
thesis's range, and it never compares LoRA specifically or reports calibration/OOD, but the
placement principle itself is not new — say this plainly, don't lead with Conv-Adapter anymore.

**BayesAdapter (arXiv:2412.09718, IJCV 2025–26)** is a second independent paper — different
Bayesian mechanism (variational Bayes over linear weights, not evidential Dirichlet) — that finds
calibration *improving* on a frozen CLIP backbone with more shots/capacity than this thesis's
grid. That's now two confirming contrasts for the RQ2 boundary claim (with BEL), not one.

**Why "just testing a combination" is a real contribution, not busywork:** because the result
wasn't predictable from prior work. Ranked by strength, strongest first — lead with #1 if asked
what's novel, don't reach for #4 first:
1. BEL and BayesAdapter (two different Bayesian mechanisms, two independent papers) both found
   calibration *improving*; we found it *degrading*, 0/20. The reason (frozen + extreme-low-budget
   vs. more capacity/data) is itself a new finding about where Bayesian calibration works and
   where it breaks. This is the single strongest claim in the thesis, and it's now triangulated
   from two directions, not one.
2. The same evidential head that's 5–51× worse-calibrated is simultaneously the *better* OOD
   detector 94% of the time — calibration quality and OOD-ranking quality are empirically
   decoupled, and this is only visible because both were measured on the same runs. Nobody in the
   surveyed literature reports both together.
3. Step 4.5 suggested evidential ≈ energy for OOD; the full grid overturned that at scale — energy
   wins ~70% of the time. We caught our own earlier claim being wrong when we scaled it up.
4. RQ1's placement/parameter-efficiency results (below) — real, useful, but the weakest of the four
   on novelty specifically, since the core placement principle replicates TSA (and, separately,
   Conv-Adapter for the locality angle).

**Be upfront about the one place where the finding does NOT contradict prior work — RQ1's
placement result.** TSA (Li, Liu & Bilen, arXiv:2107.00358, CVPR 2022) already ran this exact
comparison — frozen ResNet-18, serial vs. parallel adapter connection, 600 sampled episodic tasks,
parameter-free nearest-centroid head — and found parallel wins, same as Step 6. Separately,
Conv-Adapter (arXiv:2208.07463) found locality-preserving adapters beat 1×1/linear ones; **don't
equate this with the LoRA result** — a 1×1 convolution (our LoRA target) still preserves spatial
locality, so it isn't the same named failure mode, just a separate corroborating data point. **Say
this plainly if RQ1 comes up: the placement principle is a replication, and TSA — not
Conv-Adapter — is the closest match on backbone, protocol, and head design. What's new is a far
smaller parameter budget (down to 2, vs. TSA's 175K–1.22M), a second backbone, the LoRA-specific
comparison, and calibration/OOD numbers neither reports.** Both `docs/RESULTS_MASTER.md` §3 RQ1 /
§4.7 claim 7 (claims are now ordered by defensibility, not RQ number — this one is last,
deliberately) and `docs/DEFENCE_BRIEF.md` §5.2 now cite this directly —
check there if asked for the receipts.

**One line for the room:**
> "Every ingredient is prior work. What's new is running them together in the one regime nobody
> had — a frozen sub-3M-parameter CNN with a sub-10K-parameter adapter, measuring accuracy,
> calibration, and OOD together instead of just one. Two of my four answers actually contradict
> what the nearest prior work would have predicted."

**If pushed further — "CNNs are outdated, why not a transformer?"** (full answer in
`docs/DEFENCE_BRIEF.md`): the deployment target is MCU-class edge hardware (32–512kB SRAM), where
an 86M-parameter ViT doesn't fit under any quantization scheme — a 2025 survey measures ~180ms for
a memory-optimized transformer attention block on an STM32F746 vs. ~8–12ms for CNN inference on
the same class of device. A separate Oct-2025 study found CNNs match ViTs specifically in
low-data regimes — few-shot *is* the low-data regime. And on the same 5-way 5-shot protocol, the
31,744-parameter adapter is only 1.06pp behind a fully meta-trained DINO ViT-S while training 662×
fewer parameters, and is actually +3.56pp *ahead* of the backbone-matched DINO ResNet-50 at 788×
fewer — so the remaining gap is a ViT-vs-CNN gap, not a parameter-efficiency gap.

---

## 5. Weaknesses — volunteer these before you're asked

In the order a committee/supervisor would find them (from `docs/RESULTS_MASTER.md` §4.8 and
`docs/DEFENCE_BRIEF.md` §7):

1. **The ImageNet-pretraining confound.** Our backbones are ImageNet-pretrained, and
   MiniImageNet's "novel" test classes were literally in ImageNet's pretraining set — so those
   accuracy numbers are not from-scratch few-shot numbers in the sense the benchmark was designed
   to measure. This is documented, not hidden, but it's the single biggest thing a from-scratch
   control run would remove as an objection. **Highest-value unstarted work.**
2. **No transformer arm in our own grid.** We argue CNNs are the right edge choice using
   *published* ViT numbers, not a ViT we ran ourselves under our exact protocol. A ViT-Tiny/
   DeiT-Small arm would convert that argument into a measurement — most direct answer to the
   "CNNs are outdated" objection.
3. **Only 3 seeds — and effectively 1 for the two baselines.** Full-FT and Linear-Probe have zero
   seed variance *by construction*: Full-FT starts from fixed pretrained weights and Linear-Probe
   has no trainable parameters at all, so neither has any randomly-initialized tensor for the seed
   to perturb. Any claim resting on a small margin over those two rows should be read with that in
   mind (e.g. the MobileNet-matches-Full-FT claim is phrased as "matches," not "beats," because
   the +0.27pp margin is smaller than that cell's own seed spread).
4. **The energy score beats the Bayesian one in ~70% of comparisons** (RQ3). Volunteer this — it's
   already in the write-up, and a reviewer who finds it unprompted weighs it more heavily.
5. **One frozen recipe across all 120 cells.** LR, LoRA rank, KL weight, evidence affine — all
   VAL-tuned once (Step 4.5, one configuration) and carried unchanged across two backbones, two
   datasets, two shot regimes. Deliberate design choice so the grid is a controlled comparison, not
   40 independently-tuned numbers — but it means the grid answers "how do these axes compare under
   one fixed recipe," not "what's the best achievable number per cell."
6. **Evidential calibration has no post-hoc fix in this codebase.** Softmax gets temperature
   scaling; evidential has no equivalent, so the TS comparison is inherently favorable to softmax.
   Still the right comparison (TS is cheap and standard) but state the asymmetry.
7. **TinyImageNet OOD-set exclusion caveat.** A test-isolation check for TinyImageNet's
   class-exclusion was traced by code inspection but never confirmed by a logged exclusion count
   from a real run (`step_writeups/step9.txt` §5). All TinyImageNet near-OOD numbers carry this
   caveat.
8. **Scope was trimmed from the original proposal.** ConvNeXt-Nano backbone, CUB-200-2011, and the
   ISIC medical dataset were all in the original proposal and are now optional Step-12 backlog
   items, not run. If asked why: time budget — the core 2×2×2×4×2 grid (120 runs, ~36 wall-hours)
   was prioritized to get all four RQs a numerical answer over breadth across more datasets/
   backbones. This is a legitimate scope-vs-depth tradeoff to state plainly, not something to
   dance around.

---

## 6. Two real bugs found during Step 11 closeout — say this plainly if latency comes up

Worth mentioning proactively if RQ4/Step 11 comes up, because it shows the verification process
worked, not that the process is sloppy:

1. A crash in an optional bonus feature discarded a correct measurement before it could save
   (found and fixed 08-07/08-09).
2. A **silent** bug — no crash, no failing test — where three independent latency-selection
   functions across `scripts/pareto_plots.py`, `make_master_tables.py`, and
   `make_results_master.py` all picked this repo's own noisy dev-laptop latency instead of the
   canonical Kaggle CPU number, by a dict-insertion-order accident. Errors up to 47% on individual
   cells, caught only by a manual cross-check. Fixed, and every downstream artifact (Table 8,
   `pareto_frontier.json`, all six Pareto PNGs) was regenerated from corrected data. **This bug
   class doesn't have a regression test yet — flagged as the top open follow-up.**

---

## 7. Anticipated questions — Q&A bank

**Q: What's the single biggest result?**
A: A 6,928-parameter adapter on a frozen MobileNetV3-Small matches full fine-tuning of an
11.18M-parameter ResNet-18 on CIFAR-FS 5-shot — 0.06% of the parameter budget for the same
accuracy. That's RQ1, and it's the cleanest result in the thesis (16/16, no exceptions).

**Q: Isn't "parallel beats sequential/LoRA for CNN adapters" already known?**
A: Yes — and the closest match is even closer than we first found. TSA (arXiv:2107.00358, CVPR
2022) already ran this exact comparison on a frozen ResNet-18, 600 sampled episodic tasks, a
parameter-free nearest-centroid head, and found parallel wins — same backbone, same protocol, same
head design, same answer as our Step 6. Conv-Adapter is a secondary precedent for a narrower point
(locality-preserving adapters beat 1×1 ones). RQ1's contribution isn't discovering the placement
principle, it's confirming it survives a far smaller parameter budget than TSA tested (down to 2,
vs. TSA's 175K–1.22M), a second backbone, the LoRA-specific comparison, and an evidential-
uncertainty layer with calibration/OOD reported — none of which TSA or Conv-Adapter do. Volunteer
this before it's asked; it's now cited in `RESULTS_MASTER.md` and `DEFENCE_BRIEF.md` §5.2.

**Q: Isn't "freeze a backbone, add a small adapter, do few-shot" already Tip-Adapter/CLIP-Adapter?**
A: The general recipe, yes — and that needed checking rather than assuming, so both papers were
read in full rather than name-dropped. Three verified differences: (1) they train and test on the
*same* classes — Tip-Adapter's own paper contrasts itself with meta-learning protocols that split
into disjoint category subsets, and instead evaluates on one fixed test set per dataset, not
sampled episodes with held-out classes; (2) neither reports ECE, Brier, or OOD-detection AUROC;
(3) their adapters are 17× to orders-of-magnitude larger than this thesis's 6,928–31,746 range.
The idea is shared, the disjoint-class episodic protocol and the parameter scale are not — full
writeup in `docs/DEFENCE_BRIEF.md` §5.3.

**Q: Did the central hypothesis (evidential = better calibration) work?**
A: No — cleanly, 0/20 matched comparisons, and it's not a tuning artifact (a VAL sweep already
showed the ECE surface is flat). But it's a well-powered negative result across 2 datasets × 2
shots × 2 backbones × 4 adapters, it's independently supported by 2026 theory, and it's honestly
more interesting than a "yes" would have been, because it locates a boundary — frozen backbone,
extreme-low parameter budget — that two independent prior papers (BEL's evidential Dirichlet, and
BayesAdapter's variational Bayes, arXiv:2412.09718) didn't test and both found the opposite result
outside of.

**Q: So was the Bayesian machinery worth building at all?**
A: For OOD detection, yes — it beats every softmax-probability-based score 94% of the time, and
the advantage grows as data shrinks (exactly the RQ3 prediction). For calibration, no. And a free
energy score beats it on OOD 70% of the time. The honest framing is: the Bayesian head is a better
*ranker* of confidence, not a better *calibrated* one, and it's not the unconditionally best OOD
score available.

**Q: How confident are you in the 120-run grid — could this be noise?**
A: Seed spread (3 seeds per cell) exceeds 1pp in only 2 of 40 configurations; everywhere else it's
≤0.74pp. Differences of >2pp between cells are real, not seed noise. The two baseline rows
(Full-FT, Linear-Probe) have zero seed variance by construction (see §5.3 above) — that's the one
place to be careful about "n=3."

**Q: What would you do with one more month?**
A: In priority order: (1) the from-scratch backbone control run — removes the ImageNet-pretraining
confound entirely; (2) a ViT-Tiny/DeiT-Small arm under our own protocol — converts the "CNNs are
the right edge choice" argument from literature-citation into a measurement; (3) a regression test
for the silent latency-selection bug class found in Step 11.

**Q: Why didn't you run CUB-200 / ISIC / ConvNeXt-Nano like the proposal said?**
A: Time budget. The 120-run core grid (all 4 RQs × 2 datasets × 2 shots × 2 backbones × 4 adapters
× 2 heads × 3 seeds, ~36 wall-hours) was prioritized over proposal breadth so every RQ would have a
numerical answer. Those three are explicit optional Step-12 backlog items, not dropped silently.

**Q: Is the code/pipeline solid, or are these one-off scripts?**
A: Reproducible by design: fixed episode seeds, `set_seed()` covering python/numpy/torch, and every
eval run on the same config is required to produce byte-identical `metrics.json`. The grid's
seed-42 cells were spot-checked against Step 6's earlier committed metrics and reproduced exactly
(29/29 and 55/55 numeric keys). Two real bugs *were* found late (§6), which is why there's now an
open action item to add regression coverage for the bug class that produces no error/crash at all.

**Q: What's next — more experiments or start writing?**
A: All four RQs have numerical answers as of Step 11 closing 2026-08-09. `progress.txt` frames the
choice explicitly as Step 12 (optional breadth) vs. Phase 6 (thesis writing) — this is the actual
decision for this meeting.

---

## 8. Cheat sheet — numbers to have on hand

| Fact | Number |
|---|---|
| Total runs | 120 (40 configs × 3 seeds) |
| Test episodes per run | 600 (frozen seeds 0–599) |
| Smallest adapter tested | 6,928 params (MobileNetV3-S, CIFAR-FS) |
| Adapter matching full-FT accuracy | 6,928 params vs. 11,176,512 (0.06%) |
| Adapter beating full-FT accuracy | 31,744 params, +0.98pp (CIFAR-FS 5-shot) |
| RQ1 win rate | 16/16 parallel-bottleneck > LoRA |
| RQ2 result | 0/20 evidential ECE beats softmax ECE |
| RQ2 worst-case ECE gap | 51× worse than TS-softmax |
| RQ3 vs. MSP / TS-MSP | 93.8% / 95.0% win rate |
| RQ3 vs. energy | 30% win rate (evidential loses ~70%) |
| RQ4 recommended point | MobileNetV3-S + parallel + evidential, 11.86ms/img, 6,930 params |
| Backbone vs. adapter latency effect | 5.12× vs. 3.9–5.8% |
| Evidential inference overhead | 1.29% (below 5.91% noise floor) |

---

## 9. Source map — if your supervisor wants to see the receipts

| Question | File |
|---|---|
| Full results tables (all 120 runs) | `docs/RESULTS_MASTER.md` |
| Step-by-step status, decisions log | `progress.txt` |
| "CNNs are outdated" objection, full answer | `docs/DEFENCE_BRIEF.md` |
| RQ4 / latency / bug postmortem detail | `step_writeups/step11.txt` |
| RQ2/RQ3 grid detail | `step_writeups/step10.txt` |
| Original scope | `proposal.txt` |
