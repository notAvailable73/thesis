# Pre-Defence Brief — "Isn't a CNN backbone backdated in 2026?"

**B-PEFT: Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision with Lightweight CNN Backbones**

This document answers the objection raised at the pre-defence: *that the thesis works on outdated technology because nobody uses CNN backbones in 2025–26.* It is built from external sources verified during this session (every claim is linked) and from this project's own measured results (`results/mvt_results.json`, 120 runs — see [RESULTS_MASTER.md](RESULTS_MASTER.md)).

**Reading order for a 5-minute version:** §1 (the one-paragraph answer), §3.1–3.2 (accuracy and macro-F1 vs. parameters — the trade), §6 (the three sentences to say out loud).

> **Honesty note carried from the rest of this thesis.** Parameter efficiency is a *trade*, so this document shows what it costs in accuracy and macro-F1, not just what it saves. Where a comparison is not apples-to-apples, it says so rather than presenting the favourable number. Three things to keep straight:
> - **§3.1 is a fair accuracy comparison.** P>M>F runs the same 5-way few-shot episodic protocol on the same two datasets, so its accuracies and ours belong in the same table.
> - **§3.3 is not.** VTAB-1k gives 1,000 labelled examples per task against our 25. Its accuracies appear only so the parameter budgets aren't floating free, and must never be read against §3.1.
> - **Our own accuracies carry the ImageNet-pretraining caveat** documented in [RESULTS_MASTER.md §4.1](RESULTS_MASTER.md) — as do P>M>F's, which is why the two sit together honestly.

---

## 1. The one-paragraph answer

The objection conflates two different claims: *"transformers are the best-performing vision backbones"* (true, and this thesis does not dispute it) and *"CNN backbones are obsolete"* (false, and the 2025–26 literature says so explicitly). CNNs remain the deployed architecture on edge and MCU-class hardware because transformers do not fit there — a July 2025 survey measures a memory-optimised transformer attention block at **~180 ms** per inference on an STM32F746 against **~8–12 ms** for CNN inference on the same class of device, and reports that CNNs "dominate TinyDL". More directly relevant to this thesis: an October 2025 comparative study finds that **in low-data regimes CNNs match ViTs even when the ViTs were pretrained on large-scale datasets**, because the convolutional inductive bias substitutes for data the transformer does not have. Few-shot learning *is* the low-data regime. And the trade is quantified rather than asserted: on the *same* 5-way 5-shot protocol our 31,744-parameter adapter is **1.06 pp behind a fully meta-trained DINO ViT-S while training 662× fewer parameters**, and it is **+3.56 pp ahead** of the backbone-matched DINO>ProtoNet ResNet-50 at 788× fewer — so the remaining gap is a ViT gap, not a parameter-efficiency gap. It costs more at 1-shot and on MiniImageNet, and §3.2 states exactly how much. Meanwhile the research question the thesis asks (does Bayesian uncertainty survive parameter-efficient adaptation?) is one of the most active topics of 2025–26, with at least six papers in the last twelve months — all of them on language models, none on vision at the edge.

---

## 2. What the objection gets right, and where it stops being true

| The claim | Status | Evidence |
|---|---|---|
| "Transformers are the strongest vision backbones at scale" | **True — not disputed** | DINOv3 (2025) trains a ~7B-parameter ViT on ~1.7B images and leads dense-prediction benchmarks with frozen features ([arXiv:2508.10104](https://arxiv.org/html/2508.10104v1)) |
| "Therefore CNN backbones are obsolete" | **False** | CNNs "remain the preferred choice" for edge/real-time deployment; deployed vision systems (YOLOv5/v8, Faster R-CNN, Mask R-CNN) "continue to rely entirely on convolutional architectures" — *Scientific Reports*, **27 Nov 2025** ([DOI 10.1038/s41598-025-27856-3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12728162/)) |
| "Therefore CNNs are the wrong choice for few-shot work" | **False, and backwards** | "In small data scenarios, the inductive bias and smaller capacity of CNNs improve their performance, allowing them to match that of a ViT… CNNs achieve comparable performance in low-data regimes even when the ViTs were pretrained on large-scale datasets" — **Oct 2025** ([arXiv:2510.04794](https://arxiv.org/html/2510.04794v1)) |
| "Therefore this research question is dated" | **False** | The identical question — *does Bayesian/uncertainty machinery survive parameter-efficient adaptation?* — has ≥6 papers in 2025–26 (§5). None of them are vision-at-the-edge. That is the gap, not a dated topic. |

**The deployment reality (July 2025 TinyML→TinyDL survey, [arXiv:2506.18927](https://arxiv.org/html/2506.18927v2)):** MCU-class targets have **32–512 kB SRAM**, **under ~1 MB flash**, **20–200 MHz** clocks, frequently **no floating-point unit**, and inference budgets **under 1 mW**. The survey reports CNNs dominating this space via MobileNet/SqueezeNet/MCUNet, and records that transformer attention remains the bottleneck: even a memory-optimised Fused-Weight Self-Attention achieving a 6.19× peak-memory reduction still costs **~180 ms** on an STM32F746, versus **~8–12 ms** for CNN inference.

A ViT-B/16 at 86 M parameters does not fit in 1 MB of flash under any quantisation scheme. That is not a preference; it is arithmetic.

---

## 3. The parameter argument — the core of the defence

### 3.1 The trade, on the same task: accuracy and macro-F1 vs. parameters

**This is the table to lead with.** P>M>F is evaluated on 5-way few-shot episodes on CIFAR-FS and MiniImageNet — the *same task as ours* — so accuracy is directly comparable here. It differs in backbone and in what it trains: P>M>F meta-trains and fine-tunes the whole backbone; we freeze it and train an adapter.

P>M>F figures from [arXiv:2204.07305](https://ar5iv.labs.arxiv.org/html/2204.07305) Table 4; ViT-S/16 = 21 M, ViT-B/16 = 86 M per the DINO model cards; ours from `results/mvt_results.json`.

| Method | Backbone | Trainable | CIFAR-FS 5-shot | MiniIN 5-shot | CIFAR-FS 1-shot | MiniIN 1-shot | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|:---:|
| Sup-21k > ProtoNet | ViT-B/16 | ~85.8 M | **96.7** | **99.2** | 92.3 | 97.2 | not reported |
| CLIP > ProtoNet | ViT-B/16 | ~86.5 M | 93.2 | 98.1 | 85.3 | 93.1 | not reported |
| DINO > ProtoNet | ViT-B/16 | ~85.8 M | 92.2 | 98.4 | 84.3 | 95.3 | not reported |
| DINO > ProtoNet | ViT-S/16 | ~21 M | 92.5 | 98.0 | 81.1 | 93.1 | not reported |
| DINO > ProtoNet | ResNet-50 | ~25 M | — | 92.0 | — | 79.2 | not reported |
| BEL (evidential few-shot) | ResNet-12 | not reported | 86.92 | 79.60 | 73.96 | 63.10 | not reported |
| MetaOptNet *(from-scratch)* | ResNet-12 | not reported | 84.2 | 78.63 | 72.0 | 62.64 | not reported |
| **Ours**, parallel bottleneck | **ResNet-18, frozen** | **31,744** | **91.44** | **95.56** | 78.57 | 85.03 | **91.29 / 95.53 / 77.50 / 84.34** |
| **Ours**, parallel bottleneck | **MobileNetV3-S, frozen** | **6,928** | **90.74** | 90.10 | 78.80 | 74.92 | **90.59 / 89.97 / 77.69 / 73.71** |

*(Our macro-F1 column is in the same column order as the accuracies. **No competing method reports macro-F1 at all.**)*

### 3.2 What the parameter saving actually costs — say this before you are asked

| Ours vs. | Param saving | CIFAR-FS 5-shot | MiniIN 5-shot | CIFAR-FS 1-shot | MiniIN 1-shot |
|---|---:|---:|---:|---:|---:|
| ResNet-18 adapter vs DINO>PN **ViT-S** | **662× fewer** | **−1.06 pp** | −2.44 pp | −2.53 pp | −8.07 pp |
| ResNet-18 adapter vs DINO>PN **ViT-B** | **2,703× fewer** | **−0.76 pp** | −2.84 pp | −5.73 pp | −10.27 pp |
| ResNet-18 adapter vs CLIP>PN ViT-B | 2,725× fewer | −1.76 pp | −2.54 pp | −6.73 pp | −8.07 pp |
| ResNet-18 adapter vs Sup-21k>PN ViT-B *(best published)* | 2,703× fewer | −5.26 pp | −3.64 pp | −13.73 pp | −12.17 pp |
| ResNet-18 adapter vs DINO>PN **ResNet-50** | 788× fewer | — | **+3.56 pp** | — | **+5.83 pp** |
| MobileNetV3-S adapter vs DINO>PN ViT-S | **3,031× fewer** | −1.76 pp | −7.90 pp | −2.30 pp | −18.18 pp |

**Three separate conclusions — do not blur them into one:**

1. **At 5-shot on CIFAR-FS the saving is close to free.** 31,744 parameters, **1.06 pp** behind a fully meta-trained DINO ViT-S at **662×** fewer trainable parameters; **0.76 pp** behind ViT-B at **2,703×** fewer. This is the strongest trade in the thesis.
2. **Backbone-family-matched, we simply win.** Against DINO>ProtoNet on ResNet-50 we are **+3.56 pp** (MiniIN 5-shot) and **+5.83 pp** (1-shot) *ahead*, at 788× fewer trainable parameters. The ViT gap is a ViT gap, not a parameter-efficiency gap.
3. **The trade is genuinely bad at 1-shot and on MiniImageNet with the small backbone** — up to **−18.2 pp**. Volunteer this. The 6,928-parameter cell is competitive *on CIFAR-FS* (−1.76 pp at 3,031× fewer parameters) and is not on MiniImageNet, whose classes come from ImageNet so a larger pretrained representation carries more of the answer (§7, item 2).

### 3.3 Trainable parameters across PEFT methods (different protocol — parameters only)

The ViT numbers are the SSF paper's VTAB-1k table on ViT-B/16 (~86 M total), verified at [arXiv:2210.08823](https://ar5iv.labs.arxiv.org/html/2210.08823). **VTAB-1k gives 1,000 labelled examples per task; our protocol gives 25. Their accuracies are shown so the budgets aren't floating free, but they are NOT comparable to §3.1 and must never be tabulated against it.**

| Method | Backbone | Trainable params | VTAB-1k avg acc | vs. our MobileNetV3-S | vs. our ResNet-18 |
|---|---|---:|---:|---:|---:|
| Full fine-tuning | ViT-B/16 | 85,840,000 | 65.57 | **12,390× more** | **2,704× more** |
| Full fine-tuning *(our measured baseline)* | ResNet-18 | 11,176,512 | *(few-shot: 90.47)* | **1,613× more** | 352× more |
| VPT-Deep | ViT-B/16 | 600,000 | 69.43 | **86.6× more** | 18.9× more |
| Adapter | ViT-B/16 | 270,000 | 55.82 | **39.0× more** | 8.5× more |
| SSF | ViT-B/16 | 240,000 | **73.10** | **34.6× more** | 7.6× more |
| VPT-Shallow | ViT-B/16 | 110,000 | 64.85 | 15.9× more | 3.5× more |
| Linear probing | ViT-B/16 | 40,000 | 52.94 | 5.8× more | 1.3× more |
| CoOp | CLIP | 8,192 | *(not on VTAB-1k)* | 1.2× more | 0.3× (smaller) |
| **B-PEFT (ours)** | **ResNet-18** | **31,744** | *(few-shot: §3.1)* | — | — |
| **B-PEFT (ours)** | **MobileNetV3-Small** | **6,928** | *(few-shot: §3.1)* | — | — |

**The honest exception, and why it helps us.** CoOp is the one method in our trainable-parameter league (8,192 vs. our 6,928), and it beats our ResNet-18 configuration on the raw count. But CoOp's context vectors steer a *frozen CLIP* that must be resident at inference — the trainable budget is small while the **deployed system is 34.6× larger** (§3.4). CoOp also reports no calibration and no OOD detection ([verified at arXiv:2109.01134](https://ar5iv.labs.arxiv.org/html/2109.01134)). Don't dispute "CoOp is smaller" — point at the next table.

### 3.4 Deployed parameters: what has to be on the device

Trainable parameters are the *adaptation* cost. The *deployment* cost is the whole frozen model, and this is where the CNN choice is doing the work. Backbone sizes from the Oct 2025 comparative study's Table 1 ([arXiv:2510.04794](https://arxiv.org/html/2510.04794v1)); MobileNetV3-Small's 2.5 M from the MobileNetV3 paper as recorded in this repo's own `PAPER SUMMARIES/CNN_paper_summaries.txt`.

| Backbone at inference | Parameters | vs. our MobileNetV3-Small |
|---|---:|---:|
| DINOv3 ViT-7B | 7,000,000,000 | **2,800× larger** |
| CLIP-ViT-B/32 | 88,200,000 | 35.3× larger |
| CLIP-ViT-B/16 | 86,500,000 | **34.6× larger** |
| DINO-ViT-B/16 | 85,800,000 | 34.3× larger |
| CLIP-ResNet-101 | 56,400,000 | 22.6× larger |
| DINO-ViT-S/16 | 21,000,000 | 8.4× larger |
| ResNet-18 *(ours)* | ~11,700,000 | 4.7× larger |
| **MobileNetV3-Small *(ours)*** | **2,500,000** | — |

**The combined statement for the defence:** our deployed system is a **2.5 M-parameter frozen backbone plus 6,928 adapted parameters**, reaching **90.74 %** on CIFAR-FS 5-shot — **1.76 pp** behind a fully meta-trained DINO ViT-S that is **8.4× larger at inference** and trains **3,031× more parameters**.

### 3.5 What our parameter budget buys (measured, not claimed)

From [RESULTS_MASTER.md](RESULTS_MASTER.md), CIFAR-FS 5-way 5-shot:

| Configuration | Trainable params | Accuracy |
|---|---:|---:|
| Parallel bottleneck, ResNet-18 | 31,744 | **91.44 %** |
| Parallel bottleneck, MobileNetV3-Small | **6,928** | 90.74 % |
| Full fine-tuning, ResNet-18 | 11,176,512 | 90.47 % |
| Linear probe (no adaptation) | 0 | 87.41 % |

**6,928 trainable parameters reach the accuracy of retraining 11.18 million** — 0.06 % of the budget. That is the number to put on a slide.

---

## 4. The research question is not dated — it is one of 2026's most active

Between mid-2024 and mid-2026 the field has produced a steady stream of work on **exactly** this thesis's question: *does uncertainty quantification survive parameter-efficient adaptation, and does it stay calibrated?*

| Work | Year | What it does | Domain |
|---|---|---|---|
| [Laplace-LoRA](https://proceedings.iclr.cc/paper_files/paper/2024/file/07c256a163a7559186ec1c71e95b9ec9-Paper-Conference.pdf) (ICLR 2024) | 2024 | Post-hoc Laplace approximation over LoRA parameters; cuts ECE from 31.2 % → 2.1 % on Winogrande-small (LLaMA2-7B) at 1–5 % memory overhead | LLM |
| [BLoB](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf) (NeurIPS 2024) | 2024 | Bayesian LoRA by backpropagation | LLM |
| [LoRA-Ensemble](https://arxiv.org/html/2405.14438v5) | 2024–25 | Parameter-efficient ensembling for uncertainty in self-attention networks | **ViT** |
| [Scalable Bayesian LoRA via stochastic variational subspace inference](https://arxiv.org/pdf/2506.21408) | 2025 | Variational Bayesian LoRA at scale | LLM |
| [Calibrated Adaptation: Bayesian Stiefel Manifold Priors for Reliable PEFT](https://arxiv.org/html/2602.17809) | 2026 | Geometry-aware Bayesian prior on the Stiefel manifold; <8 % wall-clock overhead over deterministic LoRA | LLM |
| [BaRA: Bayesian Adaptive Rank Allocation for PEFT](https://arxiv.org/pdf/2606.29184) | 2026 | Bayesian rank allocation | LLM |
| [Bayesian Sparse LoRA for LLM Uncertainty Estimation](https://arxiv.org/html/2607.02182v1) | 2026 | Sparse Bayesian posterior over LoRA | LLM |
| [Bayesian Adaptation Gym](https://arxiv.org/pdf/2606.22188) | 2026 | A *benchmark* for Bayesian low-rank adaptation — the field now has enough entries to need one | Multi-modal LM |
| **B-PEFT (this thesis)** | 2026 | Evidential Dirichlet uncertainty over a parameter-free prototype head on a **frozen lightweight CNN**, ≤31.7 k trainable params, few-shot episodic, with calibration **and** OOD **and** parameter budget all reported | **Vision, edge** |

**The point to make in the room:** a *benchmark paper* for Bayesian PEFT appeared in 2026. Fields do not build benchmarks for dated questions. This thesis is asking the field's current question — in the one setting the field has not covered.

### 4.1 A 2026 theory result that independently supports our headline finding

Our RQ2 answer is negative: the evidential head is worse-calibrated than softmax in **20 out of 20** matched comparisons ([RESULTS_MASTER.md §3](RESULTS_MASTER.md)). That is not an isolated or contrarian result. A 2026 theoretical analysis of second-order/evidential classification reports that standard reverse-KL EDL objectives yield **non-vanishing epistemic uncertainty even in the limit of infinite data** ([arXiv:2606.10777](https://arxiv.org/pdf/2606.10777)), and the 2024 EDL survey ([arXiv:2409.04720](https://arxiv.org/pdf/2409.04720)) documents the KL-regulariser design this depends on.

Our empirical result and the 2026 theory point the same way. Present the negative result as *convergent evidence*, not as a failed experiment.

---

## 5. The gap — what nobody reports, across every backbone family

This is the strongest single slide, because each row was checked against the source rather than assumed.

| Literature | Representative work | Accuracy | Macro-F1 | Calibration (ECE) | OOD AUROC | Param budget | Few-shot episodic | Edge-deployable backbone |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical few-shot meta-learning | ProtoNet, MAML, R2D2, MetaOptNet | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Foundation-model few-shot | P>M>F, CoOp, Tip-Adapter, DINOv3 | ✅ | ❌ | ❌ | ❌ | partial | ✅ | ❌ |
| PEFT for vision transformers | VPT, AdaptFormer, SSF, FacT, NOAH | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PEFT for CNNs on edge | LoRA-C, LoRA-Edge, CoLoRA | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Bayesian PEFT | Laplace-LoRA, BLoB, BaRA, Stiefel-Bayes | ✅ | ❌ | ✅ | partial | ✅ | ❌ | ❌ |
| Evidential few-shot | [BEL (arXiv:2207.13137)](https://ar5iv.labs.arxiv.org/html/2207.13137) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | partial |
| TinyML / TinyDL | [TinyDL survey (arXiv:2506.18927)](https://arxiv.org/html/2506.18927v2) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **B-PEFT (this thesis)** | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Two of those ❌ columns are documented absences, not inferences:**

- The **PEFT-for-vision survey** ([arXiv:2402.02242](https://arxiv.org/html/2402.02242v1)) — a survey of the entire ViT PEFT field — **does not discuss calibration, uncertainty quantification, or OOD detection at all.**
- The **TinyML→TinyDL survey** ([arXiv:2506.18927](https://arxiv.org/html/2506.18927v2), 2025) covers quantisation extensively and makes **no mention of uncertainty estimation, calibration, or OOD detection.**

Two independent surveys, two different communities, the same blind spot. That blind spot is this thesis.

### 5.1 The closest prior work, and how ours differs

**Bayesian Evidential Learning for Few-Shot Classification** ([arXiv:2207.13137](https://ar5iv.labs.arxiv.org/html/2207.13137)) is the nearest neighbour: evidential Dirichlet uncertainty on few-shot episodes, ResNet-12/Conv-4 backbones, miniImageNet 63.10 / 79.60 and CIFAR-FS 73.96 / 86.92 (1-/5-shot), and it **does** report ECE — improving it (3.59 % vs 14.69 % baseline, miniImageNet 5-shot).

**It reports no OOD AUROC and no parameter counts, and its backbone is meta-trained rather than frozen.**

That difference is the thesis's contribution, and it must be stated carefully because BEL's calibration result points the *opposite* way to ours:

> BEL shows evidential uncertainty improving calibration when the backbone is meta-trained and two networks' evidence is fused. We show it **degrading** calibration when the backbone is frozen and the entire trainable budget is ≤31,744 parameters — including a configuration with just **2**. Both can be true: they are different regimes. Establishing *where the boundary lies* is a contribution, and it is the regime that matters for edge deployment, because meta-training a backbone is exactly what a 256 kB device cannot do.

If a reviewer raises BEL, this is the answer — and it converts an apparent contradiction into the thesis's actual finding.

---

## 6. What to say out loud

**On the backbone choice:**
> "We're not claiming CNNs beat transformers at scale — they don't. We're claiming the deployment target we chose, MCU-class edge hardware with 32–512 kB of SRAM, cannot run an 86-million-parameter ViT, and the 2025 literature agrees: a memory-optimised transformer attention block still costs 180 ms on an STM32F746 against 8–12 ms for a CNN. We also have an October 2025 comparative study finding CNNs match ViTs in low-data regimes specifically. Few-shot *is* the low-data regime."

**On parameter efficiency — and what it costs:**
> "Our adapter trains 6,928 parameters. The standard ViT PEFT methods train 240,000 to 600,000 — 35 to 87 times more. Fine-tuning a ViT-B/16 trains 85.84 million, twelve thousand times more. And our 6,928 parameters reach the accuracy of fully retraining an 11.18-million-parameter network."

**On the accuracy trade (have this ready — it will be the first follow-up):**
> "On the same 5-way 5-shot protocol, our 31,744-parameter adapter is 1.06 points behind a fully meta-trained DINO ViT-S on CIFAR-FS while training 662 times fewer parameters, and 0.76 points behind ViT-B at 2,703 times fewer. Against the backbone-matched comparison — DINO ProtoNet on ResNet-50 — we're 3.56 points *ahead* on MiniImageNet 5-shot at 788 times fewer parameters. So the gap that remains is a ViT gap, not a parameter-efficiency gap. It does cost us at 1-shot and on MiniImageNet with the smallest backbone, up to 18 points, and that's in the table too."

**On whether the question is dated:**
> "The question of whether Bayesian uncertainty survives parameter-efficient adaptation produced at least six papers in the last twelve months, including a dedicated benchmark in 2026. All of them are on language models. Two separate surveys — one covering all of ViT PEFT, one covering all of TinyML — contain no discussion of calibration or OOD detection whatsoever. We're asking the field's current question in the one place the field hasn't looked."

**If pressed on our negative calibration result:**
> "It's a negative result across 20 out of 20 matched comparisons, two datasets, two shot regimes, two backbones and four adapters — and a 2026 theoretical analysis independently shows standard evidential objectives retain non-vanishing epistemic uncertainty even with infinite data. The same head is simultaneously a *better* OOD detector than max-softmax probability in 93.8 % of our comparisons. Calibration quality and OOD-ranking quality are decoupled, and you only see that if you measure both."

---

## 7. Weaknesses a committee will still find — prepare these

Do not let these be discovered rather than volunteered.

1. **No latency measurement yet.** The edge argument currently rests on parameter counts plus the literature's measurements, not our own hardware timings. This is Step 11 and it is the most exposed gap in the story.
2. **The ImageNet-pretraining confound.** Our accuracies are not comparable to from-scratch few-shot SOTA, and MiniImageNet's classes come from ImageNet. Documented in [RESULTS_MASTER.md §4.1](RESULTS_MASTER.md); the fix is a from-scratch control run.
3. **No transformer arm in our own grid.** We argue from published ViT PEFT numbers rather than running a ViT ourselves. A ViT-Tiny or DeiT-Small arm under our exact protocol would convert an argument into a measurement — and it is the single most direct answer to this objection.
4. **The energy score beats our Bayesian one** in ~70 % of comparisons ([RESULTS_MASTER.md §3](RESULTS_MASTER.md)). Volunteer this; it is in the write-up and a reviewer who finds it unprompted will weigh it more heavily.
5. **Protocol mismatch in §3.1.** VTAB-1k vs. episodic few-shot — parameter counts transfer, accuracies do not. Say it before you are asked.

**Highest-value work to do before the defence, in order:** (1) the ViT-Tiny/DeiT-Small arm under our protocol; (2) Step 11 latency on real hardware; (3) the from-scratch control.

---

## Sources

All verified during this session; each was fetched and read rather than recalled.

**Backbone / edge deployment**
- [From Tiny Machine Learning to Tiny Deep Learning: A Survey — arXiv:2506.18927 (2025)](https://arxiv.org/html/2506.18927v2)
- [Revisiting convolutional design for efficient CNN architectures in edge-aware applications — *Scientific Reports*, 27 Nov 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12728162/)
- [A Comparative Study of Vision Transformers and CNNs for Few-Shot… — arXiv:2510.04794 (Oct 2025)](https://arxiv.org/html/2510.04794v1)
- [DINOv3 — arXiv:2508.10104 (2025)](https://arxiv.org/html/2508.10104v1)

**PEFT parameter budgets**
- [SSF: Scaling & Shifting Your Features — arXiv:2210.08823](https://ar5iv.labs.arxiv.org/html/2210.08823) (VTAB-1k table, ViT-B/16)
- [PEFT for Pre-Trained Vision Models: A Survey — arXiv:2402.02242](https://arxiv.org/html/2402.02242v1) (no calibration/uncertainty/OOD coverage)
- [CoOp: Learning to Prompt for Vision-Language Models — arXiv:2109.01134](https://ar5iv.labs.arxiv.org/html/2109.01134)
- [LoRA-C — arXiv:2410.16954](https://arxiv.org/abs/2410.16954) · [LoRA-Edge — arXiv:2511.03765](https://arxiv.org/abs/2511.03765) · [CoLoRA — arXiv:2505.18315](https://arxiv.org/html/2505.18315)

**Bayesian / uncertainty-aware PEFT**
- [Laplace-LoRA, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/07c256a163a7559186ec1c71e95b9ec9-Paper-Conference.pdf) · [BLoB, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d53575463291ea6b5a23cf6e571f59b-Paper-Conference.pdf)
- [LoRA-Ensemble — arXiv:2405.14438](https://arxiv.org/html/2405.14438v5) · [Scalable Bayesian LoRA — arXiv:2506.21408](https://arxiv.org/pdf/2506.21408)
- [Calibrated Adaptation (Stiefel-Bayes) — arXiv:2602.17809](https://arxiv.org/html/2602.17809) · [BaRA — arXiv:2606.29184](https://arxiv.org/pdf/2606.29184) · [Bayesian Sparse LoRA — arXiv:2607.02182](https://arxiv.org/html/2607.02182v1) · [Bayesian Adaptation Gym — arXiv:2606.22188](https://arxiv.org/pdf/2606.22188)

**Evidential learning / few-shot uncertainty**
- [Bayesian Evidential Learning for Few-Shot Classification — arXiv:2207.13137](https://ar5iv.labs.arxiv.org/html/2207.13137)
- [A Comprehensive Survey on Evidential Deep Learning — arXiv:2409.04720](https://arxiv.org/pdf/2409.04720)
- [Epistemic calibration in second-order classification — arXiv:2606.10777 (2026)](https://arxiv.org/pdf/2606.10777)

**Few-shot benchmarks**
- [MetaOptNet, CVPR 2019 — arXiv:1904.03758](https://arxiv.org/abs/1904.03758) · [P>M>F, CVPR 2022 — arXiv:2204.07305](https://ar5iv.labs.arxiv.org/html/2204.07305)
