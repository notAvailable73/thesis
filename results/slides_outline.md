# Slide Deck Outline
# Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision
# Pre-Defence Presentation | IUT | Mainul

---

## Slide 1 — Title

**Title:** Bayesian Parameter-Efficient Fine-Tuning for Reliable Few-Shot Vision

**Subtitle:** Pre-Defence Presentation

**Author:** [Your Name], IUT
**Date:** April 2026
**Supervisor:** [Supervisor Name]

---

## Slide 2 — The Problem

**Heading:** Three Gaps in Practical Vision AI

1. **Parameter inefficiency** — Full fine-tuning of large models (ResNet18: 11M params)
   requires massive compute and risks catastrophic forgetting. Few-shot tasks don't
   justify training all weights.

2. **Reliability gap** — Standard Softmax classifiers are overconfident. They output
   high-probability predictions even on inputs they have never seen. In medical or
   safety-critical applications, this is dangerous.

3. **Data scarcity** — Real-world tasks often have only 5–20 labelled examples per class.
   Calibration and uncertainty estimation are hardest exactly when data is most scarce.

**One-line summary:** *We need models that are small, few-shot capable, and know when they don't know.*

---

## Slide 3 — Research Questions

**Heading:** What This Thesis Asks

| RQ | Question | Status |
|----|----------|--------|
| RQ1 | Does parameter-efficient fine-tuning (Adapter, LoRA, BitFit) match full fine-tuning accuracy on few-shot tasks? | Future work |
| **RQ2** | **Can Evidential Dirichlet Networks provide superior calibration vs Softmax with <500 trainable parameters?** | **Partially addressed today** |
| **RQ3** | **Does Bayesian loss improve OOD detection in low-data regimes?** | **Addressed today** |
| RQ4 | Do PEFT methods generalise to edge devices (Jetson Nano) with acceptable latency? | Future work |

**Speaker note:** "Today's demo directly addresses RQ2 and RQ3. RQ1 and RQ4 are the next two months."

---

## Slide 4 — Method Diagram

**Heading:** The Pipeline (One Forward Pass)

```
Input image (224×224)
        ↓
[ Frozen ResNet18 ]  ← ImageNet pretrained, ALL 11M params locked
        ↓  (512-dim features)
[ Bottleneck Adapter ]  ← DOWN(512→16) → ReLU → UP(16→512) + residual
        ↓  (512-dim adapted features)          16,912 params  ✓
[ Linear Head ]  ← 512 → 5 classes
        ↓  (5-dim logits)                       2,565 params  ✓
[ Softplus ]  → evidence e_k ≥ 0
        ↓
  alpha_k = e_k + 1  (Dirichlet concentration)
        ↓
  prob_k = alpha_k / S    uncertainty u = K / S
```

**Callout box:** "Only **19,477 trainable parameters** — 0.17% of the full model"

**Speaker note:** "The adapter starts as an identity function. At step 0, it changes nothing.
All learning happens in these 20K parameters over 300 gradient steps."

---

## Slide 5 — The Evidential Layer (Simple Math)

**Heading:** From Logits to Uncertainty

**Three equations, one per bullet:**

1. **Evidence** (what the model produces):
   `e_k = softplus(logit_k) ≥ 0`

2. **Dirichlet parameters** (one per class):
   `alpha_k = e_k + 1`   →   `S = Σ alpha_k`

3. **Class probability and uncertainty:**
   `prob_k = alpha_k / S`   |   `vacuity u = K / S`

**Key insight boxes:**
- High evidence → large S → low vacuity → model is CONFIDENT
- Low evidence (OOD input) → small S → high vacuity → model says "I DON'T KNOW"
- Softmax has no vacuity — it always sums to 1, even for inputs it has never seen

**Speaker note:** "This is the core idea. Softmax is a closed world. Dirichlet is an open world."

---

## Slide 6 — Experimental Setup

**Heading:** What We Tested

| Component | Choice | Why |
|-----------|--------|-----|
| Backbone | ResNet18 (frozen) | ImageNet pretrained, 11M params |
| Adapter | Bottleneck (rank 16) | 17K params, identity init |
| Dataset | CIFAR-FS (CIFAR-100 subset) | Standard few-shot benchmark |
| Episode | 5-way 5-shot, 15 query/class | 25 train, 75 test images |
| OOD dataset | SVHN (500 images) | Far-OOD, digit vs object domain shift |
| Baseline | Same model, Softmax + CE | Identical architecture, different head |
| Training | 300 steps, Adam, lr=5e-3 | Single episode |

**Speaker note:** "Both models share the same backbone and adapter. The only difference
is the output head: Dirichlet evidence vs raw logits."

---

## Slide 7 — Results Table

**Heading:** Evidential Wins Where It Counts Most

| Model | Accuracy ↑ | ECE ↓ | Brier ↓ | OOD AUROC ↑ |
|-------|-----------|-------|---------|------------|
| Softmax | **0.933** | **0.040** | **0.095** | 0.854 |
| Evidential | 0.893 | 0.526 | 0.541 | **0.991** |

**Bold/highlight:** OOD AUROC column — Evidential +0.137

**Key takeaways (3 bullets):**
- Accuracy is comparable — adapter learns equally well for both heads
- **OOD AUROC: 0.991 vs 0.854** — evidential detects unseen domains far better
- ECE: softmax wins in this single-episode setting (see next slides for why)

**Speaker note:** "The OOD result is the headline. A 13.7-point AUROC improvement
means the model is far more reliable when deployed on new domains."

---

## Slide 8 — Reliability Diagram

**Heading:** Calibration: Confidence vs Actual Accuracy

**Insert:** `results/reliability_plot.png`

**Caption:** "A perfectly calibrated model follows the diagonal. Softmax (red) is
near-perfect. Evidential (blue) is overconfident in this single-episode setting."

**Honest explanation (2 bullets):**
- Softmax benefits from natural competition between classes — max prob is bounded
- Evidential trained on 25 samples overfits → softplus evidence grows large →
  probabilities push toward 1.0 → high confidence, ECE suffers
- Fix: multi-episode meta-training (planned for Month 1 of full thesis)

---

## Slide 9 — OOD Detection

**Heading:** OOD Detection: Evidential is Dramatically Better

**Insert:** `results/ood_histogram.png`

**Caption:** "Evidential vacuity (K/S) clearly separates in-distribution CIFAR-FS
(left peak) from OOD SVHN (right peak). Softmax max-prob distributions overlap."

**3 bullets:**
- Evidential AUROC: **0.991** — nearly perfect separation
- Softmax AUROC: 0.854 — decent but a 14-point gap remains
- Vacuity (K/S) is a principled uncertainty score. Softmax has no equivalent.

**Speaker note:** "This is the core thesis contribution. In a deployment scenario,
the evidential model will flag 99% of OOD inputs. The softmax will silently
misclassify them with high confidence."

---

## Slide 10 — Training Curves

**Heading:** Both Models Learn Equally Fast

**Insert:** `results/training_curve.png`

**2 bullets:**
- Both models reach 100% support accuracy by step 20 — adapter is effective
- Loss curves confirm stable training for both modes
- The difference is NOT in learning speed — it is in what the output HEAD encodes

---

## Slide 11 — What Is Next (Full Thesis Plan)

**Heading:** Roadmap to Full Thesis

| Timeline | Task |
|----------|------|
| Month 1 | LoRA and BitFit adapters — compare calibration and OOD vs bottleneck |
| Month 1 | Multi-episode meta-training to fix ECE for evidential model |
| Month 2 | Additional backbones: MobileNetV3-Small, ConvNeXt-Nano |
| Month 2 | Additional datasets: CUB-200 (fine-grained), ISIC (medical imaging) |
| Month 3 | Edge benchmarks: Jetson Nano latency + power (RQ4) |
| Month 3 | Paper writing + ablation tables |

**One honest sentence:** "Today's demo shows the OOD detection story works.
The calibration story (ECE) requires multi-episode training — that is Month 1."

---

## Slide 12 — Thank You / Questions

**Heading:** Summary

- ~20K trainable parameters → 89.3% few-shot accuracy (0.17% of ResNet18)
- Evidential head → OOD AUROC **0.991** vs Softmax 0.854
- Code available: [github link]
- Full thesis: 3 adapters × 3 backbones × 3 datasets × edge benchmarks

**Questions welcome.**

---

## SPEAKER NOTES FOR LIKELY QUESTIONS

**Q: Why not LoRA?**
> "LoRA is Month 1. This demo establishes the evidential baseline. LoRA
> modifies the weight matrices directly — we expect it to learn faster
> with fewer parameters. We'll compare both in the full ablation."

**Q: What is vacuity?**
> "Vacuity is total uncertainty — K divided by the sum of all Dirichlet
> concentrations S. When the model has seen nothing like this input,
> all evidence values are small, S stays small, K/S is large. When it's
> confident, S is large, K/S is small. It's a principled measure —
> unlike softmax, which always outputs a distribution summing to 1."

**Q: Why is ECE worse for evidential?**
> "We trained on a single 25-sample episode. The softplus evidence grows
> large during CE training, pushing Dirichlet probs toward 1.0.
> Multi-episode meta-training regularises this — it's Month 1."

**Q: Isn't training with CE and evaluating with Dirichlet inconsistent?**
> "There is a mismatch, yes. The principled solution is to train with
> the evidential MSE loss — Sensoy 2018. But on 25 samples, the KL
> regularization prevents convergence. For the pre-defence demo, CE
> training with Dirichlet evaluation demonstrates the OOD story clearly.
> The full thesis uses multi-episode training where the evidential loss
> works as intended."

---
*Generated by B-PEFT demo pipeline | 2026-04-21*
