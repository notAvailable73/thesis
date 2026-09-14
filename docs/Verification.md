# Literature Verification and Publishability Analysis for B-PEFT Research Questions

This report provides an independent, rigorous evaluation of the five proposed research questions (RQs) for the **B-PEFT** project (parameter-efficient fine-tuning of frozen lightweight CNN backbones for few-shot vision). We analyze the novelty of each question against the current state of literature (up to August 2026), assess their scientific and practical value, and provide a publishability verdict.

---

## 📄 Verified Literature Sources

To ground this evaluation, we reviewed recent literature in parameter-efficient fine-tuning (PEFT), model calibration, out-of-distribution (OOD) detection, and evidential deep learning (EDL). Key anchor papers include:

1. **"Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters" (arXiv:2606.00428, June 2026)**: Studies how fine-grained parameter capacity increments affect accuracy in low-rank adaptation, but does not evaluate calibration or OOD metrics.
2. **"A Systematic Comparison of Training Objectives for Out-of-Distribution Detection in Image Classification" (arXiv:2603.07571, March 2026)**: Compares training losses (Cross-Entropy, Triplet, Prototype, AP) for OOD detection under OpenOOD, noting that training objectives and OOD scoring rules are often not fully factorized. It does not cover frozen backbones, episodic few-shot evaluation, or adapters.
3. **"Be Confident in What You Know: Bayesian Parameter Efficient Fine-Tuning of Vision Foundation Models" (NeurIPS 2024)**: Explores Bayesian PEFT (B-PEFT) for vision models, focusing on calibration and accuracy under few-shot settings but primarily presenting a methodological contribution rather than a structural attribution study.
4. **"What Adapter Ensembling Actually Buys: A Pre-Registered Audit, and the Scalar That Replaces the Population" (ACL ARR, August 2026)**: Audits ensembling and calibration of adapters, highlighting that simple temperature scaling can often close the calibration gap of single adapters under certain regimes.

---

## 🔍 Detailed Verification of Proposed Research Questions

### RQ-A: Does the trainable-parameter budget or the adapter architecture govern reliability, and is it the same lever that governs accuracy?

*   **Underlying Mechanic**: The authors exploit an organic structural anomaly in their grid. For a frozen **ResNet-18**, the bottleneck adapter (31,744 parameters) is **2.6× larger** than LoRA (12,288 parameters). For **MobileNetV3-Small**, the ordering reverses: LoRA (10,752 parameters) is **1.6× larger** than the bottleneck adapter (6,928 parameters). This allows them to deconfound the effects of *architecture* (bottleneck vs. LoRA) from *parameter budget* (larger vs. smaller).
*   **Novelty Check**: 
    *   Typical PEFT papers conflate performance gains with parameter growth (i.e., a larger adapter is assumed to be better because it has more capacity). 
    *   No prior literature has leveraged a backbone-induced parameter-count reversal to dissociate adapter architecture from budget across different reliability metrics (accuracy, near-OOD, far-OOD, ECE).
    *   The closest work, *CP Tensor Adapters (Wang et al., 2026)*, sweeps parameter-capacity curves at high resolution but restricts its evaluation strictly to validation accuracy on NLP benchmarks.
*   **Scientific Value & Publishability**: **Extremely High**. This is a beautiful, clean attribution experiment. The finding that **accuracy and near-OOD AUROC are architecture-governed** (bottleneck consistently wins 16/16 regardless of size) while **expected calibration error (ECE) is budget-governed** (whichever adapter has more parameters wins 16/16) is a highly valuable, non-obvious insight. It shifts the PEFT literature from empirical benchmarking ("my adapter beats yours") to fundamental structural understanding.
*   **Critical Caveat (Reviewer Check #1)**: The author is highly honest about a major limitation: because the budget ordering reverses *with the backbone*, the "budget governs ECE" claim is mathematically confounded with "backbone-intrinsic calibration characteristics" (ResNet-18 vs. MobileNetV3-S). While the budget explanation is more parsimonious (explaining the LP $\rightarrow$ LoRA $\rightarrow$ Bottleneck progression in RQ-B), **the current grid alone cannot fully isolate this causal link**. Running the proposed rank sweep in RQ-B (varying budget while holding the backbone and adapter family fixed) is **absolutely required** to lock in this causal claim.

---

### RQ-B: Is there an interior optimum in the reliability-versus-budget curve, and does it coincide with the accuracy-optimal budget?

*   **Underlying Mechanic**: Evaluates ECE and accuracy across four budgets on CIFAR-FS × ResNet-18: Linear-Probe (0-2 params) $\rightarrow$ LoRA (12.3k) $\rightarrow$ Bottleneck (31.7k) $\rightarrow$ Full Fine-Tuning (11.18M). ECE exhibits a U-shaped curve, reaching its minimum at the 31.7k bottleneck adapter before rising again during Full-FT. Meanwhile, accuracy continues to rise monotonically up to Full-FT (at 1-shot).
*   **Novelty Check**:
    *   Prior literature paints a contradictory picture: *Guo et al. (2017)* established that model capacity degrades calibration (suggesting ECE increases with capacity), while *LoRA vs. Full Fine-Tuning (2024)* reports that Full-FT is significantly better calibrated than LoRA.
    *   No work has unified these perspectives by demonstrating an **interior optimum (U-shape)** specifically for the *trainable* capacity under a frozen backbone. Underfitting the confidence function occurs at low budgets (LP), while overfitting occurs at extremely high budgets (Full-FT).
*   **Scientific Value & Publishability**: **High (Conditional on further experiments)**. If confirmed, this is highly publishable because it directly challenges the standard industry practice of selecting adapter capacity solely based on validation accuracy. It proves that maximizing accuracy silently degrades calibration reliability.
*   **Actionable Recommendation**: As the authors note, this is currently their weakest-evidenced claim because it relies on only four points and confounds adapter type with budget. To make this load-bearing, they must execute the **rank sweep** proposed in §4 (varying bottleneck rank across ~7 values with 3 seeds $\approx$ 21 runs, requiring ~6 GPU hours). This is a tiny computational cost for a massive increase in scientific rigor.

---

### RQ-C: When the training objective and the readout score are fully factorised, which produces the uncertainty benefit?

*   **Underlying Mechanic**: Evaluates a complete $2 \times 4$ factorial matrix: 2 training objectives (evidential/Dirichlet vs. softmax cross-entropy) crossed with 4 OOD readout scores (vacuity, MSP, TS-MSP, energy). This resolves the standard confound where evidential models are evaluated only on vacuity, and softmax models only on MSP/energy.
*   **Novelty Check**:
    *   The literature has a massive open gap here. The authors correctly cite *Genç et al. (2026)* ("A Systematic Comparison of Training Objectives for OOD Detection"), which explicitly flags that training objectives and OOD scoring rules are typically not fully factorized in evaluations.
    *   Furthermore, existing OOD benchmarks (like OpenOOD) do not evaluate these combinations under frozen backbones, parameter-efficient adapters, or episodic few-shot regimes.
*   **Scientific Value & Publishability**: **High**. This is a classic, rigorous diagnostic study. It prevents researchers from incorrectly attributing OOD performance to "evidential feature learning" when the benefit actually stems entirely from the mathematical properties of the readout score (e.g., energy vs. vacuity), or vice versa. It is highly worth publishing.

---

### RQ-D: Can an evidential head be recalibrated post-hoc by a two-parameter evidence affine, and does that preserve its OOD ranking?

*   **Underlying Mechanic**: Applies a two-parameter affine transform to evidential logits: $\text{softplus}(\text{logits} \times \text{scale} + \text{bias})$ fitted on a validation split, and measures its impact on ECE and OOD ranking.
*   **Novelty Check**:
    *   Post-hoc calibration is mature (e.g., Dirichlet calibration by *Kull et al., 2019*). The novelty is not "first post-hoc calibration," but rather evaluating how a two-parameter affine recalibration of an evidential prototype head behaves under an episodic few-shot regime.
    *   The specific focus on **OOD ranking preservation** is mathematically highly insightful. Since vacuity is computed as $K / S$ (where $S$ is the sum of Dirichlet parameters $\alpha$, which are non-linear transformations of all logits), a per-logit monotone transformation does *not* mathematically guarantee that the sample ordering of vacuity remains unchanged.
*   **Scientific Value & Publishability**: **Moderate-to-High**. This is a tight, elegant, and highly practical contribution. If the ranking is preserved, it provides an extremely cheap, edge-compatible calibration method for evidential heads. If the ranking collapses, it uncovers an important, undocumented trade-off between calibration and OOD detection in Dirichlet-based uncertainty models.

---

### RQ-E: Which design axis owns which reliability metric, and what does uncertainty cost at inference?

*   **Underlying Mechanic**: Performs an effect-size decomposition ($\eta^2$, variance explained) across a balanced, full-factorial 32-cell subgrid (dataset, shots, backbone, adapter, head). It also profiles latency on edge-proxy hardware.
*   **Novelty Check**:
    *   Most PEFT studies vary one factor at a time (OAT design) and cannot compute true variance decomposition. 
    *   Uncovering that **the head axis explains 84.0% of calibration variance and only 0.2% of accuracy variance** is a striking, clean separation of concerns. 
    *   Showing that the apparent correlation between ECE and OOD AUROC is purely a "head effect" (and collapses to orthogonality once controlled for head type) is a major conceptual contribution.
*   **Scientific Value & Publishability**: **Very High**. This provides an invaluable "map" of the design space for practitioners. It proves that uncertainty estimation is practically "free" at inference (evidential head costs only 1.29% more than softmax, which is below the measurement noise floor), and its trade-offs are strictly design-time decisions (governed by the head and budget).

---

## 🛠️ Reviewer Feedback on Statistical & Methodological Framing (§7)

1.  **Backbone Confound in RQ-A**: Yes, the backbone-intrinsic characteristics are currently confounded with the budget reversal. **The rank sweep proposed in RQ-B is the single most critical experiment to solve this.**
2.  **Sign Test Over 16 Pairs**: The author's skepticism is correct. A sign test assumes independent trials, but pairs sharing a backbone or dataset are not independent. To make this statistically rigorous, the authors should report **hierarchical/mixed-effects models** or at least perform a **Wilcoxon signed-rank test** taking group blocking into account.
3.  **$\eta^2$ on Aggregated Seeds**: Decomposing variance on seed-averaged observations artificially suppresses the "seed noise" variance (inflating the main effects). **The $\eta^2$ decomposition should absolutely be recomputed on the raw, per-seed observations (all 96 or 120 runs)**. This will allow the "Residual" column to honestly reflect seed-to-seed variance and validate whether the main effects remain dominant.

---

## 🎯 Verdict on the Proposed Reframing

**The proposed transition is brilliant and highly recommended.**

*   **Original Framing**: *"Bayesian PEFT for reliable few-shot vision"* — A weak, method-focused paper where the proposed method (evidential/Bayesian) frequently loses to simpler baselines (like softmax with energy or temperature scaling). This would likely be rejected or routed to a minor workshop.
*   **Proposed Framing**: *"What governs reliability under parameter-efficient adaptation?"* — An **attribution and systematic-understanding study** over a balanced factorial grid. Here, the "negative" results are no longer failures; they are critical, load-bearing measurements that establish which design axes govern which metrics.

**Conclusion**: The five proposed research questions are highly novel, mutually reinforcing, and timely. If the authors execute the minor additional experiments (the bottleneck rank sweep for RQ-B and the evaluation-only logit factorization for RQ-C), this paper will have an excellent chance of acceptance at a top-tier machine learning or computer vision conference (NeurIPS, ICML, or ECCV/CVPR).

---
*Analysis completed on August 21, 2026.*