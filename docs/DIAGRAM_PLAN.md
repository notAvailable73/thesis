# Diagram and Figure Plan — B-PEFT Thesis Report and Defence Slides (must-have only)

This plan lists only the figures the report and the defence cannot do without. Each figure has: what to
draw, where it goes, where its numbers come from, a draft caption, and what not to imply.

**Ground rules**

- **Every figure is new.** No existing PNG from `results/` is reused.
- **Numbers come from `docs/guide/` and `docs/DEFENCE_DECK_25_SLIDES.md` only.** When a plot needs raw
  values, read them from a result data file (`results/*.json`, `results/rq3_matched/`), never from an older
  write-up.
- **Chapters** follow `docs/guide/07_introduction.md` §7.9 and the reference theses: Ch1 Introduction, Ch2
  Literature Review, Ch3 Methodology, Ch4 Experimental Setup, Ch5 Results, Ch6 Discussion, Ch7 Conclusion.
- **Numbering** as in the reference theses: `Figure <chapter>.<n>`, caption below the figure, cited in the
  text before it appears, listed in the List of Figures.

**16 figures**: 8 drawn diagrams + 8 data plots. Two pairs were merged to save work: adapters + parameter
reversal, and the two RQ4 plots.

**Update 2026-09-15: the RQ2/RQ4 completion run.** `notebooks/rq2_completion_rq1_interactions.ipynb` (Kaggle
run `notebook28109c47fb`) retrained the 21 lost CIFAR-FS 5-shot checkpoints, and all 21 reproduce the grid
exactly. Phase A now covers **120/120** models, RQ4 covers **60** evidential models, and RQ1's two-way
interactions are computed. The cards below for F1, F6, F7, F11, F12, F15 and F16 use the new numbers.
Values marked **⟳** could not be read from the notebook printout. Re-derive them from
`results/rq_completion/*.json` after unzipping `rq_completion_results.zip` at the repo root, and don't reuse
the old 99-model value. `results/rq_summary.json` still holds the 99-model numbers, so use it only for
before/after comparisons.

---

## 0. Fix these while drawing

| # | Problem | What to do |
|---|---|---|
| 1 | Deck Slide 12's sketch puts the **adapter after the backbone**. It actually sits **inside** it: the bottleneck at the last block of each of the 4 stages, LoRA inside one 1×1 conv (`09_methodology.md` §9.1, §9.3.2). | Draw the adapter inside the backbone box (F4, F5). |
| 2 | Slide 12 labels the backbone "11.7 M / 2.5 M". Those are **full-model** sizes. The frozen parts actually used are **11,176,512** (ResNet-18) and **927,008** (MobileNetV3-Small trunk). | Use the used counts. |
| 3 | The RQ1 η² values (76.1%, 82.9%, …) used to exist only as text in the guide. The completion run recomputed them, reproducing all four published rows to 2 d.p., and stored them in `results/rq_completion/rq1_interactions.json` → `outcomes.*.main_effects_only`. | Plot F11 from that file. Two metric keys are easy to get wrong: ECE is `ece_pooled` (not per-episode), and the TinyImageNet row is an OLS Type-II fit on 95 rows. |
| 4 | In RQ2 the **design** factor explained 23.2% (far) and **56.9%** (near) of AUROC variance on the 99 models, more than the score on near-OOD. | ⟳ Re-read both shares at 120 models from `results/rq_completion/rq2_rq4_summary.json` → `rq2_after_all_records.*.eta_squared.design`, then mention them in F12's caption so the headline isn't read as "score explains most of everything". |

---

## 1. Style (set once)

- **Tools:** draw diagrams in draw.io (diagrams.net) or TikZ. Make all 8 plots from **one matplotlib
  script** with a shared style, so they match and regenerate together.
- **Colours** (Okabe–Ito, safe for colour-blind readers):

  | Meaning | Colour |
  |---|---|
  | Frozen | grey `#9E9E9E` + lock icon |
  | Trainable | orange `#E69F00` |
  | Parameter-free | white fill, dashed outline |
  | Softmax | blue `#0072B2` |
  | TS-softmax | sky blue `#56B4E9` |
  | Evidential | vermillion `#D55E00` |
  | Bottleneck | green `#009E73` |
  | LoRA | pink `#CC79A7` |

- **Markers and panels:** backbone = marker shape (ResNet-18 ●, MobileNetV3-Small ▲); dataset = panel
  (CIFAR-FS left, MiniImageNet right).
- **Error bars:** name them in every caption. Full FT and Linear Probe get none; footnote them as "no random
  initialisation, so the 3 seeds are identical".
- **Output:** vector `.pdf` for the thesis in `report/figures/`; 300-dpi `.png` with larger fonts for slides.
- **Wording:** never "proves", "causes", "always", "first", "163×", or "calibration follows the budget".

---

## 2. The list, in build order

Build in this order: it covers the deck's load-bearing slides (15, 21A/B, 19, 20, 22) first.

| Order | ID | Figure | Kind | Report | Slide |
|---:|---|---|---|---|---|
| 1 | F4 | System pipeline | diagram | Ch3 overview | 12 |
| 2 | F7 | Experimental design | diagram | Ch3 analysis design | 15 |
| 3 | F14 | RQ3 matched-budget result | plot | Ch5 RQ3 | 21B |
| 4 | F11 | RQ1 variance explained (η²) | plot | Ch5 RQ1 | 19 |
| 5 | F12 | RQ2 objectives × scores | plot | Ch5 RQ2 | 20 |
| 6 | F15 | RQ4 refit: ECE and OOD ranking | plot | Ch5 RQ4 | 22 |
| 7 | F5 | The two adapters + parameter reversal | diagram + bars | Ch3 adapters | 13, 21A |
| 8 | F13 | RQ3 confound in the grid | plot | Ch5 RQ3 | 21A |
| 9 | F16 | Which design choice governs which property | diagram | Ch6 synthesis, Ch7 | 2, 25 |
| 10 | F3 | Softmax vs Dirichlet on the simplex | plot | Ch2 EDL | 6, 14 |
| 11 | F9 | Accuracy vs trainable parameters | plot | Ch5 grid overview | 18 |
| 12 | F10 | Calibration of every matched pair | plot | Ch5 grid overview | 19 |
| 13 | F2 | Episodic few-shot protocol | diagram | Ch2 few-shot | 6 |
| 14 | F1 | Research roadmap | diagram | Ch1 RQs | 10 |
| 15 | F6 | Two readouts of the same logits | diagram + curve | Ch3 readouts | 14 |
| 16 | F8 | RQ3 matched-budget design and decision rule | diagram | Ch3 RQ3 design | 21B |

**Slides with no figure** fall back to their own text or tables: 1, 3, 4, 5, 7, 8, 9, 11, 16, 17, 23, 24.

---

## 3. Figure cards

### F1 — Research roadmap (Ch1)

- **Draw.** A top-down tree:
  - top: the overarching question (`07_introduction.md` §7.6);
  - four boxes: **RQ1 Attribution · RQ2 Objective vs score · RQ3 Architecture vs budget · RQ4 Remediation**;
  - under each, its experiment: RQ1 → Step 10 grid (120 runs); RQ2 → Phase A re-scoring (all 120 models);
    RQ3 → grid 16 pairs + matched-budget (48 runs, pre-registered); RQ4 → Phase A refit (60 evidential
    models);
  - under each experiment, its Ch5 section.
- **Caption.** "Research questions, the experiments that answer them, and where each is reported."
- **Watch out.** Final RQ numbering only. Leave the Orig-RQs out.

### F2 — Episodic few-shot protocol (Ch2)

- **Draw.**
  - Left: 100 classes as three blocks, **64 train / 16 val / 20 test**, with a "disjoint" bracket.
  - Right: one **5-way 5-shot episode**, a 5×5 support grid and a 5×15 query block (75 images).
  - Arrows show which split each episode type draws from. Counter: "600 fixed test episodes".
- **Source.** `01_what_we_built.md` §1.3–1.4; `09_methodology.md` §9.2.
- **Caption.** "Episodic 5-way K-shot evaluation. Each episode samples 5 classes with 1 or 5 labelled support
  images and 15 queries per class; test episodes use only classes never seen during adaptation."
- **Watch out.** Don't imply MiniImageNet test classes are unseen by the *backbone*: ImageNet pretraining saw
  them.

### F3 — Softmax vs Dirichlet on the simplex (Ch2)

- **Draw.** 3-class triangles (state in the caption that the thesis uses K = 5).
  - Row 1, **softmax**: single points (confident; confused, near the centre; OOD but still near a corner).
  - Row 2, **Dirichlet densities**: α = (20, 1, 1) peaked; (8, 8, 1) conflicting; (1.1, 1.1, 1.1) flat,
    high vacuity. Write α, S and u = K/S beside each.
- **Source.** Formulas in `09_methodology.md` §9.3.4; α values are illustrative.
- **Caption.** "Softmax outputs one probability vector. An evidential head outputs Dirichlet parameters
  α = e + 1; low total evidence S gives high vacuity u = K/S. Illustrative, K = 3."
- **Watch out.** This is the textbook picture, not how the trained models behave. The thesis's evidential
  head is **worse calibrated** than softmax (0/20).

### F4 — System pipeline (Ch3) — the main method figure

- **Draw.** Left to right:
  1. support + query/OOD images;
  2. a grey **frozen backbone** (4 stages) with small **orange adapters inside**;
  3. pooled features (512-d / 576-d);
  4. a dashed **prototype head**: c_k = mean support feature, logits z = 10 × cosine similarity;
  5. a split into **Softmax** (p = softmax(z) → MSP, TS-MSP, energy) and **Evidential**
     (e = softplus(a·z + b), α = e + 1, S = Σα, p = α/S, u = K/S → vacuity), with an orange tag "a, b: 2
     learnable scalars".
- **Annotate on the diagram.**
  - FROZEN: "11,176,512 / 927,008 params, never updated".
  - TRAINABLE: "6,928 – 31,744 params".
  - PARAMETER-FREE: "a class = mean of its support embeddings".
- **Source.** `09_methodology.md` §9.1, §9.3.
- **Caption.** "Framework overview. A frozen ImageNet-pretrained CNN with a small trainable adapter embeds
  support and query images; a parameter-free prototype head produces cosine-similarity logits, read either
  as a softmax or as Dirichlet evidence."
- **Watch out.**
  - Adapter inside the backbone (§0 #1).
  - Cosine × 10, not L2.
  - Energy uses raw z in both readouts.

### F5 — The two adapters + parameter reversal (Ch3) — merged figure

- **(a) Draw**, two zooms side by side (ResNet-18 shown; list the MobileNetV3-S sites in text):
  - **Bottleneck-parallel.** Residual block B_s with a side branch: 1×1 conv C→r → ReLU → 1×1 conv r→C,
    added at the output: y = B(x) + W_up ReLU(W_down x). Note "W_up = 0 at init". Placed at the last block
    of all 4 stages (C = 64/128/256/512).
  - **LoRA.** Frozen 1×1 conv W₀ with A, B inside: W = W₀ + (α/r)·BA, B = 0 at init. One layer only,
    layer4.0.downsample.0 (256→512).
- **(b) Plot**, grouped bars with a linear y-axis:
  - ResNet-18: bottleneck **31,744** vs LoRA **12,288** ("bottleneck larger, 2.58×");
  - MobileNetV3-S: **6,928** vs **10,752** ("LoRA larger, 1.55×");
  - an arrow between the groups: "ordering reverses".
- **Source.** `09_methodology.md` §9.3.2; `01_what_we_built.md` §1.2.
- **Caption.** "(a) The parallel bottleneck adds a zero-initialised side branch at every stage; LoRA adds a
  zero-initialised low-rank update inside one frozen 1×1 convolution (rank 16). (b) Because channel widths
  differ, the larger adapter is the bottleneck on ResNet-18 and LoRA on MobileNetV3-Small, which is what
  lets RQ3 separate architecture from budget."
- **Watch out.** The reversal was **noticed, not designed**. Keep the "coverage explains accuracy" idea out
  of the caption; it is an untested interpretation.
- **Slides.** Slide 13 uses (a) + (b); Slide 21A reuses (b).

### F6 — Two readouts of the same logits (Ch3)

- **(a) Draw.** One 5-bar logit vector z splits into softmax probabilities and the evidential chain
  a·z + b → softplus → +1 → α → p = α/S, u = K/S. Note: "a matched pair differs by exactly 2 parameters
  (31,746 vs 31,744)".
- **(b) Plot.** Evidence e = softplus(a·z + b) for z ∈ [−10, 10], three curves:
  - init (2, −6);
  - median trained affine;
  - median refit affine.
- **Source.** `09_methodology.md` §9.3.4, §9.6. ⟳ Medians over the 60 evidential models from
  `results/rq_completion/rq2_rq4_summary.json` → `rq4_rows[*].affine_trained` / `affine_refit`. The old
  48-model medians were trained ≈ (2.93, −6.30) and refit scale ≈ 8.3. The 12 new CIFAR-FS 5-shot models
  refit to scales of about 6–10 in the run log, so the medians may move.
- **Caption.** "Both readouts use the same prototype logits; the evidential readout maps them to Dirichlet
  evidence through a learnable affine and softplus. (b) Evidence curves at initialisation, after training,
  and after the post-hoc refit (Section 5.x)."
- **Watch out.** (a, b) are **learnable**, not frozen at (2, −6).
- **Slide version.** Slide 14 uses (a) only.

### F7 — Experimental design (Ch3) — the deck centrepiece

- **Draw.**
  - **Centre**: five columns of two tiles each:
    - Dataset {CIFAR-FS, MiniImageNet}
    - Shots {1, 5}
    - Backbone {ResNet-18, MobileNetV3-S}
    - Adapter {Bottleneck, LoRA}
    - Head {softmax, evidential}
  - Beneath: "2⁵ = 32 + 8 baseline cells (Full FT, Linear Probe; ResNet-18 + CIFAR-FS) = 40 × 3 seeds =
    **120 runs**".
  - **Right**: three distinct cards:
    1. **Phase A re-scoring**: 120/120 checkpoints → RQ2 + RQ4. 99 were recovered; the 21 lost CIFAR-FS
       5-shot adapter checkpoints were retrained from their grid configs, and all 21 reproduce the grid
       exactly (regression guard exact, `best_val_epoch` 21/21).
    2. **Matched-budget, pre-registered**: 48 runs → RQ3.
    3. **Rank sweep**: 21 runs → former RQ5.
  - Footer: "189 training runs (120 + 48 + 21) · 600 frozen test episodes each · 0 run errors".
- **Source.** `04_experiments.md` §4.1, §4.3–4.7.
- **Caption.** "Evidence base: a balanced five-axis factorial for variance attribution, plus a checkpoint
  re-scoring, a pre-registered matched-budget experiment and a controlled rank sweep, each answering a
  question the grid alone cannot."
- **Watch out.** Don't count Phase A in the 189. The 21 retrained checkpoints repeat grid runs exactly, so
  they aren't new runs either. Don't mix them up with the rank sweep's 21 runs.

### F8 — RQ3 matched-budget design and decision rule (Ch3)

- **(a) Draw** (dumbbells or a small table graphic): bottleneck vs LoRA parameters per backbone × level:

  | Backbone | Level | Bottleneck rank → params | LoRA rank → params | Mismatch |
  |---|---|---|---|---:|
  | ResNet-18 | L | 6 → 12,504 | 16 → 12,288 | 1.76% |
  | ResNet-18 | H | 16 → 31,744 | 41 → 31,488 | 0.81% |
  | MobileNetV3-S | L | 16 → 6,928 | 10 → 6,720 | 3.10% |
  | MobileNetV3-S | H | 22 → 9,448 | 14 → 9,408 | 0.43% |

  Before matching, the grid pairs differed by 55–158%.
- **(b) Draw** the decision flowchart, per (backbone × readout) cell, with ΔECE = LoRA − bottleneck and
  σ = pooled seed SD:
  - **H3.2 budget**: |Δ_matched| ≤ 50% of |Δ_unmatched| and ≤ 2σ;
  - **H3.1 architecture**: same sign on both backbones and > 2σ;
  - **H3.2-alt backbone**: keeps its per-backbone unmatched sign and > 2σ;
  - a hypothesis is supported if it fires in ≥ 3 of 4 cells, else "inconclusive".
  - Stamp: "fixed before any deciding run".
- **Source.** `04_experiments.md` §4.7; `09_methodology.md` §9.9.4.
- **Caption.** "Matched-budget experiment (MiniImageNet 5-shot, 48 runs): (a) budgets matched within each
  backbone; (b) the pre-registered decision rule."
- **Watch out.** Copy the thresholds exactly as registered.

### F9 — Accuracy vs trainable parameters (Ch5)

- **Plot.** Two panels (CIFAR-FS 5-shot | 1-shot). x = trainable params (log), with Linear Probe on a "0"
  stub; y = accuracy %; softmax head; ±1 SD. A dashed line at Full FT.

  | Configuration | Params | 5-shot | 1-shot |
  |---|---:|---:|---:|
  | Bottleneck, ResNet-18 | 31,744 | 91.44 | 78.57 |
  | Bottleneck, MobileNetV3-S | 6,928 | 90.74 | 78.80 |
  | LoRA, ResNet-18 | 12,288 | 86.25 | 75.44 |
  | LoRA, MobileNetV3-S | 10,752 | 88.05 | 75.43 |
  | Full FT (ResNet-18) | 11,176,512 | 90.47 | 81.14 |
  | Linear Probe | 0 | 87.41 | 70.25 |

- **Source.** `03_results.md` §3.11; raw values in `results/mvt_results.json`.
- **Caption.** "Accuracy against trainable parameters on CIFAR-FS (softmax, mean ± SD over 3 seeds). At
  5-shot a 31,744-parameter adapter exceeds full fine-tuning of the same backbone by 0.98 points and a
  6,928-parameter adapter on MobileNetV3-Small matches it; at 1-shot full fine-tuning leads by 2.57 points.
  Full FT and Linear Probe were run on ResNet-18 only."
- **Watch out.** "Matches", not "beats", for MobileNetV3-S: it is inside the seed spread and a
  cross-backbone comparison.
- **Slide version.** Slide 18 uses the 5-shot panel only.

### F10 — Calibration of every matched pair (Ch5)

- **Plot.** A dot/dumbbell chart, one row per matched configuration (20 rows), grouped by dataset and shot.
  Each row has three markers on an ECE axis: TS-softmax, softmax, evidential.
- **Source.** `03_results.md` §3.11 (ECE, TS-ECE columns).
- **Caption.** "Pooled ECE for every matched softmax/evidential pair. Evidential is worse calibrated than
  softmax in 20 of 20 pairs; temperature-scaled softmax is best in every row."
- **Watch out.** Say in the text that temperature scaling isn't available to the evidential head in this
  codebase, so that comparison favours softmax by construction.

### F11 — RQ1: variance explained per design axis (Ch5)

- **Plot.** A 4×6 heatmap (works for both thesis and slide), rows = outcomes, columns = Dataset, Shots,
  Backbone, Adapter, Head, Residual. Values in %:

  | Outcome | Dataset | Shots | Backbone | Adapter | Head | Residual |
  |---|---:|---:|---:|---:|---:|---:|
  | Accuracy | 1.8 | **76.1** | 3.9 | 9.1 | 0.2 | 9.0 |
  | ECE | 2.0 | 2.1 | 4.6 | 0.6 | **82.9** | 7.7 |
  | Far-OOD AUROC (SVHN) | 1.2 | 22.7 | 11.5 | 1.7 | **39.0** | 23.9 |
  | Near-OOD AUROC (TinyImageNet) | 1.6 | **42.6** | 0.6 | 22.0 | 21.0 | 11.4 |

- **Source.** `results/rq_completion/rq1_interactions.json` → `outcomes.*.main_effects_only` (§0 #3). These
  match `03_results.md` §3.3.
- **Caption.** "Share of variance (η², main effects) explained by each design axis, over 96 per-seed
  observations of the balanced 32-cell factorial (95 for near-OOD). Shot count dominates accuracy; head
  interpretation dominates calibration and explains 0.2% of accuracy variance."
- **Watch out.**
  - The heatmap is still main effects only, but the interactions are now **computed**. Adding the ten
    two-way terms shrinks the residual from 9.0 / 7.7 / 23.9 / 11.4% to **1.3 / 2.1 / 9.6 / 6.2%**
    (accuracy / ECE / far / near). The largest term is usually `dataset × backbone` (5.3% of accuracy, 9.2%
    of far-OOD). `backbone × adapter` is 3.3% of ECE and near zero for OOD. Put this in a sentence or a
    small companion table, not in the heatmap.
  - Say that interaction shares are **less certain** than the main effects next to them. The seed
    bootstrap understates within-cell variance with only 3 seeds.
  - One of the 96 observations (CIFAR-FS / 5-shot / MobileNetV3-S / LoRA / evidential / seed 42) is a
    20-episode smoke-test value in `mvt_results.json`. Replacing it with the same model's 600-episode
    scores moves the main effects by at most 0.7 pp (near-OOD shots 42.6 → 43.3%). Footnote it; don't
    replot.
  - Don't write "we discovered"; the general pattern goes back to Guo et al. 2017.

### F12 — RQ2: 2 objectives × 4 scores (Ch5)

- **Plot.** Two panels (far | near). x = score, paired bars = training objective, y = mean AUROC:

  Values are for all 120 models, from the completion run. Each bar has 120 observations: 20 designs × 3 seeds
  × 2 pools in that group.

  | Pool | Objective | MSP | TS-MSP | Energy | Vacuity (VAL-fit) |
  |---|---|---:|---:|---:|---:|
  | Far | evidential-trained | 0.7993 | 0.7868 | **0.9185** | 0.9134 |
  | Far | softmax-trained | 0.8014 | 0.7850 | **0.9342** | 0.9307 |
  | Near | evidential-trained | 0.7916 | 0.7805 | **0.8414** | 0.8347 |
  | Near | softmax-trained | 0.7770 | 0.7634 | **0.8371** | 0.8233 |

  Text on the panels: η² score **42.7%** vs objective **0.17%** (far); **14.7%** vs **0.63%** (near).
  (On the 99 recovered models it was 43.7% vs 0.27% and 13.0% vs 0.60%.)
- **Source.** `results/rq_completion/rq2_rq4_summary.json` → `rq2_tables_after` and `rq2_after_all_records`.
  The design is fully crossed: 20/20 designs complete, no zero-observation cells. `results/rq_summary.json`
  holds the superseded 99-model version.
- **Caption.** "OOD AUROC by scoring rule and training objective over all 120 models. The same score gives
  similar AUROC whichever objective trained the logits. Of these two factors the score explains far more
  variance (42.7% vs 0.17% far; 14.7% vs 0.63% near); the model configuration itself explains a further
  ⟳ (far) and ⟳ (near)." Fill both ⟳ from §0 #4.
- **Watch out.**
  - No ratio anywhere. The ratio used to be unstable because one design had only its softmax arm; with
    that fixed it is stable, but the no-ratio rule still applies.
  - Don't use the stored `std` as error bars: it is the spread across designs, not uncertainty.
  - Coverage is complete (120/120). The "99/120" caveat is gone. If the text mentions how coverage was
    completed, say the 21 retrained checkpoints reproduce the grid exactly.

### F13 — RQ3 part 1: the confound in the grid (Ch5)

- **Plot.** 16 matched pairs on the y-axis, grouped into a ResNet-18 block and a MobileNetV3-S block. Two
  columns:
  - **ΔAccuracy** (bottleneck − LoRA, pp);
  - **ΔECE** (LoRA − bottleneck).
  - Colour each block by which arm is larger. Filled markers = beyond 2σ, hollow = not.
  - What it should show: all 16 accuracy deltas on one side of zero; ECE deltas positive on ResNet-18 and
    negative on MobileNetV3-S.
- **Source.** Pair values from `results/mvt_results.json`; counts in `03_results.md` §3.5.
- **Caption.** "Bottleneck vs LoRA across the 16 matched grid pairs. The accuracy winner stays the same when
  the budget ordering reverses; the calibration winner changes with it. Because the reversal coincides with
  the backbone change, the grid cannot separate a budget effect from a backbone effect."
- **Watch out.** Only 10/16 ECE pairs clear 2σ (all 8 MiniImageNet pairs, 2 of 8 CIFAR-FS). The hollow
  markers show this honestly.
- **Slide version.** Slide 21A: this plot with F5(b) beside it.

### F14 — RQ3 part 2: matched-budget result (Ch5) — strongest result

- **(a) Plot.** Four cell groups, each with an **unmatched** (hatched) and a **matched** (solid) bar,
  ±2σ on both, and a zero line. Collapse ratio above each group; the hypothesis that fired underneath.

  | Cell | ΔECE unmatched | ΔECE matched | Collapse ratio | Fired |
  |---|---:|---:|---:|---|
  | ResNet-18 / evidential | +0.1111 | +0.1123 | 1.01 | backbone |
  | ResNet-18 / softmax | +0.1080 | +0.0991 | 0.92 | backbone |
  | MobileNetV3-S / evidential | −0.0124 | −0.0062 | 0.50 | backbone |
  | MobileNetV3-S / softmax | −0.0152 | −0.0060 | 0.61 | none (below 2σ) |

- **(b) Plot**, a small strip: 8 ΔAccuracy and 8 Δnear-OOD AUROC (bottleneck − LoRA) at matched budget,
  with 2σ bars, all on one side of zero.
- **Banner:** "backbone 3/4 (threshold 3) · budget 0/4 · architecture 0/4".
- **Source.** `results/rq3_matched/verdict.json` (`matched_table`, `unmatched_baseline`, `decision`,
  `secondary`); `03_results.md` §3.5.
- **Caption.** "(a) ΔECE (LoRA − bottleneck) before and after equalising trainable parameters, ±2σ.
  Equalising the budget leaves the ResNet-18 gap intact and roughly halves the MobileNetV3-Small gap
  without changing its sign, which is evidence against parameter budget as the primary explanation. (b) At
  matched budget the bottleneck still wins accuracy and near-OOD AUROC in 8 of 8 comparisons, beyond 2σ.
  Far-OOD shows no clear adapter effect (5/8)."
- **Watch out.** Never "calibration follows the budget". Never "the backbone causes it": which backbone
  property matters is **unidentified**.

### F15 — RQ4: refit, calibration and OOD ranking (Ch5) — merged figure

- **(a) Plot.** A dumbbell with 60 rows (evidential checkpoints), sorted by ECE before. Per row:
  - ECE before (hollow) → after (solid);
  - a blue tick for the plain softmax ECE of the same design;
  - a sky-blue tick for TS-softmax ECE.
  - Summary box:
    - improved **60/60**, mean ΔECE **−0.139**; mean before → after ⟳ (was 0.327 → 0.190 on 48);
    - still worse than softmax ⟳/60, mean ratio ⟳ (was 48/48, 2.18×);
    - worse than TS-softmax ⟳/60, mean ratio ⟳ (was 48/48, 11.9×).
- **(b) Plot.** A strip of ΔAUROC (after − before) for the 240 comparisons, one column per OOD pool. Dashed
  line at **−0.005**, the "preserved" region shaded. Counts: **192/240 (80%) preserved**, mean **+0.003**,
  worst ⟳ (was ≈ −0.03 on 48), min Spearman ρ **0.866** (was 0.921).
  - The 12 new CIFAR-FS 5-shot cells are the harder part: 42/48 preserved, mean ΔAUROC −0.002, and they
    contain the new minimum ρ. Consider marking them (e.g. a different marker) instead of hiding the shift
    in the pooled count.
- **Source.** `results/rq_completion/rq2_rq4_summary.json` → `rq4_rows[*]` (`ece_before`, `ece_after`,
  `ece_softmax_ref`, `ece_ts_ref`, `auroc_delta__<pool>`, `rank_rho__<pool>`), `rq4_after`,
  `rq4_new_cells_only`. `results/rq_summary.json` → `rq2_rows` / `rq2_verdict` holds the superseded
  48-model version (retired key names).
- **Caption.** "Post-hoc refit of the two evidence parameters on validation episodes. (a) ECE improves in all
  60 checkpoints but stays above plain and temperature-scaled softmax in ⟳ of 60. (b) OOD AUROC stays within
  0.005 in 192 of 240 comparisons; the rest drop by up to about ⟳."
- **Watch out.**
  - The "still worse than softmax" line is not optional. Re-count it on 60 before writing it; it was only
    checked on 48.
  - Never "fixes calibration".
  - Never "always preserves ranking". Rank correlation now goes as low as 0.866.

### F16 — Which design choice governs which property (Ch6, Ch7)

- **Draw** a two-column map:
  - left: Shots · Adapter architecture · Adapter budget · Backbone · Head interpretation · Scoring rule ·
    Training objective · Post-hoc refit;
  - right: Accuracy · Calibration · OOD ranking.
  - **Thick** edges (found):
    - Shots → Accuracy (RQ1);
    - Adapter architecture → Accuracy and near-OOD (RQ3);
    - Head → Calibration (RQ1);
    - Backbone → Calibration gap between adapters (RQ3; tag "RQ1 interaction agrees", since the
      `backbone × adapter` term takes 3.3% of ECE variance);
    - Scoring rule → OOD (RQ2).
  - **Crossed-out** edges (tested, not supported): Budget → Calibration (0/4); Training objective → OOD
    (< 1%; 0.17% far, 0.63% near at 120 models).
  - **Partial** arrow: Refit → Calibration ("improves 60/60, gap to softmax remains"; ⟳ re-check the
    softmax gap on the 12 new cells).
- **Source.** `10_discussion_and_conclusion.md` §10.1; deck Slide 25.
- **Caption.** "Summary of findings: different design choices govern different aspects of reliability.
  Solid edges: dominant effects found. Crossed edges: explanations tested and not supported."
- **Watch out.** Don't scale edge widths by η²; the numbers come from different analyses. Footnote that
  near-OOD also depends on shots (42.6%).
- **Slides.** Slide 2 (small) and Slide 25 (large).

---

## 4. Checklist per figure

- [ ] Every number traced to a result file or a `docs/guide/` section.
- [ ] Error bars named in the caption.
- [ ] Final RQ numbering; no banned phrases (§1).
- [ ] Colours and markers as in §1; readable in greyscale and at slide size.
- [ ] Vector file saved in `report/figures/`; plotting script kept.
- [ ] Cited in the text before it appears; listed in the List of Figures.
