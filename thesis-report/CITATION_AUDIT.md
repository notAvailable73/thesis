# Citation Audit — the teammate's 33-paper list vs. what the repo actually uses

**Date:** 2026-09-13 · **Scope:** every `.md` / `.txt` / `.py` / `.yaml` / `.ipynb` in the repo,
excluding `PAPER SUMMARIES/` itself · **Method:** the list was checked in both directions —
(a) is each listed paper actually used, (b) is anything used that is not listed.

---

## 0. Headline

| | |
|---|---|
| Papers in the teammate's list | **32** (the header says 33 — miscount) |
| Papers in `PAPER SUMMARIES/*.txt` | **32** — the list is a 1:1 transcription of the summary folder |
| Of those 32, load-bearing in the work | **18** |
| Of those 32, related-work mention only | **6** |
| Of those 32, **never used anywhere** | **8** |
| Works used in the repo but **absent from the list** | **~60** (43 of them already carry arXiv IDs in `docs/`) |

**The core problem:** the list is the *reading set* (what was summarised in July 2026), not the
*citation set*. Everything the project has cited since — the entire novelty-check and
positioning literature in `docs/DEFENCE_BRIEF.md`, `docs/RESULTS_MASTER.md` §4,
`docs/RQ_RESULTS_SUMMARY.md` and `docs/RQ_SUPERVISOR_REPORT.md` — is missing from it, including
the datasets and the primary backbone.

---

## 1. The count is off by one

The list header says 33. Counting the entries: PEFT 8 + Bayesian PEFT 5 + EDL 5 + Calibration 2 +
OOD 6 + Few-Shot 4 + CNN 2 = **32**. This matches the summary files exactly
(`PEFT_summaries.txt` 8, `B-peft_paper_summaries.txt` 5, `EDL_paper_summaries.txt` 5,
`Callibration_paper_summaries.txt` 2, `ODD_paper_summaries.txt` 6, `FEW SHOTpaper_summaries.txt` 4,
`CNN_paper_summaries.txt` 2). No paper is missing from the transcription; the total is just wrong.

---

## 2. Tier A — load-bearing (18 of 32) ✅ cite these, they justify implementation choices

| # | Paper | Where it is actually used |
|---|---|---|
| 1 | Houlsby, *PEFT for NLP* | `src/adapters/bottleneck.py` — the bottleneck adapter form; Steps 1, 6 |
| 2 | Hu, *LoRA* | `src/adapters/lora.py:1` (`"LoRA (Hu 2021)"`); Step 5 |
| 3 | Ben-Zaken, *BitFit* | `src/adapters/bitfit.py:1` (`"BitFit (Ben-Zaken 2022)"`); Step 5, `configs/exp_phase3_bitfit_*.yaml` |
| 4 | Chen, *Conv-Adapter* | `src/adapters/placement.py:27` — the 1×1 "loses locality" caveat; Step 6 / RQ1 |
| 5 | Aleem, *ConvLoRA + AdaBN* | `src/adapters/lora.py:22` — conv-form LoRA; `step_writeups/step5.txt:29` |
| 6 | Zhong, *Convolution Meets LoRA (SAM)* | `step_writeups/step5.txt:29,40` — low-data conv-LoRA evidence |
| 7 | He, *Towards a Unified View* | `step_writeups/step6.txt:36` — serial-vs-parallel vocabulary for RQ1 |
| 8 | Sensoy, *EDL* | The entire evidential head + loss (`src/losses/evidential.py`, `src/heads/prototype_head.py`) |
| 9 | Guo, *On Calibration* | `src/evaluators/temperature.py` — temperature scaling; ECE/reliability; RQ4 |
| 10 | Hendrycks, *MSP Baseline* | The MSP OOD baseline every head is scored against |
| 11 | Liu, *Energy-based OOD* | The energy score — the honest competitor; drives the Step 10 RQ3 correction |
| 12 | Yang, *OpenOOD* | Near/far OOD split design — `src/datasets/tinyimagenet_ood.py:1`, `src/datasets/cifar_fs.py:260` |
| 13 | Zhang, *OpenOOD v1.5* | Benchmark protocol references in `docs/RQ_RESULTS_SUMMARY.md`, `docs/NEW_RQS.md` |
| 14 | Chen, *A Closer Look* | `configs/exp_phase2_softmax.yaml:26` — the linear-probe baseline; Step 9 confound analysis |
| 15 | Vinyals, *Matching Networks* | MiniImageNet provenance, `step_writeups/step9.txt:114`; comparison table row |
| 16 | Snell, *Prototypical Networks* | `src/heads/prototype_head.py` + the whole episodic protocol (5-way, 15 query, 600 episodes) |
| 17 | Howard, *MobileNetV3* | `src/backbones/mobilenetv3.py:6` — second backbone; Step 8 |
| 18 | Malinin, *Prior Networks* | `src/datasets/tinyimagenet_ood.py:7` — near-OOD degradation rationale; Step 7 |

## 3. Tier B — related-work / passing mention only (6 of 32) ⚠️ cite, but not as method grounding

| # | Paper | Extent of use |
|---|---|---|
| 19 | Yang, *Laplace-LoRA* | Related-work table row only (`docs/DEFENCE_SLIDE_PLAN.md:324`, `SUPERVISOR_MEETING_PREP.md:167`) |
| 20 | Wang, *BLoB* | Same table rows. (The `BLoB` grep hits in `src/vgpu/` are `save_blob()` — unrelated.) |
| 21 | Gao, *EDL Survey* | One substantive use: `docs/DEFENCE_BRIEF.md:150` (KL-regulariser design) |
| 22 | Finn, *MAML* | Comparison table row + literature map; no implementation |
| 23 | Lakshminarayanan, *Deep Ensembles* | One line, `step_writeups/step11.txt:561` (ensemble cost comparison) |
| 24 | Amini, *Deep Evidential Regression* | One line, `step_writeups/step1.txt:411`. It is a *regression* paper — the thesis is classification |

## 4. Tier C — never used anywhere (8 of 32) ❌ do not cite unless you add the content

Verified by targeted grep across `src/ scripts/ docs/ step_writeups/ configs/ tests/ notebooks/ *.txt *.md`.
Zero substantive hits each.

| # | Paper | Note |
|---|---|---|
| 25 | Meo, *Bayesian-LoRA* | 0 hits. The summary itself warns it is quantization/compression, **not** uncertainty |
| 26 | Chai, *BH-PEFT* | 0 hits (the "Chai" grep matches only `chair` in the CIFAR-100 class list) |
| 27 | Shi, *Training-Free Bayesianization (TFB)* | 0 hits |
| 28 | Ulmer, *Prior and Posterior Networks* | 0 hits |
| 29 | Han, *PEFT for Large Models: A Survey* | 0 hits |
| 30 | Liang, *ODIN* | 0 hits — ODIN was never implemented as an OOD score |
| 31 | Lee, *Mahalanobis* | Single hit: `docs/superpowers/specs/…step4-5…-design.md:69` "No Mahalanobis OOD score (defer to Step 7)". Step 7 never added it |
| 32 | Liu, *ConvNeXt* | Only appears as **dropped scope** — "ConvNeXt-Nano … deferred", Step 12 backlog |

These four Bayesian-PEFT papers (25–27 plus Laplace-LoRA/BLoB in Tier B) are the *motivating*
literature for the thesis title, so a Related Work chapter can legitimately cite them as
background. What they cannot support is any implementation claim — nothing in `src/` derives
from them. Say "background", not "method".

---

## 5. Missing — used in the repo, absent from the list

### 5.1 🔴 Critical: datasets and backbone — currently uncited *anywhere* in the repo

Without these the protocol section cannot be written, and the frozen splits have no provenance.

| Work | Why it is mandatory | Current state |
|---|---|---|
| **Bertinetto et al.**, *Meta-learning with differentiable closed-form solvers* (R2D2) | Defines the **CIFAR-FS split** — the frozen `data/cifar_fs_split.json`. 126 references in the repo, no citation | Named, never cited |
| **Ravi & Larochelle**, *Optimization as a Model for Few-Shot Learning* | Defines the **MiniImageNet 64/16/20 split** (`src/datasets/mini_imagenet.py:1`). `step_writeups/step9.txt:118` **explicitly flags this as an unfilled gap** | Named, never cited |
| **He et al.**, *Deep Residual Learning* (ResNet) | The primary backbone, every result | 1 passing mention |
| **Krizhevsky**, CIFAR-10/100 tech report | Source of CIFAR-FS and the CIFAR-100-heldout near-OOD pool | 0 citations |
| **Netzer et al.**, *Reading Digits in Natural Images* (SVHN) | The far-OOD pool, 219 references | 0 citations |
| **Le & Yang**, *Tiny ImageNet Visual Recognition Challenge* | The TinyImageNet near-OOD pool | 0 citations |
| **Deng et al.**, *ImageNet* | The pretraining source the whole frozen-backbone argument rests on | 0 citations |

### 5.2 🟠 Closest prior work — the docs say these must be cited and differentiated

| Work | arXiv | Role |
|---|---|---|
| Li, Liu & Bilen, **TSA** (CVPR 2022) | 2107.00358 | **Primary closest prior work for RQ1** — appears in 9 separate docs |
| Bateni et al., **FiT** | 2206.08671 | Same backbone/protocol/head design |
| Zhang et al., **Tip-Adapter** | 2207.09519 | Closest "frozen backbone + adapter + few-shot" recipe |
| Gao et al., **CLIP-Adapter** | 2110.04544 | Same |
| Morales-Álvarez et al., **BayesAdapter** (IJCV) | 2412.09718 | Second confirming case for the RQ2 boundary |
| **BEL** — Bayesian Evidential Learning for Few-Shot | 2207.13137 | The nearest evidential few-shot work; appears in the results tables |
| Hu et al., **P>M>F** (CVPR 2022) | 2204.07305 | Source of the §4.4.1 accuracy comparison numbers |
| Lee et al., **MetaOptNet** (CVPR 2019) | 1904.03758 | Source of the entire §4.2 baseline table |
| **MetaQDA** | 2101.02833 | RQ2-boundary comparison row |

### 5.3 🟠 Concurrent work the docs instruct you to cite *proactively*

These are open action items (`T8.1`, `T4.3`, etc. in `docs/NEW_RQS_TASK_PLAN.md`) — each is a
near-miss a reviewer could raise as prior art.

| Work | arXiv |
|---|---|
| *One Model, Many Behaviors* (WACV 2026) — objective × score ANOVA, overlaps RQ1 | 2601.10836 |
| *A Systematic Comparison of Training Objectives for OOD Detection* (2026) — RQ2 novelty anchor | 2603.07571 |
| Softmax as a special case of evidential (theory) | 2605.22746 |
| Vacuity class-cardinality critique | 2605.06382 |
| Accuracy-preserving post-hoc calibration via invertible logit transforms | 2608.10372 |
| Density-informed EDL recalibration | 2602.01477 |
| *Invascal* | 2606.00069 |
| Epistemic calibration in second-order classification | 2606.10777 |
| CP tensor adapters (2026) | 2606.00428 |
| *A Benchmark Study on Calibration* (ICLR 2024) | 2308.11838 |
| Calibration double descent | 2302.09369 |

### 5.4 🟠 Contradictions and robustness — the thesis has prepared responses to these

| Work | arXiv | Note |
|---|---|---|
| **LoRA-Ensemble** | 2405.14438 | The documented partial contradiction — `RQ_RESULTS_SUMMARY.md` §7 has a backup slide for it |
| On Fairness of Low-Rank Adaptation | 2405.17512 | Ruled out as prior art for RQ3 |
| EDL hyperparameter sensitivity | 2510.08938 | Cited to pre-empt "your EDL was undertuned" |
| EDL hyperparameter sensitivity | 2410.00393 | Same |
| *LoRA vs Full Fine-tuning: An Illusion of Equivalence* | 2410.21228 | ⚠️ `NEW_RQS_TASK_PLAN.md` flags this citation as **needing re-verification before use** |

### 5.5 🟡 Calibration / uncertainty theory

| Work | arXiv |
|---|---|
| Minderer et al. 2021, *Revisiting the Calibration of Modern Neural Networks* | 2106.07998 |
| Kull et al. 2019, *Dirichlet calibration* (NeurIPS) | — (name collision with our Dirichlet head; the docs require an explicit differentiation paragraph) |
| Bengs, Hüllermeier & Waegeman (NeurIPS 2022) | 2203.06102 |
| Shen et al., *Is EDL a Mirage?* (NeurIPS 2024) | 2402.06160 |
| Bitterwolf et al. 2022 | — |
| Charpentier et al. 2020 (PostNet) | — (cited at `step_writeups/step1.txt:411`) |

### 5.6 🟡 Positioning — backbone currency, edge PEFT, Bayesian PEFT landscape

Used mainly in `docs/DEFENCE_BRIEF.md` to answer "isn't a CNN backdated in 2026?" and to show
the reporting gap.

| Work | arXiv |
|---|---|
| TinyML → TinyDL survey | 2506.18927 |
| CNN-vs-ViT few-shot low-data study | 2510.04794 |
| DINOv3 | 2508.10104 |
| SSF (VTAB-1k budget table) | 2210.08823 |
| PEFT for Pre-Trained Vision Models: A Survey | 2402.02242 |
| CoOp | 2109.01134 |
| LoRA-C | 2410.16954 |
| LoRA-Edge | 2511.03765 |
| CoLoRA | 2505.18315 |
| Scalable Bayesian LoRA | 2506.21408 |
| Calibrated Adaptation (Stiefel-Bayes) | 2602.17809 |
| BaRA | 2606.29184 |
| Bayesian Sparse LoRA | 2607.02182 |
| Bayesian Adaptation Gym | 2606.22188 |
| *Revisiting convolutional design for efficient CNN architectures* (Sci. Reports 2025) | — |
| **Bayesian-PEFT** (Pandey, Pyakurel, Yu, NeurIPS 2024) | — ⚠️ **unverified** — repeated extraction failures; `SUPERVISOR_PRESENTATION_SLIDES.md:607` flags it as the single closest-*named* paper and an open risk |
| Close-call preprint checked and ruled out | 2602.15283 |

### 5.7 ⚪ Comparison-table rows (secondary citations, via MetaOptNet / P>M>F tables)

Relation Networks (Sung), TADAM (Oreshkin), LEO (Rusu), DINO (Caron), CLIP (Radford).
These appear as rows in `docs/RESULTS_MASTER.md` §4.2–4.4. Cite the source table if you prefer,
but each row's method should be attributable.

---

## 6. What to do

1. **Fix the count** — the list is 32, not 33.
2. **Add §5.1 immediately.** Seven citations, all mandatory, all currently missing. The
   Bertinetto and Ravi & Larochelle gaps are the worst: the frozen splits are the repo's
   central reproducibility claim and they have no provenance in the bibliography.
3. **Move Tier C (§4) to a Background subsection or drop it.** Citing ODIN, Mahalanobis,
   ConvNeXt, or the three unused Bayesian-PEFT papers as if they informed the method is not
   defensible — a supervisor who checks will find nothing behind them.
4. **Merge §5.2–5.6 in.** These are not optional extras; `RQ_RESULTS_SUMMARY.md` and
   `DEFENCE_BRIEF.md` already build arguments on them, and several are recorded as open
   pre-emptive-citation action items.
5. **Resolve the two flagged uncertainties** before the bibliography is frozen:
   arXiv:2410.21228 (needs re-verification) and Pandey et al. *Bayesian-PEFT* (unverified).
6. **Pick a style now** (IEEE numeric per `DEFENCE_SLIDE_PLAN.md` SLIDE 32) and add a
   `refs.bib` — at ~92 entries this is past the point where manual management is safe.

---

## 7. Verification pass (2026-09-13) — `docs/refs.bib` built, 3 errors found

`docs/refs.bib` now holds the **citation set only**: 83 entries, the 8 read-but-unused papers
excluded. Every entry carries a provenance tag — `[V-web]` 44 (metadata fetched from arXiv this
session), `[V-pdf]` 24 (read off the PDFs, via `PAPER SUMMARIES/`), `[CHECK]` 15 (not
machine-verified — confirm before submission).

Coverage was re-checked mechanically: every arXiv work cited anywhere in `docs/` or `progress.txt`
now has an entry. No duplicate keys, no unbalanced braces.

All 48 arXiv identifiers in the repo resolve to real papers. But three are **wrong in a way that
matters**:

### 🔴 7.1 `arXiv:2302.09369` is not the paper the docs claim

`NEW_RQS.md`, `NEW_RQS_TASK_PLAN.md` and `RQ_RESULTS_SUMMARY.md` cite it as the *"calibration
double descent"* phenomenon, supporting the demoted interior-optimum hypothesis (former RQ5).

It is actually **"Calibrating the Rigged Lottery: Making All Tickets Reliable"** (Lei, Zhang, Xu,
Mallick, 2023) — about calibration under **sparse training / lottery-ticket pruning**. Not double
descent, not parameter-budget curves.

**Action:** find a real source for "calibration double descent" or delete the sentence. It is in
`refs.bib` as `lei2023riggedlottery` with a warning comment so the claim cannot be carried forward
silently.

### 🔴 7.2 FiT is attributed to the wrong authors

`DEFENCE_BRIEF.md`, `RESULTS_MASTER.md` and `SUPERVISOR_PRESENTATION_SLIDES.md` all cite
arXiv:2206.08671 as **"Bateni et al., FiT"**. The actual authors are **Shysheya, Bronskill,
Patacchiola, Nowozin and Turner**, and it is **ICLR 2023**, not a preprint. (Bateni is an author on
Simple CNAPS — a different paper.) Corrected in `refs.bib` as `shysheya2023fit`; the three docs
still carry the wrong name.

### 🟠 7.3 `arXiv:2510.04794` is being used outside its domain

`DEFENCE_BRIEF.md` §2 uses it to argue CNNs stay competitive with ViTs in low-data **few-shot
classification**. The paper is *"A Comparative Study of Vision Transformers and CNNs for Few-Shot
**Rigid Transformation and Fundamental Matrix Estimation**"* — geometric estimation, not
classification. The quoted sentence is real, but the task domain is not ours.

**Action:** re-scope the claim ("in few-shot geometric estimation, …") or drop the citation. It is
the weakest link in the "is a CNN backdated in 2026?" defence and an examiner who opens the paper
will see it immediately.

### Still unresolved (carried over)

- **Pandey et al., "Bayesian-PEFT" (NeurIPS 2024)** — still unverified; in `refs.bib` with a
  `note` field marking it. It is described as the closest-*named* paper, so this needs a human read.
- **arXiv:2410.21228** — metadata confirmed (Shuttleworth, Andreas, Torralba, Sharma), but
  `NEW_RQS_TASK_PLAN.md` flags the *use* of its claim as needing re-verification. Unchanged.

### Minor corrections folded into `refs.bib`

- CLIP-Adapter is now published in **IJCV** (accepted 2025), not a bare preprint.
- LoRA-Ensemble is **TMLR 2026**, not a bare preprint.
- CoLoRA (2505.18315) is specifically an **OCT medical-imaging** study — noted inline.
- MetaQDA's real title is *"Shallow Bayesian Meta Learning for Real-World Few-Shot Recognition"*
  (ICCV 2021); the docs only ever use the method name.

---

## 8. Role classification (2026-09-13) — which chapter each citation serves

Every entry in `docs/refs.bib` now carries a `% ROLE:` tag above it. Three roles, because they map
to three different chapters and three different standards of care:

- **`IMPL`** — the implementation, protocol or data actually derives from it. Methodology chapter.
  If challenged, you must be able to point at the file. These are the citations that have to be right.
- **`COMP`** — its numbers appear in our comparison tables, or we benchmark against it. Results
  chapter. The risk here is protocol mismatch, not attribution.
- **`LIT`** — discussed, positioned against, or pre-empted. Related Work. No number of ours depends
  on it.

Entries can hold two roles (e.g. ProtoNet is both the head we implement and a row in the baseline table).

| Role | Entries |
|---|---:|
| `IMPL` only | 11 |
| `IMPL+COMP` | 8 |
| `IMPL+LIT` | 6 |
| `COMP` only | 10 |
| `COMP+LIT` | 14 |
| `LIT` only | 38 |
| **Total** | **87** |

Counting each role separately: **25 implementation**, **32 comparison**, **58 literature**.

### The 25 `IMPL` citations — the load-bearing set

These are the ones an examiner can check against the code.

| Citation | What in the repo depends on it |
|---|---|
| `houlsby2019adapters` | `src/adapters/bottleneck.py` |
| `hu2022lora` | `src/adapters/lora.py` |
| `benzaken2022bitfit` | `src/adapters/bitfit.py` |
| `aleem2024convlora` | `src/adapters/lora.py:22` — the conv-form LoRA |
| `chen2024convadapter` | `src/adapters/placement.py:27` — the 1×1 locality caveat |
| `zhong2024convlorasam` | Step 5 justification for conv-LoRA in low-data |
| `he2022unified` | Step 6 serial-vs-parallel taxonomy (RQ1) |
| `sensoy2018edl` | `src/losses/evidential.py`, `PrototypeHead.to_evidence()` |
| `snell2017protonet` | `src/heads/prototype_head.py` + the episodic protocol |
| `guo2017calibration` | `src/evaluators/temperature.py` |
| `hendrycks2017baseline` | The MSP score in the evaluator |
| `liu2020energy` | The energy score in the evaluator |
| `malinin2018priornetworks` | `src/datasets/tinyimagenet_ood.py:7` — near-OOD rationale |
| `yang2022openood`, `zhang2024openoodv15` | Near/far OOD split design |
| `chen2019closerlook` | `configs/exp_phase2_softmax.yaml:26` — linear-probe baseline |
| `he2016resnet` | `src/backbones/resnet18.py` |
| `howard2019mobilenetv3` | `src/backbones/mobilenetv3.py` |
| `bertinetto2019r2d2` | The frozen CIFAR-FS split |
| `ravi2017optimization` | The frozen MiniImageNet 64/16/20 split |
| `vinyals2016matching` | MiniImageNet origin |
| `krizhevsky2009cifar`, `netzer2011svhn`, `le2015tinyimagenet`, `deng2009imagenet` | The datasets and pretraining source |

### Four more papers added in this pass

Found by auditing the comparison tables themselves rather than the prose:

- **`jia2022vpt`** (Visual Prompt Tuning, ECCV 2022) — **`COMP`**. VPT-Deep and VPT-Shallow are rows
  in the VTAB-1k budget table (`DEFENCE_BRIEF.md` §3.3, `RESULTS_MASTER.md` §4.4.3) with real
  numbers attached. It had no entry.
- **`chen2022adaptformer`**, **`jie2023fact`**, **`zhang2022noah`** — **`LIT`**. Named in the
  field-coverage matrix (`RESULTS_MASTER.md` §4.6) as examples of ViT-side PEFT that report no
  calibration or OOD. No numbers, but the table names them.

### Where the risk sits

- The `[CHECK]` and `IMPL` sets overlap on five entries — `he2016resnet`, `deng2009imagenet`,
  `krizhevsky2009cifar`, `netzer2011svhn`, `le2015tinyimagenet`. These are canonical and stable, but
  they are Methodology-chapter citations that were never machine-verified. Spot-check them first.
- `ravi2017optimization` is `IMPL+COMP` **and** `[CHECK]` — it fixes a frozen split *and* supplies a
  baseline row, and its metadata came from a search rather than the paper. Verify it directly.
- Every `LIT`-only entry is lower stakes: if one is wrong, a sentence changes, not a result.

---

## Provenance

- Listed-paper ground truth: `PAPER SUMMARIES/*.txt`, `==== PAPER` headers.
- Usage evidence: recursive grep over `src/ scripts/ configs/ docs/ step_writeups/ tests/ notebooks/`
  and the root `.txt`/`.md` trackers.
- Missing-citation evidence: all `arXiv:NNNN.NNNNN` occurrences in `docs/*.md`, `progress.txt` and
  `notebooks/*.ipynb` (48 distinct IDs, 5 of which are already on the list), plus named-author
  searches for works cited without an arXiv ID.
