# B-PEFT Thesis Guide — Start Here

This folder explains the whole project in plain language. It covers what we built, how the research
changed over time, what we found, every experiment we ran, what is still open, and where everything
lives in the repo.

It was written on 2026-09-14, after all experiments were finished.

## The project in one paragraph

We take an image model that was already trained on ImageNet (ResNet-18 or MobileNetV3-Small) and
**freeze** it. We add a **small trainable part** (an adapter) and teach the model new classes from only
1 or 5 example images per class. Then we check three things: is it **accurate**, is its confidence
**honest** (calibration), and can it **tell when an image is from something it has never seen**
(out-of-distribution, or OOD, detection). We compare two ways of reading the model's output: normal
**softmax**, and an **evidential** (Dirichlet) reading that also gives an uncertainty score.

## The main result in five lines

1. **How many example images you have** mostly decides accuracy. **Which output reading you use** mostly
   decides calibration. The two barely affect each other. *(RQ1)*
2. OOD detection depends on **how you score the output**, not on **how the model was trained**. *(RQ2)*
3. Adapter **design** decides accuracy. But calibration depends on the **backbone**, not on how many
   parameters the adapter has. This is our strongest result. *(RQ3)* — independently corroborated on
   2026-09-15 by the `backbone:adapter` interaction term (3.28% of ECE variance, p<1e-6).
4. The evidential head's poor calibration can be **improved after training** by refitting two numbers.
   It got better in 60 of 60 cases, though it is still worse than plain softmax in all 60. *(RQ4)*
5. Several things we expected at the start turned out **wrong**. We report those openly.

## Read in this order

| # | File | What it tells you | Time |
|---|---|---|---|
| 1 | [01_what_we_built.md](01_what_we_built.md) | The model, data, test protocol, and key words explained | 10 min |
| 2 | [02_research_story.md](02_research_story.md) | What we assumed, what actually happened, and why the plan changed | 15 min |
| 3 | [03_results.md](03_results.md) | Final answers to every research question, with numbers and safe wording | 20 min |
| 4 | [04_experiments.md](04_experiments.md) | Every experiment: what ran, how many runs, which files, what was checked | 15 min |
| 5 | [05_problems_and_open_work.md](05_problems_and_open_work.md) | Known problems in our documents, limits, and what is left to do | 10 min |
| 6 | [06_repo_map.md](06_repo_map.md) | Where every file is, how to run things, and the rules of the repo | 10 min |

### Thesis chapter drafts (added 2026-09-14)

Files 01–06 are a *map* of the project. Files 07–10 are *source material for the thesis chapters*, written
in thesis register, with citation keys from `docs/refs.bib` and a source tag on every claim about other
papers.

| # | File | Thesis chapter | What it contains |
|---|---|---|---|
| 7 | [07_introduction.md](07_introduction.md) | Abstract, Ch1 | Abstract draft, motivation, problem statement, challenges, research gap summary, objectives, RQs, contributions with novelty labels, scope, thesis layout |
| 8 | [08_literature_review.md](08_literature_review.md) | Ch2 | Backbones, few-shot learning, PEFT, calibration, Bayesian PEFT, evidential DL, OOD detection; gap synthesis G1–G5 mapped to RQs; citation clean-up list |
| 9 | [09_methodology.md](09_methodology.md) | Ch3–4 | Equations and hyperparameters checked against the code, analysis design per RQ, datasets and OOD pools, reproducibility, deviations table, literature grounding |
| 10 | [10_discussion_and_conclusion.md](10_discussion_and_conclusion.md) | Ch6–7 | The four answers as one picture, interpretation vs prior work, practical guidance, threats to validity, conclusion, future work |

Chapter 5 (Results) is `03_results.md`; §3.11 has the full 40-cell grid tables.

If you only have five minutes, read this page, then the table at the top of `03_results.md`, then
`05_problems_and_open_work.md`.

**Update 2026-09-15 — coverage completed.** The 21 missing Phase A checkpoints were trained and scored:
coverage is now **120/120, fully crossed**, all 21 reproducing the committed grid exactly. RQ1's two-way
interactions were computed at the same time. Read `results/rq_completion/REPORT.md`, then
`04_experiments.md` §4.8. Consequences: RQ2 is now 42.7%/0.17% (far), RQ4 is 60 cells not 48, "main
effects only" is no longer a limitation, and a new data-quality issue was found (a 20-episode smoke run
in the committed grid — `05_problems_and_open_work.md` B5).

**Corrections made on 2026-09-14** (details in `05_problems_and_open_work.md` A6–A8):

- the evidence `(scale, bias)` is **learnable**, initialised at (2, −6), not frozen;
- the prototype head uses **cosine similarity × 10**;
- the grid's `kl_weight_max` 0.1 is not the value the Step 4.5 VAL sweep chose (0.05).

## How this guide relates to the other documents

This guide is a **map and summary**. It does not replace the detailed documents. When you need the full
evidence for a claim, each section here names its source file.

The most important detailed documents are:

- `docs/RQ_SUPERVISOR_REPORT.md` — the formal report on the four research questions
- `docs/RQ_RESULTS_SUMMARY.md` — full evidence for each research question
- `docs/RESULTS_MASTER.md` — all result tables from the 120-run experiment grid
- `docs/DEFENCE_SLIDE_PLAN.md` — the plan for the defence presentation
- `progress.txt` — the step-by-step status log and decision history

**Warning about numbering.** The project has used more than one set of research-question numbers. This
guide uses the **final four** (RQ1–RQ4). The proposal's original questions are called **Orig-RQ1…4**.
`progress.txt` and `step_writeups/step1–11.txt` use the *original* numbering. See
`03_results.md` for how the two sets map to each other.
