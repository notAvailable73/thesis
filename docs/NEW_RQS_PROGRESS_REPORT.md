# New-RQs Progress Report — Phase A landed, Phase B pending

**Date:** 2026-08-25
**Scope:** the Kaggle session in `notebooks/notebookde0b8011ea.ipynb` (an executed copy of
`notebooks/new_rqs.ipynb`), plus `results/new_rqs_results.zip` and `results/new_rqs_logits.zip`.
**Status of this document:** independent audit. Every number in §3–§4 was **recomputed from the
per-cell JSONs in the results zip**, not transcribed from the notebook's stdout, per the repo
convention since Step 9.

---

## 1. Headline

Phase A ran cleanly and produced the strongest new result the project has: **RQ-C (the
objective × score factorial) is decisively answered, and it says the uncertainty benefit is
almost entirely a property of the readout score, not of evidential training.**

Phase B — the bottleneck rank sweep, which `docs/Verification.md` calls *"the single most
critical experiment"* — **did not run** (`RUN_PHASE_B = False`). Its configs are generated and
guard-verified; it is ~21 runs / ~6 GPU-h away.

Nothing has been merged into the repo yet. `results/rq_factorial/` and `results/rq5/` do not
exist locally; the results live only inside the two zips.

---

## 2. What actually ran

| | |
|---|---|
| Phase 0 — checkpoint recovery | **99 / 120** Step-10 checkpoints recovered from attached artifacts |
| Pre-flight self-test | **5 / 5 passed** |
| Phase A — T1 factorial + T2 affine refit | **99 cells ok, 0 errors, 21 `no_checkpoint`** |
| Phase B — T5 rank sweep | **skipped** (`RUN_PHASE_B = False`) |
| Wall time | **5.51 GPU-h** across 99 cells (mean 200 s/cell) |
| Artifacts out | `new_rqs_results.zip` (129 files, 0.3 MB) · `new_rqs_logits.zip` (99 `.npz`, 1.04 GB) |

Recovered checkpoint coverage — **the 21 gaps are not randomly distributed**:

| slice | recovered |
|---|---|
| `cifar_fs / 1-shot` | 36 / 36 |
| **`cifar_fs / 5-shot`** | **15 / 36** |
| `mini_imagenet / 1-shot` | 24 / 24 |
| `mini_imagenet / 5-shot` | 24 / 24 |

All 21 missing cells are `cifar_fs/5shot` PEFT cells (parallel + LoRA, both backbones).

---

## 3. The good

**3.1 The regression guard is perfect, and it is the most valuable thing in this run.**
All 99 cells report `regression_guard.status == "exact"` — bit-for-bit agreement with the
committed Step 10 `results/grid/*_metrics.json`, `max|diff| = 0.00e+00`. The new factorial
evaluator provably did not move a single Step-10 number. Given the Step 11 §8.7 incident (a
silent wrong-number bug that no test caught), having this check run per-cell and record its
verdict in the output JSON is exactly the right instinct.

**3.2 VAL discipline held.** I verified independently: **0 of 99** cells used anything other
than VAL seeds `10000..10099` (n=100). Temperature and the evidence affine were fitted on VAL
only, in both arms, by the same procedure. The project's single most important scientific
convention survived a refactor.

**3.3 The self-test earns its 3 minutes.** It proved the diagonal is untouched, that energy on
an evidential model equals `logsumexp` of the same raw logits, that two runs are byte-identical,
and that the rank→param closed form lands on **31,746 at rank 16** — matching the value
`mvt_results.json` already recorded. That last check is what guarantees the rank sweep passes
*through* the Step 10 point rather than parallel to it.

**3.4 RQ-C is a genuinely strong, publishable result.** Recomputed on matched
(design, seed, pool) pairs:

*Holding the readout fixed, does training evidentially help?*

| readout | far-OOD Δ(evid−softmax) | near-OOD Δ |
|---|---|---|
| `msp` | +0.0008 (41/96 wins) | +0.0161 (69/96) |
| `energy` | **−0.0159** (28/96) | +0.0043 (50/96) |
| `ts_msp` | +0.0047 (47/96) | +0.0188 (76/96) |
| `vacuity` | **−0.0176** (34/96) | +0.0113 (75/96) |

*Holding the training fixed, does the readout matter?*

| | far-OOD `energy − msp` | far-OOD `vacuity − msp` |
|---|---|---|
| trained evidential | **+0.1252** | +0.1197 |
| trained softmax | **+0.1395** | +0.1358 |

The readout moves AUROC by **~0.13**; the training objective moves it by **≤0.019, and on
far-OOD in the wrong direction**. Vacuity computed on a *softmax-trained* model works fine — in
fact better than on an evidential-trained one. The variance attribution agrees: `score` η² is
0.437 vs `objective` η² 0.0027 on far-OOD.

**3.5 The energy-vs-vacuity correction is confirmed and strengthened.** Matched head-to-head,
energy beats vacuity **162/198 far** (+0.0045) and **183/198 near** (+0.0104). This independently
re-confirms the 2026-08-06 correction in `CLAUDE.md` / `progress.txt` at higher resolution.

**3.6 Persisting the logits is the best structural decision in the session.** 1.04 GB of
`results/rq_logits/*.npz` means every future post-hoc scoring question is a re-analysis, not a
retrain. This session cost 5.5 GPU-h precisely because that dump didn't exist before. It now does.

**3.7 The Step 11 lesson was applied.** The session log was teed to disk and travels in the
results zip, so a crash would have been diagnosable by design rather than by luck.

---

## 4. The bad

### 4.1 `full_ft` and `linear_probe` have **zero seed variance by construction** — their error bars are fictitious

Their three "seeds" produce **identical results to 6 decimal places** — same accuracy, same
temperature, same `best_val_epoch`, same everything — while `bottleneck` and `lora` cells vary
normally.

```
cifar_fs/5shot/r18/full_ft/evidential   seed 42/43/44 -> acc 0.888178 / 0.888178 / 0.888178, T=0.8027 all
cifar_fs/5shot/r18/linear_probe/evid    seed 42/43/44 -> acc 0.874133 / 0.874133 / 0.874133, T=0.2209 all
cifar_fs/1shot/r18/bottleneck/evid      seed 42/43/44 -> acc 0.789111 / 0.788622 / 0.797933   (varies, as expected)
```

**Root cause** (traced in-tree, not inferred): `configs/base.yaml:79` sets
`train_seed_offset: 20000` as a **fixed constant**, and
[scripts/train.py:270](scripts/train.py#L270) reads it directly — the training episode stream is
`seed_offset + episode_idx`, so it is **identical across `cfg.seed` values**. `set_seed(cfg.seed)`
therefore only perturbs *random weight initialisation*. `bottleneck` and `lora` have randomly
initialised adapters, so they vary. `full_ft` starts from deterministic pretrained ResNet-18
weights and `linear_probe` has 2 config-initialised parameters — **neither has any randomness at
all**, so all three seeds retrace the same trajectory.

The checkpoint `.pt` files differ in CRC (they embed the seed), which is why this was never
noticed — but the Step 10 artifact zips already show it plainly: the per-seed `reliability.png`,
`ood_histogram.png` and `confusion_matrix.png` are **CRC-identical across seeds 42/43/44** for
both adapters.

**Consequences, in order of severity:**
- Any seed CI / std / error bar reported for `full_ft` or `linear_probe` — in Step 10,
  Step 11's Pareto analysis, `docs/RESULTS_MASTER.md`, and now this run — is **exactly zero by
  construction**, not a measurement.
- More broadly: the 3 seeds used everywhere in this thesis measure **initialisation variance
  only, never data-order variance**. That is a narrower claim than "3 seeds" normally implies and
  should be stated explicitly wherever seed robustness is asserted.
- Roughly two-thirds of the GPU time spent on `full_ft` and `linear_probe` cells (across Step 10
  *and* this run) re-computed a result already on disk.

This is **inherited from Step 10's design, not introduced by this session** — but this session is
where it became visible.

### 4.2 The η² `balanced=True` flag does not catch the imbalance that is actually present

`rq_aggregate.eta_squared` computes `balanced = len(set(cell_counts.values())) == 1`. That checks
whether all **present** cells have equal counts — it can never see a cell that is **entirely
absent**, because an empty cell contributes no key to the dict.

`cifar_fs/5shot/mobilenetv3_small/lora` has a softmax arm (3 seeds recovered) and **no evidential
arm** (those checkpoints are in the 21-cell gap). So one whole (design × objective) cell is
missing, the design is not orthogonal, and the flag still reports `balanced: true`. The module's
own docstring says this flag exists because *"a silently-wrong variance attribution is precisely
the failure mode T4.1 flagged"* — the guard has the failure mode it was written to prevent.

I recomputed η² dropping that design entirely:

| pool | | n | η²(score) | η²(design) | η²(objective) | score/objective |
|---|---|---|---|---|---|---|
| far | as reported | 792 | 0.437 | 0.232 | 0.0027 | **163×** |
| far | balanced subset | 768 | 0.440 | 0.225 | 0.0011 | **394×** |
| near | as reported | 792 | 0.130 | 0.569 | 0.0060 | 21.6× |
| near | balanced subset | 768 | 0.129 | 0.570 | 0.0069 | 18.8× |

**The qualitative conclusion is robust** (score ≫ objective either way — this is good news for
RQ-C). But the **headline ratio is not stable**: 163× vs 394× on far-OOD, because η²(objective)
is so near zero that the ratio is numerically fragile. *Do not quote "163×" in the thesis.* Quote
the η² values, or say "two-to-three orders of magnitude."

### 4.3 The 21-cell gap sits exactly where the thesis is most quoted

The missing cells are all `cifar_fs / 5-shot` PEFT cells — parallel bottleneck and LoRA, on both
backbones. That slice contains the project's most-cited configuration (the Step 4.5 / RQ1
headline, `r18 / parallel / 5-shot`). So the new factorial has its **thinnest coverage precisely
where the thesis makes its strongest claims**. Reviewers will notice.

### 4.4 The RQ-B budget curve cannot be built at 5-shot from this run

Only 2 of the 4 budget points survived at `cifar_fs/5shot/resnet18` (`linear_probe` and
`full_ft`) — and by §4.1 those are the two with **no seed variance at all**. The 1-shot curve is
complete, and it *does* show the claimed U-shape:

| params | adapter | accuracy | ECE (evidential native) | ECE (TS-softmax) |
|---:|---|---:|---:|---:|
| 2 | `linear_probe` | 0.7025 | 0.4397 | 0.0642 |
| 12,290 | `lora` | 0.7376 | 0.2970 | 0.0448 |
| **31,746** | **`bottleneck`** | 0.7919 | **0.2765** ← min | 0.0345 |
| 11,176,514 | `full_ft` | 0.8036 | 0.3207 | 0.0337 |

**But note the last column.** The U-shape exists in *evidential-native* ECE only. Under
temperature-scaled softmax, ECE is essentially **monotonically improving** with budget
(0.0642 → 0.0448 → 0.0345 → 0.0337) — no interior optimum. So RQ-B's framing in
`docs/Verification.md` ("maximizing accuracy silently degrades calibration reliability") is
**not supported in general** — it is a property of the evidential evidence map, not of
calibration. This materially narrows what RQ-B can claim, and it is better to find that now than
in review.

### 4.5 Phase B — the experiment that de-confounds budget — did not run

`docs/Verification.md` §RQ-A and §RQ-B both state the rank sweep is required to convert the
budget claims from correlational to causal, and calls it *"the single most critical experiment."*
It is the only experiment that varies budget with **backbone and adapter family held fixed**.
The configs exist (`configs/rq5/`, 21 files + `_index.json`), the T5.6 "only rank and seed differ"
guard passed (47 keys checked), and the parameter closed form was verified across all 7 ranks
(2,886 → 124,098). It just needs the GPU time.

### 4.6 Three incompatible RQ numbering schemes are now in play

| scheme | source | meaning of "RQ1" |
|---|---|---|
| RQ1–RQ4 | `CLAUDE.md`, `proposal.txt` | adapter placement (serial vs parallel) |
| RQ1–RQ5 | the new notebook, `NEW_RQS_TASK_PLAN.md` | the objective × score factorial |
| RQ-A–RQ-E | `docs/Verification.md` | budget vs architecture attribution |

"RQ1" currently denotes three different questions across documents in the same repo, and
"RQ3" denotes two. This will produce a citation error in the thesis text — it is a
when-not-if. Pick one scheme and propagate before any more prose is written.

### 4.7 Nothing is merged

`results/rq_factorial/` and `results/rq5/` do not exist in the working tree. `progress.txt` has no
entry for this session, `docs/NEW_RQS_TASK_PLAN.md`'s T0/T1/T2 boxes are unticked, and the four
new modules (`rq_core.py`, `rq5_sweep.py`, `rq_drivers.py`, `rq_aggregate.py`) exist only inside
the results zip. Per repo convention these are for a human to review and commit — but until then
5.5 GPU-h of work is one deleted zip away from being lost.

---

## 5. What the results say, stated honestly

**RQ-C (factorial) — answered, and it is a real contribution.** The OOD benefit attributed to
evidential training is ~90% attributable to the readout score. Evidential *training* is worth
+0.011 to +0.019 AUROC on near-OOD and **−0.016 to −0.018 on far-OOD**. Energy remains the best
single readout, beating vacuity 162/198 (far) and 183/198 (near).

**RQ-D (affine recalibration) — works, doesn't change the verdict.** The VAL-fitted evidence
affine improves ECE in **48/48** cells, mean 0.327 → 0.190. But the recalibrated evidential head
is still **11.9× worse than TS-softmax in 48/48 cells**, and **2.18× worse than plain,
uncalibrated softmax in 48/48**. OOD ranking mostly survives (mean ΔAUROC +0.004, min Spearman ρ
0.921), though **42 / 192** comparisons degrade by more than 0.005 — so "ranking is preserved"
should be stated as "largely preserved, with measurable exceptions," and the T2.8 question
(*can a per-logit monotone map reorder vacuity?*) is answered **yes, empirically**.

**Net effect on the thesis story:** these results further weaken the *method* contribution
(evidential/Bayesian PEFT loses to simpler baselines on both calibration and OOD) and further
strengthen the *attribution* contribution. That is the reframing `docs/Verification.md`
recommends, and this run is direct evidence for it.

---

## 6. Recommended next actions, in order

1. **Merge the results zip** into `results/rq_factorial/`, re-run the aggregation locally, and
   record the T0 verdict in `progress.txt` + `docs/NEW_RQS_TASK_PLAN.md`. *(No GPU. Do this
   first — it de-risks everything else.)*
2. **Fix the `balanced` check** to detect absent cells, and re-report η² on the balanced subset.
   Replace the "163×" figure everywhere it has been written down. *(No GPU, ~30 min.)*
3. **Decide the seed question (§4.1).** Either (a) document `full_ft` / `linear_probe` as n=1 and
   strip their error bars from every table — honest and free; or (b) make `train_seed_offset`
   depend on `cfg.seed` and re-run — correct but expensive and it would break the byte-identical
   reproduction of every earlier step. **(a) is the right call**; (b) violates the hard constraint
   carried since Step 8.
4. **Run Phase B** (21 runs, ~6 GPU-h). Highest scientific value per GPU-hour currently
   available, and the configs are already generated and guard-verified.
5. **Fill the 21-cell `cifar_fs/5shot` gap** with `ALLOW_RETRAIN=True` (~7 GPU-h). Needed for the
   5-shot budget curve and to remove §4.3.
6. **Unify the RQ numbering** before writing any more prose.

---

## 7. Verification note

Independently recomputed from `results/rq_factorial/*.json` for this report: the regression-guard
tally (99/99 exact), the VAL-seed tripwire (99/99 correct), the 2×4 factorial on matched pairs,
the η² balanced-subset comparison, the energy-vs-vacuity head-to-head, the RQ-D ECE ratios, and
the budget curve. The seed-variance root cause was traced to `configs/base.yaml:79` and
[scripts/train.py:270](scripts/train.py#L270) in-tree and cross-checked against CRCs in the three
Step-10 artifact zips. Nothing in §3–§5 is transcribed from notebook stdout.
