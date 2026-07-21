# Step 6 — Adapter Placement Study (RQ1) — Design

**Date:** 2026-07-21
**Phase:** 3 · **RQ:** RQ1 (primary — this is where RQ1 is answered)
**Spec source:** implementation.txt §6 · progress.txt STEP 6 · thesis_implementation_instructions.txt (paper-grounded reasoning)
**Depends on:** Step 5 (CLOSED)

## 1. Goal

Answer RQ1 — *adapter placement (serial vs parallel) as an accuracy vs. parameter-count
tradeoff* — by inserting the **same Bottleneck-style adapter** at three positions in a
frozen ResNet-18 and comparing them on CIFAR-FS (5-way 5-shot, 600 test episodes) for
**both heads** (evidential + softmax):

- **post_pool** — existing Step-4/4.5 Bottleneck on the pooled `(B,512)` feature (baseline of record).
- **serial** — adapter in-line inside the ResNet blocks (sequential composition).
- **parallel** — adapter alongside the ResNet blocks (parallel composition).

Deliverables: 4 new result JSONs (+ 2 reused post_pool), a comparison bar chart, and
`step_writeups/step6.txt` with the RQ1 Pareto paragraph.

## 2. Paper-grounded reasoning (per thesis_implementation_instructions.txt §5)

- **Which papers:** Conv-Adapter (Chen 2022, PEFT_summaries §4) — the single most direct
  precedent: an explicit serial-vs-parallel placement study *inside* CNNs. Unified View
  (He 2022, PEFT_summaries §7) — supplies the vocabulary: *insertion form* (sequential vs
  parallel) × *composition* (plain add `Δh` vs scaled add `Δh·s`).
- **PROS / FIT:** cleanly answers RQ1 by holding the adapter *form* fixed and varying only
  *where* it sits. Conv-Adapter reports **parallel > sequential for classification**; Step 6
  tests whether that replicates in *our* setting (few-shot, frozen backbone, parameter-free
  prototype head, evidential + softmax) — a genuinely open question, not a re-proof.
- **CONS / caveat (flagged):** we use a **1×1 channel bottleneck**, which Conv-Adapter
  names as the form that "loses locality" (their core failure mode; they prefer depth-wise
  separable K×K). We accept this **deliberately** to isolate placement — the alternative
  (depth-wise-separable) would change the adapter's identity and confound placement with
  adapter-form. Documented so that weak in-block numbers read as an honest RQ1 result.
- **Deviation from the paper:** plain **add** composition for *both* serial and parallel
  (no learnable scale). Unified-View's scaled-parallel is a known improvement we skip on
  purpose, so serial vs parallel differ *only* in insertion position.
- **Untested assumption carried in:** the evidence-affine (scale 2, bias −6) was
  VAL-selected on *post-pool* Bottleneck features (Step 4.5); reused unchanged for the
  in-block placements (same as Step 5's carried-in assumption). Per-placement VAL tuning is
  out of Step 6 scope.

## 3. Adapter form (decided)

**1×1 channel bottleneck**, the conv analogue of the post-pool `Linear` Bottleneck, held
identical across all placements:

```
Conv1x1Bottleneck(C, r):
    down: Conv2d(C, r, kernel_size=1)
    act:  ReLU
    up:   Conv2d(r, C, kernel_size=1)   # weight & bias ZERO-init -> identity at start
    body(z) = up(act(down(z)))          # (B,C,H,W) -> (B,C,H,W)
```

Rank `r = 16` (matches post-pool Bottleneck). Zero-init `up` ⇒ the placed model is
byte-identical to the frozen backbone at initialisation (required by the identity test).

## 4. Placement mechanism

New file **`src/adapters/placement.py`**:

- `register_serial_adapter(block, mod)` — forward hook: `out' = out + mod.body(out)`
  (adapter transforms the block **output**, in-line / sequential).
- `register_parallel_adapter(block, mod)` — forward hook: `out' = out + mod.body(input[0])`
  (adapter runs on the block **input**, summed at the output / parallel).
- `PlacementAdapter(spec, dim, backbone)` — holds a `ModuleList` of per-stage
  `Conv1x1Bottleneck`s, registers the hooks on the chosen blocks, advertises
  `backbone_trainable = True`, and its own `forward` is **identity** on the pooled vector
  (the adaptation lives inside the backbone — same pattern as the LoRA adapter).

**Placement points:** `cfg.adapter.block_ids` selects **stages** (layer1–4, ids `[0,1,2,3]`,
default all four). Within a selected stage the adapter attaches to the **final BasicBlock**,
where block input and output share shape `(C,H,W)` — so both serial and parallel compose
without any shape mismatch. `C ∈ {64,128,256,512}` for stages 0–3.

**Wiring (the one subtle point):** placement adapters live *inside* the frozen backbone, so
the backbone must run **with autograd** or no gradient reaches them. We reuse the existing
`backbone_trainable = True` path in `BPEFTModel.adapter_features` (backbone weights stay
`requires_grad = False`; only the inserted bottlenecks train) — identical to how
LoRA/BitFit already work. The inserted modules live in `model.adapter` (not inside
`model.backbone`), so the trainer's `freeze_backbone` logic never touches them and the
optimiser picks them up via `[p for p in model.parameters() if p.requires_grad]`.
**No trainer/evaluator changes required.**

Minor, principled deviation from the spec wording: a forward hook fires at the *block
boundary* (after the block's own residual add), i.e. Conv-Adapter's "Residual-Sequential /
Residual-Parallel" schemes — the natural reading of "forward-hook … at the block output."

## 5. Config surface

Extend the `bottleneck` adapter with a placement selector (keeps "same adapter, different
placement" semantics):

```yaml
adapter:
  type: bottleneck
  rank: 16
  placement: post_pool | serial | parallel   # default post_pool (existing behaviour)
  block_ids: [0, 1, 2, 3]                     # stages to place on (serial/parallel only)
```

`build_adapter` routes `placement in {serial, parallel}` to `PlacementAdapter`; `post_pool`
(or absent) keeps the existing `BottleneckAdapter`. **4 new configs:**
`configs/exp_phase3_placement_{serial,parallel}_{evidential,softmax}.yaml`, each extending
the Step-4.5 evidential-retuned / softmax phase-2 config and overriding only `adapter.*`.

## 6. Runs, results, plot, writeup

- Train + eval the 4 new configs, 600 episodes, `--use-tinyimagenet`. Use a distinct
  `--results-suffix` per placement so filenames don't collide:
  `results/phase3_placement_serial_bottleneck_prototype-{head}_metrics.json` and
  `..._placement_parallel_...`.
- **Reuse** the existing post-pool Bottleneck JSONs (`results/step45_bottleneck_*`) as the
  third placement — no re-run. → satisfies "3 placements × 2 heads = 6 JSONs".
- **Plot:** `scripts/step6_placement_plot.py` — grouped bar chart of accuracy / OOD-AUROC /
  ECE for 3 placements × 2 heads, 300 dpi PNG under `results/`.
- **Writeup:** `step_writeups/step6.txt` (same structure as step5.txt) with the RQ1 Pareto
  paragraph: "serial gives X% acc / Y params; parallel X'/Y'; post-pool X''/Y''; best
  Pareto position is [variant]", plus whether Conv-Adapter's parallel>serial replicated.

## 7. Tests — `tests/test_placement.py`

1. **Shape-preservation smoke test** (spec RISK): forward a random `(2,3,224,224)` tensor
   through a hooked model → output shape unchanged vs the frozen backbone.
2. **Identity at init:** zero-init `up` ⇒ placed-model pooled output equals the frozen
   backbone's, for both serial and parallel.
3. **Trainability:** only adapter params have `requires_grad=True`; backbone params stay
   frozen; param counts logged and match the analytical `Σ 2·C·r` over selected stages.
4. **Gradient flow:** after one backward, the inserted adapter params have non-None grads.

## 8. Kaggle notebook — `notebooks/step6_placement.ipynb`

Kaggle-flavoured (not Colab). Environment: `/kaggle/working` (writable, persists as
notebook output), `/kaggle/input/<dataset>` (read-only attached data), GPU = Kaggle T4
(Settings → Accelerator), **Internet = ON** (Settings). No `google.colab`, no Drive mount.

**Staging (decided): git clone repo + attached Kaggle Dataset for data.**

Cells:
0. **GPU check** — `torch.cuda.is_available()` assert (Kaggle T4).
1. **Clone + install** — `git clone https://github.com/notAvailable73/thesis.git /kaggle/working/thesis` (Internet ON), `chdir`,
   `pip -q install -r requirements.txt`, assert `src/adapters/placement.py` present.
2. **Stage data from the attached Kaggle Dataset** — copy from
   `/kaggle/input/<bpeft-data>/` into the repo's `data/` tree at the paths the loaders
   short-circuit on: `cifar-100-python.tar.gz → data/`, `tiny-imagenet-200.zip → data/`,
   `test_32x32.mat → data/svhn/`. (md5-check CIFAR-100 = `eb9058c3a382ffc7106e4002c42a8d85`.)
   No external downloads — Kaggle-native equivalent of the Colab Drive-staging trick.
   *One-time user setup:* create a Kaggle Dataset containing those three archives at its
   root and attach it to the notebook.
3. **Build split** — `python scripts/build_cifar_fs_split.py`; assert canonical (not synthetic).
4. **(optional) pytest** — `python -u -m pytest -v` (should stay green: existing 91 + placement tests).
5. **Run the 4 placement configs** — loop `serial/parallel × evidential/softmax`,
   train.py + evaluate.py (600 ep, `--use-tinyimagenet`, distinct `--results-suffix`),
   skipping any config whose metrics JSON already exists (resumable).
6. **Summary table** — read the 4 placement JSONs + the 2 post-pool refs into a DataFrame.
7. **Plot** — run `scripts/step6_placement_plot.py`.
8. **Persist** — results already live under `/kaggle/working/thesis/results` (persisted as
   output); note "Save Version" to snapshot. No Drive copy needed.

## 9. Exit criteria (from implementation.txt §6)

- [ ] ≥3 placement variants (post_pool reused + serial + parallel) × 2 heads → 6 result JSONs.
- [ ] Trainable param counts reported per placement.
- [ ] RQ1 paragraph written (serial X/Y, parallel X'/Y', post-pool X''/Y'', best Pareto).
- [ ] `pytest tests/` still passes with `test_placement.py` added.

## 10. Non-goals / YAGNI

- No depth-wise-separable / K×K adapter (would confound placement — explicitly out).
- No learnable/scaled composition (kept plain-add to isolate placement).
- No per-placement VAL re-tuning of the evidence-affine (Step 10's grid, not here).
- No new trainer/evaluator code paths — placement reuses the `backbone_trainable` wiring.
- Not committed to git by the assistant (repo convention: a human reviews and commits).
