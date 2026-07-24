# Sentinel — Accuracy Evaluation Plan

How to measure the real-world accuracy of the **Sentinel** app on real-life data,
end to end. This is the "simulation" workflow: enroll a few reference photos per
product, run a labelled test set through detection, and compute the metrics that
matter for the thesis.

> **Important context.** The Sentinel app in `app/` is an *interactive dashboard*,
> not an accuracy harness. Its API (`app/backend/api/routes.py`) only exposes
> single-image operations — enroll a product, detect **one** image. There is **no
> batch loop, no ground-truth comparison, and no accuracy metric anywhere in the
> app today.** Measuring accuracy therefore requires a small evaluation script
> (see [Phase 3–5](#phase-3--enroll-few-shot-the-known-products)) that drives the
> HTTP API over a whole dataset. This document is the plan for that.
>
> For how the app's ML model works, how to run training, and how the trained
> thesis model would connect to the app, see
> [ml-model-and-training.md](ml-model-and-training.md).

---

## Phase 0 — Understand what "accuracy" means here

The app makes one of three decisions per query image:

| Decision  | Meaning |
|-----------|---------|
| `MATCH`   | Confident and in-distribution → auto-accept as product X. |
| `REVIEW`  | Known but top-1 belief below the floor → ambiguous, a human confirms. |
| `UNKNOWN` | Vacuity ≥ threshold → out-of-distribution / foreign object → manual inspection. |

So "accuracy" is really **two** distinct measurements, and you should report both:

1. **Classification accuracy** — for images of *enrolled* products, is the predicted
   product correct?
2. **OOD detection** — are genuinely unfamiliar items correctly flagged `UNKNOWN`
   instead of being confidently mislabelled?

Decide up front which you are reporting. The thesis story ("honest AI") lives in
measurement (2) plus the evidential-vs-softmax confidence contrast.

---

## Phase 1 — Prepare the real-life data

1. Pick the products to enroll (e.g. 5–20 known products).
2. For **each product**, split its photos into two disjoint sets:
   - a **reference set** (1–10 images) → used for enrollment only,
   - a **test set** (the remaining images) → used to measure accuracy, **never**
     enrolled.
3. Add a pool of **"unknown" images** — items/products you deliberately do *not*
   enroll. These exercise the OOD / `UNKNOWN` path.
4. Lay it out so a script can read the ground-truth label from the folder name:

   ```
   data/eval/
     reference/<product_name>/*.jpg     # enrollment photos (1–10 each)
     test/<product_name>/*.jpg          # query photos; ground-truth = folder name
     test/__unknown__/*.jpg             # should come out as UNKNOWN
   ```

**Rules of hygiene**
- Reference and test sets must not overlap (no image used for both).
- Keep 1–10 reference images per product (the API enforces `min_shots`/`max_shots`).
- The `__unknown__` items must be things no enrolled product covers.

---

## Phase 2 — Start the app

```bash
./app/run.sh            # serves API + frontend on http://localhost:8000
```

Then verify the backbone weights are real, not random:

```bash
curl -s http://localhost:8000/api/health
```

- `weights_status` **must not** be `"random"`. If it is, ImageNet ResNet18 weights
  haven't been cached yet — the pipeline runs but identification quality is
  degraded, so any accuracy number is meaningless. Get the weights cached first.
- Note `device` and `embedding_dim` for the write-up.

---

## Phase 3 — Enroll (few-shot) the known products

For each `reference/<product_name>/` folder, POST its images to the enrollment
endpoint:

- **Endpoint:** `POST /api/products`
- **Form fields:** `name=<product_name>`, `images=<one or more files>`

After enrolling every product, confirm the registry:

```bash
curl -s http://localhost:8000/api/products
```

Each known product should appear exactly once. Enrollment is just averaging the
reference embeddings into a prototype — no training happens.

---

## Phase 4 — Run the test set through detection

Loop over **every** image under `test/`, POST it to detection, and record the
result:

- **Endpoint:** `POST /api/detect`
- **Form field:** `image=<file>`

For each image, log a row containing:

| Field | Source |
|-------|--------|
| `ground_truth`        | the test subfolder name (`__unknown__` for OOD items) |
| `decision`            | `MATCH` / `REVIEW` / `UNKNOWN` |
| `predicted_name`      | from the response |
| `confidence`          | evidential top-1 belief |
| `uncertainty`         | vacuity `u` |
| `softmax_confidence`  | overconfident baseline (for the contrast) |

Save all rows to a CSV or JSON — this is the raw material for Phase 5 and for any
plots.

---

## Phase 5 — Compute the metrics

From the recorded rows:

- **Classification accuracy** — over the *known-product* test images, the fraction
  where `predicted_name == ground_truth`. Decide how to treat `REVIEW` (usually
  counted as "not auto-accepted"; report coverage separately).
- **OOD detection** —
  - *Recall (unknown flagged):* fraction of `__unknown__` images with
    `decision == UNKNOWN`.
  - *False positives:* fraction of *known* items wrongly flagged `UNKNOWN`.
- **The "honest AI" contrast** — on the `__unknown__` items, compare evidential
  `confidence` against `softmax_confidence`. Softmax stays overconfident; the
  evidential belief collapses (high vacuity). This side-by-side is the key plot.

Suggested summary table:

| Metric | Value |
|--------|-------|
| Classification accuracy (known)      | … |
| Auto-accept coverage (MATCH rate)    | … |
| Unknown recall (OOD flagged)         | … |
| Unknown false-positive rate          | … |
| Mean softmax conf. on unknowns       | … |
| Mean evidential conf. on unknowns    | … |

---

## Phase 6 — Threshold sweep (optional but recommended)

Re-run Phases 4–5 while varying the decision knobs to draw operating curves:

- `SENTINEL_UNCERTAINTY_THRESHOLD` — vacuity above which an item is `UNKNOWN`.
- `SENTINEL_REVIEW_MARGIN` — confidence band that triggers `REVIEW`.
- `SENTINEL_COSINE_SCALE` — sharpness of the evidence mapping.

These are read from `app/backend/core/config.py` and are env-var overridable, so a
sweep is just restarting the app with different values (or, better, having the
evaluation script vary them). Plot accuracy-vs-coverage and OOD precision/recall.

---

## What still needs to be built

Phases 3–5 are manual through the UI today, which is impractical for a full test
set. The missing piece is a batch evaluation script — e.g. `app/scripts/evaluate.py`
— that:

1. walks the `data/eval/` layout,
2. enrolls every product (Phase 3),
3. runs every test image through `/api/detect` (Phase 4),
4. writes the per-image CSV and the summary metrics table (Phase 5),
5. optionally loops over threshold values (Phase 6).

That script is the next deliverable once the dataset is in place.

---

## Quick checklist

- [ ] Dataset laid out as `data/eval/{reference,test}/…`, reference ∩ test = ∅
- [ ] `__unknown__` items collected for the OOD test
- [ ] App running; `/api/health` shows real (non-`random`) weights
- [ ] All products enrolled; `/api/products` lists each once
- [ ] Every test image scored and logged to CSV/JSON
- [ ] Classification accuracy + OOD metrics computed
- [ ] Evidential-vs-softmax contrast plotted
- [ ] (Optional) threshold sweep for operating curves
