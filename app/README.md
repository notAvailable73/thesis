# Sentinel — Industrial Product Detector

A web application demonstrating **Bayesian Parameter-Efficient Fine-Tuning (B-PEFT)**
for a real-world industrial inspection scenario.

> Teach the system a handful of reference photos per product (few-shot enrollment),
> then let it inspect items on the line. It identifies the product, reports an
> **honest confidence**, and — crucially — **flags unknown / anomalous items** it has
> never seen instead of confidently mislabelling them. Unknown items are routed to
> manual inspection.

This is the "honest AI" story of the thesis, made tangible:
**few-shot learning + evidential uncertainty + out-of-distribution (OOD) detection.**

---

## Why this design

| Thesis concept | In this app |
|---|---|
| Frozen backbone (ResNet18) | `backend/ml/backbone.py` — feature extractor, weights frozen |
| Parameter-free prototype head | `backend/services/prototype_store.py` — a product = mean of its reference embeddings |
| Few-shot enrollment | "Register product" with 1–10 reference photos |
| Evidential / Dirichlet uncertainty (Sensoy 2018) | `backend/ml/evidential.py` — evidence → belief + vacuity |
| OOD detection | high vacuity ⇒ `UNKNOWN` decision ⇒ manual inspection |
| Softmax overconfidence (the baseline) | shown side-by-side so the contrast is visible |

No runtime training happens: enrollment is just averaging embeddings, and detection
is vector math + one frozen forward pass (~16 ms/image on CPU). Runs on low-spec hardware.

---

## Architecture

```
Browser (SPA dashboard)
        │  fetch()  JSON + multipart
        ▼
FastAPI  (backend/main.py)
        │
   ┌────┴─────────────── api/routes.py  (HTTP layer only)
   │
   ├── services/model_service.py   embed(image) -> 512-d vector   (holds frozen backbone)
   ├── services/prototype_store.py registry of enrolled products   (persists to data/)
   └── services/detector.py        query vector -> decision + evidential + softmax
        │
        └── ml/  backbone · transforms · evidential   (pure, web-agnostic)
```

Layers only depend downward (frontend → api → services → ml). Each layer is
swappable in isolation — e.g. change the backbone, or the OOD rule, without
touching the API or UI.

---

## Run

```bash
# from the repo root, with the project venv active
source .venv/bin/activate
python -m app.backend.main            # serves API + frontend on http://localhost:8000
```

Then open <http://localhost:8000>.

The first launch loads ImageNet-pretrained ResNet18 weights (cached by torchvision).
If they aren't downloaded yet, the app still starts with random-initialised weights
(pipeline works; identification quality is degraded until real weights are cached) and
`/api/health` reports `weights: "random"`.

## Configuration

See `backend/core/config.py` (env-var overridable). Key knobs:

- `SENTINEL_UNCERTAINTY_THRESHOLD` — vacuity above which an item is called `UNKNOWN`.
- `SENTINEL_REVIEW_MARGIN` — confidence band that triggers `REVIEW` instead of `MATCH`.
- `SENTINEL_COSINE_SCALE` — sharpness of the evidence mapping.
```
