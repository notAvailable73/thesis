"""API routes.

Services are resolved from ``request.app.state.container`` (wired in main.py),
so routes stay decoupled from construction/ordering concerns and are easy to
test with a fake container.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app import __version__
from app.backend.core.config import settings
from app.backend.core.logging import get_logger
from app.backend.schemas.models import (
    ConfigOut,
    DetectionOut,
    HealthOut,
    ProductListOut,
    ProductOut,
)
from app.backend.services.detector import NoProductsRegistered

log = get_logger("api.routes")
router = APIRouter(prefix="/api")


def _container(request: Request):
    return request.app.state.container


# --- Health / config --------------------------------------------------------
@router.get("/health", response_model=HealthOut, tags=["system"])
def health(request: Request) -> HealthOut:
    c = _container(request)
    return HealthOut(
        status="ok",
        version=__version__,
        weights_status=c.model.weights_status,
        device=c.model.device,
        embedding_dim=c.model.embedding_dim,
        product_count=c.store.count(),
        adapter_status=c.model.adapter_status,
        adapter_type=c.model.adapter_type,
        adapter_placement=c.model.adapter_placement,
        checkpoint_val_accuracy=c.model.checkpoint_val_accuracy,
    )


@router.get("/config", response_model=ConfigOut, tags=["system"])
def get_config(request: Request) -> ConfigOut:
    c = _container(request)
    return ConfigOut(
        uncertainty_threshold=settings.uncertainty_threshold,
        review_confidence_floor=settings.review_confidence_floor,
        # Live values: the trained checkpoint's learned scale/bias when
        # loaded, else the static config defaults (see ModelService).
        cosine_scale=c.model.evidence_scale,
        evidence_bias=c.model.evidence_bias,
        image_size=settings.image_size,
        min_shots=settings.min_shots,
        max_shots=settings.max_shots,
    )


# --- Products ---------------------------------------------------------------
@router.get("/products", response_model=ProductListOut, tags=["products"])
def list_products(request: Request) -> ProductListOut:
    c = _container(request)
    products = c.store.list_products()
    return ProductListOut(
        products=[ProductOut.from_product(p) for p in products],
        count=len(products),
    )


@router.post("/products", response_model=ProductOut, status_code=201, tags=["products"])
async def register_product(
    request: Request,
    name: str = Form(...),
    images: list[UploadFile] = File(...),
) -> ProductOut:
    c = _container(request)

    n = len(images)
    if n < settings.min_shots or n > settings.max_shots:
        raise HTTPException(
            status_code=422,
            detail=f"Provide {settings.min_shots}-{settings.max_shots} reference "
            f"images (got {n}).",
        )

    raw_images: list[bytes] = []
    for f in images:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"Empty file: {f.filename}")
        raw_images.append(data)

    try:
        embeddings = c.model.embed_many(raw_images)
        product = c.store.register(name=name, embeddings=embeddings,
                                   first_image=raw_images[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ProductOut.from_product(product)


@router.delete("/products/{product_id}", status_code=204, tags=["products"])
def delete_product(request: Request, product_id: str) -> None:
    c = _container(request)
    if not c.store.delete(product_id):
        raise HTTPException(status_code=404, detail="Product not found.")


# --- Detection --------------------------------------------------------------
@router.post("/detect", response_model=DetectionOut, tags=["detection"])
async def detect(request: Request, image: UploadFile = File(...)) -> DetectionOut:
    c = _container(request)
    data = await image.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty image upload.")
    try:
        detection = c.detector.detect(data)
    except NoProductsRegistered as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DetectionOut.from_detection(detection)
