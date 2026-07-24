"""FastAPI application entry point.

Run with:  python -m app.backend.main

Assembles the app: builds the service container at startup, mounts the JSON API
under /api, and serves the static SPA frontend at /.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.backend.api.routes import router as api_router
from app.backend.container import Container
from app.backend.core.config import settings
from app.backend.core.logging import get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy init (loading the backbone) happens here, once, at startup.
    log.info("Starting Sentinel v%s …", __version__)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app.state.container = Container(settings)
    log.info("Startup complete. Frontend: http://%s:%d",
             settings.host, settings.port)
    yield
    log.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel — Industrial Product Detector",
        description="B-PEFT few-shot detection with evidential (honest) uncertainty.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(api_router)

    # No-cache on every response: without an explicit Cache-Control, browsers
    # apply heuristic caching to static JS/CSS (roughly ~10% of file age) and
    # can serve a stale module across a plain reload. `no-cache` still lets
    # the browser use ETag/Last-Modified for a cheap 304 -- it just forces a
    # revalidation round-trip every time, which matters far more during active
    # frontend development than the negligible latency cost.
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    # Serve the SPA. index.html at "/", assets under their paths. Mounted last
    # so it doesn't shadow /api routes.
    frontend = settings.frontend_dir
    if frontend.exists():
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(frontend / "index.html")

        app.mount("/", StaticFiles(directory=str(frontend)), name="frontend")
    else:
        log.warning("Frontend directory not found at %s", frontend)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="warning",  # our own logger handles app logs
    )


if __name__ == "__main__":
    main()
