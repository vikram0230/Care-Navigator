"""FastAPI application entrypoint with CORS, lifespan, and Prometheus metrics."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import get_settings
from api.deps import reset_chroma_singleton, reset_redis_singleton
from api.routes import health, ingest, rag

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str) -> None:
    """Configure root logging for the API process."""
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)
    logger.info(
        "Care Navigator API starting (environment=%s)",
        settings.ENVIRONMENT,
    )
    try:
        yield
    finally:
        logger.info("Care Navigator API shutting down")
        reset_chroma_singleton()
        reset_redis_singleton()
        get_settings.cache_clear()


app = FastAPI(
    title="Care Navigator API",
    description=(
        "Multi-tenant RAG-powered health benefits Q&A "
        "(Stages 1–6: health, metrics, RAG, tenancy/policy, Redis caching, async Celery ingest)."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

# PRODUCTION NOTE: In production this would restrict origins to the Streamlit UI
# and corporate SSO domains instead of wildcard CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
)
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(health.router)
app.include_router(rag.router)
app.include_router(ingest.router)


def _json_safe_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make Pydantic validation errors JSON-serializable (``ctx`` may hold exceptions)."""
    safe: List[Dict[str, Any]] = []
    for err in errors:
        item = dict(err)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {k: str(v) for k, v in ctx.items()}
        safe.append(item)
    return safe


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return structured 422 responses for request validation errors."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": _json_safe_validation_errors(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors and return 500 without leaking internals."""
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.exception("Unhandled error for %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
