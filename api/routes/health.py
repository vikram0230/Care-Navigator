"""Health check endpoint for load balancers and Kubernetes probes."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from api.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness and readiness probe",
    status_code=status.HTTP_200_OK,
)
async def health() -> Dict[str, Any]:
    """Return service health metadata.

    Returns:
        JSON object with status and environment labels.

    Raises:
        HTTPException: If configuration cannot be loaded (unexpected).
    """
    try:
        settings = get_settings()
    except ValueError as exc:
        logger.exception("Invalid application configuration during health check")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface any unexpected init failure
        logger.exception("Unexpected error during health check")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from exc

    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "service": "care-navigator-api",
    }
