"""Celery application configuration for async workers (Stage 1 minimal)."""

import logging
import os
from typing import Any

from celery import Celery
from celery.exceptions import CeleryError

logger = logging.getLogger(__name__)


def _build_celery() -> Celery:
    """Construct Celery app with broker and result backend from settings."""
    try:
        from api.config import get_settings

        settings = get_settings()
        broker = settings.CELERY_BROKER_URL
        backend = settings.CELERY_RESULT_BACKEND
    except Exception as exc:  # noqa: BLE001 — fall back to env for worker bootstrap
        logger.warning("Using environment for Celery broker (settings unavailable: %s)", exc)
        broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
        backend = os.environ.get(
            "CELERY_RESULT_BACKEND",
            "redis://localhost:6379/1",
        )

    app = Celery(
        "care_navigator",
        broker=broker,
        backend=backend,
        include=[],
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )
    return app


celery_app = _build_celery()


@celery_app.task(name="workers.celery_app.health_ping", bind=True)
def health_ping(self: Any) -> str:
    """Lightweight task used for worker connectivity checks.

    Returns:
        A static success token.

    Raises:
        CeleryError: Re-raised if task context fails unexpectedly.
    """
    try:
        return "pong"
    except CeleryError:
        raise
    except Exception as exc:
        logger.exception("health_ping task failed")
        raise CeleryError(str(exc)) from exc
