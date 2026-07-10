"""Stage 6 Celery tasks: async PDF ingest and bundled seed-data reseed, with optional webhooks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import redis

from api.cache import invalidate_company_answer_cache
from api.config import get_settings
from api.llm_client import get_embedding_model
from vectordb.chroma_client import get_chroma_client
from vectordb.collections import create_collections
from vectordb.ingestion import PDFIngestionPipeline
from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _notify_webhook(url: Optional[str], payload: Dict[str, Any], timeout: float) -> None:
    """Best-effort webhook POST. Never raises — a broken webhook must not fail the task."""
    if not url:
        return
    try:
        httpx.post(url, json=payload, timeout=timeout)
    except Exception:
        logger.warning("Webhook POST to %s failed", url, exc_info=True)


def _invalidate_caches_for_ingest(settings: Any, company_id: str, tier: int) -> None:
    """Flush cached answers affected by this ingest: one company for tier 2, every company for tier 1."""
    try:
        redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        if tier == 2:
            invalidate_company_answer_cache(redis_client, company_id)
        else:
            for cid in settings.company_id_list:
                invalidate_company_answer_cache(redis_client, cid)
    except redis.exceptions.RedisError:
        logger.warning("Could not reach Redis to invalidate answer cache after ingest", exc_info=True)


@celery_app.task(name="workers.ingest_tasks.ingest_pdf_task", bind=True)
def ingest_pdf_task(
    self: Any,
    *,
    pdf_path: str,
    company_id: str,
    tier: int,
    doc_type: str,
    plan_year: str,
    source_override: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Chunk, embed, and store one PDF; invalidate affected answer caches; notify a webhook."""
    settings = get_settings()
    try:
        client = get_chroma_client(settings)
        embeddings = get_embedding_model(settings)
        create_collections(client, settings)
        pipeline = PDFIngestionPipeline(
            client,
            embeddings,
            embedding_sub_batch_size=settings.EMBEDDING_SUB_BATCH_SIZE,
            embedding_inter_batch_delay_seconds=settings.EMBEDDING_INTER_BATCH_DELAY_SECONDS,
        )
        chunks = pipeline.ingest_pdf(
            Path(pdf_path),
            company_id=company_id,
            tier=tier,
            doc_type=doc_type,
            plan_year=plan_year,
            source_override=source_override,
        )
    except Exception as exc:
        _notify_webhook(
            webhook_url,
            {
                "event": "ingest.completed",
                "task_id": self.request.id,
                "status": "failed",
                "company_id": company_id,
                "tier": tier,
                "error": str(exc),
            },
            settings.INGEST_WEBHOOK_TIMEOUT_SECONDS,
        )
        raise

    _invalidate_caches_for_ingest(settings, company_id, tier)

    result: Dict[str, Any] = {
        "status": "success",
        "chunks": chunks,
        "company_id": company_id,
        "tier": tier,
        "doc_type": doc_type,
        "source": source_override,
    }
    _notify_webhook(
        webhook_url,
        {"event": "ingest.completed", "task_id": self.request.id, **result},
        settings.INGEST_WEBHOOK_TIMEOUT_SECONDS,
    )
    return result


@celery_app.task(name="workers.ingest_tasks.reseed_task", bind=True)
def reseed_task(
    self: Any,
    *,
    reset: bool = False,
    full: bool = False,
    webhook_url: Optional[str] = None,
) -> Dict[str, int]:
    """Re-run the bundled seed-data ingest (scripts.seed_documents.run_ingestion) asynchronously."""
    settings = get_settings()
    from scripts.seed_documents import run_ingestion  # local import: script has a CLI __main__ guard

    try:
        counts = run_ingestion(reset_chroma=reset, full=full)
    except Exception as exc:
        _notify_webhook(
            webhook_url,
            {
                "event": "reseed.completed",
                "task_id": self.request.id,
                "status": "failed",
                "error": str(exc),
            },
            settings.INGEST_WEBHOOK_TIMEOUT_SECONDS,
        )
        raise

    _notify_webhook(
        webhook_url,
        {
            "event": "reseed.completed",
            "task_id": self.request.id,
            "status": "success",
            "counts": counts,
        },
        settings.INGEST_WEBHOOK_TIMEOUT_SECONDS,
    )
    return counts
