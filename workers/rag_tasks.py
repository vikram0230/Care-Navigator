"""Async RAG query task: runs the full retrieval + generation pipeline off the request thread.

This wraps the existing synchronous ``api.services.rag.run_rag_query`` unchanged; the only new
concern is (de)serialization at the Celery boundary — ``conversation_history`` arrives as a list of
plain dicts and the response goes back as a dict (``RagQueryResponse.model_dump()``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import redis

from api.config import get_settings
from api.schemas.rag import ConversationTurn
from api.services.rag import run_rag_query
from vectordb.chroma_client import get_chroma_client
from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _worker_redis(settings: Any) -> Optional["redis.Redis"]:
    """Best-effort Redis client so the worker can read/write the answer cache; ``None`` if unreachable.

    Keeping the worker cache-aware means a cache entry written here is later served by the API's
    synchronous short-circuit — so a repeated question never re-queues.
    """
    try:
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        return client
    except redis.exceptions.RedisError:
        logger.warning("Worker could not reach Redis; RAG answer cache disabled for this task", exc_info=True)
        return None


@celery_app.task(name="workers.rag_tasks.rag_query_task", bind=True)
def rag_query_task(
    self: Any,
    *,
    question: str,
    company_id: str,
    filter_doc_types: Optional[List[str]] = None,
    filter_plan_years: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run the RAG pipeline and return a serialized ``RagQueryResponse`` (answer + citations)."""
    settings = get_settings()
    chroma = get_chroma_client(settings)
    redis_client = _worker_redis(settings)
    history = (
        [ConversationTurn(**turn) for turn in conversation_history]
        if conversation_history
        else None
    )
    response = run_rag_query(
        chroma_client=chroma,
        settings=settings,
        question=question,
        company_id=company_id,
        filter_doc_types=filter_doc_types,
        filter_plan_years=filter_plan_years,
        conversation_history=history,
        redis_client=redis_client,
    )
    return response.model_dump()
