"""Stage 5 Redis caching: exact-match answer cache + embedding cache.

Every function here is fail-open: a down or erroring Redis must never turn a
working RAG request into a failure, so all Redis calls are wrapped and logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import List, Optional

import redis

from api.schemas.rag import RagQueryResponse

logger = logging.getLogger(__name__)

_ANSWER_PREFIX = "rag:answer:v1"
_EMBED_PREFIX = "rag:embed:v1"
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_question(question: str) -> str:
    """Collapse whitespace and casefold so trivial phrasing differences still hit."""
    return _WHITESPACE_RE.sub(" ", question.strip()).casefold()


def _hash(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8"))
    return digest.hexdigest()


def answer_cache_key(
    company_id: str,
    question: str,
    filter_doc_types: Optional[List[str]],
    filter_plan_years: Optional[List[str]],
    embedding_model: str,
    chat_model: str,
) -> str:
    """Build a deterministic cache key for one (tenant, question, filters, models) tuple."""
    doc_types = ",".join(sorted(filter_doc_types)) if filter_doc_types else ""
    plan_years = ",".join(sorted(filter_plan_years)) if filter_plan_years else ""
    digest = _hash(
        _normalize_question(question),
        doc_types,
        plan_years,
        embedding_model,
        chat_model,
    )
    return f"{_ANSWER_PREFIX}:{company_id}:{digest}"


def embedding_cache_key(model: str, text: str) -> str:
    """Build a deterministic cache key for a question's embedding vector."""
    digest = _hash(model, text)
    return f"{_EMBED_PREFIX}:{digest}"


def get_cached_answer(redis_client: "redis.Redis", key: str) -> Optional[RagQueryResponse]:
    """Return the cached answer for ``key``, or ``None`` on a miss or any Redis/parse error."""
    try:
        raw = redis_client.get(key)
    except redis.exceptions.RedisError:
        logger.warning("Redis GET failed for answer cache key %s", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return RagQueryResponse.model_validate_json(raw)
    except Exception:
        logger.warning("Malformed cached answer JSON for key %s", key, exc_info=True)
        return None


def set_cached_answer(
    redis_client: "redis.Redis",
    key: str,
    response: RagQueryResponse,
    ttl_seconds: int,
) -> None:
    """Store ``response`` under ``key`` with a TTL; swallows Redis errors."""
    try:
        redis_client.setex(key, ttl_seconds, response.model_dump_json())
    except redis.exceptions.RedisError:
        logger.warning("Redis SETEX failed for answer cache key %s", key, exc_info=True)


def get_cached_embedding(redis_client: "redis.Redis", key: str) -> Optional[List[float]]:
    """Return the cached embedding vector for ``key``, or ``None`` on a miss or error."""
    try:
        raw = redis_client.get(key)
    except redis.exceptions.RedisError:
        logger.warning("Redis GET failed for embedding cache key %s", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        vector = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Malformed cached embedding JSON for key %s", key, exc_info=True)
        return None
    if not isinstance(vector, list):
        return None
    return [float(v) for v in vector]


def set_cached_embedding(
    redis_client: "redis.Redis",
    key: str,
    vector: List[float],
    ttl_seconds: int,
) -> None:
    """Store an embedding ``vector`` under ``key`` with a TTL; swallows Redis errors."""
    try:
        redis_client.setex(key, ttl_seconds, json.dumps(vector))
    except redis.exceptions.RedisError:
        logger.warning("Redis SETEX failed for embedding cache key %s", key, exc_info=True)


def invalidate_company_answer_cache(redis_client: "redis.Redis", company_id: str) -> int:
    """Delete every cached answer for ``company_id``. Returns count removed (0 on Redis error)."""
    pattern = f"{_ANSWER_PREFIX}:{company_id}:*"
    try:
        keys = list(redis_client.scan_iter(match=pattern))
        if not keys:
            return 0
        redis_client.delete(*keys)
        return len(keys)
    except redis.exceptions.RedisError:
        logger.warning("Redis cache invalidation failed for company_id=%s", company_id, exc_info=True)
        return 0
