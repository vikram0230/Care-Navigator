"""Process-wide singletons for expensive clients (Chroma HTTP, Redis)."""

from typing import Optional

import redis
from chromadb import ClientAPI

from api.config import Settings, get_settings
from vectordb.chroma_client import get_chroma_client

_chroma_client: Optional[ClientAPI] = None
_redis_client: Optional[redis.Redis] = None


def get_chroma_singleton() -> ClientAPI:
    """Return a cached Chroma HTTP client (one per worker process)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = get_chroma_client()
    return _chroma_client


def reset_chroma_singleton() -> None:
    """Drop the cached client (for tests)."""
    global _chroma_client
    _chroma_client = None


def get_redis_singleton(settings: Optional[Settings] = None) -> "redis.Redis":
    """Return a cached Redis client (one per worker process).

    Construction does not connect eagerly; a down Redis surfaces (and is
    swallowed) on first command inside ``api.cache``.
    """
    global _redis_client
    if _redis_client is None:
        resolved = settings or get_settings()
        _redis_client = redis.Redis.from_url(
            resolved.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def reset_redis_singleton() -> None:
    """Drop the cached Redis client (for tests)."""
    global _redis_client
    _redis_client = None
