"""Process-wide singletons for expensive clients (Chroma HTTP)."""

from typing import Optional

from chromadb import ClientAPI

from vectordb.chroma_client import get_chroma_client

_chroma_client: Optional[ClientAPI] = None


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
