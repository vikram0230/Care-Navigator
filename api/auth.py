"""Optional API key verification for protected HTTP routes (Stages 4 and 6)."""

from typing import Optional

from fastapi import HTTPException, status

from api.config import get_settings


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.strip():
        return None
    parts = authorization.strip().split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        tok = parts[1].strip()
        return tok or None
    return None


def _check_api_key(
    authorization: Optional[str],
    x_api_key: Optional[str],
    allowed: set[str],
) -> None:
    """Require a token in ``allowed`` via ``Authorization: Bearer`` or ``X-API-Key``.

    A no-op when ``allowed`` is empty (local development default).
    """
    if not allowed:
        return
    token = _bearer_token(authorization) or (x_api_key.strip() if x_api_key else None)
    if not token or token not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing or invalid API key. "
                "Use Authorization: Bearer <key> or X-API-Key header."
            ),
        )


def check_rag_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """Require a valid API key when ``RAG_API_KEYS`` is non-empty (Stage 4)."""
    _check_api_key(authorization, x_api_key, get_settings().rag_api_key_set)


def check_ingest_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """Require a valid API key when ``INGEST_API_KEYS`` is non-empty (Stage 6).

    Independent of ``RAG_API_KEYS`` — a valid RAG key does not authorize ingest routes.
    """
    _check_api_key(authorization, x_api_key, get_settings().ingest_api_key_set)
