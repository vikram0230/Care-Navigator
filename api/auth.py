"""Optional API key verification for protected HTTP routes (Stage 4)."""

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


def check_rag_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """Require a valid API key when ``RAG_API_KEYS`` is non-empty.

    Accepts ``Authorization: Bearer <token>`` or ``X-API-Key`` header.
    When ``RAG_API_KEYS`` is empty, this is a no-op (local development default).
    """
    settings = get_settings()
    allowed = settings.rag_api_key_set
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
