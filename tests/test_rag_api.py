"""Stage 3 RAG route tests (validation and configuration; no live Chroma/Ollama)."""

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.deps import reset_chroma_singleton


def test_rag_query_unknown_company_returns_400(client: TestClient) -> None:
    """``company_id`` must be listed in COMPANY_IDS."""
    response = client.post(
        "/rag/query",
        json={"question": "What is my deductible?", "company_id": "unknown_corp"},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Unknown company_id" in detail


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "company_id": "bcbs"},
        {"question": "   ", "company_id": "bcbs"},
    ],
)
def test_rag_query_empty_question_returns_422(
    client: TestClient,
    payload: Dict[str, Any],
) -> None:
    """Question must be non-empty after validation."""
    response = client.post("/rag/query", json=payload)
    assert response.status_code == 422


def test_rag_query_missing_ollama_base_url_returns_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLLAMA_BASE_URL must be set for RAG."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.2")
    get_settings.cache_clear()
    reset_chroma_singleton()
    try:
        response = client.post(
            "/rag/query",
            json={"question": "Test question?", "company_id": "bcbs"},
        )
        assert response.status_code == 503
        assert "OLLAMA_BASE_URL" in response.json().get("detail", "")
    finally:
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        get_settings.cache_clear()
        reset_chroma_singleton()


def test_openapi_lists_rag_query(client: TestClient) -> None:
    """OpenAPI schema includes POST /rag/query."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert "/rag/query" in paths
    assert "post" in paths["/rag/query"]
