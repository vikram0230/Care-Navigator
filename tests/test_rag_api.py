"""Stage 3–4 RAG route tests (validation and configuration; no live Chroma/Ollama)."""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.deps import reset_chroma_singleton
from api.schemas.rag import RagQueryResponse


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


def test_rag_query_missing_api_key_returns_401(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When RAG_API_KEYS is set, requests must include a valid key."""
    monkeypatch.setenv("RAG_API_KEYS", "integration-test-secret")
    get_settings.cache_clear()
    reset_chroma_singleton()
    try:
        response = client.post(
            "/rag/query",
            json={"question": "What is my deductible?", "company_id": "bcbs"},
        )
        assert response.status_code == 401
    finally:
        monkeypatch.delenv("RAG_API_KEYS", raising=False)
        get_settings.cache_clear()
        reset_chroma_singleton()


def test_rag_query_accepts_x_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid X-API-Key passes the gate."""
    monkeypatch.setenv("RAG_API_KEYS", "good-key")
    get_settings.cache_clear()
    reset_chroma_singleton()
    try:
        stub = RagQueryResponse(answer="stub", citations=[])
        with patch("api.routes.rag.asyncio.to_thread", new_callable=AsyncMock, return_value=stub):
            response = client.post(
                "/rag/query",
                json={"question": "What is my deductible?", "company_id": "bcbs"},
                headers={"X-API-Key": "good-key"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body.get("answer") == "stub"
        assert body.get("cache_hit") is False
    finally:
        monkeypatch.delenv("RAG_API_KEYS", raising=False)
        get_settings.cache_clear()
        reset_chroma_singleton()


def test_rag_query_passes_redis_client_to_service(
    client: TestClient,
) -> None:
    """The route fetches the Redis singleton and forwards it into ``run_rag_query``."""
    stub = RagQueryResponse(answer="stub", citations=[])
    fake_redis = object()
    with patch("api.routes.rag.get_redis_singleton", return_value=fake_redis), \
        patch("api.routes.rag.asyncio.to_thread", new_callable=AsyncMock, return_value=stub) as mock_to_thread:
        response = client.post(
            "/rag/query",
            json={"question": "What is my deductible?", "company_id": "bcbs"},
        )
    assert response.status_code == 200
    _, kwargs = mock_to_thread.call_args
    assert kwargs.get("redis_client") is fake_redis


def test_rag_query_accepts_conversation_history(
    client: TestClient,
) -> None:
    """A valid conversation_history is accepted and forwarded to the service."""
    stub = RagQueryResponse(answer="stub", citations=[])
    with patch("api.routes.rag.asyncio.to_thread", new_callable=AsyncMock, return_value=stub) as mock_to_thread:
        response = client.post(
            "/rag/query",
            json={
                "question": "And my copay?",
                "company_id": "bcbs",
                "conversation_history": [
                    {"question": "What is my deductible?", "answer": "Your deductible is $500."},
                ],
            },
        )
    assert response.status_code == 200
    _, kwargs = mock_to_thread.call_args
    history = kwargs.get("conversation_history")
    assert history is not None
    assert history[0].question == "What is my deductible?"
    assert history[0].answer == "Your deductible is $500."


def test_rag_query_rejects_malformed_conversation_turn(client: TestClient) -> None:
    """A history turn missing 'answer' fails validation with 422."""
    response = client.post(
        "/rag/query",
        json={
            "question": "And my copay?",
            "company_id": "bcbs",
            "conversation_history": [{"question": "What is my deductible?"}],
        },
    )
    assert response.status_code == 422


def test_rag_query_rejects_null_byte_question(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guardrails return 400 for disallowed characters."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.2")
    get_settings.cache_clear()
    reset_chroma_singleton()
    try:
        response = client.post(
            "/rag/query",
            json={"question": "bad\x00question", "company_id": "bcbs"},
        )
        assert response.status_code == 400
        assert "Question failed validation" in response.json().get("detail", "")
    finally:
        get_settings.cache_clear()
        reset_chroma_singleton()


def test_openapi_lists_rag_query(client: TestClient) -> None:
    """OpenAPI schema includes POST /rag/query."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert "/rag/query" in paths
    assert "post" in paths["/rag/query"]
