"""Async RAG query Celery task tests — no live Chroma/Ollama/Redis (all mocked)."""

from unittest.mock import MagicMock, patch

import redis as redis_mod

from api.schemas.rag import RagQueryResponse
from tests.conftest import SettingsForTests
from workers.rag_tasks import rag_query_task


def _settings(**overrides: object) -> SettingsForTests:
    defaults = dict(
        COMPANY_IDS="bcbs,wells_fargo",
        REDIS_URL="redis://localhost:6379/0",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        OLLAMA_CHAT_MODEL="llama3.2",
    )
    defaults.update(overrides)
    return SettingsForTests(**defaults)


def test_rag_query_task_runs_pipeline_and_returns_serialized_response() -> None:
    settings = _settings()
    resp = RagQueryResponse(answer="Your deductible is $500.", citations=[], cache_hit=False)
    with patch("workers.rag_tasks.get_settings", return_value=settings), \
        patch("workers.rag_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.rag_tasks.redis.Redis.from_url", return_value=MagicMock()), \
        patch("workers.rag_tasks.run_rag_query", return_value=resp) as mock_run:
        out = rag_query_task(question="What is my deductible?", company_id="bcbs")

    assert out["answer"] == "Your deductible is $500."
    assert out["cache_hit"] is False
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["company_id"] == "bcbs"


def test_rag_query_task_converts_conversation_history_to_turns() -> None:
    settings = _settings()
    resp = RagQueryResponse(answer="ok", citations=[])
    with patch("workers.rag_tasks.get_settings", return_value=settings), \
        patch("workers.rag_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.rag_tasks.redis.Redis.from_url", return_value=MagicMock()), \
        patch("workers.rag_tasks.run_rag_query", return_value=resp) as mock_run:
        rag_query_task(
            question="And my copay?",
            company_id="bcbs",
            conversation_history=[{"question": "What is my deductible?", "answer": "It is $500."}],
        )

    history = mock_run.call_args.kwargs["conversation_history"]
    assert history is not None
    assert history[0].question == "What is my deductible?"
    assert history[0].answer == "It is $500."


def test_rag_query_task_passes_none_redis_when_unreachable() -> None:
    settings = _settings()
    resp = RagQueryResponse(answer="ok", citations=[])
    with patch("workers.rag_tasks.get_settings", return_value=settings), \
        patch("workers.rag_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.rag_tasks.redis.Redis.from_url", side_effect=redis_mod.exceptions.RedisError("down")), \
        patch("workers.rag_tasks.run_rag_query", return_value=resp) as mock_run:
        rag_query_task(question="q", company_id="bcbs")

    assert mock_run.call_args.kwargs["redis_client"] is None
