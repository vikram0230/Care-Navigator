"""Stage 5 caching and Stage 7 conversation-memory behavior of ``run_rag_query``.

Mocked Chroma/Ollama/Redis throughout — no live services required.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import redis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.cache import answer_cache_key, embedding_cache_key
from api.schemas.rag import ConversationTurn
from api.services.rag import run_rag_query
from tests.conftest import SettingsForTests


def _settings(**overrides: Any) -> SettingsForTests:
    defaults: Dict[str, Any] = dict(
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        OLLAMA_CHAT_MODEL="llama3.2",
        COMPANY_IDS="bcbs,wells_fargo",
        RAG_ANSWER_CACHE_ENABLED=True,
        RAG_ANSWER_CACHE_TTL_SECONDS=3600,
        RAG_EMBEDDING_CACHE_ENABLED=True,
        RAG_EMBEDDING_CACHE_TTL_SECONDS=86400,
        RAG_MAX_CONVERSATION_TURNS=6,
    )
    defaults.update(overrides)
    return SettingsForTests(**defaults)


def _mock_collection(hits: Optional[List[Dict[str, Any]]] = None) -> MagicMock:
    hits = hits or []
    coll = MagicMock()
    coll.query.return_value = {
        "documents": [[h["document"] for h in hits]],
        "metadatas": [[h.get("metadata", {}) for h in hits]],
        "distances": [[h.get("distance", 0.5) for h in hits]],
    }
    return coll


def _patched_run(**settings_overrides: Any):
    """Context manager stack: patches get_collection/get_embedding_model/get_chat_model."""
    settings = _settings(**settings_overrides)
    tier1_coll = _mock_collection(
        [{"document": "Tier 1 passage.", "metadata": {"source": "global.pdf"}, "distance": 0.2}],
    )
    tier2_coll = _mock_collection(
        [{"document": "Tier 2 passage.", "metadata": {"source": "bcbs.pdf"}, "distance": 0.1}],
    )

    def _get_collection_side_effect(_client: Any, company_id: str, tier: int, settings: Any = None) -> MagicMock:
        return tier1_coll if tier == 1 else tier2_coll

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    chat = MagicMock()
    chat.invoke.return_value = SimpleNamespace(content="The answer is 42.")

    return settings, tier1_coll, tier2_coll, embeddings, chat, _get_collection_side_effect


def test_run_rag_query_cache_hit_skips_embedding_and_chroma_and_llm() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    key = answer_cache_key(
        "bcbs",
        "What is my deductible?",
        None,
        None,
        settings.OLLAMA_EMBEDDING_MODEL,
        settings.OLLAMA_CHAT_MODEL,
    )
    cached_json = (
        '{"answer": "Cached deductible answer.", "citations": [], "cache_hit": false}'
    )
    redis_client = MagicMock()
    redis_client.get.return_value = cached_json

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            redis_client=redis_client,
        )

    assert result.cache_hit is True
    assert result.answer == "Cached deductible answer."
    embeddings.embed_query.assert_not_called()
    chat.invoke.assert_not_called()
    tier1_coll.query.assert_not_called()
    tier2_coll.query.assert_not_called()
    redis_client.get.assert_called_once_with(key)


def test_run_rag_query_cache_miss_runs_full_flow_and_stores_answer() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    redis_client = MagicMock()
    redis_client.get.return_value = None

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            redis_client=redis_client,
        )

    assert result.cache_hit is False
    assert result.answer == "The answer is 42."
    embeddings.embed_query.assert_called_once()
    chat.invoke.assert_called_once()
    tier1_coll.query.assert_called_once()
    tier2_coll.query.assert_called_once()

    answer_key = answer_cache_key(
        "bcbs",
        "What is my deductible?",
        None,
        None,
        settings.OLLAMA_EMBEDDING_MODEL,
        settings.OLLAMA_CHAT_MODEL,
    )
    redis_client.setex.assert_any_call(
        answer_key,
        settings.RAG_ANSWER_CACHE_TTL_SECONDS,
        result.model_dump_json(),
    )


def test_run_rag_query_embedding_cache_hit_skips_embed_call() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    embed_key = embedding_cache_key(settings.OLLAMA_EMBEDDING_MODEL, "What is my deductible?")

    redis_client = MagicMock()

    def _get_side_effect(key: str) -> Optional[str]:
        if key == embed_key:
            return "[0.9, 0.8, 0.7]"
        return None

    redis_client.get.side_effect = _get_side_effect

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            filter_doc_types=["benefits"],
            redis_client=redis_client,
        )

    embeddings.embed_query.assert_not_called()
    chat.invoke.assert_called_once()
    tier1_call_kwargs = tier1_coll.query.call_args.kwargs
    assert tier1_call_kwargs["query_embeddings"] == [[0.9, 0.8, 0.7]]
    assert result.cache_hit is False


def test_run_rag_query_answer_cache_disabled_bypasses_cache() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run(
        RAG_ANSWER_CACHE_ENABLED=False,
    )
    redis_client = MagicMock()
    redis_client.get.return_value = (
        '{"answer": "Stale cached answer.", "citations": [], "cache_hit": false}'
    )

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            redis_client=redis_client,
        )

    assert result.answer == "The answer is 42."
    assert result.cache_hit is False
    chat.invoke.assert_called_once()
    answer_key = answer_cache_key(
        "bcbs",
        "What is my deductible?",
        None,
        None,
        settings.OLLAMA_EMBEDDING_MODEL,
        settings.OLLAMA_CHAT_MODEL,
    )
    for call in redis_client.setex.call_args_list:
        assert call.args[0] != answer_key


def test_run_rag_query_redis_client_none_runs_normally_without_touching_redis() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            redis_client=None,
        )

    assert result.answer == "The answer is 42."
    assert result.cache_hit is False


def test_run_rag_query_threads_conversation_history_into_messages() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    history = [
        ConversationTurn(question="What is my deductible?", answer="Your deductible is $500."),
        ConversationTurn(question="And my copay?", answer="Your copay is $20."),
    ]

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What about out-of-network?",
            company_id="bcbs",
            conversation_history=history,
            redis_client=None,
        )

    messages = chat.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "What is my deductible?"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "Your deductible is $500."
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "And my copay?"
    assert isinstance(messages[4], AIMessage)
    assert messages[4].content == "Your copay is $20."
    assert isinstance(messages[5], HumanMessage)
    assert "What about out-of-network?" in messages[5].content
    assert len(messages) == 6


def test_run_rag_query_trims_history_to_max_turns() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run(
        RAG_MAX_CONVERSATION_TURNS=2,
    )
    history = [
        ConversationTurn(question=f"Question {i}?", answer=f"Answer {i}.")
        for i in range(1, 6)  # 5 turns, cap is 2 -> keep the most recent 2
    ]

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="Final question?",
            company_id="bcbs",
            conversation_history=history,
            redis_client=None,
        )

    messages = chat.invoke.call_args.args[0]
    # SystemMessage + 2 kept turns (Human/AI each) + current HumanMessage = 6
    assert len(messages) == 6
    assert messages[1].content == "Question 4?"
    assert messages[2].content == "Answer 4."
    assert messages[3].content == "Question 5?"
    assert messages[4].content == "Answer 5."


def test_run_rag_query_zero_max_conversation_turns_drops_all_history() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run(
        RAG_MAX_CONVERSATION_TURNS=0,
    )
    history = [ConversationTurn(question="Q?", answer="A.")]

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="Final question?",
            company_id="bcbs",
            conversation_history=history,
            redis_client=None,
        )

    messages = chat.invoke.call_args.args[0]
    assert len(messages) == 2  # SystemMessage + current HumanMessage only


def test_run_rag_query_bypasses_answer_cache_when_history_present() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    history = [ConversationTurn(question="What is my deductible?", answer="Your deductible is $500.")]
    redis_client = MagicMock()
    redis_client.get.return_value = (
        '{"answer": "Should not be used.", "citations": [], "cache_hit": false}'
    )

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What about out-of-network?",
            company_id="bcbs",
            conversation_history=history,
            redis_client=redis_client,
        )

    assert result.answer == "The answer is 42."
    assert result.cache_hit is False
    chat.invoke.assert_called_once()

    answer_key = answer_cache_key(
        "bcbs",
        "What about out-of-network?",
        None,
        None,
        settings.OLLAMA_EMBEDDING_MODEL,
        settings.OLLAMA_CHAT_MODEL,
    )
    # Never read from or written to the answer cache when history is present.
    assert all(call.args[0] != answer_key for call in redis_client.get.call_args_list)
    for call in redis_client.setex.call_args_list:
        assert call.args[0] != answer_key


def test_run_rag_query_history_still_uses_embedding_cache() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    history = [ConversationTurn(question="What is my deductible?", answer="Your deductible is $500.")]
    embed_key = embedding_cache_key(settings.OLLAMA_EMBEDDING_MODEL, "What about out-of-network?")

    redis_client = MagicMock()

    def _get_side_effect(key: str) -> Optional[str]:
        if key == embed_key:
            return "[0.5, 0.5, 0.5]"
        return None

    redis_client.get.side_effect = _get_side_effect

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What about out-of-network?",
            company_id="bcbs",
            conversation_history=history,
            redis_client=redis_client,
        )

    embeddings.embed_query.assert_not_called()


def test_run_rag_query_redis_failure_falls_back_gracefully() -> None:
    settings, tier1_coll, tier2_coll, embeddings, chat, get_collection_fn = _patched_run()
    redis_client = MagicMock()
    redis_client.get.side_effect = redis.exceptions.ConnectionError("redis down")
    redis_client.setex.side_effect = redis.exceptions.ConnectionError("redis down")

    with patch("api.services.rag.get_collection", side_effect=get_collection_fn), \
        patch("api.services.rag.get_embedding_model", return_value=embeddings), \
        patch("api.services.rag.get_chat_model", return_value=chat):
        result = run_rag_query(
            chroma_client=MagicMock(),
            settings=settings,
            question="What is my deductible?",
            company_id="bcbs",
            redis_client=redis_client,
        )

    assert result.answer == "The answer is 42."
    assert result.cache_hit is False
