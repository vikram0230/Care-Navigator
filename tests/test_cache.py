"""Stage 5 Redis cache helpers: key builders, get/set, invalidation (all fail-open)."""

from unittest.mock import MagicMock

import redis

from api.cache import (
    answer_cache_key,
    embedding_cache_key,
    get_cached_answer,
    get_cached_embedding,
    invalidate_company_answer_cache,
    set_cached_answer,
    set_cached_embedding,
)
from api.schemas.rag import RagCitation, RagQueryResponse

_SAMPLE_RESPONSE = RagQueryResponse(
    answer="Your deductible is $500.",
    citations=[
        RagCitation(index=1, source="bcbs.pdf", excerpt="Deductible: $500"),
    ],
)


def test_answer_cache_key_deterministic() -> None:
    key1 = answer_cache_key("bcbs", "What is my deductible?", None, None, "nomic-embed-text", "llama3.2")
    key2 = answer_cache_key("bcbs", "What is my deductible?", None, None, "nomic-embed-text", "llama3.2")
    assert key1 == key2
    assert key1.startswith("rag:answer:v1:bcbs:")


def test_answer_cache_key_filter_order_independent() -> None:
    key1 = answer_cache_key("bcbs", "q", ["benefits", "formulary"], ["2025", "2024"], "m", "c")
    key2 = answer_cache_key("bcbs", "q", ["formulary", "benefits"], ["2024", "2025"], "m", "c")
    assert key1 == key2


def test_answer_cache_key_differs_by_company() -> None:
    key1 = answer_cache_key("bcbs", "q", None, None, "m", "c")
    key2 = answer_cache_key("wells_fargo", "q", None, None, "m", "c")
    assert key1 != key2


def test_answer_cache_key_differs_by_question() -> None:
    key1 = answer_cache_key("bcbs", "What is my deductible?", None, None, "m", "c")
    key2 = answer_cache_key("bcbs", "What is my copay?", None, None, "m", "c")
    assert key1 != key2


def test_answer_cache_key_differs_by_filters() -> None:
    key1 = answer_cache_key("bcbs", "q", ["benefits"], None, "m", "c")
    key2 = answer_cache_key("bcbs", "q", ["formulary"], None, "m", "c")
    assert key1 != key2


def test_answer_cache_key_differs_by_model_names() -> None:
    key1 = answer_cache_key("bcbs", "q", None, None, "nomic-embed-text", "llama3.2")
    key2 = answer_cache_key("bcbs", "q", None, None, "other-embed", "llama3.2")
    assert key1 != key2


def test_answer_cache_key_normalizes_whitespace_and_case() -> None:
    key1 = answer_cache_key("bcbs", "What is my deductible?", None, None, "m", "c")
    key2 = answer_cache_key("bcbs", "  what   is My deductible?  ", None, None, "m", "c")
    assert key1 == key2


def test_embedding_cache_key_deterministic() -> None:
    key1 = embedding_cache_key("nomic-embed-text", "What is my deductible?")
    key2 = embedding_cache_key("nomic-embed-text", "What is my deductible?")
    assert key1 == key2
    assert key1.startswith("rag:embed:v1:")


def test_embedding_cache_key_differs_by_model() -> None:
    key1 = embedding_cache_key("model-a", "same text")
    key2 = embedding_cache_key("model-b", "same text")
    assert key1 != key2


def test_embedding_cache_key_differs_by_text() -> None:
    key1 = embedding_cache_key("m", "text one")
    key2 = embedding_cache_key("m", "text two")
    assert key1 != key2


def test_get_cached_answer_hit_returns_parsed_response() -> None:
    client = MagicMock()
    client.get.return_value = _SAMPLE_RESPONSE.model_dump_json()
    result = get_cached_answer(client, "rag:answer:v1:bcbs:abc")
    assert result == _SAMPLE_RESPONSE


def test_get_cached_answer_miss_returns_none() -> None:
    client = MagicMock()
    client.get.return_value = None
    assert get_cached_answer(client, "rag:answer:v1:bcbs:abc") is None


def test_get_cached_answer_redis_error_returns_none() -> None:
    client = MagicMock()
    client.get.side_effect = redis.exceptions.ConnectionError("down")
    assert get_cached_answer(client, "rag:answer:v1:bcbs:abc") is None


def test_get_cached_answer_malformed_json_returns_none() -> None:
    client = MagicMock()
    client.get.return_value = "not json"
    assert get_cached_answer(client, "rag:answer:v1:bcbs:abc") is None


def test_set_cached_answer_calls_setex_with_ttl_and_json() -> None:
    client = MagicMock()
    set_cached_answer(client, "rag:answer:v1:bcbs:abc", _SAMPLE_RESPONSE, ttl_seconds=3600)
    client.setex.assert_called_once()
    args, _ = client.setex.call_args
    assert args[0] == "rag:answer:v1:bcbs:abc"
    assert args[1] == 3600
    assert args[2] == _SAMPLE_RESPONSE.model_dump_json()


def test_set_cached_answer_redis_error_does_not_raise() -> None:
    client = MagicMock()
    client.setex.side_effect = redis.exceptions.ConnectionError("down")
    set_cached_answer(client, "rag:answer:v1:bcbs:abc", _SAMPLE_RESPONSE, ttl_seconds=3600)


def test_get_cached_embedding_hit_returns_vector() -> None:
    client = MagicMock()
    client.get.return_value = "[0.1, 0.2, 0.3]"
    assert get_cached_embedding(client, "rag:embed:v1:abc") == [0.1, 0.2, 0.3]


def test_get_cached_embedding_miss_returns_none() -> None:
    client = MagicMock()
    client.get.return_value = None
    assert get_cached_embedding(client, "rag:embed:v1:abc") is None


def test_get_cached_embedding_redis_error_returns_none() -> None:
    client = MagicMock()
    client.get.side_effect = redis.exceptions.ConnectionError("down")
    assert get_cached_embedding(client, "rag:embed:v1:abc") is None


def test_set_cached_embedding_calls_setex_with_ttl_and_json() -> None:
    client = MagicMock()
    set_cached_embedding(client, "rag:embed:v1:abc", [0.1, 0.2], ttl_seconds=86400)
    client.setex.assert_called_once_with("rag:embed:v1:abc", 86400, "[0.1, 0.2]")


def test_set_cached_embedding_redis_error_does_not_raise() -> None:
    client = MagicMock()
    client.setex.side_effect = redis.exceptions.ConnectionError("down")
    set_cached_embedding(client, "rag:embed:v1:abc", [0.1], ttl_seconds=86400)


def test_invalidate_company_answer_cache_deletes_matching_keys() -> None:
    client = MagicMock()
    client.scan_iter.return_value = iter(
        ["rag:answer:v1:bcbs:a", "rag:answer:v1:bcbs:b"],
    )
    count = invalidate_company_answer_cache(client, "bcbs")
    assert count == 2
    client.scan_iter.assert_called_once_with(match="rag:answer:v1:bcbs:*")
    client.delete.assert_called_once_with("rag:answer:v1:bcbs:a", "rag:answer:v1:bcbs:b")


def test_invalidate_company_answer_cache_no_keys_returns_zero() -> None:
    client = MagicMock()
    client.scan_iter.return_value = iter([])
    assert invalidate_company_answer_cache(client, "bcbs") == 0
    client.delete.assert_not_called()


def test_invalidate_company_answer_cache_redis_error_returns_zero() -> None:
    client = MagicMock()
    client.scan_iter.side_effect = redis.exceptions.ConnectionError("down")
    assert invalidate_company_answer_cache(client, "bcbs") == 0
