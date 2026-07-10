"""Stage 4/6: API key gating for RAG and ingest routes (independent key sets)."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from api.auth import check_ingest_api_key, check_rag_api_key
from tests.conftest import SettingsForTests


def _settings_with(**overrides: str) -> SettingsForTests:
    defaults = {"RAG_API_KEYS": "", "INGEST_API_KEYS": ""}
    defaults.update(overrides)
    return SettingsForTests(**defaults)


def test_check_rag_api_key_noop_when_empty() -> None:
    with patch("api.auth.get_settings", return_value=_settings_with()):
        check_rag_api_key(None, None)  # must not raise


def test_check_ingest_api_key_noop_when_empty() -> None:
    with patch("api.auth.get_settings", return_value=_settings_with()):
        check_ingest_api_key(None, None)  # must not raise


def test_check_rag_api_key_accepts_valid_bearer() -> None:
    settings = _settings_with(RAG_API_KEYS="good-key")
    with patch("api.auth.get_settings", return_value=settings):
        check_rag_api_key("Bearer good-key", None)  # must not raise


def test_check_rag_api_key_accepts_valid_x_api_key() -> None:
    settings = _settings_with(RAG_API_KEYS="good-key")
    with patch("api.auth.get_settings", return_value=settings):
        check_rag_api_key(None, "good-key")  # must not raise


def test_check_rag_api_key_rejects_missing() -> None:
    settings = _settings_with(RAG_API_KEYS="good-key")
    with patch("api.auth.get_settings", return_value=settings):
        with pytest.raises(HTTPException) as exc_info:
            check_rag_api_key(None, None)
    assert exc_info.value.status_code == 401


def test_check_rag_api_key_rejects_invalid_token() -> None:
    settings = _settings_with(RAG_API_KEYS="good-key")
    with patch("api.auth.get_settings", return_value=settings):
        with pytest.raises(HTTPException) as exc_info:
            check_rag_api_key("Bearer wrong-key", None)
    assert exc_info.value.status_code == 401


def test_check_ingest_api_key_accepts_valid_bearer() -> None:
    settings = _settings_with(INGEST_API_KEYS="ingest-key")
    with patch("api.auth.get_settings", return_value=settings):
        check_ingest_api_key("Bearer ingest-key", None)  # must not raise


def test_check_ingest_api_key_rejects_missing() -> None:
    settings = _settings_with(INGEST_API_KEYS="ingest-key")
    with patch("api.auth.get_settings", return_value=settings):
        with pytest.raises(HTTPException) as exc_info:
            check_ingest_api_key(None, None)
    assert exc_info.value.status_code == 401


def test_rag_key_does_not_satisfy_ingest_gate() -> None:
    """RAG_API_KEYS and INGEST_API_KEYS are independent key sets."""
    settings = _settings_with(RAG_API_KEYS="rag-key", INGEST_API_KEYS="ingest-key")
    with patch("api.auth.get_settings", return_value=settings):
        with pytest.raises(HTTPException) as exc_info:
            check_ingest_api_key("Bearer rag-key", None)
    assert exc_info.value.status_code == 401


def test_ingest_key_does_not_satisfy_rag_gate() -> None:
    settings = _settings_with(RAG_API_KEYS="rag-key", INGEST_API_KEYS="ingest-key")
    with patch("api.auth.get_settings", return_value=settings):
        with pytest.raises(HTTPException) as exc_info:
            check_rag_api_key("Bearer ingest-key", None)
    assert exc_info.value.status_code == 401
