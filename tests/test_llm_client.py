"""Ollama LLM client validation (embeddings and chat)."""

import pytest

from api.llm_client import get_chat_model, get_embedding_model
from tests.conftest import SettingsForTests


def test_get_embedding_model_ollama_raises_when_base_url_missing() -> None:
    settings = SettingsForTests(
        OLLAMA_BASE_URL="",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
    )
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        get_embedding_model(settings=settings)


def test_get_embedding_model_ollama_raises_when_embedding_model_missing() -> None:
    settings = SettingsForTests(
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_EMBEDDING_MODEL="",
    )
    with pytest.raises(ValueError, match="OLLAMA_EMBEDDING_MODEL"):
        get_embedding_model(settings=settings)


def test_get_chat_model_ollama_raises_when_chat_model_missing() -> None:
    settings = SettingsForTests(
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_CHAT_MODEL="",
    )
    with pytest.raises(ValueError, match="OLLAMA_CHAT_MODEL"):
        get_chat_model(settings=settings)
