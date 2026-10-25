"""LLM client validation: Ollama embeddings and Hugging Face chat."""

from unittest.mock import patch

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


def test_get_chat_model_raises_when_token_missing() -> None:
    settings = SettingsForTests(
        HF_TOKEN="",
        HF_CHAT_MODEL="meta-llama/Llama-3.1-8B-Instruct:novita",
    )
    with pytest.raises(ValueError, match="HF_TOKEN"):
        get_chat_model(settings=settings)


def test_get_chat_model_raises_when_model_missing() -> None:
    settings = SettingsForTests(
        HF_TOKEN="test-token",
        HF_CHAT_MODEL="",
    )
    with pytest.raises(ValueError, match="HF_CHAT_MODEL"):
        get_chat_model(settings=settings)


def test_get_chat_model_raises_when_base_url_missing() -> None:
    settings = SettingsForTests(
        HF_TOKEN="test-token",
        HF_CHAT_MODEL="meta-llama/Llama-3.1-8B-Instruct:novita",
        HF_INFERENCE_BASE_URL="",
    )
    with pytest.raises(ValueError, match="HF_INFERENCE_BASE_URL"):
        get_chat_model(settings=settings)


def test_get_chat_model_pins_novita() -> None:
    settings = SettingsForTests(
        HF_INFERENCE_BASE_URL="https://router.huggingface.co/v1",
        HF_CHAT_MODEL="meta-llama/Llama-3.1-8B-Instruct:novita",
        HF_TOKEN="test-token",
    )
    with patch("api.llm_client.ChatOpenAI") as chat_cls:
        get_chat_model(settings=settings)
    chat_cls.assert_called_once_with(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        base_url="https://router.huggingface.co/v1",
        api_key="test-token",
        temperature=0.1,
    )
