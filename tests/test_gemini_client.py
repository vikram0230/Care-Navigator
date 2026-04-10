"""Tests for Gemini client factory (configuration guardrails)."""

import pytest

from api.gemini_client import get_gemini_chat_model, get_gemini_embeddings
from tests.conftest import SettingsForTests


def test_get_gemini_embeddings_raises_when_key_missing() -> None:
    """Factory must fail fast when GEMINI_API_KEY is unset."""
    settings = SettingsForTests(GEMINI_API_KEY="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_gemini_embeddings(settings=settings)


def test_get_gemini_chat_model_raises_when_key_missing() -> None:
    """Chat factory must fail fast when GEMINI_API_KEY is unset."""
    settings = SettingsForTests(GEMINI_API_KEY="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_gemini_chat_model(settings=settings)
