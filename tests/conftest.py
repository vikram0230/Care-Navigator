"""Shared pytest fixtures for API tests."""

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

# Load app only after test-friendly defaults are set (before Settings is cached).
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("COMPANY_IDS", "bcbs,wells_fargo")
# Neutralize local .env bleed-through: the deployment sets RAG_ASYNC_ENABLED=true, but the default
# route tests exercise the synchronous path. Setting it here (os.environ wins over .env) makes the
# suite hermetic; the async-path tests opt in with monkeypatch.setenv("RAG_ASYNC_ENABLED", "true").
os.environ.setdefault("RAG_ASYNC_ENABLED", "false")

from api.config import Settings, get_settings
from api.deps import reset_chroma_singleton
from api.main import app


class SettingsForTests(Settings):
    """``Settings`` that ignore ``.env`` for deterministic unit tests."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a synchronous TestClient with lifespan hooks executed."""
    get_settings.cache_clear()
    reset_chroma_singleton()
    with TestClient(app) as test_client:
        yield test_client
    reset_chroma_singleton()
    get_settings.cache_clear()
