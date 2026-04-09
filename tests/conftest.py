"""Shared pytest fixtures for API tests."""

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

# Load app only after test-friendly defaults are set (before Settings is cached).
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("COMPANY_IDS", "acme_corp,globex_inc")

from api.config import get_settings
from api.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a synchronous TestClient with lifespan hooks executed."""
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
