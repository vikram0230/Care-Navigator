"""Stage 1 regression tests: health probe and Prometheus metrics."""

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "method,path,min_status,max_status",
    [
        ("GET", "/health", 200, 200),
        ("GET", "/metrics", 200, 200),
        ("GET", "/openapi.json", 200, 200),
    ],
)
def test_public_endpoints_respond(
    client: TestClient,
    method: str,
    path: str,
    min_status: int,
    max_status: int,
) -> None:
    """Core Stage 1 routes return successful HTTP status codes."""
    response = client.request(method, path)
    assert min_status <= response.status_code <= max_status


def test_health_payload_shape(client: TestClient) -> None:
    """GET /health returns structured JSON for orchestration probes."""
    response = client.get("/health")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["status"] == "healthy"
    assert data["environment"] in ("development", "production")
    assert data["service"] == "care-navigator-api"


def test_metrics_contains_httpInstrumentation(client: TestClient) -> None:
    """GET /metrics exposes Prometheus text with HTTP request metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "http_requests_total" in text or "http_request" in text
