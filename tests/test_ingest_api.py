"""Stage 6 ingest route tests: auth gating, upload validation, reseed enqueue, status polling.

No real Celery broker/worker or Chroma/Ollama needed — ``.delay()`` and ``AsyncResult`` are mocked.
"""

from pathlib import Path
from typing import Any, Dict, Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.deps import reset_chroma_singleton

_MIN_PDF = b"%PDF-1.4\n%dummy pdf bytes for upload tests\n%%EOF"


@pytest.fixture
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point INGEST_UPLOAD_DIR at an isolated temp directory for the duration of a test."""
    monkeypatch.setenv("INGEST_UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    reset_chroma_singleton()
    yield tmp_path
    monkeypatch.delenv("INGEST_UPLOAD_DIR", raising=False)
    get_settings.cache_clear()
    reset_chroma_singleton()


def test_ingest_upload_requires_ingest_api_key(
    client: TestClient,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_API_KEYS", "ingest-secret")
    get_settings.cache_clear()
    try:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "bcbs", "tier": "2"},
        )
        assert response.status_code == 401
    finally:
        monkeypatch.delenv("INGEST_API_KEYS", raising=False)
        get_settings.cache_clear()


def test_ingest_upload_rag_key_alone_is_not_sufficient(
    client: TestClient,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEYS", "rag-secret")
    monkeypatch.setenv("INGEST_API_KEYS", "ingest-secret")
    get_settings.cache_clear()
    try:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "bcbs", "tier": "2"},
            headers={"X-API-Key": "rag-secret"},
        )
        assert response.status_code == 401
    finally:
        monkeypatch.delenv("RAG_API_KEYS", raising=False)
        monkeypatch.delenv("INGEST_API_KEYS", raising=False)
        get_settings.cache_clear()


def test_ingest_upload_valid_pdf_enqueues_task_and_saves_file(
    client: TestClient,
    upload_dir: Path,
) -> None:
    fake_task = MagicMock()
    fake_task.id = "task-123"
    with patch("api.routes.ingest.ingest_pdf_task.delay", return_value=fake_task) as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "bcbs", "tier": "2", "plan_year": "2025"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body == {"task_id": "task-123", "status": "queued"}

    mock_delay.assert_called_once()
    kwargs = mock_delay.call_args.kwargs
    assert kwargs["company_id"] == "bcbs"
    assert kwargs["tier"] == 2
    assert kwargs["source_override"] == "benefits.pdf"
    saved_path = Path(kwargs["pdf_path"])
    assert saved_path.parent == upload_dir
    assert saved_path.exists()
    assert saved_path.read_bytes() == _MIN_PDF


def test_ingest_upload_infers_doc_type_when_not_supplied(
    client: TestClient,
    upload_dir: Path,
) -> None:
    fake_task = MagicMock()
    fake_task.id = "task-456"
    with patch("api.routes.ingest.ingest_pdf_task.delay", return_value=fake_task) as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("wells-fargo-formulary.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "wells_fargo", "tier": "2"},
        )
    assert response.status_code == 202
    assert mock_delay.call_args.kwargs["doc_type"] == "formulary"


def test_ingest_upload_rejects_non_pdf_extension(
    client: TestClient,
    upload_dir: Path,
) -> None:
    with patch("api.routes.ingest.ingest_pdf_task.delay") as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("notes.txt", b"just text", "text/plain")},
            data={"company_id": "bcbs", "tier": "2"},
        )
    assert response.status_code == 400
    mock_delay.assert_not_called()


def test_ingest_upload_rejects_wrong_content_type(
    client: TestClient,
    upload_dir: Path,
) -> None:
    with patch("api.routes.ingest.ingest_pdf_task.delay") as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "text/plain")},
            data={"company_id": "bcbs", "tier": "2"},
        )
    assert response.status_code == 400
    mock_delay.assert_not_called()


def test_ingest_upload_rejects_oversized_file(
    client: TestClient,
    upload_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    oversized = b"%PDF-1.4\n" + (b"0" * (2 * 1024 * 1024))
    try:
        with patch("api.routes.ingest.ingest_pdf_task.delay") as mock_delay:
            response = client.post(
                "/ingest/upload",
                files={"file": ("big.pdf", oversized, "application/pdf")},
                data={"company_id": "bcbs", "tier": "2"},
            )
        assert response.status_code == 413
        mock_delay.assert_not_called()
        assert list(upload_dir.iterdir()) == []
    finally:
        monkeypatch.delenv("INGEST_MAX_UPLOAD_MB", raising=False)
        get_settings.cache_clear()


def test_ingest_upload_rejects_unknown_company_id_for_tier2(
    client: TestClient,
    upload_dir: Path,
) -> None:
    with patch("api.routes.ingest.ingest_pdf_task.delay") as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "unknown_corp", "tier": "2"},
        )
    assert response.status_code == 400
    mock_delay.assert_not_called()


def test_ingest_upload_rejects_invalid_tier(
    client: TestClient,
    upload_dir: Path,
) -> None:
    with patch("api.routes.ingest.ingest_pdf_task.delay") as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("benefits.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "bcbs", "tier": "3"},
        )
    assert response.status_code == 400
    mock_delay.assert_not_called()


def test_ingest_upload_sanitizes_path_traversal_filename(
    client: TestClient,
    upload_dir: Path,
) -> None:
    fake_task = MagicMock()
    fake_task.id = "task-789"
    with patch("api.routes.ingest.ingest_pdf_task.delay", return_value=fake_task) as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("../../evil.pdf", _MIN_PDF, "application/pdf")},
            data={"company_id": "bcbs", "tier": "2"},
        )
    assert response.status_code == 202
    saved_path = Path(mock_delay.call_args.kwargs["pdf_path"])
    assert saved_path.parent == upload_dir
    assert ".." not in saved_path.parts
    assert saved_path.name.endswith("_evil.pdf")


def test_ingest_upload_tier1_forces_global_company(
    client: TestClient,
    upload_dir: Path,
) -> None:
    fake_task = MagicMock()
    fake_task.id = "task-t1"
    with patch("api.routes.ingest.ingest_pdf_task.delay", return_value=fake_task) as mock_delay:
        response = client.post(
            "/ingest/upload",
            files={"file": ("guidelines.pdf", _MIN_PDF, "application/pdf")},
            data={"tier": "1"},
        )
    assert response.status_code == 202
    assert mock_delay.call_args.kwargs["company_id"] == "global"


def test_ingest_reseed_requires_ingest_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_API_KEYS", "ingest-secret")
    get_settings.cache_clear()
    try:
        response = client.post("/ingest/reseed", json={"reset": False, "full": False})
        assert response.status_code == 401
    finally:
        monkeypatch.delenv("INGEST_API_KEYS", raising=False)
        get_settings.cache_clear()


def test_ingest_reseed_enqueues_task(client: TestClient) -> None:
    fake_task = MagicMock()
    fake_task.id = "reseed-task-1"
    with patch("api.routes.ingest.reseed_task.delay", return_value=fake_task) as mock_delay:
        response = client.post(
            "/ingest/reseed",
            json={"reset": True, "full": False, "webhook_url": "https://example.com/hook"},
        )
    assert response.status_code == 202
    assert response.json() == {"task_id": "reseed-task-1", "status": "queued"}
    mock_delay.assert_called_once_with(reset=True, full=False, webhook_url="https://example.com/hook")


def test_ingest_reseed_rejects_non_http_webhook_url(client: TestClient) -> None:
    response = client.post(
        "/ingest/reseed",
        json={"reset": False, "full": False, "webhook_url": "javascript:alert(1)"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("celery_state", "celery_result", "expected_result", "expected_error"),
    [
        ("PENDING", None, None, None),
        ("SUCCESS", {"chunks": 5}, {"chunks": 5}, None),
        ("FAILURE", RuntimeError("boom"), None, "boom"),
    ],
)
def test_ingest_status_maps_celery_states(
    client: TestClient,
    celery_state: str,
    celery_result: Any,
    expected_result: Dict[str, Any],
    expected_error: str,
) -> None:
    fake_async_result = MagicMock()
    fake_async_result.state = celery_state
    fake_async_result.result = celery_result
    with patch("api.routes.ingest.celery_app.AsyncResult", return_value=fake_async_result):
        response = client.get("/ingest/status/some-task-id")
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "some-task-id"
    assert body["state"] == celery_state
    assert body["result"] == expected_result
    assert body["error"] == expected_error


def test_ingest_status_requires_ingest_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_API_KEYS", "ingest-secret")
    get_settings.cache_clear()
    try:
        response = client.get("/ingest/status/some-task-id")
        assert response.status_code == 401
    finally:
        monkeypatch.delenv("INGEST_API_KEYS", raising=False)
        get_settings.cache_clear()
