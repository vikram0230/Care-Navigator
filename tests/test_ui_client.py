"""Stage 7 UI HTTP client (ui/api_client.py) — pure functions, mocked httpx, no live API."""

from unittest.mock import MagicMock, patch

import httpx

from ui.api_client import (
    ask_question,
    fetch_health,
    get_ingest_status,
    trigger_reseed,
    upload_pdf,
)


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_fetch_health_ok() -> None:
    resp = _mock_response({"status": "healthy"})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = resp
        result = fetch_health("http://api:8000")
    assert result.ok is True
    assert "healthy" in result.message


def test_fetch_health_connection_error() -> None:
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("down")
        result = fetch_health("http://api:8000")
    assert result.ok is False
    assert "Could not reach API" in result.message


def test_ask_question_sends_expected_body_and_no_key_header() -> None:
    resp = _mock_response({"answer": "Your deductible is $500.", "citations": [], "cache_hit": False})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = resp
        result = ask_question(
            "http://api:8000",
            question="What is my deductible?",
            company_id="bcbs",
        )
    assert result.ok is True
    assert result.answer == "Your deductible is $500."
    url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
    assert url == "http://api:8000/rag/query"
    assert kwargs["json"] == {"question": "What is my deductible?", "company_id": "bcbs"}
    assert kwargs["headers"] == {}


def test_ask_question_uses_a_generous_timeout_for_cold_ollama_starts() -> None:
    """A cold Ollama model load alone can take 15s+ before the first token; 30s is too tight."""
    resp = _mock_response({"answer": "ok", "citations": [], "cache_hit": False})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = resp
        ask_question("http://api:8000", question="q", company_id="bcbs")
    timeout = mock_client_cls.call_args.kwargs["timeout"]
    assert timeout >= 60.0


def test_ask_question_attaches_api_key_header() -> None:
    resp = _mock_response({"answer": "ok", "citations": [], "cache_hit": False})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = resp
        ask_question(
            "http://api:8000",
            question="q",
            company_id="bcbs",
            rag_api_key="secret-key",
        )
    kwargs = mock_post.call_args.kwargs
    assert kwargs["headers"] == {"X-API-Key": "secret-key"}


def test_ask_question_includes_conversation_history_when_present() -> None:
    resp = _mock_response({"answer": "ok", "citations": [], "cache_hit": False})
    history = [{"question": "What is my deductible?", "answer": "Your deductible is $500."}]
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = resp
        ask_question(
            "http://api:8000",
            question="And my copay?",
            company_id="bcbs",
            conversation_history=history,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["conversation_history"] == history


def test_ask_question_http_error_returns_structured_result() -> None:
    resp = _mock_response({"detail": "Unknown company_id"}, status_code=400)
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = resp
        result = ask_question("http://api:8000", question="q", company_id="unknown")
    assert result.ok is False
    assert result.error is not None
    assert "400" in result.error


def test_ask_question_timeout_returns_structured_result() -> None:
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timed out")
        result = ask_question("http://api:8000", question="q", company_id="bcbs")
    assert result.ok is False
    assert result.error is not None


def test_upload_pdf_builds_multipart_fields() -> None:
    resp = _mock_response({"task_id": "task-1", "status": "queued"})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = resp
        result = upload_pdf(
            "http://api:8000",
            file_bytes=b"%PDF-1.4",
            filename="benefits.pdf",
            company_id="bcbs",
            tier=2,
            plan_year="2026",
            ingest_api_key="ingest-secret",
        )
    assert result.ok is True
    assert result.task_id == "task-1"
    _, kwargs = mock_post.call_args.args, mock_post.call_args.kwargs
    assert kwargs["data"]["company_id"] == "bcbs"
    assert kwargs["data"]["tier"] == "2"
    assert kwargs["data"]["plan_year"] == "2026"
    assert kwargs["files"]["file"][0] == "benefits.pdf"
    assert kwargs["headers"] == {"X-API-Key": "ingest-secret"}


def test_trigger_reseed_sends_reset_and_full_flags() -> None:
    resp = _mock_response({"task_id": "reseed-1", "status": "queued"})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = resp
        result = trigger_reseed("http://api:8000", reset=True, full=False, ingest_api_key="key")
    assert result.ok is True
    assert result.task_id == "reseed-1"
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"] == {"reset": True, "full": False}
    assert kwargs["headers"] == {"X-API-Key": "key"}


def test_get_ingest_status_parses_state_and_result() -> None:
    resp = _mock_response({"task_id": "t1", "state": "SUCCESS", "result": {"chunks": 5}, "error": None})
    with patch("ui.api_client.httpx.Client") as mock_client_cls:
        mock_get = mock_client_cls.return_value.__enter__.return_value.get
        mock_get.return_value = resp
        result = get_ingest_status("http://api:8000", "t1", ingest_api_key="key")
    assert result.ok is True
    assert result.state == "SUCCESS"
    assert result.result == {"chunks": 5}
    url = mock_get.call_args.args[0]
    assert url == "http://api:8000/ingest/status/t1"
    assert mock_get.call_args.kwargs["headers"] == {"X-API-Key": "key"}
