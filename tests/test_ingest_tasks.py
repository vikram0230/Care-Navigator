"""Stage 6 Celery tasks: async PDF ingest + bundled reseed, with fail-open cache flush and webhooks."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.conftest import SettingsForTests
from workers.ingest_tasks import _notify_webhook, ingest_pdf_task, reseed_task


def _settings(**overrides: object) -> SettingsForTests:
    defaults = dict(
        COMPANY_IDS="bcbs,wells_fargo",
        INGEST_WEBHOOK_TIMEOUT_SECONDS=5.0,
        EMBEDDING_SUB_BATCH_SIZE=50,
        EMBEDDING_INTER_BATCH_DELAY_SECONDS=0.0,
        REDIS_URL="redis://localhost:6379/0",
    )
    defaults.update(overrides)
    return SettingsForTests(**defaults)


def test_notify_webhook_noop_when_url_none() -> None:
    with patch("workers.ingest_tasks.httpx.post") as mock_post:
        _notify_webhook(None, {"a": 1}, 5.0)
    mock_post.assert_not_called()


def test_notify_webhook_posts_payload() -> None:
    with patch("workers.ingest_tasks.httpx.post") as mock_post:
        _notify_webhook("https://example.com/hook", {"a": 1}, 5.0)
    mock_post.assert_called_once_with("https://example.com/hook", json={"a": 1}, timeout=5.0)


def test_notify_webhook_swallows_httpx_errors() -> None:
    with patch("workers.ingest_tasks.httpx.post", side_effect=httpx.ConnectError("down")):
        _notify_webhook("https://example.com/hook", {"a": 1}, 5.0)  # must not raise


def test_ingest_pdf_task_success_tier2_invalidates_single_company_and_notifies_webhook() -> None:
    settings = _settings()
    pipeline_instance = MagicMock()
    pipeline_instance.ingest_pdf.return_value = 7

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("workers.ingest_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.ingest_tasks.get_embedding_model", return_value=MagicMock()), \
        patch("workers.ingest_tasks.create_collections"), \
        patch("workers.ingest_tasks.PDFIngestionPipeline", return_value=pipeline_instance), \
        patch("workers.ingest_tasks.invalidate_company_answer_cache") as mock_invalidate, \
        patch("workers.ingest_tasks.redis.Redis.from_url", return_value=MagicMock()), \
        patch("workers.ingest_tasks.httpx.post") as mock_post:
        result = ingest_pdf_task(
            pdf_path="/app/data/uploads/abc_benefits.pdf",
            company_id="bcbs",
            tier=2,
            doc_type="benefits",
            plan_year="2025",
            source_override="benefits.pdf",
            webhook_url="https://example.com/hook",
        )

    assert result["status"] == "success"
    assert result["chunks"] == 7
    assert result["company_id"] == "bcbs"
    mock_invalidate.assert_called_once()
    assert mock_invalidate.call_args.args[1] == "bcbs"
    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["status"] == "success"
    assert payload["chunks"] == 7


def test_ingest_pdf_task_success_tier1_invalidates_all_companies() -> None:
    settings = _settings(COMPANY_IDS="bcbs,wells_fargo")
    pipeline_instance = MagicMock()
    pipeline_instance.ingest_pdf.return_value = 3

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("workers.ingest_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.ingest_tasks.get_embedding_model", return_value=MagicMock()), \
        patch("workers.ingest_tasks.create_collections"), \
        patch("workers.ingest_tasks.PDFIngestionPipeline", return_value=pipeline_instance), \
        patch("workers.ingest_tasks.invalidate_company_answer_cache") as mock_invalidate, \
        patch("workers.ingest_tasks.redis.Redis.from_url", return_value=MagicMock()), \
        patch("workers.ingest_tasks.httpx.post"):
        ingest_pdf_task(
            pdf_path="/app/data/uploads/abc_immunization.pdf",
            company_id="global",
            tier=1,
            doc_type="guidelines",
            plan_year="2025",
        )

    called_companies = {call.args[1] for call in mock_invalidate.call_args_list}
    assert called_companies == {"bcbs", "wells_fargo"}


def test_ingest_pdf_task_failure_notifies_webhook_and_reraises() -> None:
    settings = _settings()

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("workers.ingest_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.ingest_tasks.get_embedding_model", return_value=MagicMock()), \
        patch("workers.ingest_tasks.create_collections"), \
        patch("workers.ingest_tasks.PDFIngestionPipeline", side_effect=RuntimeError("embedding failed")), \
        patch("workers.ingest_tasks.httpx.post") as mock_post:
        with pytest.raises(RuntimeError, match="embedding failed"):
            ingest_pdf_task(
                pdf_path="/app/data/uploads/bad.pdf",
                company_id="bcbs",
                tier=2,
                doc_type="benefits",
                plan_year="2025",
                webhook_url="https://example.com/hook",
            )

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["status"] == "failed"
    assert "embedding failed" in payload["error"]


def test_ingest_pdf_task_no_webhook_url_skips_notification() -> None:
    settings = _settings()
    pipeline_instance = MagicMock()
    pipeline_instance.ingest_pdf.return_value = 1

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("workers.ingest_tasks.get_chroma_client", return_value=MagicMock()), \
        patch("workers.ingest_tasks.get_embedding_model", return_value=MagicMock()), \
        patch("workers.ingest_tasks.create_collections"), \
        patch("workers.ingest_tasks.PDFIngestionPipeline", return_value=pipeline_instance), \
        patch("workers.ingest_tasks.invalidate_company_answer_cache"), \
        patch("workers.ingest_tasks.redis.Redis.from_url", return_value=MagicMock()), \
        patch("workers.ingest_tasks.httpx.post") as mock_post:
        ingest_pdf_task(
            pdf_path="/app/data/uploads/abc.pdf",
            company_id="bcbs",
            tier=2,
            doc_type="benefits",
            plan_year="2025",
        )

    mock_post.assert_not_called()


def test_reseed_task_success_calls_run_ingestion_and_notifies_webhook() -> None:
    settings = _settings()

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("scripts.seed_documents.run_ingestion", return_value={"tier1:a.pdf": 4}) as mock_run, \
        patch("workers.ingest_tasks.httpx.post") as mock_post:
        result = reseed_task(reset=True, full=False, webhook_url="https://example.com/hook")

    mock_run.assert_called_once_with(reset_chroma=True, full=False)
    assert result == {"tier1:a.pdf": 4}
    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["status"] == "success"
    assert payload["counts"] == {"tier1:a.pdf": 4}


def test_reseed_task_failure_notifies_webhook_and_reraises() -> None:
    settings = _settings()

    with patch("workers.ingest_tasks.get_settings", return_value=settings), \
        patch("scripts.seed_documents.run_ingestion", side_effect=ValueError("bad config")), \
        patch("workers.ingest_tasks.httpx.post") as mock_post:
        with pytest.raises(ValueError, match="bad config"):
            reseed_task(reset=False, full=True, webhook_url="https://example.com/hook")

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["status"] == "failed"
    assert "bad config" in payload["error"]
