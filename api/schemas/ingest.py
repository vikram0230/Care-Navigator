"""Pydantic models for the Stage 6 async ingest HTTP API."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class IngestReseedRequest(BaseModel):
    """Trigger an async re-run of the bundled seed-data ingest."""

    reset: bool = Field(
        False,
        description="Wipe Chroma before seeding (development only; scripts/seed_documents.py enforces this).",
    )
    full: bool = Field(
        False,
        description="Ingest every global PDF plus all tier-2 employer PDFs, not just the minimal seed file.",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Optional URL the worker POSTs a completion/failure payload to when the task finishes.",
    )

    @field_validator("webhook_url")
    @classmethod
    def webhook_url_looks_like_http_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not (stripped.startswith("http://") or stripped.startswith("https://")):
            raise ValueError("webhook_url must start with http:// or https://")
        return stripped


class IngestTaskAccepted(BaseModel):
    """Returned immediately after an ingest task is enqueued."""

    task_id: str = Field(..., description="Celery task id; poll GET /ingest/status/{task_id}.")
    status: str = Field("queued", description="Always 'queued' at enqueue time.")


class IngestStatusResponse(BaseModel):
    """Current state of a previously enqueued ingest task."""

    task_id: str = Field(..., description="Celery task id.")
    state: str = Field(..., description="PENDING, STARTED, SUCCESS, FAILURE, or RETRY.")
    result: Optional[Dict[str, Any]] = Field(None, description="Task return value, present on SUCCESS.")
    error: Optional[str] = Field(None, description="Error message, present on FAILURE.")
