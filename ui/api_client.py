"""Pure HTTP client helpers for the Streamlit UI (Stage 7).

No Streamlit imports here so this module is testable in isolation. Every function
returns a small result dataclass instead of raising — network/HTTP errors become a
human-readable ``error`` string so ``ui/app.py`` can render them, never crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

_DEFAULT_TIMEOUT = 30.0
# ask_question does a full synchronous round trip (embed + Chroma retrieval + Ollama chat
# generation) inside a single request. Uncached follow-up turns bypass the answer cache and
# re-run the full pipeline every time — measured at ~130s on local llama3.2 — so a 90s timeout
# reliably errored mid-conversation. Give it real headroom until the async LLM queue lands.
# (The UI shows a live progress indicator during this wait; see ui/app.py.)
_ASK_TIMEOUT = 300.0
_STATUS_TIMEOUT = 10.0
_HEALTH_TIMEOUT = 5.0


@dataclass
class HealthResult:
    ok: bool
    message: str


@dataclass
class AskResult:
    ok: bool
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class IngestEnqueueResult:
    ok: bool
    task_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class IngestStatusResult:
    """``error`` here is the *task's* reported failure, not a transport error (see ``ok``)."""

    ok: bool
    state: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return f"{exc.response.status_code}: {detail}"
    if isinstance(exc, httpx.RequestError):
        return f"Could not reach API ({exc})"
    return str(exc)


def fetch_health(api_base: str) -> HealthResult:
    """Call ``GET /health``."""
    url = f"{api_base.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=_HEALTH_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            return HealthResult(ok=True, message=f"API status: {data.get('status', 'unknown')}")
    except Exception as exc:
        return HealthResult(ok=False, message=_error_message(exc))


def ask_question(
    api_base: str,
    *,
    question: str,
    company_id: str,
    filter_doc_types: Optional[List[str]] = None,
    filter_plan_years: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    rag_api_key: Optional[str] = None,
) -> AskResult:
    """Call ``POST /rag/query``."""
    url = f"{api_base.rstrip('/')}/rag/query"
    body: Dict[str, Any] = {"question": question, "company_id": company_id}
    if filter_doc_types:
        body["filter_doc_types"] = filter_doc_types
    if filter_plan_years:
        body["filter_plan_years"] = filter_plan_years
    if conversation_history:
        body["conversation_history"] = conversation_history
    try:
        with httpx.Client(timeout=_ASK_TIMEOUT) as client:
            response = client.post(url, json=body, headers=_auth_headers(rag_api_key))
            response.raise_for_status()
            data = response.json()
            return AskResult(
                ok=True,
                answer=data.get("answer", ""),
                citations=data.get("citations", []),
                cache_hit=bool(data.get("cache_hit", False)),
            )
    except Exception as exc:
        return AskResult(ok=False, error=_error_message(exc))


def upload_pdf(
    api_base: str,
    *,
    file_bytes: bytes,
    filename: str,
    company_id: str,
    tier: int,
    doc_type: Optional[str] = None,
    plan_year: str = "2025",
    ingest_api_key: Optional[str] = None,
) -> IngestEnqueueResult:
    """Call ``POST /ingest/upload`` (multipart)."""
    url = f"{api_base.rstrip('/')}/ingest/upload"
    data: Dict[str, str] = {"company_id": company_id, "tier": str(tier), "plan_year": plan_year}
    if doc_type:
        data["doc_type"] = doc_type
    files = {"file": (filename, file_bytes, "application/pdf")}
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, data=data, files=files, headers=_auth_headers(ingest_api_key))
            response.raise_for_status()
            payload = response.json()
            return IngestEnqueueResult(ok=True, task_id=payload.get("task_id"))
    except Exception as exc:
        return IngestEnqueueResult(ok=False, error=_error_message(exc))


def trigger_reseed(
    api_base: str,
    *,
    reset: bool = False,
    full: bool = False,
    ingest_api_key: Optional[str] = None,
) -> IngestEnqueueResult:
    """Call ``POST /ingest/reseed``."""
    url = f"{api_base.rstrip('/')}/ingest/reseed"
    body = {"reset": reset, "full": full}
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, json=body, headers=_auth_headers(ingest_api_key))
            response.raise_for_status()
            payload = response.json()
            return IngestEnqueueResult(ok=True, task_id=payload.get("task_id"))
    except Exception as exc:
        return IngestEnqueueResult(ok=False, error=_error_message(exc))


def get_ingest_status(
    api_base: str,
    task_id: str,
    ingest_api_key: Optional[str] = None,
) -> IngestStatusResult:
    """Call ``GET /ingest/status/{task_id}``."""
    url = f"{api_base.rstrip('/')}/ingest/status/{task_id}"
    try:
        with httpx.Client(timeout=_STATUS_TIMEOUT) as client:
            response = client.get(url, headers=_auth_headers(ingest_api_key))
            response.raise_for_status()
            payload = response.json()
            return IngestStatusResult(
                ok=True,
                state=payload.get("state"),
                result=payload.get("result"),
                error=payload.get("error"),
            )
    except Exception as exc:
        return IngestStatusResult(ok=False, error=_error_message(exc))
