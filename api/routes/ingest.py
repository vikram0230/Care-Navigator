"""Stage 6 async ingest endpoints: PDF upload and bundled reseed, both enqueue Celery tasks."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from api.auth import check_ingest_api_key
from api.config import get_settings
from api.schemas.ingest import IngestReseedRequest, IngestStatusResponse, IngestTaskAccepted
from vectordb.ingestion import PDFIngestionPipeline
from workers.celery_app import celery_app
from workers.ingest_tasks import ingest_pdf_task, reseed_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


def _require_ingest_key(request: Request) -> None:
    check_ingest_api_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )


@router.post(
    "/upload",
    response_model=IngestTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and enqueue async ingestion",
    responses={
        401: {"description": "Missing or invalid API key when INGEST_API_KEYS is configured."},
        400: {"description": "Invalid file type, tier, or company_id."},
        413: {"description": "File exceeds INGEST_MAX_UPLOAD_MB."},
    },
)
async def ingest_upload(
    request: Request,
    file: UploadFile = File(...),
    company_id: str = Form(""),
    tier: int = Form(...),
    doc_type: Optional[str] = Form(None),
    plan_year: str = Form("2025"),
    webhook_url: Optional[str] = Form(None),
) -> IngestTaskAccepted:
    """Save an uploaded PDF to the shared upload volume and enqueue ``ingest_pdf_task``."""
    _require_ingest_key(request)
    settings = get_settings()

    if tier not in (1, 2):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="tier must be 1 or 2")

    cid = "global"
    if tier == 2:
        cid = company_id.strip()
        if cid not in settings.company_id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown company_id {cid!r}; allowed: {settings.company_id_list}",
            )

    original_name = file.filename or ""
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .pdf uploads are accepted")
    if file.content_type and file.content_type not in _PDF_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unexpected content type {file.content_type!r}; expected a PDF",
        )

    upload_dir = Path(settings.INGEST_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Path(...).name strips any directory components from the caller-supplied filename,
    # so a value like "../../evil.pdf" cannot escape upload_dir.
    safe_name = Path(original_name).name
    dest_path = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"

    max_bytes = settings.INGEST_MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds INGEST_MAX_UPLOAD_MB={settings.INGEST_MAX_UPLOAD_MB}",
                    )
                out.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    resolved_doc_type = (
        doc_type.strip()
        if doc_type and doc_type.strip()
        else PDFIngestionPipeline._infer_doc_type(Path(original_name).stem, "benefits")
    )

    task = ingest_pdf_task.delay(
        pdf_path=str(dest_path),
        company_id=cid,
        tier=tier,
        doc_type=resolved_doc_type,
        plan_year=plan_year.strip() or "2025",
        source_override=original_name,
        webhook_url=webhook_url,
    )
    return IngestTaskAccepted(task_id=task.id)


@router.post(
    "/reseed",
    response_model=IngestTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run the bundled seed-data ingest asynchronously",
    responses={
        401: {"description": "Missing or invalid API key when INGEST_API_KEYS is configured."},
    },
)
async def ingest_reseed(request: Request, body: IngestReseedRequest) -> IngestTaskAccepted:
    """Enqueue ``reseed_task``, which wraps ``scripts.seed_documents.run_ingestion``."""
    _require_ingest_key(request)
    task = reseed_task.delay(reset=body.reset, full=body.full, webhook_url=body.webhook_url)
    return IngestTaskAccepted(task_id=task.id)


@router.get(
    "/status/{task_id}",
    response_model=IngestStatusResponse,
    summary="Poll the state of a previously enqueued ingest task",
    responses={
        401: {"description": "Missing or invalid API key when INGEST_API_KEYS is configured."},
    },
)
async def ingest_status(request: Request, task_id: str) -> IngestStatusResponse:
    """Look up a Celery task's current state via the result backend."""
    _require_ingest_key(request)
    async_result = celery_app.AsyncResult(task_id)
    state = async_result.state
    result = None
    error = None
    if state == "SUCCESS":
        result = async_result.result
    elif state == "FAILURE":
        error = str(async_result.result)
    return IngestStatusResponse(task_id=task_id, state=state, result=result, error=error)
