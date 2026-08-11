"""Stage 3 RAG endpoint: retrieve from Chroma and answer via Ollama."""

import asyncio
import logging
from typing import Union

from chromadb.errors import ChromaError
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.auth import check_rag_api_key
from api.cache import answer_cache_key, get_cached_answer
from api.config import Settings, get_settings
from api.deps import get_chroma_singleton, get_redis_singleton
from api.schemas.rag import (
    RagQueryAccepted,
    RagQueryRequest,
    RagQueryResponse,
    RagQueryStatus,
)
from api.services.rag import _validate_rag_question, run_rag_query
from workers.celery_app import celery_app
from workers.rag_tasks import rag_query_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


def _require_llm_configured(settings: Settings) -> None:
    """Raise HTTP 503 if Ollama URL or model names are missing."""
    if not settings.OLLAMA_BASE_URL.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OLLAMA_BASE_URL is not configured",
        )
    if not settings.OLLAMA_EMBEDDING_MODEL.strip() or not settings.OLLAMA_CHAT_MODEL.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OLLAMA_EMBEDDING_MODEL and OLLAMA_CHAT_MODEL are required",
        )


def _try_answer_cache(settings: Settings, body: RagQueryRequest, cid: str) -> Union[RagQueryResponse, None]:
    """Synchronous exact-match cache short-circuit (skipped for follow-up turns). None on miss."""
    if not settings.RAG_ANSWER_CACHE_ENABLED or body.conversation_history:
        return None
    redis_client = get_redis_singleton()
    if redis_client is None:
        return None
    key = answer_cache_key(
        cid,
        body.question,
        body.filter_doc_types,
        body.filter_plan_years,
        settings.OLLAMA_EMBEDDING_MODEL,
        settings.OLLAMA_CHAT_MODEL,
    )
    cached = get_cached_answer(redis_client, key)
    if cached is None:
        return None
    return cached.model_copy(update={"cache_hit": True})


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Ask a benefits question (RAG)",
    status_code=status.HTTP_200_OK,
    responses={
        202: {"model": RagQueryAccepted, "description": "Enqueued (RAG_ASYNC_ENABLED); poll /rag/status/{task_id}."},
        401: {"description": "Missing or invalid API key when RAG_API_KEYS is configured."},
    },
)
async def rag_query(
    request: Request, body: RagQueryRequest
) -> Union[RagQueryResponse, JSONResponse]:
    """Retrieve relevant chunks from global + employer collections and generate an answer.

    When ``RAG_ASYNC_ENABLED`` the LLM work is enqueued on Celery and this returns ``202`` with a
    ``task_id`` (poll ``GET /rag/status/{task_id}``); an exact-match answer-cache hit still returns
    ``200`` immediately. Otherwise the query runs synchronously in-request.
    """
    check_rag_api_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    settings = get_settings()
    cid = body.company_id.strip()
    if cid not in settings.company_id_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown company_id {cid!r}; allowed: {settings.company_id_list}",
        )
    _require_llm_configured(settings)

    if settings.RAG_ASYNC_ENABLED:
        # Validate synchronously before enqueuing so a malformed question fails fast with 400
        # instead of becoming a queued task that errors later.
        try:
            _validate_rag_question(body.question)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        cached = _try_answer_cache(settings, body, cid)
        if cached is not None:
            return cached
        task = rag_query_task.delay(
            question=body.question,
            company_id=body.company_id,
            filter_doc_types=body.filter_doc_types,
            filter_plan_years=body.filter_plan_years,
            conversation_history=(
                [turn.model_dump() for turn in body.conversation_history]
                if body.conversation_history
                else None
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=RagQueryAccepted(task_id=task.id).model_dump(),
        )

    try:
        chroma = get_chroma_singleton()
        redis_client = get_redis_singleton()
        return await asyncio.to_thread(
            run_rag_query,
            chroma_client=chroma,
            settings=settings,
            question=body.question,
            company_id=body.company_id,
            filter_doc_types=body.filter_doc_types,
            filter_plan_years=body.filter_plan_years,
            conversation_history=body.conversation_history,
            redis_client=redis_client,
        )
    except ValueError as exc:
        msg = str(exc)
        if "Unknown company_id" in msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg,
            ) from exc
        if msg.startswith("Question failed validation:"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg,
            ) from exc
        if "OLLAMA_" in msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg,
        ) from exc
    except ChromaError as exc:
        logger.exception("Chroma error during RAG query")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error",
        ) from exc
    except ConnectionError as exc:
        logger.exception("Chroma connection error during RAG query")
        detail = str(exc).strip() or "Could not reach vector store"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from exc
    except RuntimeError as exc:
        logger.exception("LLM error during RAG query")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc) or "LLM generation failed",
        ) from exc


@router.get(
    "/status/{task_id}",
    response_model=RagQueryStatus,
    summary="Poll the state of an async RAG query",
    responses={
        401: {"description": "Missing or invalid API key when RAG_API_KEYS is configured."},
    },
)
async def rag_status(request: Request, task_id: str) -> RagQueryStatus:
    """Look up an enqueued RAG query's state (and answer, on SUCCESS) via the Celery result backend."""
    check_rag_api_key(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
    )
    async_result = celery_app.AsyncResult(task_id)
    state = async_result.state
    result = None
    error = None
    if state == "SUCCESS":
        result = async_result.result  # dict; RagQueryStatus coerces it to RagQueryResponse
    elif state == "FAILURE":
        error = str(async_result.result)
    return RagQueryStatus(task_id=task_id, state=state, result=result, error=error)
