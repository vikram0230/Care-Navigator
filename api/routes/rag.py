"""Stage 3 RAG endpoint: retrieve from Chroma and answer via Ollama."""

import asyncio
import logging

from chromadb.errors import ChromaError
from fastapi import APIRouter, HTTPException, status

from api.config import Settings, get_settings
from api.deps import get_chroma_singleton
from api.schemas.rag import RagQueryRequest, RagQueryResponse
from api.services.rag import run_rag_query

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


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Ask a benefits question (RAG)",
    status_code=status.HTTP_200_OK,
)
async def rag_query(body: RagQueryRequest) -> RagQueryResponse:
    """Retrieve relevant chunks from global + employer collections and generate an answer."""
    settings = get_settings()
    cid = body.company_id.strip()
    if cid not in settings.company_id_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown company_id {cid!r}; allowed: {settings.company_id_list}",
        )
    _require_llm_configured(settings)
    try:
        chroma = get_chroma_singleton()
        return await asyncio.to_thread(
            run_rag_query,
            chroma_client=chroma,
            settings=settings,
            question=body.question,
            company_id=body.company_id,
        )
    except ValueError as exc:
        msg = str(exc)
        if "Unknown company_id" in msg:
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
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach vector store",
        ) from exc
    except RuntimeError as exc:
        logger.exception("LLM error during RAG query")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc) or "LLM generation failed",
        ) from exc
