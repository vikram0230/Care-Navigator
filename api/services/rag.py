"""Retrieve from Chroma (tier 1 + tier 2) and generate an answer via Ollama."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from chromadb import ClientAPI
from chromadb.errors import ChromaError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from api.config import Settings
from api.llm_client import get_chat_model, get_embedding_model
from api.schemas.rag import RagCitation, RagQueryResponse
from vectordb.collections import get_collection

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 28000
_EXCERPT_LEN = 280

_SYSTEM_PROMPT = """You are Care Navigator, a careful assistant for employee health benefits and coverage questions.
Use ONLY the numbered context passages below. If the answer is not contained in them, say you do not have enough information in the provided materials and suggest the employee contact their HR or benefits administrator.
When you use information from a passage, cite it with the bracketed index like [1] or [2] matching the passage number.
Do not invent plan details, dollar amounts, or coverage rules that are not explicitly supported by the context."""


@dataclass(frozen=True)
class _Hit:
    distance: float
    document: str
    metadata: Dict[str, Any]
    tier: int


def _embed_query(embeddings: Embeddings, text: str) -> List[float]:
    vec = embeddings.embed_query(text)
    return list(vec)


def _chroma_query(
    collection: Any,
    query_embedding: List[float],
    k: int,
    tier: int,
) -> List[_Hit]:
    if k <= 0:
        return []
    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except ChromaError:
        logger.exception("Chroma query failed")
        raise

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    dists = raw.get("distances") or []
    if not docs or not docs[0]:
        return []

    out: List[_Hit] = []
    for i, text in enumerate(docs[0]):
        if text is None:
            continue
        meta = (metas[0][i] if metas and metas[0] and i < len(metas[0]) else {}) or {}
        dist_list = dists[0] if dists and dists[0] else []
        dist = float(dist_list[i]) if i < len(dist_list) else 1.0
        out.append(_Hit(distance=dist, document=str(text), metadata=dict(meta), tier=tier))
    return out


def _merge_hits(tier1_hits: List[_Hit], tier2_hits: List[_Hit], cap: int) -> List[_Hit]:
    merged = sorted(tier1_hits + tier2_hits, key=lambda h: h.distance)
    return merged[:cap]


def _build_context_and_citations(hits: List[_Hit]) -> tuple[str, List[RagCitation]]:
    """Build prompt context and citations for the chunks that fit within the char budget."""
    parts: List[str] = []
    citations: List[RagCitation] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] (tier {h.tier})\n{h.document.strip()}\n"
        if total + len(block) > _MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
        excerpt = h.document.strip().replace("\n", " ")
        if len(excerpt) > _EXCERPT_LEN:
            excerpt = excerpt[: _EXCERPT_LEN - 1] + "…"
        meta = h.metadata
        citations.append(
            RagCitation(
                index=i,
                source=meta.get("source"),
                doc_type=meta.get("doc_type"),
                plan_year=meta.get("plan_year"),
                company_id=meta.get("company_id"),
                tier=h.tier,
                distance=h.distance,
                excerpt=excerpt,
            )
        )
    return "\n".join(parts).strip(), citations


def run_rag_query(
    *,
    chroma_client: ClientAPI,
    settings: Settings,
    question: str,
    company_id: str,
) -> RagQueryResponse:
    """Embed ``question``, retrieve from global tier 1 and employer tier 2, then call the chat model.

    Raises:
        ValueError: Unknown ``company_id`` or missing Ollama configuration.
        ChromaError: Vector store query failure.
        RuntimeError: LLM invocation failure.
    """
    cid = company_id.strip()
    if cid not in settings.company_id_list:
        raise ValueError(
            f"Unknown company_id {cid!r}; allowed: {settings.company_id_list}",
        )

    embeddings = get_embedding_model(settings)
    chat: BaseChatModel = get_chat_model(settings)
    qvec = _embed_query(embeddings, question)

    coll_t1 = get_collection(chroma_client, "global", 1, settings=settings)
    coll_t2 = get_collection(chroma_client, cid, 2, settings=settings)

    hits_t1 = _chroma_query(coll_t1, qvec, settings.RAG_TIER1_TOP_K, tier=1)
    hits_t2 = _chroma_query(coll_t2, qvec, settings.RAG_TIER2_TOP_K, tier=2)
    hits = _merge_hits(hits_t1, hits_t2, settings.RAG_MAX_CONTEXT_CHUNKS)

    if not hits:
        context_block = "(No matching passages were retrieved from the knowledge base.)"
        citations: List[RagCitation] = []
    else:
        context_block, citations = _build_context_and_citations(hits)

    user_content = (
        f"Context passages:\n{context_block}\n\n"
        f"Employee question (company tenant: {cid}):\n{question.strip()}"
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    try:
        ai_msg = chat.invoke(messages)
    except Exception as exc:
        logger.exception("Chat model invoke failed")
        raise RuntimeError("LLM generation failed") from exc

    content = getattr(ai_msg, "content", None)
    answer = content if isinstance(content, str) else str(content or "")

    return RagQueryResponse(answer=answer.strip(), citations=citations)
