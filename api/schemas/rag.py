"""Pydantic models for RAG HTTP API (Stages 3–4, 7)."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ConversationTurn(BaseModel):
    """One prior Q&A pair, replayed into the LLM prompt for conversational context (Stage 7)."""

    question: str = Field(..., min_length=1, max_length=4000, description="A prior user question.")
    answer: str = Field(..., min_length=1, max_length=8000, description="The assistant's prior answer.")


class RagQueryRequest(BaseModel):
    """User question scoped to one employer (tier-2 tenant)."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language benefits or coverage question.",
    )
    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Tenant id (must appear in COMPANY_IDS); tier-1 global docs are always included.",
    )
    filter_doc_types: Optional[List[str]] = Field(
        None,
        max_length=16,
        description=(
            "Optional metadata filter: only chunks whose ``doc_type`` is one of these values "
            "(e.g. benefits, formulary, guidelines)."
        ),
    )
    filter_plan_years: Optional[List[str]] = Field(
        None,
        max_length=16,
        description="Optional metadata filter: only chunks whose ``plan_year`` is one of these values.",
    )
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None,
        max_length=30,
        description=(
            "Stage 7: prior turns in this conversation, oldest first. Threaded into the LLM "
            "prompt (trimmed server-side to RAG_MAX_CONVERSATION_TURNS) so follow-up questions "
            "are contextual. A non-empty history bypasses the Stage 5 exact-match answer cache."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_stripped_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped

    @field_validator("company_id")
    @classmethod
    def company_id_stripped(cls, value: str) -> str:
        return value.strip()

    @field_validator("filter_doc_types")
    @classmethod
    def filter_doc_types_normalized(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return None
        out = [v.strip() for v in value if v and str(v).strip()]
        return out or None

    @field_validator("filter_plan_years")
    @classmethod
    def filter_plan_years_normalized(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return None
        out = [v.strip() for v in value if v and str(v).strip()]
        return out or None


class RagCitation(BaseModel):
    """One retrieved passage used as context (and for UI source display)."""

    index: int = Field(..., ge=1, description="1-based index matching context markers in the prompt.")
    source: Optional[str] = Field(None, description="Original PDF or document filename from metadata.")
    doc_type: Optional[str] = Field(None, description="e.g. benefits, formulary, guidelines.")
    plan_year: Optional[str] = Field(None, description="Plan or content year if present.")
    company_id: Optional[str] = Field(None, description="global or employer id from chunk metadata.")
    tier: Optional[int] = Field(None, description="1 = shared global, 2 = employer-specific.")
    distance: Optional[float] = Field(
        None,
        description="Chroma distance for the hit (lower is more similar for cosine space).",
    )
    excerpt: str = Field(
        ...,
        description="Short preview of the chunk (not necessarily full text).",
    )


class RagQueryResponse(BaseModel):
    """LLM answer plus structured citations for transparency."""

    answer: str = Field(..., description="Model response grounded in retrieved context.")
    citations: List[RagCitation] = Field(
        default_factory=list,
        description="Chunks fed into the model, in the same order as [1], [2], … in the prompt.",
    )
    cache_hit: bool = Field(
        False,
        description="True when served from the Stage 5 exact-match Redis answer cache.",
    )


class RagQueryAccepted(BaseModel):
    """Returned (HTTP 202) when RAG_ASYNC_ENABLED and the query is enqueued for a worker."""

    task_id: str = Field(..., description="Celery task id; poll GET /rag/status/{task_id}.")
    status: str = Field("queued", description="Always 'queued' at enqueue time.")


class RagQueryStatus(BaseModel):
    """Current state of an async RAG query enqueued via POST /rag/query."""

    task_id: str = Field(..., description="Celery task id.")
    state: str = Field(..., description="PENDING, STARTED, SUCCESS, FAILURE, or RETRY.")
    result: Optional[RagQueryResponse] = Field(
        None, description="The full answer + citations, present on SUCCESS.",
    )
    error: Optional[str] = Field(None, description="Error message, present on FAILURE.")
