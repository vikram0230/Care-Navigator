"""Pydantic models for RAG HTTP API (Stage 3)."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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
