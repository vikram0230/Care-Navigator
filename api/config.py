"""Application configuration loaded from environment variables via pydantic-settings."""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API and shared workers.

    All secrets and service endpoints are read from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP API base URL.",
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Ollama model for embeddings; pull with `ollama pull`.",
    )
    OLLAMA_CHAT_MODEL: str = Field(
        default="llama3.2",
        description="Ollama model for RAG chat; pull with `ollama pull`.",
    )
    EMBEDDING_SUB_BATCH_SIZE: int = Field(
        default=50,
        ge=1,
        le=250,
        description="Max texts per embed_documents call; smaller batches ease memory load on Ollama.",
    )
    EMBEDDING_INTER_BATCH_DELAY_SECONDS: float = Field(
        default=0.2,
        ge=0.0,
        description="Seconds to sleep between embedding sub-batches (gentle pacing for local Ollama).",
    )
    RAG_TIER1_TOP_K: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Max Chroma chunks to retrieve from shared global_tier1 per query.",
    )
    RAG_TIER2_TOP_K: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Max Chroma chunks to retrieve from the employer tier-2 collection per query.",
    )
    RAG_MAX_CONTEXT_CHUNKS: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Cap on chunks passed to the LLM after merging tier-1 and tier-2 hits.",
    )
    RAG_API_KEYS: str = Field(
        default="",
        description=(
            "Comma-separated API tokens. When non-empty, POST /rag/query requires "
            "Authorization: Bearer <token> or X-API-Key header matching one of these values."
        ),
    )
    RAG_ANSWER_CACHE_ENABLED: bool = Field(
        default=True,
        description="Stage 5: exact-match Redis cache for full RAG answers (fails open if Redis is down).",
    )
    RAG_ANSWER_CACHE_TTL_SECONDS: int = Field(
        default=3600,
        ge=0,
        description="TTL for cached answers; invalidated early on re-ingest (see scripts/seed_documents.py).",
    )
    RAG_EMBEDDING_CACHE_ENABLED: bool = Field(
        default=True,
        description="Stage 5: Redis cache for question embeddings, keyed by embedding model + text.",
    )
    RAG_EMBEDDING_CACHE_TTL_SECONDS: int = Field(
        default=86400,
        ge=0,
        description="TTL for cached question embeddings.",
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching and optional session storage.",
    )
    CHROMA_HOST: str = Field(
        default="localhost",
        description="ChromaDB server hostname.",
    )
    CHROMA_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="ChromaDB server port.",
    )
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery message broker URL.",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL.",
    )
    COMPANY_IDS: str = Field(
        default="bcbs,wells_fargo",
        description="Comma-separated list of valid tenant company IDs.",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level name.",
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="deployment environment label (development or production).",
    )

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Ensure log level is a valid uppercase name."""
        if isinstance(value, str):
            upper = value.strip().upper()
            valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if upper not in valid:
                raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}")
            return upper
        raise TypeError("LOG_LEVEL must be a string")

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        """Restrict environment to known values."""
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower not in {"development", "production"}:
                raise ValueError("ENVIRONMENT must be 'development' or 'production'")
            return lower
        raise TypeError("ENVIRONMENT must be a string")

    @property
    def company_id_list(self) -> List[str]:
        """Parse COMPANY_IDS into a list of non-empty strings."""
        return [c.strip() for c in self.COMPANY_IDS.split(",") if c.strip()]

    @property
    def rag_api_key_set(self) -> set[str]:
        """Non-empty API keys from ``RAG_API_KEYS``."""
        return {k.strip() for k in self.RAG_API_KEYS.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton per process)."""
    return Settings()
