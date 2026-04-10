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

    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key (AI Studio) for embeddings and chat.",
    )
    GEMINI_CHAT_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model id for conversational RAG (e.g. gemini-2.0-flash).",
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description=(
            "Gemini embedding model for RAG (e.g. gemini-embedding-001). "
            "Legacy text-embedding-004 was retired from the API; see Gemini embeddings docs."
        ),
    )
    EMBEDDING_SUB_BATCH_SIZE: int = Field(
        default=50,
        ge=1,
        le=250,
        description=(
            "Texts per embedding API round-trip. Each LangChain batch can map to one HTTP "
            "embed_content call; large PDFs need small batches + delay to avoid free-tier bursts."
        ),
    )
    EMBEDDING_INTER_BATCH_DELAY_SECONDS: float = Field(
        default=0.7,
        ge=0.0,
        description=(
            "Seconds to sleep between embedding sub-batches (free tier: ~100 embed requests/minute)."
        ),
    )
    EMBEDDING_RATE_LIMIT_MAX_RETRIES: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Retries per sub-batch when the provider returns 429 / RESOURCE_EXHAUSTED.",
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


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton per process)."""
    return Settings()
