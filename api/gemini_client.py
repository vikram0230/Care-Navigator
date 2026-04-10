"""Construct LangChain Gemini chat and embedding clients from application settings."""

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from api.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_gemini_embeddings(settings: Optional[Settings] = None) -> GoogleGenerativeAIEmbeddings:
    """Build a Gemini embedding model for Chroma ingestion and semantic cache.

    Args:
        settings: Optional settings override; defaults to cached ``get_settings()``.

    Returns:
        Configured ``GoogleGenerativeAIEmbeddings`` instance.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is missing or whitespace-only.
    """
    resolved = settings or get_settings()
    key = resolved.GEMINI_API_KEY.strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set; add it to the environment or .env file.",
        )
    try:
        # PRODUCTION NOTE: In production this would use a local embedding model
        # for HIPAA compliance instead of sending PHI to Google APIs.
        return GoogleGenerativeAIEmbeddings(
            model=resolved.GEMINI_EMBEDDING_MODEL,
            google_api_key=key,
        )
    except Exception as exc:
        logger.exception("Failed to initialize GoogleGenerativeAIEmbeddings")
        raise RuntimeError("Could not create Gemini embeddings client") from exc


def get_gemini_chat_model(settings: Optional[Settings] = None) -> ChatGoogleGenerativeAI:
    """Build a Gemini chat model for RAG answer generation.

    Args:
        settings: Optional settings override; defaults to cached ``get_settings()``.

    Returns:
        Configured ``ChatGoogleGenerativeAI`` instance.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is missing or whitespace-only.
    """
    resolved = settings or get_settings()
    key = resolved.GEMINI_API_KEY.strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set; add it to the environment or .env file.",
        )
    try:
        # PRODUCTION NOTE: In production this would use local Llama 3 for HIPAA compliance.
        return ChatGoogleGenerativeAI(
            model=resolved.GEMINI_CHAT_MODEL,
            google_api_key=key,
            temperature=0.1,
        )
    except Exception as exc:
        logger.exception("Failed to initialize ChatGoogleGenerativeAI")
        raise RuntimeError("Could not create Gemini chat client") from exc
