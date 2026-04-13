"""LangChain clients for Ollama (embeddings + chat)."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from api.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_embedding_model(settings: Optional[Settings] = None) -> Embeddings:
    """Build ``OllamaEmbeddings`` from settings.

    Raises:
        ValueError: Missing Ollama URL or embedding model name.
        RuntimeError: Client initialization failed.
    """
    resolved = settings or get_settings()
    base = resolved.OLLAMA_BASE_URL.strip()
    if not base:
        raise ValueError("OLLAMA_BASE_URL is not set.")
    model = resolved.OLLAMA_EMBEDDING_MODEL.strip()
    if not model:
        raise ValueError("OLLAMA_EMBEDDING_MODEL is not set.")
    try:
        return OllamaEmbeddings(model=model, base_url=base)
    except Exception as exc:
        logger.exception("Failed to initialize OllamaEmbeddings")
        raise RuntimeError("Could not create Ollama embeddings client") from exc


def get_chat_model(settings: Optional[Settings] = None) -> BaseChatModel:
    """Build ``ChatOllama`` from settings.

    Raises:
        ValueError: Missing Ollama URL or chat model name.
        RuntimeError: Client initialization failed.
    """
    resolved = settings or get_settings()
    base = resolved.OLLAMA_BASE_URL.strip()
    if not base:
        raise ValueError("OLLAMA_BASE_URL is not set.")
    model = resolved.OLLAMA_CHAT_MODEL.strip()
    if not model:
        raise ValueError("OLLAMA_CHAT_MODEL is not set.")
    try:
        return ChatOllama(model=model, base_url=base, temperature=0.1)
    except Exception as exc:
        logger.exception("Failed to initialize ChatOllama")
        raise RuntimeError("Could not create Ollama chat client") from exc
