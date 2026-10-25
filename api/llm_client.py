"""LangChain clients: Ollama embeddings and Hugging Face chat."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

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
    """Build a Hugging Face chat client pinned to the configured provider.

    Raises:
        ValueError: Missing base URL, model id, or token.
        RuntimeError: Client initialization failed.
    """
    resolved = settings or get_settings()
    base = resolved.HF_INFERENCE_BASE_URL.strip()
    if not base:
        raise ValueError("HF_INFERENCE_BASE_URL is not set.")
    model = resolved.HF_CHAT_MODEL.strip()
    if not model:
        raise ValueError("HF_CHAT_MODEL is not set.")
    token = resolved.HF_TOKEN.strip()
    if not token:
        raise ValueError("HF_TOKEN is not set.")
    try:
        return ChatOpenAI(model=model, base_url=base, api_key=token, temperature=0.1)
    except Exception as exc:
        logger.exception("Failed to initialize Hugging Face chat client")
        raise RuntimeError("Could not create Hugging Face chat client") from exc
