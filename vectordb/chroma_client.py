"""Factory for a persistent ChromaDB HTTP client."""

import logging
from typing import Optional

import chromadb
from chromadb import ClientAPI
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError

from api.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_chroma_client(settings: Optional[Settings] = None) -> ClientAPI:
    """Create a ChromaDB HTTP client pointed at the configured host and port.

    Args:
        settings: Optional ``Settings`` override; defaults to ``get_settings()``.

    Returns:
        Connected ``ClientAPI`` instance.

    Raises:
        ConnectionError: If the server is unreachable or heartbeat fails.
        ChromaError: For other Chroma-specific failures.
    """
    resolved = settings or get_settings()
    host = resolved.CHROMA_HOST.strip()
    port = resolved.CHROMA_PORT
    try:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        client.heartbeat()
        logger.info("Connected to ChromaDB at %s:%s", host, port)
        return client
    except ChromaError as exc:
        logger.exception("ChromaDB error connecting to %s:%s", host, port)
        raise ConnectionError(
            f"Could not connect to ChromaDB at {host}:{port}: {exc}",
        ) from exc
    except OSError as exc:
        logger.exception("Network error connecting to ChromaDB at %s:%s", host, port)
        raise ConnectionError(
            f"Network failure reaching ChromaDB at {host}:{port}: {exc}",
        ) from exc
