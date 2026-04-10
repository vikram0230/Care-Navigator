"""Multi-tenant Chroma collection naming and lifecycle."""

import logging
from typing import List, Optional

from chromadb import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

from api.config import Settings, get_settings

logger = logging.getLogger(__name__)

GLOBAL_TIER1: str = "global_tier1"


def collection_name(company_id: str, tier: int) -> str:
    """Return the canonical collection name for a tenant tier.

    Tier 1 is shared global content (``global_tier1``). Tier 2 is per-company
    (``{company_id}_tier2``).

    Args:
        company_id: Tenant id (ignored for tier 1; required for tier 2).
        tier: Document tier ``1`` or ``2``.

    Returns:
        Chroma collection name string.

    Raises:
        ValueError: If ``tier`` is not 1 or 2.
    """
    if tier == 1:
        return GLOBAL_TIER1
    if tier == 2:
        if not company_id or not company_id.strip():
            raise ValueError("company_id is required for tier 2 collections")
        return f"{company_id.strip()}_tier2"
    raise ValueError("tier must be 1 or 2")


def create_collections(
    client: ClientAPI,
    settings: Optional[Settings] = None,
) -> List[str]:
    """Ensure global tier-1 and each configured company's tier-2 collections exist.

    Args:
        client: Connected Chroma client.
        settings: Optional settings; defaults to ``get_settings()``.

    Returns:
        List of collection names that were ensured (created or already present).

    Raises:
        ChromaError: If Chroma rejects collection creation.
        ValueError: If ``COMPANY_IDS`` is empty.
    """
    resolved = settings or get_settings()
    companies = resolved.company_id_list
    if not companies:
        raise ValueError("COMPANY_IDS must list at least one company id")

    names: List[str] = [GLOBAL_TIER1]
    names.extend(f"{cid}_tier2" for cid in companies)

    created: List[str] = []
    try:
        for name in names:
            client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            created.append(name)
            logger.debug("Ensured Chroma collection %r", name)
    except ChromaError:
        logger.exception("Failed to create or fetch collection")
        raise
    logger.info("Ensured %d Chroma collections", len(created))
    return created


def get_collection(
    client: ClientAPI,
    company_id: str,
    tier: int,
    settings: Optional[Settings] = None,
) -> Collection:
    """Resolve and return an existing collection for a tenant tier.

    Args:
        client: Connected Chroma client.
        company_id: Tenant id (may be ``global`` for tier-1 metadata context).
        tier: ``1`` (shared global) or ``2`` (company-specific).
        settings: Used to validate tier-2 ``company_id`` against ``COMPANY_IDS``.

    Returns:
        The Chroma ``Collection`` instance.

    Raises:
        ValueError: If tier/company combination is invalid.
        ChromaError: If the collection does not exist.
    """
    resolved = settings or get_settings()
    if tier == 2:
        cid = company_id.strip()
        if cid not in resolved.company_id_list:
            raise ValueError(
                f"Unknown company_id {cid!r}; expected one of {resolved.company_id_list}",
            )
    name = collection_name(company_id, tier)
    try:
        return client.get_collection(name=name)
    except ChromaError:
        logger.warning("Collection %r not found", name)
        raise


def list_collections(client: ClientAPI) -> List[str]:
    """Return sorted collection names visible to this client.

    Args:
        client: Connected Chroma client.

    Returns:
        Alphabetically sorted list of collection names.
    """
    try:
        cols = client.list_collections()
    except ChromaError as exc:
        logger.exception("Failed to list Chroma collections")
        raise
    names = [c.name for c in cols if getattr(c, "name", None)]
    return sorted(names)
