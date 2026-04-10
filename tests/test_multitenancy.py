"""Multitenancy rules for Chroma collections and Stage 2 seed content."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from api.config import get_settings
from api.gemini_client import get_gemini_embeddings
from vectordb.chroma_client import get_chroma_client
from vectordb.collections import (
    GLOBAL_TIER1,
    collection_name,
    create_collections,
    get_collection,
    list_collections,
)
from tests.conftest import SettingsForTests

# Substrings expected in extracted text from real seed PDFs (heuristic integration checks).
GLOBAL_IMMUNIZATION_HINT = "immunization"
BCBS_PLAN_HINT = "Blue Cross"
WELLS_FARGO_HINT = "Wells Fargo"

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_collection_name_tier1_is_global() -> None:
    """Tier 1 maps to the shared ``global_tier1`` collection."""
    assert collection_name("ignored", 1) == GLOBAL_TIER1


def test_collection_name_tier2_per_company() -> None:
    """Tier 2 maps to ``{company_id}_tier2``."""
    assert collection_name("bcbs", 2) == "bcbs_tier2"
    assert collection_name("wells_fargo", 2) == "wells_fargo_tier2"


def test_collection_name_invalid_tier() -> None:
    """Unsupported tiers raise ``ValueError``."""
    with pytest.raises(ValueError, match="tier"):
        collection_name("bcbs", 3)


def test_collection_name_tier2_requires_company() -> None:
    """Tier 2 rejects blank company ids."""
    with pytest.raises(ValueError, match="company_id"):
        collection_name("   ", 2)


def test_get_collection_rejects_unknown_company() -> None:
    """Tier 2 lookups validate ``company_id`` against ``COMPANY_IDS``."""
    settings = SettingsForTests(
        COMPANY_IDS="bcbs,wells_fargo",
        GEMINI_API_KEY="dummy",
    )
    fake_client = MagicMock()
    with pytest.raises(ValueError, match="Unknown company_id"):
        get_collection(fake_client, "other_corp", 2, settings=settings)
    fake_client.get_collection.assert_not_called()


@pytest.mark.integration
@pytest.fixture(scope="module")
def chroma_seeded_environment() -> Dict[str, Any]:
    """Reset Chroma, run ``seed_documents.py``, and return client helpers.

    Skips unless ``RUN_CHROMA_INTEGRATION`` is truthy and ``GEMINI_API_KEY`` is set.
    """
    if os.environ.get("RUN_CHROMA_INTEGRATION", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        pytest.skip("Set RUN_CHROMA_INTEGRATION=1 to run Chroma integration tests")

    # Resolve secrets from the same source as the app (``.env`` via pydantic-settings),
    # not only ``os.environ`` — developers often keep GEMINI_API_KEY in ``.env`` only.
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.GEMINI_API_KEY.strip():
        pytest.skip(
            "GEMINI_API_KEY is required for embedding; set it in .env or the environment",
        )

    env = os.environ.copy()
    env.setdefault("ENVIRONMENT", "development")
    env["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    env["CHROMA_HOST"] = settings.CHROMA_HOST
    env["CHROMA_PORT"] = str(settings.CHROMA_PORT)
    env["COMPANY_IDS"] = settings.COMPANY_IDS
    # Subprocess must resolve ``api`` / ``vectordb`` like Docker's PYTHONPATH=/app
    _pp = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        _pp = f"{_pp}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = _pp

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "seed_documents.py"),
            "--reset",
            "--full",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"seed_documents.py failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

    client = get_chroma_client(settings)
    embeddings = get_gemini_embeddings(settings)
    return {"client": client, "settings": settings, "embeddings": embeddings}


@pytest.mark.integration
def test_seed_script_exits_zero(chroma_seeded_environment: Dict[str, Any]) -> None:
    """Fixture running ``seed_documents.py`` implies the script completed successfully."""
    assert chroma_seeded_environment["client"] is not None


@pytest.mark.integration
def test_list_collections_includes_tier1_and_tier2(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """``list_collections`` returns global and per-tenant tier-2 names."""
    client = chroma_seeded_environment["client"]
    names = list_collections(client)
    assert GLOBAL_TIER1 in names
    assert "bcbs_tier2" in names
    assert "wells_fargo_tier2" in names


@pytest.mark.integration
def test_global_tier1_retrievable_immunization_content(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """Shared tier-1 chunks include immunization / schedule language from CDC PDF."""
    client = chroma_seeded_environment["client"]
    embeddings = chroma_seeded_environment["embeddings"]
    coll = get_collection(client, "global", 1)
    query_vec = embeddings.embed_query("adult immunization vaccine schedule recommendations")
    result = coll.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["documents"],
    )
    blob = "\n".join(result["documents"][0])
    assert GLOBAL_IMMUNIZATION_HINT.lower() in blob.lower()


@pytest.mark.integration
def test_bcbs_tier2_excludes_wells_fargo_employer_text(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """BCBS tier-2 collection must not contain Wells Fargo employer summary text."""
    client = chroma_seeded_environment["client"]
    embeddings = chroma_seeded_environment["embeddings"]
    bcbs = get_collection(client, "bcbs", 2)
    query_vec = embeddings.embed_query(
        "Wells Fargo regular and fixed term employees benefits summary 2025",
    )
    result = bcbs.query(
        query_embeddings=[query_vec],
        n_results=12,
        include=["documents"],
    )
    blob = "\n".join(result["documents"][0])
    assert WELLS_FARGO_HINT.lower() not in blob.lower()


@pytest.mark.integration
def test_bcbs_tier2_includes_bcbs_content(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """BCBS tier-2 retrieval returns Blue Cross / BCBS plan language."""
    client = chroma_seeded_environment["client"]
    embeddings = chroma_seeded_environment["embeddings"]
    bcbs = get_collection(client, "bcbs", 2)
    query_vec = embeddings.embed_query("Blue Cross medical insurance benefits coverage")
    result = bcbs.query(
        query_embeddings=[query_vec],
        n_results=12,
        include=["documents"],
    )
    blob = "\n".join(result["documents"][0])
    assert BCBS_PLAN_HINT in blob


@pytest.mark.integration
def test_wells_fargo_tier2_contains_employer_branding(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """Wells Fargo tier-2 collection returns employer-branded benefits content."""
    client = chroma_seeded_environment["client"]
    embeddings = chroma_seeded_environment["embeddings"]
    wells = get_collection(client, "wells_fargo", 2)
    query_vec = embeddings.embed_query(
        "employee benefits summary medical dental vision Wells Fargo",
    )
    result = wells.query(
        query_embeddings=[query_vec],
        n_results=12,
        include=["documents"],
    )
    blob = "\n".join(result["documents"][0])
    assert WELLS_FARGO_HINT in blob


@pytest.mark.integration
def test_metadata_filter_formulary_only_bcbs(
    chroma_seeded_environment: Dict[str, Any],
) -> None:
    """Chroma ``where`` filters restrict results to formulary metadata."""
    client = chroma_seeded_environment["client"]
    embeddings = chroma_seeded_environment["embeddings"]
    bcbs = get_collection(client, "bcbs", 2)
    query_vec = embeddings.embed_query("prescription drug list tier copay formulary")
    result = bcbs.query(
        query_embeddings=[query_vec],
        n_results=10,
        where={"doc_type": "formulary"},
        include=["metadatas", "documents"],
    )
    metas = result["metadatas"][0]
    assert metas, "expected at least one formulary hit"
    for meta in metas:
        assert meta is not None
        assert meta.get("doc_type") == "formulary"
