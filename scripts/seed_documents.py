#!/usr/bin/env python3
"""Ingest bundled benefits / formulary / guideline PDFs into ChromaDB (multi-tenant seed)."""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

# Running ``python scripts/seed_documents.py`` puts ``scripts/`` on ``sys.path``, not the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(REPO_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

import redis

from api.cache import invalidate_company_answer_cache
from api.config import Settings, get_settings
from api.llm_client import get_embedding_model
from vectordb.chroma_client import get_chroma_client
from vectordb.collections import create_collections
from vectordb.ingestion import PDFIngestionPipeline

logger = logging.getLogger(__name__)

SEED_ROOT = REPO_ROOT / "data" / "seed"

# Default seed mode ingests only this file (fewer embedding API calls; avoids free-tier bursts).
MINIMAL_SEED_RELATIVE_PATH = "global/adult-combined-schedule.pdf"


@dataclass(frozen=True)
class SeedFileSpec:
    """One PDF to ingest under ``data/seed`` with metadata."""

    relative_path: str
    doc_type: str
    plan_year: str


# Tier 1: every ``*.pdf`` under ``data/seed/global/`` (recursively) is ingested into ``global_tier1``.

# Tier 2: employer-specific benefits + formulary PDFs (paths relative to ``data/seed``).
TIER2_SEED_FILES: Mapping[str, Tuple[SeedFileSpec, ...]] = {
    "bcbs": (
        SeedFileSpec("bcbs/BCBS_Benefits.pdf", "benefits", "2025"),
        SeedFileSpec("bcbs/BCBS-drug-list-il-2025.pdf", "formulary", "2025"),
    ),
    "wells_fargo": (
        SeedFileSpec(
            "wells_fargo/wells-fargo-2025-benefits-summary-for-regular-and-fixed-term-employees.pdf",
            "benefits",
            "2025",
        ),
        SeedFileSpec("wells_fargo/CVS_Value_Formulary_OE.pdf", "formulary", "2025"),
    ),
}


def _collect_tier1_global_pdf_jobs(plan_year: str = "2025") -> List[Tuple[Path, str, str]]:
    """Discover all PDFs under ``data/seed/global`` and assign ``doc_type`` from the filename.

    Args:
        plan_year: Metadata ``plan_year`` stored on every global chunk.

    Returns:
        List of ``(absolute_path, doc_type, plan_year)`` sorted by path.

    Raises:
        FileNotFoundError: If the global directory is missing or contains no PDFs.
    """
    global_dir = (SEED_ROOT / "global").resolve()
    if not global_dir.is_dir():
        raise FileNotFoundError(f"Missing tier-1 seed directory: {global_dir}")

    pdfs: List[Path] = []
    for path in global_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path.resolve())

    pdfs = sorted(set(pdfs))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under {global_dir}")

    jobs: List[Tuple[Path, str, str]] = []
    for path in pdfs:
        doc_type = PDFIngestionPipeline._infer_doc_type(path.stem, "guidelines")
        jobs.append((path, doc_type, plan_year))
    return jobs


def _resolve_specs(settings: Settings) -> Tuple[List[Tuple[Path, str, str]], List[Tuple[str, Path, str, str]]]:
    """Build absolute paths for tier-1 and tier-2 specs; validate every file exists."""
    tier1_jobs = _collect_tier1_global_pdf_jobs()

    companies = settings.company_id_list
    if not companies:
        raise ValueError("COMPANY_IDS must list at least one tier-2 company id")

    tier2_jobs: List[Tuple[str, Path, str, str]] = []
    for company_id in companies:
        specs = TIER2_SEED_FILES.get(company_id)
        if specs is None:
            raise ValueError(
                f"No seed PDF manifest for company_id={company_id!r}. "
                f"Supported ids: {sorted(TIER2_SEED_FILES)}. "
                f"Update scripts/seed_documents.py TIER2_SEED_FILES or COMPANY_IDS.",
            )
        for spec in specs:
            path = (SEED_ROOT / spec.relative_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing tier-2 seed PDF for {company_id}: {path}",
                )
            tier2_jobs.append((company_id, path, spec.doc_type, spec.plan_year))

    return tier1_jobs, tier2_jobs


def _resolve_minimal_specs() -> Tuple[List[Tuple[Path, str, str]], List[Tuple[str, Path, str, str]]]:
    """Single tier-1 PDF for lightweight local seeding (rate limits / iteration speed)."""
    path = (SEED_ROOT / MINIMAL_SEED_RELATIVE_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Minimal seed file missing: {path} (expected under data/seed/)",
        )
    doc_type = PDFIngestionPipeline._infer_doc_type(path.stem, "guidelines")
    return [(path, doc_type, "2025")], []


def _invalidate_answer_caches(settings: Settings) -> None:
    """Flush Stage 5 cached answers for every tenant after a (re-)ingest.

    Blanket per-company flush regardless of whether tier 1 or tier 2 changed,
    since tier-1 content is merged into every company's answers. A cache-flush
    failure must never fail the ingest run, so Redis errors are logged and
    swallowed here.
    """
    try:
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        for company_id in settings.company_id_list:
            removed = invalidate_company_answer_cache(client, company_id)
            logger.info("Invalidated %s cached answer(s) for company_id=%s", removed, company_id)
    except redis.exceptions.RedisError:
        logger.warning("Could not reach Redis to invalidate answer caches after ingest", exc_info=True)


def run_ingestion(reset_chroma: bool, *, full: bool) -> Dict[str, int]:
    """Ensure collections exist and ingest PDFs (minimal one-file or full manifest).

    Args:
        reset_chroma: When True and ``ENVIRONMENT=development``, wipe Chroma before ingest.
        full: When True, ingest every global PDF and all tier-2 employer PDFs. When False,
            ingest only ``MINIMAL_SEED_RELATIVE_PATH`` (one global PDF) into ``global_tier1``.

    Returns:
        Mapping of logical label (e.g. ``tier1:adult-combined-schedule.pdf``) to chunk count.
    """
    settings = get_settings()
    if not settings.OLLAMA_BASE_URL.strip():
        raise ValueError("OLLAMA_BASE_URL is required for seed ingestion")
    if not settings.OLLAMA_EMBEDDING_MODEL.strip():
        raise ValueError("OLLAMA_EMBEDDING_MODEL is required for seed ingestion")

    if full:
        tier1_jobs, tier2_jobs = _resolve_specs(settings)
    else:
        tier1_jobs, tier2_jobs = _resolve_minimal_specs()
        logger.info(
            "Minimal seed: only %s (pass --full to ingest all global + employer PDFs)",
            MINIMAL_SEED_RELATIVE_PATH,
        )

    client = get_chroma_client(settings)
    if reset_chroma:
        if settings.ENVIRONMENT != "development":
            raise ValueError(
                "--reset is only allowed when ENVIRONMENT=development "
                "(refuses to wipe production Chroma data).",
            )
        logger.warning("Resetting ChromaDB (all collections will be removed)")
        client.reset()

    create_collections(client, settings)
    embeddings = get_embedding_model(settings)
    pipeline = PDFIngestionPipeline(
        client,
        embeddings,
        embedding_sub_batch_size=settings.EMBEDDING_SUB_BATCH_SIZE,
        embedding_inter_batch_delay_seconds=settings.EMBEDDING_INTER_BATCH_DELAY_SECONDS,
    )

    counts: Dict[str, int] = {}

    for path, doc_type, plan_year in tier1_jobs:
        label = f"tier1:{path.name}"
        n = pipeline.ingest_pdf(
            path,
            company_id="global",
            tier=1,
            doc_type=doc_type,
            plan_year=plan_year,
        )
        counts[label] = n
        logger.info("Ingested tier-1 %s (%s chunks)", path.name, n)

    for company_id, path, doc_type, plan_year in tier2_jobs:
        label = f"{company_id}:{path.name}"
        n = pipeline.ingest_pdf(
            path,
            company_id=company_id,
            tier=2,
            doc_type=doc_type,
            plan_year=plan_year,
        )
        counts[label] = n
        logger.info("Ingested tier-2 %s / %s (%s chunks)", company_id, path.name, n)

    _invalidate_answer_caches(settings)

    return counts


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest PDFs into ChromaDB. By default only one global file is ingested; "
            "use --full for all global PDFs plus employer tier-2 PDFs."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Ingest every PDF under data/seed/global/ and all tier-2 files for COMPANY_IDS. "
            "Default is a single file to reduce embedding API usage."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Call client.reset() before seeding (development only; requires ENVIRONMENT=development).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for this script.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    try:
        run_ingestion(reset_chroma=args.reset, full=args.full)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 3
    except ConnectionError as exc:
        logger.error("%s", exc)
        return 4
    except Exception as exc:
        logger.exception("Seed ingestion failed: %s", exc)
        return 1

    logger.info("Seed ingestion completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
