"""PDF chunking, embedding, and ChromaDB upsert for multi-tenant RAG."""

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Google free tier counts one ``embed_content`` HTTP call per batch; LangChain issues one call
# per internal batch (up to 100 texts). A single large PDF can therefore trigger 100+ calls in
# seconds if we embed all chunks in one ``embed_documents`` — not “many files”, one document.

from chromadb import ClientAPI
from chromadb.errors import ChromaError
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from vectordb.collections import collection_name

logger = logging.getLogger(__name__)


def _is_embedding_rate_limited(exc: BaseException) -> bool:
    """Return True if the exception indicates Gemini / Google embedding quota or 429."""
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or ("rate" in text and "limit" in text)
    )


def _retry_after_seconds_from_error(exc: BaseException) -> Optional[float]:
    """Parse ``retry in Ns`` from Google error payloads when present."""
    match = re.search(r"retry in ([\d.]+)\s*s", str(exc), flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _string_metadata(meta: Mapping[str, Any]) -> Dict[str, str]:
    """Coerce metadata values to strings for Chroma compatibility."""
    out: Dict[str, str] = {}
    for key, value in meta.items():
        if value is None:
            continue
        out[str(key)] = str(value)
    return out


class PDFIngestionPipeline:
    """Load PDFs, split text, embed with Gemini (or any LangChain embeddings), and store."""

    def __init__(
        self,
        chroma_client: ClientAPI,
        embeddings: Embeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        *,
        embedding_sub_batch_size: int = 50,
        embedding_inter_batch_delay_seconds: float = 0.7,
        embedding_rate_limit_max_retries: int = 12,
    ) -> None:
        """Initialize splitter and retain clients for ingestion.

        Args:
            chroma_client: Connected Chroma HTTP client.
            embeddings: LangChain ``Embeddings`` implementation (e.g. Gemini).
            chunk_size: Target maximum chunk length in characters.
            chunk_overlap: Overlap between consecutive chunks.
            embedding_sub_batch_size: Max texts per ``embed_documents`` call (maps to HTTP batches).
            embedding_inter_batch_delay_seconds: Pause between sub-batches to respect provider RPM.
            embedding_rate_limit_max_retries: Retries when a sub-batch returns 429 / quota errors.
        """
        self._client = chroma_client
        self._embeddings = embeddings
        self._embed_sub_batch_size = max(1, embedding_sub_batch_size)
        self._embed_inter_batch_delay_seconds = max(0.0, embedding_inter_batch_delay_seconds)
        self._embed_rate_limit_max_retries = max(1, embedding_rate_limit_max_retries)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, text: str) -> List[str]:
        """Split raw document text using ``RecursiveCharacterTextSplitter``.

        Args:
            text: Full document body.

        Returns:
            Non-empty list of chunk strings.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text or not str(text).strip():
            raise ValueError("text must be non-empty")
        stripped = str(text).strip()
        chunks = self._splitter.split_text(stripped)
        return chunks if chunks else [stripped]

    def embed_and_store(
        self,
        collection_name_str: str,
        chunks: List[str],
        metadatas: List[Mapping[str, Any]],
    ) -> int:
        """Embed chunks and insert into the given collection.

        Metadata keys ``tier``, ``company_id``, ``doc_type``, ``source``, and
        ``plan_year`` are stored as strings for downstream filtering.

        Args:
            collection_name_str: Target Chroma collection name.
            chunks: Text segments to embed.
            metadatas: One metadata mapping per chunk (same length as ``chunks``).

        Returns:
            Number of vectors written.

        Raises:
            ValueError: On length mismatch or empty input.
            RuntimeError: If the embedding provider fails.
            ChromaError: If Chroma rejects the write.
        """
        if not chunks:
            raise ValueError("chunks must be non-empty")
        if len(metadatas) != len(chunks):
            raise ValueError("metadatas must have the same length as chunks")
        meta_str = [_string_metadata(m) for m in metadatas]
        vectors: List[List[float]] = []
        sub = self._embed_sub_batch_size
        for start in range(0, len(chunks), sub):
            end = min(start + sub, len(chunks))
            batch = chunks[start:end]
            attempt = 0
            while True:
                try:
                    # Google GenAI: ``batch_size=len(batch)`` → one HTTP ``embed_content`` per slice.
                    # Base ``Embeddings`` protocol has no ``batch_size``; fall back if unsupported.
                    embed = self._embeddings.embed_documents
                    try:
                        part = embed(batch, batch_size=len(batch))  # type: ignore[call-arg]
                    except TypeError:
                        part = embed(batch)
                    vectors.extend(part)
                    break
                except Exception as exc:
                    if (
                        _is_embedding_rate_limited(exc)
                        and attempt < self._embed_rate_limit_max_retries
                    ):
                        wait = _retry_after_seconds_from_error(exc)
                        if wait is None:
                            wait = min(60.0, 2.0 ** attempt)
                        logger.warning(
                            "Embedding sub-batch %s-%s rate limited; sleeping %.1fs (attempt %s/%s)",
                            start,
                            end,
                            wait,
                            attempt + 1,
                            self._embed_rate_limit_max_retries,
                        )
                        time.sleep(wait)
                        attempt += 1
                        continue
                    logger.exception("Embedding failed for collection %s", collection_name_str)
                    raise RuntimeError("Embedding request failed") from exc
            if end < len(chunks) and self._embed_inter_batch_delay_seconds > 0:
                time.sleep(self._embed_inter_batch_delay_seconds)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned unexpected vector count")
        ids = [str(uuid.uuid4()) for _ in chunks]
        try:
            coll = self._client.get_collection(collection_name_str)
        except ChromaError:
            logger.exception("Missing collection %s", collection_name_str)
            raise
        try:
            coll.add(
                ids=ids,
                embeddings=vectors,
                documents=chunks,
                metadatas=meta_str,
            )
        except ChromaError:
            logger.exception("Chroma add failed for %s", collection_name_str)
            raise
        return len(chunks)

    def ingest_pdf(
        self,
        pdf_path: Path,
        company_id: str,
        tier: int,
        doc_type: str,
        plan_year: str,
        source_override: Optional[str] = None,
    ) -> int:
        """Ingest a single PDF into the collection implied by ``tier`` / ``company_id``.

        Args:
            pdf_path: Path to the PDF file.
            company_id: ``global`` for tier 1; tenant id for tier 2.
            tier: ``1`` (shared) or ``2`` (tenant).
            doc_type: Logical document type (e.g. ``benefits``, ``formulary``).
            plan_year: Plan or content year label stored in metadata.
            source_override: Optional filename override for metadata ``source``.

        Returns:
            Number of chunks stored.

        Raises:
            FileNotFoundError: If the path is missing.
            ValueError: If the PDF has no extractable text.
        """
        path = pdf_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        text = self._read_pdf_text(path)
        chunks = self.chunk_document(text)
        cid_for_name = company_id.strip() if tier == 2 else "global"
        coll_name = collection_name(cid_for_name, tier)
        meta_company = company_id.strip() if tier == 2 else "global"
        source = source_override or path.name
        metadatas: List[Dict[str, Any]] = [
            {
                "tier": str(tier),
                "company_id": meta_company,
                "doc_type": doc_type,
                "source": source,
                "plan_year": str(plan_year),
            }
            for _ in chunks
        ]
        return self.embed_and_store(coll_name, chunks, metadatas)

    def ingest_directory(
        self,
        directory: Path,
        company_id: str,
        tier: int,
        plan_year: str = "2025",
        default_doc_type: str = "benefits",
    ) -> Dict[str, int]:
        """Recursively ingest every ``*.pdf`` under ``directory``.

        Args:
            directory: Root folder to scan.
            company_id: Passed to ``ingest_pdf`` (``global`` for tier 1).
            tier: ``1`` or ``2``.
            plan_year: Metadata plan year.
            default_doc_type: Used when filename heuristics do not match.

        Returns:
            Map of relative PDF path to chunk count.

        Raises:
            NotADirectoryError: If ``directory`` is not a folder.
        """
        root = directory.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")
        pdf_files = sorted(root.rglob("*.pdf"))
        results: Dict[str, int] = {}
        for pdf in pdf_files:
            doc_type = self._infer_doc_type(pdf.stem, default_doc_type)
            rel = str(pdf.relative_to(root))
            count = self.ingest_pdf(
                pdf,
                company_id=company_id,
                tier=tier,
                doc_type=doc_type,
                plan_year=plan_year,
            )
            results[rel] = count
        return results

    @staticmethod
    def _infer_doc_type(stem: str, default: str) -> str:
        """Guess ``doc_type`` from a filename stem."""
        lower = stem.lower()
        if "formulary" in lower:
            return "formulary"
        if "benefit" in lower:
            return "benefits"
        if "cdc" in lower or "guideline" in lower:
            return "guidelines"
        return default

    @staticmethod
    def _read_pdf_text(path: Path) -> str:
        """Extract joined text from all pages of a PDF."""
        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise ValueError(f"Invalid or unreadable PDF: {path}") from exc
        parts: List[str] = []
        for index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text()
            except Exception as exc:
                logger.warning(
                    "Skipping page %s of %s: %s",
                    index,
                    path,
                    exc,
                )
                continue
            if extracted and extracted.strip():
                parts.append(extracted.strip())
        body = "\n".join(parts)
        if not body.strip():
            raise ValueError(f"No extractable text in PDF: {path}")
        return body
