"""Pure ranking-quality metrics for RAG retrieval evaluation.

All functions operate on a *ranked* list of relevance flags (``rel``), index 0 = top hit.
They have no I/O and no dependency on Chroma/Ollama so they are unit-testable in isolation.

Relevance labeling (see ``eval/run_eval.py``) is **source-level**: a retrieved chunk counts as
relevant when its ``source`` metadata (the PDF filename) is one of the gold item's
``expected_sources``. Source filenames are stable across re-seeding; chunk IDs are random UUIDs
minted at ingest, so they cannot be used as labels.

Denominator conventions, stated explicitly because they are easy to get subtly wrong:

- ``precision_at_k``   — hits in top-k divided by ``min(k, len(rel))``.
- ``recall_at_k``      — hits in top-k divided by ``total_relevant``. When ``total_relevant`` is
  omitted it defaults to ``sum(rel)`` == the number of relevant chunks in the retrieved candidate
  window (tier1_top_k + tier2_top_k). This is **recall within the retrieved candidate set**, not
  corpus-level recall — corpus recall would require exhaustively labeling every chunk of every PDF.
- ``ndcg_at_k``        — binary gains; IDCG uses ``min(k, total_relevant)`` ideal relevant slots.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence


def relevance_flags(ranked_sources: Sequence[str], relevant_sources: Iterable[str]) -> List[bool]:
    """Map a ranked list of citation ``source`` values to relevance booleans."""
    relevant = set(relevant_sources)
    return [src in relevant for src in ranked_sources]


def hit_rate_at_k(rel: Sequence[bool], k: int) -> float:
    """1.0 if at least one relevant item appears in the top ``k``, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(rel[:k]) else 0.0


def precision_at_k(rel: Sequence[bool], k: int) -> float:
    """Fraction of the top ``k`` retrieved items that are relevant (denominator ``min(k, len)``)."""
    if k <= 0:
        return 0.0
    top = rel[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r) / len(top)


def recall_at_k(rel: Sequence[bool], k: int, total_relevant: Optional[int] = None) -> float:
    """Fraction of relevant items found within the top ``k``.

    ``total_relevant`` defaults to ``sum(rel)`` (recall within the retrieved candidate window).
    """
    if k <= 0:
        return 0.0
    total = total_relevant if total_relevant is not None else sum(1 for r in rel if r)
    if total <= 0:
        return 0.0
    hits = sum(1 for r in rel[:k] if r)
    return hits / total


def mrr(rel: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none present)."""
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def dcg_at_k(rel: Sequence[bool], k: int) -> float:
    """Discounted cumulative gain over the top ``k`` with binary gains."""
    if k <= 0:
        return 0.0
    return sum((1.0 if r else 0.0) / math.log2(i + 1) for i, r in enumerate(rel[:k], start=1))


def ndcg_at_k(rel: Sequence[bool], k: int, total_relevant: Optional[int] = None) -> float:
    """Normalized DCG at ``k``: ``dcg / idcg`` with an ideal ranking of all relevant items first."""
    if k <= 0:
        return 0.0
    dcg = dcg_at_k(rel, k)
    total = total_relevant if total_relevant is not None else sum(1 for r in rel if r)
    ideal_hits = min(k, total)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
