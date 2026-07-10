"""Chroma ``where`` metadata filters for RAG retrieval (Stage 4)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def chroma_where_for_rag_filters(
    filter_doc_types: Optional[List[str]],
    filter_plan_years: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """Build a Chroma metadata ``where`` clause from optional doc_type / plan_year filters.

    Returns ``None`` when no filters are set (full collection query).
    """
    clauses: List[Dict[str, Any]] = []

    if filter_doc_types:
        vals = [v.strip() for v in filter_doc_types if v and str(v).strip()]
        if vals:
            if len(vals) == 1:
                clauses.append({"doc_type": vals[0]})
            else:
                clauses.append({"doc_type": {"$in": vals}})

    if filter_plan_years:
        vals_y = [v.strip() for v in filter_plan_years if v and str(v).strip()]
        if vals_y:
            if len(vals_y) == 1:
                clauses.append({"plan_year": vals_y[0]})
            else:
                clauses.append({"plan_year": {"$in": vals_y}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
