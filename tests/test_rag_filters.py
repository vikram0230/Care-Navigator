"""Chroma metadata filter construction for RAG (Stage 4)."""

from api.rag_filters import chroma_where_for_rag_filters


def test_chroma_where_none_when_no_filters() -> None:
    assert chroma_where_for_rag_filters(None, None) is None
    assert chroma_where_for_rag_filters([], []) is None


def test_chroma_where_single_doc_type() -> None:
    assert chroma_where_for_rag_filters(["benefits"], None) == {"doc_type": "benefits"}


def test_chroma_where_multiple_doc_types() -> None:
    w = chroma_where_for_rag_filters(["benefits", "formulary"], None)
    assert w == {"doc_type": {"$in": ["benefits", "formulary"]}}


def test_chroma_where_combined_and() -> None:
    w = chroma_where_for_rag_filters(["benefits"], ["2025"])
    assert w == {"$and": [{"doc_type": "benefits"}, {"plan_year": "2025"}]}
