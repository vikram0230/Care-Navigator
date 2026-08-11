"""Unit tests for eval/retrieval_metrics.py — hand-computed expected values, no live services."""

import math

import pytest

from eval.retrieval_metrics import (
    dcg_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_flags,
)


def test_relevance_flags_maps_sources_to_booleans():
    ranked = ["a.pdf", "b.pdf", "a.pdf", "c.pdf"]
    assert relevance_flags(ranked, {"a.pdf"}) == [True, False, True, False]
    assert relevance_flags(ranked, {"a.pdf", "c.pdf"}) == [True, False, True, True]
    assert relevance_flags(ranked, set()) == [False, False, False, False]


def test_perfect_ranking():
    rel = [True, True, False, False]
    assert precision_at_k(rel, 2) == 1.0
    assert recall_at_k(rel, 2, total_relevant=2) == 1.0
    assert hit_rate_at_k(rel, 2) == 1.0
    assert mrr(rel) == 1.0
    # dcg == idcg for a perfect prefix -> ndcg 1.0
    assert ndcg_at_k(rel, 2) == pytest.approx(1.0)


def test_no_relevant_hits_are_all_zero():
    rel = [False, False, False]
    assert precision_at_k(rel, 3) == 0.0
    assert recall_at_k(rel, 3) == 0.0
    assert hit_rate_at_k(rel, 3) == 0.0
    assert mrr(rel) == 0.0
    assert ndcg_at_k(rel, 3) == 0.0


def test_first_relevant_at_rank_three():
    rel = [False, False, True]
    assert mrr(rel) == pytest.approx(1 / 3)
    assert precision_at_k(rel, 3) == pytest.approx(1 / 3)
    assert hit_rate_at_k(rel, 3) == 1.0
    assert recall_at_k(rel, 3, total_relevant=1) == 1.0
    # dcg@3 = 1/log2(4) = 0.5 ; idcg (1 relevant) = 1/log2(2) = 1.0 ; ndcg = 0.5
    assert ndcg_at_k(rel, 3, total_relevant=1) == pytest.approx(0.5)


def test_partial_ranking_ndcg():
    rel = [True, False, True, False]
    assert precision_at_k(rel, 2) == 0.5
    assert recall_at_k(rel, 2, total_relevant=2) == 0.5
    assert mrr(rel) == 1.0
    # dcg@2 = 1/log2(2) + 0 = 1.0 ; idcg@2 = 1/log2(2)+1/log2(3) = 1 + 0.63093
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(rel, 2, total_relevant=2) == pytest.approx(expected)


def test_k_larger_than_list_uses_available_length():
    rel = [True]
    assert precision_at_k(rel, 5) == 1.0  # denominator is min(k, len) == 1
    assert hit_rate_at_k(rel, 5) == 1.0
    assert mrr(rel) == 1.0
    assert recall_at_k(rel, 5, total_relevant=1) == 1.0


def test_recall_defaults_total_relevant_to_sum_of_flags():
    rel = [True, False, True, False]
    # default total_relevant == sum(rel) == 2 ; top-2 has 1 hit -> 0.5
    assert recall_at_k(rel, 2) == 0.5
    assert recall_at_k(rel, 4) == 1.0


def test_precision_denominator_is_min_k_len_on_short_lists():
    rel = [True, False]
    # top-3 requested but only 2 items -> 1 hit / 2 == 0.5
    assert precision_at_k(rel, 3) == 0.5


def test_zero_and_negative_k_are_zero():
    rel = [True, True]
    for fn in (hit_rate_at_k, precision_at_k, recall_at_k, ndcg_at_k):
        assert fn(rel, 0) == 0.0
        assert fn(rel, -1) == 0.0
    assert dcg_at_k(rel, 0) == 0.0


def test_dcg_matches_manual_computation():
    rel = [True, True, False]
    # 1/log2(2) + 1/log2(3) + 0
    assert dcg_at_k(rel, 3) == pytest.approx(1.0 + 1.0 / math.log2(3))
