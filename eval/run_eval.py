#!/usr/bin/env python3
"""Run the retrieval evaluation against the live stack and write a metrics report.

Reconstructs production retrieval exactly by reusing ``api.services.rag`` internals
(``_chroma_query`` / ``_merge_hits``) — same top-k, same distance merge, same metadata
filters — but stops before the (slow, local) LLM generation, since retrieval metrics only
need the ranked chunk sources. Generation-side quality (faithfulness / correctness) is a
separate LLM-as-judge pass, deliberately out of scope here.

Guardrail refusal (out_of_scope / injection items) DOES need the model's answer, so it is
opt-in via ``--guardrails``, which calls the live ``POST /rag/query`` for those items only.

Usage (from repo root, with the Compose stack up and the host venv):

    .venv/bin/python eval/run_eval.py                     # retrieval + tenancy + latency
    .venv/bin/python eval/run_eval.py --guardrails        # also check refusals (slow, hits LLM)
    .venv/bin/python eval/run_eval.py --chroma-port 8001  # default already 8001 on the host
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.config import get_settings  # noqa: E402
from api.llm_client import get_embedding_model  # noqa: E402
from api.rag_filters import chroma_where_for_rag_filters  # noqa: E402
from api.services.rag import _chroma_query, _merge_hits  # noqa: E402
from vectordb.chroma_client import get_chroma_client  # noqa: E402
from vectordb.collections import get_collection  # noqa: E402

from eval.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_flags,
)

K_VALUES = [1, 3, 5, 10]
SCORED_CATEGORIES = {"normal", "filter"}
GUARDRAIL_CATEGORIES = {"out_of_scope", "injection"}

# Phrases that indicate the assistant correctly declined / deferred to HR.
# NOTE: this list is inherently brittle — a model can refuse in wording it doesn't contain,
# producing false negatives (an actual refusal scored as not-refused). That fragility is the
# core reason the deferred LLM-as-judge pass exists; treat this heuristic as a first-pass signal
# to be eyeballed against the answer column, not a ground-truth refusal classifier.
_DECLINE_MARKERS = [
    "contact hr",
    "human resources",
    "benefits administrator",
    "do not have enough information",
    "don't have enough information",
    "not have enough information",
    "cannot help",
    "can't help",
    "unable to",
    "i can't",
    "i cannot",
    "can't fulfill",
    "cannot fulfill",
    "can't provide",
    "cannot provide",
    "i must decline",
    "decline to",
    "not about benefits",
    "outside", "unrelated",
]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _retrieve(settings, emb, chroma_client, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mirror api.services.rag.run_rag_query retrieval (pre-LLM) for one gold item."""
    cid = item["company_id"]
    where = chroma_where_for_rag_filters(item.get("filter_doc_types"), item.get("filter_plan_years"))
    qvec = list(emb.embed_query(item["question"]))
    coll_t1 = get_collection(chroma_client, "global", 1, settings=settings)
    coll_t2 = get_collection(chroma_client, cid, 2, settings=settings)
    hits_t1 = _chroma_query(coll_t1, qvec, settings.RAG_TIER1_TOP_K, tier=1, where=where)
    hits_t2 = _chroma_query(coll_t2, qvec, settings.RAG_TIER2_TOP_K, tier=2, where=where)
    merged = _merge_hits(hits_t1, hits_t2, settings.RAG_MAX_CONTEXT_CHUNKS)
    return [
        {
            "source": h.metadata.get("source"),
            "doc_type": h.metadata.get("doc_type"),
            "company_id": h.metadata.get("company_id"),
            "tier": h.tier,
            "distance": h.distance,
        }
        for h in merged
    ]


def _check_refusal(settings, api_base: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Call the live API and heuristically decide whether the answer declined the request."""
    import httpx

    headers = {"Content-Type": "application/json"}
    keys = sorted(settings.rag_api_key_set)
    if keys:
        headers["X-API-Key"] = keys[0]
    payload = {"question": item["question"], "company_id": item["company_id"]}
    try:
        resp = httpx.post(f"{api_base}/rag/query", json=payload, headers=headers, timeout=300.0)
        resp.raise_for_status()
        answer = resp.json().get("answer", "")
    except Exception as exc:  # pragma: no cover - network path
        return {"refused": None, "error": str(exc), "answer": ""}
    lower = answer.lower()
    refused = any(marker in lower for marker in _DECLINE_MARKERS)
    return {"refused": refused, "error": None, "answer": answer}


def evaluate(gold: List[Dict[str, Any]], settings, api_base: str, do_guardrails: bool) -> Dict[str, Any]:
    emb = get_embedding_model(settings)
    chroma_client = get_chroma_client(settings)

    per_item: List[Dict[str, Any]] = []
    latencies: List[float] = []
    leaked_citations = 0
    total_citations = 0

    for item in gold:
        cid = item["company_id"]
        category = item["category"]
        expected = set(item.get("expected_sources") or [])

        t0 = time.perf_counter()
        hits = _retrieve(settings, emb, chroma_client, item)
        latencies.append(time.perf_counter() - t0)

        ranked_sources = [h["source"] for h in hits]
        rel = relevance_flags(ranked_sources, expected)

        # tenant isolation: nothing outside {global, cid} may appear
        allowed = {"global", cid}
        leaks = [h for h in hits if h["company_id"] not in allowed]
        leaked_citations += len(leaks)
        total_citations += len(hits)

        record: Dict[str, Any] = {
            "id": item["id"],
            "category": category,
            "company_id": cid,
            "n_hits": len(hits),
            "top_source": ranked_sources[0] if ranked_sources else None,
            "leaked": len(leaks),
        }

        if category in SCORED_CATEGORIES and expected:
            record["mrr"] = mrr(rel)
            for k in K_VALUES:
                record[f"hit@{k}"] = hit_rate_at_k(rel, k)
                record[f"precision@{k}"] = precision_at_k(rel, k)
                record[f"recall@{k}"] = recall_at_k(rel, k)
                record[f"ndcg@{k}"] = ndcg_at_k(rel, k)

        # filter correctness: every citation must match the requested doc_type
        if category == "filter" and item.get("expected_doc_type"):
            want = item["expected_doc_type"]
            record["filter_ok"] = all(h["doc_type"] == want for h in hits) if hits else False

        per_item.append(record)

    # guardrail pass (opt-in; hits the live LLM)
    guardrail_results: List[Dict[str, Any]] = []
    if do_guardrails:
        for item in gold:
            if item["category"] not in GUARDRAIL_CATEGORIES:
                continue
            res = _check_refusal(settings, api_base, item)
            guardrail_results.append({"id": item["id"], "category": item["category"], **res})

    return {
        "per_item": per_item,
        "latencies": latencies,
        "leaked_citations": leaked_citations,
        "total_citations": total_citations,
        "guardrails": guardrail_results,
    }


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate(results: Dict[str, Any]) -> Dict[str, Any]:
    scored = [r for r in results["per_item"] if "mrr" in r]
    by_metric: Dict[str, float] = {"mrr": _mean([r["mrr"] for r in scored])}
    for k in K_VALUES:
        for m in ("hit", "precision", "recall", "ndcg"):
            key = f"{m}@{k}"
            by_metric[key] = _mean([r[key] for r in scored])

    # per category (normal vs filter) and per tier
    categories: Dict[str, Dict[str, float]] = {}
    for cat in sorted({r["category"] for r in scored}):
        subset = [r for r in scored if r["category"] == cat]
        categories[cat] = {
            "n": len(subset),
            "hit@5": _mean([r["hit@5"] for r in subset]),
            "mrr": _mean([r["mrr"] for r in subset]),
            "ndcg@5": _mean([r["ndcg@5"] for r in subset]),
        }

    filter_items = [r for r in results["per_item"] if "filter_ok" in r]
    filter_ok = _mean([1.0 if r["filter_ok"] else 0.0 for r in filter_items]) if filter_items else None

    lat = results["latencies"]
    guardrails = results["guardrails"]
    refused = [g for g in guardrails if g.get("refused") is True]

    return {
        "n_scored": len(scored),
        "overall": by_metric,
        "by_category": categories,
        "filter_ok_rate": filter_ok,
        "leak_rate": (results["leaked_citations"] / results["total_citations"]) if results["total_citations"] else 0.0,
        "leaked_citations": results["leaked_citations"],
        "total_citations": results["total_citations"],
        "latency_p50": _percentile(lat, 50),
        "latency_p95": _percentile(lat, 95),
        "latency_mean": _mean(lat),
        "guardrail_n": len(guardrails),
        "guardrail_refused": len(refused),
    }


def render_report(agg: Dict[str, Any], results: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Care Navigator — Retrieval Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {meta['generated']}")
    lines.append(f"- Gold items: {meta['n_gold']} ({agg['n_scored']} scored: normal + filter)")
    lines.append(f"- Retrieval: tier1_top_k={meta['k1']}, tier2_top_k={meta['k2']}, cap={meta['cap']}")
    lines.append(f"- Embedding model: `{meta['embed_model']}`")
    lines.append("")
    lines.append("Relevance is **source-level** (a chunk counts as relevant when its `source` PDF is in "
                 "the gold `expected_sources`). Recall here is within the retrieved candidate window "
                 "(tier1_top_k + tier2_top_k), not corpus-level. LLM-as-judge (faithfulness/correctness) "
                 "is a separate deferred pass.")
    lines.append("")

    lines.append("## Metric definitions")
    lines.append("")
    lines.append("All five are **ranking** metrics: retrieval returns the top-k chunks in rank order, "
                 "each chunk is labeled relevant (✓) or not (✗) by the gold set, and the metric "
                 "scores that ordering. Values below are the mean over scored items.")
    lines.append("")
    lines.append("| Metric | What it measures | Reach for it when |")
    lines.append("|--------|------------------|-------------------|")
    lines.append("| **Precision@k** | Fraction of the top-k that is relevant. Ignores rank within the cut "
                 "and ignores misses. | Junk in the top slots is costly (limited LLM context / user attention). |")
    lines.append("| **Recall@k** | Fraction of all relevant items found in the top-k. Punishes misses; "
                 "ignores order. *Here: recall within the retrieved candidate window (tier1_k+tier2_k), not "
                 "corpus-level.* | **The RAG-critical one** — if the answer chunk is never retrieved, the LLM cannot answer. |")
    lines.append("| **Hit Rate@k** | 1 if ≥ 1 relevant item is in the top-k, else 0 (per query, then averaged). "
                 "Most forgiving. | One good chunk is enough; coarse “did retrieval whiff?” gate. |")
    lines.append("| **MRR** | Mean of 1/(rank of the first relevant item). Rank 1→1.0, rank 2→0.5, rank 3→0.33…; "
                 "ignores everything after the first hit. | The single best answer must land at/near the top. |")
    lines.append("| **nDCG@k** | Discounted cumulative gain (gain / log₂(rank+1)) normalized by the ideal "
                 "ordering, so 0–1 and comparable across queries. Rewards many relevant items ranked high. | "
                 "One well-rounded headline number for overall ranking quality. |")
    lines.append("")
    lines.append("Notes: Precision and Recall trade off as k grows (recall up, precision down). MRR and Hit "
                 "Rate look only at the *first* relevant item; nDCG weighs the *whole* ordering; Precision/Recall "
                 "ignore order *within* the cut. For RAG, prioritize Recall@k (can the answer be found?) then "
                 "MRR/nDCG (is it ranked high enough to survive the context cap?).")
    lines.append("")

    lines.append("## Retrieval metrics (mean over scored items)")
    lines.append("")
    lines.append(f"**MRR = {agg['overall']['mrr']:.3f}**")
    lines.append("")
    lines.append("| k | Hit Rate@k | Precision@k | Recall@k | nDCG@k |")
    lines.append("|---|-----------|-------------|----------|--------|")
    for k in K_VALUES:
        o = agg["overall"]
        lines.append(f"| {k} | {o[f'hit@{k}']:.3f} | {o[f'precision@{k}']:.3f} | "
                     f"{o[f'recall@{k}']:.3f} | {o[f'ndcg@{k}']:.3f} |")
    lines.append("")

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | n | Hit@5 | MRR | nDCG@5 |")
    lines.append("|----------|---|-------|-----|--------|")
    for cat, c in agg["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {c['hit@5']:.3f} | {c['mrr']:.3f} | {c['ndcg@5']:.3f} |")
    lines.append("")

    lines.append("## Tenant isolation & filters")
    lines.append("")
    lines.append(f"- **Cross-tenant leakage rate: {agg['leak_rate']:.3f}** "
                 f"({agg['leaked_citations']} leaked / {agg['total_citations']} citations) — expect 0.")
    if agg["filter_ok_rate"] is not None:
        lines.append(f"- **Filter correctness: {agg['filter_ok_rate']:.3f}** "
                     "(fraction of filter items where every citation matched the requested `doc_type`).")
    lines.append("")

    lines.append("## Retrieval latency (embed + Chroma, pre-LLM)")
    lines.append("")
    lines.append(f"- p50 = {agg['latency_p50']*1000:.0f} ms · p95 = {agg['latency_p95']*1000:.0f} ms · "
                 f"mean = {agg['latency_mean']*1000:.0f} ms")
    lines.append("- End-to-end latency is dominated by LLM generation, measured separately: "
                 "~49 ms on an exact-match cache hit vs ~133 s cold (local llama3.2).")
    lines.append("")

    if agg["guardrail_n"]:
        lines.append("## Guardrails (out-of-scope + injection)")
        lines.append("")
        lines.append(f"- Refusal rate (decline-phrase heuristic): "
                     f"{agg['guardrail_refused']}/{agg['guardrail_n']} declined.")
        lines.append("")
        lines.append("| id | category | refused? | answer (truncated) |")
        lines.append("|----|----------|----------|--------------------|")
        for g in results["guardrails"]:
            ans = (g.get("answer") or g.get("error") or "").replace("\n", " ")[:80]
            lines.append(f"| {g['id']} | {g['category']} | {g.get('refused')} | {ans} |")
        lines.append("")

    lines.append("## Per-item detail")
    lines.append("")
    lines.append("| id | category | Hit@5 | MRR | nDCG@5 | leaked | top source |")
    lines.append("|----|----------|-------|-----|--------|--------|------------|")
    for r in results["per_item"]:
        h5 = f"{r['hit@5']:.2f}" if "hit@5" in r else "—"
        mr = f"{r['mrr']:.2f}" if "mrr" in r else "—"
        n5 = f"{r['ndcg@5']:.2f}" if "ndcg@5" in r else "—"
        top = (r["top_source"] or "—")[:38]
        lines.append(f"| {r['id']} | {r['category']} | {h5} | {mr} | {n5} | {r['leaked']} | {top} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Care Navigator retrieval evaluation harness.")
    parser.add_argument("--gold", default=str(REPO_ROOT / "eval" / "gold.yaml"))
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--guardrails", action="store_true",
                        help="Also check out_of_scope/injection refusals via the live LLM (slow).")
    parser.add_argument("--outdir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args()

    settings = get_settings().model_copy(update={
        "CHROMA_HOST": args.chroma_host,
        "CHROMA_PORT": args.chroma_port,
        "OLLAMA_BASE_URL": args.ollama_url,
    })

    gold = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8"))
    print(f"Loaded {len(gold)} gold items. Running retrieval (no LLM)...", flush=True)

    results = evaluate(gold, settings, args.api_base, args.guardrails)
    agg = aggregate(results)

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_gold": len(gold),
        "k1": settings.RAG_TIER1_TOP_K,
        "k2": settings.RAG_TIER2_TOP_K,
        "cap": settings.RAG_MAX_CONTEXT_CHUNKS,
        "embed_model": settings.OLLAMA_EMBEDDING_MODEL,
    }
    report = render_report(agg, results, meta)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (outdir / f"{stamp}.json").write_text(
        json.dumps({"meta": meta, "aggregate": agg, "per_item": results["per_item"],
                    "guardrails": results["guardrails"]}, indent=2),
        encoding="utf-8",
    )
    (outdir / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWrote {outdir/'report.md'} and {outdir/(stamp + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
