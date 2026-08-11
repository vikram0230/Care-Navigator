# Care Navigator — Progress & Roadmap

Living progress tracker. The `README.md` is a formal, current-state overview and intentionally
does **not** carry status/roadmap; that lives here.

## Delivery stages (all shipped)

| Stage | Status | Scope |
|-------|--------|--------|
| **1 — Foundation** | Done | FastAPI `/health`, `/metrics`, CORS, Dockerfiles, Compose (Redis, Chroma, API, Celery, Streamlit shell, Prometheus, Grafana); minimal Celery app |
| **2 — Vector DB and seed** | Done | Chroma HTTP client, collection naming (`global_tier1`, `{company}_tier2`), PDF → chunk → embed → upsert, `seed_documents.py`, embedding sub-batching + inter-batch delay for Ollama, pytest + optional Chroma/Ollama integration tests |
| **3 — RAG API** | Done | `POST /rag/query`: embed question, retrieve from `global_tier1` + `{company_id}_tier2`, merge by distance, answer + citations via Ollama; OpenAPI under `/docs` |
| **4 — Tenancy and policy** | Done | Optional API keys (`RAG_API_KEYS` + `Authorization: Bearer` / `X-API-Key`); Chroma `filter_doc_types` / `filter_plan_years`; question validation + hardened system prompt (benefits scope, prompt-injection resistance); `company_id` validated against `COMPANY_IDS`. SSO/OIDC not implemented — swap the auth helper or add middleware when wiring an IdP. |
| **5 — Caching** | Done (exact-match) | Redis answer cache (`api/cache.py`, keyed by company + normalized question + filters + model names, TTL `RAG_ANSWER_CACHE_TTL_SECONDS`, invalidated per-company at the end of seed ingestion) and embedding cache (keyed by model + question text, TTL `RAG_EMBEDDING_CACHE_TTL_SECONDS`); both fail open if Redis is down. |
| **6 — Async operations** | Done | Celery tasks (`workers/ingest_tasks.py`): `ingest_pdf_task` and `reseed_task`. HTTP surface (`api/routes/ingest.py`, gated by dedicated `INGEST_API_KEYS`): `POST /ingest/upload`, `POST /ingest/reseed`, `GET /ingest/status/{task_id}`. Optional best-effort `webhook_url`. |
| **7 — Product UI** | Done | Streamlit (`ui/app.py` + `ui/api_client.py`): tenant selector, chat thread with citations + cache-hit badge, "Clear conversation"; conversation memory in `POST /rag/query` (`conversation_history`, bounded by `RAG_MAX_CONVERSATION_TURNS`, bypasses the answer cache on follow-ups); Ingest tab over the Stage 6 endpoints. |

Embedding ingestion uses sub-batching + optional inter-batch delay (`EMBEDDING_SUB_BATCH_SIZE`,
`EMBEDDING_INTER_BATCH_DELAY_SECONDS`) to avoid hammering local Ollama on large PDFs.

## Recent work (evaluation + UX iteration)

- **Retrieval evaluation harness (`eval/`)** — hand-curated `gold.yaml` (33 items across
  `normal`/`filter`/`out_of_scope`/`injection`), pure unit-tested metric math
  (`retrieval_metrics.py`: Recall@k, Precision@k, Hit Rate@k, MRR, nDCG@k), and a runner
  (`run_eval.py`) that reconstructs production retrieval (reusing `api.services.rag._chroma_query`
  / `_merge_hits`) and reports cross-tenant leakage, filter correctness, retrieval latency, and an
  optional `--guardrails` refusal pass. Baseline: MRR ≈ 0.98, Hit@5 = 1.00, nDCG@5 ≈ 0.96, leakage
  0/378, filter correctness 1.00, guardrail refusals 6/6.
- **Chat UX** — `st.chat_input` pinned to the app bottom (moved out of the tab); a live
  "Thinking… Ns" progress indicator that runs the request off-thread; HTTP ask-timeout raised
  90 s → 300 s (uncached follow-ups run the full pipeline, measured 130–280 s on local llama3.2);
  `$` escaped in rendered answers/citations so dollar amounts aren't parsed as LaTeX math.

## Planned — next: async LLM queue + polling

`POST /rag/query` still calls Ollama **synchronously** in-request. Because uncached follow-up turns
re-run the full pipeline (~130–280 s locally), the request blocks and only a large client timeout
keeps it alive. The fix mirrors the Stage 6 ingest pattern:

1. `api/schemas/rag.py` — `RagQueryAccepted {task_id, status}` and `RagQueryStatus {task_id, state, result, error}`.
2. `workers/rag_tasks.py` — `rag_query_task` wrapping the existing `run_rag_query` unchanged.
3. `workers/celery_app.py` — register the new task module (`include=[...]`).
4. `api/routes/rag.py` — check the L1 answer cache **synchronously** (instant hits stay `200`); on a
   miss, `rag_query_task.delay(...)` and return `202 {task_id}`; add `GET /rag/status/{task_id}`
   (reuse the ingest `AsyncResult` logic).
5. `api/config.py` — `RAG_ASYNC_ENABLED` (feature flag) + `RAG_LLM_RATE_LIMIT` (Celery `rate_limit`).
6. `ui/api_client.py` + `ui/app.py` — `ask_question` handles 200 (cache) vs 202 (task); new
   `get_rag_status`; the existing `_ask_with_progress` poll loop swaps the thread future for status
   polling — same live indicator.
7. Tests — `tests/test_rag_tasks.py`, extend `tests/test_rag_api.py` (cache→200, miss→202, status
   mapping), `tests/test_ui_client.py` (200 vs 202).

Run the RAG worker `--concurrency=1` (single local Ollama); `rate_limit` provides backpressure; set a
short `result_expires`. A queue makes the wait non-blocking/observable and enables rate limiting,
retries, and fair per-tenant queues — it does not make generation faster. Token streaming (SSE) is
the lighter alternative if the only goal is perceived latency.

## Future work (deferred)

- **Semantic (fuzzy) answer cache** — L1 currently exact-match only; fuzzy/semantic (>0.95 cosine)
  matching needs a vector index over past cached questions + threshold tuning (GPTCache or a custom
  Redis layer).
- **L2 chunk-retrieval cache** — cache Chroma hits directly (skip vector search on hit), distinct
  from the embedding cache which only avoids re-embedding.
- **Cache warming** after upload / bulk re-ingest; green→blue cache promotion (ingest already
  invalidates stale answers; pre-warming replacements is pending).
- **Grafana dashboards** — L1/L2/miss rates, p95 latency, queue depth, per-company breakdown.
- **Webhook hardening** — the ingest `webhook_url` is a single best-effort POST with no SSRF
  validation and no retry/backoff.
- **Evaluation Phase 2 — LLM-as-judge** — faithfulness/groundedness, answer correctness vs the
  `reference_answer` already in `gold.yaml`, answer relevance. Deferred because local llama3.2
  self-judging is too noisy to lead with; revisit judge-model choice then.
- **Compliance** — data residency, key management, model hosting documented for PHI.

## Scaling (planned)

| Area | Direction |
|------|-----------|
| Workers | KEDA (or similar) to scale Celery on Redis queue depth, not only CPU |
| Ingress | Multi-region active-active (e.g. Route 53 across `us-east-1` / `us-west-2`); stateless API tiers |
| Chroma | Shard/partition by `company_id` as tenant count grows |
| Open enrollment | Blue-green (or canary) cache swap so new corpora don't cause a thundering herd on cold caches |
