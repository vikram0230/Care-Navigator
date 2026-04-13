# Care Navigator

Multi-tenant RAG foundation for **health benefits Q&A**: ingest employer and reference PDFs into **ChromaDB**, embed with **Ollama**, and expose a **FastAPI** service with observability and worker hooks for future retrieval-augmented chat.

## Goals

- **Tenant-aware knowledge**: shared global content (tier 1) plus per-employer documents (tier 2), stored in separate Chroma collections.
- **Production-shaped layout**: API, Redis, Celery worker, Streamlit shell, Prometheus, and Grafana in Docker Compose.
- **Local LLM**: **Ollama** for embeddings and chat in Docker Compose, `.env.example`, and application code (Phase C: Google GenAI / Gemini removed).

## Reference architecture (target end-state)

The diagram below is the **portfolio target**: L1/L2 Redis caches, Celery, multi-tenant Chroma (`global_tier1`, `bcbs_tier2`, `wells_fargo_tier2`), a local or hosted LLM, ingestion, cache warming, and observability. The codebase today implements **Stages 1–3** (API shell, Chroma + seed, **`POST /rag/query`**, Ollama); L1/L2, full Streamlit chat, and cache warming remain **planned**.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryTextColor':'#111111','secondaryTextColor':'#1a1a1a','tertiaryTextColor':'#111111','lineColor':'#374151','textColor':'#111111','mainBkg':'#ffffff','nodeBorder':'#374151','clusterBkg':'#f3f4f6','clusterBorder':'#4b5563','titleColor':'#000000','edgeLabelBackground':'#ffffff','nodeTextColor':'#111111'}}}%%
flowchart TD
    User(["Employee\nBCBS or Wells Fargo"])

    User --> UI

    subgraph DC["Docker Compose — local dev or demo K8s"]

        UI["Streamlit UI\nCompany switcher · chat history · cache hit indicator"]

        UI --> API

        API["FastAPI plus LangChain\nRAG pipeline · company_id isolation · conversation memory"]

        API -->|"check L1"| L1
        API -->|"check L2"| L2
        API -->|"enqueue"| BROKER

        subgraph REDIS["Redis"]
            L1["L1 answer cache\nSemantic similarity above 0.95\nTTL 90 to 365 days by doc type"]
            L2["L2 chunk cache\nCached Chroma chunks\nSkips vector search on hit"]
            BROKER["Celery broker\nAsync LLM queue\nRate limit e.g. 100 req per minute"]
        end

        L2 -->|"cache miss retrieve"| CHROMA
        BROKER --> WORKER

        subgraph CHROMA["ChromaDB — multi-tenant collections"]
            T1["Shared medical reference\nPreventive care and screening guides\nSame for every employer"]
            T2A["Blue Cross employer pack\nMedical plan and prescription list"]
            T2B["Wells Fargo employer pack\nBenefits summary and prescription list"]
        end

        WORKER["Celery worker\nLLM and heavy tasks\nRetry on failure"]

        CHROMA -->|"chunks plus context"| LLM
        WORKER --> LLM

        LLM["Ollama (or similar)\nEmbeddings and chat\nProd and HIPAA: see Production and compliance below"]

        LLM -->|"answer"| API

        subgraph INGEST["Ingestion and cache warming"]
            PDF["PDF ingestion\nChunk · embed · store"]
            WARM["Cache warming\nGenerate questions then pre-warm\nBlue-green swap"]
        end

        PDF -->|"embed and store"| CHROMA
        WARM -->|"pre-warm answers"| L1

        subgraph OBS["Observability"]
            PROM["Prometheus\nCache hits · latency\nQueue depth · LLM calls"]
            GRAF["Grafana\nL1 L2 miss rate · p95 latency\nBreakdown by company"]
        end

        PROM --> GRAF

    end

    style DC     fill:#eef2f7,stroke:#1e293b,stroke-width:2px,stroke-dasharray:5 5,color:#0f172a
    style REDIS  fill:#fff7ed,stroke:#9a3412,stroke-width:2px,color:#431407
    style CHROMA fill:#eff6ff,stroke:#1d4ed8,stroke-width:2px,color:#1e3a5f
    style INGEST fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#14532d
    style OBS    fill:#faf5ff,stroke:#7e22ce,stroke-width:2px,color:#3b0764

    style User   fill:#e5e7eb,stroke:#374151,stroke-width:2px,color:#111827
    style UI     fill:#99f6e4,stroke:#0f766e,stroke-width:2px,color:#042f2e
    style API    fill:#99f6e4,stroke:#0f766e,stroke-width:2px,color:#042f2e
    style L1     fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#422006
    style L2     fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#422006
    style BROKER fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#422006
    style T1     fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
    style T2A    fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
    style T2B    fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
    style WORKER fill:#fdba74,stroke:#c2410c,stroke-width:2px,color:#431407
    style LLM    fill:#fdba74,stroke:#c2410c,stroke-width:2px,color:#431407
    style PDF    fill:#bbf7d0,stroke:#166534,stroke-width:2px,color:#14532d
    style WARM   fill:#bbf7d0,stroke:#166534,stroke-width:2px,color:#14532d
    style PROM   fill:#e9d5ff,stroke:#6b21a8,stroke-width:2px,color:#3b0764
    style GRAF   fill:#e9d5ff,stroke:#6b21a8,stroke-width:2px,color:#3b0764
```

> **Implemented today:** FastAPI `/health`, `/metrics`, **`POST /rag/query`** (Stage 3), Compose stack, PDF → chunk → embed → upsert via `scripts/seed_documents.py`, and optional integration tests. RAG reads Chroma (`global_tier1` + employer tier 2) and calls **Ollama**; configure `OLLAMA_*`, Chroma host/port, and seed data for end-to-end use.

### Component summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Streamlit | Company switcher, chat, cache hit indicator (planned beyond Stage 1 shell) |
| API | FastAPI + LangChain | RAG pipeline, tenant isolation, conversation memory |
| L1 cache | Redis (planned; optional GPTCache-style semantic layer) | Semantic answer cache, TTL by document type |
| L2 cache | Redis | Chunk retrieval cache; skip Chroma on hit |
| Queue | Redis + Celery | Async LLM and ingest tasks; rate limits |
| Vector DB | ChromaDB | `global_tier1` plus `bcbs_tier2`, `wells_fargo_tier2` (see `COMPANY_IDS` in config) |
| LLM / embeddings | **Ollama** | `api/llm_client.py` (`OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_CHAT_MODEL`) |
| Ingestion | Python + LangChain + `pypdf` | PDF chunk, embed, store (`scripts/seed_documents.py`, `vectordb/ingestion.py`) |
| Cache warming | Celery (planned) | Pre-warm L1 after upload; blue-green cache swap |
| Monitoring | Prometheus + Grafana | Cache hit rate, latency p95, queue depth |

### Cache check order (planned)

```
Query arrives
    │
    ▼
L1 semantic cache ──── HIT ──► return answer instantly
    │
    MISS
    │
    ▼
L2 chunk cache ──────── HIT ──► skip ChromaDB, call LLM only
    │
    MISS
    │
    ▼
ChromaDB retrieval ──────────► full pipeline → cache result in L1 + L2
```

**Tenants in seed data:** Default `COMPANY_IDS` are `bcbs` and `wells_fargo`. Add employers via `scripts/seed_documents.py` (`TIER2_SEED_FILES`) and `COMPANY_IDS`.

## Production and compliance

- **PHI / HIPAA**: For regulated workloads, prefer **Google Cloud Vertex AI** with appropriate agreements, or **on-cluster** models (e.g. Llama family) so sensitive payloads stay in your boundary. Cloud consumer API keys alone are not a HIPAA program.
- **Collection names:** `global_tier1`, `bcbs_tier2`, `wells_fargo_tier2` for the default seed layout.

## Scaling (planned)

| Area | Direction |
|------|-----------|
| Workers | **KEDA** (or similar) to scale Celery on **Redis queue depth**, not only CPU |
| Ingress | Multi-region active-active (e.g. Route 53 across `us-east-1` / `us-west-2`) when moving beyond Compose; health checks and stateless API tiers |
| Chroma | Shard or partition by `company_id` as tenant count grows (e.g. dedicated instances or StatefulSets per tenant group) |
| Open enrollment | **Blue-green** (or canary) cache swap so new corpora do not cause a thundering herd on cold L1/L2 |

## Future work

- **Stages 4–7** (see roadmap below): auth, metadata filters, Redis L1/L2, Celery ingest/reindex, full Streamlit chat with citations and cache-hit UI.
- Wire **L1/L2** and Celery tasks into FastAPI (stages 5–6).
- Extend **RAG** with optional metadata filters and authenticated tenant resolution (stage 4).
- **Cache warming** after PDF upload or bulk re-ingest; **green → blue** promotion for caches.
- **Grafana**: L1/L2/miss rates, p95 latency, queue depth, **per-company** breakdown (`bcbs` vs `wells_fargo`).
- **L1 implementation**: optional **GPTCache** or custom semantic layer over Redis if plain keys are insufficient; TTLs by doc type.
- **Ingest API**: upload-driven re-ingest behind auth, enqueue to workers instead of blocking the API.
- **Compliance**: data residency, key management, and model hosting choices documented for PHI.

## Tech stack

| Area | Choice |
|------|--------|
| API | FastAPI, Uvicorn, Pydantic Settings |
| LLM / embeddings | **Ollama** (`api/llm_client.py`) |
| Vectors | ChromaDB 0.5.x (HTTP client) |
| Ingestion | LangChain text splitters, `pypdf`, Ollama embeddings |
| Workers | Celery + Redis |
| UI (placeholder) | Streamlit — calls API `/health` only |
| Tests | pytest (unit + optional Chroma/Ollama integration) |

Python **3.12** is expected (see `requirements.txt` / Dockerfile notes).

## Repository layout

| Path | Purpose |
|------|---------|
| `api/` | FastAPI app, config, deps, `llm_client.py` (Ollama), `routes/`, `schemas/`, `services/rag.py` |
| `vectordb/` | Chroma client, collection naming, PDF ingestion pipeline |
| `scripts/seed_documents.py` | CLI to create collections and ingest `data/seed/` PDFs |
| `data/seed/` | Tier-1 global PDFs under `global/`; tier-2 PDFs per employer |
| `workers/` | Celery application |
| `ui/` | Streamlit entrypoint |
| `tests/` | API tests, LLM client tests, multitenancy / integration tests |
| `monitoring/prometheus.yml` | Prometheus scrape config for the API |

## Progress and roadmap (stages)

| Stage | Status | Scope |
|-------|--------|--------|
| **1 — Foundation** | Done | FastAPI `/health`, `/metrics`, CORS, Dockerfiles, Compose (Redis, Chroma, API, Celery, Streamlit shell, Prometheus, Grafana); minimal Celery app |
| **2 — Vector DB and seed** | Done | Chroma HTTP client, collection naming (`global_tier1`, `{company}_tier2`), PDF → chunk → embed → upsert, `seed_documents.py`, embedding sub-batching + inter-batch delay for Ollama, pytest + optional Chroma/Ollama integration tests |
| **3 — RAG API** | Done | `POST /rag/query`: embed question, retrieve from `global_tier1` + `{company_id}_tier2`, merge by distance, answer + citations via **Ollama**; OpenAPI under `/docs` |
| **4 — Tenancy and policy** | Planned | Auth / SSO, metadata filters (e.g. doc_type, plan_year), stronger guardrails; today `company_id` is validated against `COMPANY_IDS` only |
| **5 — Caching** | Planned | Redis: session or conversation state, optional embedding cache, optional answer cache with TTL and invalidation on re-ingest |
| **6 — Async operations** | Planned | Celery tasks: large/batch ingest, reindex, optional webhooks; API enqueues work instead of blocking on big PDFs |
| **7 — Product UI** | Planned | Streamlit: tenant selection, chat thread, rendered citations / sources, aligned with the RAG API (replaces health-only placeholder) |

Embedding ingestion uses **sub-batching** and an optional **inter-batch delay** (`EMBEDDING_SUB_BATCH_SIZE`, `EMBEDDING_INTER_BATCH_DELAY_SECONDS` in `api/config.py`) to avoid hammering local Ollama on large PDFs.

## LLM (Phase C — Ollama only)

The stack uses **Ollama** for embeddings and chat everywhere (ingest, RAG, tests that hit a real model). There is no `LLM_PROVIDER` switch and no Google GenAI dependency.

- **`OLLAMA_BASE_URL`:** `http://ollama:11434` in Compose, `http://localhost:11434` on the host when talking to a local daemon.
- **`OLLAMA_EMBEDDING_MODEL`** / **`OLLAMA_CHAT_MODEL`:** must match pulled models. After the first `docker compose up`, for example:  
  `docker exec -it care-navigator-ollama ollama pull nomic-embed-text && docker exec -it care-navigator-ollama ollama pull llama3.2`  
  The **ollama** service has a healthcheck (`ollama list`); API / worker / Streamlit **wait for it** before starting.

**Important:** Changing the embedding model requires **re-ingesting** Chroma (`scripts/seed_documents.py --reset …`).

## Quick start (Docker Compose)

1. Copy environment template and set Ollama/Chroma URLs if needed:

   ```bash
   cp .env.example .env
   # Pull models after containers are up — see LLM section above.
   ```

2. Start the stack:

   ```bash
   docker compose up --build
   ```

3. Useful URLs (host):

   - API: [http://localhost:8000](http://localhost:8000) — OpenAPI at `/docs`
   - API health: [http://localhost:8000/health](http://localhost:8000/health)
   - Streamlit: [http://localhost:8501](http://localhost:8501)
   - Prometheus: [http://localhost:9090](http://localhost:9090)
   - Grafana: [http://localhost:3000](http://localhost:3000)
   - Chroma (mapped from container 8000): **host port 8001** (see below)

Compose sets `CHROMA_HOST=chromadb` and port `8000` inside the network. On the host, Chroma is published as **8001 → 8000**. Ollama is on **11434**.

## Local Python (without full Compose)

1. Create a venv and install dependencies:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run Redis and Chroma (e.g. `docker compose up redis chromadb -d`).

3. In `.env`, point Chroma at the host-mapped port:

   - `CHROMA_HOST=localhost`
   - `CHROMA_PORT=8001`

4. Run the API:

   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Seeding Chroma

Requires a reachable Chroma instance and **Ollama** URL + embedding model (same variables as the API).

```bash
# Default: one small global PDF (fewer embed calls)
python scripts/seed_documents.py

# Wipe collections in development, then re-seed
python scripts/seed_documents.py --reset

# All PDFs under data/seed/global/ plus tier-2 PDFs for COMPANY_IDS
python scripts/seed_documents.py --full

# With reset + full ingest
python scripts/seed_documents.py --reset --full
```

`--reset` only runs when `ENVIRONMENT=development`. Tier-2 manifests live in `scripts/seed_documents.py` (`TIER2_SEED_FILES`); extend that mapping if you add employers.

## Tests

```bash
# Fast suite (no live Chroma/Ollama)
pytest tests/ -m "not integration"

# Integration: requires Chroma, seed data, and Ollama (reachable from host for embed)
RUN_CHROMA_INTEGRATION=1 pytest tests/test_multitenancy.py -m integration -q
```

## Verifying ingested content

After seeding, you can confirm row counts and sample documents via the Chroma client (`collection.count()`, `collection.get(...)`) or run a `collection.query(...)` using the same embedding model as ingestion. The integration tests in `tests/test_multitenancy.py` embed a query string and assert expected phrases appear in top results.

## Configuration

See [`.env.example`](.env.example) for variables: Ollama URLs and model names, Redis/Celery URLs, Chroma host/port, `COMPANY_IDS`, logging, `EMBEDDING_*` (ingest pacing), and optional RAG limits (`RAG_TIER1_TOP_K`, etc.).

**RAG HTTP:** `POST /rag/query` with JSON `{"question": "...", "company_id": "bcbs"}` (or `wells_fargo`). Responses include `answer` and `citations`. Requires seeded Chroma and Ollama configured (`OLLAMA_*`).

## License / data

Benefits and formulary PDFs under `data/seed/` are for development; ensure you have rights to any proprietary documents you add.
