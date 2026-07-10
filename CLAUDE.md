# Working with Claude on Care Navigator

This file documents the workflow established for building out this project's roadmap stages. Keep it updated whenever the user gives new standing instructions — it should always reflect how work actually proceeds here, not how it used to.

## Stage workflow

Each roadmap stage (see README.md's "Progress and roadmap (stages)" table) follows this sequence:

1. **Explore first.** Read the relevant existing code (config, routes, services, tests) before proposing anything. Reuse existing helpers/patterns instead of rebuilding them.
2. **Plan mode for anything non-trivial.** New stages are multi-file, architecturally significant changes — always enter plan mode. Use `AskUserQuestion` to resolve real scope decisions (e.g. "exact-match cache vs. semantic cache", "reuse RAG_API_KEYS vs. a dedicated key") before writing the plan. Don't ask about approval itself — `ExitPlanMode` handles that.
3. **Write the plan file** with: Context (why this stage, what prompted it), what existing code is being reused (with file paths), concrete design per file, a **Tests** section enumerating test cases up front, and a **Verification** section (fast suite, manual smoke test commands).
4. **After approval: tests first, then implementation.** Write the new/updated test files before or alongside the implementation, then build the implementation to make them pass. Run the fast suite after each meaningful chunk, not just at the end.
5. **Update README.md as part of the same change**, not as an afterthought: flip the stage's status in the roadmap table, update the component summary table, add/adjust a "Future work" note for anything explicitly descoped, and document any new HTTP endpoints with example `curl` calls.
6. **Report results with numbers**, not just "tests pass" — e.g. "37 new tests, 95/95 fast suite passing."
7. **Never commit or push without being asked.** When asked, stage exactly the files belonging to the change (no `git add -A`), write a commit message explaining *why*, and only push if explicitly told to.

## Testing conventions

- Fast suite: `pytest tests/ -m "not integration"`. Integration suite (real Chroma + Ollama): `RUN_CHROMA_INTEGRATION=1 pytest tests/test_multitenancy.py -m integration -q` — slow (~6 min), only run when asked or before a stage is declared fully verified.
- **Known local gotcha:** the developer's own `.env` has `RAG_API_KEYS` set, which breaks any fast test that doesn't explicitly monkeypatch it. Always run the fast suite as `RAG_API_KEYS= pytest tests/ -m "not integration" -q` to avoid false failures from local environment bleed-through.
- No live services in unit tests. Mock Chroma/Redis/Celery/httpx clients with `unittest.mock.MagicMock`/`patch`, matching the existing style in `tests/test_multitenancy.py` (Chroma), `tests/test_rag_service.py` (service-layer mocking), and `tests/test_ingest_tasks.py` (Celery task + webhook mocking). Celery tasks are tested by calling them directly (not via `.delay()`) — a `@celery_app.task`-decorated function is still a plain callable outside a real worker.
- New route/service modules get their own test file named after the module (`api/cache.py` → `tests/test_cache.py`), not folded into an unrelated existing file.
- Project venv is at `.venv/` — activate it (`source .venv/bin/activate`) before running `pytest`/`python` directly; don't assume the system Python has the dependencies.

## Project-specific facts worth remembering

- Settings are `pydantic-settings`, loaded from `.env` (real `Settings()`) in the running app, and from a hermetic `SettingsForTests` (no `.env` file) in most direct unit tests — see `tests/conftest.py`.
- Fail-open is the standing design rule for anything backed by Redis (Stage 5 cache, Stage 6 cache invalidation after ingest): a down or erroring Redis must never turn a working request into a failure. New Redis-touching code should follow the same `try/except redis.exceptions.RedisError: log + no-op` pattern already in `api/cache.py`.
- Auth is header-based, optional-by-default API keys (`Authorization: Bearer <token>` or `X-API-Key`), gated per-capability: `RAG_API_KEYS` (read/query) and `INGEST_API_KEYS` (write/ingest) are intentionally separate key spaces — see `api/auth.py`.
- `docker compose` stack is normally already running locally during development (redis, chromadb, ollama, api, celery-worker, streamlit, prometheus, grafana) with `nomic-embed-text` and `llama3.2` already pulled — check `docker ps` before assuming services need to be started.
- Streamlit's `streamlit.testing.v1.AppTest` re-execs `ui/app.py`'s source on every `.run()`, disconnected from whatever's cached in `sys.modules['ui.app']`. Mocks must target the module a Streamlit script imports names *from* (e.g. `patch("ui.api_client.ask_question", ...)`), not the name as bound in `ui.app` itself — patching `ui.app.<fn>` is silently never invoked. `st.file_uploader` has no AppTest widget proxy; inject a fake uploaded file (an object with `.getvalue()` and `.name`) directly into `at.session_state[<file_uploader's key>]` immediately before the `.run()` that reads it — it does not persist across an intervening `.run()` the way a real browser session would.

## Current state

All 7 roadmap stages are done. See README.md's "Progress and roadmap (stages)" table and "Future work" section for exactly what's shipped vs. explicitly deferred (semantic/fuzzy caching, L2 chunk cache, cache warming, LLM-queue rate limiting, webhook SSRF hardening).
