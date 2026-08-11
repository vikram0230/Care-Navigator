"""Streamlit UI (Stage 7): chat with conversation memory, citations, and an ingest panel."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import streamlit as st

from ui.api_client import (
    AskResult,
    ask_question,
    fetch_health,
    get_ingest_status,
    get_rag_status,
    trigger_reseed,
    upload_pdf,
)

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://api:8000")

# How often the progress indicator refreshes the elapsed-time label while waiting on the LLM.
_PROGRESS_POLL_SECONDS = 0.3
# How often to poll GET /rag/status/{task_id} when the API is in async mode (returned a 202).
_STATUS_POLL_SECONDS = 1.0


def _company_ids() -> List[str]:
    raw = os.environ.get("COMPANY_IDS", "bcbs,wells_fargo")
    return [c.strip() for c in raw.split(",") if c.strip()]


def _parse_csv(value: str) -> Optional[List[str]]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts or None


def _rag_api_key() -> Optional[str]:
    return os.environ.get("STREAMLIT_RAG_API_KEY", "") or None


def _ingest_api_key() -> Optional[str]:
    return os.environ.get("STREAMLIT_INGEST_API_KEY", "") or None


def _init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List[Dict[str, Any]]: question, answer, citations, cache_hit
    if "last_ingest_task_id" not in st.session_state:
        st.session_state.last_ingest_task_id = ""


def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Settings")
    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL, key="api_base_input")
    company_id = st.sidebar.selectbox("Tenant (company_id)", options=_company_ids(), key="company_select")
    doc_types_raw = st.sidebar.text_input(
        "Filter doc_type(s), comma-separated", value="", key="filter_doc_types_input",
    )
    plan_years_raw = st.sidebar.text_input(
        "Filter plan_year(s), comma-separated", value="", key="filter_plan_years_input",
    )
    if st.sidebar.button("Check API health", key="health_check_button"):
        health = fetch_health(api_base)
        if health.ok:
            st.sidebar.success(health.message)
        else:
            st.sidebar.error(health.message)
    return {
        "api_base": api_base,
        "company_id": company_id,
        "filter_doc_types": _parse_csv(doc_types_raw),
        "filter_plan_years": _parse_csv(plan_years_raw),
    }


def _md_safe(text: str) -> str:
    """Escape ``$`` so Streamlit's markdown doesn't render dollar amounts as LaTeX math.

    Benefits/formulary answers are full of paired dollar figures (e.g. "$2,000 ... $4,000"),
    which Streamlit otherwise interprets as an inline math span and garbles.
    """
    return (text or "").replace("$", "\\$")


def _render_citations(citations: List[Dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"**[{c.get('index')}]** {c.get('source') or 'unknown source'} "
                f"— tier {c.get('tier')}, doc_type={c.get('doc_type')}, plan_year={c.get('plan_year')}\n\n"
                f"> {_md_safe(c.get('excerpt', ''))}",
            )


def _render_chat_tab(settings: Dict[str, Any]) -> None:
    """Render the conversation so far. The input box itself is pinned app-bottom in ``main``."""
    st.subheader("Ask a benefits question")

    if not st.session_state.chat_history:
        st.info("Ask a question using the box at the bottom of the page.")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(_md_safe(turn["question"]))
        with st.chat_message("assistant"):
            st.markdown(_md_safe(turn["answer"]))
            if turn.get("cache_hit"):
                st.caption("⚡ served from cache")
            _render_citations(turn.get("citations") or [])

    if st.button("Clear conversation", key="clear_conversation_button"):
        st.session_state.chat_history = []
        st.rerun()


def _ask_and_wait(
    settings: Dict[str, Any],
    question: str,
    history_payload: List[Dict[str, str]],
) -> AskResult:
    """Ask the API and, in async mode, poll GET /rag/status until the task reaches a terminal state.

    In synchronous mode ``ask_question`` returns the final answer directly (``pending`` is False)
    and this returns immediately. In async mode it returns a 202 (``pending`` + ``task_id``) and we
    poll until SUCCESS/FAILURE. Either way the caller sees one ``AskResult``.
    """
    result = ask_question(
        settings["api_base"],
        question=question,
        company_id=settings["company_id"],
        filter_doc_types=settings["filter_doc_types"],
        filter_plan_years=settings["filter_plan_years"],
        conversation_history=history_payload,
        rag_api_key=_rag_api_key(),
    )
    if not (result.ok and result.pending and result.task_id):
        return result
    task_id = result.task_id
    while True:
        status_result = get_rag_status(settings["api_base"], task_id, rag_api_key=_rag_api_key())
        if not (status_result.ok and status_result.pending):
            return status_result
        time.sleep(_STATUS_POLL_SECONDS)


def _ask_with_progress(
    progress_area: Any,
    settings: Dict[str, Any],
    question: str,
    history_payload: List[Dict[str, str]],
) -> AskResult:
    """Run ``_ask_and_wait`` on a worker thread, updating ``progress_area`` with elapsed time.

    Streamlit scripts are synchronous, so the request (and, in async mode, its status polling) runs
    off-thread while the main thread repaints a live "Thinking… Ns" label.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_ask_and_wait, settings, question, history_payload)
        start = time.monotonic()
        while not future.done():
            elapsed = int(time.monotonic() - start)
            progress_area.markdown(
                f"🤔 **Thinking… {elapsed}s** &nbsp; "
                "<span style='opacity:0.6'>local models can take 1–2 min; "
                "uncached follow-ups run the full pipeline</span>",
                unsafe_allow_html=True,
            )
            time.sleep(_PROGRESS_POLL_SECONDS)
        return future.result()


def _handle_chat_input(settings: Dict[str, Any]) -> None:
    """Top-level chat input (pinned to the app bottom) plus its progress indicator."""
    question = st.chat_input("Ask about your benefits...")
    if not question:
        return
    history_payload = [
        {"question": t["question"], "answer": t["answer"]} for t in st.session_state.chat_history
    ]
    progress_area = st.empty()
    result = _ask_with_progress(progress_area, settings, question, history_payload)
    progress_area.empty()
    if result.ok:
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.answer or "",
                "citations": result.citations,
                "cache_hit": result.cache_hit,
            },
        )
        st.rerun()
    else:
        st.error(result.error or "Request failed")


def _render_ingest_tab(settings: Dict[str, Any]) -> None:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"], key="pdf_uploader")
    tier = st.radio(
        "Tier",
        options=[1, 2],
        format_func=lambda t: "1 — shared global" if t == 1 else "2 — employer-specific",
        key="tier_radio",
    )
    upload_company_id = settings["company_id"]
    if tier == 2:
        upload_company_id = st.selectbox(
            "Employer (tier 2)", options=_company_ids(), key="ingest_company_select",
        )
    doc_type = st.text_input(
        "doc_type (optional — inferred from filename if blank)", value="", key="doc_type_input",
    )
    plan_year = st.text_input("plan_year", value="2025", key="plan_year_input")

    if st.button("Upload", key="upload_button"):
        if uploaded is None:
            st.warning("Choose a PDF file first.")
        else:
            result = upload_pdf(
                settings["api_base"],
                file_bytes=uploaded.getvalue(),
                filename=uploaded.name,
                company_id=upload_company_id,
                tier=tier,
                doc_type=doc_type or None,
                plan_year=plan_year,
                ingest_api_key=_ingest_api_key(),
            )
            if result.ok:
                st.session_state.last_ingest_task_id = result.task_id or ""
                st.success(f"Queued: task_id={result.task_id}")
            else:
                st.error(result.error or "Upload failed")

    st.divider()
    st.subheader("Re-seed bundled data")
    reset = st.checkbox("Reset (wipe Chroma first — development only)", key="reseed_reset_checkbox")
    full = st.checkbox("Full (all global + tier-2 PDFs)", key="reseed_full_checkbox")
    if st.button("Re-seed", key="reseed_button"):
        result = trigger_reseed(
            settings["api_base"], reset=reset, full=full, ingest_api_key=_ingest_api_key(),
        )
        if result.ok:
            st.session_state.last_ingest_task_id = result.task_id or ""
            st.success(f"Queued: task_id={result.task_id}")
        else:
            st.error(result.error or "Reseed failed")

    st.divider()
    st.subheader("Check task status")
    task_id = st.text_input("Task ID", value=st.session_state.last_ingest_task_id, key="task_id_input")
    if st.button("Check status", key="check_status_button"):
        if not task_id:
            st.warning("Enter a task_id first.")
        else:
            result = get_ingest_status(settings["api_base"], task_id, ingest_api_key=_ingest_api_key())
            if result.ok:
                st.write(f"State: **{result.state}**")
                if result.result:
                    st.json(result.result)
                if result.error:
                    st.error(result.error)
            else:
                st.error(result.error or "Status check failed")


def main() -> None:
    """Render the Stage 7 chat + ingest UI."""
    st.set_page_config(page_title="Care Navigator", page_icon="🩺", layout="wide")
    st.title("Care Navigator")
    st.caption("Multi-tenant benefits Q&A — conversation memory, citations, and async PDF ingest")

    _init_session_state()
    settings = _render_sidebar()

    chat_tab, ingest_tab = st.tabs(["Chat", "Ingest"])
    with chat_tab:
        _render_chat_tab(settings)
    with ingest_tab:
        _render_ingest_tab(settings)

    # Pinned to the app bottom (Claude-Code style): st.chat_input only sticks to the viewport
    # bottom when it is a direct child of the app body, not when nested inside st.tabs.
    _handle_chat_input(settings)


if __name__ == "__main__":
    main()
