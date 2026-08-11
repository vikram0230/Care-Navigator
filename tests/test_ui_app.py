"""Stage 7 Streamlit app tests via ``streamlit.testing.v1.AppTest`` — no live API/Ollama/Chroma.

AppTest re-execs ``ui/app.py``'s source on every ``.run()``, disconnected from whatever is
cached in ``sys.modules['ui.app']`` — so mocks must target ``ui.api_client.<fn>`` (the module
``ui/app.py`` imports names *from*), not ``ui.app.<fn>``. Verified empirically before writing
these tests: patching ``ui.app.ask_question`` is silently never invoked; patching
``ui.api_client.ask_question`` is.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from ui.api_client import AskResult, IngestEnqueueResult, IngestStatusResult

_APP_PATH = "ui/app.py"


@dataclass
class _FakeUploadedFile:
    _bytes: bytes
    name: str

    def getvalue(self) -> bytes:
        return self._bytes


@pytest.fixture(autouse=True)
def _company_ids_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPANY_IDS", "bcbs,wells_fargo")
    monkeypatch.delenv("STREAMLIT_RAG_API_KEY", raising=False)
    monkeypatch.delenv("STREAMLIT_INGEST_API_KEY", raising=False)


def test_sidebar_tenant_selectbox_populated_from_company_ids() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    assert at.exception == []
    select = at.selectbox(key="company_select")
    assert select.options == ["bcbs", "wells_fargo"]


def test_chat_question_renders_answer_and_citations() -> None:
    at = AppTest.from_file(_APP_PATH)
    fake_result = AskResult(
        ok=True,
        answer="Your deductible is $500.",
        citations=[
            {
                "index": 1,
                "source": "bcbs.pdf",
                "tier": 2,
                "doc_type": "benefits",
                "plan_year": "2025",
                "excerpt": "Deductible: $500",
            },
        ],
        cache_hit=False,
    )
    with patch("ui.api_client.ask_question", return_value=fake_result) as mock_ask:
        at.run()
        at.chat_input[0].set_value("What is my deductible?")
        at.run()

    assert at.exception == []
    mock_ask.assert_called_once()
    assert len(at.chat_message) == 2  # one user turn, one assistant turn
    markdown_texts = [m.value for cm in at.chat_message for m in cm.markdown]
    assert any("What is my deductible?" in t for t in markdown_texts)
    # answer is rendered with $ escaped (see _md_safe) so Streamlit doesn't treat it as LaTeX
    assert any("Your deductible is \\$500." in t for t in markdown_texts)
    assert len(at.expander) == 1  # citations expander rendered


def test_dollar_amounts_in_answer_are_escaped_not_latex() -> None:
    at = AppTest.from_file(_APP_PATH)
    fake_result = AskResult(
        ok=True,
        answer="The out-of-pocket limit is $2,000 and the family limit is $4,000.",
        citations=[],
        cache_hit=False,
    )
    with patch("ui.api_client.ask_question", return_value=fake_result):
        at.run()
        at.chat_input[0].set_value("What is my out-of-pocket max?")
        at.run()

    assert at.exception == []
    markdown_texts = [m.value for cm in at.chat_message for m in cm.markdown]
    answer_text = next(t for t in markdown_texts if "out-of-pocket limit" in t)
    # every literal $ must be backslash-escaped so no paired $...$ math span survives
    assert "\\$2,000" in answer_text and "\\$4,000" in answer_text
    assert "$" not in answer_text.replace("\\$", "")


def test_second_question_receives_first_turn_in_history() -> None:
    at = AppTest.from_file(_APP_PATH)
    first_result = AskResult(ok=True, answer="Your deductible is $500.", citations=[], cache_hit=False)
    second_result = AskResult(ok=True, answer="Your copay is $20.", citations=[], cache_hit=False)

    with patch("ui.api_client.ask_question", side_effect=[first_result, second_result]) as mock_ask:
        at.run()
        at.chat_input[0].set_value("What is my deductible?")
        at.run()
        at.chat_input[0].set_value("And my copay?")
        at.run()

    assert at.exception == []
    assert mock_ask.call_count == 2
    second_call_kwargs = mock_ask.call_args_list[1].kwargs
    history = second_call_kwargs["conversation_history"]
    assert history == [{"question": "What is my deductible?", "answer": "Your deductible is $500."}]


def test_ask_question_error_renders_error_and_does_not_add_history() -> None:
    at = AppTest.from_file(_APP_PATH)
    error_result = AskResult(ok=False, error="502: Vector store error")

    with patch("ui.api_client.ask_question", return_value=error_result):
        at.run()
        at.chat_input[0].set_value("What is my deductible?")
        at.run()

    assert at.exception == []
    assert len(at.chat_message) == 0
    assert len(at.error) == 1
    assert "502" in at.error[0].value


def test_clear_conversation_resets_history() -> None:
    at = AppTest.from_file(_APP_PATH)
    fake_result = AskResult(ok=True, answer="Your deductible is $500.", citations=[], cache_hit=False)
    with patch("ui.api_client.ask_question", return_value=fake_result):
        at.run()
        at.chat_input[0].set_value("What is my deductible?")
        at.run()
    assert len(at.session_state["chat_history"]) == 1

    at.button(key="clear_conversation_button").click().run()
    assert at.session_state["chat_history"] == []
    assert len(at.chat_message) == 0


def test_async_mode_polls_status_then_renders_answer() -> None:
    """When ask_question returns a pending 202, the app polls get_rag_status and renders SUCCESS."""
    at = AppTest.from_file(_APP_PATH)
    pending = AskResult(ok=True, pending=True, task_id="rag-1")
    done = AskResult(ok=True, answer="Your deductible is $500.", citations=[], cache_hit=False)
    with patch("ui.api_client.ask_question", return_value=pending), \
        patch("ui.api_client.get_rag_status", return_value=done) as mock_status:
        at.run()
        at.chat_input[0].set_value("What is my deductible?")
        at.run()

    assert at.exception == []
    mock_status.assert_called()  # polled the status endpoint
    markdown_texts = [m.value for cm in at.chat_message for m in cm.markdown]
    assert any("Your deductible is \\$500." in t for t in markdown_texts)


def test_ask_with_progress_runs_offthread_and_returns_result() -> None:
    import ui.app as app

    updates: list = []

    class _FakeArea:
        def markdown(self, text: str, **_: object) -> None:
            updates.append(text)

    result = AskResult(ok=True, answer="hello", citations=[], cache_hit=False)
    settings = {
        "api_base": "http://api:8000",
        "company_id": "bcbs",
        "filter_doc_types": None,
        "filter_plan_years": None,
    }
    with patch("ui.app.ask_question", return_value=result) as mock_ask:
        out = app._ask_with_progress(_FakeArea(), settings, "What is my deductible?", [])

    assert out is result
    mock_ask.assert_called_once()
    kwargs = mock_ask.call_args.kwargs
    assert kwargs["company_id"] == "bcbs"
    assert kwargs["conversation_history"] == []


def test_ingest_upload_calls_api_client_with_expected_args() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    at.radio(key="tier_radio").set_value(2)
    at.run()
    at.selectbox(key="ingest_company_select").set_value("wells_fargo")
    at.text_input(key="plan_year_input").set_value("2026")
    at.run()

    # A widget's session_state value must be re-set immediately before the run that reads it —
    # it does not persist across an intervening .run() the way a real browser session would.
    at.session_state["pdf_uploader"] = _FakeUploadedFile(b"%PDF-1.4 fake", "benefits.pdf")
    enqueue_result = IngestEnqueueResult(ok=True, task_id="task-123")
    with patch("ui.api_client.upload_pdf", return_value=enqueue_result) as mock_upload:
        at.button(key="upload_button").click().run()

    assert at.exception == []
    mock_upload.assert_called_once()
    kwargs = mock_upload.call_args.kwargs
    assert kwargs["filename"] == "benefits.pdf"
    assert kwargs["company_id"] == "wells_fargo"
    assert kwargs["tier"] == 2
    assert kwargs["plan_year"] == "2026"
    assert at.session_state["last_ingest_task_id"] == "task-123"


def test_ingest_upload_without_file_shows_warning() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    with patch("ui.api_client.upload_pdf") as mock_upload:
        at.button(key="upload_button").click().run()
    mock_upload.assert_not_called()
    assert len(at.warning) == 1


def test_ingest_reseed_calls_trigger_reseed_with_flags() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    at.checkbox(key="reseed_reset_checkbox").set_value(True)
    at.checkbox(key="reseed_full_checkbox").set_value(False)
    at.run()

    enqueue_result = IngestEnqueueResult(ok=True, task_id="reseed-456")
    with patch("ui.api_client.trigger_reseed", return_value=enqueue_result) as mock_reseed:
        at.button(key="reseed_button").click().run()

    assert at.exception == []
    mock_reseed.assert_called_once()
    kwargs = mock_reseed.call_args.kwargs
    assert kwargs["reset"] is True
    assert kwargs["full"] is False
    assert at.session_state["last_ingest_task_id"] == "reseed-456"


def test_ingest_check_status_renders_state() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    at.text_input(key="task_id_input").set_value("task-789")
    at.run()

    status_result = IngestStatusResult(ok=True, state="SUCCESS", result={"chunks": 5}, error=None)
    with patch("ui.api_client.get_ingest_status", return_value=status_result) as mock_status:
        at.button(key="check_status_button").click().run()

    assert at.exception == []
    mock_status.assert_called_once()
    assert mock_status.call_args.args[1] == "task-789"
    markdown_texts = [m.value for m in at.markdown]
    assert any("SUCCESS" in t for t in markdown_texts)


def test_ingest_check_status_without_task_id_shows_warning() -> None:
    at = AppTest.from_file(_APP_PATH)
    at.run()
    with patch("ui.api_client.get_ingest_status") as mock_status:
        at.button(key="check_status_button").click().run()
    mock_status.assert_not_called()
    assert len(at.warning) == 1
