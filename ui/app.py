"""Streamlit UI entrypoint (Stage 1 placeholder until Stage 7)."""

import logging
from typing import NoReturn

import httpx
import streamlit as st

logger = logging.getLogger(__name__)

# Default API base inside Docker Compose network
DEFAULT_API_URL = "http://api:8000"


def _fetch_health(api_base: str) -> tuple[bool, str]:
    """Call the FastAPI /health endpoint and return (ok, message)."""
    url = f"{api_base.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            status_label = data.get("status", "unknown")
            return True, f"API status: {status_label}"
    except httpx.HTTPStatusError as exc:
        logger.warning("Health HTTP error: %s", exc)
        return False, f"API returned {exc.response.status_code}"
    except httpx.RequestError as exc:
        logger.warning("Health request failed: %s", exc)
        return False, f"Could not reach API ({exc!s})"
    except ValueError as exc:
        logger.warning("Invalid health JSON: %s", exc)
        return False, "Invalid response from API"


def main() -> None:
    """Render minimal Stage 1 UI with API connectivity check."""
    st.set_page_config(page_title="Care Navigator", page_icon="🩺", layout="centered")
    st.title("Care Navigator")
    st.caption("Stage 1 — foundation (full chat UI arrives in Stage 7)")

    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)
    if st.button("Check API health"):
        ok, message = _fetch_health(api_base)
        if ok:
            st.success(message)
        else:
            st.error(message)


if __name__ == "__main__":
    main()
