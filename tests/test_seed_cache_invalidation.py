"""Stage 5: seed_documents.py flushes stale answer caches after ingest."""

from unittest.mock import MagicMock, patch

import redis

from scripts.seed_documents import _invalidate_answer_caches
from tests.conftest import SettingsForTests


def test_invalidate_answer_caches_calls_invalidate_per_company() -> None:
    settings = SettingsForTests(COMPANY_IDS="bcbs,wells_fargo", REDIS_URL="redis://localhost:6379/0")
    fake_client = MagicMock()

    with patch("scripts.seed_documents.redis.Redis.from_url", return_value=fake_client), \
        patch("scripts.seed_documents.invalidate_company_answer_cache") as mock_invalidate:
        _invalidate_answer_caches(settings)

    assert mock_invalidate.call_count == 2
    called_companies = {call.args[1] for call in mock_invalidate.call_args_list}
    assert called_companies == {"bcbs", "wells_fargo"}


def test_invalidate_answer_caches_swallows_connection_errors() -> None:
    settings = SettingsForTests(COMPANY_IDS="bcbs", REDIS_URL="redis://localhost:6379/0")

    with patch(
        "scripts.seed_documents.redis.Redis.from_url",
        side_effect=redis.exceptions.ConnectionError("down"),
    ):
        _invalidate_answer_caches(settings)  # must not raise
