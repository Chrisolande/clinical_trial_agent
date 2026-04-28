from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from clinical_trial_agent.memory import EpisodicMemory


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "test-salt")
    monkeypatch.setenv("DB_ENCRYPTION_KEY", Fernet.generate_key().decode("utf-8"))
    monkeypatch.setenv("DATABASE_URI", "postgresql://user:pass@localhost/test")


@pytest.fixture
def mock_asyncpg_pool() -> AsyncMock:
    pool = AsyncMock()
    pool.acquire = MagicMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_connection() -> AsyncMock:
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetch = AsyncMock()
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock()
    return conn


@patch("asyncpg.create_pool")
async def test_invalidate_and_list_and_purge(
    mock_create_pool: AsyncMock,
    mock_env: None,
    mock_asyncpg_pool: AsyncMock,
    mock_connection: AsyncMock,
) -> None:
    async def _create_pool(*args, **kwargs):
        return mock_asyncpg_pool

    mock_create_pool.side_effect = _create_pool
    mock_asyncpg_pool.acquire.return_value.__aenter__.return_value = mock_connection
    mock_asyncpg_pool.acquire.return_value.__aexit__.return_value = None
    mock_connection.fetchrow.return_value = None

    mock_connection.execute.return_value = "DELETE 1"
    async with EpisodicMemory() as mem:
        removed = await mem.invalidate({"age": 50})
        assert removed is True

    mock_connection.execute.return_value = "DELETE 0"
    async with EpisodicMemory() as mem:
        removed = await mem.invalidate({"age": 50})
        assert removed is False

    now = datetime.now(UTC)
    expires = now + timedelta(days=30)
    mock_connection.fetch.return_value = [
        {"profile_hash": "abc123", "created_at": now, "expires_at": expires}
    ]
    async with EpisodicMemory() as mem:
        runs = await mem.list_runs()
        assert len(runs) == 1

    mock_connection.execute.return_value = "DELETE 5"
    async with EpisodicMemory() as mem:
        count = await mem.purge_expired()
        assert count == 5


@patch("asyncpg.create_pool")
async def test_audit_feedback_and_erase(
    mock_create_pool: AsyncMock,
    mock_env: None,
    mock_asyncpg_pool: AsyncMock,
    mock_connection: AsyncMock,
) -> None:
    async def _create_pool(*args, **kwargs):
        return mock_asyncpg_pool

    mock_create_pool.side_effect = _create_pool
    mock_asyncpg_pool.acquire.return_value.__aenter__.return_value = mock_connection
    mock_asyncpg_pool.acquire.return_value.__aexit__.return_value = None
    mock_connection.fetchrow.return_value = None
    mock_connection.execute.return_value = None

    async with EpisodicMemory() as mem:
        await mem.write_pipeline_audit(
            patient_profile={"age": 50},
            run_id="run-123",
            outcome_tier_counts={"tier1": 5},
            model_version="gpt-4",
            consent_flag=True,
        )
        audit_insert_call = next(
            call
            for call in mock_connection.execute.call_args_list
            if "INSERT INTO pipeline_audit_log" in call.args[0]
        )
        audit_insert_args = audit_insert_call.args
        assert isinstance(audit_insert_args[6], str)
        assert audit_insert_args[6] == '{"tier1": 5}'
        await mem.save_feedback(
            patient_profile={"age": 50},
            run_id="run-123",
            nct_id="NCT123",
            verdict="confirmed",
            note="Good match",
        )
        await mem.erase_profile("abc123")

        assert mock_connection.execute.call_count >= 3

    now = datetime.now(UTC)
    mock_connection.fetch.return_value = [
        {
            "profile_hash": "abc123",
            "run_id": "run-123",
            "timestamp": now,
            "outcome_tier_counts": {"tier1": 5},
            "model_version": "gpt-4",
            "consent_flag": True,
        }
    ]
    async with EpisodicMemory() as mem:
        audits = await mem.list_pipeline_audit("abc123")
        assert audits[0]["run_id"] == "run-123"
        assert audits[0]["outcome_tier_counts"] == {"tier1": 5}

    mock_connection.fetch.return_value = [
        {
            "profile_hash": "abc123",
            "run_id": "run-legacy",
            "timestamp": now,
            "outcome_tier_counts": '{"tier2": 3}',
            "model_version": "gpt-4",
            "consent_flag": True,
        }
    ]
    async with EpisodicMemory() as mem:
        audits = await mem.list_pipeline_audit("abc123")
        assert audits[0]["run_id"] == "run-legacy"
        assert audits[0]["outcome_tier_counts"] == {"tier2": 3}

    mock_connection.fetch.return_value = [
        {
            "profile_hash": "abc123",
            "run_id": "run-123",
            "nct_id": "NCT123",
            "verdict": "confirmed",
            "note": "Good match",
            "created_at": now,
        }
    ]
    async with EpisodicMemory() as mem:
        feedback = await mem.list_feedback("abc123")
        assert feedback[0]["nct_id"] == "NCT123"


def test_memory_feedback_methods_exist() -> None:
    assert hasattr(EpisodicMemory, "save_feedback")
    assert hasattr(EpisodicMemory, "list_feedback")
