from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from memory import EpisodicMemory, _patient_hash, get_checkpointer


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "test-salt")
    monkeypatch.setenv("DB_ENCRYPTION_KEY", Fernet.generate_key().decode("utf-8"))
    monkeypatch.setenv("DATABASE_URI", "postgresql://user:pass@localhost/test")


def test_get_checkpointer_success(mock_env: None) -> None:
    mock_saver = MagicMock()
    mock_instance = MagicMock()
    mock_saver.from_conn_string.return_value = mock_instance

    with patch.dict(
        "sys.modules",
        {"langgraph.checkpoint.postgres.aio": MagicMock(AsyncPostgresSaver=mock_saver)},
    ):
        result = get_checkpointer()
        assert result == mock_instance


def test_get_checkpointer_import_error(mock_env: None) -> None:
    import sys

    original_modules = sys.modules.copy()
    if "langgraph.checkpoint.postgres.aio" in sys.modules:
        del sys.modules["langgraph.checkpoint.postgres.aio"]

    try:
        with patch.dict("sys.modules", {"langgraph.checkpoint.postgres.aio": None}):
            result = get_checkpointer()
            assert result is None
    finally:
        sys.modules = original_modules


def test_get_checkpointer_with_custom_dsn(mock_env: None) -> None:
    mock_saver = MagicMock()
    mock_instance = MagicMock()
    mock_saver.from_conn_string.return_value = mock_instance

    with patch.dict(
        "sys.modules",
        {"langgraph.checkpoint.postgres.aio": MagicMock(AsyncPostgresSaver=mock_saver)},
    ):
        result = get_checkpointer(dsn="postgresql://custom/db")
        mock_saver.from_conn_string.assert_called_once_with("postgresql://custom/db")
        assert result == mock_instance


@pytest.mark.asyncio
async def test_memory_insert_lookup_invalidate_and_purge() -> None:
    if os.getenv("RUN_TESTCONTAINERS", "false").lower() != "true":
        pytest.skip("Set RUN_TESTCONTAINERS=true to run containerized DB tests")

    pytest.importorskip("testcontainers.postgres")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        os.environ["DATABASE_URI"] = pg.get_connection_url()
        os.environ["MEMORY_DB_DSN"] = pg.get_connection_url()
        os.environ["PROFILE_HASH_SALT"] = "test-salt"
        os.environ["DB_ENCRYPTION_KEY"] = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="

        memory = EpisodicMemory()
        await memory.init()
        try:
            profile = {"age": 60, "sex": "male", "conditions": ["NSCLC"]}
            payload = {"report_text": "ok", "trial_scores": []}
            await memory.put_run(profile, payload)
            got = await memory.get_run(profile)
            assert got is not None
            assert got.get("report_text") == "ok"
            removed = await memory.invalidate(profile)
            assert removed
            assert await memory.get_run(profile) is None
            purged = await memory.purge_expired()
            assert purged >= 0
        finally:
            await memory.close()


@pytest.mark.asyncio
async def test_patient_hash_requires_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_HASH_SALT", raising=False)
    with pytest.raises(RuntimeError):
        _patient_hash({"age": 50})


@pytest.mark.asyncio
async def test_memory_requires_encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("PROFILE_HASH_SALT", "salt")
    with pytest.raises(RuntimeError):
        EpisodicMemory()
