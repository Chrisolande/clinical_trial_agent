from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from memory import EpisodicMemory, _serialize_encrypted_json


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


@patch("memory.get_settings")
async def test_episodic_memory_init_requires_encryption_key(
    mock_settings: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_mock = MagicMock()
    settings_mock.memory_db_dsn = "postgresql://localhost/test"
    settings_mock.memory_ttl_days = 30
    mock_settings.return_value = settings_mock

    monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("PROFILE_HASH_SALT", "salt")

    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY must be set"):
        EpisodicMemory()


async def test_episodic_memory_init_success(mock_env: None) -> None:
    mem = EpisodicMemory()
    assert mem._ttl_days > 0
    assert mem._fernet is not None


@patch("asyncpg.create_pool")
async def test_episodic_memory_init_and_close(
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

    async with EpisodicMemory() as mem:
        assert mem._pool is not None

    mock_create_pool.assert_called_once()
    mock_asyncpg_pool.close.assert_called_once()


@patch("asyncpg.create_pool")
async def test_lookup_cache_miss(
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
    mock_connection.fetchrow.side_effect = [None, None]

    async with EpisodicMemory() as mem:
        profile = {"age": 50}
        result = await mem.lookup(profile)
        assert result is None


@patch("asyncpg.create_pool")
async def test_lookup_cache_hit(
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

    async with EpisodicMemory() as mem:
        profile = {"age": 50}
        result_data = {"trials": ["NCT123"]}
        encrypted = _serialize_encrypted_json(result_data, mem._fernet)
        mock_connection.fetchrow.reset_mock()
        mock_connection.fetchrow.return_value = {"result_json": encrypted}

        result = await mem.lookup(profile)
        assert result == result_data


@patch("asyncpg.create_pool")
async def test_lookup_corrupted_data_returns_none(
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
    mock_connection.fetchrow.side_effect = [None, {"result_json": "corrupted-data"}]

    async with EpisodicMemory() as mem:
        profile = {"age": 50}
        result = await mem.lookup(profile)
        assert result is None


@patch("asyncpg.create_pool")
async def test_store(
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
        profile = {"age": 50}
        result = {"trials": ["NCT123"]}
        await mem.store(profile, result)

        assert mock_connection.execute.call_count >= 1
        call_args = mock_connection.execute.call_args
        assert "INSERT INTO patient_runs" in call_args[0][0]
