from __future__ import annotations

from typing import Any

import pytest
from tools import postgres_base


class _AcquireCtx:
    def __init__(self, conn: Any):
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


class _Conn:
    def __init__(self, execute_result: str = "DELETE 2") -> None:
        self.execute_result = execute_result
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return self.execute_result


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self.closed = False

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)

    async def close(self) -> None:
        self.closed = True


class _Store(postgres_base.PostgresBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.setup_called = False

    async def _setup_schema(self, conn: Any) -> None:
        _ = conn
        self.setup_called = True


def test_redact_dsn() -> None:
    assert postgres_base.redact_dsn("postgresql://localhost/db") == "postgresql://localhost/db"
    assert (
        postgres_base.redact_dsn(
            "postgresql://user:example-password@localhost:5432/db"  # pragma: allowlist secret
        )
        == "postgresql://user:***@localhost:5432/db"
    )


@pytest.mark.asyncio
async def test_postgres_base_init_context_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    pool = _Pool(conn)

    async def fake_create_pool(**kwargs: Any) -> _Pool:
        assert kwargs["dsn"] == "postgresql://dsn"
        assert kwargs["min_size"] == 1
        assert kwargs["max_size"] == 3
        return pool

    monkeypatch.setattr(postgres_base.asyncpg, "create_pool", fake_create_pool)

    store = _Store(dsn="postgresql://dsn", pool_min=1, pool_max=3)
    await store.init()
    assert store.setup_called is True

    await store.init()  # no-op when already initialized
    assert store._pool is pool

    async with _Store(dsn="postgresql://dsn", pool_min=1, pool_max=3) as cm_store:
        monkeypatch.setattr(postgres_base.asyncpg, "create_pool", fake_create_pool)
        await cm_store.init()
    assert pool.closed is True

    store._pool = pool
    await store.close()
    assert store._pool is None


def test_pool_or_raise() -> None:
    store = _Store(dsn="postgresql://dsn")
    with pytest.raises(RuntimeError, match="not initialized"):
        store._pool_or_raise()


@pytest.mark.asyncio
async def test_purge_expired_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn(execute_result="DELETE 5")
    pool = _Pool(conn)
    store = _Store(dsn="postgresql://dsn")
    store._pool = pool

    removed = await store._purge_expired("patient_runs")
    assert removed == 5
    assert "DELETE FROM patient_runs" in conn.executed[0][0]

    with pytest.raises(ValueError, match="Unsupported table"):
        await store._purge_expired("unknown")

    with pytest.raises(ValueError, match="Unsupported timestamp column"):
        await store._purge_expired("patient_runs", timestamp_col="bad")

    monkeypatch.setitem(postgres_base._PURGE_QUERIES, "patient_runs", "")
    with pytest.raises(ValueError, match="Unsupported table"):
        await store._purge_expired("patient_runs")
