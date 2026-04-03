from __future__ import annotations

from urllib.parse import urlparse, urlunparse

import asyncpg
from config import get_settings
from loguru import logger

_ALLOWED_TABLES: frozenset[str] = frozenset({"patient_runs", "llm_cache", "pipeline_audit_log"})
_ALLOWED_COLUMNS: frozenset[str] = frozenset({"expires_at", "timestamp"})


def redact_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    if parsed.password is None:
        return dsn
    # netloc = parsed.netloc
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    auth = f"{user}:***@" if user else "***@"
    redacted = parsed._replace(netloc=f"{auth}{host}{port}")
    return urlunparse(redacted)


class PostgresBase:
    """Shared connection pool lifecycle for all PostgreSQL-backed stores."""

    def __init__(
        self,
        dsn: str | None = None,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        self._dsn = dsn or get_settings().memory_db_dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> PostgresBase:
        await self.init()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object | None
    ) -> None:
        _ = (exc_type, exc, tb)
        await self.close()

    async def init(self) -> None:
        """Create connection pool and run schema setup. Call once at startup."""
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        async with self._pool.acquire() as conn:
            await self._setup_schema(conn)
        logger.info(
            f"{self.__class__.__name__} pool ready",
            dsn=self._dsn,
        )

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info(f"{self.__class__.__name__} pool closed")

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call init() first.")
        return self._pool

    async def _setup_schema(self, conn: asyncpg.Connection) -> None:
        """Override in subclass to run CREATE TABLE statements."""
        raise NotImplementedError

    async def _purge_expired(self, table: str, timestamp_col: str = "expires_at") -> int:
        """Generic expired-row purge. Pass the table name."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"Unsupported table for purge_expired: {table}")
        if timestamp_col not in _ALLOWED_COLUMNS:
            raise ValueError(f"Unsupported timestamp column for purge_expired: {timestamp_col}")
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute(f"DELETE FROM {table} WHERE {timestamp_col} <= $1", now)
        removed = int(result.split()[-1])
        if removed:
            logger.info(
                "Purged expired entries",
                table=table,
                count=removed,
            )
        return removed
