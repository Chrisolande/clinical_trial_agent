from __future__ import annotations

import asyncpg
from config import settings
from loguru import logger


class PostgresBase:
    """Shared connection pool lifecycle for all PostgreSQL-backed stores."""

    def __init__(
        self,
        dsn: str = settings.memory_db_dsn,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: asyncpg.Pool | None = None

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
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE {timestamp_col} <= $1",
                now,  # nosec B608
            )
        removed = int(result.split()[-1])
        if removed:
            logger.info(
                "Purged expired entries",
                table=table,
                count=removed,
            )
        return removed
