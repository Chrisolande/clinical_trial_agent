"""PostgreSQL-backed episodic memory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from config import settings
from loguru import logger
from tools.postgres_base import PostgresBase

_DDL = """

    CREATE TABLE IF NOT EXISTS schema_version (

        version INTEGER PRIMARY KEY

    );

    CREATE TABLE IF NOT EXISTS patient_runs (

        profile_hash  TEXT PRIMARY KEY,

        result_json   JSONB NOT NULL,

        created_at    TIMESTAMPTZ NOT NULL,

        expires_at    TIMESTAMPTZ NOT NULL

    );

    CREATE INDEX IF NOT EXISTS idx_patient_runs_expires_at

        ON patient_runs (expires_at);

"""


_SCHEMA_VERSION = 1


def _patient_hash(patient_profile: dict[str, Any]) -> str:
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)

    return hashlib.sha256(canonical.encode()).hexdigest()


class EpisodicMemory(PostgresBase):
    """PostgreSQL-backed async store for completed patient pipeline runs."""

    def __init__(
        self,
        dsn: str = settings.memory_db_dsn,
        ttl_days: int = settings.memory_ttl_days,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        super().__init__(dsn=dsn, pool_min=pool_min, pool_max=pool_max)
        self._ttl_days = ttl_days

    async def _setup_schema(self, conn: asyncpg.Connection) -> None:
        async with conn.transaction():
            await conn.execute(_DDL)
            row = await conn.fetchrow("SELECT version FROM schema_version LIMIT 1")
            if row is None:
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES ($1)",
                    _SCHEMA_VERSION,
                )

    async def lookup(self, patient_profile: dict[str, Any]) -> dict[str, Any] | None:
        key = _patient_hash(patient_profile)
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT result_json FROM patient_runs
                WHERE profile_hash = $1 AND expires_at > $2
                """,
                key,
                now,
            )
        if row:
            raw = row["result_json"]
            if isinstance(raw, dict):
                logger.info("Episodic memory hit", profile_hash=key[:8])
                return raw
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "Episodic memory payload string was not valid JSON; ignoring cache entry"
                    )
                else:
                    if isinstance(parsed, dict):
                        logger.info("Episodic memory hit", profile_hash=key[:8])
                        return parsed
            logger.warning(
                "Episodic memory payload had unexpected type {}; ignoring cache entry",
                type(raw).__name__,
            )
        logger.debug("Episodic memory miss", profile_hash=key[:8])
        return None

    async def store(
        self,
        patient_profile: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        key = _patient_hash(patient_profile)
        now = datetime.now(UTC)
        expires = now + timedelta(days=self._ttl_days)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_runs
                    (profile_hash, result_json, created_at, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (profile_hash) DO UPDATE SET
                    result_json = EXCLUDED.result_json,
                    created_at  = EXCLUDED.created_at,
                    expires_at  = EXCLUDED.expires_at
                """,
                key,
                json.dumps(result, default=str),
                now,
                expires,
            )
        logger.info("Episodic memory stored", profile_hash=key[:8], ttl_days=self._ttl_days)

    async def purge_expired(self) -> int:
        return await self._purge_expired("patient_runs")

    async def list_runs(self) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, created_at, expires_at
                FROM patient_runs WHERE expires_at > $1
                ORDER BY created_at DESC
                """,
                now,
            )
        return [
            {
                "profile_hash": r["profile_hash"],
                "created_at": r["created_at"].isoformat(),
                "expires_at": r["expires_at"].isoformat(),
            }
            for r in rows
        ]

    async def invalidate(self, patient_profile: dict[str, Any]) -> bool:
        key = _patient_hash(patient_profile)
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute("DELETE FROM patient_runs WHERE profile_hash = $1", key)
        removed = int(result.split()[-1]) > 0
        if removed:
            logger.info("Episodic memory invalidated", profile_hash=key[:8])
        return removed


def get_checkpointer(dsn: str = settings.database_uri) -> Any | None:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        return AsyncPostgresSaver.from_conn_string(dsn)
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed - checkpointing disabled. "
            "Run: pip install langgraph-checkpoint-postgres"
        )
        return None
