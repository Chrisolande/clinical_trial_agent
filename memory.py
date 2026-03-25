from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

_DEFAULT_DSN = os.getenv(
    "MEMORY_DB_DSN",
    "postgresql://localhost:5432/clinical_trial_memory",
)
_DEFAULT_DB = os.getenv("MEMORY_DB_PATH", "/tmp/clinical_trial_memory.db")
_SCHEMA_VERSION = 1
_DEFAULT_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", "30"))

_CREATE_SCHEMA_VERSION = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )
"""

_CREATE_RUNS = """
    CREATE TABLE IF NOT EXISTS patient_runs (
        profile_hash  TEXT PRIMARY KEY,
        result_json   JSONB NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL,
        expires_at    TIMESTAMPTZ NOT NULL
    )
"""

_CREATE_EXPIRES_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_patient_runs_expires_at
    ON patient_runs (expires_at)
"""


def _patient_hash(patient_profile: dict[str, Any]) -> str:
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _ensure_schema(conn: asyncpg.Connection) -> None:
    async with conn.transaction():
        await conn.execute(_CREATE_SCHEMA_VERSION)
        await conn.execute(_CREATE_RUNS)
        await conn.execute(_CREATE_EXPIRES_INDEX)
        row = await conn.fetchrow("SELECT version FROM schema_version LIMIT 1")
        if row is None:
            await conn.execute(
                "INSERT INTO schema_version (version) VALUES ($1)",
                _SCHEMA_VERSION,
            )


# Add Episodic Memory with postgress!


class EpisodicMemory:
    pass
