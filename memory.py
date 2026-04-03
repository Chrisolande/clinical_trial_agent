"""PostgreSQL-backed episodic memory."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from config import get_settings
from cryptography.fernet import Fernet
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

    CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key    TEXT PRIMARY KEY,
        value_json   JSONB NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL,
        expires_at   TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_llm_cache_expires_at
        ON llm_cache (expires_at);

    CREATE TABLE IF NOT EXISTS pipeline_audit_log (
        id SERIAL PRIMARY KEY,
        profile_hash TEXT NOT NULL,
        run_id TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        outcome_tier_counts JSONB NOT NULL,
        model_version TEXT NOT NULL,
        consent_flag BOOLEAN NOT NULL
    );

    CREATE TABLE IF NOT EXISTS physician_feedback (
        id SERIAL PRIMARY KEY,
        profile_hash TEXT NOT NULL,
        run_id TEXT NOT NULL,
        nct_id TEXT NOT NULL,
        verdict TEXT NOT NULL,
        note TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    );
"""

_SCHEMA_VERSION = 3


def _get_profile_hash_salt() -> str:
    salt = os.getenv("PROFILE_HASH_SALT", "").strip()
    if not salt:
        raise RuntimeError("PROFILE_HASH_SALT must be set for patient profile hashing")
    return salt


def _get_fernet_key() -> bytes:
    key = os.getenv("DB_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("DB_ENCRYPTION_KEY must be set for encrypted memory storage")
    base64.urlsafe_b64decode(key.encode("utf-8"))
    return key.encode("utf-8")


def _serialize_encrypted_json(payload: dict[str, Any], fernet: Fernet) -> str:
    return fernet.encrypt(json.dumps(payload, default=str).encode("utf-8")).decode("utf-8")


def _deserialize_encrypted_json(payload: str, fernet: Fernet) -> dict[str, Any] | None:
    try:
        decrypted = fernet.decrypt(payload.encode("utf-8")).decode("utf-8")
        parsed = json.loads(decrypted)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _patient_hash(patient_profile: dict[str, Any]) -> str:
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)
    salted = f"{_get_profile_hash_salt()}::{canonical}"
    return hashlib.sha256(salted.encode()).hexdigest()


class EpisodicMemory(PostgresBase):
    """PostgreSQL-backed async store for completed patient pipeline runs."""

    def __init__(
        self,
        dsn: str | None = None,
        ttl_days: int | None = None,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        settings = get_settings()
        super().__init__(dsn=dsn or settings.memory_db_dsn, pool_min=pool_min, pool_max=pool_max)
        self._ttl_days = ttl_days if ttl_days is not None else settings.memory_ttl_days
        self._fernet = Fernet(_get_fernet_key())

    async def _setup_schema(self, conn: asyncpg.Connection) -> None:
        async with conn.transaction():
            await conn.execute(_DDL)
            row = await conn.fetchrow("SELECT version FROM schema_version LIMIT 1")
            if row is None:
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES ($1)", _SCHEMA_VERSION
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
            if isinstance(raw, str):
                parsed = _deserialize_encrypted_json(raw, self._fernet)
                if isinstance(parsed, dict):
                    logger.info("Episodic memory hit", profile_hash=key[:8])
                    return parsed
            logger.warning("Episodic memory payload could not be decrypted; ignoring cache entry")
        logger.debug("Episodic memory miss", profile_hash=key[:8])
        return None

    async def store(self, patient_profile: dict[str, Any], result: dict[str, Any]) -> None:
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
                json.dumps(_serialize_encrypted_json(result, self._fernet)),
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

    async def write_pipeline_audit(
        self,
        patient_profile: dict[str, Any],
        run_id: str,
        outcome_tier_counts: dict[str, int],
        model_version: str,
        consent_flag: bool,
    ) -> None:
        key = _patient_hash(patient_profile)
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pipeline_audit_log
                    (profile_hash, run_id, timestamp, outcome_tier_counts, model_version, consent_flag)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                key,
                run_id,
                now,
                json.dumps(outcome_tier_counts),
                model_version,
                consent_flag,
            )

    async def list_pipeline_audit(self, profile_hash: str) -> list[dict[str, Any]]:
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, run_id, timestamp, outcome_tier_counts, model_version, consent_flag
                FROM pipeline_audit_log
                WHERE profile_hash = $1
                ORDER BY timestamp DESC
                """,
                profile_hash,
            )
        return [
            {
                "profile_hash": r["profile_hash"],
                "run_id": r["run_id"],
                "timestamp": r["timestamp"].isoformat(),
                "outcome_tier_counts": dict(r["outcome_tier_counts"]),
                "model_version": r["model_version"],
                "consent_flag": bool(r["consent_flag"]),
            }
            for r in rows
        ]

    async def save_feedback(
        self,
        patient_profile: dict[str, Any],
        run_id: str,
        nct_id: str,
        verdict: str,
        note: str,
    ) -> None:
        key = _patient_hash(patient_profile)
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO physician_feedback
                    (profile_hash, run_id, nct_id, verdict, note, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                key,
                run_id,
                nct_id,
                verdict,
                note,
                now,
            )

    async def list_feedback(self, profile_hash: str) -> list[dict[str, Any]]:
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, run_id, nct_id, verdict, note, created_at
                FROM physician_feedback
                WHERE profile_hash = $1
                ORDER BY created_at DESC
                """,
                profile_hash,
            )
        return [
            {
                "profile_hash": r["profile_hash"],
                "run_id": r["run_id"],
                "nct_id": r["nct_id"],
                "verdict": r["verdict"],
                "note": r["note"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def erase_profile(self, profile_hash: str) -> None:
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute("DELETE FROM patient_runs WHERE profile_hash = $1", profile_hash)
            await conn.execute(
                "DELETE FROM pipeline_audit_log WHERE profile_hash = $1", profile_hash
            )
            await conn.execute(
                "DELETE FROM physician_feedback WHERE profile_hash = $1", profile_hash
            )


def get_checkpointer(dsn: str | None = None) -> Any | None:
    active_dsn = dsn or get_settings().database_uri
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        return AsyncPostgresSaver.from_conn_string(active_dsn)
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed - checkpointing disabled. "
            "Run: pip install langgraph-checkpoint-postgres"
        )
        return None
