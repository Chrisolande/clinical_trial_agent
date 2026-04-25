"""PostgreSQL-backed episodic memory."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from cryptography.fernet import Fernet
from loguru import logger
from tools.memory_audit_feedback import MemoryAuditFeedbackMixin
from tools.memory_helpers import (
    DDL as _DDL,
)
from tools.memory_helpers import (
    SCHEMA_VERSION as _SCHEMA_VERSION,
)
from tools.memory_helpers import (
    deserialize_encrypted_json as _deserialize_encrypted_json,
)
from tools.memory_helpers import (
    get_checkpointer,
)
from tools.memory_helpers import (
    get_fernet_key as _get_fernet_key,
)
from tools.memory_helpers import (
    get_profile_hash_salt as _get_profile_hash_salt,
)
from tools.memory_helpers import (
    serialize_encrypted_json as _serialize_encrypted_json,
)
from tools.postgres_base import PostgresBase

from clinical_trial_agent.config import get_settings

__all__ = [
    "EpisodicMemory",
    "_patient_hash",
    "_resolve_profile_hash",
    "_serialize_encrypted_json",
    "get_checkpointer",
]


def _tenant_context(
    tenant_id: str | None = None, facility_id: str | None = None
) -> tuple[str, str]:
    settings = get_settings()
    resolved_tenant = settings.tenant_id if tenant_id is None else tenant_id
    resolved_facility = settings.facility_id if facility_id is None else facility_id
    tenant = str(resolved_tenant).strip()
    facility = str(resolved_facility).strip()
    if tenant == "default-tenant" or facility == "default-facility":
        raise RuntimeError(
            "tenant_id and facility_id must be explicitly configured for clinical data access"
        )
    if not tenant or not facility:
        raise RuntimeError("tenant_id and facility_id are required for clinical data access")
    return tenant, facility


def _patient_hash(
    patient_profile: dict[str, Any],
    *,
    tenant_id: str | None = None,
    facility_id: str | None = None,
) -> str:
    tenant, facility = _tenant_context(tenant_id=tenant_id, facility_id=facility_id)
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)
    salted = f"{_get_profile_hash_salt()}::{tenant}::{facility}::{canonical}"
    return hashlib.sha256(salted.encode()).hexdigest()


def _resolve_profile_hash(
    profile_hash_or_profile: str | dict[str, Any],
    *,
    tenant_id: str,
    facility_id: str,
) -> str:
    if isinstance(profile_hash_or_profile, dict):
        return _patient_hash(
            profile_hash_or_profile,
            tenant_id=tenant_id,
            facility_id=facility_id,
        )
    profile_hash = str(profile_hash_or_profile).strip()
    if not profile_hash:
        raise RuntimeError("profile_hash is required for profile-scoped memory operations")
    return profile_hash


class EpisodicMemory(MemoryAuditFeedbackMixin, PostgresBase):
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

    def _tenant_context_for_memory(self) -> tuple[str, str]:
        return _tenant_context()

    def _patient_hash_for_memory(
        self,
        patient_profile: dict[str, Any],
        *,
        tenant_id: str | None = None,
        facility_id: str | None = None,
    ) -> str:
        return _patient_hash(patient_profile, tenant_id=tenant_id, facility_id=facility_id)

    def _resolve_profile_hash_for_memory(
        self,
        profile_hash_or_profile: str | dict[str, Any],
        *,
        tenant_id: str,
        facility_id: str,
    ) -> str:
        return _resolve_profile_hash(
            profile_hash_or_profile,
            tenant_id=tenant_id,
            facility_id=facility_id,
        )

    async def _setup_schema(self, conn: asyncpg.Connection) -> None:
        async with conn.transaction():
            await conn.execute(_DDL)
            await conn.execute("ALTER TABLE patient_runs ADD COLUMN IF NOT EXISTS tenant_id TEXT")
            await conn.execute("ALTER TABLE patient_runs ADD COLUMN IF NOT EXISTS facility_id TEXT")
            await conn.execute(
                "ALTER TABLE pipeline_audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE pipeline_audit_log ADD COLUMN IF NOT EXISTS facility_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE physician_feedback ADD COLUMN IF NOT EXISTS tenant_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE physician_feedback ADD COLUMN IF NOT EXISTS facility_id TEXT"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patient_runs_tenant_facility ON patient_runs (tenant_id, facility_id, created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_audit_log_tenant_facility ON pipeline_audit_log (tenant_id, facility_id, timestamp DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_physician_feedback_tenant_facility ON physician_feedback (tenant_id, facility_id, created_at DESC)"
            )
            row = await conn.fetchrow("SELECT version FROM schema_version LIMIT 1")
            current_version = int(row["version"]) if row and "version" in row else 0

            if current_version < 6:
                await conn.execute("ALTER TABLE llm_cache ADD COLUMN IF NOT EXISTS prefix TEXT")
                needs_backfill = await conn.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM llm_cache WHERE prefix IS NULL LIMIT 1)"
                )
                if bool(needs_backfill):
                    await conn.execute("UPDATE llm_cache SET prefix = '' WHERE prefix IS NULL")
                await conn.execute("ALTER TABLE llm_cache ALTER COLUMN prefix SET NOT NULL")
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_cache_prefix ON llm_cache (prefix)"
                )

            if row is None:
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES ($1)", _SCHEMA_VERSION
                )
            elif current_version < _SCHEMA_VERSION:
                await conn.execute("UPDATE schema_version SET version = $1", _SCHEMA_VERSION)

    async def lookup(self, patient_profile: dict[str, Any]) -> dict[str, Any] | None:
        tenant_id, facility_id = _tenant_context()
        key = _patient_hash(patient_profile, tenant_id=tenant_id, facility_id=facility_id)
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT result_json FROM patient_runs
                WHERE profile_hash = $1
                  AND tenant_id = $2
                  AND facility_id = $3
                  AND expires_at > $4
                """,
                key,
                tenant_id,
                facility_id,
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
        tenant_id, facility_id = _tenant_context()
        key = _patient_hash(patient_profile, tenant_id=tenant_id, facility_id=facility_id)
        now = datetime.now(UTC)
        expires = now + timedelta(days=self._ttl_days)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO patient_runs
                    (profile_hash, tenant_id, facility_id, result_json, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (profile_hash) DO UPDATE SET
                    tenant_id   = EXCLUDED.tenant_id,
                    facility_id = EXCLUDED.facility_id,
                    result_json = EXCLUDED.result_json,
                    created_at  = EXCLUDED.created_at,
                    expires_at  = EXCLUDED.expires_at
                """,
                key,
                tenant_id,
                facility_id,
                json.dumps(_serialize_encrypted_json(result, self._fernet)),
                now,
                expires,
            )
        logger.info("Episodic memory stored", profile_hash=key[:8], ttl_days=self._ttl_days)

    async def purge_expired(self) -> int:
        removed = await self._purge_expired("patient_runs")
        return int(removed)

    async def list_runs(self) -> list[dict[str, Any]]:
        tenant_id, facility_id = _tenant_context()
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, created_at, expires_at
                FROM patient_runs
                WHERE tenant_id = $1 AND facility_id = $2 AND expires_at > $3
                ORDER BY created_at DESC
                """,
                tenant_id,
                facility_id,
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
        tenant_id, facility_id = _tenant_context()
        key = _patient_hash(patient_profile, tenant_id=tenant_id, facility_id=facility_id)
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM patient_runs WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3",
                key,
                tenant_id,
                facility_id,
            )
        removed = int(result.split()[-1]) > 0
        if removed:
            logger.info("Episodic memory invalidated", profile_hash=key[:8])
        return removed
