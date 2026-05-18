"""PostgreSQL-backed LLM verdict cache."""

import hashlib
import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from diskcache import Cache, JSONDisk
from loguru import logger

from clinical_trial_agent.config import get_settings
from clinical_trial_agent.constants import (
    ELIGIBILITY_CACHE_SCHEMA_VERSION,
    ELIGIBILITY_EVIDENCE_CONTRACT_VERSION,
)
from tools.postgres_base import PostgresBase

_cache = Cache(get_settings().cache_dir, disk=JSONDisk)
_eligibility_memory_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_ELIGIBILITY_DISK_CACHE_ENV = "ELIGIBILITY_VERDICT_ALLOW_DISK_CACHE"

_DDL = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key   TEXT PRIMARY KEY,
        prefix      TEXT NOT NULL,
        value_json  JSONB NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL,
        expires_at  TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_llm_cache_expires_at ON llm_cache (expires_at);
    CREATE INDEX IF NOT EXISTS idx_llm_cache_prefix ON llm_cache (prefix);
"""


def _make_key(prefix: str, data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_cached(prefix: str, params: Any) -> Any | None:
    if not get_settings().use_cache:
        return None
    key = _make_key(prefix, params)
    try:
        return _cache.get(key)
    except Exception as exc:
        logger.warning("Disk cache get failed", key=key, error=str(exc))
        return None


def set_cached(
    prefix: str,
    params: Any,
    value: Any,
    ttl_seconds: int | None = None,
) -> None:
    if not get_settings().use_cache:
        return
    key = _make_key(prefix, params)
    ttl = ttl_seconds if ttl_seconds is not None else get_settings().cache_ttl_seconds
    try:
        _cache.set(key, value, expire=ttl)
    except Exception as exc:
        logger.warning("Disk cache set failed", key=key, error=str(exc))


class LLMCache(PostgresBase):
    """PostgreSQL-backed async cache for LLM verdict calls."""

    async def _setup_schema(self, conn: asyncpg.Connection) -> None:
        async with conn.transaction():
            await conn.execute(_DDL)

    async def get(self, prefix: str, params: Any) -> Any | None:
        key = _make_key(prefix, params)
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT value_json FROM llm_cache
                WHERE cache_key = $1 AND expires_at > $2
                """,
                key,
                now,
            )
        if row:
            logger.debug("Cache hit", key=key, prefix=prefix)
            return row["value_json"]
        logger.debug("Cache miss", key=key, prefix=prefix)
        return None

    async def set(
        self,
        prefix: str,
        params: Any,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        key = _make_key(prefix, params)
        now = datetime.now(UTC)
        ttl = ttl_seconds if ttl_seconds is not None else get_settings().cache_ttl_seconds
        expires = now + timedelta(seconds=ttl)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_cache
                    (cache_key, prefix, value_json, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (cache_key) DO UPDATE SET
                    value_json = EXCLUDED.value_json,
                    created_at = EXCLUDED.created_at,
                    expires_at = EXCLUDED.expires_at
                """,
                key,
                prefix,
                json.dumps(value, default=str),
                now,
                expires,
            )
        logger.debug("Cache set", key=key, prefix=prefix, ttl_seconds=ttl_seconds)

    async def invalidate(self, prefix: str, params: Any) -> bool:
        key = _make_key(prefix, params)
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute("DELETE FROM llm_cache WHERE cache_key = $1", key)
        removed = int(result.split()[-1]) > 0
        if removed:
            logger.info("Cache entry invalidated", key=key)
        return removed

    async def invalidate_prefix(self, prefix: str) -> int:
        async with self._pool_or_raise().acquire() as conn:
            result = await conn.execute("DELETE FROM llm_cache WHERE prefix = $1", prefix)
        removed = int(result.split()[-1])
        if removed:
            logger.info("Cache prefix invalidated", prefix=prefix, count=removed)
        return removed

    async def purge_expired(self) -> int:
        removed = await self._purge_expired("llm_cache")
        return int(removed)

    async def stats(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    prefix,
                    COUNT(*) AS total,
                    SUM(CASE WHEN expires_at > $1 THEN 1 ELSE 0 END) AS active,
                    SUM(CASE WHEN expires_at <= $1 THEN 1 ELSE 0 END) AS expired
                FROM llm_cache
                GROUP BY prefix
                ORDER BY prefix
                """,
                now,
            )
        return {
            r["prefix"]: {
                "total": r["total"],
                "active": r["active"],
                "expired": r["expired"],
            }
            for r in rows
        }


def _ttl_from_primary_completion_date(primary_completion_date: str | None) -> int:
    default_ttl = int(get_settings().cache_ttl_seconds)
    if not primary_completion_date:
        return default_ttl
    raw = str(primary_completion_date).strip()
    if not raw:
        return default_ttl
    try:
        date_part = raw.split("T", 1)[0]
        completion = datetime.fromisoformat(date_part).replace(tzinfo=UTC)
    except ValueError:
        return default_ttl
    now = datetime.now(UTC)
    if completion <= now:
        return default_ttl
    ttl = int((completion - now).total_seconds())
    return max(60, ttl)


def _eligibility_disk_cache_enabled() -> bool:
    return _truthy_env(_ELIGIBILITY_DISK_CACHE_ENV)


def _eligibility_cache_params(nct_id: str, profile_hash: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "nct_id": str(nct_id),
        "profile_hash": str(profile_hash),
        "tenant_id": settings.tenant_id,
        "facility_id": settings.facility_id,
        "privacy_mode": settings.llm_privacy_mode,
        "cache_schema_version": ELIGIBILITY_CACHE_SCHEMA_VERSION,
        "evidence_contract_version": ELIGIBILITY_EVIDENCE_CONTRACT_VERSION,
    }


def _eligibility_cache_metadata(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_schema_version": ELIGIBILITY_CACHE_SCHEMA_VERSION,
        "evidence_contract_version": ELIGIBILITY_EVIDENCE_CONTRACT_VERSION,
        "scope": params,
    }


def _wrap_eligibility_verdict(
    verdict: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "_cache_metadata": _eligibility_cache_metadata(params),
        "verdict": {
            **verdict,
            "evidence_contract_version": ELIGIBILITY_EVIDENCE_CONTRACT_VERSION,
        },
    }


def _verdict_rows_have_evidence_metadata(verdict: dict[str, Any]) -> bool:
    rows = verdict.get("verdicts", [])
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        if "source_type" not in row:
            return False
        evidence_refs = row.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            return False
        if not all(isinstance(ref, dict) and ref.get("source_type") for ref in evidence_refs):
            return False
    return True


def _unwrap_eligibility_verdict(
    payload: Any, expected_params: dict[str, Any]
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("_cache_metadata")
    verdict = payload.get("verdict")
    if not isinstance(metadata, dict) or not isinstance(verdict, dict):
        return None
    if metadata.get("cache_schema_version") != ELIGIBILITY_CACHE_SCHEMA_VERSION:
        return None
    if metadata.get("evidence_contract_version") != ELIGIBILITY_EVIDENCE_CONTRACT_VERSION:
        return None
    if metadata.get("scope") != expected_params:
        return None
    if verdict.get("evidence_contract_version") != ELIGIBILITY_EVIDENCE_CONTRACT_VERSION:
        return None
    if not _verdict_rows_have_evidence_metadata(verdict):
        return None
    return dict(verdict)


def _get_memory_cached_eligibility(params: dict[str, Any]) -> dict[str, Any] | None:
    key = _make_key("eligibility_verdict", params)
    entry = _eligibility_memory_cache.get(key)
    if entry is None:
        return None
    expires_at, payload = entry
    if expires_at <= time.time():
        _eligibility_memory_cache.pop(key, None)
        return None
    return payload


def _set_memory_cached_eligibility(
    params: dict[str, Any], payload: dict[str, Any], ttl_seconds: int
) -> None:
    key = _make_key("eligibility_verdict", params)
    _eligibility_memory_cache[key] = (time.time() + ttl_seconds, payload)


def get_cached_eligibility_verdict(nct_id: str, profile_hash: str) -> dict[str, Any] | None:
    if not get_settings().use_cache:
        return None
    params = _eligibility_cache_params(nct_id, profile_hash)
    result = _get_memory_cached_eligibility(params)
    if result is None and _eligibility_disk_cache_enabled():
        result = get_cached("eligibility_verdict", params)
    verdict = _unwrap_eligibility_verdict(result, params)
    if verdict is None and result is not None:
        logger.info("Ignoring stale or untrusted eligibility verdict cache entry", nct_id=nct_id)
    return verdict


def set_cached_eligibility_verdict(
    nct_id: str,
    profile_hash: str,
    verdict: dict[str, Any],
    primary_completion_date: str | None = None,
) -> None:
    if not get_settings().use_cache:
        return
    params = _eligibility_cache_params(nct_id, profile_hash)
    ttl = _ttl_from_primary_completion_date(primary_completion_date)
    payload = _wrap_eligibility_verdict(verdict, params)
    _set_memory_cached_eligibility(params, payload, ttl)
    if _eligibility_disk_cache_enabled():
        set_cached("eligibility_verdict", params, payload, ttl_seconds=ttl)
