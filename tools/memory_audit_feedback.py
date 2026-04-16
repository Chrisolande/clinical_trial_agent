"""Audit and feedback methods for episodic memory."""

import json
from datetime import UTC, datetime
from typing import Any, Protocol


class _MemoryScopeProtocol(Protocol):
    def _tenant_context_for_memory(self) -> tuple[str, str]: ...

    def _patient_hash_for_memory(
        self,
        patient_profile: dict[str, Any],
        *,
        tenant_id: str | None = None,
        facility_id: str | None = None,
    ) -> str: ...

    def _resolve_profile_hash_for_memory(
        self,
        profile_hash_or_profile: str | dict[str, Any],
        *,
        tenant_id: str,
        facility_id: str,
    ) -> str: ...

    def _pool_or_raise(self) -> Any: ...


class MemoryAuditFeedbackMixin:
    async def write_pipeline_audit(
        self: _MemoryScopeProtocol,
        patient_profile: dict[str, Any],
        run_id: str,
        outcome_tier_counts: dict[str, int],
        model_version: str,
        consent_flag: bool,
    ) -> None:
        tenant_id, facility_id = self._tenant_context_for_memory()
        key = self._patient_hash_for_memory(
            patient_profile, tenant_id=tenant_id, facility_id=facility_id
        )
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pipeline_audit_log
                    (tenant_id, facility_id, profile_hash, run_id, timestamp, outcome_tier_counts, model_version, consent_flag)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                tenant_id,
                facility_id,
                key,
                run_id,
                now,
                json.dumps(outcome_tier_counts),
                model_version,
                consent_flag,
            )

    async def list_pipeline_audit(
        self: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]
    ) -> list[dict[str, Any]]:
        tenant_id, facility_id = self._tenant_context_for_memory()
        scoped_profile_hash = self._resolve_profile_hash_for_memory(
            profile_hash,
            tenant_id=tenant_id,
            facility_id=facility_id,
        )
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, run_id, timestamp, outcome_tier_counts, model_version, consent_flag
                FROM pipeline_audit_log
                WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3
                ORDER BY timestamp DESC
                """,
                scoped_profile_hash,
                tenant_id,
                facility_id,
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
        self: _MemoryScopeProtocol,
        patient_profile: dict[str, Any],
        run_id: str,
        nct_id: str,
        verdict: str,
        note: str,
    ) -> None:
        tenant_id, facility_id = self._tenant_context_for_memory()
        key = self._patient_hash_for_memory(
            patient_profile, tenant_id=tenant_id, facility_id=facility_id
        )
        now = datetime.now(UTC)
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO physician_feedback
                    (tenant_id, facility_id, profile_hash, run_id, nct_id, verdict, note, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                tenant_id,
                facility_id,
                key,
                run_id,
                nct_id,
                verdict,
                note,
                now,
            )

    async def list_feedback(
        self: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]
    ) -> list[dict[str, Any]]:
        tenant_id, facility_id = self._tenant_context_for_memory()
        scoped_profile_hash = self._resolve_profile_hash_for_memory(
            profile_hash,
            tenant_id=tenant_id,
            facility_id=facility_id,
        )
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_hash, run_id, nct_id, verdict, note, created_at
                FROM physician_feedback
                WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3
                ORDER BY created_at DESC
                """,
                scoped_profile_hash,
                tenant_id,
                facility_id,
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

    async def erase_profile(self: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]) -> None:
        tenant_id, facility_id = self._tenant_context_for_memory()
        scoped_profile_hash = self._resolve_profile_hash_for_memory(
            profile_hash,
            tenant_id=tenant_id,
            facility_id=facility_id,
        )
        async with self._pool_or_raise().acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM patient_runs WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3",
                scoped_profile_hash,
                tenant_id,
                facility_id,
            )
            await conn.execute(
                "DELETE FROM pipeline_audit_log WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3",
                scoped_profile_hash,
                tenant_id,
                facility_id,
            )
            await conn.execute(
                "DELETE FROM physician_feedback WHERE profile_hash = $1 AND tenant_id = $2 AND facility_id = $3",
                scoped_profile_hash,
                tenant_id,
                facility_id,
            )
