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


def _scoped_profile_key_for_patient(
    scope: _MemoryScopeProtocol, patient_profile: dict[str, Any]
) -> tuple[str, str, str]:
    tenant_id, facility_id = scope._tenant_context_for_memory()
    key = scope._patient_hash_for_memory(
        patient_profile,
        tenant_id=tenant_id,
        facility_id=facility_id,
    )
    return tenant_id, facility_id, key


def _scoped_profile_key(
    scope: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]
) -> tuple[str, str, str]:
    tenant_id, facility_id = scope._tenant_context_for_memory()
    scoped_profile_hash = scope._resolve_profile_hash_for_memory(
        profile_hash,
        tenant_id=tenant_id,
        facility_id=facility_id,
    )
    return tenant_id, facility_id, scoped_profile_hash


class MemoryAuditFeedbackMixin:
    async def write_pipeline_audit(
        self: _MemoryScopeProtocol,
        patient_profile: dict[str, Any],
        run_id: str,
        outcome_tier_counts: dict[str, int],
        model_version: str,
        consent_flag: bool,
    ) -> None:
        tenant_id, facility_id, key = _scoped_profile_key_for_patient(self, patient_profile)
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
                json.dumps(outcome_tier_counts, sort_keys=True),
                model_version,
                consent_flag,
            )

    async def list_pipeline_audit(
        self: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]
    ) -> list[dict[str, Any]]:
        tenant_id, facility_id, scoped_profile_hash = _scoped_profile_key(self, profile_hash)
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
        audits: list[dict[str, Any]] = []
        for r in rows:
            raw_counts = r["outcome_tier_counts"]
            if isinstance(raw_counts, str):
                try:
                    decoded = json.loads(raw_counts)
                except json.JSONDecodeError:
                    decoded = {}
            elif isinstance(raw_counts, dict):
                decoded = raw_counts
            else:
                decoded = {}

            counts = (
                {str(k): int(v) for k, v in decoded.items()} if isinstance(decoded, dict) else {}
            )

            audits.append(
                {
                    "profile_hash": r["profile_hash"],
                    "run_id": r["run_id"],
                    "timestamp": r["timestamp"].isoformat(),
                    "outcome_tier_counts": counts,
                    "model_version": r["model_version"],
                    "consent_flag": bool(r["consent_flag"]),
                }
            )
        return audits

    async def save_feedback(
        self: _MemoryScopeProtocol,
        patient_profile: dict[str, Any],
        run_id: str,
        nct_id: str,
        verdict: str,
        note: str,
    ) -> None:
        normalized_verdict = verdict.strip().lower()
        if normalized_verdict not in {"confirmed", "rejected"}:
            raise ValueError("verdict must be confirmed or rejected")
        tenant_id, facility_id, key = _scoped_profile_key_for_patient(self, patient_profile)
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
                normalized_verdict,
                note,
                now,
            )

    async def list_feedback(
        self: _MemoryScopeProtocol, profile_hash: str | dict[str, Any]
    ) -> list[dict[str, Any]]:
        tenant_id, facility_id, scoped_profile_hash = _scoped_profile_key(self, profile_hash)
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
        tenant_id, facility_id, scoped_profile_hash = _scoped_profile_key(self, profile_hash)
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
