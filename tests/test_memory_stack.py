from __future__ import annotations

import builtins
import json
import types
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.fernet import Fernet
from tools import memory_audit_feedback, memory_helpers, memory_schema

from clinical_trial_agent import memory as memory_module


class _AsyncTx:
    async def __aenter__(self) -> _AsyncTx:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


class _AcquireCtx:
    def __init__(self, conn: Any):
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


class _FakePool:
    def __init__(self, conn: Any):
        self._conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


class _FakeConn:
    def __init__(
        self,
        *,
        fetchrow_result: dict[str, Any] | None = None,
        fetch_result: list[dict[str, Any]] | None = None,
        execute_result: str = "DELETE 1",
        needs_backfill: bool = False,
    ) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_result = fetch_result or []
        self.execute_result = execute_result
        self.needs_backfill = needs_backfill
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncTx:
        return _AsyncTx()

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return self.execute_result

    async def fetchrow(self, _sql: str, *_args: Any) -> dict[str, Any] | None:
        return self.fetchrow_result

    async def fetchval(self, _sql: str, *_args: Any) -> bool:
        return self.needs_backfill

    async def fetch(self, _sql: str, *_args: Any) -> list[dict[str, Any]]:
        return self.fetch_result


class _Scope(memory_audit_feedback.MemoryAuditFeedbackMixin):
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def _tenant_context_for_memory(self) -> tuple[str, str]:
        return ("tenant-a", "facility-a")

    def _patient_hash_for_memory(
        self,
        patient_profile: dict[str, Any],
        *,
        tenant_id: str | None = None,
        facility_id: str | None = None,
    ) -> str:
        return f"{tenant_id}:{facility_id}:{patient_profile['id']}"

    def _resolve_profile_hash_for_memory(
        self,
        profile_hash_or_profile: str | dict[str, Any],
        *,
        tenant_id: str,
        facility_id: str,
    ) -> str:
        if isinstance(profile_hash_or_profile, dict):
            return f"{tenant_id}:{facility_id}:{profile_hash_or_profile['id']}"
        return f"{tenant_id}:{facility_id}:{profile_hash_or_profile}"

    def _pool_or_raise(self) -> _FakePool:
        return _FakePool(self._conn)


def test_memory_helpers_profile_hash_salt_and_fernet_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_HASH_SALT", raising=False)
    with pytest.raises(RuntimeError, match="PROFILE_HASH_SALT must be set"):
        memory_helpers.get_profile_hash_salt()

    monkeypatch.setenv("PROFILE_HASH_SALT", "salt")
    assert memory_helpers.get_profile_hash_salt() == "salt"

    monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY must be set"):
        memory_helpers.get_fernet_key()

    monkeypatch.setenv("DB_ENCRYPTION_KEY", "%%%%")
    assert memory_helpers.get_fernet_key() == b"%%%%"

    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("DB_ENCRYPTION_KEY", key)
    assert memory_helpers.get_fernet_key() == key.encode("utf-8")


def test_memory_helpers_serialize_deserialize() -> None:
    fernet = Fernet(Fernet.generate_key())
    payload = {"x": 1}
    encrypted = memory_helpers.serialize_encrypted_json(payload, fernet)
    assert memory_helpers.deserialize_encrypted_json(encrypted, fernet) == payload
    assert memory_helpers.deserialize_encrypted_json("bad-token", fernet) is None
    encrypted_list = fernet.encrypt(json.dumps([1, 2]).encode("utf-8")).decode("utf-8")
    assert memory_helpers.deserialize_encrypted_json(encrypted_list, fernet) is None


def test_memory_helpers_get_checkpointer_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_helpers, "get_settings", lambda: SimpleNamespace(database_uri="db://dsn")
    )
    real_import = builtins.__import__

    def import_error(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "langgraph.checkpoint.postgres.aio":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_error)
    assert memory_helpers.get_checkpointer() is None

    module = types.ModuleType("langgraph.checkpoint.postgres.aio")

    class Saver:
        @staticmethod
        def from_conn_string(dsn: str) -> dict[str, str]:
            return {"dsn": dsn}

    module.AsyncPostgresSaver = Saver

    def import_success(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "langgraph.checkpoint.postgres.aio":
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_success)
    assert memory_helpers.get_checkpointer() == {"dsn": "db://dsn"}

    class BadSaver:
        @staticmethod
        def from_conn_string(_dsn: str) -> None:
            raise RuntimeError("boom")

    module.AsyncPostgresSaver = BadSaver
    assert memory_helpers.get_checkpointer("db://override") is None


@pytest.mark.asyncio
async def test_setup_memory_schema_branches() -> None:
    conn = _FakeConn(fetchrow_result=None, needs_backfill=True)
    await memory_schema.setup_memory_schema(conn, ddl="CREATE TABLE x();", schema_version=8)
    sqls = "\n".join(sql for sql, _ in conn.executed)
    assert "CREATE TABLE x();" in sqls
    assert "ADD COLUMN IF NOT EXISTS tenant_id" in sqls
    assert "ALTER TABLE llm_cache ADD COLUMN IF NOT EXISTS prefix TEXT" in sqls
    assert "ALTER TABLE pipeline_audit_log" in sqls
    assert "patient_runs_tenant_nonempty" in sqls
    assert "INSERT INTO schema_version" in sqls

    conn_v6 = _FakeConn(fetchrow_result={"version": 6}, needs_backfill=False)
    await memory_schema.setup_memory_schema(conn_v6, ddl="CREATE TABLE x();", schema_version=8)
    sqls_v6 = "\n".join(sql for sql, _ in conn_v6.executed)
    assert "ALTER TABLE llm_cache ADD COLUMN IF NOT EXISTS prefix TEXT" not in sqls_v6
    assert "ALTER TABLE pipeline_audit_log" in sqls_v6
    assert "UPDATE schema_version SET version" in sqls_v6

    conn_v8 = _FakeConn(fetchrow_result={"version": 8}, needs_backfill=False)
    await memory_schema.setup_memory_schema(conn_v8, ddl="CREATE TABLE x();", schema_version=8)
    sqls_v8 = "\n".join(sql for sql, _ in conn_v8.executed)
    assert "UPDATE schema_version SET version" not in sqls_v8


@pytest.mark.asyncio
async def test_memory_audit_feedback_helpers_and_methods() -> None:
    now = datetime.now(UTC)
    conn = _FakeConn(
        fetch_result=[
            {
                "profile_hash": "h1",
                "run_id": "r1",
                "timestamp": now,
                "outcome_tier_counts": '{"strong": "2"}',
                "model_version": "m1",
                "consent_flag": 1,
            },
            {
                "profile_hash": "h1",
                "run_id": "r2",
                "timestamp": now,
                "outcome_tier_counts": {"weak": 3},
                "model_version": "m2",
                "consent_flag": 0,
            },
            {
                "profile_hash": "h1",
                "run_id": "r3",
                "timestamp": now,
                "outcome_tier_counts": "not-json",
                "model_version": "m3",
                "consent_flag": 0,
            },
        ]
    )
    scope = _Scope(conn)

    assert memory_audit_feedback._scoped_profile_key_for_patient(scope, {"id": "p1"}) == (
        "tenant-a",
        "facility-a",
        "tenant-a:facility-a:p1",
    )
    assert memory_audit_feedback._scoped_profile_key(scope, "raw-hash") == (
        "tenant-a",
        "facility-a",
        "tenant-a:facility-a:raw-hash",
    )

    await scope.write_pipeline_audit({"id": "p1"}, "run-1", {"strong": 1}, "model", True)
    assert any("INSERT INTO pipeline_audit_log" in sql for sql, _ in conn.executed)

    audits = await scope.list_pipeline_audit("raw-hash")
    assert audits[0]["outcome_tier_counts"] == {"strong": 2}
    assert audits[1]["outcome_tier_counts"] == {"weak": 3}
    assert audits[2]["outcome_tier_counts"] == {}

    with pytest.raises(ValueError, match="verdict must be confirmed or rejected"):
        await scope.save_feedback({"id": "p1"}, "run", "NCT1", "unknown", "note")
    await scope.save_feedback({"id": "p1"}, "run", "NCT1", "CONFIRMED", "note")
    assert any("INSERT INTO physician_feedback" in sql for sql, _ in conn.executed)

    conn.fetch_result = [
        {
            "profile_hash": "h1",
            "run_id": "r1",
            "nct_id": "NCT1",
            "verdict": "confirmed",
            "note": "ok",
            "created_at": now,
        }
    ]
    feedback = await scope.list_feedback({"id": "p1"})
    assert feedback[0]["verdict"] == "confirmed"
    assert feedback[0]["created_at"] == now.isoformat()

    await scope.erase_profile("raw-hash")
    deletes = [sql for sql, _ in conn.executed if sql.startswith("DELETE FROM")]
    assert len(deletes) >= 3


def test_memory_tenant_context_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_module,
        "get_settings",
        lambda: SimpleNamespace(tenant_id="default-tenant", facility_id="facility-a"),
    )
    with pytest.raises(RuntimeError, match="must be explicitly configured"):
        memory_module._tenant_context()

    monkeypatch.setattr(
        memory_module,
        "get_settings",
        lambda: SimpleNamespace(tenant_id="tenant-a", facility_id=" "),
    )
    with pytest.raises(RuntimeError, match="are required"):
        memory_module._tenant_context()

    monkeypatch.setattr(
        memory_module,
        "get_settings",
        lambda: SimpleNamespace(tenant_id="tenant-a", facility_id="facility-a"),
    )
    monkeypatch.setattr(memory_module, "_get_profile_hash_salt", lambda: "salt")
    hash_a = memory_module._patient_hash({"age": 1})
    hash_b = memory_module._patient_hash({"age": 1}, tenant_id="tenant-b", facility_id="facility-a")
    assert hash_a != hash_b
    assert len(hash_a) == 64

    monkeypatch.setattr(memory_module, "_patient_hash", lambda *_args, **_kwargs: "resolved")
    assert (
        memory_module._resolve_profile_hash(
            {"age": 1}, tenant_id="tenant-a", facility_id="facility-a"
        )
        == "resolved"
    )
    assert (
        memory_module._resolve_profile_hash(
            " profile ", tenant_id="tenant-a", facility_id="facility-a"
        )
        == "profile"
    )
    with pytest.raises(RuntimeError, match="profile_hash is required"):
        memory_module._resolve_profile_hash(" ", tenant_id="tenant-a", facility_id="facility-a")


@pytest.mark.asyncio
async def test_episodic_memory_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_module,
        "get_settings",
        lambda: SimpleNamespace(
            memory_db_dsn="postgresql://x",
            memory_ttl_days=2,
            tenant_id="tenant-a",
            facility_id="facility-a",
        ),
    )
    monkeypatch.setattr(memory_module, "_get_fernet_key", lambda: Fernet.generate_key())
    memory = memory_module.EpisodicMemory()

    monkeypatch.setattr(memory_module, "_patient_hash", lambda *_args, **_kwargs: "hash-1")
    encrypted = memory_module._serialize_encrypted_json({"ok": True}, memory._fernet)
    conn_hit = _FakeConn(fetchrow_result={"result_json": encrypted})
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_hit))

    assert await memory.lookup({"age": 60}) == {"ok": True}

    conn_bad = _FakeConn(fetchrow_result={"result_json": "not-encrypted"})
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_bad))
    assert await memory.lookup({"age": 60}) is None

    conn_store = _FakeConn()
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_store))
    await memory.store({"age": 60}, {"report": "ok"})
    assert any("INSERT INTO patient_runs" in sql for sql, _ in conn_store.executed)

    now = datetime.now(UTC)
    conn_list = _FakeConn(
        fetch_result=[{"profile_hash": "h", "created_at": now, "expires_at": now}]
    )
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_list))
    runs = await memory.list_runs()
    assert runs == [
        {"profile_hash": "h", "created_at": now.isoformat(), "expires_at": now.isoformat()}
    ]

    conn_inv = _FakeConn(execute_result="DELETE 1")
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_inv))
    assert await memory.invalidate({"age": 60}) is True

    conn_inv0 = _FakeConn(execute_result="DELETE 0")
    monkeypatch.setattr(memory, "_pool_or_raise", lambda: _FakePool(conn_inv0))
    assert await memory.invalidate({"age": 60}) is False

    async def fake_purge(_table: str) -> int:
        return 9

    monkeypatch.setattr(memory, "_purge_expired", fake_purge)
    assert await memory.purge_expired() == 9
