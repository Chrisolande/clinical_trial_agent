from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cli
import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def sample_profile(tmp_path: Path) -> Path:
    profile = {"age": 50, "primary_condition": "Condition X"}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path


def test_run_command(sample_profile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySupervisor:
        async def ainvoke(
            self,
            patient_profile: dict[str, Any],
            *,
            thread_id: str,
            recursion_limit: int = 25,
        ) -> dict[str, Any]:
            _ = (patient_profile, thread_id, recursion_limit)
            return {"report_text": "ok"}

    class DummyContext:
        async def __aenter__(self) -> DummySupervisor:
            return DummySupervisor()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = (exc_type, exc, tb)
            return None

    monkeypatch.setattr(cli, "compile_supervisor_graph", lambda: DummyContext())
    result = runner.invoke(cli.app, ["run", str(sample_profile)])
    assert result.exit_code == 0


def test_search_command(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search_trials(**_: Any) -> dict[str, Any]:
        return {
            "studies": [{"nct_id": "NCT1", "brief_title": "Trial", "overall_status": "RECRUITING"}]
        }

    monkeypatch.setattr(cli, "search_trials", fake_search_trials)
    result = runner.invoke(cli.app, ["search"])
    assert result.exit_code == 0


def test_memory_list_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMemory:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def list_runs(self) -> list[dict[str, str]]:
            return [{"profile_hash": "abc", "created_at": "c", "expires_at": "e"}]

    async def fake_with_memory() -> DummyMemory:
        return DummyMemory()

    monkeypatch.setattr(cli, "_with_memory", fake_with_memory)
    result = runner.invoke(cli.app, ["memory", "list"])
    assert result.exit_code == 0


def test_memory_purge_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMemory:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def purge_expired(self) -> int:
            return 1

    async def fake_with_memory() -> DummyMemory:
        return DummyMemory()

    monkeypatch.setattr(cli, "_with_memory", fake_with_memory)
    result = runner.invoke(cli.app, ["memory", "purge"])
    assert result.exit_code == 0


def test_memory_invalidate_command(sample_profile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMemory:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def invalidate(self, patient_profile: dict[str, Any]) -> bool:
            _ = patient_profile
            return True

    async def fake_with_memory() -> DummyMemory:
        return DummyMemory()

    monkeypatch.setattr(cli, "_with_memory", fake_with_memory)
    result = runner.invoke(cli.app, ["memory", "invalidate", str(sample_profile)])
    assert result.exit_code == 0


def test_validate_env_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEnv:
        database_uri = "postgres://x"
        memory_db_dsn = "postgres://x"
        deepseek_ready = True
        llm_call_timeout_seconds = 20.0
        retrieval_internal_max_retries = 1
        max_trials_for_eligibility = 10

    monkeypatch.setattr(cli, "validate_or_raise", lambda: DummyEnv())
    result = runner.invoke(cli.app, ["validate-env"])
    assert result.exit_code == 0
