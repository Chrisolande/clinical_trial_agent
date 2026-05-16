import pytest

from clinical_trial_agent import validate_env


def test_inspect_environment_redacts_and_deepseek_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySecret:
        def get_secret_value(self) -> str:
            return "k"

    class DummySettings:
        database_uri = "postgresql://example-host/db"
        memory_db_dsn = "postgresql://example-host/db"
        deepseek_api_key = DummySecret()

    monkeypatch.setattr(validate_env, "bootstrap_environment", lambda: None)
    monkeypatch.setattr(validate_env, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(validate_env, "redact_dsn", lambda dsn: "redacted")
    status = validate_env.inspect_environment()
    assert status.database_uri == "redacted"
    assert status.memory_db_dsn == "redacted"
    assert status.deepseek_ready is True


@pytest.mark.asyncio
async def test_validate_or_raise_async_rejects_mismatched_dsns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validate_env,
        "inspect_environment",
        lambda: validate_env.EnvStatus("a", "b", True),
    )
    monkeypatch.setattr(
        validate_env, "get_settings", lambda: type("S", (), {"database_uri": "a"})()
    )
    with pytest.raises(RuntimeError, match="not synchronized"):
        await validate_env.validate_or_raise_async()


@pytest.mark.asyncio
async def test_validate_or_raise_async_requires_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validate_env,
        "inspect_environment",
        lambda: validate_env.EnvStatus("a", "a", False),
    )
    monkeypatch.setattr(
        validate_env, "get_settings", lambda: type("S", (), {"database_uri": "a"})()
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        await validate_env.validate_or_raise_async()


@pytest.mark.asyncio
async def test_validate_or_raise_async_pings_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        validate_env,
        "inspect_environment",
        lambda: validate_env.EnvStatus("a", "a", True),
    )
    monkeypatch.setattr(
        validate_env, "get_settings", lambda: type("S", (), {"database_uri": "dsn"})()
    )
    called = {"ping": False}

    async def ping(_dsn: str) -> None:
        called["ping"] = True

    monkeypatch.setattr(validate_env, "_ping_database", ping)
    await validate_env.validate_or_raise_async()
    assert called["ping"] is True
