import pytest
from config import Settings, get_settings


def test_settings_provider_normalization() -> None:
    settings = Settings(
        llm_provider="OPENAI", database_uri="postgresql://x", memory_db_dsn="postgresql://x"
    )
    assert settings.llm_provider == "openai"


def test_settings_dsn_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must point to the same database"):
        Settings(database_uri="postgresql://a", memory_db_dsn="postgresql://b")


def test_settings_max_trials_defaults_to_query_limit() -> None:
    settings = Settings(
        max_trials_per_query=17, database_uri="postgresql://x", memory_db_dsn="postgresql://x"
    )
    assert settings.max_trials_for_eligibility == 17


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URI", "postgresql://cache")
    monkeypatch.setenv("MEMORY_DB_DSN", "postgresql://cache")
    first = get_settings()
    second = get_settings()
    assert first is second
