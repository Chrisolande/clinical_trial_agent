import pytest

from clinical_trial_agent.config import Settings, get_settings


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


def test_settings_privacy_mode_normalization() -> None:
    settings = Settings(
        llm_privacy_mode="DEIDENTIFIED",
        database_uri="postgresql://x",
        memory_db_dsn="postgresql://x",
    )
    assert settings.llm_privacy_mode == "deidentified"


def test_settings_ctgov_proxy_url_allows_loopback_http() -> None:
    settings = Settings(
        ctgov_proxy_url="http://localhost:8000/ctgov/search",
        database_uri="postgresql://x",
        memory_db_dsn="postgresql://x",
    )
    assert settings.ctgov_proxy_url == "http://localhost:8000/ctgov/search"


def test_settings_ctgov_proxy_url_rejects_remote_http() -> None:
    with pytest.raises(ValueError, match="CTGOV_PROXY_URL must be https"):
        Settings(
            ctgov_proxy_url="http://example.com/ctgov/search",
            database_uri="postgresql://x",
            memory_db_dsn="postgresql://x",
        )


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URI", "postgresql://cache")
    monkeypatch.setenv("MEMORY_DB_DSN", "postgresql://cache")
    first = get_settings()
    second = get_settings()
    assert first is second
