from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

_DEFAULT_DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_local_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip())


def bootstrap_environment(*, load_local_env: bool = True) -> None:
    if load_local_env:
        _load_local_env_file()

    memory_dsn = os.getenv("MEMORY_DB_DSN")
    database_uri = os.getenv("DATABASE_URI")

    if not memory_dsn and not database_uri:
        memory_dsn = _DEFAULT_DB_URI

    if memory_dsn and not database_uri:
        os.environ["DATABASE_URI"] = memory_dsn
    elif database_uri and not memory_dsn:
        os.environ["MEMORY_DB_DSN"] = database_uri


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    retry_max_attempts: int
    retry_min_wait_seconds: float
    retry_max_wait_seconds: float
    retry_jitter: float
    viable_trial_threshold: int
    min_match_score: float
    criteria_text_max_chars: int
    use_cache: bool
    database_uri: str
    memory_db_dsn: str
    ctgov_user_agent: str
    ctgov_accept: str
    ctgov_retry_attempts: int
    ctgov_retry_backoff_base: float
    ctgov_base_url: str
    max_trials_per_query: int
    cache_ttl_seconds: int
    cache_dir: str
    memory_ttl_days: int
    log_level: str
    max_retry_attempts: int = 5


def load_settings() -> Settings:
    bootstrap_environment()
    database_uri = os.getenv("DATABASE_URI", _DEFAULT_DB_URI)
    memory_dsn = os.getenv("MEMORY_DB_DSN", database_uri)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
        retry_min_wait_seconds=float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1.0")),
        retry_max_wait_seconds=float(os.getenv("RETRY_MAX_WAIT_SECONDS", "30.0")),
        retry_jitter=float(os.getenv("RETRY_JITTER", "0.5")),
        viable_trial_threshold=int(os.getenv("VIABLE_TRIAL_THRESHOLD", "3")),
        min_match_score=float(os.getenv("MIN_MATCH_SCORE", "0.3")),
        criteria_text_max_chars=int(os.getenv("CRITERIA_TEXT_MAX_CHARS", "8000")),
        use_cache=_as_bool(os.getenv("USE_CACHE"), default=True),
        database_uri=database_uri,
        memory_db_dsn=memory_dsn,
        ctgov_user_agent=os.getenv("CTGOV_USER_AGENT", "clinical-trial-agent"),
        ctgov_accept=os.getenv("CTGOV_ACCEPT", "application/json"),
        ctgov_retry_attempts=int(os.getenv("CTGOV_RETRY_ATTEMPTS", "5")),
        ctgov_retry_backoff_base=float(os.getenv("CTGOV_RETRY_BACKOFF_BASE", "2.0")),
        ctgov_base_url=os.getenv("CTGOV_BASE_URL", "https://clinicaltrials.gov/api/v2"),
        max_trials_per_query=int(os.getenv("MAX_TRIALS_PER_QUERY", "10")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", str(3600 * 24))),
        cache_dir=os.getenv("CACHE_DIR", "/tmp/clinical_trial_cache"),
        memory_ttl_days=int(os.getenv("MEMORY_TTL_DAYS", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


settings = load_settings()


def get_llm() -> Any:
    provider = settings.llm_provider

    if provider in {"gemini", "auto"}:
        if settings.gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                if provider == "gemini":
                    raise RuntimeError(
                        "LLM_PROVIDER=gemini requires langchain-google-genai. "
                        "Install it with: pip install langchain-google-genai"
                    ) from exc
            else:
                return ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    temperature=0.0,
                    google_api_key=settings.gemini_api_key,
                )
        elif provider == "gemini":
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY or GOOGLE_API_KEY.")

    if provider not in {"openai", "auto", "gemini"}:
        raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}. Use gemini, openai, or auto.")

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when using the OpenAI provider.")
    return ChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
