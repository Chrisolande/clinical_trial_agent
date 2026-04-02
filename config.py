from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

_DEFAULT_DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
TIER_ORDER: dict[str, int] = {"disqualified": 0, "weak": 1, "moderate": 2, "strong": 3}


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
    deepseek_api_key: str
    deepseek_model: str
    retry_max_attempts: int
    retry_min_wait_seconds: float
    retry_max_wait_seconds: float
    retry_jitter: float
    min_match_tier: str
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
    supervisor_use_react: bool
    supervisor_agent_timeout_seconds: float
    llm_call_timeout_seconds: float
    retrieval_internal_max_retries: int
    max_trials_for_eligibility: int
    one_pass_mode: bool
    tavily_api_key: str
    tavily_max_results: int
    tavily_max_trials_to_enrich: int
    tavily_enable_ctgov_supplement: bool
    max_retry_attempts: int = 5


def load_settings() -> Settings:
    bootstrap_environment()
    database_uri = os.getenv("DATABASE_URI", _DEFAULT_DB_URI)
    memory_dsn = os.getenv("MEMORY_DB_DSN", database_uri)
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
        retry_min_wait_seconds=float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1.0")),
        retry_max_wait_seconds=float(os.getenv("RETRY_MAX_WAIT_SECONDS", "30.0")),
        retry_jitter=float(os.getenv("RETRY_JITTER", "0.5")),
        min_match_tier=os.getenv("MIN_MATCH_TIER", "moderate"),
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
        supervisor_use_react=_as_bool(os.getenv("SUPERVISOR_USE_REACT"), default=False),
        supervisor_agent_timeout_seconds=float(os.getenv("SUPERVISOR_AGENT_TIMEOUT_SECONDS", "45")),
        llm_call_timeout_seconds=float(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "20")),
        retrieval_internal_max_retries=int(os.getenv("RETRIEVAL_INTERNAL_MAX_RETRIES", "0")),
        max_trials_for_eligibility=int(
            os.getenv(
                "MAX_TRIALS_FOR_ELIGIBILITY",
                os.getenv("MAX_TRIALS_PER_QUERY", "10"),
            )
        ),
        one_pass_mode=_as_bool(os.getenv("ONE_PASS_MODE"), default=True),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        tavily_max_results=int(os.getenv("TAVILY_MAX_RESULTS", "3")),
        tavily_max_trials_to_enrich=int(os.getenv("TAVILY_MAX_TRIALS_TO_ENRICH", "8")),
        tavily_enable_ctgov_supplement=_as_bool(
            os.getenv("TAVILY_ENABLE_CTGOV_SUPPLEMENT"),
            default=True,
        ),
        max_retry_attempts=int(os.getenv("MAX_RETRY_ATTEMPTS", "1")),
    )


settings = load_settings()


def get_llm() -> Any:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required.")
    return ChatDeepSeek(
        model=settings.deepseek_model,
        temperature=0.0,
        timeout=settings.llm_call_timeout_seconds,
        max_retries=0,
        api_key=SecretStr(settings.deepseek_api_key),
    )
