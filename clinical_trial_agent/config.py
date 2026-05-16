import functools
import ipaddress
import os
import tempfile
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tools.llm_factory import build_llm_client, is_local_provider

_ = load_dotenv()
_DEFAULT_DB_URI: Final[str] = "postgresql://postgres:postgres@localhost:5433/postgres"
TIER_ORDER: Final[dict[str, int]] = {
    "disqualified": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


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
        os.environ["DATABASE_URI"] = _DEFAULT_DB_URI
        os.environ["MEMORY_DB_DSN"] = _DEFAULT_DB_URI
        return

    if memory_dsn and not database_uri:
        os.environ["DATABASE_URI"] = memory_dsn
    elif database_uri and not memory_dsn:
        os.environ["MEMORY_DB_DSN"] = database_uri


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    llm_provider: Literal["deepseek", "openai", "anthropic", "ollama"] = "deepseek"
    llm_privacy_mode: Literal["blocked", "deidentified", "full_consent", "local_only"] = "blocked"
    deepseek_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    deepseek_model: str = "deepseek-chat"

    retry_max_attempts: int = 3
    retry_min_wait_seconds: float = 1.0
    retry_max_wait_seconds: float = 30.0
    retry_jitter: float = 0.5

    min_match_tier: str = "moderate"
    criteria_text_max_chars: int = 8000
    use_cache: bool = True

    database_uri: str = _DEFAULT_DB_URI
    memory_db_dsn: str = _DEFAULT_DB_URI

    ctgov_user_agent: str = "clinical-trial-agent"
    ctgov_accept: str = "application/json"
    ctgov_retry_attempts: int = 5
    ctgov_retry_backoff_base: float = 2.0
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_transport_mode: Literal["get", "post"] = "get"
    ctgov_proxy_url: str | None = None
    ctgov_proxy_token: SecretStr = Field(default=SecretStr(""), repr=False)

    max_trials_per_query: int = 10
    cache_ttl_seconds: int = 3600 * 24
    cache_dir: str = Field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "clinical_trial_cache")
    )
    memory_ttl_days: int = 30

    tenant_id: str = "default-tenant"
    facility_id: str = "default-facility"

    log_level: str = "INFO"
    supervisor_use_react: bool = False
    supervisor_agent_timeout_seconds: float = 45.0
    llm_call_timeout_seconds: float = 60.0
    retrieval_internal_max_retries: int = 2
    max_trials_for_eligibility: int | None = None
    one_pass_mode: bool = False

    tavily_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    tavily_max_results: int = 3
    tavily_max_trials_to_enrich: int = 8
    tavily_enable_ctgov_supplement: bool = True

    max_retry_attempts: int = 2

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(_cls, value: object) -> str:
        normalized = str(value or "deepseek").strip().lower()
        if normalized in {"deepseek", "openai", "anthropic", "ollama"}:
            return normalized
        raise ValueError("LLM_PROVIDER must be one of: deepseek, openai, anthropic, ollama")

    @field_validator("llm_privacy_mode", mode="before")
    @classmethod
    def _normalize_privacy_mode(_cls, value: object) -> str:
        normalized = str(value or "blocked").strip().lower()
        if normalized in {"blocked", "deidentified", "full_consent", "local_only"}:
            return normalized
        raise ValueError(
            "LLM_PRIVACY_MODE must be one of: blocked, deidentified, full_consent, local_only"
        )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(_cls, value: object) -> str:
        return str(value or "INFO").upper()

    @field_validator("ctgov_proxy_url")
    @classmethod
    def _validate_ctgov_proxy_url(_cls, value: str | None) -> str | None:
        if not value:
            return None
        parsed = urlparse(value)
        if not parsed.netloc:
            raise ValueError("CTGOV_PROXY_URL must include a host")
        if parsed.scheme == "https":
            return value
        if parsed.scheme == "http" and parsed.hostname is not None:
            try:
                ip = ipaddress.ip_address(parsed.hostname)
            except ValueError:
                if parsed.hostname.lower() == "localhost":
                    return value
            else:
                if ip.is_loopback:
                    return value
        raise ValueError(
            "CTGOV_PROXY_URL must be https, or http://localhost / 127.0.0.1 for local development"
        )

    @model_validator(mode="after")
    def _validate_dsn_consistency(self) -> "Settings":
        if self.database_uri != self.memory_db_dsn:
            raise ValueError(
                "DATABASE_URI and MEMORY_DB_DSN must point to the same database. "
                f"DATABASE_URI={self.database_uri!r}, MEMORY_DB_DSN={self.memory_db_dsn!r}"
            )
        if self.max_trials_for_eligibility is None:
            self.max_trials_for_eligibility = self.max_trials_per_query
        return self


def load_settings() -> Settings:
    bootstrap_environment()
    return Settings()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


class _SettingsProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings = _SettingsProxy()


def external_llm_requires_consent() -> bool:
    return not is_local_provider(get_settings().llm_provider)


def has_external_llm_consent() -> bool:
    return os.environ.get("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "false").strip().lower() == "true"


def require_external_llm_consent(*, node_name: str | None = None) -> None:
    if not external_llm_requires_consent():
        return
    if has_external_llm_consent():
        return
    context = f" for {node_name}" if node_name else ""
    raise RuntimeError(
        "CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required before sending patient data "
        f"to external LLMs{context}."
    )


def is_llm_provider_local() -> bool:
    return is_local_provider(get_settings().llm_provider)


def get_llm(*, contains_phi: bool = True, node_name: str | None = None) -> Any:
    """Return an LLM client, enforcing explicit consent for PHI-bearing calls by default."""
    if contains_phi:
        require_external_llm_consent(node_name=node_name)
    return build_llm_client(get_settings())
