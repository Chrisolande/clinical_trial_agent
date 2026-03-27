from __future__ import annotations

import os
from dataclasses import dataclass

from config import bootstrap_environment, settings


@dataclass(frozen=True)
class EnvStatus:
    database_uri: str
    memory_db_dsn: str
    llm_provider: str
    gemini_ready: bool
    openai_ready: bool


def inspect_environment() -> EnvStatus:
    bootstrap_environment()
    return EnvStatus(
        database_uri=settings.database_uri,
        memory_db_dsn=settings.memory_db_dsn,
        llm_provider=settings.llm_provider,
        gemini_ready=bool(settings.gemini_api_key),
        openai_ready=bool(settings.openai_api_key),
    )


def validate_or_raise() -> EnvStatus:
    status = inspect_environment()
    if status.database_uri != status.memory_db_dsn:
        raise RuntimeError(
            "DATABASE_URI and MEMORY_DB_DSN are not synchronized. "
            f"DATABASE_URI={status.database_uri!r}, MEMORY_DB_DSN={status.memory_db_dsn!r}"
        )

    if status.llm_provider == "gemini" and not status.gemini_ready:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY/GOOGLE_API_KEY is missing.")

    if status.llm_provider == "openai" and not status.openai_ready:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")

    return status


if __name__ == "__main__":
    env_status = validate_or_raise()
    print("Environment is valid")
    print(f"DATABASE_URI={env_status.database_uri}")
    print(f"MEMORY_DB_DSN={env_status.memory_db_dsn}")
    print(f"LLM_PROVIDER={env_status.llm_provider}")
    print(f"GEMINI_READY={env_status.gemini_ready}")
    print(f"OPENAI_READY={env_status.openai_ready}")
    print(f"CHECKPOINTER_BACKEND={os.getenv('DATABASE_URI', '')}")
