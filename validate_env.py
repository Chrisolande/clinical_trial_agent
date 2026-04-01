from __future__ import annotations

import os
from dataclasses import dataclass

from config import bootstrap_environment, settings


@dataclass(frozen=True)
class EnvStatus:
    database_uri: str
    memory_db_dsn: str
    deepseek_ready: bool


def inspect_environment() -> EnvStatus:
    bootstrap_environment()
    return EnvStatus(
        database_uri=settings.database_uri,
        memory_db_dsn=settings.memory_db_dsn,
        deepseek_ready=bool(settings.deepseek_api_key),
    )


def validate_or_raise() -> EnvStatus:
    status = inspect_environment()
    if status.database_uri != status.memory_db_dsn:
        raise RuntimeError(
            "DATABASE_URI and MEMORY_DB_DSN are not synchronized. "
            f"DATABASE_URI={status.database_uri!r}, MEMORY_DB_DSN={status.memory_db_dsn!r}"
        )
    if not status.deepseek_ready:
        raise RuntimeError("DEEPSEEK_API_KEY is missing.")
    return status


if __name__ == "__main__":
    env_status = validate_or_raise()
    print("Environment is valid")
    print(f"DATABASE_URI={env_status.database_uri}")
    print(f"MEMORY_DB_DSN={env_status.memory_db_dsn}")
    print(f"DEEPSEEK_READY={env_status.deepseek_ready}")
    print(f"CHECKPOINTER_BACKEND={os.getenv('DATABASE_URI', '')}")
