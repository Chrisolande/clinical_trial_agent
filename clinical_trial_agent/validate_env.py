import asyncio
from dataclasses import dataclass

import asyncpg
from loguru import logger
from tools.postgres_base import redact_dsn

from clinical_trial_agent.config import bootstrap_environment, get_settings


@dataclass(frozen=True)
class EnvStatus:
    database_uri: str
    memory_db_dsn: str
    deepseek_ready: bool


def inspect_environment() -> EnvStatus:
    bootstrap_environment()
    settings = get_settings()
    return EnvStatus(
        database_uri=redact_dsn(settings.database_uri),
        memory_db_dsn=redact_dsn(settings.memory_db_dsn),
        deepseek_ready=bool(settings.deepseek_api_key.get_secret_value()),
    )


async def _ping_database(dsn: str) -> None:
    conn = await asyncpg.connect(dsn=dsn, timeout=5)
    try:
        await conn.fetchval("SELECT 1")
    finally:
        await conn.close()


async def validate_or_raise_async() -> EnvStatus:
    status = inspect_environment()
    settings = get_settings()

    if status.database_uri != status.memory_db_dsn:
        raise RuntimeError(
            "DATABASE_URI and MEMORY_DB_DSN are not synchronized. "
            f"DATABASE_URI={status.database_uri!r}, MEMORY_DB_DSN={status.memory_db_dsn!r}"
        )
    if not status.deepseek_ready:
        raise RuntimeError("DEEPSEEK_API_KEY is missing.")

    await _ping_database(settings.database_uri)
    return status


def validate_or_raise() -> EnvStatus:
    return asyncio.run(validate_or_raise_async())


if __name__ == "__main__":
    env_status = validate_or_raise()
    logger.info("Environment is valid")
    logger.info("DATABASE_URI={}", env_status.database_uri)
    logger.info("MEMORY_DB_DSN={}", env_status.memory_db_dsn)
    logger.info("DEEPSEEK_READY={}", env_status.deepseek_ready)
