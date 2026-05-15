from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical_trial_agent.validate_env import (
    EnvStatus,
    inspect_environment,
    validate_or_raise_async,
)


def test_inspect_environment():
    with (
        patch("clinical_trial_agent.validate_env.bootstrap_environment"),
        patch("clinical_trial_agent.validate_env.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            database_uri="postgresql://user:pass@localhost/db",  # pragma: allowlist secret
            memory_db_dsn="postgresql://user:pass@localhost/db",  # pragma: allowlist secret
            deepseek_api_key=MagicMock(get_secret_value=lambda: "key"),
        )
        status = inspect_environment()
        assert isinstance(status, EnvStatus)
        assert "redacted" in status.database_uri.lower() or "localhost" in status.database_uri
        assert status.deepseek_ready is True


@pytest.mark.asyncio
async def test_validate_or_raise_async_success():
    with (
        patch("clinical_trial_agent.validate_env.inspect_environment") as mock_inspect,
        patch("clinical_trial_agent.validate_env.get_settings") as mock_settings,
        patch("clinical_trial_agent.validate_env._ping_database", new_callable=AsyncMock),
    ):
        mock_inspect.return_value = EnvStatus(
            database_uri="db", memory_db_dsn="db", deepseek_ready=True
        )
        mock_settings.return_value = MagicMock(database_uri="db")

        status = await validate_or_raise_async()
        assert status.deepseek_ready is True


@pytest.mark.asyncio
async def test_validate_or_raise_async_mismatch():
    with patch("clinical_trial_agent.validate_env.inspect_environment") as mock_inspect:
        mock_inspect.return_value = EnvStatus(
            database_uri="db1", memory_db_dsn="db2", deepseek_ready=True
        )
        with pytest.raises(RuntimeError, match="not synchronized"):
            await validate_or_raise_async()


@pytest.mark.asyncio
async def test_validate_or_raise_async_no_api_key():
    with patch("clinical_trial_agent.validate_env.inspect_environment") as mock_inspect:
        mock_inspect.return_value = EnvStatus(
            database_uri="db", memory_db_dsn="db", deepseek_ready=False
        )
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is missing"):
            await validate_or_raise_async()
