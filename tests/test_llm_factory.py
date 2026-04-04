from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr
from tools import llm_factory
from tools.llm_factory import build_llm_client, is_local_provider


def _settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider=provider,
        deepseek_api_key=__import__("pydantic").SecretStr("k"),
        deepseek_model="deepseek-chat",
        llm_call_timeout_seconds=5.0,
    )


def test_is_local_provider() -> None:
    assert llm_factory.is_local_provider("ollama")
    assert not llm_factory.is_local_provider("deepseek")


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError):
        llm_factory.build_llm_client(_settings("unknown"))


def test_is_local_provider_ollama() -> None:
    """Test is_local_provider returns True for ollama."""
    assert is_local_provider("ollama")
    assert is_local_provider("OLLAMA")
    assert is_local_provider("  ollama  ")


def test_is_local_provider_not_ollama() -> None:
    """Test is_local_provider returns False for other providers."""
    assert not is_local_provider("openai")
    assert not is_local_provider("deepseek")
    assert not is_local_provider("anthropic")
    assert not is_local_provider("")


@patch("langchain_deepseek.ChatDeepSeek")
def test_build_llm_client_deepseek(mock_chat_deepseek: MagicMock) -> None:
    """Test build_llm_client with deepseek provider."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = SecretStr("test-key")
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_deepseek.return_value = mock_instance

    result = build_llm_client(mock_settings)
    assert result == mock_instance
    mock_chat_deepseek.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.0,
        timeout=60,
        max_retries=0,
        api_key=SecretStr("test-key"),
    )


@patch("langchain_deepseek.ChatDeepSeek")
def test_build_llm_client_deepseek_missing_key(mock_chat_deepseek: MagicMock) -> None:
    """Test build_llm_client raises when deepseek key is missing."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = SecretStr("")
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_call_timeout_seconds = 60

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        build_llm_client(mock_settings)


@patch("langchain_openai.ChatOpenAI")
def test_build_llm_client_openai(
    mock_chat_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client with openai provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4")

    mock_settings = MagicMock()
    mock_settings.llm_provider = "openai"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    result = build_llm_client(mock_settings)
    assert result == mock_instance
    mock_chat_openai.assert_called_once_with(
        model="gpt-4",
        temperature=0.0,
        timeout=60,
    )


@patch("langchain_openai.ChatOpenAI")
def test_build_llm_client_openai_default_model(
    mock_chat_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client uses default openai model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    mock_settings = MagicMock()
    mock_settings.llm_provider = "openai"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    build_llm_client(mock_settings)
    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"


@patch("langchain_openai.ChatOpenAI")
def test_build_llm_client_openai_missing_key(
    mock_chat_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client raises when openai key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_settings = MagicMock()
    mock_settings.llm_provider = "openai"
    mock_settings.llm_call_timeout_seconds = 60

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        build_llm_client(mock_settings)


@patch("langchain_anthropic.ChatAnthropic")
def test_build_llm_client_anthropic(
    mock_chat_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client with anthropic provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus")

    mock_settings = MagicMock()
    mock_settings.llm_provider = "anthropic"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_anthropic.return_value = mock_instance

    result = build_llm_client(mock_settings)
    assert result == mock_instance
    mock_chat_anthropic.assert_called_once_with(
        model_name="claude-3-opus",
        temperature=0.0,
        timeout=60,
        stop=None,
    )


@patch("langchain_anthropic.ChatAnthropic")
def test_build_llm_client_anthropic_default_model(
    mock_chat_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client uses default anthropic model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    mock_settings = MagicMock()
    mock_settings.llm_provider = "anthropic"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_anthropic.return_value = mock_instance

    build_llm_client(mock_settings)
    call_kwargs = mock_chat_anthropic.call_args.kwargs
    assert call_kwargs["model_name"] == "claude-3-5-haiku-latest"


@patch("langchain_anthropic.ChatAnthropic")
def test_build_llm_client_anthropic_missing_key(
    mock_chat_anthropic: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client raises when anthropic key is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    mock_settings = MagicMock()
    mock_settings.llm_provider = "anthropic"
    mock_settings.llm_call_timeout_seconds = 60

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
        build_llm_client(mock_settings)


@patch("langchain_ollama.ChatOllama")
def test_build_llm_client_ollama(
    mock_chat_ollama: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client with ollama provider."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")

    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"

    mock_instance = MagicMock()
    mock_chat_ollama.return_value = mock_instance

    result = build_llm_client(mock_settings)
    assert result == mock_instance
    mock_chat_ollama.assert_called_once_with(
        model="llama3.2",
        temperature=0.0,
    )


@patch("langchain_ollama.ChatOllama")
def test_build_llm_client_ollama_default_model(
    mock_chat_ollama: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test build_llm_client uses default ollama model."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"

    mock_instance = MagicMock()
    mock_chat_ollama.return_value = mock_instance

    build_llm_client(mock_settings)
    call_kwargs = mock_chat_ollama.call_args.kwargs
    assert call_kwargs["model"] == "llama3.1"


def test_build_llm_client_unsupported_provider() -> None:
    """Test build_llm_client raises for unsupported provider."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "unsupported"

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: unsupported"):
        build_llm_client(mock_settings)


@patch("langchain_deepseek.ChatDeepSeek")
def test_build_llm_client_case_insensitive(mock_chat_deepseek: MagicMock) -> None:
    """Test build_llm_client handles provider case insensitively."""
    mock_settings = MagicMock()
    mock_settings.llm_provider = "  DEEPSEEK  "
    mock_settings.deepseek_api_key = SecretStr("test-key")
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_call_timeout_seconds = 60

    mock_instance = MagicMock()
    mock_chat_deepseek.return_value = mock_instance
    result = build_llm_client(mock_settings)
    assert result == mock_instance
