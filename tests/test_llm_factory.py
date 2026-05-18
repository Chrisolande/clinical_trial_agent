from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from tools import llm_factory


def _settings(provider: str, *, deepseek_key: str = "k") -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider=provider,
        deepseek_api_key=SimpleNamespace(get_secret_value=lambda: deepseek_key),
        deepseek_model="deepseek-chat",
        llm_call_timeout_seconds=12.0,
    )


def _install_import_stub(
    monkeypatch: pytest.MonkeyPatch, *, class_name: str, constructor_name: str | None = None
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    module = types.ModuleType("fake_mod")
    ctor = constructor_name or class_name

    class _Client:
        def __init__(self, **kwargs):
            calls.append({"kwargs": kwargs})
            self.kwargs = kwargs

    setattr(module, ctor, _Client)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        targets = {
            "langchain_deepseek": ("ChatDeepSeek", class_name == "ChatDeepSeek"),
            "langchain_openai": ("ChatOpenAI", class_name == "ChatOpenAI"),
            "langchain_anthropic": ("ChatAnthropic", class_name == "ChatAnthropic"),
            "langchain_ollama": ("ChatOllama", class_name == "ChatOllama"),
        }
        wanted = targets.get(name)
        if wanted and wanted[1]:
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return calls


def test_is_local_provider() -> None:
    assert llm_factory.is_local_provider(" ollama ")
    assert not llm_factory.is_local_provider("openai")


def test_build_llm_client_deepseek_requires_key() -> None:
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        llm_factory.build_llm_client(_settings("deepseek", deepseek_key=""))


def test_build_llm_client_deepseek_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_import_stub(monkeypatch, class_name="ChatDeepSeek")
    client = llm_factory.build_llm_client(_settings("deepseek", deepseek_key="secret"))
    assert client is not None
    kwargs = calls[0]["kwargs"]
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["temperature"] == 0.0
    assert kwargs["timeout"] == 12.0


def test_build_llm_client_openai_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        llm_factory.build_llm_client(_settings("openai"))

    calls = _install_import_stub(monkeypatch, class_name="ChatOpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    client = llm_factory.build_llm_client(_settings("openai"))
    assert client is not None
    kwargs = calls[0]["kwargs"]
    assert kwargs == {"model": "gpt-4o", "temperature": 0.0, "timeout": 12.0}


def test_build_llm_client_anthropic_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
        llm_factory.build_llm_client(_settings("anthropic"))

    calls = _install_import_stub(monkeypatch, class_name="ChatAnthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ok")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    client = llm_factory.build_llm_client(_settings("anthropic"))
    assert client is not None
    kwargs = calls[0]["kwargs"]
    assert kwargs["model_name"] == "claude-x"
    assert kwargs["stop"] is None


def test_build_llm_client_ollama_and_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_import_stub(monkeypatch, class_name="ChatOllama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    client = llm_factory.build_llm_client(_settings("ollama"))
    assert client is not None
    assert calls[0]["kwargs"] == {"model": "llama3", "temperature": 0.0}

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm_factory.build_llm_client(_settings("unknown"))
