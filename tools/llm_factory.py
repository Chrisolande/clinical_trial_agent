import os
from typing import Any

from loguru import logger
from pydantic import SecretStr


def is_local_provider(provider: str) -> bool:
    return provider.strip().lower() == "ollama"


def build_llm_client(settings: Any) -> Any:
    provider = str(settings.llm_provider).strip().lower()

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        api_key = settings.deepseek_api_key.get_secret_value()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required.")
        return ChatDeepSeek(
            model=settings.deepseek_model,
            temperature=0.0,
            timeout=settings.llm_call_timeout_seconds,
            max_retries=0,
            api_key=SecretStr(api_key),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.0, timeout=settings.llm_call_timeout_seconds)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        return ChatAnthropic(
            model_name=model, temperature=0.0, timeout=settings.llm_call_timeout_seconds, stop=None
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        logger.info("Using local Ollama provider; external clinical data consent gate is bypassed.")
        return ChatOllama(model=model, temperature=0.0)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
