from __future__ import annotations

import logging

import openai
import pytest
from tools import retry as retry_module


def test_tenacity_bridge_emits_warning_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[tuple[str, str]] = []

    class _DummyLogger:
        def opt(self, **_kwargs: object) -> _DummyLogger:
            return self

        def log(self, level: str, msg: str) -> None:
            logs.append((level, msg))

    monkeypatch.setattr(retry_module, "logger", _DummyLogger())

    handler = retry_module._LoguruLoggingHandler()
    record = logging.LogRecord(
        name="tenacity.retry",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="retrying",
        args=(),
        exc_info=None,
    )
    handler.emit(record)
    assert logs == [("WARNING", "retrying")]


def test_llm_retry_exhaustion_raises_retry_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "2")
    from config import get_settings

    get_settings.cache_clear()

    attempts = {"n": 0}

    @retry_module.llm_retry
    def _always_timeout() -> str:
        attempts["n"] += 1
        raise TimeoutError("boom")

    with pytest.raises(TimeoutError):
        _always_timeout()
    assert attempts["n"] == 2


def test_transient_openai_error_types_included() -> None:
    expected = {
        TimeoutError,
        ConnectionError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    }
    assert expected.issubset(set(retry_module._TRANSIENT_LLM_ERRORS))
