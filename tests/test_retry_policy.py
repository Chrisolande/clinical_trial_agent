import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("openai")

from tools import retry as retry_mod


class _RetrySettings:
    retry_max_attempts = 2
    retry_min_wait_seconds = 0.0
    retry_max_wait_seconds = 0.0
    retry_jitter = 0.0
    log_level = "INFO"


@pytest.fixture(autouse=True)
def _patch_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_mod, "get_settings", lambda: _RetrySettings())


@pytest.mark.asyncio
async def test_http_retry_retries_transient_status() -> None:
    request = httpx.Request("GET", "https://example.com")
    response_503 = httpx.Response(status_code=503, request=request)
    calls = 0

    @retry_mod.http_retry
    async def _operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.HTTPStatusError("server error", request=request, response=response_503)
        return "ok"

    result = await _operation()
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_http_retry_does_not_retry_non_transient_status() -> None:
    request = httpx.Request("GET", "https://example.com")
    response_400 = httpx.Response(status_code=400, request=request)
    calls = 0

    @retry_mod.http_retry
    async def _operation() -> str:
        nonlocal calls
        calls += 1
        raise httpx.HTTPStatusError("bad request", request=request, response=response_400)

    with pytest.raises(httpx.HTTPStatusError):
        await _operation()
    assert calls == 1


@pytest.mark.asyncio
async def test_llm_retry_retries_timeout_error() -> None:
    calls = 0

    @retry_mod.llm_retry
    async def _operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timeout")
        return "ok"

    result = await _operation()
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_llm_retry_does_not_retry_non_transient_error() -> None:
    calls = 0

    @retry_mod.llm_retry
    async def _operation() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("non-transient")

    with pytest.raises(ValueError):
        await _operation()
    assert calls == 1
