from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import httpx
from config import settings
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

__all__ = ["RetryError", "http_retry", "llm_retry"]
std_logger = logging.getLogger("tenacity.retry")
P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])

_TRANSIENT_LLM_ERRORS = (TimeoutError, ConnectionError)
_TRANSIENT_HTTP_ERRORS = (httpx.HTTPStatusError, httpx.RequestError)


def _base_retry(retry_condition: Any) -> Any:
    return retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential_jitter(
            initial=settings.retry_min_wait_seconds,
            max=settings.retry_max_wait_seconds,
            jitter=settings.retry_jitter,
        ),
        retry=retry_condition,
        before_sleep=before_sleep_log(std_logger, logging.INFO),
        reraise=True,
    )


def llm_retry(func: Callable[P, R]) -> Callable[P, R]:
    wrapped = _base_retry(retry_if_exception_type(_TRANSIENT_LLM_ERRORS))(func)
    return cast("Callable[P, R]", wrapped)


def http_retry(func: Callable[P, R]) -> Callable[P, R]:
    wrapped = _base_retry(retry_if_exception_type(_TRANSIENT_HTTP_ERRORS))(func)
    return cast("Callable[P, R]", wrapped)
