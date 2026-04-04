import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import httpx
import openai
from config import get_settings
from loguru import logger
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

__all__ = ["RetryError", "http_retry", "llm_retry"]
P = ParamSpec("P")
R = TypeVar("R")
std_logger = logging.getLogger("tenacity.retry")

_TRANSIENT_LLM_ERRORS = (
    TimeoutError,
    ConnectionError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)
_TRANSIENT_HTTP_ERRORS = (httpx.HTTPStatusError, httpx.RequestError)


class _LoguruLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info, depth=6).log(record.levelname, record.getMessage())


def _ensure_tenacity_logger_bridge() -> None:
    if not any(isinstance(handler, _LoguruLoggingHandler) for handler in std_logger.handlers):
        std_logger.handlers.clear()
        std_logger.addHandler(_LoguruLoggingHandler())
        std_logger.propagate = False
    std_logger.setLevel(getattr(logging, get_settings().log_level, logging.INFO))


def _base_retry(retry_condition: Any) -> Any:
    _ensure_tenacity_logger_bridge()
    return retry(
        stop=stop_after_attempt(get_settings().retry_max_attempts),
        wait=wait_exponential_jitter(
            initial=get_settings().retry_min_wait_seconds,
            max=get_settings().retry_max_wait_seconds,
            jitter=get_settings().retry_jitter,
        ),
        retry=retry_condition,
        before_sleep=before_sleep_log(cast("Any", std_logger), logging.INFO),
        reraise=True,
    )


def llm_retry[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    wrapped = _base_retry(retry_if_exception_type(_TRANSIENT_LLM_ERRORS))(func)
    return cast("Callable[P, R]", wrapped)


def http_retry[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    wrapped = _base_retry(retry_if_exception_type(_TRANSIENT_HTTP_ERRORS))(func)
    return cast("Callable[P, R]", wrapped)
