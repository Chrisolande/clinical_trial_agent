import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import httpx
import openai
from loguru import logger
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from clinical_trial_agent.config import get_settings

__all__ = ["RetryError", "http_retry", "llm_retry"]
std_logger = logging.getLogger("tenacity.retry")

_TRANSIENT_LLM_ERRORS = (
    TimeoutError,
    ConnectionError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)
P = ParamSpec("P")
R = TypeVar("R")


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
        before_sleep=before_sleep_log(std_logger, logging.INFO),
        reraise=True,
    )


def _is_retryable_http_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = int(exc.response.status_code)
        return status == 429 or status >= 500
    return False


def _is_retryable_llm_exception(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_LLM_ERRORS):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = int(exc.status_code)
        return status == 429 or status >= 500
    return False


def llm_retry(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
    wrapped = _base_retry(retry_if_exception(_is_retryable_llm_exception))(func)
    return cast("Callable[P, R]", wrapped)  # tenacity decorator erases callable generics


def http_retry(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
    wrapped = _base_retry(retry_if_exception(_is_retryable_http_exception))(func)
    return cast("Callable[P, R]", wrapped)  # tenacity decorator erases callable generics
