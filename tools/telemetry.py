import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger


def configure_tracing() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


def _coerce_slo_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


@contextmanager
def trace_span(name: str, **attrs: Any) -> Iterator[None]:
    started = time.perf_counter()
    slo_ms = _coerce_slo_ms(attrs.pop("slo_ms", None))
    span_logger = logger.bind(span=name, **attrs)
    span_logger.info("span.start")
    try:
        yield
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        span_logger.bind(duration_ms=elapsed_ms, slo_ms=slo_ms).exception("span.error")
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if slo_ms is not None and elapsed_ms > slo_ms:
        span_logger.bind(duration_ms=elapsed_ms, slo_ms=slo_ms).warning("span.slo_miss")
    span_logger.bind(duration_ms=elapsed_ms, slo_ms=slo_ms).info("span.finish")
