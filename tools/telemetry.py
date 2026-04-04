import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def configure_tracing() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


@contextmanager
def trace_span(_name: str, **_attrs: Any) -> Iterator[None]:
    yield
