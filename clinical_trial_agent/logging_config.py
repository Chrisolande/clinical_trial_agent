"""Centralised logging configuration."""

import json
import sys
from typing import Any

from loguru import logger
from tools.telemetry import configure_tracing

from clinical_trial_agent.config import get_settings


def _json_sink(message: Any) -> None:
    record = message.record
    payload = {
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "name": record["name"],
        "message": record["message"],
    }
    sys.stderr.write(json.dumps(payload) + "\n")


def configure_logging(log_format: str = "text") -> None:
    """Configure loguru with consistent format and noise suppression."""
    level = get_settings().log_level

    configure_tracing()
    logger.remove()
    if log_format == "json":
        logger.add(_json_sink, level=level, backtrace=True, diagnose=level == "DEBUG")
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DDTHH:mm:ss}</green> "
                "[<cyan>{name}</cyan>] "
                "<level>{level}</level>  "
                "{message}"
            ),
            colorize=True,
            backtrace=True,
            diagnose=level == "DEBUG",
        )

    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "langchain", "langgraph"):
            logger.disable(noisy)
