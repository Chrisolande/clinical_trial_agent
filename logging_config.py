"""Centralised logging configuration."""

from __future__ import annotations

import sys

from config import settings
from loguru import logger


def configure_logging() -> None:
    """Configure loguru with consistent format and noise suppression."""
    level = settings.log_level

    logger.remove()
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
        for noisy in (
            "httpx",
            "httpcore",
            "langchain",
            "langgraph",
        ):
            logger.disable(noisy)
