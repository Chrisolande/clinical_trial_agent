"""Centralised logging configuration."""

from __future__ import annotations

import os
import sys

from loguru import logger


def configure_logging() -> None:
    """Configure loguru with consistent format and noise suppression."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()

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
            "openai._base_client",
            "langchain",
            "langgraph",
        ):
            logger.disable(noisy)
