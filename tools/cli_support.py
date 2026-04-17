"""Utility helpers shared by CLI commands."""

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from agents.patient_parser import parse_patient_profile
from agents.supervisor import compile_supervisor_graph
from rich.console import Console

from clinical_trial_agent.memory import EpisodicMemory

MAX_PROFILE_BYTES = 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    if size_bytes > MAX_PROFILE_BYTES:
        raise typer.BadParameter(f"Profile JSON exceeds 1 MB limit: {size_bytes} bytes")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to load JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


async def with_memory() -> EpisodicMemory:
    memory = EpisodicMemory()
    await memory.init()
    return memory


def validate_webhook_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise typer.BadParameter("--webhook-url must be a valid http(s) URL")


def load_text_profile(path: Path) -> str:
    size_bytes = path.stat().st_size
    if size_bytes > MAX_PROFILE_BYTES:
        raise typer.BadParameter(f"Profile text exceeds 1 MB limit: {size_bytes} bytes")
    return path.read_text(encoding="utf-8")


async def parse_profile_text(raw_text: str) -> dict[str, Any]:
    return await parse_patient_profile(raw_text)


async def run_pipeline(
    patient_profile: dict[str, Any], thread_id: str, *, stream: bool, console: Console
) -> dict[str, Any]:
    async with compile_supervisor_graph() as supervisor:
        if stream:
            console.print("[cyan]event:[/cyan] supervisor.start")
        result = await supervisor.ainvoke(patient_profile, thread_id=thread_id, recursion_limit=25)
        if stream:
            console.print("[cyan]event:[/cyan] supervisor.complete")
        if not isinstance(result, dict):
            raise RuntimeError(f"Supervisor returned unexpected type: {type(result)!r}")
        return result
