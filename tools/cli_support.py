"""Utility helpers shared by CLI commands."""

import ipaddress
import json
import os
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


_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("100.100.100.200"),
}


def _allowed_webhook_hosts() -> set[str]:
    raw = os.getenv("WEBHOOK_ALLOWED_HOSTS", "")
    return {host.strip().lower() for host in raw.split(",") if host.strip()}


def _is_blocked_webhook_host(hostname: str) -> bool:
    host = hostname.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip in _METADATA_IPS
    )


def validate_webhook_url(url: str, *, allow_local: bool = False) -> None:
    parsed = urlparse(url)
    _validate_webhook_scheme(parsed.scheme, parsed.netloc, parsed.hostname, allow_local)
    if parsed.username or parsed.password:
        raise typer.BadParameter("--webhook-url must not include embedded credentials")
    hostname = parsed.hostname
    if not hostname:
        raise typer.BadParameter("--webhook-url must include a host")

    allowed_hosts = _allowed_webhook_hosts()
    if allowed_hosts and hostname.lower() not in allowed_hosts:
        raise typer.BadParameter("--webhook-url host is not in WEBHOOK_ALLOWED_HOSTS")

    env_allows_local = os.getenv("ALLOW_LOCAL_WEBHOOK", "false").strip().lower() == "true"
    if _is_blocked_webhook_host(hostname) and not (allow_local or env_allows_local):
        raise typer.BadParameter("--webhook-url targets a local or private address")


def _validate_webhook_scheme(
    scheme: str, netloc: str, hostname: str | None, allow_local: bool
) -> None:
    if scheme == "https" and netloc:
        return
    if allow_local and scheme == "http" and hostname:
        return
    raise typer.BadParameter("--webhook-url must be a valid HTTPS URL")


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
