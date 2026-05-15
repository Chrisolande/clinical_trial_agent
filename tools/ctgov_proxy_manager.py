"""Auto-start the ClinicalTrials.gov proxy when it is configured but not reachable."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

_PROXY_PROCESS: subprocess.Popen[bytes] | None = None
_DEFAULT_PROXY_PORT = 8321
_DEFAULT_PROXY_HOST = "127.0.0.1"
_HEALTH_TIMEOUT = 10.0
_HEALTH_POLL_INTERVAL = 0.3


def _parse_proxy_url(url: str) -> tuple[str, int]:
    """Extract host and port from the configured proxy URL."""
    parsed = urlparse(url)
    host = parsed.hostname or _DEFAULT_PROXY_HOST
    port = parsed.port or _DEFAULT_PROXY_PORT
    return host, port


def _health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/healthz"


async def _is_proxy_reachable(host: str, port: int) -> bool:
    """Probe the proxy's /healthz endpoint."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(_health_url(host, port))
            return bool(response.status_code == 200)
    except (httpx.RequestError, httpx.HTTPStatusError):
        return False


def _shutdown_proxy() -> None:
    """Terminate the proxy subprocess if we spawned it."""
    global _PROXY_PROCESS
    if _PROXY_PROCESS is None:
        return
    proc = _PROXY_PROCESS
    _PROXY_PROCESS = None
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(OSError):
            proc.kill()
    logger.info("ClinicalTrials.gov proxy subprocess terminated")


def _spawn_proxy(host: str, port: int) -> subprocess.Popen[bytes]:
    """Spawn uvicorn running ctgov_proxy:app as a background subprocess."""
    project_root = str(Path(__file__).resolve().parent.parent)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "ctgov_proxy:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    logger.info("Starting ClinicalTrials.gov proxy: {}", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    atexit.register(_shutdown_proxy)
    return proc


async def _wait_for_proxy(host: str, port: int) -> None:
    """Poll the proxy health endpoint until it responds or timeout."""
    elapsed = 0.0
    while elapsed < _HEALTH_TIMEOUT:
        if await _is_proxy_reachable(host, port):
            return
        await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        elapsed += _HEALTH_POLL_INTERVAL

    global _PROXY_PROCESS
    if _PROXY_PROCESS is not None and _PROXY_PROCESS.poll() is not None:
        stderr_output = ""
        if _PROXY_PROCESS.stderr:
            stderr_output = _PROXY_PROCESS.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ClinicalTrials.gov proxy process exited unexpectedly "
            f"(code={_PROXY_PROCESS.returncode}): {stderr_output}"
        )
    raise RuntimeError(
        f"ClinicalTrials.gov proxy did not become healthy within {_HEALTH_TIMEOUT}s "
        f"at {_health_url(host, port)}"
    )


async def ensure_proxy() -> None:
    """Ensure the ClinicalTrials.gov proxy is running.

    - If ``CTGOV_PROXY_URL`` is set, probe it and auto-start if unreachable.
    - If ``CTGOV_PROXY_URL`` is **not** set, set it to a default local address
      and start the proxy automatically.

    The proxy subprocess is terminated on process exit via ``atexit``.
    """
    global _PROXY_PROCESS

    proxy_url = os.environ.get("CTGOV_PROXY_URL", "").strip()

    if not proxy_url:
        # No proxy configured - set up a default and start it
        host, port = _DEFAULT_PROXY_HOST, _DEFAULT_PROXY_PORT
        proxy_url = f"http://{host}:{port}/ctgov/search"
        os.environ["CTGOV_PROXY_URL"] = proxy_url
        logger.info("CTGOV_PROXY_URL not set - defaulting to {} and auto-starting proxy", proxy_url)
    else:
        host, port = _parse_proxy_url(proxy_url)

    # Already reachable - nothing to do
    if await _is_proxy_reachable(host, port):
        logger.info("ClinicalTrials.gov proxy already running at {}:{}", host, port)
        return

    # Already spawned by us but hasn't become healthy yet
    if _PROXY_PROCESS is not None:
        logger.warning("Proxy subprocess already spawned but not yet healthy - waiting")
        await _wait_for_proxy(host, port)
        return

    # Spawn the proxy
    _PROXY_PROCESS = _spawn_proxy(host, port)
    logger.info("Waiting for ClinicalTrials.gov proxy to become healthy...")
    await _wait_for_proxy(host, port)
    logger.info("ClinicalTrials.gov proxy is ready at {}:{}", host, port)
