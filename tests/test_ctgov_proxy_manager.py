import os
import subprocess
from pathlib import Path

import pytest
from tools import ctgov_proxy_manager

PROJECT_ROOT = str(Path(ctgov_proxy_manager.__file__).resolve().parent.parent)


@pytest.fixture(autouse=True)
def reset_proxy_manager(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CTGOV_PROXY_URL", raising=False)
    ctgov_proxy_manager._PROXY_PROCESS = None
    yield
    ctgov_proxy_manager._PROXY_PROCESS = None


@pytest.mark.asyncio
async def test_ensure_proxy_spawns_documented_ctgov_proxy_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def unreachable(_host: str, _port: int) -> bool:
        return False

    async def healthy_after_spawn(_host: str, _port: int) -> None:
        return None

    class FakeProcess:
        stderr = None
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(
        cmd: list[str],
        cwd: str,
        stdout: int,
        stderr: int,
    ) -> FakeProcess:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return FakeProcess()

    monkeypatch.setattr(ctgov_proxy_manager, "_is_proxy_reachable", unreachable)
    monkeypatch.setattr(ctgov_proxy_manager, "_wait_for_proxy", healthy_after_spawn)
    monkeypatch.setattr(ctgov_proxy_manager.subprocess, "Popen", fake_popen)

    await ctgov_proxy_manager.ensure_proxy()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "ctgov_proxy:app" in cmd
    assert "--host" in cmd
    assert "--port" in cmd
    assert os.environ["CTGOV_PROXY_URL"] == "http://127.0.0.1:8321/ctgov/search"
    assert captured["cwd"] == PROJECT_ROOT
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.PIPE
