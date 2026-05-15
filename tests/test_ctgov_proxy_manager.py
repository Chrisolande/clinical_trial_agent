import os
import signal
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tools.ctgov_proxy_manager import _shutdown_proxy, ensure_proxy


@pytest.mark.asyncio
async def test_ensure_proxy_already_running():
    with patch(
        "tools.ctgov_proxy_manager._is_proxy_reachable", new_callable=AsyncMock
    ) as mock_reachable:
        mock_reachable.return_value = True
        with patch.dict(os.environ, {"CTGOV_PROXY_URL": "http://localhost:8321/ctgov/search"}):
            await ensure_proxy()
            mock_reachable.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_proxy_spawns_new():
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None

    with (
        patch(
            "tools.ctgov_proxy_manager._is_proxy_reachable", new_callable=AsyncMock
        ) as mock_reachable,
        patch("tools.ctgov_proxy_manager._spawn_proxy") as mock_spawn,
        patch("tools.ctgov_proxy_manager._wait_for_proxy", new_callable=AsyncMock) as mock_wait,
    ):
        mock_reachable.side_effect = [False, True]
        mock_spawn.return_value = mock_proc

        with patch.dict(os.environ, {"CTGOV_PROXY_URL": ""}):
            await ensure_proxy()
            mock_spawn.assert_called_once()
            mock_wait.assert_called_once()
            assert os.environ["CTGOV_PROXY_URL"] == "http://127.0.0.1:8321/ctgov/search"


@pytest.mark.asyncio
async def test_shutdown_proxy():
    mock_proc = MagicMock(spec=subprocess.Popen)
    with patch("tools.ctgov_proxy_manager._PROXY_PROCESS", mock_proc):
        _shutdown_proxy()
        mock_proc.send_signal.assert_called_with(signal.SIGTERM)
        mock_proc.wait.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_proxy_timeout():
    from tools.ctgov_proxy_manager import _wait_for_proxy

    with (
        patch(
            "tools.ctgov_proxy_manager._is_proxy_reachable", new_callable=AsyncMock
        ) as mock_reachable,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_reachable.return_value = False
        with (
            patch("tools.ctgov_proxy_manager._HEALTH_TIMEOUT", 0.1),
            patch("tools.ctgov_proxy_manager._HEALTH_POLL_INTERVAL", 0.05),
            pytest.raises(RuntimeError, match="did not become healthy"),
        ):
            await _wait_for_proxy("localhost", 8321)


def test_shutdown_proxy_already_dead():
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.send_signal.side_effect = OSError("Already dead")
    with patch("tools.ctgov_proxy_manager._PROXY_PROCESS", mock_proc):
        _shutdown_proxy()
        mock_proc.kill.assert_called_once()
