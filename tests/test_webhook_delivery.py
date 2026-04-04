from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cli
import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_webhook_delivery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = tmp_path / "p.json"
    profile.write_text(json.dumps({"age": 60, "conditions": ["NSCLC"]}), encoding="utf-8")

    async def _run_pipeline(
        _profile: dict[str, Any], _thread_id: str, *, stream: bool
    ) -> dict[str, Any]:
        _ = stream
        return {"report_text": "ok", "report_json": {"summary": {}}}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            _ = (exc_type, exc, tb)

        async def post(self, _url: str, json: dict[str, Any]) -> _Resp:  # type: ignore[override]
            assert "run_id" in json
            return _Resp()

    monkeypatch.setattr(cli, "_run_pipeline", _run_pipeline)
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda timeout: _Client())

    result = runner.invoke(
        cli.app, ["run", str(profile), "--webhook-url", "https://example.test/hook"]
    )
    assert result.exit_code == 0
    assert "run_id=" in result.stdout
