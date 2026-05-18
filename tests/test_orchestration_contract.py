import contextlib
import io
from typing import Any

import pytest
from rich.console import Console


@pytest.mark.asyncio
async def test_cli_run_pipeline_uses_supervisor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import cli_support

    events: list[tuple[Any, ...]] = []

    class DummySupervisor:
        async def ainvoke(
            self,
            patient_profile: dict[str, Any],
            *,
            thread_id: str,
            recursion_limit: int,
        ) -> dict[str, Any]:
            events.append(("ainvoke", patient_profile, thread_id, recursion_limit))
            return {"report_json": {"ok": True}, "report_text": "done"}

    @contextlib.asynccontextmanager
    async def fake_compile_supervisor_graph() -> Any:
        events.append(("compile_enter",))
        try:
            yield DummySupervisor()
        finally:
            events.append(("compile_exit",))

    monkeypatch.setattr(cli_support, "compile_supervisor_graph", fake_compile_supervisor_graph)

    result = await cli_support.run_pipeline(
        {"age": 42},
        "contract-thread",
        stream=True,
        console=Console(file=io.StringIO()),
    )

    assert result == {"report_json": {"ok": True}, "report_text": "done"}
    assert events == [
        ("compile_enter",),
        ("ainvoke", {"age": 42}, "contract-thread", 25),
        ("compile_exit",),
    ]


@pytest.mark.asyncio
async def test_supervisor_orchestrator_delegates_to_shared_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.supervisor as supervisor_module
    from agents.supervisor import SupervisorOrchestrator

    calls: list[tuple[Any, dict[str, Any], str, Any]] = []

    async def fake_run_tools_pipeline(
        orchestrator: Any,
        patient_profile: dict[str, Any],
        *,
        thread_id: str,
        memory: Any,
    ) -> dict[str, Any]:
        calls.append((orchestrator, patient_profile, thread_id, memory))
        return {"report_json": {"ok": True}, "report_text": "done"}

    monkeypatch.setattr(supervisor_module, "run_tools_pipeline", fake_run_tools_pipeline)

    orchestrator = SupervisorOrchestrator()
    memory = object()
    result = await orchestrator._run_tools_pipeline(
        {"age": 42}, thread_id="contract-thread", memory=memory
    )

    assert result == {"report_json": {"ok": True}, "report_text": "done"}
    assert calls == [(orchestrator, {"age": 42}, "contract-thread", memory)]
