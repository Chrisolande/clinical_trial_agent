from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import typer
from agents.patient_parser import parse_patient_profile
from agents.supervisor import compile_supervisor_graph
from async_typer import AsyncTyper
from clinical_trials import search_trials
from config import get_settings
from logging_config import configure_logging
from memory import EpisodicMemory
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from validate_env import validate_or_raise

app = AsyncTyper(help="Clinical Trial Agent CLI")
memory_app = AsyncTyper(help="Episodic memory operations")
app.add_typer(memory_app, name="memory")
console = Console()
MAX_PROFILE_BYTES = 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    if size_bytes > MAX_PROFILE_BYTES:
        raise typer.BadParameter(f"Profile JSON exceeds 1 MB limit: {size_bytes} bytes")
    try:
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON from {path}: {exc}") from exc


async def _with_memory() -> EpisodicMemory:
    memory = EpisodicMemory()
    await memory.init()
    return memory


def _load_text_profile(path: Path) -> str:
    size_bytes = path.stat().st_size
    if size_bytes > MAX_PROFILE_BYTES:
        raise typer.BadParameter(f"Profile text exceeds 1 MB limit: {size_bytes} bytes")
    return path.read_text(encoding="utf-8")


async def _run_pipeline(
    patient_profile: dict[str, Any], thread_id: str, *, stream: bool
) -> dict[str, Any]:
    async with compile_supervisor_graph() as supervisor:
        if (
            stream
            and hasattr(supervisor, "_react_agent")
            and hasattr(supervisor._react_agent, "astream_events")
        ):
            async for evt in supervisor._react_agent.astream_events(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Match this patient to trials by calling tools in order: "
                                "run_retrieval -> run_eligibility -> run_synthesis."
                            ),
                        }
                    ]
                },
                config={"configurable": {"thread_id": thread_id}},
                version="v1",
            ):
                event_name = str(evt.get("event", ""))
                if event_name:
                    console.print(f"[cyan]event:[/cyan] {event_name}")
        return cast(
            "dict[str, Any]",
            await supervisor.ainvoke(patient_profile, thread_id=thread_id, recursion_limit=25),
        )


@app.async_command("run", help="Run full supervisor pipeline for a patient profile input file.")
async def run(
    profile_path: Path,
    input_format: str = typer.Option("json", "--input-format", help="json or text"),
    output_format: str = typer.Option("text", "--output-format", help="text or json"),
    stream: bool = typer.Option(False, "--stream", help="Stream intermediate events"),
    webhook_url: str | None = typer.Option(
        None, "--webhook-url", help="Optional webhook callback URL"
    ),
    log_format: str = typer.Option("text", "--log-format", help="text or json"),
) -> None:
    configure_logging(log_format=log_format)
    try:
        if input_format == "text":
            raw_text = _load_text_profile(profile_path)
            patient_profile = await parse_patient_profile(raw_text)
        else:
            patient_profile = _load_json(profile_path)

        thread_id = profile_path.stem
        result = await _run_pipeline(patient_profile, thread_id, stream=stream)

        if webhook_url:
            payload = {
                "run_id": thread_id,
                "profile_hash": result.get("profile_hash", ""),
                "outcome_summary": {
                    "report_text": result.get("report_text", ""),
                    "has_report_json": isinstance(result.get("report_json"), dict),
                },
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()
            console.print(f"Queued run with webhook delivery. run_id={thread_id}")
            return

        if output_format == "json":
            console.print(json.dumps(result.get("report_json") or result, indent=2, default=str))
        else:
            report_text = result.get("report_text") or json.dumps(result, indent=2, default=str)
            console.print(Panel(report_text, title="Clinical Trial Match Report"))
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@app.async_command("search", help="Search ClinicalTrials.gov and display trials in a rich table.")
async def search(
    condition: str | None = None, intervention: str | None = None, page_size: int = 10
) -> None:
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Searching ClinicalTrials.gov...", total=None)
            result = await search_trials(
                condition=condition, intervention=intervention, page_size=page_size
            )
        studies = result.get("studies", [])
        table = Table(title="ClinicalTrials.gov Search Results")
        table.add_column("NCT ID")
        table.add_column("Brief Title")
        table.add_column("Status")
        for study in studies:
            table.add_row(
                str(study.get("nct_id", "")),
                str(study.get("brief_title", ""))[:80],
                str(study.get("overall_status", "")),
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@memory_app.async_command("list", help="List active episodic memory entries.")
async def memory_list() -> None:
    memory = await _with_memory()
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Loading memory entries...", total=None)
            rows = await memory.list_runs()
        table = Table(title="Active Episodic Memory Entries")
        table.add_column("Profile Hash")
        table.add_column("Created At")
        table.add_column("Expires At")
        for row in rows:
            table.add_row(row["profile_hash"], row["created_at"], row["expires_at"])
        console.print(table)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


@memory_app.async_command("purge", help="Purge expired episodic memory entries.")
async def memory_purge() -> None:
    memory = await _with_memory()
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Purging expired memory entries...", total=None)
            purged = await memory.purge_expired()
        console.print(f"Purged entries: {purged}")
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


@memory_app.async_command(
    "invalidate", help="Invalidate cached episodic memory for a patient profile JSON file."
)
async def memory_invalidate(profile_path: Path) -> None:
    memory = await _with_memory()
    try:
        patient_profile = _load_json(profile_path)
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Invalidating cached profile...", total=None)
            removed = await memory.invalidate(patient_profile)
        console.print(f"Invalidated: {removed}")
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


@app.async_command("validate-env", help="Run environment validation checks.")
async def validate_env() -> None:
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Validating environment...", total=None)
            env = validate_or_raise()
        table = Table(title="Environment Status")
        table.add_column("Setting")
        table.add_column("Value")
        table.add_row("DATABASE_URI", env.database_uri)
        table.add_row("MEMORY_DB_DSN", env.memory_db_dsn)
        table.add_row("DEEPSEEK_READY", str(env.deepseek_ready))
        table.add_row("LLM_CALL_TIMEOUT_SECONDS", str(get_settings().llm_call_timeout_seconds))
        table.add_row(
            "RETRIEVAL_INTERNAL_MAX_RETRIES", str(get_settings().retrieval_internal_max_retries)
        )
        table.add_row("MAX_TRIALS_FOR_ELIGIBILITY", str(get_settings().max_trials_for_eligibility))
        table.add_row("MAX_TRIALS_PER_QUERY", str(get_settings().max_trials_per_query))
        table.add_row(
            "TAVILY_ENABLE_CTGOV_SUPPLEMENT", str(get_settings().tavily_enable_ctgov_supplement)
        )
        table.add_row("TAVILY_MAX_RESULTS", str(get_settings().tavily_max_results))
        table.add_row(
            "TAVILY_MAX_TRIALS_TO_ENRICH", str(get_settings().tavily_max_trials_to_enrich)
        )
        console.print(table)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@app.async_command("erase-profile", help="Erase all stored records for a profile hash.")
async def erase_profile(hash: str = typer.Option(..., "--hash")) -> None:
    memory = await _with_memory()
    try:
        await memory.erase_profile(hash)
        console.print(f"Erased records for profile hash: {hash}")
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


@app.async_command("feedback", help="Store physician feedback for a trial verdict.")
async def feedback(
    profile_path: Path = typer.Option(..., "--profile"),
    run_id: str = typer.Option(..., "--run-id"),
    nct_id: str = typer.Option(..., "--nct-id"),
    verdict: str = typer.Option(..., "--verdict"),
    note: str = typer.Option(..., "--note"),
) -> None:
    if verdict not in {"confirmed", "rejected"}:
        raise typer.BadParameter("--verdict must be confirmed or rejected")

    memory = await _with_memory()
    try:
        patient_profile = _load_json(profile_path)
        await memory.save_feedback(patient_profile, run_id, nct_id, verdict, note)
        console.print("Feedback saved")
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


configure_logging()
