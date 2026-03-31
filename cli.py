from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.supervisor import compile_supervisor_graph
from async_typer import AsyncTyper
from clinical_trials import search_trials
from config import settings
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON from {path}: {exc}") from exc


async def _with_memory() -> EpisodicMemory:
    memory = EpisodicMemory()
    await memory.init()
    return memory


@app.async_command("run")
async def run(profile_path: Path) -> None:
    """Run full supervisor pipeline for a patient profile JSON file."""
    try:
        patient_profile = _load_json(profile_path)
        async with compile_supervisor_graph() as supervisor:
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), console=console
            ) as progress:
                progress.add_task("Running supervisor pipeline...", total=None)
                result = await supervisor.ainvoke(
                    patient_profile,
                    thread_id=profile_path.stem,
                    recursion_limit=25,
                )
        report_text = result.get("report_text") or json.dumps(result, indent=2)
        console.print(Panel(report_text, title="Clinical Trial Match Report"))
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@app.async_command("search")
async def search(
    condition: str | None = None,
    intervention: str | None = None,
    page_size: int = 10,
) -> None:
    """Search ClinicalTrials.gov and display trials in a rich table."""
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Searching ClinicalTrials.gov...", total=None)
            result = await search_trials(
                condition=condition,
                intervention=intervention,
                page_size=page_size,
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


@memory_app.async_command("list")
async def memory_list() -> None:
    """List active episodic memory entries."""
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


@memory_app.async_command("purge")
async def memory_purge() -> None:
    """Purge expired episodic memory entries."""
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


@memory_app.async_command("invalidate")
async def memory_invalidate(profile_path: Path) -> None:
    """Invalidate cached episodic memory entry for a patient profile JSON file."""
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


@app.async_command("validate-env")
async def validate_env() -> None:
    """Run environment validation checks."""
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
        table.add_row("LLM_PROVIDER", env.llm_provider)
        table.add_row("GEMINI_READY", str(env.gemini_ready))
        table.add_row("OPENAI_READY", str(env.openai_ready))
        table.add_row("MAX_TRIALS_PER_QUERY", str(settings.max_trials_per_query))
        console.print(table)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
