import hashlib
import hmac
import json
import os
import warnings
from pathlib import Path
from typing import Literal

import httpx
import typer
from async_typer import AsyncTyper
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from tools.cli_support import (
    load_json as _load_json,
)
from tools.cli_support import (
    load_text_profile as _load_text_profile,
)
from tools.cli_support import (
    parse_profile_text as _parse_profile_text,
)
from tools.cli_support import (
    run_pipeline as _run_pipeline,
)
from tools.cli_support import (
    validate_webhook_url as _validate_webhook_url,
)
from tools.cli_support import (
    with_memory as _with_memory,
)
from tools.ctgov_proxy_manager import ensure_proxy as _ensure_proxy
from tools.errors import ClinicalTrialAgentError

from clinical_trial_agent.clinical_trials import search_trials
from clinical_trial_agent.config import get_settings
from clinical_trial_agent.logging_config import configure_logging
from clinical_trial_agent.validate_env import validate_or_raise_async

warnings.filterwarnings("ignore", category=SyntaxWarning)
app = AsyncTyper(help="Clinical Trial Agent CLI")
memory_app = AsyncTyper(help="Episodic memory operations")
app.add_typer(memory_app, name="memory")
console = Console()

HANDLED_EXCEPTIONS = (
    ClinicalTrialAgentError,
    RuntimeError,
    ValueError,
    typer.BadParameter,
    httpx.HTTPError,
)


@app.async_command("run", help="Run full supervisor pipeline for a patient profile input file.")
async def run(
    profile_path: Path,
    input_format: Literal["json", "text"] = typer.Option(
        "json", "--input-format", help="json or text"
    ),
    output_format: Literal["text", "json"] = typer.Option(
        "text", "--output-format", help="text or json"
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream intermediate events"),
    webhook_url: str | None = typer.Option(
        None, "--webhook-url", help="Optional webhook callback URL"
    ),
    allow_local_webhook: bool = typer.Option(
        False,
        "--allow-local-webhook",
        help="Allow local/private webhook URLs for development only",
    ),
    log_format: str = typer.Option("text", "--log-format", help="text or json"),
) -> None:
    configure_logging(log_format=log_format)
    try:
        await _ensure_proxy()
        if input_format == "text":
            patient_profile = await _parse_profile_text(_load_text_profile(profile_path))
        else:
            patient_profile = _load_json(profile_path)

        thread_id = profile_path.stem
        result = await _run_pipeline(patient_profile, thread_id, stream=stream, console=console)

        if webhook_url:
            _validate_webhook_url(webhook_url, allow_local=allow_local_webhook)
            payload = {
                "run_id": thread_id,
                "profile_hash": result.get("profile_hash", ""),
                "outcome_summary": {
                    "report_text": result.get("report_text", ""),
                    "has_report_json": isinstance(result.get("report_json"), dict),
                },
            }
            body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
            headers: dict[str, str] = {}
            signing_secret = os.getenv("WEBHOOK_SIGNING_SECRET", "")
            if signing_secret:
                digest = hmac.new(
                    signing_secret.encode("utf-8"),
                    body,
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Clinical-Trial-Agent-Signature"] = f"sha256={digest}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    webhook_url,
                    content=body,
                    headers={**headers, "Content-Type": "application/json"},
                )
                response.raise_for_status()
            console.print(f"Queued run with webhook delivery. run_id={thread_id}")
            return

        if output_format == "json":
            console.print(json.dumps(result.get("report_json") or result, indent=2, default=str))
        else:
            report_text = result.get("report_text") or json.dumps(result, indent=2, default=str)
            console.print(Panel(report_text, title="Clinical Trial Match Report"))
    except HANDLED_EXCEPTIONS as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@app.async_command("search", help="Search ClinicalTrials.gov and display trials in a rich table.")
async def search(
    condition: str | None = typer.Option(None, "--condition", help="Condition/disease query"),
    intervention: str | None = typer.Option(None, "--intervention", help="Intervention query"),
    term: str | None = typer.Option(
        None, "--term", help="Free-text ClinicalTrials.gov query.term search"
    ),
    status: list[str] | None = typer.Option(
        None,
        "--status",
        help="ClinicalTrials.gov overall status filter. Can be repeated.",
    ),
    page_size: int = typer.Option(10, "--page-size", min=1, max=1000),
) -> None:
    try:
        await _ensure_proxy()
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), console=console
        ) as progress:
            progress.add_task("Searching ClinicalTrials.gov...", total=None)
            result = await search_trials(
                condition=condition,
                intervention=intervention,
                term=term,
                status=status,
                page_size=page_size,
            )
        studies = result.get("studies", [])
        error = result.get("error")
        if isinstance(error, dict):
            status = error.get("status_code")
            status_text = f" (status={status})" if status is not None else ""
            message = str(error.get("message", "unknown error"))
            console.print(
                f"[yellow]ClinicalTrials.gov search error{status_text}: {message}[/yellow]"
            )
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
    except HANDLED_EXCEPTIONS as exc:
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
    except HANDLED_EXCEPTIONS as exc:
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
    except HANDLED_EXCEPTIONS as exc:
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
    except HANDLED_EXCEPTIONS as exc:
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
            env = await validate_or_raise_async()
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
        console.print(table)
    except HANDLED_EXCEPTIONS as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


@app.async_command("erase-profile", help="Erase all stored records for a profile hash.")
async def erase_profile(hash: str = typer.Option(..., "--hash")) -> None:
    memory = await _with_memory()
    try:
        await memory.erase_profile(hash)
        console.print(f"Erased records for profile hash: {hash}")
    except HANDLED_EXCEPTIONS as exc:
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
    except HANDLED_EXCEPTIONS as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        await memory.close()


configure_logging()


if __name__ == "__main__":
    app()
