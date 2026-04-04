# Clinical Trial Agent

[![License](https://img.shields.io/github/license/Chrisolande/clinical_trial_agent)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Chrisolande/clinical_trial_agent/ci-cd.yml?branch=main)](https://github.com/Chrisolande/clinical_trial_agent/actions)
[![codecov](https://img.shields.io/codecov/c/github/Chrisolande/clinical_trial_agent)](https://codecov.io/gh/Chrisolande/clinical_trial_agent)

Async multi-agent clinical trial matching built with **LangGraph**.

The pipeline takes a patient profile, retrieves candidate studies from ClinicalTrials.gov, evaluates eligibility, and synthesizes a ranked report for clinical review.

> [!WARNING]
> This project is **not certified for real patient care workflows** without additional compliance, legal, and security review.

## What it does

- Retrieves trials from ClinicalTrials.gov
- Scores trial-patient fit (`strong`, `moderate`, `weak`, `disqualified`)
- Produces human-readable and JSON outputs
- Persists episodic memory and cache data in PostgreSQL

## Architecture

```mermaid
flowchart TD
    CLI[CLI / LangGraph App] --> SUP[Supervisor]
    SUP --> RET[Retrieval Subgraph]
    SUP --> ELI[Eligibility Subgraph]
    SUP --> SYN[Synthesis Subgraph]

    RET --> CTGOV[ClinicalTrials.gov API]
    ELI --> LLM[LLM Provider]
    SUP --> DB[(PostgreSQL Memory + Cache)]
    SYN --> OUT[Report JSON + Text]
```

See `docs/architecture.mermaid` for the maintained source diagram.

## Requirements

- Python 3.13+
- PostgreSQL 16+
- `uv` (recommended)

## Quick start

```bash
git clone https://github.com/Chrisolande/clinical_trial_agent.git
cd clinical_trial_agent
docker compose up db
uv sync
```

Validate environment before first run:

```bash
uv run python validate_env.py
# or
uv run clinical-trial-agent validate-env
```

## Configuration

Minimum required environment variables:

```bash
DEEPSEEK_API_KEY=your-key
DATABASE_URI=postgresql://postgres:postgres@localhost:5432/postgres
MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5432/postgres
PROFILE_HASH_SALT=your-salt
DB_ENCRYPTION_KEY=base64-32-byte-key
```

> [!IMPORTANT]
> `DATABASE_URI` and `MEMORY_DB_DSN` should point to the same database in default local setup.

## CLI usage

```bash
# Full pipeline
uv run clinical-trial-agent run ./patient_profile.json

# Search only
uv run clinical-trial-agent search --condition "non-small cell lung cancer"

# Memory operations
uv run clinical-trial-agent memory list
uv run clinical-trial-agent memory purge
uv run clinical-trial-agent memory invalidate ./patient_profile.json
```

## Development workflow

```bash
make check
```

`make check` runs linting, type-checking, tests, security checks, and complexity checks.

Current repository coverage gate is **80%+**.

## LangGraph dev server

```bash
uv run langgraph dev --config langgraph.json --no-browser
```

Exported graphs:

- `clinical_trial_agent`
- `retrieval`
- `eligibility`
- `synthesis`

## Data flow note

If Tavily enrichment is enabled, trial NCT IDs and titles may be sent to Tavily APIs. Review retention and compliance requirements before enabling this in sensitive environments.

## Contributing

Contributor process, branch conventions, lockfile workflow, and local quality expectations are documented in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Security

Vulnerability reporting process, consent/data-flow expectations, and clinical-data safety boundaries are documented in **[SECURITY.md](SECURITY.md)**.

## Project layout

```text
agents/         Core orchestration and reasoning modules
subagents/      retrieval/, eligibility/, synthesis/ graphs and nodes
models/         Pydantic models
tools/          Cache, retry, DB, validator, telemetry utilities
prompts/        Prompt templates
tests/          Unit + integration test suite
cli.py          Typer CLI entrypoint
memory.py       Episodic memory persistence
```
