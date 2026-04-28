# Clinical Trial Agent

[![CI](https://img.shields.io/github/actions/workflow/status/Chrisolande/clinical_trial_agent/ci-cd.yml?branch=main)](https://github.com/Chrisolande/clinical_trial_agent/actions)
[![codecov](https://img.shields.io/codecov/c/github/Chrisolande/clinical_trial_agent)](https://codecov.io/gh/Chrisolande/clinical_trial_agent)
[![License](https://img.shields.io/github/license/Chrisolande/clinical_trial_agent)](LICENSE)

Async, multi-agent clinical trial matching built with **LangGraph**.

The pipeline takes a patient profile, retrieves candidate studies from ClinicalTrials.gov, evaluates eligibility, and produces a ranked report for clinical review. It also includes PostgreSQL-backed episodic memory, cached verdicts, and fail-closed safety controls for clinical-data workflows.

> [!WARNING]
> This project is **not certified for real patient care workflows**. Use it only after the appropriate clinical, legal, security, and compliance review.

## Highlights

- LangGraph orchestration with retrieval, eligibility, and synthesis subgraphs
- CLI for full runs, search-only queries, environment validation, and memory operations
- Ranked trial tiers: `strong`, `moderate`, `weak`, `disqualified`
- PostgreSQL-backed memory and cache layers
- Fail-closed consent, tenant, facility, and secret-scanning guardrails
- Rich text and JSON report output

## How It Works

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

The maintained architecture and data-flow notes live in [`docs/architecture.mermaid`](docs/architecture.mermaid).

## What You Can Do

- Search ClinicalTrials.gov from the CLI
- Run an end-to-end matching pipeline from a patient profile file
- Validate environment and fail-closed runtime settings
- Persist episodic memory, feedback, and cache entries in PostgreSQL
- Inspect the LangGraph app locally with the dev server

## Requirements

- Python 3.11 through 3.13
- `uv`
- PostgreSQL 16+ for memory and cache-backed workflows
- `nltk` stopwords corpus
- A configured LLM provider key for the provider you use

## Quick Start

```bash
git clone https://github.com/Chrisolande/clinical_trial_agent.git
cd clinical_trial_agent
uv sync --locked --group dev
docker compose up db
```

Run the environment check before the first pipeline execution:

```bash
uv run clinical-trial-agent validate-env
```

> [!TIP]
> The repo ships with a dev container, so VS Code Dev Containers and GitHub Codespaces can use the same `uv sync --group dev` setup.

## Configuration

Set the required environment variables before running the pipeline:

```bash
DEEPSEEK_API_KEY=your-key
DATABASE_URI=postgresql://postgres:postgres@localhost:5432/postgres
MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5432/postgres
PROFILE_HASH_SALT=your-salt
DB_ENCRYPTION_KEY=base64-encoded-32-byte-key
TENANT_ID=your-tenant
FACILITY_ID=your-facility
CLINICAL_DATA_EXTERNAL_LLM_CONSENT=false
CTGOV_TRANSPORT_MODE=get
```

Optional retrieval settings:

```bash
CTGOV_PROXY_URL=https://your-internal-proxy/ctgov/search
```

> [!IMPORTANT]
> `DATABASE_URI` and `MEMORY_DB_DSN` should point to the same database in the default local setup.

## Usage

### Full pipeline

```bash
uv run clinical-trial-agent run ./patient_profile.json
```

### Search ClinicalTrials.gov

```bash
uv run clinical-trial-agent search --condition "non-small cell lung cancer"
```

### Validate the environment

```bash
uv run clinical-trial-agent validate-env
```

### Inspect episodic memory

```bash
uv run clinical-trial-agent memory list
uv run clinical-trial-agent memory purge
uv run clinical-trial-agent memory invalidate ./patient_profile.json
```

## Output

The pipeline produces:

- A ranked report in text form
- A structured JSON report
- Trial tiers and eligibility verdict details
- Information gaps and QA issues
- Retrieval errors and remediation context when applicable

## Safety and Guardrails

- PHI-bearing retrieval uses a fail-closed transport policy
- External LLM calls require explicit consent when PHI is involved
- QA checks can block synthesis on critical inconsistencies
- Medication safety rules disqualify trials with known contraindications
- Tenant and facility defaults are rejected in runtime checks and CI
- Secret scanning is enforced in CI with a tracked baseline

## Development

```bash
make check
```

This runs linting, type checking, tests, security checks, and complexity checks.

Recommended local workflow:

```bash
uv sync --locked --group dev
uv run clinical-trial-agent validate-env
uv run make check
uv run clinical-trial-agent run ./patient_profile.json
```

Coverage is gated at **75%** in CI.

## Docker and Dev Containers

- `docker-compose.yml` starts the application and PostgreSQL locally
- `.devcontainer/devcontainer.json` boots the same `uv`-based dev environment
- The Docker image build uses the locked dependency set from `uv.lock`

## LangGraph Dev Server

```bash
uv run langgraph dev --config langgraph.json --no-browser
```

Exported graphs:

- `clinical_trial_agent`
- `retrieval`
- `eligibility`
- `synthesis`

## Project Layout

```text
agents/                    Orchestration and reasoning modules
subagents/                 Retrieval, eligibility, and synthesis graphs
models/                    Pydantic models
prompts/                   Prompt templates
tools/                     Cache, retry, DB, validation, and telemetry helpers
clinical_trial_agent/      CLI, config, memory, and pipeline entrypoints
tests/                     Unit and integration tests
```

For runbooks and operational commands, see [`docs/runbook.md`](docs/runbook.md) and [`docs/ops_playbook.md`](docs/ops_playbook.md).
