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
- A configured LLM provider key for the provider you use

## Quick Start

```bash
git clone https://github.com/Chrisolande/clinical_trial_agent.git
cd clinical_trial_agent
uv sync --locked --all-groups
docker compose up -d db
```

Download the required NLTK data:

```bash
uv run python -m nltk.downloader stopwords
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
DATABASE_URI=postgresql://postgres:postgres@localhost:5433/postgres
MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5433/postgres
PROFILE_HASH_SALT=your-salt
DB_ENCRYPTION_KEY=base64-encoded-32-byte-key
TENANT_ID=your-tenant
FACILITY_ID=your-facility
CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true
LLM_PRIVACY_MODE=full_consent
CTGOV_TRANSPORT_MODE=get
```

Optional retrieval settings:

```bash
CTGOV_PROXY_URL=http://localhost:8000/ctgov/search
```

> [!IMPORTANT]
> `DATABASE_URI` and `MEMORY_DB_DSN` should point to the same database in the default local setup.

> [!TIP]
> The ClinicalTrials.gov proxy is **auto-started** by the CLI when `CTGOV_PROXY_URL` is set but the proxy is not reachable (see [ClinicalTrials.gov Proxy](#clinicaltrialsgov-proxy) below). You do not need to start it manually.

## Run The Default Synthetic Profile

`patient_profile.json` is a synthetic fixture. The CLI uses the same
fail-closed clinical-data controls as a real run, so local execution needs
PostgreSQL and a configured LLM provider.

Start PostgreSQL from Compose:

```bash
docker compose up -d db
docker compose ps
```

The compose file maps PostgreSQL to host port `5433`, so use that port in local
CLI environment variables.

Run the default profile:

```bash
export DATABASE_URI=postgresql://postgres:postgres@localhost:5433/postgres
export MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5433/postgres
export PROFILE_HASH_SALT=local-dev-salt
export DB_ENCRYPTION_KEY=MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=
export TENANT_ID=local-tenant
export FACILITY_ID=local-facility
export TAVILY_ENABLE_CTGOV_SUPPLEMENT=false

export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your-deepseek-key
export LLM_PRIVACY_MODE=full_consent
export CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true

uv run clinical-trial-agent run ./patient_profile.json
```

The CLI will **automatically start** the ClinicalTrials.gov proxy if it is not
already running. You do not need to start it in a separate terminal.

Expected behavior:

- The first run starts the proxy, retrieves trials, initializes PostgreSQL
  memory, and prints a clinical review report.
- A later run with the same tenant/facility/salt can be served from episodic
  memory.
- If the LLM key is invalid, the pipeline should still fail closed and produce a
  conservative fallback report rather than a high-confidence match.
- The proxy subprocess is terminated automatically when the CLI exits.

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
- External LLM prompts are controlled by `LLM_PRIVACY_MODE`: `blocked`,
  `deidentified`, `full_consent`, or `local_only`
- QA checks can block synthesis on critical inconsistencies
- Medication safety rules disqualify trials with known contraindications
- Tenant and facility defaults are rejected in runtime checks and CI
- Secret scanning is enforced in CI with a tracked baseline
- Webhook callbacks require HTTPS by default, block local/private targets, and
  can be HMAC signed with `WEBHOOK_SIGNING_SECRET`

Criteria provenance is included in trial records and reports. Official
ClinicalTrials.gov criteria are verified/full; snippet-only criteria are
unverified/partial and cannot produce a `strong` tier.

Physician feedback is tenant/facility scoped. Confirmed feedback slightly boosts
the same scoped profile/trial in ranking, while rejected feedback slightly
penalizes it.

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

Run offline clinical evaluation with synthetic fixtures:

```bash
make eval
```

The reports are written to `evaluation/reports/latest_eval_report.json` and
`evaluation/reports/latest_eval_report.md`. See `evaluation/README.md` for
metric definitions, thresholds, prompt-injection checks, and calibration
interpretation.

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

## ClinicalTrials.gov Proxy

The agent retrieves clinical trial data through a lightweight PHI-safe proxy
(`ctgov_proxy.py`). The proxy is **auto-managed** by the CLI:

- If `CTGOV_PROXY_URL` is set (e.g. `http://localhost:8000/ctgov/search`), the
  CLI probes its `/healthz` endpoint before each run. If unreachable, it
  automatically spawns `uvicorn ctgov_proxy:app` on the configured host/port.
- If `CTGOV_PROXY_URL` is **not set**, the CLI defaults to
  `http://127.0.0.1:8321/ctgov/search` and starts the proxy there.
- The proxy subprocess is terminated via `atexit` when the CLI exits.

To start the proxy manually (e.g. for development or shared use):

```bash
uv run uvicorn ctgov_proxy:app --host 127.0.0.1 --port 8000
```

If the proxy is already running when the CLI starts, it is reused as-is.

## Project Layout

```text
agents/                    Orchestration and reasoning modules
subagents/                 Retrieval, eligibility, and synthesis graphs
models/                    Pydantic models
prompts/                   Prompt templates
tools/                     Cache, retry, DB, validation, and telemetry helpers
clinical_trial_agent/      CLI, config, memory, and pipeline entrypoints
ctgov_proxy.py             Minimal PHI-safe proxy for ClinicalTrials.gov
tests/                     Unit and integration tests
```

For runbooks and operational commands, see [`docs/runbook.md`](docs/runbook.md) and [`docs/ops_playbook.md`](docs/ops_playbook.md).
