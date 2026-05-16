# TrialMatch Agent

[![CI](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/ci.yml)
[![Release](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/release.yml)
[![Weekly](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/weekly.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/weekly.yml)
[![codecov](https://codecov.io/gh/Chrisolande/TrialMatch-Agent/branch/main/graph/badge.svg)](https://codecov.io/gh/Chrisolande/TrialMatch-Agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

Async, multi-agent clinical trial matching built with **LangGraph**.

The pipeline takes a patient profile, retrieves candidate studies from ClinicalTrials.gov, assesses eligibility, and produces a ranked report for clinical review. It also includes PostgreSQL-backed episodic memory, cached verdicts, and fail-closed safety controls for clinical-data workflows.

> [!WARNING]
> This project is **not certified for real patient care workflows**. Use it only after the appropriate clinical, legal, security, and compliance review.

## Highlights

- LangGraph orchestration with retrieval, eligibility, and synthesis subgraphs
- CLI for full runs, search-only queries, environment validation, feedback, and memory operations
- Ranked trial tiers: `strong`, `moderate`, `weak`, `disqualified`
- PostgreSQL-backed memory and cache layers
- Fail-closed consent, tenant, facility, and secret-scanning guardrails
- Rich text and JSON report output
- Docker, Compose, LangGraph dev server, and CI/CD workflow support

## How It Works

```mermaid
flowchart TD
    CLI[CLI / LangGraph App] --> SUP[Supervisor]
    SUP --> RET[Retrieval Subgraph]
    SUP --> ELI[Eligibility Subgraph]
    SUP --> SYN[Synthesis Subgraph]

    RET --> CTGOV[ClinicalTrials.gov API / Proxy]
    ELI --> LLM[Configured LLM Provider]
    SUP --> DB[(PostgreSQL Memory + Cache)]
    SUP --> CHK[(LangGraph Checkpointer)]
    SYN --> OUT[Report JSON + Text]
```

The maintained architecture and data-flow notes live in [`docs/architecture.mermaid`](docs/architecture.mermaid).

## What You Can Do

- Search ClinicalTrials.gov from the CLI
- Run an end-to-end matching pipeline from a patient profile file
- Validate environment and fail-closed runtime settings
- Persist episodic memory, feedback, and cache entries in PostgreSQL
- Generate structured reports for clinical review
- Inspect the LangGraph app locally with the dev server

## Requirements

- Python `>=3.11,<3.13`
- `uv`
- Docker and Docker Compose
- PostgreSQL 16+ for memory and cache-backed workflows
- A configured LLM provider key for the provider you use

> [!NOTE]
> `patient_profile.json` is a synthetic fixture. Never replace it with real patient data.

## Quick Start

```bash
git clone https://github.com/Chrisolande/TrialMatch-Agent.git
cd TrialMatch-Agent

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
> The repo ships with a dev container, so VS Code Dev Containers and GitHub Codespaces can use the same `uv`-based setup.

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
TAVILY_ENABLE_CTGOV_SUPPLEMENT=false
```

> [!IMPORTANT]
> `DATABASE_URI` and `MEMORY_DB_DSN` should point to the same database in the default local setup.

> [!CAUTION]
> Keep `LLM_PRIVACY_MODE=blocked` or `deidentified` unless you have explicitly reviewed consent and privacy requirements.

## Run The Default Synthetic Profile

Start PostgreSQL from Compose:

```bash
docker compose up -d db
docker compose ps
```

The compose file maps PostgreSQL to host port `5433`, so use that port in local CLI environment variables.

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

The CLI will automatically start the ClinicalTrials.gov proxy if it is not already running.

Expected behavior:

- The first run starts the proxy, retrieves trials, initializes PostgreSQL memory, and prints a clinical review report.
- Later runs with the same tenant, facility, and salt can be served from episodic memory.
- If the LLM key is invalid, the pipeline should fail closed and produce a conservative fallback report.
- The proxy subprocess is terminated automatically when the CLI exits.

## Usage

### Full pipeline

```bash
uv run clinical-trial-agent run ./patient_profile.json
```

### JSON output

```bash
uv run clinical-trial-agent run ./patient_profile.json --output-format json
```

### Text profile input

```bash
uv run clinical-trial-agent run ./profile.txt --input-format text
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

### Store clinician feedback

```bash
uv run clinical-trial-agent feedback \
  --profile ./patient_profile.json \
  --run-id patient_profile \
  --nct-id NCT00000000 \
  --verdict confirmed \
  --note "Clinician reviewed and confirmed."
```

## Output

The pipeline produces:

- A ranked report in text form
- A structured JSON report
- Trial tiers and eligibility verdict details
- Information gaps and QA issues
- Retrieval errors and remediation context when applicable

Example match tiers:

| Tier | Meaning |
|---|---|
| `strong` | Good apparent match, based on available verified criteria |
| `moderate` | Plausible match with some uncertainty or missing information |
| `weak` | Low-confidence match or limited criteria support |
| `disqualified` | Clear conflict with eligibility criteria |

## Safety and Guardrails

- PHI-bearing retrieval uses a fail-closed transport policy
- External LLM prompts are controlled by `LLM_PRIVACY_MODE`: `blocked`, `deidentified`, `full_consent`, or `local_only`
- QA checks can block synthesis on critical inconsistencies
- Medication safety rules disqualify trials with known contraindications
- Tenant and facility defaults are rejected in runtime checks and CI
- Secret scanning is enforced in CI with a tracked baseline
- Webhook callbacks require HTTPS by default, block local/private targets, and can be HMAC signed with `WEBHOOK_SIGNING_SECRET`

Criteria provenance is included in trial records and reports. Official ClinicalTrials.gov criteria are verified/full; snippet-only criteria are unverified/partial and cannot produce a `strong` tier.

Physician feedback is tenant/facility scoped. Confirmed feedback slightly boosts the same scoped profile/trial in ranking, while rejected feedback slightly penalizes it.

> [!IMPORTANT]
> A `strong` match should still be treated as a candidate for human review, not as a clinical recommendation.

## Development

Run the full local quality gate:

```bash
make check
```

This runs linting, type checking, tests, security checks, and complexity checks.

Recommended local workflow:

```bash
uv sync --locked --all-groups
uv run clinical-trial-agent validate-env
make check
uv run clinical-trial-agent run ./patient_profile.json
```

Run individual checks:

```bash
make lint
make typecheck
make test
make security
make complexity
```

Auto-format and fix lint issues:

```bash
make fix
```

Run tests with coverage:

```bash
uv run pytest \
  --cov=clinical_trial_agent \
  --cov=agents \
  --cov=subagents \
  --cov=tools \
  --cov=models \
  --cov-report=term-missing \
  --cov-fail-under=75
```

Coverage is gated at **75%** in the project configuration.

## Docker and Dev Containers

Build the image:

```bash
docker build -t clinical-trial-agent:local .
```

Run the image help command:

```bash
docker run --rm clinical-trial-agent:local --help
```

Start the local Compose stack:

```bash
docker compose up -d --build
```

Check services:

```bash
docker compose ps
```

Stop services:

```bash
docker compose down --remove-orphans
```

The Docker image build uses the locked dependency set from `uv.lock`.

## CI/CD

The repository uses separate workflow files:

```text
.github/workflows/
  ci.yml        # lint, tests, security checks, Docker smoke test, SBOM
  release.yml  # Docker image publishing
  weekly.yml   # scheduled integration and weekly image rebuild
```

The CI pipeline checks:

- Ruff linting and formatting
- Mypy typing
- Pytest coverage
- Bandit security checks
- Radon complexity
- detect-secrets baseline
- Docker image build
- Docker smoke test
- SBOM generation

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

The agent retrieves clinical trial data through a lightweight PHI-safe proxy (`ctgov_proxy.py`). The proxy is auto-managed by the CLI:

- If `CTGOV_PROXY_URL` is set, the CLI probes its `/healthz` endpoint before each run. If unreachable, it automatically spawns `uvicorn ctgov_proxy:app` on the configured host and port.
- If `CTGOV_PROXY_URL` is not set, the CLI defaults to `http://127.0.0.1:8321/ctgov/search` and starts the proxy there.
- The proxy subprocess is terminated via `atexit` when the CLI exits.

To start the proxy manually:

```bash
uv run uvicorn ctgov_proxy:app --host 127.0.0.1 --port 8000
```

If the proxy is already running when the CLI starts, it is reused as-is.

## Project Layout

```text
clinical_trial_agent/      CLI, config, memory, validation, and LangGraph app
agents/                    Orchestration plus focused helper submodules
subagents/                 Retrieval, eligibility, and synthesis graphs + node helpers
models/                    Pydantic domain models
prompts/                   Prompt templates
tools/                     Cache, retry, DB, validation, telemetry, and parsing helpers
docs/                      Architecture, runbook, and operations notes
tests/                     Unit and integration tests
ctgov_proxy.py             Minimal PHI-safe proxy for ClinicalTrials.gov
patient_profile.json       Synthetic test profile
langgraph.json             LangGraph dev server configuration
docker-compose.yml         Local PostgreSQL and app stack
Dockerfile                 Container build
Makefile                   Local developer commands
```

## Useful Docs

- [`docs/runbook.md`](docs/runbook.md)
- [`docs/ops_playbook.md`](docs/ops_playbook.md)
- [`docs/architecture.mermaid`](docs/architecture.mermaid)

## Roadmap Ideas

- Add `.env.example`
- Add `examples/sample_report.json`
- Add an evaluation harness with fixed synthetic profiles and expected tiers
- Add retrieval quality metrics
- Add report-quality regression tests
- Add provider-specific integration tests
- Add a lightweight demo screenshot or terminal output
- Add latency and cost benchmarking per run
