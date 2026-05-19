# TrialMatch Agent

[![CI](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/ci.yml)
[![Release](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/release.yml)
[![Weekly](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/weekly.yml/badge.svg?branch=main)](https://github.com/Chrisolande/TrialMatch-Agent/actions/workflows/weekly.yml)
[![codecov](https://codecov.io/gh/Chrisolande/TrialMatch-Agent/branch/main/graph/badge.svg)](https://codecov.io/gh/Chrisolande/TrialMatch-Agent)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

Async, multi-agent clinical trial matching with LangGraph, ClinicalTrials.gov retrieval, eligibility reasoning, PostgreSQL-backed memory, and clinician-facing reports.

> [!WARNING]
> TrialMatch Agent is not certified for real patient care workflows. Use it only after appropriate clinical, legal, security, and compliance review.

## Overview

TrialMatch Agent takes a patient profile, retrieves candidate studies from ClinicalTrials.gov, evaluates trial eligibility criteria, ranks matches, and produces a structured clinical review report.

The pipeline is designed around explicit safety boundaries:

- fail-closed privacy and consent checks before LLM calls
- tenant and facility scoped memory
- criteria provenance tracking
- conservative fallbacks for uncertain eligibility
- QA checks before report finalization

## Features

- LangGraph supervisor with retrieval, eligibility, and synthesis subgraphs
- CLI for full pipeline runs, search-only queries, validation, memory, and feedback
- ClinicalTrials.gov search through a local PHI-safe proxy
- Ranked match tiers: `strong`, `moderate`, `weak`, `disqualified`
- PostgreSQL-backed episodic memory, feedback, and cache workflows
- Structured report planning with Pydantic validation
- Text and JSON report output
- Docker Compose stack for local PostgreSQL and app execution
- CI checks for linting, typing, tests, coverage, security, complexity, and Docker smoke tests

## Architecture

```mermaid
flowchart TD
    CLI[CLI] --> SUP[Supervisor]
    SUP --> RET[Retrieval Subgraph]
    SUP --> ELI[Eligibility Subgraph]
    SUP --> SYN[Synthesis Subgraph]

    RET --> CTGOV[ClinicalTrials.gov Proxy/API]
    ELI --> LLM[Configured LLM Provider]
    SYN --> REPORT[Report JSON + Text]
    SUP --> DB[(PostgreSQL Memory + Cache)]
    SUP --> CHK[(LangGraph Checkpointer)]
```

Canonical runtime path:

```text
clinical-trial-agent run -> compile_supervisor_graph -> SupervisorOrchestrator -> run_tools_pipeline
```

The LangGraph dev exports are useful for inspection, but they are not production-equivalent because they do not include the full supervisor memory, audit, feedback, and retry behavior.

## Prerequisites

- Python `>=3.11,<3.13`
- `uv`
- Docker and Docker Compose
- PostgreSQL 16+ for memory and cache-backed workflows
- An LLM provider configured through environment variables

Supported LLM providers are configured in `clinical_trial_agent/config.py`:

- `deepseek`
- `openai`
- `anthropic`
- `ollama`

> [!NOTE]
> `patient_profile.json` is a synthetic fixture. Do not replace it with real patient data.

## Getting Started

Clone and install dependencies:

```bash
git clone https://github.com/Chrisolande/TrialMatch-Agent.git
cd TrialMatch-Agent

uv sync --locked --all-groups
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` with local values. For a Docker Compose database, use host port `5433`:

```dotenv
DATABASE_URI=postgresql://postgres:postgres@localhost:5433/postgres
MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5433/postgres
PROFILE_HASH_SALT=local-dev-salt
DB_ENCRYPTION_KEY=MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=
TENANT_ID=local-tenant
FACILITY_ID=local-facility
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key
LLM_PRIVACY_MODE=full_consent
CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true
```

Start PostgreSQL:

```bash
docker compose up -d db
```

Download required NLTK data:

```bash
uv run python -m nltk.downloader stopwords
```

Validate the environment:

```bash
uv run clinical-trial-agent validate-env
```

Run the synthetic profile:

```bash
uv run clinical-trial-agent run ./patient_profile.json
```

The CLI starts the ClinicalTrials.gov proxy automatically when needed.

## Usage

Run the full pipeline:

```bash
uv run clinical-trial-agent run ./patient_profile.json
```

Return JSON output:

```bash
uv run clinical-trial-agent run ./patient_profile.json --output-format json
```

Use text profile input:

```bash
uv run clinical-trial-agent run ./profile.txt --input-format text
```

Stream intermediate events:

```bash
uv run clinical-trial-agent run ./patient_profile.json --stream
```

Search ClinicalTrials.gov:

```bash
uv run clinical-trial-agent search \
  --condition "non-small cell lung cancer" \
  --status RECRUITING \
  --page-size 10
```

Validate configuration:

```bash
uv run clinical-trial-agent validate-env
```

Manage episodic memory:

```bash
uv run clinical-trial-agent memory list
uv run clinical-trial-agent memory purge
uv run clinical-trial-agent memory invalidate ./patient_profile.json
```

Erase records for a profile hash:

```bash
uv run clinical-trial-agent erase-profile --hash <profile-hash>
```

Store clinician feedback:

```bash
uv run clinical-trial-agent feedback \
  --profile ./patient_profile.json \
  --run-id patient_profile \
  --nct-id NCT00000000 \
  --verdict confirmed \
  --note "Clinician reviewed and confirmed."
```

Send a webhook callback:

```bash
uv run clinical-trial-agent run ./patient_profile.json \
  --webhook-url https://example.com/trialmatch/callback
```

If `WEBHOOK_SIGNING_SECRET` is set, callbacks include an HMAC signature header.

## Output

The pipeline returns:

- ranked trial cards
- eligibility verdicts and criterion-level reasoning
- information gaps
- recommended next actions
- QA and remediation context
- text report and structured JSON report

Match tiers:

| Tier | Meaning |
| --- | --- |
| `strong` | Good apparent fit based on verified criteria and no known hard exclusion |
| `moderate` | Plausible fit with important confirmations still needed |
| `weak` | Low-confidence fit or limited criteria support |
| `disqualified` | Known conflict with eligibility criteria |

> [!IMPORTANT]
> A `strong` match is still a candidate for human review, not a clinical recommendation.

## Configuration Notes

`DATABASE_URI` and `MEMORY_DB_DSN` must point to the same database in the default local setup.

Privacy modes:

| Mode | Behavior |
| --- | --- |
| `blocked` | External LLM prompts with patient data are forbidden |
| `deidentified` | External LLM prompts use deidentified clinically necessary summaries |
| `full_consent` | External LLM prompts require `CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true` |
| `local_only` | External LLM providers are forbidden |

> [!CAUTION]
> Keep `LLM_PRIVACY_MODE=blocked` or `deidentified` unless consent and privacy requirements have been explicitly reviewed.

ClinicalTrials.gov proxy behavior:

- If `CTGOV_PROXY_URL` is configured, the CLI probes its `/healthz` endpoint.
- If the proxy is unavailable, the CLI starts `uvicorn ctgov_proxy:app`.
- If `CTGOV_PROXY_URL` is not set, the local default is `http://127.0.0.1:8321/ctgov/search`.

Start the proxy manually:

```bash
uv run uvicorn ctgov_proxy:app --host 127.0.0.1 --port 8000
```

## Development

Run the full local quality gate:

```bash
make check
```

Run checks individually:

```bash
make lint
make typecheck
make test
make security
make complexity
```

Format and apply safe lint fixes:

```bash
make fix
```

Run coverage:

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

Run the LangGraph dev server:

```bash
uv run langgraph dev --config langgraph.json --no-browser
```

Exported graphs:

- `clinical_trial_agent`
- `retrieval`
- `eligibility`
- `synthesis`

## Docker

Build the image:

```bash
docker build -t clinical-trial-agent:local .
```

Run the CLI help from the image:

```bash
docker run --rm clinical-trial-agent:local --help
```

Start the local stack:

```bash
docker compose up -d --build
```

Inspect and stop services:

```bash
docker compose ps
docker compose down --remove-orphans
```

The Docker build uses the locked dependency set from `uv.lock`.

## Project Structure

```text
clinical_trial_agent/      CLI, config, memory, validation, and LangGraph app
agents/                    Supervisor and focused agent helpers
subagents/                 Retrieval, eligibility, and synthesis graphs
models/                    Pydantic domain models
prompts/                   Prompt builders
tools/                     Cache, retry, DB, validation, proxy, telemetry helpers
docs/                      Architecture, runbook, safety, and operations notes
tests/                     Unit and regression tests
ctgov_proxy.py             PHI-safe ClinicalTrials.gov proxy
patient_profile.json       Synthetic profile fixture
langgraph.json             LangGraph dev server configuration
docker-compose.yml         Local PostgreSQL and app stack
```

## Troubleshooting

Environment validation fails:

- Run `docker compose up -d db`.
- Confirm `DATABASE_URI` and `MEMORY_DB_DSN` use port `5433` for local Compose.
- Confirm `PROFILE_HASH_SALT`, `DB_ENCRYPTION_KEY`, `TENANT_ID`, and `FACILITY_ID` are set.

LLM calls fail closed:

- Check `LLM_PROVIDER`.
- Set the provider API key, or use `ollama` for local model testing.
- Confirm `LLM_PRIVACY_MODE` and `CLINICAL_DATA_EXTERNAL_LLM_CONSENT`.

ClinicalTrials.gov retrieval fails:

- Let the CLI auto-start the proxy, or start it manually with `uv run uvicorn ctgov_proxy:app --host 127.0.0.1 --port 8000`.
- If using `CTGOV_PROXY_URL`, ensure it points to a reachable `/ctgov/search` route.

Useful docs:

- [`docs/runbook.md`](docs/runbook.md)
- [`docs/ops_playbook.md`](docs/ops_playbook.md)
- [`docs/safety_model.md`](docs/safety_model.md)
- [`docs/architecture.mermaid`](docs/architecture.mermaid)
