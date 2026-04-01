# Clinical Trial Agent

Multi-agent clinical trial matching built with LangGraph and LangChain.

It orchestrates retrieval, eligibility reasoning, and report synthesis to match a patient profile to relevant trials from ClinicalTrials.gov.

## Overview

The pipeline is coordinated by a supervisor agent that calls three domain subagents:

- `retrieval`: finds candidate trials from ClinicalTrials.gov.
- `eligibility`: scores each trial against patient criteria.
- `synthesis`: ranks results and generates the final report.

The system includes PostgreSQL-backed episodic memory and LangGraph checkpointing for durable, thread-scoped runs.

## Key capabilities

- Tool-calling supervisor orchestration with retry-aware routing.
- Structured eligibility outcomes (`MEETS`, `FAILS`, `UNCERTAIN`) and scored trial ranking.
- Context-engineering controls (selection, compression, isolation, and score gating).
- Prompt-driven architecture with all prompt templates centralized in `prompts/`.
- Async CLI powered by `async-typer` and `rich`.

## Tech stack

- Python 3.11+
- LangGraph, LangChain, LangChain provider integrations
- PostgreSQL (`asyncpg`) for memory/checkpoint persistence
- `ruff`, `mypy`, `pytest`, `radon` for quality gates

## Quick start

```bash
git clone https://github.com/Chrisolande/clinical_trial_agent.git
cd clinical_trial_agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set environment variables (in shell or `.env`):

```bash
LLM_PROVIDER=gemini # or openai / auto
GEMINI_API_KEY=...
OPENAI_API_KEY=...
DATABASE_URI=postgresql://postgres:postgres@localhost:5432/postgres
MEMORY_DB_DSN=postgresql://postgres:postgres@localhost:5432/postgres
```

> [!IMPORTANT]
> `DATABASE_URI` and `MEMORY_DB_DSN` must resolve to the same database in this project setup.

Validate your environment:

```bash
clinical-trial-agent validate-env
```

## CLI usage

Get help:

```bash
clinical-trial-agent --help
```

Run full matching pipeline:

```bash
clinical-trial-agent run ./patient_profile.json
```

Search ClinicalTrials.gov directly:

```bash
clinical-trial-agent search --condition "non-small cell lung cancer" --intervention "pembrolizumab"
```

Manage episodic memory:

```bash
clinical-trial-agent memory list
clinical-trial-agent memory purge
clinical-trial-agent memory invalidate ./patient_profile.json
```

## Project layout

```text
agents/         Supervisor and domain logic
subagents/      retrieval/, eligibility/, synthesis/ LangGraph workflows
prompts/        Centralized prompt templates
models/         Pydantic models and typed state schemas
tools/          API, retry, cache, and DB helpers
cli.py          Async Typer + Rich CLI entrypoint
clinical_trials.py
memory.py
config.py
tests/
```

## Development

Run local quality checks:

```bash
ruff check .
ruff format --check .
mypy .
pytest -q
radon cc . -s -n B --exclude "tests/*"
```

> [!TIP]
> Start with `clinical-trial-agent validate-env` before running the full pipeline.
