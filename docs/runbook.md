# Developer Runbook (Canonical)

This is the single copy-paste flow for local development and CI parity.

## 1) Setup

```bash
cd .
uv sync --locked --dev
```

## 2) Validate environment

```bash
uv run clinical-trial-agent validate-env
```

## 3) Quality checks

```bash
uv run make check
```

## 4) Run the pipeline

```bash
uv run clinical-trial-agent run ./patient_profile.json
```

## CI-equivalent test command

```bash
uv run pytest --cov --cov-report=xml --cov-fail-under=75
```
