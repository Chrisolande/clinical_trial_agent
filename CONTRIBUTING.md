# Contributing

## Branch naming

Use one of:
- `fix/<short-topic>`
- `feat/<short-topic>`
- `chore/<short-topic>`
- `refactor/<short-topic>`

## Pull requests

1. Open a PR to `main` with a concise problem statement and test evidence.
2. Keep changes scoped to one logical area.
3. Reference related AGENTS.md task IDs where relevant.

## Module size and refactors

Keep Python modules focused and under ~300 LOC. For larger features, extract
helpers into adjacent submodules (for example `*_helpers.py`, `*_rules.py`, or
feature-specific helper modules) instead of growing a single file.

## Local checks

Run:

```bash
make check
```

## Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Synthetic data convention

`patient_profile.json` must remain synthetic. It includes a `_notice` field and must never be replaced with real patient data.

## Adding a new LangGraph subgraph

1. Export a compiled module-level graph constant in Python.
2. Add a corresponding entry under `graphs` in `langgraph.json`.
3. Validate it loads in LangGraph Studio/dev server.

## Lockfile update process

Use `uv` lock management:

```bash
uv lock --upgrade
```

Commit the updated `uv.lock` with your dependency changes.
