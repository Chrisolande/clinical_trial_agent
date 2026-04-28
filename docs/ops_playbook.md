# Operations Playbook

## CI debugging

```bash
cd .
uv sync --locked --group dev
uv run make check
uv run pytest --cov --cov-report=term --cov-fail-under=75
```

## Incident triage (runtime failures)

1. Confirm environment with `uv run clinical-trial-agent validate-env`.
2. Re-run with synthetic profile only.
3. Check fail-closed flags:
   - `CLINICAL_DATA_EXTERNAL_LLM_CONSENT`
   - `TENANT_ID`
   - `FACILITY_ID`
   - `CTGOV_PROXY_URL` for PHI-bearing retrieval.
4. Inspect logs for `span.error` and `span.slo_miss`.

## Safe config changes

1. Update `clinical_trial_agent/config.py`.
2. Add/adjust tests in `tests/test_config.py`.
3. Run `uv run make check`.
4. Ensure CI workflow env hardening checks still pass.
