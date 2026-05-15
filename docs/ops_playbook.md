# Operations Playbook

## CI debugging

```bash
cd .
uv sync --locked --group dev
uv run make check
uv run pytest --cov=clinical_trial_agent --cov=agents --cov=subagents --cov=tools --cov=models --cov-report=term --cov-fail-under=75
```

## Incident triage (runtime failures)

1. Confirm environment with `uv run clinical-trial-agent validate-env`.
2. Re-run with synthetic profile only.
3. Check fail-closed flags:
   - `CLINICAL_DATA_EXTERNAL_LLM_CONSENT`
   - `LLM_PRIVACY_MODE`
   - `TENANT_ID`
   - `FACILITY_ID`
   - `CTGOV_PROXY_URL` for PHI-bearing retrieval.
   - `WEBHOOK_ALLOWED_HOSTS` and `WEBHOOK_SIGNING_SECRET` for callbacks.
4. Inspect logs for `span.error` and `span.slo_miss`.

## Safety checks

- External LLM behavior is governed by `LLM_PRIVACY_MODE`; use `blocked` or
  `deidentified` unless a reviewed workflow requires `full_consent`.
- Webhook URLs are HTTPS-only by default and block localhost, private networks,
  metadata IPs, and embedded credentials.
- Criteria provenance should show `ctgov_api` and `criteria_source_verified=true`
  before interpreting a `strong` match.
- Physician feedback affects ranking only within the same tenant and facility
  scope.

## Safe config changes

1. Update `clinical_trial_agent/config.py`.
2. Add/adjust tests in `tests/test_config.py`.
3. Run `uv run make check`.
4. Ensure CI workflow env hardening checks still pass.
