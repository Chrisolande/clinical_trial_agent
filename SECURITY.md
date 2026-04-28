# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately via [GitHub Security Advisories](https://github.com/Chrisolande/clinical_trial_agent/security/advisories/new) or by contacting the project maintainer directly. Include reproduction steps, impact assessment, and suggested remediation if you have one.

## Data flows leaving the system

Patient data touches three external services:

1. **LLM providers** - patient profile fields are sent to Gemini, OpenAI (via OpenRouter), or DeepSeek for eligibility reasoning and report synthesis. Which provider runs depends on `LLM_PROVIDER`.
2. **ClinicalTrials.gov API** - trial search terms are sent to `https://clinicaltrials.gov/api/v2`. Condition/intervention terms are treated as PHI-sensitive transport inputs and are never sent as URL query params when present.
3. **Tavily** - when enrichment is enabled, NCT IDs and trial titles are sent to Tavily search APIs for additional context.

## External LLM consent gate (fail-closed)

External LLM usage is centralized through `config.get_llm()`.

- For external providers (`deepseek`, `openai`, `anthropic`), `CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true` is required before PHI-bearing calls are allowed.
- If consent is missing, runtime raises and blocks execution (fail-closed).
- Local `ollama` is treated as non-external and bypasses this gate.
- `get_llm()` defaults to `contains_phi=True` (fail-closed). `contains_phi=False` may be used only for explicitly non-PHI calls.

Do not run the agent against real patient profiles without this flag intentionally set.

## Retrieval transport controls (PHI-safe)

ClinicalTrials.gov retrieval transport is enforced in `clinical_trials._request_with_transport`:

- If `query.cond` or `query.intr` is present, transport is allowed only via `CTGOV_PROXY_URL` with a **POST** JSON payload (`{"endpoint": ..., "params": ...}`), never URL query params.
- If PHI-bearing retrieval is attempted without `CTGOV_PROXY_URL`, execution fails closed with a configuration error.
- For non-PHI retrieval, transport follows `CTGOV_TRANSPORT_MODE` (`get` default, optional `post`).
- On HTTP 403, urllib fallback is disabled for PHI-bearing requests.
- In production, `CTGOV_PROXY_URL` should be HTTPS. For local development, loopback HTTP endpoints such as `http://localhost` or `http://127.0.0.1` are accepted.

## QA gating and severity model

QA issues are structured as: `code`, `severity`, `message`.

- Supported severity values: `critical`, `high`, `medium`, `low`.
- QA is fail-closed on `critical` findings (`qa_passed=false`).
- Synthesis blocks report generation when QA fails; unresolved QA failures trigger re-evaluation signaling.

## Deterministic medication safety controls

Medication safety checks run before LLM eligibility reasoning:

- Known high-risk medication/intervention interactions are deterministic and return `severity=critical`, `disqualify=true`.
- Disqualified trials are forced to `match_tier=disqualified` with hard exclusion failure.
- Malformed medication input (missing name/malformed dose) fails closed and disqualifies the trial path.

## Tenant/facility governance (fail-closed)

Clinical-data memory, audit, and feedback access require explicit tenant context:

- `TENANT_ID` and `FACILITY_ID` must be non-empty and non-default.
- Placeholder defaults (`default-tenant`, `default-facility`) are rejected.
- Patient hash scope and all memory/audit/feedback operations are tenant+facility scoped.

## Secret management

All secrets are loaded from environment variables. The `bootstrap_environment()` function in `config.py` reads from a `.env` file at the project root using a custom parser. Key secrets/configuration:

- `GEMINI_API_KEY` / `GOOGLE_API_KEY` - Gemini provider
- `OPENAI_API_KEY` - OpenAI / OpenRouter provider
- `DEEPSEEK_API_KEY` - DeepSeek provider (validated by `validate_env.py`)
- `DATABASE_URI` / `MEMORY_DB_DSN` - PostgreSQL connection strings
- `DB_ENCRYPTION_KEY` - Fernet key used to encrypt sensitive memory payloads at rest
- `PROFILE_HASH_SALT` - salt used in patient profile hashing
- `CLINICAL_DATA_EXTERNAL_LLM_CONSENT` - external LLM consent gate flag
- `CTGOV_TRANSPORT_MODE` - ClinicalTrials.gov transport mode for non-PHI retrieval (`get`/`post`)
- `CTGOV_PROXY_URL` - optional PHI-safe proxy endpoint for retrieval POST transport (HTTPS in production; loopback HTTP accepted for local development)
- `TENANT_ID` / `FACILITY_ID` - required governance scope for clinical data memory/audit/feedback

The `.gitignore` excludes `.env`. Never commit secrets to the repository.

`DB_ENCRYPTION_KEY` must be a valid Fernet key (32 url-safe base64-encoded bytes). Generate one with:

```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

## Memory and storage

Patient pipeline results are stored in PostgreSQL via `EpisodicMemory` in `memory.py`. Before storage:

- Profile keys are SHA-256 hashed with canonical JSON and tenant/facility scope.
- Records expire after `MEMORY_TTL_DAYS` days (default: 30). Expired rows can be purged with `EpisodicMemory.purge_expired()`.
- LangGraph checkpoints use a separate `AsyncPostgresSaver` connection on `DATABASE_URI`.

For production deployments, encrypt the PostgreSQL volume at rest and restrict database access to the application user only.

## Static analysis and CI

The following checks run in CI:

- **ruff** - linting and import hygiene
- **mypy** - type checking with `disallow_untyped_defs`
- **bandit** - security-focused static analysis
- **detect-secrets** - fail-closed baseline policy + tracked-file secret scan against `.secrets.baseline`
- **CI env hardening consistency** - enforces fail-closed consent and tenant/facility defaults for CI test jobs
- **pytest-cov** - test coverage, fail threshold at 75%

The canonical local/CI command flow is documented in
`docs/runbook.md`.

Run `python validate_env.py` locally before deployment to confirm all required secrets are present and the database is reachable.

## Third-party retention

Review the data retention and processing policies for each external provider before using this system with real patient data:

- [Google AI / Gemini](https://ai.google.dev/terms) - check the Gemini API usage policies
- [OpenRouter](https://openrouter.ai/privacy) - requests may be forwarded to multiple model providers
- [DeepSeek](https://platform.deepseek.com/privacy_policy)
- [Tavily](https://tavily.com/privacy)

Policies differ significantly between providers. OpenRouter in particular routes requests through third parties, which may have their own retention terms.

## Compliance statement

This project is **not certified for real-patient clinical use**. Before deploying against actual patient data, you need independent legal, privacy, and regulatory review covering at minimum HIPAA (US), GDPR (EU), or the applicable regional framework.
