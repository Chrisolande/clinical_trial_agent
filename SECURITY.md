# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately via [GitHub Security Advisories](https://github.com/Chrisolande/clinical_trial_agent/security/advisories/new) or by contacting the project maintainer directly. Include reproduction steps, impact assessment, and suggested remediation if you have one.

## Data flows leaving the system

Patient data touches three external services:

1. **LLM providers** - patient profile fields are sent to Gemini, OpenAI (via OpenRouter), or DeepSeek for eligibility reasoning and report synthesis. Which provider runs depends on `LLM_PROVIDER`.
2. **ClinicalTrials.gov API** - trial search queries are sent to `https://clinicaltrials.gov/api/v2`. No patient fields are included; only condition and intervention terms.
3. **Tavily** - when enrichment is enabled, NCT IDs and trial titles are sent to Tavily search APIs for additional context.

## Consent gate

Set `CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true` before running any pipeline that sends patient data to an external LLM. If the variable is unset, the pipeline should fail closed. Do not run the agent against real patient profiles without this flag intentionally set.

## Secret management

All secrets are loaded from environment variables. The `bootstrap_environment()` function in `config.py` reads from a `.env` file at the project root using a custom parser. Key secrets:

- `GEMINI_API_KEY` / `GOOGLE_API_KEY` - Gemini provider
- `OPENAI_API_KEY` - OpenAI / OpenRouter provider
- `DEEPSEEK_API_KEY` - DeepSeek provider (validated by `validate_env.py`)
- `DATABASE_URI` / `MEMORY_DB_DSN` - PostgreSQL connection strings
- `DB_ENCRYPTION_KEY` - Fernet key used to encrypt sensitive memory payloads at rest
- `PROFILE_HASH_SALT` - salt applied during patient profile hashing to prevent cross-system correlation
- `CLINICAL_DATA_EXTERNAL_LLM_CONSENT` - consent gate flag (see section above)

The `.gitignore` excludes `.env`. Never commit secrets to the repository.

`DB_ENCRYPTION_KEY` must be a valid Fernet key (32 url-safe base64-encoded bytes). Generate one with:

```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

## Memory and storage

Patient pipeline results are stored in PostgreSQL via `EpisodicMemory` in `memory.py`. Before storage:

- Profile keys are SHA-256 hashed using `json.dumps(profile, sort_keys=True)` as the canonical form. This provides consistent lookup without storing raw profile fields.
- Records expire after `MEMORY_TTL_DAYS` days (default: 30). Expired rows can be purged with `EpisodicMemory.purge_expired()`.
- LangGraph checkpoints use a separate `AsyncPostgresSaver` connection on `DATABASE_URI`.

For production deployments, encrypt the PostgreSQL volume at rest and restrict database access to the application user only.

## Static analysis and CI

The following checks run in CI:

- **ruff** - linting and import hygiene
- **mypy** - type checking with `disallow_untyped_defs`
- **bandit** - security-focused static analysis
- **pytest-cov** - test coverage, fail threshold at 70%

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
