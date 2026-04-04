# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub Security Advisories or by contacting project maintainers directly. Include reproduction steps, impact, and suggested remediation.

## Data flows leaving the system

1. Patient profile fields are sent to the configured LLM provider for eligibility reasoning and synthesis.
2. When Tavily enrichment is enabled, trial NCT IDs and trial titles are sent to Tavily search APIs.

## Consent gate

Set `CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true` before any external LLM use that includes patient data. If unset, the pipeline should fail closed.

## Storage protections

1. Sensitive memory payloads should be encrypted at rest using `DB_ENCRYPTION_KEY` (Fernet).
2. Patient profile hashing should be salted using `PROFILE_HASH_SALT` to prevent cross-system correlation.

## Compliance statement

This project is **not certified for real-patient clinical use** without additional legal, privacy, and regulatory compliance review.

## Third-party retention

Review retention and processing policies for DeepSeek/LangChain providers and Tavily before production use.
