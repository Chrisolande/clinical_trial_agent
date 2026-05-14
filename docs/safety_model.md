# Safety Model

This repository is not certified for real patient care. Outputs are intended
for clinical review workflows only after local clinical, legal, security, and
compliance approval.

## Privacy Modes

`LLM_PRIVACY_MODE` controls patient-data handling for LLM prompts:

- `blocked`: external LLM prompts with patient data are forbidden.
- `deidentified`: external LLM prompts use clinically necessary fields with
  direct identifiers removed and age represented as a range.
- `full_consent`: external LLM prompts require
  `CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true`.
- `local_only`: local providers may receive clinically necessary profile
  fields; external providers are forbidden.

Local providers are not redacted merely because external consent is false.

## Webhooks

Webhook callbacks send only run metadata, profile hash, and outcome summary.
They do not send the full patient profile. HTTPS public URLs are required by
default. Localhost, loopback, private, link-local, metadata IPs, and embedded
credentials are rejected unless local development is explicitly allowed.

Set `WEBHOOK_ALLOWED_HOSTS` to a comma-separated allowlist when callbacks must
be restricted to known domains. If `WEBHOOK_SIGNING_SECRET` is configured,
callbacks include `X-Clinical-Trial-Agent-Signature: sha256=<hex>`.

## Feedback

Physician feedback is scoped by tenant, facility, and patient profile hash.
Confirmed feedback gives a small ranking boost for the same scoped profile;
rejected feedback gives a small penalty. Feedback from another tenant or
facility must not affect ranking.

## Criteria Provenance

Eligibility criteria from the ClinicalTrials.gov API are marked verified and
full. Search-snippet supplementation is marked unverified and partial. Partial
or unverified criteria cannot produce a `strong` tier unless official criteria
are later fetched.
