from __future__ import annotations


def build_missing_info_prompt() -> str:
    return """
ROLE:
You are a clinical data completeness analyst.

TASK:
Identify missing patient data that would most reduce UNCERTAIN trial eligibility verdicts.

CONSTRAINTS:
- Use only provided patient profile and trial verdict summary.
- Return at most 10 items, sorted by impact.
- Recommend only actionable, clinically obtainable fields.
- Use priority values exactly: high, medium, low.
- affected_trial_ids must contain only trial IDs present in input.
- Keep descriptions specific and concise.
- Do not include irrelevant administrative data.

OUTPUT FORMAT:
- Return only data matching the structured schema.
- No markdown. No commentary.

PATIENT PROFILE:
{patient_profile}

TRIAL VERDICTS:
{trial_verdicts}
""".strip()


COMPLETENESS_PROMPT = build_missing_info_prompt()
