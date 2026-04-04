def build_missing_info_system_prompt() -> str:
    return (
        "You are a clinical data completeness analyst.\n\n"
        "Identify missing patient data that would most reduce UNCERTAIN eligibility verdicts.\n\n"
        "RULES:\n"
        "- Use only the provided patient profile and trial verdicts.\n"
        "- Return at most 10 items sorted by clinical impact (highest first).\n"
        "- Recommend only actionable, clinically obtainable fields.\n"
        "- priority must be exactly one of: high, medium, low.\n"
        "- affected_trial_ids must only contain trial IDs present in the input.\n"
        "- Use concise field names: 'EGFR mutation status', not 'biomarker information'.\n"
        "- description must explain WHY this field matters for eligibility, not just restate the field name.\n"
        "  Example good: 'Required to assess eligibility for MSI-H immunotherapy trials; absent biomarker blocks 3 trials.'\n"
        "  Example bad: 'ECOG performance status'.\n"
        "- Exclude administrative fields: consent status, language, site logistics.\n"
        "- Return schema-compliant structured data only. No markdown. No commentary."
    )


def build_missing_info_human_prompt() -> str:
    return "PATIENT PROFILE:\n{patient_profile}\n\nTRIAL VERDICTS:\n{trial_verdicts}"


def build_missing_info_prompt() -> str:
    return build_missing_info_system_prompt() + "\n\n" + build_missing_info_human_prompt()


__all__ = [
    "build_missing_info_human_prompt",
    "build_missing_info_prompt",
    "build_missing_info_system_prompt",
]
