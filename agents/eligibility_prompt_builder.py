from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from prompts.eligibility import build_eligibility_prompt
from pydantic import BaseModel, Field
from tools.sanitizer import sanitize_patient_profile

from clinical_trial_agent.config import (
    get_settings,
    has_external_llm_consent,
    is_llm_provider_local,
)

DIRECT_IDENTIFIER_FIELDS = {
    "name",
    "first_name",
    "last_name",
    "date_of_birth",
    "dob",
    "address",
    "phone",
    "email",
    "mrn",
    "medical_record_number",
}

CLINICALLY_NECESSARY_FIELDS = {
    "age",
    "sex",
    "conditions",
    "primary_condition",
    "biomarkers",
    "lab_values",
    "medications",
    "prior_treatments",
    "contraindications",
    "ecog_performance_status",
    "stage",
    "smoking_status",
    "bmi",
}


class EligibilityPromptInput(BaseModel):
    patient_summary: str = Field(min_length=1)
    trial_summary: str = Field(min_length=1)


def _assert_privacy_mode_allows_prompt() -> None:
    settings = get_settings()
    provider_is_local = is_llm_provider_local()
    mode = settings.llm_privacy_mode
    if provider_is_local:
        return
    if mode == "blocked":
        raise RuntimeError(
            "LLM_PRIVACY_MODE=blocked forbids external LLM prompts with patient data."
        )
    if mode == "local_only":
        raise RuntimeError("LLM_PRIVACY_MODE=local_only forbids external LLM providers.")
    if mode == "full_consent" and not has_external_llm_consent():
        raise RuntimeError(
            "CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required for LLM_PRIVACY_MODE=full_consent."
        )


def _is_empty_value(value: Any) -> bool:
    return value in (None, "", [])


def _sanitize_or_redact(value: Any) -> str:
    text = "; ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    return sanitize_patient_profile(text).text or "[redacted]"


def _format_deidentified_value(key: str, value: Any) -> str:
    if key == "age" and isinstance(value, (int, float)):
        base = int(value) // 10 * 10
        return f"age_range: {base}-{base + 9}"
    return _sanitize_or_redact(value)


def _summary_mode() -> str:
    if is_llm_provider_local():
        return "full"
    return get_settings().llm_privacy_mode


def format_patient_summary(profile: dict[str, Any]) -> str:
    mode = _summary_mode()
    lines = []

    for key, val in profile.items():
        if _is_empty_value(val):
            continue
        if key in DIRECT_IDENTIFIER_FIELDS:
            continue
        if mode == "deidentified" and key not in CLINICALLY_NECESSARY_FIELDS:
            continue

        content = (
            _format_deidentified_value(key, val)
            if mode == "deidentified"
            else _sanitize_or_redact(val)
        )
        lines.append(f"{key}: {content}")

    return "\n".join(lines)


def format_trial_summary(trial: dict[str, Any], criteria: list[dict[str, Any]]) -> str:
    meta = [
        f"nct_id: {trial.get('nct_id', '')}",
        f"title: {sanitize_patient_profile(str(trial.get('brief_title', ''))).text}",
        f"phase: {trial.get('phase', '')}",
        f"status: {trial.get('overall_status', '')}",
    ]
    lines: list[str] = []
    for criterion in criteria:
        criteria_type = str(criterion.get("criteria_type", "inclusion")).lower()
        criterion_id = str(criterion.get("criterion_id", "")).strip()
        label = f"{criteria_type} {criterion_id}" if criterion_id else criteria_type
        text = sanitize_patient_profile(str(criterion.get("text", ""))).text
        lines.append(f"[{label}] {text}")
    return "\n".join([*meta, "", "criteria:", *lines])


def build_judge_messages(
    profile: dict[str, Any], trial: dict[str, Any], criteria: list[dict[str, Any]]
) -> list[BaseMessage]:
    _assert_privacy_mode_allows_prompt()
    prompt_input = EligibilityPromptInput(
        patient_summary=format_patient_summary(profile).strip(),
        trial_summary=format_trial_summary(trial, criteria).strip(),
    )
    prompt = ChatPromptTemplate.from_template(build_eligibility_prompt())
    messages = prompt.format_messages(
        patient_summary=prompt_input.patient_summary,
        trial_summary=prompt_input.trial_summary,
    )
    return [message for message in messages if isinstance(message, BaseMessage)]
