import os
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from prompts.eligibility import build_eligibility_prompt
from tools.sanitizer import sanitize_patient_profile

from config import external_llm_requires_consent


def _has_consent() -> bool:
    return os.environ.get("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "").lower() == "true"


def _assert_external_llm_consent() -> None:
    if external_llm_requires_consent() and not _has_consent():
        raise RuntimeError("CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true required for external LLMs.")


def format_patient_summary(profile: dict[str, Any]) -> str:
    pii_fields = {
        "age",
        "sex",
        "conditions",
        "primary_condition",
        "biomarkers",
        "medications",
        "prior_treatments",
    }
    consent_active = _has_consent() and external_llm_requires_consent()
    lines = []

    for key, val in profile.items():
        if val in (None, "", []):
            continue

        if key not in pii_fields:
            lines.append(f"{key}: {val}")
            continue

        if not consent_active:
            lines.append(f"{key}: [redacted]")
            continue

        if key == "age" and isinstance(val, (int, float)):
            base = int(val) // 10 * 10
            content = f"age_range: {base}-{base + 9}"
        else:
            text = "; ".join(str(v) for v in val) if isinstance(val, (list, tuple)) else str(val)
            content = sanitize_patient_profile(text).text

        lines.append(f"{key}: {content}" if content else f"{key}: [redacted]")

    return "\n".join(lines)


def format_trial_summary(trial: dict[str, Any], criteria: list[dict[str, Any]]) -> str:
    meta = [
        f"nct_id: {trial.get('nct_id', '')}",
        f"title: {sanitize_patient_profile(str(trial.get('brief_title', ''))).text}",
        f"phase: {trial.get('phase', '')}",
        f"status: {trial.get('overall_status', '')}",
    ]
    lines = [
        f"[{c.get('criteria_type', 'inclusion').lower()}] {sanitize_patient_profile(str(c.get('text', ''))).text}"
        for c in criteria
    ]
    return "\n".join([*meta, "", "criteria:", *lines])


def build_judge_messages(
    profile: dict[str, Any], trial: dict[str, Any], criteria: list[dict[str, Any]]
) -> list[BaseMessage]:
    _assert_external_llm_consent()
    prompt = ChatPromptTemplate.from_template(build_eligibility_prompt())
    messages = prompt.format_messages(
        patient_summary=format_patient_summary(profile),
        trial_summary=format_trial_summary(trial, criteria),
    )
    return [message for message in messages if isinstance(message, BaseMessage)]
