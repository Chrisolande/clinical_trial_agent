"""Biomedical terminology normalisation node."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.terminology import NormalisedTerminology
from prompts.terminology import build_terminology_prompt
from tools.retry import llm_retry

from clinical_trial_agent.config import get_llm

_chain = ChatPromptTemplate.from_template(
    build_terminology_prompt()
) | get_llm().with_structured_output(NormalisedTerminology)


def _prepare_terminology_inputs(patient_profile: dict) -> tuple[list[str], list[str]]:
    conditions = list(patient_profile.get("conditions", []))
    primary = patient_profile.get("primary_condition")
    if primary and primary not in conditions:
        conditions = [primary, *conditions]
    medications = [
        m.get("name", "") if isinstance(m, dict) else str(m)
        for m in patient_profile.get("medications", [])
    ]
    return conditions, medications


def _fallback_terminology(conditions: list[str], medications: list[str]) -> dict:
    return {
        "conditions": {},
        "medications": {},
        "primary_search_terms": conditions[:3],
        "intervention_search_terms": medications[:3],
    }


def _format_terms(conditions: list[str], medications: list[str]) -> str:
    parts = []
    if conditions:
        parts.append(f"Conditions: {', '.join(conditions)}")
    if medications:
        parts.append(f"Medications: {', '.join(medications)}")
    return "\n".join(parts)


@llm_retry
async def _invoke_terminology_llm(
    conditions: list[str], medications: list[str]
) -> NormalisedTerminology:
    result = await _chain.ainvoke(
        {"terms": _format_terms(conditions, medications)},
        config={
            "run_name": "terminology_normalize",
            "tags": ["supervisor", "terminology"],
        },
    )
    if isinstance(result, NormalisedTerminology):
        return result
    if isinstance(result, dict):
        return NormalisedTerminology.model_validate(result)
    raise TypeError(f"Unexpected terminology output type: {type(result)!r}")


async def normalize_terminology(patient_profile: dict[str, Any]) -> dict[str, object]:
    conditions, medications = _prepare_terminology_inputs(patient_profile)
    if not conditions and not medications:
        return _fallback_terminology([], [])
    try:
        result: NormalisedTerminology = await _invoke_terminology_llm(conditions, medications)
        return result.model_dump()
    except Exception as exc:
        logger.error("Terminology normalisation failed after retries: {}", exc)
    return _fallback_terminology(conditions, medications)
