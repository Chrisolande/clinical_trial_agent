"""Missing information identification node."""

import asyncio
import re
from typing import Any

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger
from models.missing_info import CompletenessAssessmentList
from prompts.missinginfo import build_missing_info_human_prompt, build_missing_info_system_prompt
from pydantic import BaseModel, Field

from agents.consent import assert_external_llm_consent
from agents.missing_info_catalog import (
    fallback_missing_info_recommendations,
    priority_from_impact,
    to_actionable_field,
)
from clinical_trial_agent.config import get_llm, get_settings

_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(build_missing_info_system_prompt()),
        HumanMessagePromptTemplate.from_template(build_missing_info_human_prompt()),
    ]
)


class MissingInfoPromptInput(BaseModel):
    patient_profile: str = Field(min_length=1)
    trial_verdicts: str = Field(min_length=1)


def _get_chain() -> Any:
    return _PROMPT | get_llm().with_structured_output(CompletenessAssessmentList)


def _build_uncertain_summary(
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    uncertain_by_field: dict[str, dict[str, Any]] = {}
    for trial_id, verdict_data in eligibility_verdicts.items():
        for v in verdict_data.get("verdicts", []):
            if v["verdict"] == "UNCERTAIN":
                text = str(v.get("criterion_text", "")).strip()
                canonical = to_actionable_field(text)
                field_id = canonical["field_id"]
                if field_id not in uncertain_by_field:
                    uncertain_by_field[field_id] = {
                        **canonical,
                        "affected_trial_ids": [],
                        "examples": [],
                    }
                if trial_id not in uncertain_by_field[field_id]["affected_trial_ids"]:
                    uncertain_by_field[field_id]["affected_trial_ids"].append(trial_id)
                if text and text not in uncertain_by_field[field_id]["examples"]:
                    uncertain_by_field[field_id]["examples"].append(text[:120])
    sorted_items = sorted(
        uncertain_by_field.values(),
        key=lambda item: (-len(item["affected_trial_ids"]), item["field_id"]),
    )
    summary = "\n".join(
        (
            f"- {item['field_id']} ({item['display_name']}) "
            f"[{', '.join(item['affected_trial_ids'][:3])}] "
            f"why_needed={item['why_needed']} "
            f"example={item['examples'][0] if item['examples'] else ''}"
        )
        for item in sorted_items[:20]
    )
    return uncertain_by_field, summary


def _format_profile_summary(profile: dict[str, Any]) -> str:
    redacted_fields = {
        "age",
        "sex",
        "conditions",
        "primary_condition",
        "biomarkers",
        "medications",
        "prior_treatments",
    }
    available: list[str] = []
    for key, value in profile.items():
        if value in (None, "", []):
            continue
        if key in redacted_fields:
            available.append(f"{key}: [redacted]")
        else:
            available.append(f"{key}: {str(value)[:60]}")
    return "\n".join(available[:20])


async def _invoke_missing_info_llm(
    patient_profile: dict[str, Any],
    uncertain_summary: str,
) -> CompletenessAssessmentList:
    assert_external_llm_consent()
    prompt_input = MissingInfoPromptInput(
        patient_profile=_format_profile_summary(patient_profile).strip(),
        trial_verdicts=uncertain_summary.strip(),
    )
    result = await _get_chain().ainvoke(
        {
            "patient_profile": prompt_input.patient_profile,
            "trial_verdicts": prompt_input.trial_verdicts,
        },
        config={
            "run_name": "missing_info",
            "tags": ["eligibility", "missing-data"],
            "timeout": get_settings().llm_call_timeout_seconds,
        },
    )
    if isinstance(result, CompletenessAssessmentList):
        return result
    if isinstance(result, dict):
        return CompletenessAssessmentList.model_validate(result)
    raise TypeError(f"Unexpected missing-info output type: {type(result)!r}")


async def identify_missing_info(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    uncertain_by_field, uncertain_summary = _build_uncertain_summary(eligibility_verdicts)
    if not uncertain_by_field:
        return []
    try:
        result = await _invoke_missing_info_llm(patient_profile, uncertain_summary)
        if result and result.results:
            return _enrich_with_trial_context(
                [item.model_dump() for item in result.results], uncertain_by_field
            )
        logger.info("Missing info model returned no items; using deterministic fallback.")
        return fallback_missing_info_recommendations(uncertain_by_field)
    except TimeoutError:
        logger.warning("Missing info identification timed out; using deterministic fallback.")
        return fallback_missing_info_recommendations(uncertain_by_field)
    except asyncio.CancelledError:
        logger.warning("Missing info identification cancelled; using deterministic fallback.")
        return fallback_missing_info_recommendations(uncertain_by_field)
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.exception("Missing info identification failed: {}", exc)
        return fallback_missing_info_recommendations(uncertain_by_field)


def _enrich_with_trial_context(
    items: list[dict[str, Any]], uncertain_by_field: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in items:
        field_id = _missing_info_field_id(item)
        context = uncertain_by_field.get(field_id, {})
        affected_ids = _merged_affected_ids(item, context)
        enriched.append(
            {
                "field_id": field_id,
                "display_name": _missing_info_display_name(item, context, field_id),
                "category": str(item.get("category") or context.get("category") or "clinical"),
                "field": _missing_info_display_name(item, context, field_id),
                "why_needed": _missing_info_rationale(item, context),
                "evidence_text": _missing_info_rationale(item, context),
                "description": _missing_info_rationale(item, context),
                "affected_trial_ids": affected_ids,
                "priority": _missing_info_priority(item, affected_ids),
            }
        )
    return enriched


def _missing_info_field_id(item: dict[str, Any]) -> str:
    field_id = str(item.get("field_id", "")).strip()
    if field_id:
        return field_id
    source = str(item.get("display_name") or item.get("field") or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", source).strip("_") or "additional_clinical_detail"


def _missing_info_display_name(item: dict[str, Any], context: dict[str, Any], field_id: str) -> str:
    return (
        str(
            item.get("display_name") or item.get("field") or context.get("display_name") or ""
        ).strip()
        or field_id.replace("_", " ").title()
    )


def _missing_info_rationale(item: dict[str, Any], context: dict[str, Any]) -> str:
    return str(
        item.get("why_needed")
        or item.get("evidence_text")
        or item.get("description")
        or context.get("why_needed")
        or ""
    ).strip()


def _merged_affected_ids(item: dict[str, Any], context: dict[str, Any]) -> list[str]:
    item_ids = [str(tid) for tid in (item.get("affected_trial_ids") or [])]
    context_ids = [str(tid) for tid in (context.get("affected_trial_ids") or [])]
    return list(dict.fromkeys(item_ids + context_ids))


def _missing_info_priority(item: dict[str, Any], affected_ids: list[str]) -> str:
    priority = str(item.get("priority", "")).lower()
    if priority in {"high", "medium", "low"}:
        return priority
    return priority_from_impact(len(set(affected_ids)))
