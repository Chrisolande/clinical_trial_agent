"""Missing information identification node."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from config import get_llm, settings
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.missing_info import CompletenessAssessmentList
from prompts.missinginfo import build_missing_info_prompt

_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_template(build_missing_info_prompt())
_CHAIN: Any = None


def _get_chain() -> Any:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = _PROMPT | get_llm().with_structured_output(CompletenessAssessmentList)
    return _CHAIN


def _build_uncertain_summary(
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], str]:
    uncertain_by_theme: dict[str, list[str]] = {}
    for trial_id, verdict_data in eligibility_verdicts.items():
        for v in verdict_data.get("verdicts", []):
            if v["verdict"] == "UNCERTAIN":
                text = v["criterion_text"][:100]
                uncertain_by_theme.setdefault(text, []).append(trial_id)
    summary = "\n".join(
        f"- [{', '.join(ids[:3])}] {text}" for text, ids in list(uncertain_by_theme.items())[:20]
    )
    return uncertain_by_theme, summary


def _format_profile_summary(profile: dict[str, Any]) -> str:
    available = [f"{k}: {str(v)[:60]}" for k, v in profile.items() if v and v != [] and v != ""]
    return "\n".join(available[:20])


async def _invoke_missing_info_llm(
    patient_profile: dict[str, Any],
    uncertain_summary: str,
) -> CompletenessAssessmentList:
    result = await asyncio.wait_for(
        _get_chain().ainvoke(
            {
                "patient_profile": _format_profile_summary(patient_profile),
                "trial_verdicts": uncertain_summary,
            },
            config={
                "run_name": "missing_info",
                "tags": ["eligibility", "missing-data"],
            },
        ),
        timeout=settings.llm_call_timeout_seconds,
    )
    return cast("CompletenessAssessmentList", result)


async def identify_missing_info(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    uncertain_by_theme, uncertain_summary = _build_uncertain_summary(eligibility_verdicts)
    if not uncertain_by_theme:
        return []

    try:
        result = await _invoke_missing_info_llm(patient_profile, uncertain_summary)
        if result and result.results:
            return [item.model_dump() for item in result.results]
        logger.info("Missing info model returned no items; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)

    except TimeoutError:
        logger.warning("Missing info identification timed out; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)
    except asyncio.CancelledError:
        logger.warning("Missing info identification cancelled; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)

    except Exception:
        logger.exception("Missing info identification failed.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)


def _fallback_missing_info_recommendations(
    uncertain_by_theme: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not uncertain_by_theme:
        return []

    items = sorted(
        uncertain_by_theme.items(),
        key=lambda kv: len(set(kv[1])),
        reverse=True,
    )[:10]

    return [
        {
            "field": text,
            "description": (
                "Patient data is missing or ambiguous for this criterion and blocks a "
                "confident eligibility verdict."
            ),
            "affected_trial_ids": list(dict.fromkeys(ids)),
            "priority": _priority_from_impact(len(set(ids))),
        }
        for text, ids in items
    ]


def _priority_from_impact(affected_count: int) -> str:
    if affected_count >= 4:
        return "high"
    if affected_count >= 2:
        return "medium"
    return "low"
