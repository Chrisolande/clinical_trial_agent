from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, cast

import openai
from config import get_llm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from models.judge_verdict import JudgeVerdict
from prompts.eligibility import build_eligibility_prompt
from pydantic import ValidationError

_FALLBACK_VERDICT: dict[str, Any] = {
    "match_tier": "weak",
    "match_score": 0.1,
    "major_criteria_assessable": False,
    "inclusion_met": [],
    "inclusion_failed": [],
    "inclusion_uncertain": [],
    "exclusion_triggered": [],
    "exclusion_uncertain": [],
    "critical_missing_info": ["Insufficient structured verdict from judge model."],
    "key_concern": "LLM response parsing failed",
    "rationale": "Could not parse model verdict; conservative weak fallback applied.",
}

_TIMEOUT_EXCEPTIONS = (
    TimeoutError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def _log_warning(msg: str, *args: Any) -> None:
    logger.warning(msg, *args)


def _log_error(msg: str, *args: Any) -> None:
    logger.opt(exception=False).error(msg, *args)


def _format_patient_summary(patient_profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in patient_profile.items():
        if value in (None, "", []):
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_trial_summary(trial: dict[str, Any], criteria: list[dict[str, Any]]) -> str:
    trial_meta = [
        f"nct_id: {trial.get('nct_id', '')}",
        f"title: {trial.get('brief_title', '')}",
        f"phase: {trial.get('phase', '')}",
        f"status: {trial.get('overall_status', '')}",
    ]
    criteria_lines = [
        f"[{str(c.get('criteria_type', 'inclusion')).lower()}] {c.get('text', '')!s}"
        for c in criteria
    ]
    return "\n".join([*trial_meta, "", "criteria:", *criteria_lines])


def _build_judge_messages(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> list[BaseMessage]:
    prompt_messages = build_eligibility_prompt(
        patient_summary=_format_patient_summary(patient_profile),
        trial_summary=_format_trial_summary(trial, criteria),
    )
    messages: list[BaseMessage] = []
    for message in prompt_messages:
        role = message.get("role", "")
        content = str(message.get("content", ""))
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
    return messages


def _parse_verdict_xml(raw_response: str, trial_id: str) -> dict[str, Any]:
    try:
        payload = raw_response.split("<verdict>")[1].split("</verdict>")[0]
        return cast("dict[str, Any]", json.loads(payload))
    except (IndexError, JSONDecodeError, TypeError):
        _log_warning(
            "Failed to parse judge verdict for {}. Raw (first 200 chars): {!r}",
            trial_id,
            raw_response[:200],
        )
        return dict(_FALLBACK_VERDICT)


def _validate_verdict(verdict_dict: dict[str, Any], trial_id: str) -> JudgeVerdict:
    try:
        return JudgeVerdict.model_validate(verdict_dict)
    except (ValidationError, TypeError, ValueError):
        _log_warning("Invalid judge verdict schema for {}", trial_id)
        return JudgeVerdict.model_validate(dict(_FALLBACK_VERDICT))


def _make_fallback(key_concern: str, rationale: str, missing: str) -> dict[str, Any]:
    fallback = dict(_FALLBACK_VERDICT)
    fallback["key_concern"] = key_concern
    fallback["rationale"] = rationale
    fallback["critical_missing_info"] = [missing]
    return fallback


async def _judge_trial(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    llm = get_llm()
    messages = _build_judge_messages(patient_profile, trial, criteria)
    response = await llm.ainvoke(
        messages,
        config={"run_name": "eligibility_judge", "tags": ["eligibility", "judge"]},
    )
    raw = (
        str(response.content)
        if isinstance(response, AIMessage)
        else str(getattr(response, "content", response))
    )
    trial_id = str(trial.get("nct_id", "unknown"))
    parsed = _parse_verdict_xml(raw, trial_id)
    return _validate_verdict(parsed, trial_id)


def _build_verdict_rows(verdict: JudgeVerdict) -> list[dict[str, Any]]:
    def rows(texts: list[str], verdict_label: str, kind: str, hard: bool) -> list[dict[str, Any]]:
        return [
            {
                "criterion_text": t,
                "verdict": verdict_label,
                "criterion_type": kind,
                "is_hard_exclusion": hard,
            }
            for t in texts
        ]

    return [
        *rows(verdict.inclusion_met, "MEETS", "inclusion", False),
        *rows(verdict.inclusion_failed, "FAILS", "inclusion", False),
        *rows(verdict.inclusion_uncertain, "UNCERTAIN", "inclusion", False),
        *rows(verdict.exclusion_triggered, "FAILS", "exclusion", True),
        *rows(verdict.exclusion_uncertain, "UNCERTAIN", "exclusion", True),
    ]


async def evaluate_criteria_batch(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_id = str(trial.get("nct_id", "unknown"))

    try:
        verdict = await _judge_trial(patient_profile, trial, all_criteria)

    except _TIMEOUT_EXCEPTIONS as exc:
        _log_warning("Eligibility judge timed out for {} ({})", trial_id, type(exc).__name__)
        verdict = _validate_verdict(
            _make_fallback(
                "Eligibility judge timeout",
                "Eligibility judge timed out; conservative weak fallback applied.",
                "Eligibility judge timed out.",
            ),
            trial_id,
        )

    except Exception as exc:
        _log_error(
            "Eligibility judge failed for {} ({}): {}",
            trial_id,
            type(exc).__name__,
            exc,
        )
        verdict = _validate_verdict(
            _make_fallback(
                f"Eligibility judge error ({type(exc).__name__})",
                "Eligibility judge execution failed; conservative weak fallback applied.",
                f"Eligibility judge raised {type(exc).__name__}.",
            ),
            trial_id,
        )

    verdicts = _build_verdict_rows(verdict)
    inclusion_meets = [v for v in verdicts if v["verdict"] == "MEETS"]
    inclusion_fails = [
        v for v in verdicts if v["verdict"] == "FAILS" and not v["is_hard_exclusion"]
    ]
    exclusion_triggered = [
        v for v in verdicts if v["verdict"] == "FAILS" and v["is_hard_exclusion"]
    ]
    uncertain = [v for v in verdicts if v["verdict"] == "UNCERTAIN"]

    return {
        "trial_id": trial_id,
        "match_score": verdict.match_score,
        "match_tier": verdict.match_tier,
        "major_criteria_assessable": verdict.major_criteria_assessable,
        "critical_missing_info": verdict.critical_missing_info,
        "key_concern": verdict.key_concern,
        "rationale": verdict.rationale,
        "verdicts": verdicts,
        "meets_count": len(inclusion_meets),
        "fails_count": len(inclusion_fails) + len(exclusion_triggered),
        "uncertain_count": len(uncertain),
        "hard_exclusion_failures": len(exclusion_triggered),
    }
