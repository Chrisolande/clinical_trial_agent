import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.report import ReportPlan
from prompts.synthesis import build_synthesis_prompt
from tools.retry import llm_retry

from clinical_trial_agent.config import get_llm, get_settings


def _enforce_conservative_wording(summary: str, critical_missing_count: int) -> str:
    if critical_missing_count <= 0:
        return summary
    banned = (
        "no concerns",
        "no major concerns",
        "none identified",
        "no significant concerns",
    )
    sanitized = summary
    for phrase in banned:
        sanitized = sanitized.replace(phrase, "meaningful unresolved concerns")
        sanitized = sanitized.replace(phrase.title(), "Meaningful unresolved concerns")
    return sanitized


def _count_tiers(scored_trials: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tiers = Counter(str(t.get("tier", "weak")) for t in scored_trials)
    return (tiers["strong"], tiers["moderate"], tiers["weak"], tiers["disqualified"])


def _describe_patient(patient_profile: dict[str, Any]) -> str:
    conds = patient_profile.get("conditions", [])
    primary = patient_profile.get("primary_condition") or (
        conds[0] if conds else "unknown condition"
    )
    age = patient_profile.get("age", "unknown age")
    sex = patient_profile.get("sex", "")
    return f"{age} year old {sex} patient with {primary}".strip()


def _collect_exec_risk_signals(scored_trials: list[dict[str, Any]]) -> dict[str, Any]:
    deduped_missing_fields = _deduped_critical_missing_fields(scored_trials)

    return {
        "hard_exclusion_count": _hard_exclusion_count(scored_trials),
        "uncertain_major_criteria_count": _uncertain_major_criteria_count(scored_trials),
        "na_criteria_count": _not_applicable_criteria_count(scored_trials),
        "critical_missing_count": len(deduped_missing_fields),
        "critical_missing_fields": deduped_missing_fields[:12],
        "trial_type_signal": _trial_type_signal(scored_trials),
    }


def _hard_exclusion_count(scored_trials: list[dict[str, Any]]) -> int:
    return sum(int(t.get("hard_exclusion_failures", 0) or 0) for t in scored_trials)


def _uncertain_major_criteria_count(scored_trials: list[dict[str, Any]]) -> int:
    return sum(1 for t in scored_trials if t.get("major_criteria_assessable") is False)


def _not_applicable_criteria_count(scored_trials: list[dict[str, Any]]) -> int:
    return sum(int(_first_trial_count(t, _NA_COUNT_KEYS)) for t in scored_trials)


_NA_COUNT_KEYS = ("na_count", "n_a_count", "not_applicable_count", "not_applicable_criteria_count")


def _first_trial_count(trial: dict[str, Any], keys: tuple[str, ...]) -> int:
    return next((int(trial.get(key) or 0) for key in keys if trial.get(key)), 0)


def _deduped_critical_missing_fields(scored_trials: list[dict[str, Any]]) -> list[str]:
    missing_fields = [
        value
        for trial in scored_trials
        for item in trial.get("critical_missing_info", []) or []
        if (value := str(item).strip())
    ]
    return list(dict.fromkeys(missing_fields))


def _trial_type_signal(scored_trials: list[dict[str, Any]]) -> str:
    type_counts = Counter(_trial_type_label(trial) for trial in scored_trials)
    return (
        ", ".join(
            f"{label}={type_counts[label]}"
            for label in ("interventional", "observational", "registry")
            if type_counts[label] > 0
        )
        or "not available"
    )


def _trial_type_label(trial: dict[str, Any]) -> str:
    raw_type = str(trial.get("study_type") or trial.get("trial_type") or "").strip().lower()
    if "observational" in raw_type:
        return "observational"
    if "registry" in raw_type:
        return "registry"
    if "interventional" in raw_type:
        return "interventional"
    return "unknown"


def _build_exec_summary_context(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _count_tiers(scored_trials)
    risk = _collect_exec_risk_signals(scored_trials)
    top_trials_summary = "\n".join(
        f"- {t.get('brief_title', '')} ({t.get('trial_id', '')}): tier={t.get('tier', 'weak')}, score={float(t.get('score', 0.0)):.2f}, concern={t.get('key_concern', '')}"
        for t in scored_trials[:5]
    )
    patient_sum = _describe_patient(patient_profile)
    return {
        "patient_summary": patient_sum,
        "strong_count": strong,
        "moderate_count": moderate,
        "excluded_count": weak + disqualified,
        "total": len(scored_trials),
        "top_trials": top_trials_summary or "No trials evaluated",
        "missing_info": "See Information Gaps section.",
        "hard_exclusion_count": risk["hard_exclusion_count"],
        "uncertain_major_criteria_count": risk["uncertain_major_criteria_count"],
        "na_criteria_count": risk["na_criteria_count"],
        "critical_missing_count": risk["critical_missing_count"],
        "critical_missing_fields": ", ".join(risk["critical_missing_fields"]) or "none identified",
        "trial_type_signal": risk["trial_type_signal"],
    }


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _retrieval_failed_with_zero_trials(
    scored_trials: list[dict[str, Any]], qa_issues: Sequence[dict[str, Any] | str]
) -> bool:
    if scored_trials:
        return False
    issue_codes = {
        str(issue.get("code", "")).strip() for issue in qa_issues if isinstance(issue, dict)
    }
    return "RETRIEVAL_FAILED_EMPTY_RESULT" in issue_codes


@llm_retry
async def _invoke_report_plan_llm(chain: Any, context: dict[str, Any]) -> ReportPlan:
    result = await asyncio.wait_for(
        chain.ainvoke(
            context,
            config={"run_name": "report_plan", "tags": ["synthesis", "report"]},
        ),
        timeout=get_settings().llm_call_timeout_seconds,
    )
    if not isinstance(result, ReportPlan):
        raise ValueError(f"Unexpected report plan result type: {type(result)}")
    return result


async def generate_report_plan(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
    eligibility_verdicts: dict[str, dict[str, Any]],
    missing_info: list[dict[str, Any]],
    qa_issues: Sequence[dict[str, Any] | str],
) -> ReportPlan:
    key_concerns = [
        {
            "trial_id": str(trial.get("trial_id", "")),
            "key_concern": str(trial.get("key_concern", "")),
        }
        for trial in scored_trials
        if str(trial.get("key_concern", "")).strip()
    ]
    critical_missing_info = [
        {
            "trial_id": str(trial.get("trial_id", "")),
            "critical_missing_info": list(trial.get("critical_missing_info", []) or []),
        }
        for trial in scored_trials
    ]
    context = {
        "patient_profile": _serialize(patient_profile),
        "patient_summary": _describe_patient(patient_profile),
        "scored_trials": _serialize(scored_trials),
        "eligibility_verdicts": _serialize(eligibility_verdicts),
        "key_concerns": _serialize(key_concerns),
        "critical_missing_info": _serialize(critical_missing_info),
        "missing_info": _serialize(missing_info),
        "qa_issues": _serialize(qa_issues),
    }
    prompt = ChatPromptTemplate.from_template(build_synthesis_prompt())
    chain = prompt | get_llm().with_structured_output(ReportPlan)

    try:
        report_plan = await _invoke_report_plan_llm(chain, context)
    except Exception as exc:
        if _retrieval_failed_with_zero_trials(scored_trials, qa_issues):
            patient_summary = _describe_patient(patient_profile)
            return ReportPlan(
                patient_summary=patient_summary,
                executive_summary=(
                    "Trial retrieval failed before eligibility fan-out completed. "
                    "No matches are reported because zero trials were retrieved/evaluated."
                ),
                bottom_line="Unable to prioritize trials because retrieval returned zero evaluated trials.",
                strong_matches=[],
                moderate_matches=[],
                information_gaps=[],
                recommended_actions=[],
                excluded_summary="No trial exclusions available because no trials were evaluated.",
                limitations=["Trial retrieval failed before eligibility fan-out completed."],
            )
        logger.error("Report plan generation failed after retries: {}", exc)
        raise RuntimeError(f"Structured report planning failed: {exc}") from exc

    critical_missing_count = len(_deduped_critical_missing_fields(scored_trials))
    report_plan.executive_summary = _enforce_conservative_wording(
        report_plan.executive_summary,
        critical_missing_count,
    )
    return report_plan


async def generate_executive_summary(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, str]:
    report_plan = await generate_report_plan(
        patient_profile=patient_profile,
        scored_trials=scored_trials,
        eligibility_verdicts={},
        missing_info=[],
        qa_issues=[],
    )
    return {
        "executive_summary": report_plan.executive_summary,
        "patient_summary": report_plan.patient_summary,
    }
