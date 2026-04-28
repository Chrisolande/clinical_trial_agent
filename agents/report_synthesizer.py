import asyncio
from collections import Counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from prompts.synthesis import build_synthesis_prompt
from pydantic import BaseModel, Field
from tools.retry import llm_retry

from clinical_trial_agent.config import get_llm, get_settings


class ExecutiveSummaryModel(BaseModel):
    executive_summary: str = Field(
        description="A 150-250 word physician-level executive summary of the trial matches."
    )
    patient_summary: str = Field(
        description="A concise 1-sentence demographic and clinical description of the patient"
    )


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
    hard_exclusion_count = sum(int(t.get("hard_exclusion_failures", 0) or 0) for t in scored_trials)
    uncertain_major_criteria_count = sum(
        1 for t in scored_trials if t.get("major_criteria_assessable") is False
    )
    na_criteria_count = sum(
        int(
            t.get("na_count")
            or t.get("n_a_count")
            or t.get("not_applicable_count")
            or t.get("not_applicable_criteria_count")
            or 0
        )
        for t in scored_trials
    )

    missing_fields: list[str] = []
    for trial in scored_trials:
        for item in trial.get("critical_missing_info", []) or []:
            value = str(item).strip()
            if value:
                missing_fields.append(value)
    deduped_missing_fields = list(dict.fromkeys(missing_fields))

    type_counts: Counter[str] = Counter()
    for trial in scored_trials:
        raw_type = str(trial.get("study_type") or trial.get("trial_type") or "").strip().lower()
        if "observational" in raw_type:
            type_counts["observational"] += 1
        elif "registry" in raw_type:
            type_counts["registry"] += 1
        elif "interventional" in raw_type:
            type_counts["interventional"] += 1
    if type_counts:
        trial_type_signal = ", ".join(
            f"{label}={type_counts[label]}"
            for label in ("interventional", "observational", "registry")
            if type_counts[label] > 0
        )
    else:
        trial_type_signal = "not available"

    return {
        "hard_exclusion_count": hard_exclusion_count,
        "uncertain_major_criteria_count": uncertain_major_criteria_count,
        "na_criteria_count": na_criteria_count,
        "critical_missing_count": len(deduped_missing_fields),
        "critical_missing_fields": deduped_missing_fields[:12],
        "trial_type_signal": trial_type_signal,
    }


def _build_exec_summary_context(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _count_tiers(scored_trials)
    risk = _collect_exec_risk_signals(scored_trials)
    top_trials_summary = "\n".join(
        f"- {t['brief_title']} ({t['trial_id']}): tier={t['tier']}, score={t['score']:.2f}, concern={t.get('key_concern', '')}"
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


@llm_retry
async def _invoke_exec_summary_llm(chain: Any, context: dict[str, Any]) -> dict[str, Any]:
    result = await asyncio.wait_for(
        chain.ainvoke(
            context,
            config={"run_name": "executive_summary", "tags": ["synthesis", "report"]},
        ),
        timeout=get_settings().llm_call_timeout_seconds,
    )
    if not isinstance(result, ExecutiveSummaryModel):
        raise ValueError(f"Unexpected report summary result type: {type(result)}")
    return result.model_dump()


async def generate_executive_summary(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, str]:
    context = _build_exec_summary_context(patient_profile, scored_trials)
    prompt = ChatPromptTemplate.from_template(build_synthesis_prompt())
    chain = prompt | get_llm().with_structured_output(ExecutiveSummaryModel)

    try:
        result = await _invoke_exec_summary_llm(chain, context)
        return {
            "executive_summary": _enforce_conservative_wording(
                result["executive_summary"], int(context["critical_missing_count"])
            ),
            "patient_summary": result["patient_summary"],
        }
    except Exception as exc:
        logger.error("Executive summary generation failed after retries: {}", exc)

    return {
        "executive_summary": (
            f"Conservative triage completed for {context['total']} trials. "
            f"Strong: {context['strong_count']}, Moderate: {context['moderate_count']}, "
            f"Excluded (weak/disqualified): {context['excluded_count']}."
        ),
        "patient_summary": context["patient_summary"],
    }
