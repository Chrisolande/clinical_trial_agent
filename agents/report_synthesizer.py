import asyncio
from collections import Counter
from typing import Any

from config import get_llm, get_settings
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from prompts.synthesis import build_synthesis_prompt
from pydantic import BaseModel, Field
from tools.retry import llm_retry


class ExecutiveSummaryModel(BaseModel):
    executive_summary: str = Field(
        description="A 150-250 word physician-level executive summary of the trial matches."
    )
    patient_summary: str = Field(
        description="A concise 1-sentence demographic and clinical description of the patient"
    )


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


def _build_exec_summary_context(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _count_tiers(scored_trials)
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
            "executive_summary": result["executive_summary"],
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
