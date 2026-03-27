import asyncio
from typing import Any

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from models.criteria import CriterionAssessment, EligibilityAssessmentList
from prompts.eligibility import EVALUATION_PROMPT


def _build_eligibility_chain(llm: ChatOpenAI) -> Any:
    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
    structured_llm = llm.with_structured_output(EligibilityAssessmentList)
    return prompt | structured_llm


def _fmt_demographics(p: dict) -> Any:
    age, sex = p.get("age"), p.get("sex")
    if not age and not sex:
        return None
    return f"Demographics: {age or 'unknown age'} year old {sex or 'unknown sex'}"


def _fmt_conditions(p: dict) -> str | None:
    primary = p.get("primary_condition")
    conds = p.get("conditions", [])
    parts = []
    if primary:
        parts.append(f"Primary condition: {primary}")
    if conds:
        parts.append(f"Conditions: {', '.join(conds[:10])}")
    return "\n".join(parts) or None


def _fmt_medications(p: dict) -> str | None:
    meds = p.get("medications", [])
    if not meds:
        return None
    names = [m.get("name", "") if isinstance(m, dict) else str(m) for m in meds[:10]]
    return f"Medications: {', '.join(names)}"


def _fmt_labs(p: dict) -> str | None:
    labs = p.get("lab_values", [])
    if not labs:
        return None
    strs = [
        f"{lab['name']}: {lab['value']} {lab.get('unit', '')}"
        for lab in labs[:10]
        if isinstance(lab, dict)
    ]
    return f"Lab values: {'; '.join(strs)}" if strs else None


def _fmt_biomarkers(p: dict) -> str | None:
    bms = p.get("biomarkers", [])
    if not bms:
        return None
    strs = [f"{b['name']}: {b['result']}" for b in bms[:5] if isinstance(b, dict)]
    return f"Biomarkers: {'; '.join(strs)}" if strs else None


def _fmt_clinical_attrs(p: dict) -> list[str]:
    parts: list[str] = []
    ecog = p.get("ecog_performance_status")
    if ecog is not None:
        parts.append(f"ECOG PS: {ecog}")
    hist = p.get("medical_history", [])
    if hist:
        parts.append(f"Medical history: {', '.join(hist[:8])}")
    prior = p.get("prior_treatments", [])
    if prior:
        parts.append(f"Prior treatments: {', '.join(prior[:8])}")
    allergies = p.get("allergies", [])
    if allergies:
        parts.append(f"Allergies: {', '.join(allergies)}")
    contra = p.get("contraindications", [])
    if contra:
        parts.append(f"Contraindications: {', '.join(contra)}")
    bmi = p.get("bmi")
    if bmi:
        parts.append(f"BMI: {bmi}")
    return parts


def _format_patient_summary(patient_profile: dict[str, Any]) -> str:
    sections = [
        _fmt_demographics(patient_profile),
        _fmt_conditions(patient_profile),
        _fmt_medications(patient_profile),
        _fmt_labs(patient_profile),
        _fmt_biomarkers(patient_profile),
    ]
    parts = [s for s in sections if s]
    parts.extend(_fmt_clinical_attrs(patient_profile))
    return "\n".join(parts)


def _format_criteria_batch(criteria: list[dict[str, Any]]) -> str:
    lines = []
    for crit in criteria:
        ctype = crit.get("criterion_type", "inclusion")
        hard = " [HARD EXCLUSION]" if crit.get("is_hard_exclusion") else ""
        lines.append(f"[{crit['criterion_id']}] [{ctype.upper()}{hard}] {crit['text']}")
    return "\n".join(lines)


async def _invoke_eligibility_llm(chain: Any, inputs: dict[str, Any]) -> list[CriterionAssessment]:
    async with asyncio.Semaphore(10):
        result = await chain.ainvoke(
            inputs,
            config={
                "run_name": "eligibility_reasoning",
                "tags": ["eligibility", "verdict"],
            },
        )
        if isinstance(result, EligibilityAssessmentList):
            return result.results
        raise ValueError(f"Unexpected eligibility result type: {type(result)}")


def _assemble_verdict(crit: dict, llm_v: CriterionAssessment):
    if not llm_v:
        verdict = "UNCERTAIN"
        justification = "LLM failed to return an assessment for this criterion."
    else:
        verdict = llm_v.verdict
        justification = llm_v.justification
    return {
        "criterion_id": crit["criterion_id"],
        "criterion_text": crit["text"],
        "criterion_type": crit.get("criterion_type", "inclusion"),
        "verdict": verdict,
        "justification": justification,
        "is_hard_exclusion": crit.get("is_hard_exclusion", False),
        "confidence": 0.8 if verdict != "UNCERTAIN" else 0.4,
    }


def _build_verdicts(
    all_criteria: list[dict], raw_verdicts: list[CriterionAssessment]
) -> list[dict]:
    lookup = {v.criterion_id: v for v in raw_verdicts}
    return [_assemble_verdict(c, lookup.get(c["criterion_id"])) for c in all_criteria]


def _count_verdicts(verdicts: list[dict]) -> dict[str, int]:
    return {
        "meets_count": sum(1 for v in verdicts if v["verdict"] == "MEETS"),
        "fails_count": sum(1 for v in verdicts if v["verdict"] == "FAILS"),
        "uncertain_count": sum(1 for v in verdicts if v["verdict"] == "UNCERTAIN"),
        "hard_exclusion_failures": sum(
            1 for v in verdicts if v["verdict"] == "FAILS" and v["is_hard_exclusion"]
        ),
    }


async def assess_patient_eligibility(
    patient_profile: dict[str, Any], criteria: list[dict[str, Any]]
) -> tuple[list[dict], dict[str, int]]:
    llm = get_llm()
    chain = _build_eligibility_chain(llm)
    inputs = {
        "patient_profile": _format_patient_summary(patient_profile),
        "criteria_list": _format_criteria_batch(criteria),
    }

    try:
        raw_verdicts = await _invoke_eligibility_llm(chain, inputs)
    except Exception as e:
        logger.warning("Failed to assess eligibility: %s", e)
        raw_verdicts = []

    final_verdicts = _build_verdicts(criteria, raw_verdicts)
    counts = _count_verdicts(final_verdicts)
    return final_verdicts, counts
