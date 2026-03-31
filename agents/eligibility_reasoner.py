import asyncio
import json
from typing import Any

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.criteria import CriterionAssessment, EligibilityAssessmentList
from prompts.eligibility import build_eligibility_prompt


def _build_eligibility_chain(llm: Any) -> Any:
    prompt = ChatPromptTemplate.from_template(build_eligibility_prompt())
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


def _derive_base_verdict(llm_v: CriterionAssessment | None) -> tuple[str, str]:
    if llm_v is None:
        return "UNCERTAIN", "LLM failed to return an assessment for this criterion."
    return llm_v.verdict, llm_v.justification


def _derive_defaults(verdict: str, is_hard_exclusion: bool) -> tuple[float, list[str]]:
    if verdict == "FAILS":
        flags = ["evidence_of_exclusion"]
        if is_hard_exclusion:
            return 0.0, [*flags, "hard_exclusion"]
        return 0.2, flags
    if verdict == "UNCERTAIN":
        return 0.4, []
    return 0.8, []


def _parse_justification_payload(
    justification: str,
    match_score: float,
    flags: list[str],
) -> tuple[str, float, list[str]]:
    try:
        payload = json.loads(justification)
    except json.JSONDecodeError:
        return justification, match_score, flags
    if not isinstance(payload, dict):
        return justification, match_score, flags
    rationale = str(payload.get("rationale", justification))
    parsed_score = payload.get("match_score")
    if isinstance(parsed_score, int | float):
        match_score = max(0.0, min(1.0, float(parsed_score)))
    parsed_flags = payload.get("flags")
    if isinstance(parsed_flags, list):
        flags = [str(flag) for flag in parsed_flags]
    return rationale, match_score, flags


def _assemble_verdict(crit: dict, llm_v: CriterionAssessment | None) -> dict[str, Any]:
    verdict, justification = _derive_base_verdict(llm_v)
    match_score, flags = _derive_defaults(verdict, bool(crit.get("is_hard_exclusion")))
    justification, match_score, flags = _parse_justification_payload(
        str(justification), match_score, flags
    )
    return {
        "criterion_id": crit["criterion_id"],
        "criterion_text": crit["text"],
        "criterion_type": crit.get("criterion_type", "inclusion"),
        "verdict": verdict,
        "justification": justification,
        "is_hard_exclusion": crit.get("is_hard_exclusion", False),
        "match_score": match_score,
        "flags": flags,
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


async def evaluate_criteria_batch(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    nct_id = trial.get("nct_id", "unknown")
    verdicts, counts = await assess_patient_eligibility(patient_profile, all_criteria)
    return {
        "trial_id": nct_id,
        "verdicts": verdicts,
        **counts,
    }
