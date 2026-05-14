import json
from typing import Any

from loguru import logger
from models.judge_verdict import JudgeVerdict
from tools.retry import llm_retry

from clinical_trial_agent.config import TIER_ORDER, get_llm

from .eligibility_fallback import FALLBACK_VERDICT, fallback_verdict_for_exception, validate_verdict
from .eligibility_prompt_builder import build_judge_messages


@llm_retry
async def _judge_trial(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    llm = get_llm(contains_phi=False, node_name="eligibility_judge")
    if hasattr(llm, "with_structured_output"):
        llm = llm.with_structured_output(JudgeVerdict)

    messages = build_judge_messages(patient_profile, trial, criteria)
    response = await llm.ainvoke(
        messages,
        config={"run_name": "eligibility_judge", "tags": ["eligibility", "judge"]},
    )
    trial_id = str(trial.get("nct_id", "unknown"))

    if isinstance(response, JudgeVerdict):
        return response
    if isinstance(response, dict):
        return validate_verdict(response, trial_id)

    content = getattr(response, "content", None)
    if isinstance(content, dict):
        return validate_verdict(content, trial_id)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return validate_verdict(parsed, trial_id)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    logger.warning(
        "Eligibility judge returned non-structured response for {}: {}",
        trial_id,
        type(response).__name__,
    )
    return validate_verdict(dict(FALLBACK_VERDICT), trial_id)


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


def _summarize_verdict_counts(
    verdicts: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    inclusion_meets = 0
    inclusion_fails = 0
    exclusion_triggered = 0
    uncertain = 0
    for verdict in verdicts:
        label = verdict.get("verdict")
        is_hard = bool(verdict.get("is_hard_exclusion"))
        if label == "MEETS":
            inclusion_meets += 1
        elif label == "FAILS":
            if is_hard:
                exclusion_triggered += 1
            else:
                inclusion_fails += 1
        elif label == "UNCERTAIN":
            uncertain += 1
    return (
        inclusion_meets,
        inclusion_fails + exclusion_triggered,
        uncertain,
        exclusion_triggered,
    )


def _build_batch_result(
    trial_id: str,
    trial: dict[str, Any],
    verdict: JudgeVerdict,
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    meets_count, fails_count, uncertain_count, hard_exclusion_failures = _summarize_verdict_counts(
        verdicts
    )
    adjusted_tier, adjusted_score = _apply_sparse_evidence_caps(
        verdict=verdict,
        meets_count=meets_count,
        fails_count=fails_count,
        uncertain_count=uncertain_count,
        hard_exclusion_failures=hard_exclusion_failures,
    )
    adjusted_tier, adjusted_score = _apply_criteria_provenance_caps(
        trial, adjusted_tier, adjusted_score
    )
    return {
        "trial_id": trial_id,
        "match_score": adjusted_score,
        "match_tier": adjusted_tier,
        "major_criteria_assessable": verdict.major_criteria_assessable,
        "critical_missing_info": verdict.critical_missing_info,
        "key_concern": verdict.key_concern,
        "rationale": verdict.rationale,
        "verdicts": verdicts,
        "meets_count": meets_count,
        "fails_count": fails_count,
        "uncertain_count": uncertain_count,
        "hard_exclusion_failures": hard_exclusion_failures,
        "criteria_source": trial.get("criteria_source", "missing"),
        "criteria_source_verified": bool(trial.get("criteria_source_verified", False)),
        "criteria_retrieved_at": trial.get("criteria_retrieved_at"),
        "criteria_completeness": trial.get("criteria_completeness", "missing"),
    }


def _apply_criteria_provenance_caps(
    trial: dict[str, Any], tier: str, score: float
) -> tuple[str, float]:
    source_verified = bool(trial.get("criteria_source_verified", False))
    completeness = str(trial.get("criteria_completeness", "missing"))
    source = str(trial.get("criteria_source", "missing"))
    if source_verified and completeness == "full":
        return tier, score
    if source == "missing" or completeness == "missing":
        if TIER_ORDER[tier] > TIER_ORDER["weak"]:
            tier = "weak"
        return tier, min(score, 0.45)
    if TIER_ORDER[tier] > TIER_ORDER["moderate"]:
        tier = "moderate"
    return tier, min(score, 0.65)


def _apply_sparse_evidence_caps(
    *,
    verdict: JudgeVerdict,
    meets_count: int,
    fails_count: int,
    uncertain_count: int,
    hard_exclusion_failures: int,
) -> tuple[str, float]:
    if hard_exclusion_failures > 0:
        return "disqualified", 0.0

    tier = verdict.match_tier
    score = verdict.match_score
    assessed_count = meets_count + fails_count
    total_count = assessed_count + uncertain_count

    if total_count == 0:
        if TIER_ORDER[tier] > TIER_ORDER["weak"]:
            tier = "weak"
        return tier, min(score, 0.25)

    if assessed_count <= 1:
        if TIER_ORDER[tier] > TIER_ORDER["weak"]:
            tier = "weak"
        return tier, min(score, 0.45)

    if assessed_count < 3 or (assessed_count / total_count) < 0.5:
        if TIER_ORDER[tier] > TIER_ORDER["moderate"]:
            tier = "moderate"
        return tier, min(score, 0.65)

    return tier, score


async def evaluate_criteria_batch(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_id = str(trial.get("nct_id", "unknown"))

    try:
        verdict = await _judge_trial(patient_profile, trial, all_criteria)
    except Exception as exc:
        verdict = fallback_verdict_for_exception(exc, trial_id, patient_profile, all_criteria)

    verdicts = _build_verdict_rows(verdict)
    return _build_batch_result(trial_id, trial, verdict, verdicts)
