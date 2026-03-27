from typing import Any

from agents import criteria_parser, eligibility_reasoner, missing_info, scorer
from config import settings
from langgraph.types import Send
from loguru import logger

from subagents.eligibility.state import EligibilityState, TrialWorkerState


def _patient_profile_to_dict(patient_profile: Any) -> dict[str, Any]:
    if isinstance(patient_profile, dict):
        return patient_profile
    if hasattr(patient_profile, "model_dump"):
        return patient_profile.model_dump()
    return {}


def _trial_id(trial: dict[str, Any]) -> str:
    return str(trial.get("nct_id") or trial.get("trial_id") or "unknown")


async def fan_out_trials(state: EligibilityState) -> dict[str, Any]:
    trials = list(state.get("trials_deduplicated") or [])
    cached = state.get("eligibility_verdicts") or {}
    new_trials = [t for t in trials if _trial_id(t) not in cached]
    logger.info(
        "Eligibility fan-out: dispatching {} trial workers ({} already cached)",
        len(new_trials),
        len(cached),
    )

    return {
        "trials_to_evaluate": new_trials,
        "processed_trials_with_criteria": [],
        "processed_verdicts": [],
        "decision_history": [
            f"Eligibility fan-out: {len(new_trials)} new trials, "
            f"{len(cached)} cached from prior round."
        ],
    }


def dispatch_trial_workers(state: EligibilityState) -> list[Send]:
    trials = list(state.get("trials_to_evaluate") or [])
    patient_profile = _patient_profile_to_dict(state.get("patient_profile"))
    return [
        Send(
            "evaluate_trial_worker",
            {"patient_profile": patient_profile, "trial": t},
        )
        for t in trials
    ]


async def evaluate_trial_worker(state: TrialWorkerState) -> dict[str, Any]:
    trial = state["trial"]
    patient_profile = _patient_profile_to_dict(state["patient_profile"])
    nct_id = _trial_id(trial)

    # Parse criteria
    try:
        parsed_trials = await criteria_parser.parse_criteria_for_trials([trial])
        trial_with_criteria = parsed_trials[0] if parsed_trials else {"trial": trial}
    except Exception as exc:
        logger.warning("Criteria parse failed for {}: {}", nct_id, exc)
        trial_with_criteria = {
            "trial": trial,
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "parse_error": str(exc),
        }

    inclusion = [
        {**c, "criteria_type": "inclusion"}
        for c in trial_with_criteria.get("inclusion_criteria", [])
    ]

    exclusion = [
        {**c, "criteria_type": "exclusion"}
        for c in trial_with_criteria.get("exclusion_criteria", [])
    ]

    all_criteria = inclusion + exclusion

    if not all_criteria:
        verdict_data = {
            "trial_id": nct_id,
            "verdicts": [],
            "meets_count": 0,
            "fails_count": 0,
            "uncertain_count": 0,
            "hard_exclusion_failures": 0,
        }
        return {
            "processed_trials_with_criteria": [trial_with_criteria],
            "processed_verdicts": [verdict_data],
        }

    # Batched eligibility call for the entire criteria
    try:
        verdict_data = await eligibility_reasoner.evaluate_criteria_batch(
            patient_profile=patient_profile, trial=trial, all_criteria=all_criteria
        )

    except Exception as exc:
        logger.warning("Eligibility reasoning failed for {}: {}", nct_id, exc)
        verdict_data = {
            "trial_id": nct_id,
            "verdicts": [],
            "meets_count": 0,
            "fails_count": 0,
            "uncertain_count": 0,
            "hard_exclusion_failures": 0,
            "error": str(exc),
        }

    return {
        "processed_trials_with_criteria": [trial_with_criteria],
        "processed_verdicts": [verdict_data],
    }


# Aggregation node


async def aggregate_results(state: EligibilityState):
    """Merge cached results with new ones then score"""
    cached = state.get("eligibility_verdicts") or {}
    new_trials_criteria = list(state.get("processed_trials_with_criteria") or [])
    new_verdicts_list = list(state.get("processed_verdicts") or [])

    # Merge cached  + new verdicts
    eligibility_verdicts = dict(cached)
    for vdata in new_verdicts_list:
        tid = vdata.get("trial_id", "")
        if tid:
            eligibility_verdicts[tid] = vdata

    # Score them
    scored = scorer.score_and_rank_trials(
        eligibility_verdicts,
        new_trials_criteria,
        trials_raw=list(state.get("trials_deduplicated") or []),
    )
    viable = scorer.count_viable_trials(scored)
    return {
        "trials_with_criteria": new_trials_criteria,
        "eligibility_verdicts": eligibility_verdicts,
        "trial_scores": scored,
        "viable_trial_count": viable,
        "decision_history": [
            f"Eligibility aggregation: {len(new_trials_criteria)} newly evaluated, "
            f"{len(cached)} from cache, {len(eligibility_verdicts)} total verdicts, "
            f"{viable} viable."
        ],
    }


async def identify_missing_info(state: EligibilityState):
    """Identify missing patient information that would resolve uncertainties."""
    patient_profile = _patient_profile_to_dict(state.get("patient_profile"))
    eligibility_verdicts = state.get("eligibility_verdicts") or {}
    # scored = state.get("trial_scores") or []
    try:
        recommendations = await missing_info.identify_missing_info(
            patient_profile, eligibility_verdicts
        )

        return {
            "missing_info_recommendations": recommendations,
            "decision_history": [f"Identified {len(recommendations)} missing info items."],
        }
    except Exception as exc:
        logger.error("identify_missing_info failed: {}", exc)
        return {
            "missing_info_recommendations": [],
            "decision_history": [f"identify_missing_info failed: {exc}"],
        }


async def assess_viability_signal(state: EligibilityState):
    """Assess if the supervisor needs to broaden retrieval."""
    viable = state.get("viable_trial_count", 0)
    needs_broadening = viable < settings.viable_trial_threshold

    broadening_msg = (
        "Signalling supervisor to broaden retrieval." if needs_broadening else "Sufficient results."
    )
    signal = (
        f"Viability assessment: {viable} viable trials "
        f"(threshold={settings.viable_trial_threshold}). "
        f"{broadening_msg}"
    )
    return {
        "retrieval_needs_broadening": needs_broadening,
        "decision_history": [signal],
    }
