import asyncio
from typing import Any

from agents import criteria_parser, eligibility_reasoner, missing_info
from langgraph.types import Send
from loguru import logger
from tools.cache import get_cached_eligibility_verdict, set_cached_eligibility_verdict

from clinical_trial_agent.config import TIER_ORDER, get_settings

from . import nodes_helpers as _nodes_helpers
from .nodes_helpers import (
    _base_verdict_counts,
    _build_scored_trial,
    _build_trial_lookup,
    _collect_all_criteria,
    _empty_criteria_worker_result,
    _is_plausibly_relevant,
    _merge_eligibility_verdicts,
    _patient_profile_to_dict,
    _profile_hash_for_cache,
    _rank_trials,
    _trial_id,
    _worker_result,
)
from .state import EligibilityState, TrialWorkerState

_LLM_SEMAPHORE: asyncio.Semaphore | None = None

_normalize_criteria = _nodes_helpers._normalize_criteria
_collect_verdict_texts = _nodes_helpers._collect_verdict_texts
_tokenize = _nodes_helpers._tokenize


def set_llm_semaphore(semaphore: asyncio.Semaphore) -> None:
    global _LLM_SEMAPHORE
    _LLM_SEMAPHORE = semaphore


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
    trial = state.get("trial")
    if not isinstance(trial, dict):
        raise ValueError("Eligibility worker missing trial payload")
    patient_profile = _patient_profile_to_dict(state.get("patient_profile"))
    nct_id = _trial_id(trial)
    patient_condition = str(
        patient_profile.get("primary_condition") or (patient_profile.get("conditions") or [""])[0]
    )

    profile_hash = _profile_hash_for_cache(patient_profile)
    cached = _cached_worker_result(trial, nct_id, profile_hash)
    if cached is not None:
        return cached

    irrelevant = _irrelevant_worker_result(trial, nct_id, patient_condition)
    if irrelevant is not None:
        return irrelevant

    trial_with_criteria = await _parse_trial_criteria(trial, nct_id)
    all_criteria = _collect_all_criteria(trial_with_criteria)

    if not all_criteria:
        return _empty_criteria_worker_result(trial_with_criteria, nct_id)

    verdict_data = await _evaluate_with_optional_semaphore(
        patient_profile=patient_profile,
        trial=trial,
        all_criteria=all_criteria,
    )
    set_cached_eligibility_verdict(
        nct_id=nct_id,
        profile_hash=profile_hash,
        verdict=verdict_data,
        primary_completion_date=str(trial.get("primary_completion_date", "")),
    )
    return _worker_result(trial_with_criteria, verdict_data)


def _cached_worker_result(
    trial: dict[str, Any], nct_id: str, profile_hash: str
) -> dict[str, Any] | None:
    cached_verdict = get_cached_eligibility_verdict(nct_id, profile_hash)
    if cached_verdict is None:
        return None
    trial_stub = {"trial": trial, "inclusion_criteria": [], "exclusion_criteria": []}
    return _worker_result(trial_stub, cached_verdict)


def _irrelevant_worker_result(
    trial: dict[str, Any], nct_id: str, patient_condition: str
) -> dict[str, Any] | None:
    if not patient_condition or _is_plausibly_relevant(trial, patient_condition):
        return None
    verdict_data = {
        "trial_id": nct_id,
        "match_score": 0.0,
        "match_tier": "disqualified",
        "major_criteria_assessable": False,
        "critical_missing_info": [
            "Trial appears condition-mismatched to patient primary condition."
        ],
        "key_concern": "Trial condition appears unrelated to patient condition",
        "rationale": (
            "Structured eligibility assessment was deferred due to low condition/title "
            "overlap with patient primary condition."
        ),
        **_base_verdict_counts(),
    }
    trial_stub = {"trial": trial, "inclusion_criteria": [], "exclusion_criteria": []}
    return _worker_result(trial_stub, verdict_data)


async def _parse_trial_criteria(trial: dict[str, Any], nct_id: str) -> dict[str, Any]:
    try:
        parsed_trials = await criteria_parser.parse_criteria_for_trials([trial])
        return parsed_trials[0] if parsed_trials else {"trial": trial}
    except Exception as exc:
        logger.warning("Criteria parse failed for {}: {}", nct_id, exc)
        return {
            "trial": trial,
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "parse_error": str(exc),
        }


async def _evaluate_with_optional_semaphore(
    *, patient_profile: dict[str, Any], trial: dict[str, Any], all_criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    if _LLM_SEMAPHORE is None:
        return await eligibility_reasoner.evaluate_criteria_batch(
            patient_profile=patient_profile, trial=trial, all_criteria=all_criteria
        )
    async with _LLM_SEMAPHORE:
        return await eligibility_reasoner.evaluate_criteria_batch(
            patient_profile=patient_profile, trial=trial, all_criteria=all_criteria
        )


def _count_viable_trials(scored: list[dict[str, Any]]) -> int:
    return sum(
        1
        for trial in scored
        if TIER_ORDER.get(str(trial.get("tier", "weak")), 0)
        >= TIER_ORDER.get(get_settings().min_match_tier, TIER_ORDER["moderate"])
    )


async def aggregate_results(state: EligibilityState) -> dict[str, Any]:
    """Merge cached results with new LLM-judge verdicts and rank by tier/score."""
    cached = state.get("eligibility_verdicts") or {}
    new_trials_criteria = list(state.get("processed_trials_with_criteria") or [])
    new_verdicts_list = list(state.get("processed_verdicts") or [])

    eligibility_verdicts = _merge_eligibility_verdicts(cached, new_verdicts_list)
    trial_lookup = _build_trial_lookup(new_trials_criteria)

    scored: list[dict[str, Any]] = []
    for trial_id, verdict in eligibility_verdicts.items():
        scored.append(_build_scored_trial(str(trial_id), verdict, trial_lookup))

    ranked = _rank_trials(scored)
    viable = _count_viable_trials(ranked)
    return {
        "trials_with_criteria": new_trials_criteria,
        "eligibility_verdicts": eligibility_verdicts,
        "trial_scores": ranked,
        "viable_trial_count": viable,
        "decision_history": [
            f"Eligibility aggregation: {len(new_trials_criteria)} newly evaluated, "
            f"{len(cached)} from cache, {len(eligibility_verdicts)} total verdicts, "
            f"{viable} viable."
        ],
    }


async def identify_missing_info(state: EligibilityState) -> dict[str, Any]:
    """Identify missing patient information that would resolve uncertainties."""
    patient_profile = _patient_profile_to_dict(state.get("patient_profile"))
    eligibility_verdicts = state.get("eligibility_verdicts") or {}
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


async def assess_viability_signal(state: EligibilityState) -> dict[str, Any]:
    """Assess if retrieval needs broadening based on number of acceptable-tier trials."""
    viable = state.get("viable_trial_count", 0)
    needs_broadening = viable < 1

    broadening_msg = (
        "Signalling supervisor to broaden retrieval." if needs_broadening else "Sufficient results."
    )
    signal = (
        f"Viability assessment: {viable} viable trials "
        f"(threshold=1 at tier >= {get_settings().min_match_tier}). "
        f"{broadening_msg}"
    )
    return {
        "retrieval_needs_broadening": needs_broadening,
        "decision_history": [signal],
    }
