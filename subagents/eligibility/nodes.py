from typing import Any, cast

from agents import criteria_parser, eligibility_reasoner, missing_info
from config import TIER_ORDER, settings
from langgraph.types import Send
from loguru import logger

from .state import EligibilityState, TrialWorkerState


def _patient_profile_to_dict(patient_profile: Any) -> dict[str, Any]:
    if isinstance(patient_profile, dict):
        return patient_profile
    if hasattr(patient_profile, "model_dump"):
        return cast("dict[str, Any]", patient_profile.model_dump())
    return {}


def _trial_id(trial: dict[str, Any]) -> str:
    return str(trial.get("nct_id") or trial.get("trial_id") or "unknown")


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    stopwords = {"a", "of", "in", "for", "the", "and", "with", "to", "or"}
    return {tok for tok in cleaned.split() if len(tok) > 2 and tok not in stopwords}


def _is_plausibly_relevant(trial: dict[str, Any], patient_condition: str) -> bool:
    patient_tokens = _tokenize(patient_condition)
    if not patient_tokens:
        return True
    trial_conditions = " ".join(str(c) for c in trial.get("conditions", []))
    trial_title = str(trial.get("brief_title", ""))
    combined_tokens = _tokenize(f"{trial_conditions} {trial_title}")
    return bool(patient_tokens & combined_tokens)


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
    patient_condition = str(
        patient_profile.get("primary_condition") or (patient_profile.get("conditions") or [""])[0]
    )

    if patient_condition and not _is_plausibly_relevant(trial, patient_condition):
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
                "Skipped LLM eligibility evaluation due to low condition/title token overlap "
                "with patient primary condition."
            ),
            "verdicts": [],
            "meets_count": 0,
            "fails_count": 0,
            "uncertain_count": 0,
            "hard_exclusion_failures": 0,
        }
        return {
            "processed_trials_with_criteria": [
                {"trial": trial, "inclusion_criteria": [], "exclusion_criteria": []}
            ],
            "processed_verdicts": [verdict_data],
        }

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
        (
            {**c, "criteria_type": "inclusion"}
            if isinstance(c, dict)
            else {"text": str(c), "criteria_type": "inclusion"}
        )
        for c in trial_with_criteria.get("inclusion_criteria", [])
    ]

    exclusion = [
        (
            {**c, "criteria_type": "exclusion"}
            if isinstance(c, dict)
            else {"text": str(c), "criteria_type": "exclusion"}
        )
        for c in trial_with_criteria.get("exclusion_criteria", [])
    ]

    all_criteria = inclusion + exclusion

    if not all_criteria:
        verdict_data = {
            "trial_id": nct_id,
            "match_score": 0.1,
            "match_tier": "weak",
            "major_criteria_assessable": False,
            "critical_missing_info": ["No parsed criteria available."],
            "key_concern": "No assessable criteria parsed",
            "rationale": "Eligibility criteria could not be parsed.",
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

    verdict_data = await eligibility_reasoner.evaluate_criteria_batch(
        patient_profile=patient_profile, trial=trial, all_criteria=all_criteria
    )

    return {
        "processed_trials_with_criteria": [trial_with_criteria],
        "processed_verdicts": [verdict_data],
    }


def _merge_eligibility_verdicts(
    cached: dict[str, Any], new_verdicts: list[dict[str, Any]]
) -> dict[str, Any]:
    merged = dict(cached)
    for verdict_data in new_verdicts:
        trial_id = str(verdict_data.get("trial_id", ""))
        if trial_id:
            merged[trial_id] = verdict_data
    return merged


def _build_trial_lookup(new_trials_criteria: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    trial_lookup: dict[str, dict[str, Any]] = {}
    for trial_with_criteria in new_trials_criteria:
        trial_data = trial_with_criteria.get("trial", {})
        trial_id = str(trial_data.get("nct_id", ""))
        if trial_id:
            trial_lookup[trial_id] = trial_data
    return trial_lookup


def _collect_verdict_texts(
    verdict_items: list[dict[str, Any]], *, verdict_name: str, criteria_type: str | None = None
) -> list[str]:
    values: list[str] = []
    for item in verdict_items:
        if item.get("verdict") != verdict_name:
            continue
        if criteria_type is not None and item.get("criterion_type") != criteria_type:
            continue
        values.append(str(item.get("criterion_text", "")))
    return values[:3]


def _build_scored_trial(
    trial_id: str,
    verdict: dict[str, Any],
    trial_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trial = trial_lookup.get(trial_id, {"nct_id": trial_id})
    verdicts = list(verdict.get("verdicts", []))
    return {
        "trial_id": trial_id,
        "brief_title": trial.get("brief_title", ""),
        "overall_status": trial.get("overall_status", ""),
        "phase": trial.get("phase"),
        "lead_sponsor": trial.get("lead_sponsor"),
        "score": float(verdict.get("match_score", 0.1)),
        "tier": str(verdict.get("match_tier", "weak")),
        "meets_count": int(verdict.get("meets_count", 0)),
        "fails_count": int(verdict.get("fails_count", 0)),
        "uncertain_count": int(verdict.get("uncertain_count", 0)),
        "hard_exclusion_failures": int(verdict.get("hard_exclusion_failures", 0)),
        "major_criteria_assessable": bool(verdict.get("major_criteria_assessable", False)),
        "key_concern": str(verdict.get("key_concern", "")),
        "critical_missing_info": list(verdict.get("critical_missing_info", [])),
        "rationale": str(verdict.get("rationale", "")),
        "key_inclusion_passed": _collect_verdict_texts(
            verdicts, verdict_name="MEETS", criteria_type="inclusion"
        ),
        "key_exclusion_failed": _collect_verdict_texts(
            verdicts, verdict_name="FAILS", criteria_type="exclusion"
        ),
        "key_uncertain": _collect_verdict_texts(verdicts, verdict_name="UNCERTAIN"),
        "locations_summary": [
            f"{loc.get('city', '')}, {loc.get('country', '')}".strip(", ")
            for loc in trial.get("locations", [])[:3]
            if isinstance(loc, dict)
        ],
        "primary_completion_date": trial.get("primary_completion_date"),
        "verdict_details": verdict,
    }


def _rank_trials(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        scored,
        key=lambda x: (
            TIER_ORDER.get(str(x.get("tier", "weak")), 0),
            float(x.get("score", 0.0)),
        ),
        reverse=True,
    )
    for rank, trial in enumerate(ranked, 1):
        trial["rank"] = rank
    return ranked


def _count_viable_trials(scored: list[dict[str, Any]]) -> int:
    return sum(
        1
        for trial in scored
        if TIER_ORDER.get(str(trial.get("tier", "weak")), 0)
        >= TIER_ORDER.get(settings.min_match_tier, TIER_ORDER["moderate"])
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
        f"(threshold=1 at tier >= {settings.min_match_tier}). "
        f"{broadening_msg}"
    )
    return {
        "retrieval_needs_broadening": needs_broadening,
        "decision_history": [signal],
    }
