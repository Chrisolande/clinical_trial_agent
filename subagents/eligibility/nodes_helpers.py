import hashlib
import json
import os
from typing import Any

from tools.medical_synonyms import expand_condition_tokens

from clinical_trial_agent.config import TIER_ORDER


def _patient_profile_to_dict(patient_profile: Any) -> dict[str, Any]:
    if isinstance(patient_profile, dict):
        return patient_profile
    if hasattr(patient_profile, "model_dump"):
        dumped = patient_profile.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _trial_id(trial: dict[str, Any]) -> str:
    return str(trial.get("nct_id") or trial.get("trial_id") or "unknown")


def _profile_hash_for_cache(patient_profile: dict[str, Any]) -> str:
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)
    salt = os.getenv("PROFILE_HASH_SALT", "")
    return hashlib.sha256(f"{salt}::{canonical}".encode()).hexdigest()


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    stopwords = {"a", "of", "in", "for", "the", "and", "with", "to", "or"}
    return {tok for tok in cleaned.split() if len(tok) > 2 and tok not in stopwords}


def _is_plausibly_relevant(trial: dict[str, Any], patient_condition: str) -> bool:
    patient_tokens = expand_condition_tokens(_tokenize(patient_condition))
    if not patient_tokens:
        return True
    trial_conditions = " ".join(str(c) for c in trial.get("conditions", []))
    trial_title = str(trial.get("brief_title", ""))
    combined_tokens = expand_condition_tokens(_tokenize(f"{trial_conditions} {trial_title}"))
    return bool(patient_tokens & combined_tokens)


def _worker_result(
    trial_with_criteria: dict[str, Any], verdict_data: dict[str, Any]
) -> dict[str, Any]:
    return {
        "processed_trials_with_criteria": [trial_with_criteria],
        "processed_verdicts": [verdict_data],
    }


def _normalize_criteria(criteria: list[Any], criteria_type: str) -> list[dict[str, Any]]:
    return [
        (
            {**c, "criteria_type": criteria_type}
            if isinstance(c, dict)
            else {"text": str(c), "criteria_type": criteria_type}
        )
        for c in criteria
    ]


def _collect_all_criteria(trial_with_criteria: dict[str, Any]) -> list[dict[str, Any]]:
    inclusion = _normalize_criteria(trial_with_criteria.get("inclusion_criteria", []), "inclusion")
    exclusion = _normalize_criteria(trial_with_criteria.get("exclusion_criteria", []), "exclusion")
    return inclusion + exclusion


def _empty_criteria_worker_result(
    trial_with_criteria: dict[str, Any], nct_id: str
) -> dict[str, Any]:
    verdict_data = {
        "trial_id": nct_id,
        "match_score": 0.1,
        "match_tier": "weak",
        "major_criteria_assessable": False,
        "critical_missing_info": ["No parsed criteria available."],
        "key_concern": "No assessable criteria parsed",
        "rationale": "Eligibility criteria could not be parsed.",
        **_base_verdict_counts(),
    }
    return _worker_result(trial_with_criteria, verdict_data)


def _base_verdict_counts() -> dict[str, Any]:
    return {
        "verdicts": [],
        "meets_count": 0,
        "fails_count": 0,
        "uncertain_count": 0,
        "hard_exclusion_failures": 0,
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
