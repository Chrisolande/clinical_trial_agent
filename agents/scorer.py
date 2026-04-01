from __future__ import annotations

from typing import Any

import numpy as np
from config import settings
from scipy.stats import binomtest
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

HARD_EXCLUSION_WEIGHT = -0.5


def _build_tier_classifier() -> DecisionTreeClassifier:
    scores = np.linspace(0.0, 1.0, 11)
    confidences = np.linspace(0.0, 1.0, 11)
    hf_rates = np.array([0.0, 0.10, 0.15, 0.20, 0.30, 0.50])

    S, C, H = np.meshgrid(scores, confidences, hf_rates)
    X = np.column_stack([S.ravel(), C.ravel(), H.ravel()])

    conditions = [
        (X[:, 2] >= 0.20),
        (X[:, 0] >= 0.7) & (X[:, 1] >= 0.5),
        (X[:, 0] >= 0.7) & (X[:, 1] >= 0.15),
        (X[:, 0] >= 0.5) & (X[:, 1] >= 0.25),
        (X[:, 0] >= 0.4) & (X[:, 1] >= 0.4),
    ]
    choices = [
        "unlikely_match",
        "strong_match",
        "possible_match",
        "possible_match",
        "possible_match",
    ]

    y = np.select(conditions, choices, default="unlikely_match")

    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X, y)
    return clf


# Initialize model at module load
_TIER_CLF = _build_tier_classifier()


def _wilson_ci(successes: int, total: int, hard_fails: int = 0) -> tuple[float, float]:
    """Scipy binomial test with a hard-fail veto."""
    if hard_fails > 0:
        return 0.0, 0.0
    if total == 0:
        return 0.0, 1.0

    successes = min(successes, total)
    ci = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="wilson")
    return round(float(ci.low), 4), round(float(ci.high), 4)


def _compute_score_confidence(
    meets: int, soft_fails: int, hard_fails: int, uncertain: int, total: int
) -> tuple[float, float]:
    known = meets + soft_fails + hard_fails
    score = 0.5 if known == 0 else (meets / known) + (hard_fails * HARD_EXCLUSION_WEIGHT)
    confidence = max(0.1, min(1.0, known / total)) if total > 0 else 0.1
    return round(max(0.0, min(1.0, score)), 4), round(confidence, 4)


def _collect_verdict_keys(verdicts: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    inc_passed = [
        v["criterion_text"]
        for v in verdicts
        if v["verdict"] == "MEETS" and v["criterion_type"] == "inclusion"
    ][:3]
    exc_failed = [v["criterion_text"] for v in verdicts if v["verdict"] == "FAILS"][:3]
    uncertain = [v["criterion_text"] for v in verdicts if v["verdict"] == "UNCERTAIN"][:3]
    return inc_passed, exc_failed, uncertain


def _format_locations(trial: dict[str, Any]) -> list[str]:
    return [
        f"{loc.get('city', '')}, {loc.get('country', '')}".strip(", ")
        for loc in trial.get("locations", [])[:3]
        if isinstance(loc, dict)
    ]


def _build_trial_lookup(
    trials_with_criteria: list[dict[str, Any]], trials_raw: list[dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for twc in trials_with_criteria:
        if nct_id := twc.get("trial", {}).get("nct_id"):
            trial_value = twc.get("trial")
            if isinstance(trial_value, dict):
                lookup[str(nct_id)] = trial_value

    for trial in trials_raw or []:
        if (nct_id := trial.get("nct_id")) and nct_id not in lookup:
            lookup[str(nct_id)] = trial
    return lookup


def score_trial(eligibility_result: dict[str, Any], trial_data: dict[str, Any]) -> dict[str, Any]:
    verdicts = eligibility_result.get("verdicts", [])
    trial = trial_data.get("trial", trial_data)

    if not verdicts:
        return _build_scored_trial(trial, 0.0, 0, 0, 0, 0)

    meets = eligibility_result.get("meets_count", 0)
    fails = eligibility_result.get("fails_count", 0)
    uncertain = eligibility_result.get("uncertain_count", 0)
    hard_fails = eligibility_result.get("hard_exclusion_failures", 0)

    total = meets + fails + uncertain
    score, confidence = _compute_score_confidence(
        meets, fails - hard_fails, hard_fails, uncertain, total
    )
    inc_passed, exc_failed, unc_list = _collect_verdict_keys(verdicts)

    return {
        "trial_id": trial.get("nct_id", ""),
        "brief_title": trial.get("brief_title", ""),
        "overall_status": trial.get("overall_status", ""),
        "phase": trial.get("phase"),
        "lead_sponsor": trial.get("lead_sponsor"),
        "score": score,
        "confidence": confidence,
        "meets_count": meets,
        "fails_count": fails,
        "uncertain_count": uncertain,
        "hard_exclusion_failures": hard_fails,
        "key_inclusion_passed": inc_passed,
        "key_exclusion_failed": exc_failed,
        "key_uncertain": unc_list,
        "locations_summary": _format_locations(trial),
        "primary_completion_date": trial.get("primary_completion_date"),
    }


def _build_scored_trial(
    trial: dict[str, Any],
    score: float,
    meets: int,
    fails: int,
    uncertain: int,
    hard_fails: int,
) -> dict[str, Any]:
    return {
        "trial_id": trial.get("nct_id", ""),
        "brief_title": trial.get("brief_title", ""),
        "overall_status": trial.get("overall_status", ""),
        "phase": trial.get("phase"),
        "lead_sponsor": trial.get("lead_sponsor"),
        "score": score,
        "confidence": 0.1,
        "meets_count": meets,
        "fails_count": fails,
        "uncertain_count": uncertain,
        "hard_exclusion_failures": hard_fails,
        "key_inclusion_passed": [],
        "key_exclusion_failed": [],
        "key_uncertain": [],
        "locations_summary": [],
        "primary_completion_date": trial.get("primary_completion_date"),
    }


def score_and_rank_trials(
    eligibility_verdicts: dict[str, dict[str, Any]],
    trials_with_criteria: list[dict[str, Any]],
    trials_raw: list | None = None,
) -> list[dict[str, Any]]:
    trial_lookup = _build_trial_lookup(trials_with_criteria, trials_raw)

    scored = [
        score_trial(verdict, trial_lookup.get(nct_id, {"nct_id": nct_id}))
        for nct_id, verdict in eligibility_verdicts.items()
    ]

    if not scored:
        return scored

    # ML Pipeline: Normalization and Prediction

    raw_scores = np.array([t["score"] for t in scored]).reshape(-1, 1)
    if raw_scores.max() == raw_scores.min():
        normalized = raw_scores.flatten()
    else:
        normalized = MinMaxScaler().fit_transform(raw_scores).flatten()

    confidences = np.array([t["confidence"] for t in scored])
    hard_fails = np.array([t["hard_exclusion_failures"] for t in scored])
    totals = np.array([t["meets_count"] + t["fails_count"] for t in scored])

    with np.errstate(divide="ignore", invalid="ignore"):
        hf_rates = np.where(totals > 0, hard_fails / totals, 0.0)

    tier_features = np.column_stack([normalized, confidences, hf_rates])

    # Predict all tiers at once
    tiers = _TIER_CLF.predict(tier_features)

    for i, trial in enumerate(scored):
        trial["score"] = round(float(normalized[i]), 4)
        trial["tier"] = tiers[i]

        tot = trial["meets_count"] + trial["fails_count"] + trial["uncertain_count"]
        trial["score_ci_lower"], trial["score_ci_upper"] = _wilson_ci(
            trial["meets_count"], tot, trial["hard_exclusion_failures"]
        )

    scored.sort(key=lambda x: (x["score"], x["confidence"], x["meets_count"]), reverse=True)

    for rank, trial in enumerate(scored, 1):
        trial["rank"] = rank

    return scored


def count_viable_trials(scored_trials: list[dict[str, Any]], min_score: float | None = None) -> int:
    """Count trials with normalized score above threshold."""
    if min_score is None:
        min_score = settings.min_match_score
    return sum(1 for t in scored_trials if t["score"] >= min_score)
