"""Trial scoring and ranking agent."""

from __future__ import annotations

from typing import Any

import numpy as np
from config import settings
from scipy.stats import binomtest, rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

HARD_EXCLUSION_WEIGHT = -0.5


def _build_tier_classifier() -> DecisionTreeClassifier:
    scores = np.linspace(0.0, 1.0, 11)
    confidences = np.linspace(0.0, 1.0, 11)
    hf_rates = [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]

    X_rows: list[list[float]] = []
    y_rows: list[str] = []

    for s in scores:
        for c in confidences:
            for hfr in hf_rates:
                if hfr >= 0.20:
                    label = "unlikely_match"
                elif s >= 0.7 and c >= 0.3:
                    label = "strong_match"
                elif s >= 0.7 and c >= 0.15 or s >= 0.5 and c >= 0.25 or s >= 0.4 and c >= 0.4:
                    label = "possible_match"
                else:
                    label = "unlikely_match"
                X_rows.append([s, c, hfr])
                y_rows.append(label)

    clf = DecisionTreeClassifier(max_depth=6, random_state=0)
    clf.fit(np.array(X_rows), y_rows)
    return clf


_TIER_CLF = _build_tier_classifier()


def _classify_tier(score: float, hard_fails: int, confidence: float, total_known: int) -> str:
    hard_fail_rate = hard_fails / total_known if total_known > 0 else 0.0
    features = np.array([[score, confidence, hard_fail_rate]])
    return str(_TIER_CLF.predict(features)[0])


# Wilson confidence interval


def _wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    successes = min(successes, total)
    ci = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="wilson")
    return round(float(ci.low), 4), round(float(ci.high), 4)


# Score computation


def _compute_score_confidence(meets, soft_fails, hard_fails, uncertain, total):
    known = meets + soft_fails + hard_fails
    score = 0.5 if known == 0 else meets / known + hard_fails * HARD_EXCLUSION_WEIGHT
    confidence = max(0.1, known / total) if total > 0 else 0.1
    score = max(0.0, min(1.0, score))
    return round(score, 4), round(confidence, 4)


# Verdict key helpers


def _collect_verdict_keys(verdicts: list[dict]) -> tuple[list, list, list]:
    inc_passed = [
        v["criterion_text"]
        for v in verdicts
        if v["verdict"] == "MEETS" and v["criterion_type"] == "inclusion"
    ][:3]
    exc_failed = [v["criterion_text"] for v in verdicts if v["verdict"] == "FAILS"][:3]
    uncertain = [v["criterion_text"] for v in verdicts if v["verdict"] == "UNCERTAIN"][:3]
    return inc_passed, exc_failed, uncertain


def _format_locations(trial: dict) -> list[str]:
    return [
        f"{loc.get('city', '')}, {loc.get('country', '')}".strip(", ")
        for loc in trial.get("locations", [])[:3]
        if isinstance(loc, dict)
    ]


# Per-trial scoring


def score_trial(eligibility_result, trial_data):
    verdicts = eligibility_result.get("verdicts", [])
    trial = trial_data.get("trial", trial_data)
    if not verdicts:
        return _build_scored_trial(trial, 0.0, 0, 0, 0, 0)

    meets = eligibility_result.get("meets_count", 0)
    fails = eligibility_result.get("fails_count", 0)
    uncertain = eligibility_result.get("uncertain_count", 0)
    hard_fails = eligibility_result.get("hard_exclusion_failures", 0)
    total_known = meets + fails
    total = len(verdicts)

    score, confidence = _compute_score_confidence(
        meets, fails - hard_fails, hard_fails, uncertain, total
    )
    inc_passed, exc_failed, unc_list = _collect_verdict_keys(verdicts)
    ci_lower, ci_upper = _wilson_ci(meets, total)

    return {
        "trial_id": trial.get("nct_id", ""),
        "brief_title": trial.get("brief_title", ""),
        "overall_status": trial.get("overall_status", ""),
        "phase": trial.get("phase"),
        "lead_sponsor": trial.get("lead_sponsor"),
        "score": round(score, 4),
        "tier": _classify_tier(score, hard_fails, confidence, total_known),
        "confidence": round(confidence, 4),
        "score_ci_lower": ci_lower,
        "score_ci_upper": ci_upper,
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
    ci_lower, ci_upper = _wilson_ci(meets, meets + fails + uncertain)
    return {
        "trial_id": trial.get("nct_id", ""),
        "brief_title": trial.get("brief_title", ""),
        "overall_status": trial.get("overall_status", ""),
        "phase": trial.get("phase"),
        "lead_sponsor": trial.get("lead_sponsor"),
        "score": score,
        "tier": _classify_tier(score, hard_fails, 0.0, meets + fails),
        "confidence": 0.1,
        "score_ci_lower": ci_lower,
        "score_ci_upper": ci_upper,
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


# Batch scoring, normalization, and ranking


def score_and_rank_trials(eligibility_verdicts, trials_with_criteria, trials_raw=None):
    trial_lookup = {
        twc.get("trial", {}).get("nct_id", ""): twc
        for twc in trials_with_criteria
        if twc.get("trial", {}).get("nct_id", "")
    }
    for trial in trials_raw or []:
        nct_id = trial.get("nct_id", "")
        if nct_id and nct_id not in trial_lookup:
            trial_lookup[nct_id] = {"trial": trial}

    scored = [
        score_trial(verdict, trial_lookup.get(nct_id, {"trial": {"nct_id": nct_id}}))
        for nct_id, verdict in eligibility_verdicts.items()
    ]
    if not scored:
        return scored

    raw_scores = np.array([t["score"] for t in scored], dtype=float).reshape(-1, 1)
    normalized = (
        raw_scores.flatten()
        if raw_scores.max() == raw_scores.min()
        else MinMaxScaler().fit_transform(raw_scores).flatten()
    )
    ranks = (len(scored) + 1) - rankdata(normalized, method="average")

    for trial, norm_score, rank in zip(scored, normalized, ranks, strict=False):
        norm_score_f = round(float(norm_score), 4)
        trial["score"] = norm_score_f
        trial["rank"] = int(rank)
        trial["score_ci_lower"], trial["score_ci_upper"] = _wilson_ci(
            trial["meets_count"], len(trial.get("key_inclusion_passed", []))
        )
        trial["tier"] = _classify_tier(
            norm_score_f,
            trial["hard_exclusion_failures"],
            trial["confidence"],
            trial["meets_count"] + trial["fails_count"],
        )

    return sorted(scored, key=lambda x: x["rank"])


def count_viable_trials(
    scored_trials: list[dict[str, Any]],
    min_score: float | None = None,
) -> int:
    """Count trials with normalized score above threshold."""
    if min_score is None:
        min_score = settings.min_match_score
    return sum(1 for t in scored_trials if t["score"] >= min_score)
