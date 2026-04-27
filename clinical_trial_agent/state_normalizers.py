from typing import Any


def require_dict(value: Any, *, source: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"{source} must return dict, got {type(value)!r}")


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_retrieval_result(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "trials_raw": as_list(value.get("trials_raw")),
        "trials_deduplicated": as_list(value.get("trials_deduplicated")),
        "search_queries": as_list(value.get("search_queries")),
        "decision_history": as_list(value.get("decision_history")),
        "errors": as_list(value.get("errors")),
    }


def normalize_eligibility_result(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "trial_scores": as_list(value.get("trial_scores")),
        "trials_with_criteria": as_list(value.get("trials_with_criteria")),
        "eligibility_verdicts": as_dict(value.get("eligibility_verdicts")),
        "decision_history": as_list(value.get("decision_history")),
        "missing_info_recommendations": as_list(value.get("missing_info_recommendations")),
        "errors": as_list(value.get("errors")),
        "retrieval_needs_broadening": bool(value.get("retrieval_needs_broadening", False)),
    }
