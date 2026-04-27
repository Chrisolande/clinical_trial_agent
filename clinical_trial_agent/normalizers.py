import json
from typing import Any


def _require_dict(value: Any, *, source: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"{source} must return dict, got {type(value)!r}")


def _normalize_retrieval_result(value: dict[str, Any]) -> dict[str, Any]:
    retrieval_errors = list(value.get("retrieval_errors", value.get("errors", [])))
    return {
        **value,
        "trials_raw": list(value.get("trials_raw", [])),
        "trials_deduplicated": list(value.get("trials_deduplicated", [])),
        "search_queries": list(value.get("search_queries", [])),
        "retrieval_errors": retrieval_errors,
        "retrieval_failed": bool(value.get("retrieval_failed", bool(retrieval_errors))),
        "errors": retrieval_errors,
    }


def _normalize_eligibility_result(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "trial_scores": list(value.get("trial_scores", [])),
        "decision_history": list(value.get("decision_history", [])),
        "missing_info_recommendations": list(value.get("missing_info_recommendations", [])),
        "retrieval_needs_broadening": bool(value.get("retrieval_needs_broadening", False)),
    }


def _normalize_supervisor_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"report_json": None, "report_text": str(value)}
    result = dict(value)
    report_json = result.get("report_json")
    if not isinstance(report_json, dict):
        result["report_json"] = None
    report_text = result.get("report_text")
    if not isinstance(report_text, str) or not report_text.strip():
        result["report_text"] = (
            json.dumps(result["report_json"], default=str)
            if isinstance(result["report_json"], dict)
            else ""
        )
    return result
