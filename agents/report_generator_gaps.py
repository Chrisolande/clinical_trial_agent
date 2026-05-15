import re
from typing import Any

_NOT_APPLICABLE_MARKERS = ("not_applicable", "not applicable", "n/a", "non-applicable")
_INTERNAL_FAILURE_MARKERS = (
    "judge model",
    "structured verdict",
    "llm",
    "parser",
    "fallback",
    "tool failed",
    "qa issue",
    "qa check",
    "model returned none",
)
_LOW_VALUE_GAP_MARKERS = {
    "criterion requires details not present in profile",
    "missing trial-specific clinical detail",
    "missing exclusion-history detail",
    "additional clinical detail",
    "n/a",
    "unknown",
    "none",
}
_PUBLIC_LOW_PRIORITY_GAP_LIMIT = 2


def _severity_for_missing_item(item: str, tier: str) -> str:
    text = item.lower()
    high_markers = [
        "egfr",
        "alk",
        "ros1",
        "her2",
        "pd-l1",
        "stage",
        "ecog",
        "karnofsky",
        "diagnosis",
        "prior treatment",
        "measurable disease",
    ]
    if any(marker in text for marker in high_markers):
        return "high"
    if tier in {"weak", "disqualified"}:
        return "medium"
    return "low"


def _priority_rank(priority: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(priority.lower(), 0)


def _normalize_gap_item(raw_item: str) -> dict[str, str]:
    slug = re.sub(r"[^a-z0-9]+", "_", raw_item.lower()).strip("_")
    return {
        "field_id": slug or "additional_clinical_detail",
        "display_name": raw_item,
        "category": "clinical",
        "why_needed": raw_item,
    }


def _gap_payload(
    normalized: dict[str, str], raw_item: str, tier: str, trial_id: str
) -> dict[str, Any]:
    return {
        "field_id": normalized["field_id"],
        "display_name": normalized["display_name"],
        "category": normalized["category"],
        "field": normalized["display_name"],
        "why_needed": normalized["why_needed"],
        "evidence_text": normalized["why_needed"],
        "description": normalized["why_needed"],
        "priority": _severity_for_missing_item(raw_item, tier),
        "affected_trial_ids": [trial_id] if trial_id else [],
    }


def _merge_gap(grouped: dict[str, dict[str, Any]], gap: dict[str, Any]) -> None:
    key = str(gap["field_id"])
    if key not in grouped:
        grouped[key] = gap
        return
    current = grouped[key]
    current["affected_trial_ids"] = sorted(
        set(current.get("affected_trial_ids", [])) | set(gap.get("affected_trial_ids", []))
    )
    if _priority_rank(str(gap.get("priority", "low"))) > _priority_rank(
        str(current.get("priority", "low"))
    ):
        current["priority"] = gap["priority"]


def _gaps_from_scored_trial(trial: dict[str, Any]) -> list[dict[str, Any]]:
    trial_id = str(trial.get("trial_id", ""))
    tier = str(trial.get("tier", "weak"))
    gaps: list[dict[str, Any]] = []
    for item in trial.get("critical_missing_info", []) or []:
        raw_item = str(item).strip()
        if not raw_item or _is_not_applicable_gap(
            {"description": raw_item, "field": raw_item, "category": ""}
        ):
            continue
        normalized = _normalize_gap_item(raw_item)
        gaps.append(_gap_payload(normalized, raw_item, tier, trial_id))
    return gaps


def _collect_information_gaps(scored_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trial in scored_trials:
        for gap in _gaps_from_scored_trial(trial):
            _merge_gap(grouped, gap)
    return sorted(grouped.values(), key=lambda x: _priority_rank(str(x["priority"])), reverse=True)


def _is_not_applicable_gap(gap: dict[str, Any]) -> bool:
    text = (
        " ".join(
            [
                str(gap.get("field_id", "")),
                str(gap.get("field", "")),
                str(gap.get("display_name", "")),
                str(gap.get("description", "")),
                str(gap.get("why_needed", "")),
                str(gap.get("category", "")),
            ]
        )
        .strip()
        .lower()
    )
    return bool(text) and any(marker in text for marker in _NOT_APPLICABLE_MARKERS)


def _fallback_gap_field_id(gap: dict[str, Any]) -> str:
    text = str(gap.get("field") or gap.get("display_name") or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or "additional_clinical_detail"


def _gap_rationale(gap: dict[str, Any]) -> str:
    return str(gap.get("why_needed") or gap.get("evidence_text") or gap.get("description") or "")


def _normalized_gap(gap: dict[str, Any]) -> dict[str, Any] | None:
    field_id = str(gap.get("field_id", "")).strip() or _fallback_gap_field_id(gap)
    if not field_id:
        return None
    normalized = dict(gap)
    display_name = str(gap.get("display_name") or gap.get("field") or "").strip()
    normalized["field_id"] = field_id
    normalized["display_name"] = display_name or field_id.replace("_", " ").title()
    normalized["field"] = normalized["display_name"]
    rationale = _gap_rationale(normalized)
    normalized["why_needed"] = rationale
    normalized["evidence_text"] = rationale
    normalized["description"] = rationale
    normalized["affected_trial_ids"] = sorted(set(normalized.get("affected_trial_ids") or []))
    return normalized


def _contains_internal_failure_text(*parts: str) -> bool:
    text = " ".join(part for part in parts if part).strip().lower()
    return bool(text) and any(marker in text for marker in _INTERNAL_FAILURE_MARKERS)


def _is_low_value_gap(gap: dict[str, Any]) -> bool:
    field = str(gap.get("field") or gap.get("display_name") or "").strip().lower()
    desc = str(gap.get("description") or gap.get("why_needed") or "").strip().lower()
    if not field and not desc:
        return True
    if field in _LOW_VALUE_GAP_MARKERS or desc in _LOW_VALUE_GAP_MARKERS:
        return True
    return bool(field and field == desc and field in _LOW_VALUE_GAP_MARKERS)


def _merge_normalized_gap(current: dict[str, Any], normalized: dict[str, Any]) -> None:
    if len(str(normalized.get("description", ""))) > len(str(current.get("description", ""))):
        rationale = str(normalized.get("description", ""))
        current["why_needed"] = rationale
        current["evidence_text"] = rationale
        current["description"] = rationale
    if _priority_rank(str(normalized.get("priority", "low"))) > _priority_rank(
        str(current.get("priority", "low"))
    ):
        current["priority"] = str(normalized.get("priority", "low")).lower()
    current["affected_trial_ids"] = sorted(
        set(current.get("affected_trial_ids", [])) | set(normalized.get("affected_trial_ids", []))
    )


def _deduplicate_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        if _is_not_applicable_gap(gap):
            continue
        normalized = _normalized_gap(gap)
        if not normalized:
            continue
        field_id = str(normalized["field_id"])
        if field_id not in seen:
            seen[field_id] = normalized
            continue
        _merge_normalized_gap(seen[field_id], normalized)
    return list(seen.values())


def _limit_public_low_priority_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    low_seen = 0
    for gap in gaps:
        if str(gap.get("priority", "medium")).lower() == "low":
            if low_seen >= _PUBLIC_LOW_PRIORITY_GAP_LIMIT:
                continue
            low_seen += 1
        output.append(gap)
    return output


def sanitize_information_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for gap in _deduplicate_gaps(gaps):
        if _is_not_applicable_gap(gap):
            continue
        normalized = _normalized_gap(gap)
        if not normalized:
            continue
        if _contains_internal_failure_text(
            str(normalized.get("field", "")),
            str(normalized.get("display_name", "")),
            str(normalized.get("description", "")),
            str(normalized.get("why_needed", "")),
        ):
            continue
        if _is_low_value_gap(normalized):
            continue
        sanitized.append(normalized)
    ordered = sorted(
        sanitized,
        key=lambda x: (
            _priority_rank(str(x.get("priority", "low"))),
            len(str(x.get("description", ""))),
        ),
        reverse=True,
    )
    return _limit_public_low_priority_gaps(ordered)


def merge_information_gaps(
    scored_trials: list[dict[str, Any]], missing_info: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = _collect_information_gaps(scored_trials)
    if missing_info:
        merged.extend(missing_info)
    return sanitize_information_gaps(merged)
