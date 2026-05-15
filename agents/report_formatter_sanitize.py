import re
from typing import Any

_LOW_VALUE_PLACEHOLDERS = {
    "criterion requires details not present in profile",
    "missing trial-specific clinical detail",
    "missing exclusion-history detail",
    "additional clinical detail",
    "n/a",
    "unknown",
    "none",
}
_NOT_APPLICABLE_MARKERS = {"not_applicable", "not applicable", "n/a", "non-applicable"}
_FORBIDDEN_PHRASES = (
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
_PUBLIC_LOW_PRIORITY_GAP_LIMIT = 2


def _forbidden_phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in phrase.split()]
    joined = r"\s+".join(parts)
    return re.compile(rf"\b{joined}\b", flags=re.IGNORECASE)


_FORBIDDEN_PATTERNS = tuple(_forbidden_phrase_pattern(phrase) for phrase in _FORBIDDEN_PHRASES)


def _contains_forbidden_phrase(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(pattern.search(lowered) for pattern in _FORBIDDEN_PATTERNS)


def _sanitize_forbidden_phrases(text: str) -> str:
    sanitized = str(text or "")
    for pattern in _FORBIDDEN_PATTERNS:
        sanitized = pattern.sub("", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" -:;,.")
    return sanitized


def _sanitize_public_text(text: str) -> str:
    cleaned = _sanitize_forbidden_phrases(text)
    if _contains_forbidden_phrase(cleaned):
        return ""
    return cleaned


def _is_low_value_placeholder(field: str, desc: str) -> bool:
    field_norm = field.strip().lower()
    desc_norm = desc.strip().lower()
    if not field_norm and not desc_norm:
        return True
    if field_norm in _LOW_VALUE_PLACEHOLDERS or desc_norm in _LOW_VALUE_PLACEHOLDERS:
        return True
    return bool(field_norm and field_norm == desc_norm and field_norm in _LOW_VALUE_PLACEHOLDERS)


def _is_not_applicable_item(item: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(item.get("field_id", "")),
            str(item.get("field", "")),
            str(item.get("display_name", "")),
            str(item.get("description", "")),
            str(item.get("why_needed", "")),
            str(item.get("category", "")),
        ]
    ).lower()
    return any(marker in blob for marker in _NOT_APPLICABLE_MARKERS)


def _priority_rank(priority: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(priority.lower(), 0)


def _dedupe_info_gaps(info_gaps: list[Any]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in info_gaps:
        if not isinstance(item, dict) or _is_not_applicable_item(item):
            continue
        field_id = str(item.get("field_id", "")).strip().lower()
        field = _sanitize_public_text(
            str(item.get("item") or item.get("field") or item.get("display_name") or item.get("field_id") or "").strip()
        )
        desc = _sanitize_public_text(
            str(item.get("reason") or item.get("description") or item.get("why_needed") or "").strip()
        )
        if (not field and not desc) or _is_low_value_placeholder(field, desc):
            continue
        key = field_id or field.lower()
        if not key:
            continue
        if key in deduped:
            existing = deduped[key]
            merged_ids = set(existing.get("affected_trial_ids", [])) | set(
                item.get("affected_trial_ids") or item.get("affects_trials", [])
            )
            existing["affected_trial_ids"] = sorted(merged_ids)
            existing["affects_trials"] = sorted(merged_ids)
            if len(desc) > len(str(existing.get("description", ""))):
                existing["description"] = desc
                existing["reason"] = desc
            action = _sanitize_public_text(str(item.get("action", "")).strip())
            if action and len(action) > len(str(existing.get("action", ""))):
                existing["action"] = action
            if _priority_rank(str(item.get("priority", "low"))) > _priority_rank(
                str(existing.get("priority", "low"))
            ):
                existing["priority"] = str(item.get("priority", "low")).lower()
            continue
        deduped[key] = {
            "field_id": field_id or key,
            "item": field or str(item.get("item") or item.get("field") or item.get("display_name") or "").strip(),
            "field": field or str(item.get("field") or item.get("display_name") or "").strip(),
            "description": desc,
            "reason": desc,
            "priority": str(item.get("priority", "medium")).lower(),
            "affected_trial_ids": sorted(set(item.get("affected_trial_ids") or item.get("affects_trials") or [])),
            "affects_trials": sorted(set(item.get("affected_trial_ids") or item.get("affects_trials") or [])),
            "action": _sanitize_public_text(str(item.get("action", "")).strip()),
            "applicable_to_patient": bool(item.get("applicable_to_patient", True)),
        }
    ordered = sorted(
        deduped.values(),
        key=lambda x: (_priority_rank(str(x.get("priority", "low"))), len(str(x.get("description", "")))),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    low_count = 0
    for item in ordered:
        if str(item.get("priority", "medium")).lower() == "low":
            if low_count >= _PUBLIC_LOW_PRIORITY_GAP_LIMIT:
                continue
            low_count += 1
        output.append(item)
    return output


def _compact_executive_summary(summary: str) -> str:
    text = _sanitize_public_text(summary)
    if not text:
        return ""
    bullet_lines = [
        line.strip("-• ").strip()
        for line in text.splitlines()
        if line.strip().startswith(("-", "•"))
    ]
    if bullet_lines:
        return " ".join(f"- {line}" for line in bullet_lines[:3])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(sentences[:3]) if sentences else text


def _extract_report_plan(report_json: dict[str, Any]) -> dict[str, Any]:
    plan = report_json.get("report_plan")
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return {}


def _cards_from_plan_or_matches(report_json: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    strong = plan.get("strong_matches", [])
    moderate = plan.get("moderate_matches", [])
    if isinstance(strong, list) and isinstance(moderate, list) and (strong or moderate):
        cards = [*strong, *moderate]
        return [card for card in cards if isinstance(card, dict)]
    cards = plan.get("trial_cards", [])
    if isinstance(cards, list) and cards:
        return [card for card in cards if isinstance(card, dict)]
    ranked = report_json.get("ranked_trials", [])
    if isinstance(ranked, list) and ranked:
        return [trial for trial in ranked if isinstance(trial, dict)]
    return list(report_json.get("strong_matches", [])) + list(report_json.get("moderate_matches", []))
