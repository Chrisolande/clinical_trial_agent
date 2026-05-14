import re
from collections import defaultdict
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
    if field_norm in _LOW_VALUE_PLACEHOLDERS:
        return True
    if desc_norm in _LOW_VALUE_PLACEHOLDERS:
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
        if not isinstance(item, dict):
            continue
        if _is_not_applicable_item(item):
            continue
        field_id = str(item.get("field_id", "")).strip().lower()
        field = _sanitize_public_text(
            str(item.get("item") or item.get("field") or item.get("display_name") or item.get("field_id") or "").strip()
        )
        desc = _sanitize_public_text(
            str(item.get("reason") or item.get("description") or item.get("why_needed") or "").strip()
        )
        if not field and not desc:
            continue
        if _is_low_value_placeholder(field, desc):
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


def _render_ranking_row(trial: dict[str, Any]) -> str:
    nct = str(trial.get("nct_id") or trial.get("trial_id") or "N/A")
    title = str(trial.get("title") or trial.get("brief_title") or "Untitled")
    reason = str(trial.get("evidence_summary") or trial.get("rationale") or trial.get("key_concern") or "").strip()
    action = str(trial.get("next_action") or "").strip()
    reason_text = _sanitize_public_text(reason) or "No concise reason supplied"
    action_text = _sanitize_public_text(action) or "No concrete action supplied"
    return (
        f"- {nct}: {title} | "
        f"tier={trial.get('tier', 'weak')} | score={float(trial.get('score', 0.0)):.2f} | "
        f"reason={reason_text} | next={action_text}"
    )


def _render_trial_group(cards: list[dict[str, Any]], tier: str, title: str) -> list[str]:
    rows = [card for card in cards if str(card.get("tier", "weak")) == tier]
    lines = [title, "-" * 40]
    if not rows:
        lines.extend(["None.", ""])
        return lines
    for row in rows:
        nct = str(row.get("nct_id") or row.get("trial_id") or "N/A")
        trial_title = str(row.get("title") or row.get("brief_title") or "Untitled")
        phase = str(row.get("phase", "")).strip() or "None"
        status = str(row.get("status", "")).strip() or "Unknown"
        recommendation = _sanitize_public_text(str(row.get("recommendation") or row.get("rationale") or ""))
        why_it_matches = [
            _sanitize_public_text(str(item))
            for item in list(row.get("why_it_matches") or [])
            if _sanitize_public_text(str(item))
        ]
        main_blockers = [
            _sanitize_public_text(str(item))
            for item in list(row.get("main_blockers") or [])
            if _sanitize_public_text(str(item))
        ]
        key_uncertainties = [
            _sanitize_public_text(str(item))
            for item in list(row.get("key_uncertainties") or [])
            if _sanitize_public_text(str(item))
        ]
        next_action = _sanitize_public_text(str(row.get("next_action", "")))

        lines.append(f"[{nct}] {trial_title}")
        lines.append(
            f"Tier: {row.get('tier', tier)} | Score: {float(row.get('score', 0.0)):.2f} | "
            f"Phase: {phase} | Status: {status}"
        )
        if recommendation:
            lines.append(f"Recommendation: {recommendation}")
        if why_it_matches:
            lines.append("Why it fits:")
            for item in why_it_matches[:5]:
                lines.append(f"- {item}")
        if main_blockers:
            lines.append("Main blockers:")
            for item in main_blockers[:5]:
                lines.append(f"- {item}")
        if key_uncertainties:
            lines.append("Key uncertainties:")
            for item in key_uncertainties[:5]:
                lines.append(f"- {item}")
        if next_action:
            lines.append(f"Next action: {next_action}")
        evidence_summary = _sanitize_public_text(str(row.get("evidence_summary", "")))
        if evidence_summary:
            lines.append(f"Evidence summary: {evidence_summary}")
        lines.append("")
    return lines


def _group_gaps_by_priority(gaps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gaps:
        grouped[str(item.get("priority", "medium")).upper()].append(item)
    return grouped


def _gap_field_and_description(item: dict[str, Any]) -> tuple[str, str]:
    field = _sanitize_public_text(
        str(item.get("item") or item.get("field") or item.get("display_name") or item.get("field_id") or "").strip()
    )
    desc = _sanitize_public_text(str(item.get("reason") or item.get("description") or "").strip())
    return field, desc


def _render_gap_priority_group(severity: str, items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return []
    lines = [f"{severity}:"]
    for item in items:
        field, desc = _gap_field_and_description(item)
        if not field and not desc:
            continue
        lines.append(f"- {field or desc}")
        if desc and desc != field:
            lines.append(f"  Why needed: {desc}")
        action = _sanitize_public_text(str(item.get("action", "")).strip())
        if action:
            lines.append(f"  Action: {action}")
        trials = list(item.get("affects_trials") or item.get("affected_trial_ids") or [])
        if trials:
            lines.append(f"  Affects trials: {', '.join(str(t) for t in trials)}")
    lines.append("")
    return lines


def _render_information_gaps(info_gaps: list[dict[str, Any]]) -> list[str]:
    cleaned = [
        gap
        for gap in _dedupe_info_gaps(info_gaps)
        if bool(gap.get("applicable_to_patient", True))
    ]
    lines = ["CRITICAL INFORMATION GAPS", "-" * 40]
    if not cleaned:
        lines.extend(["None identified after sanitization.", ""])
        return lines
    grouped = _group_gaps_by_priority(cleaned)
    for severity in ("HIGH", "MEDIUM", "LOW"):
        lines.extend(_render_gap_priority_group(severity, grouped.get(severity, [])))
    return lines


def _render_next_actions(info_gaps: list[dict[str, Any]], recommended_actions: list[dict[str, Any]]) -> list[str]:
    cleaned = [
        gap
        for gap in _dedupe_info_gaps(recommended_actions or info_gaps)
        if bool(gap.get("applicable_to_patient", True))
    ]
    lines = ["RECOMMENDED CLINICAL NEXT ACTIONS", "-" * 40]
    actionable = [gap for gap in cleaned if str(gap.get("priority", "medium")).lower() in {"high", "medium"}]
    if not actionable:
        lines.extend(["No immediate actions.", ""])
        return lines
    count = 0
    for item in actionable[:5]:
        field, desc = _gap_field_and_description(item)
        if _is_low_value_placeholder(field, desc):
            continue
        trials = ", ".join(list(item.get("affects_trials") or item.get("affected_trial_ids") or [])[:3])
        action = _sanitize_public_text(str(item.get("action", "")).strip())
        if not action:
            action = f"Obtain {field or 'missing clinical detail'}."
        if desc and desc not in action:
            action += f" {desc}"
        if trials:
            action += f" Prioritize for trials: {trials}."
        count += 1
        lines.append(f"{count}. {action}")
    lines.append("")
    return lines


def _render_excluded_trials(report_json: dict[str, Any]) -> list[str]:
    lines = ["TRIALS NOT PRIORITIZED", "-" * 40]
    excluded = list(report_json.get("excluded_trials", []))
    if not excluded:
        lines.extend([f"Count: {int(report_json.get('excluded_trial_count', 0))}", ""])
        return lines
    for trial in excluded[:10]:
        lines.append(
            f"- {trial.get('trial_id', 'N/A')}: {trial.get('brief_title', 'Untitled')} "
            f"(tier={trial.get('tier', 'weak')}, score={float(trial.get('score', 0.0)):.2f})"
        )
    if len(excluded) > 10:
        lines.append(f"- ... and {len(excluded) - 10} additional excluded trials")
    lines.append("")
    return lines


def _render_debug_sections(report_json: dict[str, Any]) -> list[str]:
    if not bool(report_json.get("debug", False)):
        return []
    qa_issues = report_json.get("qa_issues_internal", report_json.get("qa_issues", []))
    qa_remediation = report_json.get("qa_remediation_internal", report_json.get("qa_remediation", {}))
    lines: list[str] = []
    if qa_issues:
        lines.extend(["QA ISSUES", "-" * 40])
        for issue in qa_issues:
            if isinstance(issue, dict):
                sev = str(issue.get("severity", "unknown")).upper()
                code = str(issue.get("code", "UNSPECIFIED"))
                msg = str(issue.get("message", ""))
                lines.append(f"  - [{sev}] {code}: {msg}")
            else:
                lines.append(f"  - {issue}")
        lines.append("")
    if isinstance(qa_remediation, dict):
        attempts = int(qa_remediation.get("attempts", 0) or 0)
        unresolved = list(qa_remediation.get("unresolved_issues") or [])
        if attempts > 0 or unresolved:
            lines.extend(["QA REMEDIATION", "-" * 40, f"Attempts: {attempts}"])
            actions = [
                str(action.get("action", ""))
                for action in list(qa_remediation.get("actions") or [])
                if isinstance(action, dict)
            ]
            if actions:
                lines.append(f"Actions: {', '.join(actions)}")
            for issue in unresolved:
                if isinstance(issue, dict):
                    sev = str(issue.get("severity", "unknown")).upper()
                    code = str(issue.get("code", "UNSPECIFIED"))
                    msg = str(issue.get("message", ""))
                    lines.append(f"  - [{sev}] {code}: {msg}")
            lines.append("")
    return lines


def _render_methodology_and_limitations(report_json: dict[str, Any]) -> list[str]:
    methodology = _sanitize_public_text(str(report_json.get("methodology_note", "")).strip())
    plan = _extract_report_plan(report_json)
    limitations = [
        _sanitize_public_text(str(item))
        for item in list(plan.get("limitations") or [])
        if _sanitize_public_text(str(item))
    ]
    if bool(report_json.get("retrieval_failed", False)):
        limitations.append("Trial retrieval failed before fan-out; findings may be incomplete.")
    if report_json.get("retrieval_errors"):
        limitations.append("Some retrieval errors occurred.")
    limitation_line = _sanitize_public_text(" ".join(limitations)) or "None explicitly identified."
    return [
        "METHODOLOGY AND LIMITATIONS",
        "-" * 40,
        f"Methodology: {methodology or 'Not provided.'}",
        f"Limitations: {limitation_line}",
        "",
    ]


def build_text_report(report_json: dict[str, Any]) -> str:
    plan = _extract_report_plan(report_json)
    cards = _cards_from_plan_or_matches(report_json, plan)
    plan_gaps = plan.get("information_gaps", []) if isinstance(plan.get("information_gaps"), list) else []
    fallback_gaps = report_json.get("information_gaps", []) if isinstance(report_json.get("information_gaps"), list) else []
    info_gaps = plan_gaps or fallback_gaps

    patient_summary = _sanitize_public_text(
        str(plan.get("patient_summary") or report_json.get("patient_summary", "")).strip()
    )
    bottom_line = _compact_executive_summary(
        str(plan.get("bottom_line") or plan.get("executive_summary") or report_json.get("executive_summary", "")).strip()
    )
    executive_summary = _compact_executive_summary(
        str(plan.get("executive_summary") or report_json.get("executive_summary", "")).strip()
    )
    recommended_actions = (
        plan.get("recommended_actions", []) if isinstance(plan.get("recommended_actions"), list) else []
    )

    lines = [
        "=" * 70,
        "CLINICAL TRIAL MATCHING REPORT",
        f"Generated: {report_json.get('generated_at', 'N/A')}",
        "=" * 70,
        "",
        "PATIENT SNAPSHOT",
        "-" * 40,
        patient_summary or "Not provided.",
        "",
        "BOTTOM LINE",
        "-" * 40,
        bottom_line or "Insufficient information for a concise recommendation.",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        executive_summary or "Insufficient information for a concise summary.",
        "",
        "TRIAL RANKING TABLE",
        "-" * 40,
    ]

    if cards:
        for trial in cards:
            lines.append(_render_ranking_row(trial))
    else:
        lines.append("No trial cards available.")
    lines.append("")

    lines.extend(_render_trial_group(cards, "strong", "STRONG MATCHES"))
    lines.extend(_render_trial_group(cards, "moderate", "MODERATE MATCHES"))
    lines.extend(_render_information_gaps(info_gaps))
    lines.extend(_render_next_actions(info_gaps, recommended_actions))
    lines.extend(_render_excluded_trials(report_json))
    lines.extend(_render_methodology_and_limitations(report_json))
    lines.extend(_render_debug_sections(report_json))

    return "\n".join(lines)
