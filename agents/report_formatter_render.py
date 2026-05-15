from collections import defaultdict
from typing import Any

from .report_formatter_sanitize import (
    _dedupe_info_gaps,
    _is_low_value_placeholder,
    _sanitize_public_text,
)


def _render_ranking_row(trial: dict[str, Any]) -> str:
    nct = str(trial.get("nct_id") or trial.get("trial_id") or "N/A")
    title = str(trial.get("title") or trial.get("brief_title") or "Untitled")
    reason = str(
        trial.get("evidence_summary") or trial.get("rationale") or trial.get("key_concern") or ""
    ).strip()
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
        recommendation = _sanitize_public_text(
            str(row.get("recommendation") or row.get("rationale") or "")
        )
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
        str(
            item.get("item")
            or item.get("field")
            or item.get("display_name")
            or item.get("field_id")
            or ""
        ).strip()
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
        gap for gap in _dedupe_info_gaps(info_gaps) if bool(gap.get("applicable_to_patient", True))
    ]
    lines = ["CRITICAL INFORMATION GAPS", "-" * 40]
    if not cleaned:
        lines.extend(["None identified after sanitization.", ""])
        return lines
    grouped = _group_gaps_by_priority(cleaned)
    for severity in ("HIGH", "MEDIUM", "LOW"):
        lines.extend(_render_gap_priority_group(severity, grouped.get(severity, [])))
    return lines


def _render_next_actions(
    info_gaps: list[dict[str, Any]], recommended_actions: list[dict[str, Any]]
) -> list[str]:
    cleaned = [
        gap
        for gap in _dedupe_info_gaps(recommended_actions or info_gaps)
        if bool(gap.get("applicable_to_patient", True))
    ]
    lines = ["RECOMMENDED CLINICAL NEXT ACTIONS", "-" * 40]
    actionable = [
        gap for gap in cleaned if str(gap.get("priority", "medium")).lower() in {"high", "medium"}
    ]
    if not actionable:
        lines.extend(["No immediate actions.", ""])
        return lines
    count = 0
    for item in actionable[:5]:
        field, desc = _gap_field_and_description(item)
        if _is_low_value_placeholder(field, desc):
            continue
        trials = ", ".join(
            list(item.get("affects_trials") or item.get("affected_trial_ids") or [])[:3]
        )
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
    qa_remediation = report_json.get(
        "qa_remediation_internal", report_json.get("qa_remediation", {})
    )
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


def _render_methodology_and_limitations(
    report_json: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    methodology = _sanitize_public_text(str(report_json.get("methodology_note", "")).strip())
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
