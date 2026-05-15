from typing import Any

from .report_formatter_render import (
    _render_debug_sections,
    _render_excluded_trials,
    _render_information_gaps,
    _render_methodology_and_limitations,
    _render_next_actions,
    _render_ranking_row,
    _render_trial_group,
)
from .report_formatter_sanitize import (
    _cards_from_plan_or_matches,
    _compact_executive_summary,
    _extract_report_plan,
    _sanitize_forbidden_phrases,
    _sanitize_public_text,
)


def build_text_report(report_json: dict[str, Any]) -> str:
    plan = _extract_report_plan(report_json)
    cards = _cards_from_plan_or_matches(report_json, plan)
    plan_gaps = (
        plan.get("information_gaps", []) if isinstance(plan.get("information_gaps"), list) else []
    )
    fallback_gaps = (
        report_json.get("information_gaps", [])
        if isinstance(report_json.get("information_gaps"), list)
        else []
    )
    info_gaps = plan_gaps or fallback_gaps

    patient_summary = _sanitize_public_text(
        str(plan.get("patient_summary") or report_json.get("patient_summary", "")).strip()
    )
    bottom_line = _compact_executive_summary(
        str(
            plan.get("bottom_line")
            or plan.get("executive_summary")
            or report_json.get("executive_summary", "")
        ).strip()
    )
    executive_summary = _compact_executive_summary(
        str(plan.get("executive_summary") or report_json.get("executive_summary", "")).strip()
    )
    recommended_actions = (
        plan.get("recommended_actions", [])
        if isinstance(plan.get("recommended_actions"), list)
        else []
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
    lines.extend(_render_methodology_and_limitations(report_json, plan))
    lines.extend(_render_debug_sections(report_json))

    report_text = "\n".join(lines)
    if bool(report_json.get("debug", False)):
        return report_text
    return _sanitize_forbidden_phrases(report_text)
