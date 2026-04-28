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
}
_NOT_APPLICABLE_MARKERS = {"not_applicable", "not applicable", "n/a", "non-applicable"}


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


def _dedupe_info_gaps(info_gaps: list[Any]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in info_gaps:
        if not isinstance(item, dict):
            continue
        if _is_not_applicable_item(item):
            continue
        field_id = str(item.get("field_id", "")).strip().lower()
        field = str(item.get("field", "")).strip()
        desc = str(item.get("description", "")).strip()
        if _is_low_value_placeholder(field, desc):
            continue
        key = field_id or f"{field.lower()}::{desc.lower()}"
        if key in deduped:
            existing = deduped[key]
            merged_ids = set(existing.get("affected_trial_ids", [])) | set(
                item.get("affected_trial_ids", [])
            )
            existing["affected_trial_ids"] = sorted(merged_ids)
            if len(desc) > len(str(existing.get("description", ""))):
                existing["description"] = desc
            continue
        deduped[key] = dict(item)
    return list(deduped.values())


def _compact_executive_summary(summary: str) -> str:
    text = str(summary or "").strip()
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


def render_trial_entry(t: dict[str, Any], *, include_key_concern: bool) -> list[str]:
    lines = [
        f"[{t['trial_id']}] {t['brief_title']}",
        f"Tier: {t.get('tier', 'weak')} | Score: {t['score']:.2f} | "
        f"Phase: {t.get('phase', 'N/A')} | Status: {t.get('overall_status', 'N/A')}",
        f"Criteria: {t['meets_count']} met / {t['fails_count']} failed / "
        f"{t['uncertain_count']} uncertain",
    ]
    if include_key_concern and t.get("key_concern"):
        lines.append(f"Key concern: {t['key_concern']}")
    lines.append("")
    return lines


def render_tier_section(
    report_json: dict[str, Any],
    key: str,
    label: str,
    *,
    include_key_concern: bool,
) -> list[str]:
    trials = report_json.get(key, [])
    if not trials:
        return []
    lines = [f"{label} ({len(trials)}):", "-" * 40]
    for t in trials:
        lines.extend(render_trial_entry(t, include_key_concern=include_key_concern))
    return lines


def render_information_gaps(info_gaps: list[dict[str, Any]]) -> list[str]:
    cleaned = _dedupe_info_gaps(info_gaps)
    if not cleaned:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cleaned:
        grouped[str(item.get("priority", "medium")).upper()].append(item)

    lines = ["INFORMATION GAPS", "-" * 40]
    for severity in ("HIGH", "MEDIUM", "LOW"):
        items = grouped.get(severity, [])
        if not items:
            continue
        lines.append(f"{severity}:")
        for item in items:
            field = str(
                item.get("field") or item.get("display_name") or item.get("field_id") or ""
            ).strip()
            desc = str(item.get("description", "")).strip()
            if not field and not desc:
                continue
            lines.append(f"- {field}")
            if desc and desc != field:
                lines.append(f"  {desc}")
        lines.append("")
    return lines


def render_next_actions(info_gaps: list[dict[str, Any]]) -> list[str]:
    cleaned = _dedupe_info_gaps(info_gaps)
    if not cleaned:
        return []
    action_lines: list[str] = []
    high_then_medium = [
        g for g in cleaned if str(g.get("priority", "")).lower() in {"high", "medium"}
    ]
    high_then_medium = [
        g
        for g in high_then_medium
        if str(g.get("field", "")).strip().lower()
        not in {
            "criterion requires details not present in profile",
            "missing trial-specific clinical detail",
            "missing exclusion-history detail",
        }
    ]
    for item in high_then_medium[:5]:
        field = str(
            item.get("field") or item.get("display_name") or item.get("field_id") or ""
        ).strip()
        desc = str(item.get("description", "")).strip()
        trials = ", ".join(list(item.get("affected_trial_ids", []))[:3])
        if _is_low_value_placeholder(field, desc):
            continue
        action_lines.append(f"- {field}")
        if desc:
            action_lines.append(f"  Action: {desc}")
        if trials:
            action_lines.append(f"  Affects trials: {trials}")
    if not action_lines:
        return []
    lines = ["RECOMMENDED CLINICAL NEXT ACTIONS", "-" * 40, *action_lines, ""]
    return lines


def render_potential_matches(report_json: dict[str, Any]) -> list[str]:
    strong = list(report_json.get("strong_matches", []))
    moderate = list(report_json.get("moderate_matches", []))
    if strong or moderate:
        return []

    excluded = list(report_json.get("excluded_trials", []))
    if not excluded:
        return []

    candidates = [
        t
        for t in excluded
        if int(t.get("hard_exclusion_failures", 0)) == 0 and int(t.get("meets_count", 0)) > 0
    ]
    candidates = sorted(
        candidates,
        key=lambda x: (
            float(x.get("score", 0.0)),
            int(x.get("meets_count", 0)),
            -int(x.get("uncertain_count", 0)),
        ),
        reverse=True,
    )[:5]
    if not candidates:
        return []

    lines = ["POTENTIAL MATCHES PENDING ADDITIONAL DATA", "-" * 40]
    for c in candidates:
        lines.append(
            f"- [{c.get('trial_id', '')}] {c.get('brief_title', '')} | "
            f"score={float(c.get('score', 0.0)):.2f} | "
            f"met={int(c.get('meets_count', 0))} uncertain={int(c.get('uncertain_count', 0))}"
        )
        concern = str(c.get("key_concern", "")).strip()
        if concern:
            lines.append(f"  Limitation: {concern}")
    lines.append("")
    return lines


def build_text_report(report_json: dict[str, Any]) -> str:
    methodology_note = report_json.get("methodology_note", "")
    lines: list[str] = [
        "=" * 70,
        "CLINICAL TRIAL MATCHING REPORT",
        f"Generated: {report_json.get('generated_at', 'N/A')}",
        "=" * 70,
        "",
        "PATIENT: " + report_json.get("patient_summary", ""),
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        _compact_executive_summary(str(report_json.get("executive_summary", ""))),
        "",
        f"TRIALS SEARCHED: {report_json.get('total_trials_searched', 0)}  "
        f"EVALUATED: {report_json.get('total_trials_evaluated', 0)}",
        "",
    ]

    lines.extend(
        render_tier_section(
            report_json,
            "strong_matches",
            "STRONG MATCHES",
            include_key_concern=False,
        )
    )
    lines.extend(
        render_tier_section(
            report_json,
            "moderate_matches",
            "MODERATE MATCHES",
            include_key_concern=True,
        )
    )

    excluded_count = int(report_json.get("excluded_trial_count", 0))
    lines.append(
        f"Appendix: {excluded_count} trials were assessed as weak or disqualified and excluded from this report."
    )
    lines.append("")

    lines.extend(render_potential_matches(report_json))

    raw_gaps = report_json.get("information_gaps", [])
    gaps = [g for g in raw_gaps if isinstance(g, dict)] if isinstance(raw_gaps, list) else []
    gaps = _dedupe_info_gaps(gaps)
    lines.extend(render_information_gaps(gaps))
    lines.extend(render_next_actions(gaps))

    qa_issues = report_json.get("qa_issues", [])
    if qa_issues:
        lines += ["QA ISSUES", "-" * 40]
        for issue in qa_issues:
            if isinstance(issue, dict):
                sev = str(issue.get("severity", "unknown")).upper()
                code = str(issue.get("code", "UNSPECIFIED"))
                msg = str(issue.get("message", ""))
                lines.append(f"  - [{sev}] {code}: {msg}")
            else:
                lines.append(f"  - {issue}")
        lines.append("")

    qa_remediation = report_json.get("qa_remediation", {})
    if isinstance(qa_remediation, dict):
        attempts = int(qa_remediation.get("attempts", 0))
        actions = qa_remediation.get("actions", [])
        unresolved = qa_remediation.get("unresolved_issues", [])
        if attempts > 0 or unresolved:
            lines += ["QA REMEDIATION", "-" * 40]
            lines.append(f"Attempts: {attempts}")
            if isinstance(actions, list) and actions:
                lines.append(
                    "Actions: "
                    + ", ".join(
                        str(action.get("action", ""))
                        for action in actions
                        if isinstance(action, dict)
                    )
                )
            if isinstance(unresolved, list) and unresolved:
                lines.append("Unresolved critical issues:")
                for issue in unresolved:
                    if isinstance(issue, dict):
                        lines.append(
                            f"  - [{str(issue.get('severity', 'unknown')).upper()}] "
                            f"{issue.get('code', 'UNSPECIFIED')}: {issue.get('message', '')}"
                        )
            lines.append("")

    lines += ["METHODOLOGY", "-" * 40, methodology_note, ""]

    return "\n".join(lines)
