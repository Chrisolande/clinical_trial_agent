from collections import defaultdict
from typing import Any


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
    if not info_gaps:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in info_gaps:
        grouped[str(item.get("priority", "medium")).upper()].append(item)

    lines = ["INFORMATION GAPS", "-" * 40]
    for severity in ("HIGH", "MEDIUM", "LOW"):
        items = grouped.get(severity, [])
        if not items:
            continue
        lines.append(f"{severity}:")
        for item in items:
            field = str(item.get("field", "")).strip()
            desc = str(item.get("description", "")).strip()
            lines.append(f"- {field}")
            if desc and desc != field:
                lines.append(f"  {desc}")
        lines.append("")
    return lines


def render_next_actions(info_gaps: list[dict[str, Any]]) -> list[str]:
    if not info_gaps:
        return []
    lines = ["RECOMMENDED CLINICAL NEXT ACTIONS", "-" * 40]
    high_then_medium = [
        g for g in info_gaps if str(g.get("priority", "")).lower() in {"high", "medium"}
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
    for item in high_then_medium[:8]:
        field = str(item.get("field", "")).strip()
        desc = str(item.get("description", "")).strip()
        trials = ", ".join(list(item.get("affected_trial_ids", []))[:3])
        lines.append(f"- {field}")
        if desc:
            lines.append(f"  Action: {desc}")
        if trials:
            lines.append(f"  Affects trials: {trials}")
    lines.append("")
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
        report_json.get("executive_summary", ""),
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
    lines.extend(render_information_gaps(gaps))
    lines.extend(render_next_actions(gaps))

    qa_issues = report_json.get("qa_issues", [])
    if qa_issues:
        lines += ["QA ISSUES", "-" * 40] + [f"  - {issue}" for issue in qa_issues] + [""]

    lines += ["METHODOLOGY", "-" * 40, methodology_note, ""]

    return "\n".join(lines)
