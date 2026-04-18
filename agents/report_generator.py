from datetime import UTC, datetime
from typing import Any

from clinical_trial_agent.config import TIER_ORDER

from .report_formatter import build_text_report
from .report_synthesizer import generate_executive_summary

_METHODOLOGY_NOTE = (
    "This report uses an LLM-as-judge eligibility assessment with conservative tiering. "
    "Strong matches require confidence on major criteria and no disqualifying exclusion triggers. "
    "Weak/disqualified trials are excluded from the main body."
)


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


def _collect_information_gaps(scored_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trial in scored_trials:
        trial_id = str(trial.get("trial_id", ""))
        tier = str(trial.get("tier", "weak"))
        for item in trial.get("critical_missing_info", []) or []:
            key = str(item).strip()
            if not key:
                continue
            severity = _severity_for_missing_item(key, tier)
            if key not in grouped:
                grouped[key] = {
                    "field": key,
                    "description": key,
                    "priority": severity,
                    "affected_trial_ids": [trial_id] if trial_id else [],
                }
            else:
                if trial_id and trial_id not in grouped[key]["affected_trial_ids"]:
                    grouped[key]["affected_trial_ids"].append(trial_id)
                current = grouped[key]["priority"]
                if TIER_ORDER.get(severity, 0) > TIER_ORDER.get(current, 0):
                    grouped[key]["priority"] = severity

    priority_order = {"high": 3, "medium": 2, "low": 1}
    return sorted(
        grouped.values(), key=lambda x: priority_order.get(str(x["priority"]), 0), reverse=True
    )


def _deduplicate_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    for gap in gaps:
        field = str(gap.get("field", "")).strip()
        if not field:
            continue
        if field not in seen:
            seen[field] = dict(gap)
            continue
        current = seen[field]
        if len(str(gap.get("description", ""))) > len(str(current.get("description", ""))):
            current["description"] = gap.get("description", current.get("description", ""))
        current_priority = str(current.get("priority", "low")).lower()
        new_priority = str(gap.get("priority", "low")).lower()
        if priority_rank.get(new_priority, 0) > priority_rank.get(current_priority, 0):
            current["priority"] = new_priority
        current_ids = set(current.get("affected_trial_ids", []))
        new_ids = set(gap.get("affected_trial_ids", []))
        current["affected_trial_ids"] = sorted(current_ids | new_ids)
    return list(seen.values())


def _merge_information_gaps(
    scored_trials: list[dict[str, Any]], missing_info: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = _collect_information_gaps(scored_trials)
    if missing_info:
        merged.extend(missing_info)
    return _deduplicate_gaps(merged)


def _partition_trials(
    enriched_trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strong = [t for t in enriched_trials if str(t.get("tier", "weak")) == "strong"]
    moderate = [t for t in enriched_trials if str(t.get("tier", "weak")) == "moderate"]
    weak = [t for t in enriched_trials if str(t.get("tier", "weak")) == "weak"]
    disqualified = [t for t in enriched_trials if str(t.get("tier", "weak")) == "disqualified"]
    return strong, moderate, weak, disqualified


def _build_per_criterion_verdicts(verdict_details: dict[str, Any]) -> list[dict[str, str]]:
    mapping = {"MEETS": "eligible", "FAILS": "ineligible", "UNCERTAIN": "uncertain"}
    rows: list[dict[str, str]] = []
    for item in list(verdict_details.get("verdicts", [])):
        text = str(item.get("criterion_text", "")).strip()
        if not text:
            continue
        verdict = mapping.get(str(item.get("verdict", "UNCERTAIN")), "uncertain")
        rows.append(
            {
                "criterion_text": text,
                "verdict": verdict,
                "reasoning": str(item.get("reasoning", "")),
            }
        )
    return rows


def _build_trial_report_entry(
    trial: dict[str, Any], eligibility_verdicts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tid = str(trial.get("trial_id", ""))
    enriched = dict(trial)
    details = eligibility_verdicts.get(tid, {})
    enriched["verdict_details"] = details
    enriched["per_criterion_verdicts"] = _build_per_criterion_verdicts(details)
    return enriched


def _sort_by_tier_then_score(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda t: (TIER_ORDER.get(str(t.get("tier", "weak")), 0), float(t.get("score", 0.0))),
        reverse=True,
    )


def _build_report_payload(
    *,
    summary_data: dict[str, str],
    enriched_trials: list[dict[str, Any]],
    information_gaps: list[dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _partition_trials(enriched_trials)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "patient_summary": summary_data.get("patient_summary", ""),
        "executive_summary": summary_data.get("executive_summary", ""),
        "total_trials_searched": len(trials_raw),
        "total_trials_evaluated": len(enriched_trials),
        "strong_matches": strong,
        "moderate_matches": moderate,
        "excluded_trial_count": len(weak) + len(disqualified),
        "excluded_trials": weak + disqualified,
        "information_gaps": information_gaps,
        "qa_issues": qa_issues,
        "methodology_note": _METHODOLOGY_NOTE,
        "search_queries_used": list(dict.fromkeys(search_queries)),
        "decision_history": decision_history,
        "ranked_trials": enriched_trials,
    }


async def build_report(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
    missing_info: list[dict[str, Any]],
    eligibility_verdicts: dict[str, dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = _sort_by_tier_then_score(scored_trials)
    summary_data = await generate_executive_summary(patient_profile, ranked)
    enriched_trials = [_build_trial_report_entry(t, eligibility_verdicts) for t in ranked]

    merged_gaps = _merge_information_gaps(ranked, missing_info)
    return _build_report_payload(
        summary_data=summary_data,
        enriched_trials=enriched_trials,
        information_gaps=merged_gaps,
        trials_raw=trials_raw,
        search_queries=search_queries,
        decision_history=decision_history,
        qa_issues=qa_issues,
    )


__all__ = ["build_report", "build_text_report"]
