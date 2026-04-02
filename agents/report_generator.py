from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any, cast

from config import TIER_ORDER, get_llm, settings
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from prompts.synthesis import build_synthesis_prompt
from pydantic import BaseModel, Field
from tools.retry import llm_retry


class ExecutiveSummaryModel(BaseModel):
    executive_summary: str = Field(
        description="A 150-250 word physician-level executive summary of the trial matches."
    )
    patient_summary: str = Field(
        description="A concise 1-sentence demographic and clinical description of the patient"
    )


def _count_tiers(scored_trials: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tiers = Counter(str(t.get("tier", "weak")) for t in scored_trials)
    return (
        tiers["strong"],
        tiers["moderate"],
        tiers["weak"],
        tiers["disqualified"],
    )


def _describe_patient(patient_profile: dict[str, Any]) -> str:
    conds = patient_profile.get("conditions", [])
    primary = patient_profile.get("primary_condition") or (
        conds[0] if conds else "unknown condition"
    )
    age = patient_profile.get("age", "unknown age")
    sex = patient_profile.get("sex", "")
    return f"{age} year old {sex} patient with {primary}".strip()


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


def _merge_information_gaps(
    scored_trials: list[dict[str, Any]],
    missing_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = _collect_information_gaps(scored_trials)
    if missing_info:
        merged.extend(missing_info)
    return merged


def _partition_trials(
    enriched_trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strong = [t for t in enriched_trials if str(t.get("tier", "weak")) == "strong"]
    moderate = [t for t in enriched_trials if str(t.get("tier", "weak")) == "moderate"]
    weak = [t for t in enriched_trials if str(t.get("tier", "weak")) == "weak"]
    disqualified = [t for t in enriched_trials if str(t.get("tier", "weak")) == "disqualified"]
    return strong, moderate, weak, disqualified


def _build_report_payload(
    *,
    summary_data: dict[str, str],
    enriched_trials: list[dict[str, Any]],
    information_gaps: list[dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues: list[str],
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
    }


def _build_exec_summary_context(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _count_tiers(scored_trials)
    top_trials_summary = "\n".join(
        f"- {t['brief_title']} ({t['trial_id']}): tier={t['tier']}, score={t['score']:.2f}, concern={t.get('key_concern', '')}"
        for t in scored_trials[:5]
    )
    patient_sum = _describe_patient(patient_profile)
    return {
        "patient_summary": patient_sum,
        "strong_count": strong,
        "moderate_count": moderate,
        "excluded_count": weak + disqualified,
        "total": len(scored_trials),
        "top_trials": top_trials_summary or "No trials evaluated",
        "missing_info": "See Information Gaps section.",
    }


@llm_retry
async def _invoke_exec_summary_llm(chain: Any, context: dict[str, Any]) -> dict[str, Any]:
    result = await asyncio.wait_for(
        chain.ainvoke(
            context,
            config={"run_name": "executive_summary", "tags": ["synthesis", "report"]},
        ),
        timeout=settings.llm_call_timeout_seconds,
    )
    if not isinstance(result, ExecutiveSummaryModel):
        raise ValueError(f"Unexpected report summary result type: {type(result)}")
    return cast("dict[str, Any]", result.model_dump())


async def generate_executive_summary(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
) -> dict[str, str]:
    context = _build_exec_summary_context(patient_profile, scored_trials)
    prompt = ChatPromptTemplate.from_template(build_synthesis_prompt())
    chain = prompt | get_llm().with_structured_output(ExecutiveSummaryModel)

    try:
        result = await _invoke_exec_summary_llm(chain, context)
        return {
            "executive_summary": result["executive_summary"],
            "patient_summary": result["patient_summary"],
        }
    except Exception as exc:
        logger.error("Executive summary generation failed after retries: {}", exc)

    return {
        "executive_summary": (
            f"Conservative triage completed for {context['total']} trials. "
            f"Strong: {context['strong_count']}, Moderate: {context['moderate_count']}, "
            f"Excluded (weak/disqualified): {context['excluded_count']}."
        ),
        "patient_summary": context["patient_summary"],
    }


def _build_trial_report_entry(
    trial: dict[str, Any], eligibility_verdicts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tid = str(trial.get("trial_id", ""))
    enriched = dict(trial)
    enriched["verdict_details"] = eligibility_verdicts.get(tid, {})
    return enriched


_METHODOLOGY_NOTE = (
    "This report uses an LLM-as-judge eligibility assessment with conservative tiering. "
    "Strong matches require confidence on major criteria and no disqualifying exclusion triggers. "
    "Weak/disqualified trials are excluded from the main body."
)


def _sort_by_tier_then_score(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda t: (
            TIER_ORDER.get(str(t.get("tier", "weak")), 0),
            float(t.get("score", 0.0)),
        ),
        reverse=True,
    )


async def build_report(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
    missing_info: list[dict[str, Any]],
    eligibility_verdicts: dict[str, dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues: list[str],
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


def _render_trial_entry(t: dict[str, Any], *, include_key_concern: bool) -> list[str]:
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


def _render_tier_section(
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
        lines.extend(_render_trial_entry(t, include_key_concern=include_key_concern))
    return lines


def _render_information_gaps(info_gaps: list[dict[str, Any]]) -> list[str]:
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
            lines.append(f"- {item.get('field', '')}")
            lines.append(f"  {item.get('description', '')}")
        lines.append("")
    return lines


def build_text_report(report_json: dict[str, Any]) -> str:
    """Build human-readable text from report JSON."""
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
        _render_tier_section(
            report_json,
            "strong_matches",
            "STRONG MATCHES",
            include_key_concern=False,
        )
    )
    lines.extend(
        _render_tier_section(
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

    lines.extend(_render_information_gaps(list(report_json.get("information_gaps", []))))

    qa_issues = report_json.get("qa_issues", [])
    if qa_issues:
        lines += ["QA ISSUES", "-" * 40] + [f"  - {issue}" for issue in qa_issues] + [""]

    lines += [
        "METHODOLOGY",
        "-" * 40,
        report_json.get("methodology_note", ""),
        "",
    ]

    return "\n".join(lines)
