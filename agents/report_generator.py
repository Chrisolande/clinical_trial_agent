from datetime import UTC, datetime
from typing import Any

from clinical_trial_agent.config import TIER_ORDER

from .report_formatter import build_text_report
from .report_generator_gaps import merge_information_gaps
from .report_synthesizer import generate_report_plan

_METHODOLOGY_NOTE = (
    "This report uses structured eligibility assessment with conservative tiering. "
    "Strong matches require confidence on major criteria and no disqualifying exclusion triggers. "
    "Weak/disqualified trials are excluded from the main body."
)


def _normalize_qa_issues(qa_issues: list[dict[str, Any] | str]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for issue in qa_issues:
        if isinstance(issue, dict):
            normalized.append(
                {
                    "code": str(issue.get("code", "UNSPECIFIED")),
                    "severity": str(issue.get("severity", "medium")),
                    "message": str(issue.get("message", "")),
                }
            )
            continue
        message = str(issue).strip()
        if message:
            normalized.append({"code": "UNSPECIFIED", "severity": "medium", "message": message})
    return normalized


def _sanitize_qa_remediation(qa_remediation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(qa_remediation, dict):
        return {"attempts": 0, "actions": [], "unresolved_issues": []}
    return {
        "attempts": int(qa_remediation.get("attempts", 0) or 0),
        "actions": list(qa_remediation.get("actions") or []),
        "unresolved_issues": _normalize_qa_issues(
            list(qa_remediation.get("unresolved_issues") or [])
        ),
    }


def _partition_trials(
    enriched_trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strong = [t for t in enriched_trials if str(t.get("tier", "weak")) == "strong"]
    moderate = [t for t in enriched_trials if str(t.get("tier", "weak")) == "moderate"]
    weak = [t for t in enriched_trials if str(t.get("tier", "weak")) == "weak"]
    disqualified = [t for t in enriched_trials if str(t.get("tier", "weak")) == "disqualified"]
    return strong, moderate, weak, disqualified


def _build_per_criterion_verdicts(verdict_details: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = {"MEETS": "eligible", "FAILS": "ineligible", "UNCERTAIN": "uncertain"}
    rows: list[dict[str, Any]] = []
    for item in list(verdict_details.get("verdicts", [])):
        text = str(item.get("criterion_text", "")).strip()
        if not text:
            continue
        rows.append(
            {
                "criterion_id": item.get("criterion_id"),
                "criterion_text": text,
                "verdict": mapping.get(str(item.get("verdict", "UNCERTAIN")), "uncertain"),
                "reasoning": str(item.get("reasoning", "")),
                "source_type": item.get("source_type"),
                "evidence_refs": list(item.get("evidence_refs") or []),
            }
        )
    return rows


_TRACE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "be",
    "by",
    "criteria",
    "criterion",
    "for",
    "in",
    "is",
    "it",
    "known",
    "main",
    "major",
    "no",
    "of",
    "or",
    "patient",
    "pending",
    "supported",
    "the",
    "to",
    "trial",
    "with",
}

_POSITIVE_ASSERTION_TOKENS = {
    "detected",
    "expressed",
    "mutated",
    "mutation",
    "positive",
    "present",
}
_NEGATIVE_ASSERTION_TOKENS = {
    "absent",
    "negative",
    "undetected",
    "without",
    "wildtype",
}
_ELIGIBILITY_ASSERTION_TOKENS = {
    "candidate",
    "eligible",
    "eligibility",
    "fit",
    "fits",
    "match",
    "matches",
    "meets",
    "qualifies",
    "suitable",
}


def _tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
    return {token for token in cleaned.split() if len(token) > 2 and token not in _TRACE_STOPWORDS}


def _not_enough_evidence_ref(trial_id: str, claim: str) -> dict[str, Any]:
    _ = claim
    return {
        "source_type": "not_enough_evidence",
        "trial_id": trial_id,
        "note": "No supplied criterion or trial metadata evidence matched this claim.",
    }


def _has_supported_row_provenance(row: dict[str, Any]) -> bool:
    source_type = str(row.get("source_type") or "").strip()
    return source_type in {"parsed_inclusion", "parsed_exclusion"} and bool(row.get("criterion_id"))


def _row_evidence_ref(row: dict[str, Any], trial_id: str) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "").strip()
    if source_type == "not_enough_evidence":
        return _not_enough_evidence_ref(
            trial_id,
            str(row.get("criterion_text") or "criterion verdict"),
        )
    if not _has_supported_row_provenance(row):
        return _not_enough_evidence_ref(
            trial_id,
            str(row.get("criterion_text") or "criterion verdict"),
        )
    if source_type not in {"parsed_inclusion", "parsed_exclusion"}:
        criterion_type = str(row.get("criterion_type", "")).strip().lower()
        source_type = "parsed_exclusion" if criterion_type == "exclusion" else "parsed_inclusion"
    return {
        "source_type": source_type,
        "trial_id": trial_id,
        "criterion_id": row.get("criterion_id"),
        "criterion_text": str(row.get("criterion_text", "")),
        "verdict": str(row.get("verdict", "UNCERTAIN")).upper(),
    }


def _metadata_evidence_refs(
    claim: str, trial: dict[str, Any], trial_id: str
) -> list[dict[str, Any]]:
    claim_tokens = _tokens(claim)
    refs: list[dict[str, Any]] = []
    metadata_fields = ("brief_title", "overall_status", "phase", "lead_sponsor")
    for field in metadata_fields:
        value = str(trial.get(field, "")).strip()
        if not value:
            continue
        if claim_tokens & _tokens(value):
            refs.append(
                {
                    "source_type": "trial_metadata",
                    "trial_id": trial_id,
                    "metadata_field": field,
                    "metadata_value": value,
                }
            )
    return refs[:2]


def _has_conflicting_assertion(claim: str, evidence_text: str) -> bool:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence_text)
    shared_subject = (
        claim_tokens & evidence_tokens - _POSITIVE_ASSERTION_TOKENS - _NEGATIVE_ASSERTION_TOKENS
    )
    if not shared_subject:
        return False
    claim_positive = bool(claim_tokens & _POSITIVE_ASSERTION_TOKENS)
    claim_negative = bool(claim_tokens & _NEGATIVE_ASSERTION_TOKENS)
    evidence_positive = bool(evidence_tokens & _POSITIVE_ASSERTION_TOKENS)
    evidence_negative = bool(evidence_tokens & _NEGATIVE_ASSERTION_TOKENS)
    return (claim_positive and evidence_negative) or (claim_negative and evidence_positive)


def _claim_evidence_refs(
    claim: str,
    trial: dict[str, Any],
    *,
    preferred_verdicts: set[str],
) -> list[dict[str, Any]]:
    trial_id = str(trial.get("trial_id") or trial.get("nct_id") or "")
    claim_tokens = _tokens(claim)
    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for row in list((trial.get("verdict_details") or {}).get("verdicts", [])):
        verdict = str(row.get("verdict", "")).upper()
        if verdict not in preferred_verdicts:
            continue
        if _has_conflicting_assertion(claim, str(row.get("criterion_text", ""))):
            continue
        overlap = len(claim_tokens & _tokens(str(row.get("criterion_text", ""))))
        if overlap > 0:
            scored_rows.append((overlap, row))

    if scored_rows:
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [_row_evidence_ref(row, trial_id) for _, row in scored_rows[:3]]

    metadata_refs = _metadata_evidence_refs(claim, trial, trial_id)
    if metadata_refs:
        return metadata_refs

    return [_not_enough_evidence_ref(trial_id, claim)]


def _claim_evidence_entries(
    claims: list[Any],
    trial: dict[str, Any],
    *,
    preferred_verdicts: set[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for claim in claims:
        claim_text = str(claim).strip()
        if not claim_text:
            continue
        entries.append(
            {
                "claim_text": claim_text,
                "evidence_refs": _claim_evidence_refs(
                    claim_text, trial, preferred_verdicts=preferred_verdicts
                ),
            }
        )
    return entries


def _has_supporting_evidence(entry: dict[str, Any]) -> bool:
    return any(
        str(ref.get("source_type") or "") != "not_enough_evidence"
        for ref in list(entry.get("evidence_refs") or [])
        if isinstance(ref, dict)
    )


def _append_unique(values: list[Any], value: str) -> None:
    normalized = value.strip().lower()
    if normalized and normalized not in {str(item).strip().lower() for item in values}:
        values.append(value)


def _mentions_claim(text: str, claim: str) -> bool:
    text_tokens = _tokens(text)
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return False
    return claim.lower() in text.lower() or len(text_tokens & claim_tokens) >= min(
        3, len(claim_tokens)
    )


def _looks_like_positive_eligibility_claim(text: str) -> bool:
    tokens = _tokens(text)
    if not (tokens & _ELIGIBILITY_ASSERTION_TOKENS):
        return False
    conditional_terms = {"confirm", "confirmation", "if", "pending", "review", "screening"}
    return not bool(tokens & conditional_terms)


def _supported_as_positive_claim(text: str, trial: dict[str, Any]) -> bool:
    refs = _claim_evidence_refs(text, trial, preferred_verdicts={"MEETS"})
    return any(str(ref.get("source_type") or "") != "not_enough_evidence" for ref in refs)


def _demote_unsupported_text_field(
    card: dict[str, Any],
    field: str,
    trial: dict[str, Any],
    unsupported_claims: list[str],
) -> None:
    text = str(card.get(field) or "").strip()
    if not text:
        return
    mentioned_claim = next(
        (claim for claim in unsupported_claims if _mentions_claim(text, claim)), ""
    )
    unsupported_assertion = bool(mentioned_claim) or (
        _looks_like_positive_eligibility_claim(text)
        and not _supported_as_positive_claim(text, trial)
    )
    if not unsupported_assertion:
        return
    claim_text = mentioned_claim or "eligibility claim"
    card[field] = f"Uncertainty: {claim_text} needs confirmation before use as a match reason."


def _ensure_card_evidence(card: dict[str, Any], trial_lookup: dict[str, dict[str, Any]]) -> None:
    trial_id = str(card.get("nct_id") or card.get("trial_id") or "")
    trial = trial_lookup.get(trial_id, {"trial_id": trial_id})
    why_entries = _claim_evidence_entries(
        list(card.get("why_it_matches") or []),
        trial,
        preferred_verdicts={"MEETS"},
    )
    supported_why_entries = [entry for entry in why_entries if _has_supporting_evidence(entry)]
    unsupported_claims = [
        str(entry.get("claim_text", "")).strip()
        for entry in why_entries
        if not _has_supporting_evidence(entry)
    ]
    card["why_it_matches"] = [entry["claim_text"] for entry in supported_why_entries]
    card["why_it_matches_evidence"] = supported_why_entries

    uncertainties = list(card.get("key_uncertainties") or [])
    for claim in unsupported_claims:
        if claim:
            _append_unique(uncertainties, f"Unsupported match claim requires confirmation: {claim}")
    card["key_uncertainties"] = uncertainties
    _demote_unsupported_text_field(card, "recommendation", trial, unsupported_claims)
    _demote_unsupported_text_field(card, "next_action", trial, unsupported_claims)

    card["main_blockers_evidence"] = _claim_evidence_entries(
        list(card.get("main_blockers") or []),
        trial,
        preferred_verdicts={"FAILS", "UNCERTAIN"},
    )
    card["key_uncertainties_evidence"] = _claim_evidence_entries(
        uncertainties,
        trial,
        preferred_verdicts={"UNCERTAIN"},
    )
    evidence_summary = str(card.get("evidence_summary", "")).strip()
    if evidence_summary:
        card["evidence_summary_refs"] = _claim_evidence_refs(
            evidence_summary,
            trial,
            preferred_verdicts={"MEETS", "FAILS", "UNCERTAIN"},
        )


def _ensure_report_plan_evidence(
    report_plan_dict: dict[str, Any], enriched_trials: list[dict[str, Any]]
) -> dict[str, Any]:
    updated = dict(report_plan_dict)
    trial_lookup = {str(trial.get("trial_id", "")): trial for trial in enriched_trials}
    for section in ("strong_matches", "moderate_matches"):
        cards: list[dict[str, Any]] = []
        for card in list(updated.get(section) or []):
            if not isinstance(card, dict):
                continue
            enriched_card = dict(card)
            _ensure_card_evidence(enriched_card, trial_lookup)
            cards.append(enriched_card)
        updated[section] = cards
    return updated


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
    report_plan: Any,
    enriched_trials: list[dict[str, Any]],
    information_gaps: list[dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues_internal: list[dict[str, str]],
    retrieval_errors: list[str],
    qa_remediation_internal: dict[str, Any],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _partition_trials(enriched_trials)
    retrieval_failed = bool(retrieval_errors) and len(enriched_trials) == 0
    report_plan_dict = (
        report_plan.model_dump() if hasattr(report_plan, "model_dump") else dict(report_plan)
    )
    report_plan_dict = _ensure_report_plan_evidence(report_plan_dict, enriched_trials)
    effective_summary = str(report_plan_dict.get("executive_summary", ""))
    patient_summary = str(report_plan_dict.get("patient_summary", ""))
    if retrieval_failed:
        effective_summary = (
            "Trial retrieval failed before eligibility fan-out completed. "
            "No matches are reported because zero trials were retrieved/evaluated."
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_plan": report_plan_dict,
        "patient_summary": patient_summary,
        "executive_summary": effective_summary,
        "total_trials_searched": len(trials_raw),
        "total_trials_evaluated": len(enriched_trials),
        "retrieval_failed": retrieval_failed,
        "retrieval_errors": list(retrieval_errors),
        "strong_matches": strong,
        "moderate_matches": moderate,
        "excluded_trial_count": len(weak) + len(disqualified),
        "excluded_trials": weak + disqualified,
        "information_gaps": information_gaps,
        "qa_issues_internal": qa_issues_internal,
        "qa_remediation_internal": qa_remediation_internal,
        "debug": False,
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
    qa_issues: list[dict[str, Any] | str],
    retrieval_errors: list[str] | None = None,
    qa_remediation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = _sort_by_tier_then_score(scored_trials)
    enriched_trials = [_build_trial_report_entry(t, eligibility_verdicts) for t in ranked]
    merged_gaps = merge_information_gaps(ranked, missing_info)
    qa_internal = _normalize_qa_issues(qa_issues)
    report_plan = await generate_report_plan(
        patient_profile=patient_profile,
        scored_trials=enriched_trials,
        eligibility_verdicts=eligibility_verdicts,
        missing_info=merged_gaps,
        qa_issues=qa_internal,
    )
    return _build_report_payload(
        report_plan=report_plan,
        enriched_trials=enriched_trials,
        information_gaps=merged_gaps,
        trials_raw=trials_raw,
        search_queries=search_queries,
        decision_history=decision_history,
        qa_issues_internal=qa_internal,
        retrieval_errors=list(retrieval_errors or []),
        qa_remediation_internal=_sanitize_qa_remediation(qa_remediation),
    )


def _merge_information_gaps(
    scored_trials: list[dict[str, Any]], missing_info: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return merge_information_gaps(scored_trials, missing_info)


__all__ = ["build_report", "build_text_report"]
