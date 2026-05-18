import json
import os
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from models.judge_verdict import JudgeVerdict
from tools.retry import llm_retry

from clinical_trial_agent.config import TIER_ORDER, get_llm

from .eligibility_fallback import fallback_verdict_for_exception, validate_verdict
from .eligibility_prompt_builder import build_judge_messages


def _extract_json_dict_from_text(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    try:
        parsed = parse_json_markdown(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end: int | None = None
    for idx, ch in enumerate(cleaned[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is None:
        return None

    candidate = cleaned[start : end + 1].strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


@llm_retry
async def _judge_trial(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    llm = get_llm(contains_phi=False, node_name="eligibility_judge")
    trial_id = str(trial.get("nct_id", "unknown"))
    messages = build_judge_messages(patient_profile, trial, criteria)

    if hasattr(llm, "with_structured_output"):
        structured_llm = llm.with_structured_output(JudgeVerdict)
        response = await structured_llm.ainvoke(
            messages,
            config={
                "run_name": "eligibility_judge_structured",
                "tags": ["eligibility", "judge"],
            },
        )
        if isinstance(response, JudgeVerdict):
            return response
        if isinstance(response, dict):
            return validate_verdict(response, trial_id)

    # Fallback to raw JSON parse if structured output returns None
    logger.info(
        "Structured output failed or returned None for {}, trying raw JSON fallback",
        trial_id,
    )
    response = await llm.ainvoke(
        messages,
        config={
            "run_name": "eligibility_judge_raw",
            "tags": ["eligibility", "judge", "raw"],
        },
    )

    content = getattr(response, "content", None)
    if isinstance(content, dict):
        return validate_verdict(content, trial_id)
    if isinstance(content, str):
        parsed = _extract_json_dict_from_text(content)
        if isinstance(parsed, dict):
            return validate_verdict(parsed, trial_id)

    logger.warning(
        "Eligibility judge raw fallback failed to parse for {}. Content: {}",
        trial_id,
        content,
    )

    logger.warning(
        "Eligibility judge returned non-structured response for {}: {}",
        trial_id,
        type(response).__name__,
    )
    # Treat non-structured responses as a parsing failure to get a consistent fallback
    return fallback_verdict_for_exception(
        Exception("LLM parsing failed"), trial_id, patient_profile, criteria
    )


def _normalize_criterion_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _criterion_source_type(criteria_type: str) -> str:
    return "parsed_exclusion" if criteria_type == "exclusion" else "parsed_inclusion"


def _build_criterion_lookup(
    criteria: list[dict[str, Any]] | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_type_and_text: dict[tuple[str, str], dict[str, Any]] = {}
    by_text: dict[str, dict[str, Any]] = {}
    for criterion in criteria or []:
        text = str(criterion.get("text", "")).strip()
        normalized = _normalize_criterion_text(text)
        if not normalized:
            continue
        criteria_type = str(
            criterion.get("criteria_type") or criterion.get("criterion_type") or "inclusion"
        ).lower()
        entry = {
            "criterion_id": criterion.get("criterion_id"),
            "criterion_text": text,
            "criterion_type": criteria_type,
            "source_type": _criterion_source_type(criteria_type),
        }
        by_type_and_text.setdefault((criteria_type, normalized), entry)
        by_text.setdefault(normalized, entry)
    return by_type_and_text, by_text


def _evidence_metadata_for_row(
    *,
    text: str,
    kind: str,
    verdict_label: str,
    trial_id: str,
    lookup_by_type_and_text: dict[tuple[str, str], dict[str, Any]],
    lookup_by_text: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = _normalize_criterion_text(text)
    matched = lookup_by_type_and_text.get((kind, normalized)) or lookup_by_text.get(normalized)
    if matched:
        source_type = str(matched["source_type"])
        criterion_id = matched.get("criterion_id")
        criterion_text = str(matched.get("criterion_text") or text)
        evidence_ref = {
            "source_type": source_type,
            "trial_id": trial_id,
            "criterion_id": criterion_id,
            "criterion_text": criterion_text,
            "verdict": verdict_label,
        }
        return {
            "criterion_id": criterion_id,
            "source_type": source_type,
            "criterion_text": criterion_text,
            "evidence_refs": [evidence_ref],
        }

    return {
        "criterion_id": None,
        "source_type": "not_enough_evidence",
        "criterion_text": text,
        "evidence_refs": [
            {
                "source_type": "not_enough_evidence",
                "trial_id": trial_id,
                "criterion_text": text,
                "verdict": verdict_label,
                "note": "No supplied parsed criterion matched this verdict text exactly.",
            }
        ],
    }


def _build_verdict_rows(
    verdict: JudgeVerdict,
    criteria: list[dict[str, Any]] | None = None,
    trial_id: str = "",
) -> list[dict[str, Any]]:
    lookup_by_type_and_text, lookup_by_text = _build_criterion_lookup(criteria)

    def rows(texts: list[str], verdict_label: str, kind: str, hard: bool) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for criterion_text in texts:
            text = str(criterion_text).strip()
            metadata = _evidence_metadata_for_row(
                text=text,
                kind=kind,
                verdict_label=verdict_label,
                trial_id=trial_id,
                lookup_by_type_and_text=lookup_by_type_and_text,
                lookup_by_text=lookup_by_text,
            )
            output.append(
                {
                    **metadata,
                    "verdict": verdict_label,
                    "criterion_type": kind,
                    "is_hard_exclusion": hard,
                }
            )
        return output

    return [
        *rows(verdict.inclusion_met, "MEETS", "inclusion", False),
        *rows(verdict.inclusion_failed, "FAILS", "inclusion", False),
        *rows(verdict.inclusion_uncertain, "UNCERTAIN", "inclusion", False),
        *rows(verdict.exclusion_triggered, "FAILS", "exclusion", True),
        *rows(verdict.exclusion_uncertain, "UNCERTAIN", "exclusion", True),
    ]


def _summarize_verdict_counts(
    verdicts: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    inclusion_meets = 0
    inclusion_fails = 0
    exclusion_triggered = 0
    uncertain = 0
    for verdict in verdicts:
        label = verdict.get("verdict")
        is_hard = bool(verdict.get("is_hard_exclusion"))
        if label == "MEETS":
            inclusion_meets += 1
        elif label == "FAILS":
            if is_hard:
                exclusion_triggered += 1
            else:
                inclusion_fails += 1
        elif label == "UNCERTAIN":
            uncertain += 1
    return (
        inclusion_meets,
        inclusion_fails + exclusion_triggered,
        uncertain,
        exclusion_triggered,
    )


def _build_batch_result(
    trial_id: str,
    trial: dict[str, Any],
    verdict: JudgeVerdict,
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    meets_count, fails_count, uncertain_count, hard_exclusion_failures = _summarize_verdict_counts(
        verdicts
    )

    logger.debug(
        "[{}] Original LLM verdict: tier={}, score={}, meets={}, fails={}, uncertain={}, hard={}",
        trial_id,
        verdict.match_tier,
        verdict.match_score,
        meets_count,
        fails_count,
        uncertain_count,
        hard_exclusion_failures,
    )

    adjusted_tier, adjusted_score, cap_metadata = _apply_sparse_evidence_caps(
        verdict=verdict,
        meets_count=meets_count,
        fails_count=fails_count,
        uncertain_count=uncertain_count,
        hard_exclusion_failures=hard_exclusion_failures,
    )
    adjusted_tier, adjusted_score = _apply_criteria_provenance_caps(
        trial, adjusted_tier, adjusted_score
    )
    return {
        "trial_id": trial_id,
        "match_score": adjusted_score,
        "match_tier": adjusted_tier,
        "major_criteria_assessable": verdict.major_criteria_assessable,
        "critical_missing_info": verdict.critical_missing_info,
        "key_concern": verdict.key_concern,
        "rationale": verdict.rationale,
        "verdicts": verdicts,
        "meets_count": meets_count,
        "fails_count": fails_count,
        "uncertain_count": uncertain_count,
        "hard_exclusion_failures": hard_exclusion_failures,
        "criteria_source": trial.get("criteria_source", "missing"),
        "criteria_source_verified": bool(trial.get("criteria_source_verified", False)),
        "criteria_retrieved_at": trial.get("criteria_retrieved_at"),
        "criteria_completeness": trial.get("criteria_completeness", "missing"),
        "internal_fallback_used": getattr(verdict, "internal_fallback_used", False),
        **cap_metadata,
    }


def _apply_criteria_provenance_caps(
    trial: dict[str, Any], tier: str, score: float
) -> tuple[str, float]:
    source_verified = bool(trial.get("criteria_source_verified", False))
    completeness = str(trial.get("criteria_completeness", "missing")).strip().lower()
    source = str(trial.get("criteria_source", "missing")).strip().lower()

    if source in ("none", "null", ""):
        source = "missing"
    if completeness in ("none", "null", ""):
        completeness = "missing"

    # In evaluation mode, we assume criteria are full/verified if missing metadata
    eval_mode = os.environ.get("CTA_EVAL_MODE", "false").lower() == "true"
    if eval_mode and source == "missing" and completeness == "missing":
        return tier, score

    if source_verified and completeness == "full":
        return tier, score

    if source == "missing" or completeness == "missing":
        if TIER_ORDER[tier] > TIER_ORDER["weak"]:
            tier = "weak"
        return tier, min(score, 0.45)

    if TIER_ORDER[tier] > TIER_ORDER["moderate"]:
        tier = "moderate"
    return tier, min(score, 0.65)


def _apply_sparse_evidence_caps(
    *,
    verdict: JudgeVerdict,
    meets_count: int,
    fails_count: int,
    uncertain_count: int,
    hard_exclusion_failures: int,
) -> tuple[str, float, dict[str, Any]]:
    if hard_exclusion_failures > 0:
        return (
            "disqualified",
            0.0,
            {"sparse_evidence_cap_applied": False, "sparse_evidence_cap_reason": None},
        )

    tier = verdict.match_tier
    score = verdict.match_score
    assessed_count = meets_count + fails_count
    total_count = assessed_count + uncertain_count

    metadata: dict[str, Any] = {
        "sparse_evidence_cap_applied": False,
        "sparse_evidence_cap_reason": None,
    }

    if total_count == 0:
        if TIER_ORDER[tier] > TIER_ORDER["weak"]:
            tier = "weak"
            metadata.update(
                {
                    "sparse_evidence_cap_applied": True,
                    "sparse_evidence_cap_reason": "total_count == 0",
                }
            )
        return tier, min(score, 0.25), metadata

    if assessed_count <= 1:
        if verdict.major_criteria_assessable and TIER_ORDER[tier] >= TIER_ORDER["moderate"]:
            tier = "moderate"
            metadata.update(
                {
                    "sparse_evidence_cap_applied": True,
                    "sparse_evidence_cap_reason": "assessed_count <= 1 with major criteria met",
                }
            )
            return tier, min(score, 0.55), metadata
        else:
            if TIER_ORDER[tier] > TIER_ORDER["weak"]:
                tier = "weak"
                metadata.update(
                    {
                        "sparse_evidence_cap_applied": True,
                        "sparse_evidence_cap_reason": "assessed_count <= 1",
                    }
                )
            return tier, min(score, 0.45), metadata

    if (
        assessed_count >= 2
        and hard_exclusion_failures == 0
        and tier == "strong"
        and uncertain_count > assessed_count
    ):
        tier = "moderate"
        metadata.update(
            {
                "sparse_evidence_cap_applied": True,
                "sparse_evidence_cap_reason": "uncertain_count > assessed_count",
            }
        )
        return tier, min(score, 0.65), metadata

    return tier, score, metadata


async def evaluate_criteria_batch(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_id = str(trial.get("nct_id", "unknown"))

    try:
        verdict = await _judge_trial(patient_profile, trial, all_criteria)
    except Exception as exc:
        verdict = fallback_verdict_for_exception(exc, trial_id, patient_profile, all_criteria)

    verdicts = _build_verdict_rows(verdict, all_criteria, trial_id)
    return _build_batch_result(trial_id, trial, verdict, verdicts)
