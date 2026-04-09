import asyncio
import re
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.criteria import EligibilityCriterion, ParsedEligibilityCriterion
from prompts.criteria_parser import build_criteria_parser_prompt
from tools import cache
from tools.retry import llm_retry

from config import get_llm, get_settings

CriterionCategoryLiteral = Literal[
    "age", "lab", "biomarker", "diagnosis", "medication", "performance", "other"
]

_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_template(build_criteria_parser_prompt())
_CATEGORY_KEYWORDS: tuple[tuple[CriterionCategoryLiteral, tuple[str, ...]], ...] = (
    ("age", ("age", "year", "yo")),
    ("biomarker", ("egfr", "braf", "alk", "pd-l1", "mutation", "biomarker")),
    ("lab", ("creatinine", "bilirubin", "ast", "alt", "hemoglobin", "platelet", "wbc", "ldh")),
    ("performance", ("ecog", "karnofsky", "performance")),
    ("diagnosis", ("diagnosis", "histology", "pathology", "metastatic", "stage")),
    ("medication", ("treatment", "therapy", "drug", "medication", "immunotherapy", "chemotherapy")),
)


def _get_chain() -> Any:
    """Build a fresh parser chain per call to avoid stale loop-bound transports."""
    return _PROMPT | get_llm().with_structured_output(ParsedEligibilityCriterion)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _infer_category(text: str) -> CriterionCategoryLiteral:
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if _contains_any(lowered, keywords):
            return category
    return "other"


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\-•*]+\s*", "", line)
        line = re.sub(r"^\(?\d+[.)]\s*", "", line)
        line = re.sub(r"^\(?[a-zA-Z][.)]\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip(" .")
        if len(line) < 8:
            continue
        if line.lower() in {"inclusion criteria", "exclusion criteria"}:
            continue
        lines.append(line)
    # de-duplicate preserving order
    return list(dict.fromkeys(lines))


def _split_sections(text: str) -> tuple[list[str], list[str]]:
    if not text:
        return [], []
    if "Exclusion Criteria:" in text:
        inclusion_part, exclusion_part = text.split("Exclusion Criteria:", 1)
        inclusion_part = inclusion_part.replace("Inclusion Criteria:", "")
        return _clean_lines(inclusion_part), _clean_lines(exclusion_part)

    lines = _clean_lines(text.replace("Inclusion Criteria:", ""))
    inclusion: list[str] = []
    exclusion: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(
            k in lowered
            for k in ["exclude", "must not", "ineligible", "not eligible", "history of"]
        ):
            exclusion.append(line)
        else:
            inclusion.append(line)
    return inclusion, exclusion


def _fallback_parse_criteria(eligibility_text: str) -> ParsedEligibilityCriterion:
    inclusion_lines, exclusion_lines = _split_sections(eligibility_text)

    inclusion = [
        EligibilityCriterion(
            text=line,
            is_hard_exclusion=False,
            category=_infer_category(line),
        )
        for line in inclusion_lines[:60]
    ]
    exclusion = [
        EligibilityCriterion(
            text=line,
            is_hard_exclusion=True,
            category=_infer_category(line),
        )
        for line in exclusion_lines[:60]
    ]
    return ParsedEligibilityCriterion(
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
    )


async def parse_eligibility_criteria(eligibility_text: str, nct_id: str) -> dict[str, Any]:
    if not eligibility_text or len(eligibility_text) < 20:
        return {"inclusion_criteria": [], "exclusion_criteria": []}

    cache_params = {
        "nct_id": nct_id,
        "eligibility_criteria_raw": eligibility_text[: get_settings().criteria_text_max_chars],
    }

    if get_settings().use_cache:
        cached = await asyncio.to_thread(cache.get_cached, "criteria_parser", cache_params)
        if isinstance(cached, dict):
            return cached

    try:
        parsed_obj = await _invoke_criteria_llm(_get_chain(), cache_params)
        if not parsed_obj.inclusion_criteria and not parsed_obj.exclusion_criteria:
            parsed_obj = _fallback_parse_criteria(eligibility_text)
        result = parsed_obj.model_dump()
        parsed = _assign_ids(result, nct_id)

        if get_settings().use_cache:
            await asyncio.to_thread(cache.set_cached, "criteria_parser", cache_params, parsed)

        return parsed

    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(
            "Criteria parsing failed for {} ({}). Falling back to deterministic split.",
            nct_id,
            exc,
        )
        fallback = _fallback_parse_criteria(eligibility_text)
        return _assign_ids(fallback.model_dump(), nct_id)


@llm_retry
async def _invoke_criteria_llm(chain: Any, inputs: dict[str, Any]) -> ParsedEligibilityCriterion:
    result = await chain.ainvoke(
        inputs,
        config={"run_name": "criteria_parse", "tags": ["eligibility", "parse"]},
    )

    if result is None:
        raise ValueError("LLM returned None for criteria parse")

    if isinstance(result, ParsedEligibilityCriterion):
        return result
    if isinstance(result, dict):
        return ParsedEligibilityCriterion.model_validate(result)
    raise TypeError(f"Unexpected criteria parser output type: {type(result)!r}")


def _assign_ids(
    parsed: dict[str, list[dict[str, Any]]], nct_id: str
) -> dict[str, list[dict[str, Any]]]:
    for i, crit in enumerate(parsed.get("inclusion_criteria", [])):
        crit["criterion_id"] = f"{nct_id}_inc_{i}"
        crit.setdefault("criterion_type", "inclusion")
        crit.setdefault("is_hard_exclusion", False)
        crit.setdefault("category", "other")

    for i, crit in enumerate(parsed.get("exclusion_criteria", [])):
        crit["criterion_id"] = f"{nct_id}_exc_{i}"
        crit.setdefault("criterion_type", "exclusion")
        crit.setdefault("is_hard_exclusion", True)
        crit.setdefault("category", "other")

    return parsed


async def parse_criteria_for_trials(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = [
        parse_eligibility_criteria(
            (
                f"Inclusion Criteria:\n{trial.get('inclusion_criteria_parsed', '')}\n\n"
                f"Exclusion Criteria:\n{trial.get('exclusion_criteria_parsed', '')}"
            ).strip()
            if trial.get("inclusion_criteria_parsed") or trial.get("exclusion_criteria_parsed")
            else (trial.get("eligibility_criteria_raw", "") or ""),
            trial.get("nct_id", "unknown"),
        )
        for trial in trials
    ]

    results_parsed = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for trial, parsed in zip(trials, results_parsed, strict=False):
        if isinstance(parsed, BaseException):
            logger.exception(
                "gather task failed for {}: {}", trial.get("nct_id", "unknown"), parsed
            )
            parsed = {"inclusion_criteria": [], "exclusion_criteria": []}

        results.append(
            {
                "trial": trial,
                "inclusion_criteria": parsed.get("inclusion_criteria", []),
                "exclusion_criteria": parsed.get("exclusion_criteria", []),
            }
        )

    return results
